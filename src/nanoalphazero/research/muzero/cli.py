"""Installed MuZero research training and smoke entry point."""

import argparse
from datetime import datetime, timezone
import functools
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import time
import tomllib


def resolve(raw):
    from nanoalphazero.config import CONFIG_FACTORIES
    name = raw.get("env", "hex4")
    base = CONFIG_FACTORIES[name]()
    config = dict(env=name, env_id=base["env_id"], width=128, depth=2, unroll=5,
                  discount=1.0, max_steps=base["game_max_steps"],
                  roots=base["mcts_num_root_considered"], survivors=base["mcts_num_survivors"],
                  exploration_moves=base["num_exploratory_moves"],
                  selfplay_batch_size=32, train_batch_size=32, replay_batches=8,
                  updates_per_cycle=4, cycles=2, seed=0, learning_rate=1e-3,
                  weight_decay=1e-4, devices=4, platform="tpu", wandb=False,
                  checkpoint_period=10, eval_period=10, defaults="alphazero",
                  warmup_updates=0, decay_kernels_only=False,
                  exploration_mode="fixed", root_temperature=1.0,
                  value_scale=1.0, maxvisit_init=50., rescale_values=False,
                  hex_eval_period=0, diagnostic_period=base["diagnostic_period"],
                  network="vector", activation=base.get("katago_activation", "mish"),
                  use_rvgl=base.get("katago_use_rvgl", True), decision_eval_positions=0,
                  data_pipeline="episodes", staging_batches=8,
                  consume_size=base["selfplay_buffer_consume_size"],
                  replay_positions=base["replay_buffer_total_size"],
                  replay_warmup_cycles=(base["replay_buffer_warmup_steps"] + base["game_max_steps"] - 1)
                  // base["game_max_steps"])
    if raw.get("defaults", "alphazero") == "alphazero":
        train_batch = 4096 if name == "hex5" else base["train_batch_size"]
        config.update(
            width=base["conv_width"], depth=base["conv_depth"],
            selfplay_batch_size=base["selfplay_batch_size"], train_batch_size=train_batch,
            learning_rate=base["learning_rate"], weight_decay=base["weight_decay"],
            updates_per_cycle=base["cycle_n_train"],
            cycles=base["num_iters"] // base["cycle_n_train"],
            replay_batches=max(1, base["replay_buffer_total_size"] //
                               (base["selfplay_batch_size"] * base["game_max_steps"])),
            warmup_updates=base["lr_warmup_steps"],
            decay_kernels_only=base.get("weight_decay_kernels_only", False),
            exploration_mode="random_switch",
            root_temperature=base.get("exp_root_temperature", 1.) if base.get("exp_use_root_temperature") else 1.,
            value_scale=base["mcts_value_scale"], maxvisit_init=base["mcts_maxvisit_init"],
            rescale_values=base["mcts_rescale_values"],
            eval_period=base["eval_period"], checkpoint_period=base["ckpt_period"] or
            (base["num_iters"] // base["cycle_n_train"]),
        )
    elif raw.get("defaults") != "pilot_v1":
        raise ValueError("defaults must be alphazero or pilot_v1")
    unknown = set(raw) - set(config)
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    config.update(raw)
    if config["network"] == "vector":
        # These architectural options belong to the spatial production modules.
        # Record the vector tower's actual choices rather than unused defaults.
        config.update(activation="relu", use_rvgl=False)
    for key in ("width", "unroll", "max_steps", "roots", "survivors", "selfplay_batch_size",
                "train_batch_size", "replay_batches", "updates_per_cycle", "cycles", "devices"):
        if config[key] < 1:
            raise ValueError(f"{key} must be positive")
    if config["survivors"] > config["roots"]:
        raise ValueError("survivors exceeds roots")
    if config["root_temperature"] <= 0 or config["warmup_updates"] < 0:
        raise ValueError("Invalid root temperature or warmup")
    if config["exploration_mode"] not in ("fixed", "random_switch"):
        raise ValueError("Unknown exploration mode")
    if config["network"] not in ("vector", "spatial"):
        raise ValueError("Unknown network")
    if config["data_pipeline"] not in ("episodes", "staged"):
        raise ValueError("Unknown data pipeline")
    if min(config["staging_batches"], config["consume_size"], config["replay_positions"]) < 1:
        raise ValueError("Staging/replay sizes must be positive")
    if config["data_pipeline"] == "staged" and config["consume_size"] > config["selfplay_batch_size"] * config["staging_batches"] * config["max_steps"]:
        raise ValueError("consume_size exceeds staging capacity")
    if config["replay_warmup_cycles"] < 0 or config["replay_positions"] < config["consume_size"]:
        raise ValueError("Invalid replay warmup or capacity")
    if not 0 < config["discount"] <= 1:
        raise ValueError("discount must be in (0, 1]")
    if name == "hex5" and config["train_batch_size"] != 4096:
        raise ValueError("Hex 5 requires train_batch_size = 4096")
    if config["platform"] == "tpu" and config["devices"] != 4:
        raise ValueError("TPU experiments require exactly four devices")
    if config["cycles"] > 10 and not config["wandb"]:
        raise ValueError("Long runs require W&B")
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--hex-eval-engine-path")
    parser.add_argument("--hex-eval-engine-config", default="default")
    args = parser.parse_args()
    with args.config.open("rb") as stream:
        config = resolve(tomllib.load(stream))
    engine_pool = None
    if config["hex_eval_period"] or config["decision_eval_positions"]:
        if not config["env"].startswith("hex") or not args.hex_eval_engine_path:
            raise ValueError("Periodic MoHex evaluation requires Hex and --hex-eval-engine-path")
        # Match production: external processes must start before libtpu/JAX.
        from nanoalphazero.eval.hex.engine import MoHexPool
        size = int(config["env"][3:])
        engine_pool = MoHexPool(args.hex_eval_engine_path, args.hex_eval_engine_config, size, size * size)
    try:
        _run(args, config, engine_pool)
    finally:
        if engine_pool is not None:
            engine_pool.close()


def _run(args, config, engine_pool):
    # Selection precedes imports that initialize production's device mesh.
    import os
    os.environ["JAX_PLATFORMS"] = config["platform"]
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    from flax.training.train_state import TrainState
    from nanoalphazero.core import make_env
    from nanoalphazero.research.muzero import checkpoint
    from nanoalphazero.research.muzero.learning import make_collect, train_step
    from nanoalphazero.research.muzero.model import build_model
    from nanoalphazero.research.muzero.replay import make_replay, sequences
    from nanoalphazero.research.muzero.search import make_search
    from nanoalphazero.research.muzero.evaluation import make_evaluator, heldout_metrics
    from nanoalphazero.research.muzero.inspection import inspect_position_values
    from nanoalphazero.research.muzero.metrics import standardize, batch_metrics, collection_metrics, histogram_data

    devices = jax.devices()
    if len(devices) != config["devices"]:
        raise ValueError(f"Expected {config['devices']} devices, found {devices}")
    for key in ("selfplay_batch_size", "train_batch_size", "consume_size"):
        if config[key] % len(devices):
            raise ValueError(f"{key} must be divisible by device count")
    args.output.mkdir(parents=True, exist_ok=False)
    mesh = jax.sharding.Mesh(np.array(devices), ("data",))
    data = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    env = make_env(config)
    config.update(obs_shape=list(env.obs_shape), num_actions=env.num_actions,
                  architecture=f"{config['network']}-v1", target="signed-return-boundary-bootstrap",
                  latent_normalization="minmax", latent_normalization_epsilon=1e-5,
                  dynamics_gradient_scale=0.5, policy_loss="cross_entropy",
                  value_loss="scalar_mse", reward_loss="scalar_mse", max_grad_norm=1.0,
                  optimizer="adamw", initial_warmup_lr=1e-6,
                  metric_schema="alphazero-v1",
                  replay_sampling=("uniform_consumed_position" if config["data_pipeline"] == "staged"
                                   else "uniform_episode_then_uniform_real_position"))
    from nanoalphazero.config import CONFIG_FACTORIES
    inspection_config = {**CONFIG_FACTORIES[config["env"]](), "enable_sharding": True}
    if config["roots"] > env.num_actions:
        raise ValueError("roots exceeds action count")
    if args.resume and checkpoint.metadata(args.resume) != config:
        raise ValueError("Resume requires identical resolved configuration")
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    manifest = {"command": __import__("sys").argv, "devices": [str(d) for d in devices],
                "jax": jax.__version__, "started": datetime.now(timezone.utc).isoformat(),
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "git_status": subprocess.check_output(["git", "status", "--short"], text=True)}
    if engine_pool is not None:
        manifest["mohex"] = {"executable": str(engine_pool.executable),
                              "config": engine_pool.config.read_text(),
                              "sha256": hashlib.sha256(engine_pool.executable.read_bytes()).hexdigest()}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    source_root = Path(__file__).parent
    snapshot = args.output / "source-muzero"
    snapshot.mkdir()
    manifest["source_sha256"] = {}
    for source in sorted(source_root.glob("*.py")):
        shutil.copyfile(source, snapshot / source.name)
        manifest["source_sha256"][source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
    model = build_model(config)
    key, init_key = jax.random.split(jax.random.PRNGKey(config["seed"]))
    params = model.init(init_key, jnp.zeros((1, *env.obs_shape)), jnp.zeros(1, jnp.int32))["params"]
    rate = config["learning_rate"]
    if config["warmup_updates"]:
        rate = optax.join_schedules([
            optax.linear_schedule(1e-6, rate, config["warmup_updates"]),
            optax.constant_schedule(rate)], [config["warmup_updates"]])
    decay_mask = None
    if config["decay_kernels_only"]:
        from flax.traverse_util import flatten_dict, unflatten_dict
        decay_mask = unflatten_dict({p: p[-1] in ("kernel", "kernel_3x3", "kernel_1x1")
                                     for p in flatten_dict(params)})
    state = TrainState.create(apply_fn=model.apply, params=params,
                              tx=optax.chain(optax.clip_by_global_norm(1.0),
                                             optax.adamw(rate, weight_decay=config["weight_decay"], mask=decay_mask)))
    state = jax.device_put(state, replicated)
    search = make_search(model, env, config)
    collect = jax.jit(make_collect(env, model, search, config), out_shardings=data)
    dummy = jax.eval_shape(collect, state.params, key)
    example = jax.tree.map(lambda x: jnp.zeros(x.shape[1:], x.dtype), dummy)
    staging_state = None
    if config["data_pipeline"] == "staged":
        from nanoalphazero.research.muzero.staging import make_staging
        init_staging, stage_add, consume, drain, replay = make_staging(config)
        staging_shape = jax.eval_shape(init_staging, example)
        staging_sharding = jax.tree.map(lambda x: data if x.ndim else replicated, staging_shape)
        staging_state = jax.jit(init_staging, out_shardings=staging_sharding)(example)
        batch_shape = jax.eval_shape(consume, staging_state, key)[1]
        example = jax.tree.map(lambda x: jnp.zeros(x.shape[1:], x.dtype), batch_shape)
        stage_add = jax.jit(stage_add, out_shardings=(staging_sharding, replicated), donate_argnums=(0,))
    else:
        replay = make_replay(config)
    shape = jax.eval_shape(replay.init, example)
    replay_sharding = jax.tree.map(lambda x: data if x.ndim else replicated, shape)
    replay_state = jax.jit(replay.init, out_shardings=replay_sharding)(example)
    add = jax.jit(replay.add, donate_argnums=(0,), out_shardings=replay_sharding)
    if staging_state is not None:
        drain = jax.jit(drain, out_shardings=(staging_sharding, replay_sharding, replicated, replicated),
                        donate_argnums=(0, 1))
    sample = jax.jit(lambda s, k: jax.tree.map(lambda x: x[:, 0], replay.sample(s, k).experience), out_shardings=data)
    make_batch = jax.jit(functools.partial(sequences, unroll=config["unroll"]), out_shardings=data)
    train = jax.jit(functools.partial(train_step, model), in_shardings=(replicated, data),
                    out_shardings=(replicated, replicated), donate_argnums=(0,))
    start_cycle = 0
    if args.resume:
        template = {"train": state, "replay": vars(replay_state), "key": key, "cycle": jnp.array(0)}
        if staging_state is not None:
            template["staging"] = staging_state
        loaded, _ = checkpoint.load(args.resume, template)
        state = jax.device_put(loaded["train"], replicated)
        replay_state = jax.device_put(replay_state.replace(**loaded["replay"]), replay_sharding)
        key, start_cycle = loaded["key"], int(loaded["cycle"])
        if staging_state is not None:
            staging_state = jax.device_put(loaded["staging"], staging_sharding)
    run = None
    if config["wandb"]:
        import wandb
        run = wandb.init(project="nanoAlphaZero-muzero", config=config,
                         name=f"{config['env']}-{config['network']}-two-rung-seed{config['seed']}-{args.output.name}")
        manifest["wandb_url"] = run.url
        run.define_metric("runner_state/n_updates")
        run.define_metric("runner_state/n_selfplay_steps")
        run.summary["stats/num_params"] = sum(x.size for x in jax.tree.leaves(state.params))
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    started = time.monotonic()
    evaluator = make_evaluator(model, env, config)
    hex_evaluators = {}
    if engine_pool is not None:
        from nanoalphazero.eval.hex.training import HexTrainingEvaluator
        for mode in ("search", "policy"):
            hex_evaluators[mode] = HexTrainingEvaluator(
                config["hex_eval_period"], engine_pool, args.output / f"mohex-{mode}",
                board_size=inspection_config["boardsize"])
    policy_search = make_search(model, env, config, policy_only=True)
    # Separate seeds and episodes; these records are never inserted into replay.
    heldout = None
    if config["eval_period"] and config["cycles"] >= config["eval_period"]:
        if args.resume:
            heldout_path = args.resume.parent / "heldout-initial-selfplay.npz"
            if not heldout_path.exists():
                raise ValueError("Resume needs the original held-out episodes beside its checkpoint")
            with np.load(heldout_path) as saved:
                heldout = jax.device_put({name: saved[name] for name in saved.files}, data)
        else:
            heldout = collect(state.params, jax.random.PRNGKey(config["seed"] + 1000000))
        jax.block_until_ready(heldout)
        np.savez_compressed(args.output / "heldout-initial-selfplay.npz", **jax.device_get(heldout))
        initial_metrics = heldout_metrics(model, state.params, heldout, jax.random.PRNGKey(12345))
        (args.output / "heldout-initial-metrics.json").write_text(json.dumps(initial_metrics, indent=2) + "\n")
        baseline = evaluator(state.params, jax.random.PRNGKey(config["seed"] + 2000000),
                             args.output / f"eval-{start_cycle:06d}")
        print(json.dumps({"initial_evaluation": baseline}), flush=True)

    def collect_and_insert(params, rng, staged, replayed):
        rng, ck = jax.random.split(rng)
        episodes = collect(params, ck)
        jax.block_until_ready(episodes)
        if bool(jnp.any(episodes["illegal"])):
            raise RuntimeError("Illegal real self-play action")
        data_metrics = {}
        if staged is None:
            episodes = {**episodes, "game_id": replayed.current_index.astype(jnp.uint32) * config["selfplay_batch_size"]
                        + jnp.arange(config["selfplay_batch_size"], dtype=jnp.uint32) + 1}
            replayed = add(replayed, jax.tree.map(lambda x: x[:, None], episodes))
        else:
            staged, data_metrics = stage_add(staged, episodes)
            if int(data_metrics["staging/overwritten_fresh"]) or int(data_metrics["staging/malformed_episodes"]):
                raise RuntimeError("Staging overflow or malformed episode; refusing to train")
            staged, replayed, rng, drain_metrics = drain(staged, replayed, rng)
            data_metrics = {**data_metrics, **drain_metrics}
        return episodes, rng, staged, replayed, {
            k: float(v) if jnp.issubdtype(v.dtype, jnp.floating) else int(v) for k, v in data_metrics.items()}

    warmup_start = time.monotonic()
    if staging_state is not None and not args.resume:
        for warmup in range(config["replay_warmup_cycles"]):
            warmup_cycle_start = time.monotonic()
            episodes, key, staging_state, replay_state, data_metrics = collect_and_insert(
                state.params, key, staging_state, replay_state)
            record = standardize({"warmup/loop_n": warmup + 1,
                                  "warmup/loop_duration": time.monotonic() - warmup_cycle_start,
                                  "real_transitions": int(jnp.sum(episodes["length"])),
                                  "runner_state/n_updates": int(state.step), **data_metrics})
            with (args.output / "warmup.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            if warmup == 0 or (warmup + 1) % 10 == 0:
                print(json.dumps(record), flush=True)
            if run:
                run.log(record, step=warmup)
        manifest["replay_warmup_cycles"] = config["replay_warmup_cycles"]
    if run:
        run.summary["stats/warmup_duration"] = time.monotonic() - warmup_start
    learning_started = time.monotonic()
    for cycle in range(start_cycle + 1, config["cycles"] + 1):
        cycle_start = time.monotonic()
        episodes, key, staging_state, replay_state, data_metrics = collect_and_insert(
            state.params, key, staging_state, replay_state)
        if not bool(replay.can_sample(replay_state)):
            raise RuntimeError("Replay has no complete consumable batch; increase collection/warmup")
        for _ in range(config["updates_per_cycle"]):
            key, rk, bk = jax.random.split(key, 3)
            sampled = sample(replay_state, rk)
            batch = sampled if staging_state is not None else make_batch(sampled, bk)
            state, metrics = train(state, batch)
        if cycle == start_cycle + 1:
            manifest["sharding"] = {
                "selfplay_observation": str(episodes["observation"].sharding),
                "replay_observation": str(replay_state.experience["observation"].sharding),
                "training_observation": str(batch["observation"].sharding),
                "training_local_shapes": [list(s.data.shape) for s in batch["observation"].addressable_shards],
            }
        metrics = {k: float(v) for k, v in metrics.items()}
        metrics.update(data_metrics)
        if not all(np.isfinite(v) for v in metrics.values()):
            raise RuntimeError(f"Nonfinite training metrics: {metrics}")
        metrics.update(cycle=cycle, updates=int(state.step),
                       seconds=time.monotonic() - cycle_start,
                       elapsed_seconds=time.monotonic() - started,
                       completed_fraction=float(jnp.mean(episodes["terminal"])),
                       real_transitions=int(jnp.sum(episodes["length"])))
        metrics.update({
            "norms/current_lr": float(rate(state.step - 1)) if callable(rate) else float(rate),
            "muzero/selfplay/episode_length_mean": float(jnp.mean(episodes["length"])),
            "muzero/selfplay/episode_length_std": float(jnp.std(episodes["length"].astype(jnp.float32))),
            "muzero/selfplay/episode_length_min": int(jnp.min(episodes["length"])),
            "muzero/selfplay/episode_length_max": int(jnp.max(episodes["length"])),
            "muzero/selfplay/truncated_fraction": float(jnp.mean(~episodes["terminal"])),
            "runner_state/train_step": int(state.step),
            "runner_state/n_selfplay_steps": cycle * config["max_steps"],
            "norms/n_updates": int(state.step), "loop/loop_n": cycle,
            "debug/sample_rng_hash": float(jnp.sum(rk.astype(jnp.float32))),
            "debug/rbuf_current_index": int(replay_state.current_index),
        })
        metrics.update(batch_metrics(batch, inspection_config["env_allows_draws"]))
        priming_cycles = config["replay_warmup_cycles"] if staging_state is not None else 0
        metrics.update(collection_metrics(episodes, (cycle + priming_cycles) * config["max_steps"], inspection_config["env_allows_draws"]))
        if staging_state is not None:
            occupied = config["replay_positions"] // config["consume_size"] if bool(replay_state.is_full) else int(replay_state.current_index)
            metrics.update({"training/rbuf-n_is_valid": occupied * config["consume_size"],
                            "training/rbuf-n_is_fresh": occupied * config["consume_size"],
                            "training/spbuf_num_consumables": int(jnp.sum(staging_state.fresh)),
                            "training/n_slices_drained": data_metrics["drain/n_slices"]})
        diagnostic_tables = {}
        if cycle == 1 or (config["diagnostic_period"] and cycle % config["diagnostic_period"] == 0):
            metrics.update(inspect_position_values(model, state.params, env, inspection_config))
            if config["env"].startswith("hex"):
                from nanoalphazero.research.muzero.inspection import opening_head_tables
                diagnostic_tables, diagnostic_scalars = opening_head_tables(
                    model, state.params, env, inspection_config, args.output / f"diagnostics-{cycle:06d}")
                metrics.update(diagnostic_scalars)
        for mode, hex_evaluator in hex_evaluators.items():
            results = hex_evaluator.run_if_due(
                cycle, search if mode == "search" else policy_search, env,
                inspection_config, state.params, train_step=int(state.step))
            prefix = "hex_eval/" if mode == "search" else "hex_eval_policy/"
            metrics.update({k.replace("hex_eval/", prefix): v for k, v in results.items()})
        if heldout is not None and cycle % config["eval_period"] == 0:
            metrics.update(heldout_metrics(model, state.params, heldout, jax.random.PRNGKey(12345)))
            fresh = collect(state.params, jax.random.PRNGKey(config["seed"] + 3000000 + cycle))
            fresh_metrics = heldout_metrics(model, state.params, fresh, jax.random.PRNGKey(12345))
            metrics.update({k.replace("heldout/", "heldout_current/"): v for k, v in fresh_metrics.items()})
            metrics["heldout_current/real_transitions"] = int(jnp.sum(fresh["length"]))
            np.savez_compressed(args.output / f"heldout-cycle-{cycle:06d}.npz", **jax.device_get(fresh))
            scores = evaluator(state.params, jax.random.PRNGKey(config["seed"] + 2000000),
                               args.output / f"eval-{cycle:06d}")
            for label, result in scores.items():
                total = result["wins"] + result["draws"] + result["losses"]
                metrics[f"eval/{label}/seat0_score"] = (result["wins"] + .5 * result["draws"]) / total
        if cycle == config["cycles"] and config["decision_eval_positions"]:
            from nanoalphazero.research.muzero.decisions import evaluate_decisions
            result = evaluate_decisions(model, state.params, env, config, engine_pool,
                                        args.output / "exact-decisions", config["decision_eval_positions"])
            metrics.update({f"decisions/{k}": v for k, v in result.items()})
        metrics["elapsed_seconds"] = time.monotonic() - learning_started
        metrics["muzero/timing/run_seconds"] = time.monotonic() - started
        metrics["cycle_wall_seconds"] = time.monotonic() - cycle_start
        metrics["loop/loop_total_duration"] = metrics["elapsed_seconds"]
        metrics = standardize(metrics)
        print(json.dumps(metrics), flush=True)
        with (args.output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(metrics) + "\n")
        if run:
            from nanoalphazero.research.muzero.charts import wandb_board_logs
            consumed_info = None
            if staging_state is not None and data_metrics["drain/n_slices"]:
                last_slot = (replay_state.current_index - 1) % replay_state.experience["policy_mask"].shape[1]
                consumed_info = jax.tree.map(lambda x: x[:, last_slot], replay_state.experience["sample_info"])
            histograms = {name: wandb.Histogram(values) for name, values in histogram_data(episodes, batch, consumed_info).items()}
            run.log({**metrics, **histograms, **wandb_board_logs(diagnostic_tables)}, step=cycle + priming_cycles)
        if cycle % config["checkpoint_period"] == 0 or cycle == config["cycles"]:
            saved = {"train": state, "replay": vars(replay_state), "key": key, "cycle": jnp.array(cycle)}
            if staging_state is not None:
                saved["staging"] = staging_state
            checkpoint.save(args.output / f"cycle-{cycle:06d}.safetensors",
                            saved, config)
    manifest.update(finished=datetime.now(timezone.utc).isoformat(),
                    parameter_count=sum(x.size for x in jax.tree.leaves(state.params)),
                    memory_stats=[d.memory_stats() for d in devices])
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if run:
        run.summary["stats/total_learn_duration"] = time.monotonic() - learning_started
        run.finish()
