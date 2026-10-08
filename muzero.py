"""Build MuZero from the nanoalphazero components.

The cycle is visible here: advance persistent games, consume fresh starts as
K-step training samples, then train from replay. The learning objective and host
loop are visible here too. Buffer indexing,
model layers, search internals, and checkpoint serialization live in components.

Run with ``uv run python muzero.py --env hex4``. Use --help for configuration.
For Trackio, use ``uv run --extra trackio python muzero.py --enabled-trackio``.
The historical portable export is muzero-singlefile.py.
"""

import json
import os
import time
from functools import partial
from types import SimpleNamespace

# Keep JAX and device-initializing component imports inside functions so the CLI
# can select its backend before any accelerator is initialized.


def make_muzero_selfplay(config, wenv, run_mcts_fn, data_sharding=None):
    """Keep AlphaZero's action selection, auto-reset, and persistent game state."""
    import jax.numpy as jnp
    from nanoalphazero.buffers import MuZeroSelfplayOutput
    from nanoalphazero.core import make_selfplay

    original, state = make_selfplay(config, wenv, run_mcts_fn, data_sharding)

    def selfplay_fn(rng, state, params):
        following, output, metrics = original(rng, state, params)
        player = output.player[:, 0]
        reward = following.env_state.rewards[jnp.arange(player.shape[0]), player]
        sign = jnp.where(player == following.env_state.current_player, 1.0, -1.0)
        discount = jnp.where(
            following.env_state.terminated | following.env_state.truncated,
            0.0, sign * config["muzero_discount"],
        )
        output = MuZeroSelfplayOutput(
            **vars(output), transition_reward=reward[:, None], discount=discount[:, None],
        )
        return following, output, metrics

    return selfplay_fn, state


def unroll_muzero_predictions(model, params, batch, remat=False, *, return_latents=False):
    """Run h once and g/f K times; policy/value include the root prediction."""
    import jax
    import jax.numpy as jnp
    from nanoalphazero.model import scale_gradient

    def initial(params, observation):
        return model.apply({"params": params}, observation, method=model.initial)

    def recurrent(params, latent, action):
        return model.apply({"params": params}, latent, action, method=model.recurrent)

    if remat:
        initial, recurrent = jax.checkpoint(initial), jax.checkpoint(recurrent)
    observation = batch.observation
    if observation is None:
        from nanoalphazero.buffers import combine_observation_vmap
        observation = combine_observation_vmap(batch.board_bool[:, 0], batch.board_float[:, 0])
    else:
        observation = observation[:, 0].astype(jnp.float32)
    latent, logits, value = initial(params, observation)
    policies, values, rewards = [logits], [value], []
    latents = []
    for step in range(batch.action.shape[1]):
        latent, reward, logits, value = recurrent(
            params, scale_gradient(latent, 0.5), batch.action[:, step]
        )
        policies.append(logits)
        values.append(value)
        rewards.append(reward)
        if return_latents:
            latents.append(latent)
    predictions = (jnp.stack(policies, axis=1), jnp.stack(values, axis=1),
                   jnp.stack(rewards, axis=1))
    return (*predictions, jnp.stack(latents, axis=1)) if return_latents else predictions


def muzero_policy_kl(logits, target, indices=None):
    """Normalize over every action before selecting sparse target support."""
    import jax
    import jax.numpy as jnp

    target = jax.lax.stop_gradient(target)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    if indices is not None:
        log_probs = jnp.take_along_axis(log_probs, indices.astype(jnp.int32), axis=-1)
    return -jnp.sum(target * log_probs + jax.scipy.special.entr(target), axis=-1)


def masked_consistency_loss(prediction, target, mask):
    """EfficientZero-style negative cosine similarity, with a stop-gradient target."""
    import jax
    import jax.numpy as jnp

    def normalize(x):
        return x * jax.lax.rsqrt(jnp.maximum(jnp.sum(x * x, axis=-1, keepdims=True), 1e-10))

    target = jax.lax.stop_gradient(target)
    losses = -jnp.sum(normalize(prediction) * normalize(target), axis=-1)
    return jnp.sum(jnp.where(mask, losses, 0)) / jnp.maximum(jnp.sum(mask), 1)


def muzero_consistency_loss(model, params, batch, predicted_latents, remat=False):
    """Match each imagined next state to its real observation; exclude reset tails."""
    import jax
    import jax.numpy as jnp

    def flatten_time(x):
        return x.reshape((-1, *x.shape[2:]))

    observation = batch.observation
    if observation is None:
        from nanoalphazero.buffers import combine_observation_vmap
        observation = combine_observation_vmap(
            flatten_time(batch.board_bool[:, 1:]),
            flatten_time(batch.board_float[:, 1:]),
        )
    else:
        observation = flatten_time(observation[:, 1:]).astype(jnp.float32)

    def target_projection(params, observation):
        latent = model.apply({"params": params}, observation, method=model.encode)
        return model.apply({"params": params}, latent, method=model.project)

    if remat:
        target_projection = jax.checkpoint(target_projection)
    target = jax.lax.stop_gradient(target_projection(params, observation))
    prediction = model.apply(
        {"params": params}, flatten_time(predicted_latents), predict=True, method=model.project,
    )
    # A recorded final action has no stored terminal next observation. Its
    # following slot belongs to the next game and must never become a target.
    mask = batch.policy_mask[:, 1:].reshape(-1)
    return masked_consistency_loss(prediction, target, mask)


def muzero_loss(model, params, batch, remat=False, *, self_consistency_weight=0.0):
    """Policy KL + value/reward MSE, with optional temporal consistency."""
    import jax.numpy as jnp

    predictions = unroll_muzero_predictions(
        model, params, batch, remat, return_latents=self_consistency_weight > 0,
    )
    logits, values, rewards = predictions[:3]

    def mean_masked(values, mask):
        return jnp.sum(jnp.where(mask, values, 0)) / jnp.maximum(jnp.sum(mask), 1)

    kl = muzero_policy_kl(logits, batch.action_weights, batch.k_indices)
    loss_pi = mean_masked(kl, batch.policy_mask)
    loss_v = mean_masked((values - batch.reward) ** 2, batch.value_mask)
    loss_r = mean_masked((rewards - batch.transition_reward) ** 2, batch.reward_mask)
    total = loss_pi + loss_v + loss_r
    metrics = dict(loss_pi=loss_pi, loss_v=loss_v, loss_r=loss_r)
    if self_consistency_weight > 0:
        consistency = muzero_consistency_loss(model, params, batch, predictions[3], remat)
        total += self_consistency_weight * consistency
        metrics.update(loss_consistency=consistency,
                       **{"consistency/n_valid": jnp.sum(batch.policy_mask[:, 1:]).astype(jnp.float32)})
    return total, metrics


def make_muzero_train(config, model, model_state, data_sharding=None):
    """Expose the established train_fn(model_ts, batch, is_warmup) interface."""
    import jax
    import jax.numpy as jnp
    import optax
    from flax.traverse_util import flatten_dict, unflatten_dict
    from nanoalphazero.core import CustomTrainState

    rate = config["learning_rate"]
    warmup = config.get("lr_warmup_steps", 0)
    if warmup:
        rate = optax.join_schedules(
            [optax.linear_schedule(1e-6, rate, warmup), optax.constant_schedule(rate)],
            [warmup],
        )
    mask = None
    if config.get("weight_decay_kernels_only", False):
        mask = unflatten_dict({
            path: path[-1] in ("kernel", "kernel_3x3", "kernel_1x1")
            for path in flatten_dict(model_state["params"])
        })
    tx = optax.chain(
        optax.clip_by_global_norm(config.get("max_grad_norm", 1.0)),
        optax.adamw(rate, weight_decay=config.get("weight_decay", 1e-4), mask=mask),
    )
    model_ts = CustomTrainState.create(
        apply_fn=model.apply, params=model_state["params"], tx=tx,
        key=jax.random.PRNGKey(0), n_updates=jnp.int32(0),
    )
    # TrainState.create uses a Python step; make the initial carry match JIT output.
    model_ts = model_ts.replace(step=jnp.int32(0))
    if data_sharding is not None:
        replicated = jax.sharding.NamedSharding(data_sharding.mesh, jax.sharding.PartitionSpec())
        model_ts = jax.device_put(model_ts, replicated)

    def train_fn(state, batch, is_warmup):
        def objective(params):
            return muzero_loss(
                model, params, batch, config.get("muzero_remat_unroll", False),
                self_consistency_weight=(config.get("self_consistency_weight", 1.0)
                                         if config.get("exp_self_consistency", False) else 0.0),
            )

        (loss, metrics), grads = jax.value_and_grad(objective, has_aux=True)(state.params)
        updated = state.apply_gradients(grads=grads).replace(n_updates=state.n_updates + 1)
        updated = jax.lax.cond(is_warmup, lambda: state, lambda: updated)
        return updated, (loss, metrics)

    return train_fn, model_ts


def make_muzero(config, rng, data_sharding=None):
    """Assemble MuZero; expose each component for inspection and independent use."""
    import jax
    import jax.numpy as jnp
    from nanoalphazero.buffers import (
        get_dummy_muzero_selfplay_output, get_dummy_muzero_training_sample,
        make_muzero_selfplay_buffer, make_replay_buffer, check_muzero_storage_budget,
    )
    from nanoalphazero.config import validate_muzero_config
    from nanoalphazero.core import make_env, RunnerState, DATA_PARALLEL_SHARDING
    from nanoalphazero.mcts import make_muzero_mcts, make_muzero_compression_probe
    from nanoalphazero.model import make_muzero_model

    config = dict(config)
    if config.get("enable_sharding", False):
        data_sharding = data_sharding or DATA_PARALLEL_SHARDING
    else:
        data_sharding = None
    replicated = None
    if data_sharding is not None:
        replicated = jax.sharding.NamedSharding(data_sharding.mesh, jax.sharding.PartitionSpec())

    env = make_env(config)
    config.update(game_obs_shape=list(env.obs_shape), game_num_actions=env.num_actions)
    validate_muzero_config(config, env.num_actions, data_sharding.mesh.size if data_sharding else 1)
    # Legacy uint32 keys serialize directly to safetensors.
    rng, model_rng = jax.random.split(jax.random.key_data(rng))
    model, model_state = make_muzero_model(config, model_rng, replicated)
    devices = list(data_sharding.mesh.devices.flat) if data_sharding is not None else [jax.devices()[0]]
    storage = check_muzero_storage_budget(config, devices)
    run_mcts_fn = make_muzero_mcts(config, env, model, data_sharding)
    selfplay_fn, selfplay_state = make_muzero_selfplay(config, env, run_mcts_fn, data_sharding)
    selfplay_buffer, selfplay_buffer_state = make_muzero_selfplay_buffer(
        config, get_dummy_muzero_selfplay_output(config), data_sharding,
    )
    replay_buffer, replay_buffer_state = make_replay_buffer(
        config, get_dummy_muzero_training_sample(config), data_sharding,
    )
    train_fn, model_ts = make_muzero_train(config, model, model_state, data_sharding)
    runner_state = RunnerState(
        model_ts, selfplay_state, selfplay_buffer_state, replay_buffer_state, rng,
    )
    if data_sharding is not None:
        # Scalars added by the MuZero buffer also need replicated placement.
        selfplay_buffer_state = jax.device_put(selfplay_buffer_state, jax.tree.map(
            lambda x: data_sharding if x.ndim else replicated, selfplay_buffer_state,
        ))
        runner_state = runner_state._replace(
            selfplay_buffer_state=selfplay_buffer_state, rng=jax.device_put(rng, replicated),
        )

    @partial(jax.jit, donate_argnums=(0,))
    def run_fn(state, is_warmup=False):
        # 1. Advance every lane. Unfinished games survive this cycle boundary.
        def selfplay_step(state, _):
            rng, key = jax.random.split(state.rng)
            games, output, _ = selfplay_fn(key, state.selfplay_state, state.model_ts.params)
            buffered, _ = selfplay_buffer.add_backfill(
                state.selfplay_buffer_state, output, games.env_state.terminated,
                games.env_state.rewards,
            )
            # auto_reset discards the final observation on truncation. Do not
            # invent bootstrap targets. Built-in games terminate instead.
            buffered = buffered.replace(
                truncated_count=buffered.truncated_count + jnp.sum(games.env_state.truncated),
            )
            return state._replace(
                selfplay_state=games, selfplay_buffer_state=buffered, rng=rng,
            ), None

        state, _ = jax.lax.scan(selfplay_step, state, None, length=config["cycle_n_selfplay"])

        # 2. AlphaZero consume chooses fresh starts. MuZero returns K-step items.
        n_slices = state.selfplay_buffer_state.num_valid_consumable // config["selfplay_buffer_consume_size"]

        def drain_step(_, state):
            buffered, samples, _ = selfplay_buffer.consume(state.selfplay_buffer_state)
            replayed = replay_buffer.add(state.replay_buffer_state, samples)
            return state._replace(selfplay_buffer_state=buffered, replay_buffer_state=replayed)

        state = jax.lax.fori_loop(0, n_slices, drain_step, state)
        buffered = state.selfplay_buffer_state
        errors = buffered.overwritten_count + buffered.missing_count + buffered.truncated_count

        # 3. A replay item is already an unroll; Flashbax samples one item.
        def train_step(state, _):
            rng, key = jax.random.split(state.rng)
            batch = replay_buffer.sample(state.replay_buffer_state, key).experience
            batch = jax.tree.map(lambda x: x[:, 0], batch)
            model_ts, (loss, metrics) = train_fn(state.model_ts, batch, is_warmup)
            return state._replace(model_ts=model_ts, rng=rng), {"total_loss": loss, **metrics}

        def train_cycle(state):
            state, metrics = jax.lax.scan(train_step, state, None, length=config["cycle_n_train"])
            return state, jax.tree.map(lambda x: jnp.mean(x), metrics)

        ready = replay_buffer.can_sample(state.replay_buffer_state) & (errors == 0) & ~jnp.asarray(is_warmup)
        empty_metrics = {name: jnp.float32(0) for name in ("total_loss", "loss_pi", "loss_v", "loss_r")}
        if config.get("exp_self_consistency", False):
            empty_metrics.update(loss_consistency=jnp.float32(0),
                                 **{"consistency/n_valid": jnp.float32(0)})
        state, metrics = jax.lax.cond(ready, train_cycle, lambda s: (s, empty_metrics), state)
        return state, {
            **metrics, "drain/n_slices": n_slices,
            "drain/remaining_fresh": buffered.num_valid_consumable,
            "selfplay_buffer/error_count": errors,
            "selfplay_buffer/overwritten_count": buffered.overwritten_count,
            "selfplay_buffer/missing_count": buffered.missing_count,
            "selfplay_buffer/truncated_count": buffered.truncated_count,
            "runner_state/n_updates": state.model_ts.n_updates,
            "runner_state/n_selfplay_steps": state.selfplay_state.step_count,
            "train/ready": ready,
        }

    return SimpleNamespace(
        config=config, env=env, model=model, run_fn=run_fn, runner_state=runner_state,
        run_mcts_fn=run_mcts_fn, selfplay_fn=selfplay_fn, selfplay_buffer=selfplay_buffer,
        replay_buffer=replay_buffer, train_fn=train_fn, data_sharding=data_sharding,
        buffer_storage=storage,
        compression_probe=(make_muzero_compression_probe(config, env, model)
                           if config.get("exp_bnk_action_weights", False) else None),
    )


def run_muzero(algorithm, *, num_iters=None, checkpoint_path=None, resume=None, evaluator=None):
    """Compile once, warm up replay, then run and monitor persistent cycles."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from nanoalphazero.checkpoint import (
        save_muzero_checkpoint, load_muzero_checkpoint, muzero_checkpoint_payload,
    )

    config = algorithm.config
    state = algorithm.runner_state
    start_cycle = 0
    include_replay = config.get("muzero_checkpoint_replay", True)
    if resume is not None:
        payload, saved_config = load_muzero_checkpoint(
            resume, muzero_checkpoint_payload(state, 0, include_replay)
        )
        if saved_config != config:
            raise ValueError("MuZero resume requires identical resolved configuration")
        start_cycle = int(payload["cycle"])
        if include_replay:
            state = payload["runner_state"]
        else:
            state = state._replace(model_ts=payload["model_ts"], rng=payload["rng"])
            print("Compact checkpoint restored; restarting games and warming up empty buffers", flush=True)
        if algorithm.data_sharding is not None:
            # Restore the original placement, including replicated scalar counters.
            placement = jax.tree.map(lambda x: x.sharding, algorithm.runner_state)
            state = jax.device_put(state, placement)

    def check(metrics):
        if int(metrics["selfplay_buffer/error_count"]):
            raise RuntimeError(
                "MuZero self-play buffer lost required transitions or encountered "
                "unsupported truncation. No optimizer update was applied in this cycle. "
                "Inspect buffer counters and increase selfplay_buffer_max_len if needed. "
                f"Counters: { {name: int(value) for name, value in metrics.items() if name.startswith('selfplay_buffer/')} }"
            )
        if not all(np.isfinite(value) for value in metrics.values()):
            raise RuntimeError("Nonfinite MuZero training metrics")

    run = None
    # Compile the entire persistent cycle before spending time on warmup.
    # Warmup is a dynamic boolean, so the same executable also covers training.
    print("Compiling MuZero cycle", flush=True)
    started = time.monotonic()
    run_cycle = algorithm.run_fn.lower(state, jnp.bool_(True)).compile()
    analysis = run_cycle.memory_analysis()
    compiled_memory = {name: getattr(analysis, name, 0) for name in (
        "argument_size_in_bytes", "output_size_in_bytes", "alias_size_in_bytes", "temp_size_in_bytes",
    )}
    print(json.dumps({"compile_seconds": time.monotonic() - started,
                      "compiled_memory": compiled_memory}), flush=True)
    if config.get("enable_trackio", False):
        try:
            import trackio
        except ModuleNotFoundError as exc:
            if exc.name != "trackio":
                raise
            raise RuntimeError(
                "Trackio logging requires the optional dependency: "
                "uv run --extra trackio python muzero.py --enabled-trackio"
            ) from exc
        run = trackio.init(project="nanoAlphaZero-muzero", config=config)
    elif config.get("enable_wandb", False):
        import wandb
        run = wandb.init(project="nanoAlphaZero-muzero", config=config)
    try:
        head_probe = None
        if evaluator is not None:
            evaluator.run(start_cycle, state, force=True)
        if resume is None or not include_replay:
            num_warmup_cycles = (
                config.get("replay_buffer_warmup_steps", 100) // config["cycle_n_selfplay"]
            )
            for warmup in range(num_warmup_cycles):
                started = time.monotonic()
                state, metrics = run_cycle(state, jnp.bool_(True))
                check(jax.device_get(metrics))
                print(json.dumps({"warmup_cycle": warmup + 1,
                                  "timing/cycle_seconds": time.monotonic() - started}), flush=True)
        # Match AlphaZero's host-loop budget, including its step counter on resume.
        budget = config["num_iters"] if num_iters is None else num_iters
        n_cycles = max(0, (budget - int(state.model_ts.step)) // config["cycle_n_selfplay"])
        cycles = start_cycle + n_cycles
        cycle = start_cycle
        for cycle in range(start_cycle + 1, cycles + 1):
            started = time.monotonic()
            state, metrics = run_cycle(state, jnp.bool_(False))
            metrics = {name: float(value) for name, value in jax.device_get(metrics).items()}
            # device_get synchronizes execution; exclude diagnostics and disk I/O.
            metrics["timing/cycle_seconds"] = time.monotonic() - started
            diagnostic_logs = {}
            period = config.get("diagnostic_period", 50)
            if period and (cycle == start_cycle + 1 or cycle % period == 0 or cycle == cycles):
                if config.get("env_id", "").startswith("hex"):
                    from nanoalphazero.eval.muzero import make_muzero_head_probe, head_diagnostic_logs
                    if head_probe is None:
                        head_probe = make_muzero_head_probe(algorithm.model, algorithm.env, config)
                    head_metrics, tables, histograms = head_probe(state.model_ts.params)
                    metrics.update(head_metrics)
                    if run is not None:
                        backend = "trackio" if config.get("enable_trackio", False) else "wandb"
                        diagnostic_logs = head_diagnostic_logs(tables, histograms, backend)
                if algorithm.compression_probe is not None:
                    # Inspection has an independent key and never advances training RNG.
                    probe = algorithm.compression_probe(
                        state.model_ts.params, state.selfplay_state.env_state,
                        jax.random.fold_in(state.rng, cycle),
                    )
                    metrics.update({name: float(value) for name, value in jax.device_get(probe).items()})
                devices = (list(algorithm.data_sharding.mesh.devices.flat)
                           if algorithm.data_sharding is not None else [jax.devices()[0]])
                memory = [device.memory_stats() or {} for device in devices]
                for name in ("bytes_in_use", "peak_bytes_in_use", "bytes_limit"):
                    if all(name in stats for stats in memory):
                        metrics[f"memory/{name}_max"] = max(stats[name] for stats in memory)
            check(metrics)
            if evaluator is not None:
                metrics.update(evaluator.run(cycle, state, force=cycle == cycles))
            print(json.dumps({"cycle": cycle, **metrics}), flush=True)
            if run is not None:
                run.log({**metrics, **diagnostic_logs}, step=cycle)
            if checkpoint_path and (cycle == start_cycle + 1 or (
                config.get("ckpt_period") and cycle % config["ckpt_period"] == 0
            )):
                save_muzero_checkpoint(
                    checkpoint_path, muzero_checkpoint_payload(state, cycle, include_replay), config
                )
        if checkpoint_path:
            save_muzero_checkpoint(
                checkpoint_path, muzero_checkpoint_payload(state, cycle, include_replay), config
            )
    finally:
        if run is not None:
            run.finish()
    return state


def main(argv=None):
    from nanoalphazero.cli import parse_muzero_args
    from nanoalphazero.config import get_muzero_config

    args = parse_muzero_args(argv)
    # Backend selection precedes imports of core.py, which creates its device mesh.
    if args.platform:
        os.environ["JAX_PLATFORMS"] = args.platform
    overrides = {name: value for name, value in vars(args).items()
                 if value is not None and name not in ("env", "save", "resume", "platform")
                 and not name.startswith("hex_eval_")}
    config = get_muzero_config(args.env, **overrides)
    pool = None
    try:
        # Start subprocess engines before JAX/libtpu owns accelerator resources.
        if args.hex_eval_period:
            from nanoalphazero.eval.hex.engine import MoHexPool
            size = config["boardsize"]
            pool = MoHexPool(args.hex_eval_engine_path, args.hex_eval_engine_config, size, size * size)
        import jax
        from nanoalphazero.eval.hex.training import MuZeroHexEvaluator
        algorithm = make_muzero(config, jax.random.PRNGKey(config["seed"]))
        evaluator = (MuZeroHexEvaluator(algorithm, pool, args.hex_eval_period, args.hex_eval_output)
                     if pool is not None else None)
        run_muzero(algorithm, checkpoint_path=args.save, resume=args.resume, evaluator=evaluator)
    finally:
        if pool is not None:
            pool.close()


if __name__ == "__main__":
    main()
