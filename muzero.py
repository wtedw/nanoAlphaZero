"""Build MuZero from the nanoalphazero components.

The cycle is visible here: advance persistent games, consume fresh starts as
K-step training samples, then train from replay. Return calculation, sequence
indexing, terminal masks, and differentiable unrolling live in the components.

Run with ``uv run python muzero.py --env hex4``. Use --help for configuration.
The historical portable export is muzero-singlefile.py.
"""

from functools import partial
from types import SimpleNamespace


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


def make_muzero(config, rng, data_sharding=None):
    """Assemble MuZero; expose each component for inspection and independent use."""
    import jax
    import jax.numpy as jnp
    from nanoalphazero.buffers import (
        get_dummy_muzero_selfplay_output, get_dummy_muzero_training_sample,
        make_muzero_selfplay_buffer, make_replay_buffer, estimate_muzero_storage,
    )
    from nanoalphazero.core import make_env, RunnerState, DATA_PARALLEL_SHARDING
    from nanoalphazero.mcts import make_muzero_mcts, make_muzero_compression_probe
    from nanoalphazero.model import make_muzero_model
    from nanoalphazero.training import make_muzero_train

    config = dict(config)
    positive = (
        "selfplay_batch_size", "train_batch_size", "selfplay_buffer_consume_size",
        "selfplay_buffer_max_len", "replay_buffer_max_len", "cycle_n_selfplay",
        "cycle_n_train", "muzero_unroll_steps",
    )
    for name in positive:
        if config[name] < 1:
            raise ValueError(f"{name} must be positive")
    if not 0 < config["muzero_discount"] <= 1:
        raise ValueError("muzero_discount must be in (0, 1]")
    if not 0 < config.get("muzero_memory_fraction", 0.75) <= 1:
        raise ValueError("muzero_memory_fraction must be in (0, 1]")
    for name in ("exp_bnk_action_weights", "muzero_checkpoint_replay"):
        if not isinstance(config.get(name, False), bool):
            raise ValueError(f"{name} must be boolean")
    if config["selfplay_buffer_max_len"] <= config["muzero_unroll_steps"]:
        raise ValueError("Self-play storage must hold at least K+1 positions")
    if config["selfplay_buffer_consume_size"] > config["selfplay_batch_size"] * config["selfplay_buffer_max_len"]:
        raise ValueError("consume_size exceeds self-play storage")
    if config["selfplay_buffer_add_batch_size"] != config["selfplay_batch_size"]:
        raise ValueError("Self-play buffer lane count must match selfplay_batch_size")
    if config["replay_buffer_add_batch_size"] != config["selfplay_buffer_consume_size"]:
        raise ValueError("Replay insertion batch must match selfplay_buffer_consume_size")
    if config["replay_buffer_sample_batch_size"] != config["train_batch_size"]:
        raise ValueError("Replay sample batch must match train_batch_size")
    for prefix in ("selfplay_buffer", "replay_buffer"):
        if not 1 <= config[f"{prefix}_min_len"] <= config[f"{prefix}_max_len"]:
            raise ValueError(f"{prefix}_min_len must fit within its capacity")
    if config.get("muzero_warmup_cycles", 1) < 0 or config["num_iters"] < 0:
        raise ValueError("Warmup cycles and num_iters cannot be negative")
    if config.get("muzero_diagnostic_period", 50) < 0:
        raise ValueError("muzero_diagnostic_period cannot be negative")

    if config.get("enable_sharding", False):
        data_sharding = data_sharding or DATA_PARALLEL_SHARDING
    else:
        data_sharding = None
    replicated = None
    if data_sharding is not None:
        replicated = jax.sharding.NamedSharding(data_sharding.mesh, jax.sharding.PartitionSpec())
        for name in ("selfplay_batch_size", "train_batch_size", "selfplay_buffer_consume_size"):
            if config[name] % data_sharding.mesh.size:
                raise ValueError(f"{name} must be divisible by the device count")

    env = make_env(config)
    config.update(game_obs_shape=list(env.obs_shape), game_num_actions=env.num_actions)
    if config.get("exp_bnk_action_weights", False):
        top_k = config["mcts_num_k_actions"]
        if type(top_k) is not int or not 1 <= top_k <= env.num_actions:
            raise ValueError("mcts_num_k_actions must be an integer within the action space")
    if not 1 <= config["mcts_num_survivors"] <= config["mcts_num_root_considered"] <= env.num_actions:
        raise ValueError("Search requires 1 <= survivors <= roots <= action count")
    # Legacy uint32 keys serialize directly to safetensors.
    rng, model_rng = jax.random.split(jax.random.key_data(rng))
    model, model_state = make_muzero_model(config, model_rng, replicated)
    import json
    devices = list(data_sharding.mesh.devices.flat) if data_sharding is not None else [jax.devices()[0]]
    storage = estimate_muzero_storage(config, len(devices))
    print(json.dumps({"buffer_storage": storage}), flush=True)
    per_device = sum(spec["bytes_per_device"] for spec in storage.values())
    for device in devices:
        stats = device.memory_stats() or {}
        limit = stats.get("bytes_limit")
        if limit is not None and per_device + stats.get("bytes_in_use", 0) > config.get("muzero_memory_fraction", 0.75) * limit:
            raise ValueError(
                f"MuZero buffers require {per_device / 2**30:.2f} GiB/device and exceed "
                "the persistent-memory budget. Reduce replay/self-play capacity; "
                "replay items each contain a full unroll. Temporary arrays are additional."
            )
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


def main(argv=None):
    from nanoalphazero.cli import parse_muzero_args
    from nanoalphazero.config import get_muzero_config

    args = parse_muzero_args(argv)
    # Backend selection precedes imports of core.py, which creates its device mesh.
    import os
    if args.platform:
        os.environ["JAX_PLATFORMS"] = args.platform
    import jax
    from nanoalphazero.training import run_muzero

    overrides = {name: value for name, value in vars(args).items()
                 if value is not None and name not in ("env", "save", "resume", "platform")}
    config = get_muzero_config(args.env, **overrides)
    algorithm = make_muzero(config, jax.random.PRNGKey(config["seed"]))
    run_muzero(algorithm, checkpoint_path=args.save, resume=args.resume)


if __name__ == "__main__":
    main()
