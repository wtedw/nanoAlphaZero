"""Compare reversible replay layouts without changing the training entry point.

CPU correctness control (four virtual devices):
    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \
      uv run pytest -q tests/test_muzero_replay_layout.py

After the TPU is free, run each layout in a SEPARATE process, retaining logs:
    uv run python scripts/probe_muzero_replay_layout.py --platform tpu \
      --layout baseline --mode cycle --batch 4096 --capacity 512 \
      --depth 10 --width 256
    # Repeat with --layout features, then --layout matrix.

Chess defaults to 512 self-play slots, K=5, BNK-128 targets, consistency,
rematerialization, and 20 self-play/20 optimizer steps per cycle. Sizes, cycle
lengths, consistency, and replay seeding are configurable. Synthetic replay
exercises training immediately, NOT learning. --no-seed-replay waits for real
completed games; inspect drain counts and update counts rather than assuming
every cycle trains. --warmup-iterations excludes calls from timing summaries,
but these calls DO train (unlike the production replay warmup).
It writes no checkpoints or tracker data. No production factory is changed on disk.
For HLO diagnostics, optionally set XLA_FLAGS=--xla_dump_to=NEW_DIRECTORY, but
large full-cycle dumps require substantial free disk space.

Observed baseline: JAX 0.8.1/libtpu 0.0.30, four TPU v4, b10c256, batch 4096,
replay capacity 512, K=5. Compile failed: 37.58 GiB versus 30.75 GiB HBM.
The following are compiler-reported TEMPORARY allocations, not storage estimates:
  consistency_board_float: bf16[1024,512,5,8,8,2], 640 MiB -> 10 GiB
  consistency_board_bool:  u8[1024,512,5,936], 2.29 GiB -> 4 GiB
  action_weights:          f32[1024,512,6,128], 1.5 GiB -> 2 GiB
  k_indices:               u16[1024,512,6,128], 768 MiB -> 1 GiB
  board_bool:              u8[1024,512,936], 468 MiB -> 512 MiB
  masks:                   pred[1024,512,5 or 6], 2.5/3 MiB -> 64 MiB each
The root board_float allocation was already unpadded (128 MiB).

Candidate layouts:
  features: flatten each stored sample field, e.g. (6,8,8,2) -> (768,).
    Flashbax gathers flat features; only the sampled batch is reshaped for loss.
    This also folds K into the BNK and packed-observation feature axes.
  matrix: additionally store each field as (lanes, capacity * features), so
    small masks do not remain trailing dimensions of persistent state.
    Flashbax still sees (lanes, capacity, features) internally; XLA may undo
    the benefit, which is why the full-cycle compiler result is authoritative.

Both variants retain every bit, dtype, item, RNG decision, and time index.
Logical storage bytes are unchanged. Checkpoint layouts would change, so these
remain isolated experiments pending production integration/migration handling.

Historical TPU comparison at commit 81716d8 (separate root/future observation
fields), on 2026-10-07: both candidates executed 40 optimizer updates
with finite losses and zero buffer errors at the failing baseline's dimensions.
XLA's padded temporary payload fell from 21.69 GiB (9.62 GiB unpadded) to
10.01 GiB (9.74 GiB unpadded) for features and 6.96 GiB (6.96 GiB unpadded)
for matrix. Matrix also had higher initialization allocation peaks, so do not
infer overall peak memory solely from its lower execution reservation.
These two-cycle synthetic-replay probes did not exercise completed-game drain
or long-run wraparound. Results: evals/muzero/replay-layout-padding-20261007.json.
The probe now uses unified observation sequences; repeat TPU measurements before
applying those historical numbers to this new schema.
Use scripts/benchmark_muzero_layouts.py to automate fresh-process comparisons.
Replay mode fills the complete ring outside timing. Add uses a dynamic input
batch; sample returns all fields; roundtrip includes observation decoding and a
policy checksum (unused gathered fields can be optimized out). Cycle mode is
the end-to-end control, including self-play, drain, sampling and optimization.
Only replay changes layout; self-play is always original. Per-field shard sizes
are logical sizes, NOT measured TPU tile allocation. Compiler memory_analysis,
allocator counters, and optional raw HLO/driver reports remain separate.
See https://docs.cloud.google.com/tpu/docs/performance-guide for TPU tiling.
"""

import argparse
from functools import partial
import importlib.util
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import statistics
import time
from unittest.mock import patch


def make_layout_replay_buffer(config, dummy, data_sharding=None, *, layout, factory=None):
    """Adapt only storage shapes; retain Flashbax's add/sample implementation."""
    import jax
    from nanoalphazero.buffers import make_replay_buffer

    factory = factory or make_replay_buffer
    if layout in ("baseline", "original"):
        return factory(config, dummy, data_sharding)
    if layout not in ("features", "matrix"):
        raise ValueError(layout)
    capacity = config["replay_buffer_max_len"]

    def encode(batch, leading):
        return jax.tree.map(lambda x: x.reshape((*x.shape[:leading], -1)), batch)

    def decode(batch):
        return jax.tree.map(lambda x, example: x.reshape((*x.shape[:2], *example.shape)), batch, dummy)

    inner, state = factory(config, encode(dummy, 0), data_sharding)

    def store(state):
        if layout == "matrix":
            return state.replace(experience=jax.tree.map(
                lambda x: x.reshape((x.shape[0], -1)), state.experience))
        return state

    def restore(state):
        if layout == "matrix":
            return state.replace(experience=jax.tree.map(
                lambda x: x.reshape((x.shape[0], capacity, -1)), state.experience))
        return state

    def init(example):
        return store(inner.init(encode(example, 0)))

    @partial(jax.jit, donate_argnums=(0,))
    def add(state, batch):
        return store(inner.add(restore(state), encode(batch, 2)))

    @jax.jit
    def sample(state, key):
        result = inner.sample(restore(state), key)
        return result.replace(experience=decode(result.experience))

    return inner.replace(init=init, add=add, sample=sample), store(state)


def check_tpu_ownership():
    """Check before importing JAX; never stop an owner or remove its lock."""
    paths = [Path('/tmp/libtpu_lockfile'), *Path('/dev').glob('accel*')]
    paths = [str(path) for path in paths if path.exists()]
    if not paths:
        return
    result = subprocess.run(['fuser', *paths], capture_output=True, text=True)
    if result.returncode == 0:
        raise RuntimeError(f'TPU occupied; leave its owner untouched: {result.stdout.strip()}')
    if result.returncode != 1 or result.stderr.strip():
        raise RuntimeError(f'Could not establish TPU ownership: {result.stderr.strip()}')


def make_synthetic_batch(dummy, batch_size, sharding=None):
    """Nonzero supervised targets; not real game trajectories."""
    import jax
    import jax.numpy as jnp

    sample = dummy.replace(
        action_weights=jnp.full_like(dummy.action_weights, 1 / dummy.action_weights.shape[-1]),
        k_indices=(None if dummy.k_indices is None else jnp.broadcast_to(
            jnp.arange(dummy.k_indices.shape[-1], dtype=dummy.k_indices.dtype), dummy.k_indices.shape)),
        reward=jnp.full_like(dummy.reward, .25),
        policy_mask=jnp.ones_like(dummy.policy_mask),
        value_mask=jnp.ones_like(dummy.value_mask),
        reward_mask=jnp.ones_like(dummy.reward_mask),
    )
    batch = jax.tree.map(lambda x: jnp.broadcast_to(x, (batch_size, 1, *x.shape)), sample)
    return jax.device_put(batch, sharding) if sharding is not None else batch


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--platform', choices=('cpu', 'tpu'), default='cpu')
    parser.add_argument('--layout', choices=('original', 'baseline', 'features', 'matrix'), required=True)
    parser.add_argument('--mode', choices=('replay', 'cycle'), default='replay')
    parser.add_argument('--operation', choices=('add', 'sample', 'roundtrip'), default='roundtrip')
    parser.add_argument('--env', choices=('chess', 'hex4', 'hex5'), default='chess')
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--capacity', type=int, default=16)
    parser.add_argument('--depth', type=int, default=1)
    parser.add_argument('--width', type=int, default=8)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--warmup-iterations', type=int, default=2)
    parser.add_argument('--selfplay-capacity', type=int, default=512)
    parser.add_argument('--selfplay-steps', type=int, default=20)
    parser.add_argument('--train-steps', type=int, default=20)
    parser.add_argument('--unroll', type=int, default=5)
    parser.add_argument('--consistency', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--seed-replay', action=argparse.BooleanOptionalAction, default=True,
                        help='Seed synthetic replay in cycle mode; disable to wait for real completed games.')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(argv)
    if args.iterations < 1 or args.warmup_iterations < 0:
        parser.error('iterations must be positive and warmup-iterations nonnegative')
    if args.platform == 'tpu':
        check_tpu_ownership()
    os.environ['JAX_PLATFORMS'] = args.platform

    import jax
    import jax.numpy as jnp
    from nanoalphazero import buffers
    from nanoalphazero.config import get_muzero_config
    from nanoalphazero.core import DATA_PARALLEL_SHARDING, make_env

    def emit(**data):
        print(json.dumps(data), flush=True)

    def memory():
        return [device.memory_stats() or {} for device in jax.devices()]

    cfg = get_muzero_config(
        args.env, conv_depth=args.depth, conv_width=args.width,
        selfplay_batch_size=args.batch, train_batch_size=args.batch,
        selfplay_buffer_consume_size=args.batch, selfplay_buffer_max_len=args.selfplay_capacity,
        selfplay_buffer_min_len=1, replay_buffer_min_len=1,
        replay_buffer_max_len=args.capacity, exp_self_consistency=args.consistency,
        cycle_n_selfplay=args.selfplay_steps, cycle_n_train=args.train_steps,
        muzero_unroll_steps=args.unroll,
        muzero_exploration_losses="none",
    )
    env = make_env(cfg)
    cfg.update(game_obs_shape=list(env.obs_shape), game_num_actions=env.num_actions)
    dummy = buffers.get_dummy_muzero_training_sample(cfg)
    factory = partial(make_layout_replay_buffer, layout=args.layout, factory=buffers.make_replay_buffer)
    sharding = DATA_PARALLEL_SHARDING
    batch = make_synthetic_batch(dummy, args.batch, sharding)
    def version(name):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    emit(stage='start', layout=args.layout, mode=args.mode, config=cfg,
         options=vars(args), versions={name: version(name) for name in ('jax', 'jaxlib', 'libtpu', 'flashbax')},
         devices=[{'id': d.id, 'kind': d.device_kind, 'platform': d.platform} for d in jax.devices()])
    initialized_at = time.monotonic()

    if args.mode == 'cycle':
        spec = importlib.util.spec_from_file_location('muzero_layout_runner', Path(__file__).resolve().parents[1] / 'muzero.py')
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        # Process-local substitution, never modify the package or another process.
        with patch.object(buffers, 'make_replay_buffer', factory):
            algorithm = runner.make_muzero(cfg, jax.random.PRNGKey(args.seed))
        state = algorithm.runner_state
        if args.seed_replay:
            state = state._replace(replay_buffer_state=algorithm.replay_buffer.add(state.replay_buffer_state, batch))
        fn, extra = algorithm.run_fn, jnp.bool_(False)
        stored = state.replay_buffer_state
    else:
        replay, state = factory(cfg, dummy, sharding)
        # Populate all T insertion slots, including wrap, outside timed calls.
        @partial(jax.jit, donate_argnums=(0,))
        def fill(state, batch):
            return jax.lax.fori_loop(0, args.capacity + 1, lambda _, s: replay.add(s, batch), state)

        state = fill(state, batch)

        @partial(jax.jit, donate_argnums=(0,) if args.operation != 'sample' else ())
        def fn(state, key, batch):
            if args.operation != 'sample':
                state = replay.add(state, batch)
            if args.operation == 'add':
                return state, {'index': state.current_index}
            sample = replay.sample(state, key).experience
            if args.operation == 'sample':
                # Return every field to prevent dead-code elimination of gathers.
                return sample
            if sample.observation is None:
                observed = buffers.combine_observation_vmap(
                    sample.board_bool.reshape((-1, *dummy.board_bool.shape[1:])),
                    sample.board_float.reshape((-1, *dummy.board_float.shape[1:])))
            else:
                observed = sample.observation.astype(jnp.float32)
            return state, {'checksum': jnp.sum(observed) + jnp.sum(sample.action_weights)}

        extra = jax.random.PRNGKey(1)
        stored = state

    jax.block_until_ready(state)
    def fields(experience):
        return {name: {'shape': list(x.shape), 'dtype': str(x.dtype),
                'logical_bytes': x.size * x.dtype.itemsize,
                'local_shards': [{'shape': list(s.data.shape), 'logical_bytes': s.data.nbytes,
                                  'device': str(s.device)} for s in x.addressable_shards]}
                for name, x in vars(experience).items() if x is not None}

    emit(stage='storage', fields=fields(stored.experience),
         selfplay_fields=fields(state.selfplay_buffer_state.experience) if args.mode == 'cycle' else None,
         initialization_seconds=time.monotonic()-initialized_at, memory=memory())
    started = time.monotonic()
    emit(stage='compiling')
    call_args = (state, extra) if args.mode == 'cycle' else (state, extra, batch)
    executable = fn.lower(*call_args).compile()
    analysis = executable.memory_analysis()
    sizes = {name: getattr(analysis, name, 0) for name in (
        'argument_size_in_bytes', 'output_size_in_bytes', 'alias_size_in_bytes', 'temp_size_in_bytes')}
    emit(stage='compiled', seconds=time.monotonic()-started, compiled_memory=sizes, memory=memory())
    durations = []
    updates = 0
    for iteration in range(args.warmup_iterations + args.iterations):
        if args.mode == 'replay':
            extra = jax.random.fold_in(jax.random.PRNGKey(args.seed), iteration)
        started = time.monotonic()
        if args.mode == 'cycle':
            state, (metrics, arrays) = executable(state, extra)
            jax.block_until_ready(arrays)
        elif args.operation == 'sample':
            metrics = executable(state, extra, batch)
        else:
            state, metrics = executable(state, extra, batch)
        jax.block_until_ready((state, metrics))
        elapsed = time.monotonic()-started
        if args.mode == 'replay' and args.operation == 'sample':
            # Keep reduction/host transfer outside timed sample execution.
            metrics = {'checksum': sum(float(jnp.sum(x.astype(jnp.float32))) for x in jax.tree.leaves(metrics))}
        else:
            metrics = {name: float(value) for name, value in jax.device_get(metrics).items()}
        assert all(math.isfinite(value) for value in metrics.values())
        if args.mode == 'cycle':
            assert metrics['selfplay_buffer/error_count'] == 0
            updates += int(metrics['train/ready']) * cfg['cycle_n_train']
            assert metrics['runner_state/n_updates'] == updates
            if args.seed_replay:
                assert metrics['train/ready']
        measured = iteration >= args.warmup_iterations
        if measured:
            durations.append(elapsed)
        emit(stage='executed', iteration=iteration, seconds=elapsed,
             device_call_seconds=elapsed, measured=measured, metrics=metrics, memory=memory())
    emit(stage='passed', median_seconds=statistics.median(durations),
         min_seconds=min(durations), max_seconds=max(durations), optimizer_updates=updates,
         measured_iterations=len(durations), compiled_executions=args.iterations+args.warmup_iterations)


if __name__ == '__main__':
    main()
