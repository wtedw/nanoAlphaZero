"""Profile MuZero or AlphaZero MCTS, self-play, or a reduced full cycle.

Run from the repository root: just prof-analyze chess mcts --batch 256
Timing excludes compilation, warmup, and trace collection. Each invocation gets
a new output directory. Cycle mode includes buffers and training; the isolated
modes do not. No checkpoints or trackers are used.
"""

import argparse
from contextlib import redirect_stdout
import io
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

from analyze_profile_trace import main as analyze_trace
from probe_muzero_replay_layout import check_tpu_ownership


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("game", choices=("chess",))
    parser.add_argument("mode", choices=("mcts", "selfplay", "cycle"))
    parser.add_argument("--algorithm", choices=("muzero", "alphazero"), default="muzero")
    parser.add_argument("--batch", "--selfplay-batch-size", type=int, default=256)
    parser.add_argument("--root-considered", type=int)
    parser.add_argument("--survivors", type=int)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--unroll", type=int, default=5,
                        help="MuZero cycle training unroll (default: 5)")
    parser.add_argument("--remat-blocks", action=argparse.BooleanOptionalAction, default=None,
                        help="Override MuZero block checkpointing in cycle mode")
    parser.add_argument("--remat-unroll", action=argparse.BooleanOptionalAction, default=None,
                        help="Override MuZero unroll checkpointing in cycle mode")
    parser.add_argument("--check-equivalence", action="store_true",
                        help="Compare MuZero losses, gradients and updates to both checkpoint levels enabled")
    parser.add_argument("--only-time", action="store_true")
    parser.add_argument("--no-analyze", action="store_true",
                        help="Save the trace without printing the XLA text summary")
    parser.add_argument("--output-parent", type=Path, default=Path("/tmp/maz-profiles"))
    args = parser.parse_args()
    if (args.remat_blocks is not None or args.remat_unroll is not None or args.check_equivalence):
        if args.mode != "cycle" or args.algorithm != "muzero":
            parser.error("checkpoint overrides and equivalence checks require MuZero cycle mode")
    if min(args.batch, args.iterations, args.warmup) < 1:
        parser.error("batch, iterations, and warmup must be positive")
    if args.root_considered is not None and args.root_considered < 1:
        parser.error("root-considered must be positive")
    if args.survivors is not None and args.survivors < 1:
        parser.error("survivors must be positive")

    if args.mode == "cycle":
        from profile_cycle import run_cycle_profile
        run_cycle_profile(args)
        return

    check_tpu_ownership()
    os.environ["JAX_PLATFORMS"] = "tpu"
    import jax
    from nanoalphazero.config import get_muzero_config, get_chess_config
    from nanoalphazero.core import make_env, make_selfplay, DATA_PARALLEL_SHARDING, REPLICATED_SHARDING
    from nanoalphazero.mcts import make_muzero_mcts, make_mcts
    from nanoalphazero.model import make_muzero_model, make_model

    if args.batch % jax.device_count():
        parser.error("batch must be divisible by device count")
    overrides = dict(selfplay_batch_size=args.batch)
    if args.root_considered is not None:
        overrides["mcts_num_root_considered"] = args.root_considered
    if args.survivors is not None:
        overrides["mcts_num_survivors"] = args.survivors
    if args.algorithm == "muzero":
        config = get_muzero_config(args.game, **overrides)
    else:
        config = {**get_chess_config(), **overrides, "algorithm": "alphazero"}
    if config["mcts_num_survivors"] > config["mcts_num_root_considered"]:
        parser.error("survivors must not exceed root-considered")
    args.output_parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=f"{args.algorithm}-{args.mode}-", dir=args.output_parent))
    print(f"Output: {output}", flush=True)
    env = make_env(config)
    config.update(game_obs_shape=tuple(env.obs_shape), game_num_actions=env.num_actions)
    if config["mcts_num_root_considered"] > env.num_actions:
        parser.error("root-considered must not exceed action count")
    key, model_key, state_key = jax.random.split(jax.random.PRNGKey(args.seed), 3)
    key = jax.device_put(key, REPLICATED_SHARDING)
    model_factory = make_muzero_model if args.algorithm == "muzero" else make_model
    model, variables = model_factory(config, model_key, REPLICATED_SHARDING)
    params = variables["params"]
    search_factory = make_muzero_mcts if args.algorithm == "muzero" else make_mcts
    search = search_factory(config, env, model, DATA_PARALLEL_SHARDING)
    scale = config["mcts_gumbel_scale"] if config["mcts_use_gumbel"] else 0.0
    if args.mode == "selfplay":
        selfplay_factory = make_selfplay
        if args.algorithm == "muzero":
            spec = importlib.util.spec_from_file_location(
                "muzero_profile_runner", Path(__file__).resolve().parents[1] / "muzero.py"
            )
            runner = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(runner)
            selfplay_factory = runner.make_muzero_selfplay
        selfplay_fn, states = selfplay_factory(
            config, env, search, DATA_PARALLEL_SHARDING
        )
        state_sharding = jax.tree.map(
            lambda x: DATA_PARALLEL_SHARDING if x.ndim else REPLICATED_SHARDING, states
        )
        states = jax.device_put(states, state_sharding)

        def advance(key, states, params):
            key, step_key = jax.random.split(key)
            following, samples, metrics = selfplay_fn(step_key, states, params)
            return key, following, samples, metrics

        step = jax.jit(
            advance,
            out_shardings=(REPLICATED_SHARDING, state_sharding,
                           DATA_PARALLEL_SHARDING, REPLICATED_SHARDING),
        )
    else:
        states = jax.device_put(
            env.init(jax.random.split(state_key, args.batch)), DATA_PARALLEL_SHARDING
        )
    jax.block_until_ready((params, states, key))
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Compiling b{config['conv_depth']}c{config['conv_width']}, batch={args.batch}, "
          f"roots={config['mcts_num_root_considered']}, survivors={config['mcts_num_survivors']}", flush=True)
    started = time.perf_counter()
    lowered = (step.lower(key, states, params) if args.mode == "selfplay"
               else search.lower(key, states, params, scale, args.batch))
    executable = lowered.compile()
    compile_seconds = time.perf_counter() - started

    def call():
        nonlocal key, states
        if args.mode == "selfplay":
            result = jax.block_until_ready(executable(key, states, params))
            key, states = result[:2]
            return result
        return jax.block_until_ready(executable(key, states, params, scale))

    print(f"Compiled in {compile_seconds:.2f}s; warming up", flush=True)
    for _ in range(args.warmup):
        call()
    durations = []
    for _ in range(args.iterations):
        started = time.perf_counter()
        call()
        durations.append(time.perf_counter() - started)
    memory = executable.memory_analysis()
    report = dict(
        algorithm=args.algorithm, mode=args.mode,
        positions="advancing games from initial" if args.mode == "selfplay" else "initial",
        parameters="random", seed=args.seed, warmup_calls=args.warmup,
        devices=[str(d) for d in jax.devices()], batch=args.batch,
        compile_seconds=compile_seconds, seconds=durations,
        median_seconds=statistics.median(durations),
        positions_per_second=args.batch / statistics.median(durations),
        compiled_memory=str(memory),
        revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        worktree_status=subprocess.check_output(["git", "status", "--short"], text=True),
    )
    (output / "timings.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if not args.only_time:
        print(f"Collecting one additional {args.mode} trace", flush=True)
        with jax.profiler.trace(str(output)):
            with jax.profiler.TraceAnnotation(f"{args.algorithm}_{args.mode}"):
                call()
        print(f"Trace saved to {output}\nView in TensorBoard: just xp {output}", flush=True)
        if args.no_analyze:
            return
        original_argv = sys.argv
        summary = io.StringIO()
        try:
            sys.argv = ["analyze_profile_trace.py", str(output), "--mode", args.mode]
            with redirect_stdout(summary):
                analyze_trace()
        finally:
            sys.argv = original_argv
        (output / "analysis.txt").write_text(summary.getvalue())
        print(summary.getvalue(), flush=True)


if __name__ == "__main__":
    main()
