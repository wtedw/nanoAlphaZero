"""Full production cycles with one self-play, one drain batch, and one update.

Called by profile_muzero.py. Preparation uses a legal scripted mating game and
one-hot policy targets, not learned game data. Only the first start in each lane
is fresh so both algorithms drain exactly one batch. Copies restoring this
fixture are synchronized outside timing. The real model/search and optimizer
execute inside each measured production run_fn; parameters keep updating.
"""

from contextlib import redirect_stdout
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

from analyze_profile_trace import main as analyze_trace
from probe_muzero_replay_layout import check_tpu_ownership


def run_cycle_profile(args):
    if not 1 <= args.unroll <= 5:
        raise ValueError("The seven-ply fixture supports unroll lengths 1 through 5")
    check_tpu_ownership()
    os.environ["JAX_PLATFORMS"] = "tpu"
    import chess
    import jax
    import jax.numpy as jnp
    import numpy as np
    from nanoalphazero.config import get_chess_config, get_muzero_config
    from nanoalphazero.core import make_alphazero, make_selfplay, DATA_PARALLEL_SHARDING
    from nanoalphazero.eval.chess.action import PgxAction
    from nanoalphazero.mcts import PolicyOutput, compress_muzero_policy

    if args.batch % jax.device_count():
        raise ValueError("Batch must divide evenly over devices")
    cfg = get_muzero_config("chess") if args.algorithm == "muzero" else get_chess_config()
    cfg.update(algorithm=args.algorithm, cycle_n_selfplay=1, cycle_n_train=1,
               selfplay_batch_size=args.batch, train_batch_size=args.batch,
               selfplay_buffer_consume_size=args.batch,
               selfplay_buffer_add_batch_size=args.batch,
               selfplay_buffer_sample_batch_size=args.batch,
               replay_buffer_add_batch_size=args.batch,
               replay_buffer_sample_batch_size=args.batch,
               replay_buffer_max_len=16, replay_buffer_min_len=1,
               replay_buffer_total_size=args.batch * 16,
               muzero_unroll_steps=args.unroll, enable_wandb=False)
    for arg, name in ((args.remat_blocks, "muzero_remat_blocks"),
                      (args.remat_unroll, "muzero_remat_unroll")):
        if arg is not None:
            if args.algorithm != "muzero":
                raise ValueError("Checkpoint overrides apply only to MuZero")
            cfg[name] = arg
    for arg, name in ((args.root_considered, "mcts_num_root_considered"),
                      (args.survivors, "mcts_num_survivors")):
        if arg is not None:
            cfg[name] = arg
    if not 1 <= cfg["mcts_num_survivors"] <= cfg["mcts_num_root_considered"] <= 4672:
        raise ValueError("Invalid search rungs")
    args.output_parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=f"{args.algorithm}-cycle-", dir=args.output_parent))
    print(f"Output: {output}", flush=True)
    fixture_factory = make_selfplay
    if args.algorithm == "muzero":
        spec = importlib.util.spec_from_file_location(
            "muzero_cycle_profile", Path(__file__).resolve().parents[1] / "muzero.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        algorithm = module.make_muzero(cfg, jax.random.PRNGKey(args.seed))
        fixture_factory = module.make_muzero_selfplay
    else:
        algorithm = make_alphazero(cfg, jax.random.PRNGKey(args.seed))
    cfg = algorithm.config
    (output / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    state = algorithm.runner_state
    # Scholar's mate: seven legal transitions, with a full five-step window.
    board = chess.Board()
    actions = []
    for uci in ("e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7"):
        move = chess.Move.from_uci(uci)
        assert move in board.legal_moves
        actions.append(PgxAction.encode(move, board))
        board.push(move)
    assert board.is_checkmate()
    actions = jnp.asarray(actions)

    def scripted_search(key, env_state, params, scale, batch):
        action = actions[env_state._step_count]
        weights = jax.nn.one_hot(action, cfg["game_num_actions"])
        result = PolicyOutput(action=action, action_weights=weights, visit_counts=weights)
        if cfg.get("exp_bnk_action_weights", False):
            result, _ = compress_muzero_policy(result, cfg["mcts_num_k_actions"])
            result = result.replace(bnk_k_indices=result.bnk_k_indices.astype(
                state.selfplay_buffer_state.experience.k_indices.dtype))
        return result

    fixture_fn, games = fixture_factory(cfg, algorithm.env, scripted_search, DATA_PARALLEL_SHARDING)
    games = games._replace(switch_step=jnp.zeros_like(games.switch_step))

    @jax.jit
    def prepare(games, buffer):
        def body(carry, index):
            games, buffer = carry
            games, samples, _ = fixture_fn(jax.random.PRNGKey(index), games, state.model_ts.params)
            buffer, _ = algorithm.selfplay_buffer.add_backfill(
                buffer, samples, games.env_state.terminated, games.env_state.rewards)
            return (games, buffer), None
        (games, buffer), _ = jax.lax.scan(body, (games, buffer), jnp.arange(7))
        exp = buffer.experience
        fresh = exp.is_valid_sample & (exp.global_step_id == 0)
        buffer = buffer.replace(experience=exp.replace(
            is_valid_sample=fresh, is_fresh_i8=fresh.astype(jnp.int8)),
            num_valid_consumable=jnp.sum(fresh))
        return games, buffer

    print("Preparing completed-game fixture", flush=True)
    finished, fixture_buffer = jax.block_until_ready(prepare(games, state.selfplay_buffer_state))
    assert bool(jnp.all(finished.env_state.terminated))
    assert int(fixture_buffer.num_valid_consumable) == args.batch
    if args.check_equivalence:
        if args.algorithm != "muzero":
            raise ValueError("Equivalence checks apply only to MuZero")
        from check_muzero_training import check_training_equivalence
        samples = jax.jit(lambda b: algorithm.selfplay_buffer.consume(b)[1])(fixture_buffer)
        samples = jax.tree.map(lambda x: x[:, 0], samples)
        check_training_equivalence(module, cfg, algorithm.model, state.model_ts, samples, output)
    # Start the timed self-play call on a live board; preserve unique step IDs.
    fixture_games = state.selfplay_state._replace(
        step_count=finished.step_count, next_game_id=finished.next_game_id)

    def restore_fixture(state):
        return state._replace(
            selfplay_state=jax.tree.map(lambda x: x.copy(), fixture_games),
            selfplay_buffer_state=jax.tree.map(lambda x: x.copy(), fixture_buffer))

    state = jax.block_until_ready(restore_fixture(state))
    flag = jnp.bool_(False)
    print("Compiling production cycle: selfplay=1, drain batch=1, train=1", flush=True)
    started = time.perf_counter()
    executable = algorithm.run_fn.lower(state, flag).compile()
    compile_seconds = time.perf_counter() - started
    print(f"Compiled in {compile_seconds:.2f}s", flush=True)
    # Persist every leaf's layout, not merely the mesh or requested sharding.
    def sharding_report(tree):
        return [{"path": jax.tree_util.keystr(path), "shape": list(x.shape),
                 "dtype": str(x.dtype), "sharding": str(x.sharding),
                 "fully_replicated": x.is_fully_replicated,
                 "shards": [{"device": str(s.device), "index": str(s.index),
                             "shape": list(s.data.shape)} for s in x.addressable_shards]}
                for path, x in jax.tree_util.tree_leaves_with_path(tree)]
    (output / "sharding-input.json").write_text(json.dumps(sharding_report(state), indent=2))
    # Keep optimized HLO for inspecting intermediate sample and compute layouts.
    (output / "optimized-hlo.txt").write_text(executable.as_text())
    durations, records = [], []
    updates = 0

    def call(measured):
        nonlocal state, updates
        started = time.perf_counter()
        state, result = executable(state, flag)
        jax.block_until_ready((state, result))
        elapsed = time.perf_counter() - started
        metrics = {k: float(np.asarray(v).reshape(-1)[-1])
                   for k, v in jax.device_get(result[0]).items()}
        updates += 1
        assert metrics["runner_state/n_updates"] == updates, metrics
        assert metrics["drain/n_slices"] == 1, metrics
        assert math.isfinite(metrics["total_loss"]), metrics
        assert metrics.get("selfplay_buffer/error_count", 0) == 0, metrics
        assert metrics.get("train_batch/n_is_invalid", 0) == 0, metrics
        if measured:
            durations.append(elapsed)
        records.append({"measured": measured, "seconds": elapsed, "metrics": metrics})
        return elapsed

    for iteration in range(args.warmup + args.iterations):
        state = jax.block_until_ready(restore_fixture(state))
        elapsed = call(iteration >= args.warmup)
        print(f"Cycle {iteration + 1}: {elapsed * 1000:.2f} ms, drain=1, update={updates}", flush=True)
    (output / "sharding-output.json").write_text(json.dumps(sharding_report(state), indent=2))
    report = dict(algorithm=args.algorithm, mode="cycle", batch=args.batch,
                  revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  remat_blocks=cfg.get("muzero_remat_blocks"),
                  remat_unroll=cfg.get("muzero_remat_unroll"),
                  fixture="scripted Scholar's mate; first start per lane; restored outside timing",
                  compile_seconds=compile_seconds, seconds=durations,
                  median_seconds=statistics.median(durations),
                  positions_per_second=args.batch / statistics.median(durations),
                  compiled_memory=str(executable.memory_analysis()), records=records,
                  devices=[{"name": str(d), "kind": d.device_kind} for d in jax.devices()])
    (output / "timings.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2), flush=True)
    if not args.only_time:
        state = jax.block_until_ready(restore_fixture(state))
        with jax.profiler.trace(str(output)):
            with jax.profiler.TraceAnnotation(f"{args.algorithm}_cycle"):
                call(False)
        print(f"Trace saved to {output}\nView: just xp {output}", flush=True)
        if not args.no_analyze:
            original_argv, summary = sys.argv, io.StringIO()
            try:
                sys.argv = ["analyze_profile_trace.py", str(output), "--mode", "cycle",
                            "--json", str(output / "analysis.json")]
                with redirect_stdout(summary):
                    analyze_trace()
            finally:
                sys.argv = original_argv
            (output / "analysis.txt").write_text(summary.getvalue())
            print(summary.getvalue(), flush=True)
