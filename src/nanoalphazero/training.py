"""Host-side AlphaZero training runtime, monitoring, and diagnostics."""

import atexit
import os
from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import pgx1

from nanoalphazero.eval.ladder import make_ladder
from nanoalphazero.eval.acpl import make_acpl
# Preserve existing helper import paths for callers outside the training host.
from nanoalphazero.eval.matches import (
    _legal_mask_from_state, _random_legal_actions,
    all_opening_actions, run_eval_match, evaluate_vs,
)
from nanoalphazero.checkpoint import (
    _CHECKPOINT_FORMAT,
    _CHECKPOINT_FORMAT_VERSION,
    checkpoint_model_config,
    save_checkpoint,
)
from nanoalphazero.core import make_alphazero, DATA_PARALLEL_SHARDING


# =============================================================================
# Training loop
# =============================================================================
def _ascii_loss_chart(series_map, height=12, max_width=70):
    """Render one or more value series as a compact, colored terminal line chart.

    `series_map` is a dict {name: list-of-values}; all series share one y-axis.
    A single list is also accepted (rendered as one unnamed series).
    """
    RESET = "\033[0m"
    COLORS = {"total": "", "value": "\033[36m", "policy": "\033[33m"}  # cyan / yellow
    if not isinstance(series_map, dict):
        series_map = {"loss": series_map}

    def downsample(s):
        s = [float(v) for v in s if v == v]  # drop NaNs
        if len(s) <= max_width:
            return s
        bucket = len(s) / max_width  # bucket-average so long runs stay readable
        out = []
        for i in range(max_width):
            chunk = s[int(i * bucket) : int((i + 1) * bucket)] or [s[int(i * bucket)]]
            out.append(sum(chunk) / len(chunk))
        return out

    cleaned = {n: downsample(s) for n, s in series_map.items()}
    cleaned = {n: s for n, s in cleaned.items() if len(s) >= 2}
    if not cleaned:
        return ""
    allvals = [v for s in cleaned.values() for v in s]
    minimum, maximum = min(allvals), max(allvals)
    interval = (maximum - minimum) or 1.0
    ratio = height / interval
    min2 = int(round(minimum * ratio))
    max2 = int(round(maximum * ratio))
    rows = max(1, max2 - min2)
    offset = 10
    width = max(len(s) for s in cleaned.values()) + offset
    grid = [[" "] * width for _ in range(rows + 1)]
    cgrid = [[""] * width for _ in range(rows + 1)]
    # y-axis labels + axis ticks
    for y in range(min2, max2 + 1):
        label = f"{maximum - (y - min2) * interval / rows:8.4f}"
        for i, ch in enumerate(label):
            grid[y - min2][i] = ch
        grid[y - min2][offset - 1] = "┤"  # ┤
    # plot each series with box-drawing connectors
    for name, s in cleaned.items():
        col = COLORS.get(name, "")

        def put(r, c, ch, col=col):
            grid[r][c] = ch
            cgrid[r][c] = col

        for x in range(len(s) - 1):
            y0 = int(round(s[x] * ratio) - min2)
            y1 = int(round(s[x + 1] * ratio) - min2)
            if y0 == y1:
                put(rows - y0, x + offset, "─")  # ─
            else:
                put(rows - y1, x + offset, "╰" if y0 > y1 else "╭")  # ╰ ╭
                put(rows - y0, x + offset, "╮" if y0 > y1 else "╯")  # ╮ ╯
                for y in range(min(y0, y1) + 1, max(y0, y1)):
                    put(rows - y, x + offset, "│")  # │
    lines = []
    for r in range(rows + 1):
        cells = []
        for c in range(width):
            ch, col = grid[r][c], cgrid[r][c]
            cells.append(col + ch + RESET if (col and ch != " ") else ch)
        lines.append("".join(cells).rstrip())
    legend = "  ".join(f"{COLORS.get(n, '')}● {n}{RESET}" for n in cleaned)
    return "\n".join(lines) + "\n  " + legend


def _fixed_point_mismatches(in_state, out_info):
    """Compare run_fn's train-path OUTPUT against its INPUT runner_state, leaf by
    leaf. Any difference in dtype, shape, or sharding means runner_state is not a
    fixed point: feeding the output back next cycle retraces -> a second executable
    -> doubled HBM scratch -> OOM.

    in_state: the concrete runner_state (leaves expose .dtype/.shape/.sharding).
    out_info: ShapeDtypeStruct pytree from Compiled.out_info (same attrs).
    Returns a list of (path_str, field, in_value, out_value) for each mismatch.
    """
    in_leaves = jax.tree_util.tree_leaves_with_path(in_state)
    out_leaves = jax.tree_util.tree_leaves_with_path(out_info)
    mismatches = []
    for (path, x), (_, y) in zip(in_leaves, out_leaves):
        where = jax.tree_util.keystr(path)
        if x.dtype != y.dtype:
            mismatches.append((where, "dtype", x.dtype, y.dtype))
        if x.shape != y.shape:
            mismatches.append((where, "shape", x.shape, y.shape))
        if x.sharding != y.sharding:
            mismatches.append((where, "sharding", x.sharding, y.sharding))
    return mismatches


def _probe_executables(run_fn, runner_state):
    """Pre-flight, zero extra HBM: AOT-compile run_fn for BOTH is_warmup branches so
    a compile / scratch-reservation OOM surfaces now rather than after a long
    warmup, then verify the train (is_warmup=False) branch's OUTPUT matches its
    INPUT in dtype/shape/sharding -- i.e. runner_state is a fixed point, so feeding
    it back after warmup won't retrace into a second executable -> doubled HBM ->
    OOM.

    .lower()/.compile() traces and compiles but never *executes*, so run_fn's arg-0
    donation is not triggered: runner_state stays live for warmup and nothing is
    copied (the old throwaway-copy probe transiently doubled HBM and OOM'd chess).
    is_warmup is traced into one shared executable, but the train output's concrete
    sharding/dtype is what historically diverged, so we inspect that branch.
    """
    out_info = None
    for warmup in (False, True):
        # .compile() forces XLA compilation + program load, so a compile-time /
        # scratch-reservation OOM (the "Error loading program ... reserve NN.NG")
        # surfaces here, before the long warmup.
        compiled = run_fn.lower(runner_state, jnp.array(warmup)).compile()
        if warmup is False:
            out_info = compiled.out_info[0]  # (runner_state, metrics) -> [0]

    mismatches = _fixed_point_mismatches(runner_state, out_info)
    if not mismatches:
        print(
            "[probe] OK: train-path output dtype/shape/sharding == input; no "
            "post-warmup recompile/OOM.",
            flush=True,
        )
    else:
        print(
            f"[probe] WARN: {len(mismatches)} leaf field(s) change on the train "
            f"path -> feeding output back will recompile -> likely OOM after "
            f"warmup. First: {mismatches[0]}",
            flush=True,
        )
    # runner_state was never donated/executed -- caller uses it as-is.

class _Tee:
    """Duplicate writes to several streams at once (terminal + log file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def _start_run_logfile(config):
    """Mirror all stdout to logs/[env]_[timestamp].txt for the rest of the run."""
    os.makedirs("logs", exist_ok=True)
    env_name = config.get("game_name", config["env_id"])
    log_path = os.path.join(
        "logs", f"{env_name}_{time.strftime('%Y%m%d-%H%M%S')}.txt"
    )
    log_file = open(log_path, "w", buffering=1)  # line-buffered: flush each line
    prev_stdout = sys.stdout
    sys.stdout = _Tee(prev_stdout, log_file)

    def _close():  # restore + close on normal exit, Ctrl+C, or crash
        sys.stdout = prev_stdout
        log_file.close()

    atexit.register(_close)
    return log_path


def _init_wandb(config: dict):
    """Initialize optional host-0 experiment tracking."""
    if not config.get("enable_wandb", False) or jax.process_index() != 0:
        return None
    import wandb

    kwargs = {
        "project": config.get("wandb_project", "nanoAlphaZero"),
        "config": config,
    }
    for config_key, wandb_key in (
        ("wandb_entity", "entity"),
        ("wandb_name", "name"),
        ("wandb_group", "group"),
        ("wandb_tags", "tags"),
    ):
        if config.get(config_key) is not None:
            kwargs[wandb_key] = config[config_key]
    run = wandb.init(**kwargs)
    print(f"✅ W&B run: {run.url}", flush=True)
    return run


def _upload_wandb_checkpoint(
    wandb_run,
    path: str,
    config: dict,
    *,
    cycle: int,
    final: bool = False,
) -> None:
    """Upload one locally saved checkpoint as a versioned W&B artifact."""
    if wandb_run is None:
        return
    import wandb

    game_name = config.get("game_name", config["env_id"])
    artifact_name = config.get(
        "wandb_artifact_name", f"nanoalphazero-{game_name}"
    )
    artifact = wandb.Artifact(
        artifact_name,
        type="model",
        metadata={
            "cycle": int(cycle),
            "format": _CHECKPOINT_FORMAT,
            "format_version": _CHECKPOINT_FORMAT_VERSION,
            "model_config": checkpoint_model_config(config),
        },
    )
    artifact.add_file(path, name=os.path.basename(path))
    aliases = ["latest", f"cycle-{int(cycle)}"]
    if final:
        aliases.append("final")
    wandb_run.log_artifact(artifact, aliases=aliases)
    print(
        f"☁️  Uploaded checkpoint artifact {artifact_name}:{aliases[-1]}",
        flush=True,
    )


def run_alphazero(config, ckpt_path=None, hex_eval_engine_pool=None):
    log_path = _start_run_logfile(config)
    print(f"Logging this run to {log_path}")

    rng = jax.random.PRNGKey(42)

    rng, az_rng = jax.random.split(rng)
    az = make_alphazero(config, az_rng)

    run_fn = az.run_fn
    runner_state = az.runner_state
    wenv = az.env
    resolved_config = az.config
    wandb_run = _init_wandb(resolved_config)

    print("Successfully initialized all components.")

    # Optional pre-flight: AOT-compile both is_warmup branches now (no execution,
    # no copy) so a recompile-induced OOM surfaces before the long warmup, not after.
    if config.get("debug_probe_executables", False):
        _probe_executables(run_fn, runner_state)

    # --- Warmup Phase: run cycles with is_warmup=True (model frozen) to prime the
    # buffers before real training starts. ---
    num_warmup_cycles = (
        config.get("replay_buffer_warmup_steps", 100) // config["cycle_n_selfplay"]
    )
    print(f"Starting buffer warmup for {num_warmup_cycles} cycles...")
    start_time = time.time()

    for cycle_i in range(num_warmup_cycles):
        call_start = time.time()
        runner_state, _ = run_fn(runner_state, jnp.array(True))
        runner_state.model_ts.step.block_until_ready()
        print(
            f"  Warmup {cycle_i}/{num_warmup_cycles} | {time.time() - call_start:.2f}s"
        )

    warmup_duration = time.time() - start_time
    print(f"Warmup finished in {warmup_duration:.1f}s.")

    num_params = sum(x.size for x in jax.tree.leaves(runner_state.model_ts.params))
    print(f"Model has {num_params:,} parameters.")
    if wandb_run is not None:
        wandb_run.summary["model/num_params"] = int(num_params)
        wandb_run.summary["timing/warmup_seconds"] = float(warmup_duration)

    # --- Main Training Cycle ---
    total_steps = runner_state.model_ts.step
    n_cycles = (config["num_iters"] - total_steps) // config["cycle_n_selfplay"]
    print(f"Starting training for {n_cycles} cycles...")

    cycle_total_duration = 0
    start_time = time.time()

    loss_history = {
        "total": [],
        "value": [],
        "policy": [],
    }  # per-cycle, for ASCII chart
    ep_step_std_history = []  # per-cycle selfplay/ep_step_std, for ASCII chart
    norm_history = {"grad": [], "param": []}  # per-cycle grad/param norms, ASCII chart
    chart_period = config.get("loss_chart_period", 50)

    ladder_fn, ladder_state = make_ladder(
        config, env=wenv, policy_fn=az.run_mcts_fn, chart_fn=_ascii_loss_chart,
    )
    eval_acpl_fn, acpl_state = make_acpl(
        resolved_config, env=wenv, policy_fn=az.run_mcts_fn,
        data_sharding=DATA_PARALLEL_SHARDING if config.get("enable_sharding", False) else None,
    )
    ckpt_period = config.get("ckpt_period")
    hex_eval_period = int(resolved_config.get("hex_eval_period", 0))
    if hex_eval_period and hex_eval_engine_pool is None:
        raise ValueError("Hex engine evaluation is enabled without a MoHex process bank")
    hex_evaluator = None
    if hex_eval_period:
        from nanoalphazero.eval.hex.training import HexTrainingEvaluator

        hex_evaluator = HexTrainingEvaluator(
            hex_eval_period,
            hex_eval_engine_pool,
            Path(log_path).with_suffix(".hex-eval"),
            board_size=int(resolved_config["boardsize"]),
        )

    # Graceful Ctrl+C: the first press finishes the in-flight cycle, then breaks
    # out of training so we still save params + play against the model. A second
    # press restores the default handler and aborts hard.
    import signal

    interrupt = {"flag": False}

    def _handle_sigint(signum, frame):
        if interrupt["flag"]:
            signal.signal(signal.SIGINT, orig_sigint)
            raise KeyboardInterrupt
        interrupt["flag"] = True
        print(
            "\n⚠️  Ctrl+C — will stop after this cycle, then save + play. "
            "Press again to abort.",
            flush=True,
        )

    orig_sigint = signal.signal(signal.SIGINT, _handle_sigint)
    completed_cycle = 0

    for cycle_n in range(1, int(n_cycles) + 1):
        cycle_start_time = time.time()

        runner_state, (all_scalar_metrics, _) = run_fn(runner_state, jnp.array(False))
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), runner_state)

        cycle_duration = time.time() - cycle_start_time
        cycle_total_duration += cycle_duration

        # Log last train step's metrics, grouped by phase (selfplay / drain / train)
        last = jax.tree_util.tree_map(lambda x: float(x[-1]), all_scalar_metrics)
        online_metrics = {
            **last,
            "cycle": cycle_n,
            "timing/cycle_seconds": cycle_duration,
            "runner_state/train_step": int(runner_state.model_ts.step),
        }
        print(
            f"[{config.get('game_name', config['env_id'])}] "
            f"Cycle {cycle_n}/{int(n_cycles)} | {cycle_duration:.2f}s\n"
            f"  phase1 selfplay | "
            f"p1_wins={last.get('selfplay/p1_wins', 0):.0f} "
            f"p2_wins={last.get('selfplay/p2_wins', 0):.0f} "
            f"ties={last.get('selfplay/p_just_tied', 0):.0f} "
            f"n_legal_avg_mid={last.get('selfplay/n_legal_moves_avg_mid', 0):.2f} "
            f"ep_step_std={last.get('selfplay/ep_step_std', 0):.2f}\n"
            f"  phase2 drain    | "
            f"consumable={last.get('drain/num_valid_consumable', 0):.0f} "
            f"slices={last.get('drain/n_slices', 0):.0f}\n"
            f"  phase3 train    | "
            f"loss={last.get('total_loss', 0):.4f} "
            f"loss_v={last.get('loss_v', 0):.4f} "
            f"loss_pi={last.get('loss_pi', 0):.4f} "
            f"grad_norm={last.get('norms/grad_norm', 0):.3f} "
            f"param_norm={last.get('norms/param_norm', 0):.3f} "
            f"batch[r+={last.get('train_batch/n_reward_pos', 0):.0f} "
            f"r-={last.get('train_batch/n_reward_neg', 0):.0f} "
            f"r0={last.get('train_batch/n_reward_zero', 0):.0f} "
            f"valid={last.get('train_batch/n_is_valid', 0):.0f} "
            f"invalid={last.get('train_batch/n_is_invalid', 0):.0f}]",
            flush=True,
        )

        loss_history["total"].append(last.get("total_loss", float("nan")))
        loss_history["value"].append(last.get("loss_v", float("nan")))
        loss_history["policy"].append(last.get("loss_pi", float("nan")))
        if (
            len(loss_history["total"]) > 4000
        ):  # cap memory; chart bucket-averages anyway
            loss_history = {k: v[-4000:] for k, v in loss_history.items()}
        ep_step_std_history.append(last.get("selfplay/ep_step_std", float("nan")))
        if len(ep_step_std_history) > 4000:
            ep_step_std_history = ep_step_std_history[-4000:]
        norm_history["grad"].append(last.get("norms/grad_norm", float("nan")))
        norm_history["param"].append(last.get("norms/param_norm", float("nan")))
        if len(norm_history["grad"]) > 4000:
            norm_history = {k: v[-4000:] for k, v in norm_history.items()}
        if (
            chart_period
            and cycle_n % chart_period == 0
            and len(loss_history["total"]) >= 2
        ):
            n_pts = len(loss_history["total"])
            print(f"\n── loss over last {n_pts} cycles ──", flush=True)
            print(
                _ascii_loss_chart(
                    loss_history,
                    height=config.get("chart_height", 24),
                    max_width=config.get("chart_width", 160),
                ),
                flush=True,
            )
            print(flush=True)
            print(
                f"── selfplay/ep_step_std over last {n_pts} cycles ──",
                flush=True,
            )
            print(
                _ascii_loss_chart({"ep_step_std": ep_step_std_history}),
                flush=True,
            )
            print(flush=True)
            # grad/param norms on separate small (default-size) charts: they live
            # on very different scales, so a shared y-axis would flatten grad_norm.
            print(f"── grad_norm over last {n_pts} cycles ──", flush=True)
            print(_ascii_loss_chart({"grad_norm": norm_history["grad"]}), flush=True)
            print(flush=True)
            print(f"── param_norm over last {n_pts} cycles ──", flush=True)
            print(_ascii_loss_chart({"param_norm": norm_history["param"]}), flush=True)
            print(flush=True)

        is_diagnostic_time = (cycle_n == 1) or (
            cycle_n % config.get("diagnostic_period", 100) == 0
        )
        if is_diagnostic_time and config["env_id"] == "tic_tac_toe":
            _run_ttt_diagnostics(runner_state.model_ts, wenv, config)
        # hex value-head-vs-perfect-play ASCII diagnostic
        elif is_diagnostic_time and config["env_id"].startswith("hex"):
            _run_hex_diagnostics(runner_state.model_ts, wenv, config)
        # connect4 opening-column value-vs-perfect-play diagnostic
        elif is_diagnostic_time and config["env_id"] == "connect_four":
            _run_connect4_diagnostics(runner_state.model_ts, wenv, config)
        # go policy-logit + value-head ASCII diagnostic
        elif is_diagnostic_time and config["env_id"].startswith("go"):
            _run_go_diagnostics(runner_state.model_ts, wenv, config)

        ladder_state, ladder_metrics = ladder_fn(
            ladder_state, cycle_n, runner_state.model_ts.params,
        )
        online_metrics.update(ladder_metrics)
        acpl_state, acpl_result = eval_acpl_fn(acpl_state, cycle_n, runner_state.model_ts.params)
        online_metrics.update(acpl_result.metrics)

        if hex_evaluator is not None:
            online_metrics.update(hex_evaluator.run_if_due(
                cycle_n,
                az.run_mcts_fn,
                wenv,
                resolved_config,
                runner_state.model_ts.params,
                train_step=int(runner_state.model_ts.step),
            ))

        if ckpt_period and ckpt_path and cycle_n % ckpt_period == 0:
            print(f"\n--- Checkpoint at cycle {cycle_n} ---", flush=True)
            save_checkpoint(runner_state.model_ts.params, resolved_config, ckpt_path)
            _upload_wandb_checkpoint(
                wandb_run,
                ckpt_path,
                resolved_config,
                cycle=cycle_n,
            )
            online_metrics["checkpoint/saved"] = 1

        if wandb_run is not None:
            wandb_run.log(online_metrics, step=cycle_n)
        completed_cycle = cycle_n

        if interrupt["flag"]:
            print(f"Stopping early at cycle {cycle_n} (Ctrl+C).", flush=True)
            break

    signal.signal(signal.SIGINT, orig_sigint)  # restore default Ctrl+C handling
    total_duration = time.time() - start_time
    print(f"Training finished in {total_duration:.1f}s.")
    if ckpt_path:
        save_checkpoint(runner_state.model_ts.params, resolved_config, ckpt_path)
        _upload_wandb_checkpoint(
            wandb_run,
            ckpt_path,
            resolved_config,
            cycle=completed_cycle,
            final=True,
        )
    if wandb_run is not None:
        wandb_run.summary["timing/training_seconds"] = float(total_duration)
        wandb_run.summary["cycle/final"] = int(completed_cycle)
        wandb_run.finish()
    return runner_state


# =============================================================================
# Compression utils (chess)
# =============================================================================
# Chess observation is (8,8,119); we compress the bool channels into packed uint8
# and store the legal-action mask as a uint32 bitset. This drastically reduces
# replay-buffer memory vs. storing the raw (8,8,119) float obs + (4672,) bool mask.

# =============================================================================
# Game-specific diagnostics & perfect-play tables
# =============================================================================
def _run_ttt_diagnostics(model_ts, wenv, config):
    boardsize = config.get("boardsize", 3)
    env = pgx1.make("tic_tac_toe")
    dummy_state = env.init(jax.random.PRNGKey(0))
    obs = env.observe(dummy_state, dummy_state.current_player)
    obs_b = obs[jnp.newaxis, ...]
    legal_b = dummy_state.legal_action_mask[jnp.newaxis, ...]

    logits, values = model_ts.apply_fn(
        {"params": model_ts.params}, obs_b, legal_b, deterministic=True
    )
    logits = logits.flatten()
    value = float(values.flatten()[0])

    print(f"  Diagnostics: P1 argmax={int(jnp.argmax(logits))}, value={value:.3f}")
    logits_2d = np.array(logits).reshape((boardsize, boardsize))

    print(f"  Logits (empty board), value={value:.3f}:")
    for r in range(boardsize):
        print("    " + "  ".join(f"{logits_2d[r, c]:+.2f}" for c in range(boardsize)))


# hex perfect-play ground truth
# Value matrix from the perspective of the player to move (White) *after* Black's
# opening. A Black-winning opening leaves White in a lost position -> value -1.0;
# a Black-losing opening leaves White winning -> value +1.0.
def _get_hex_perfect_play_values(boardsize: int):
    from nanoalphazero.eval.hex.perfect_play import perfect_play_values

    return perfect_play_values(boardsize)


def _run_hex_diagnostics(model_ts, wenv, config):
    boardsize = config["boardsize"]
    batch_size = boardsize * boardsize

    # One blank board per opening; each plays a distinct first move.
    env_state = wenv.init_dummy_estate(batch_size=batch_size)
    all_moves = jnp.arange(batch_size)
    env_state = wenv.step(env_state, all_moves)
    obs = wenv.observe(env_state, env_state.current_player)

    _logits, values = model_ts.apply_fn(
        {"params": model_ts.params},
        obs,
        env_state.legal_action_mask,
        deterministic=True,
    )
    values_2d = np.array(values.flatten()).reshape((boardsize, boardsize))

    print(
        "\n--- Hex value-head after each Black opening "
        "(value = White-to-move perspective; negative => Black-winning) ---"
    )
    for r in range(boardsize):
        print("  " + " ".join(f"{values_2d[r, c]:+.2f}" for c in range(boardsize)))
    gt = _get_hex_perfect_play_values(boardsize)
    if gt is None:
        print(f"  (no ground-truth perfect-play table for {boardsize}x{boardsize})")
        return

    # B = Black-winning opening (gt -1), . = Black-losing opening (gt +1)
    print("  Perfect play (B=Black wins / .=Black loses), [x]=model sign mismatch:")
    pred_sign = np.sign(values_2d)
    for r in range(boardsize):
        cells = []
        for c in range(boardsize):
            truth = "B" if gt[r, c] < 0 else "."
            mismatch = pred_sign[r, c] != np.sign(gt[r, c])
            cells.append(f"[{truth}]" if mismatch else f" {truth} ")
        print("  " + "".join(cells))

    mse = float(np.mean((values_2d - gt) ** 2))
    sign_acc = float(np.mean(pred_sign == np.sign(gt)))
    print(f"  MSE vs perfect = {mse:.4f} | sign accuracy = {sign_acc:.3f}")


# connect4 perfect-play opening values + ASCII diagnostic
# Connect 4 is solved: with perfect play, P1 wins iff they open in the center
# column (col 4 / 0-indexed 3); the two adjacent columns (3 & 5 / idx 2 & 4) draw;
# the four edge columns (1,2,6,7 / idx 0,1,5,6) are losses for P1.
# Values are from the perspective of the player to move *after* the opening (P2):
#   P1-win opening   -> P2 is lost   -> -1.0
#   draw opening     ->  0.0
#   P1-loss opening  -> P2 wins      -> +1.0
def _get_connect4_perfect_play_values():
    # idx:            0    1    2    3    4    5    6
    vals = np.array([+1.0, +1.0, 0.0, -1.0, 0.0, +1.0, +1.0], dtype=np.float32)
    labels = ["L", "L", "D", "W", "D", "L", "L"]  # outcome for P1
    return vals, labels


def _run_connect4_diagnostics(model_ts, wenv, config):
    num_cols = 7
    # One blank board per opening column; each plays a distinct first move.
    env_state = wenv.init_dummy_estate(batch_size=num_cols)
    all_moves = jnp.arange(num_cols)
    env_state = wenv.step(env_state, all_moves)
    obs = wenv.observe(env_state, env_state.current_player)

    _logits, values = model_ts.apply_fn(
        {"params": model_ts.params},
        obs,
        env_state.legal_action_mask,
        deterministic=True,
    )
    values = np.array(values.flatten())  # (7,) P2-to-move perspective

    gt, labels = _get_connect4_perfect_play_values()

    print(
        "\n--- Connect4 value-head after each opening column "
        "(value = P2-to-move perspective; negative => P1-winning) ---"
    )
    print("  col:    " + "  ".join(f"{c+1:>5d}" for c in range(num_cols)))
    print("  value:  " + "  ".join(f"{values[c]:+.2f}" for c in range(num_cols)))
    print(
        "  perfect:"
        + "  ".join(f"{labels[c]:>5s}" for c in range(num_cols))
        + "    (W=P1 wins, D=draw, L=P1 loses)"
    )

    # Directional correctness: W -> value<0, L -> value>0, D -> |value| small.
    draw_tol = 0.5
    correct = []
    for c in range(num_cols):
        if labels[c] == "W":
            correct.append(values[c] < 0)
        elif labels[c] == "L":
            correct.append(values[c] > 0)
        else:  # draw
            correct.append(abs(values[c]) < draw_tol)
    print("  match:  " + "  ".join("  ok " if ok else "  x  " for ok in correct))

    mse = float(np.mean((values - gt) ** 2))
    acc = float(np.mean(correct))
    print(f"  MSE vs perfect = {mse:.4f} | directional accuracy = {acc:.3f}")


def _run_go_diagnostics(model_ts, wenv, config):
    """Go diagnostic: policy logits on the empty board and the value head after
    each non-pass opening move, both laid out as boardsize x boardsize grids.

    The pass action (index boardsize**2) is excluded from both grids; its policy
    logit is printed inline next to the empty-board value for reference.
    """
    boardsize = config["boardsize"]
    n = boardsize * boardsize  # number of board points (pass is action index n)

    # --- Empty-board policy logits (non-pass actions) + value ---
    env_state = wenv.init_dummy_estate(batch_size=1)
    obs = wenv.observe(env_state, env_state.current_player)
    logits, values = model_ts.apply_fn(
        {"params": model_ts.params},
        obs,
        env_state.legal_action_mask,
        deterministic=True,
    )
    logits = np.array(logits.flatten())
    pass_logit = float(logits[n])
    value0 = float(values.flatten()[0])
    logits_2d = logits[:n].reshape((boardsize, boardsize))

    print(
        f"\n--- Go policy logits on empty board "
        f"(value={value0:+.3f}, pass logit={pass_logit:+.2f}) ---"
    )
    for r in range(boardsize):
        print("  " + " ".join(f"{logits_2d[r, c]:+.2f}" for c in range(boardsize)))

    # --- Value head after each non-pass opening move ---
    # One blank board per opening point; each plays a distinct first move.
    env_state = wenv.init_dummy_estate(batch_size=n)
    all_moves = jnp.arange(n)  # 0..n-1: every board point (excludes pass=n)
    env_state = wenv.step(env_state, all_moves)
    obs = wenv.observe(env_state, env_state.current_player)
    _logits, values = model_ts.apply_fn(
        {"params": model_ts.params},
        obs,
        env_state.legal_action_mask,
        deterministic=True,
    )
    values_2d = np.array(values.flatten()).reshape((boardsize, boardsize))

    print(
        "\n--- Go value head after each opening move "
        "(value = opponent-to-move perspective; negative => opening side winning) ---"
    )
    for r in range(boardsize):
        print("  " + " ".join(f"{values_2d[r, c]:+.2f}" for c in range(boardsize)))
