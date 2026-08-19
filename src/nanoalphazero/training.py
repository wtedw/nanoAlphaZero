"""Host-side AlphaZero training runtime, monitoring, and diagnostics."""

import atexit
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import pgx1

from nanoalphazero.buffers import unpack_bitmask_vmap
from nanoalphazero.checkpoint import (
    _CHECKPOINT_FORMAT,
    _CHECKPOINT_FORMAT_VERSION,
    checkpoint_model_config,
    save_checkpoint,
)
from nanoalphazero.core import make_alphazero


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


def run_alphazero(config, ckpt_path=None, *, custom_env=None):
    log_path = _start_run_logfile(config)
    print(f"Logging this run to {log_path}")

    rng = jax.random.PRNGKey(42)

    rng, az_rng = jax.random.split(rng)
    az = make_alphazero(config, az_rng, custom_env=custom_env)

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

    # --- Strength eval setup: a GATED ELO LADDER.
    # The anchor is a frozen opponent we measure ΔElo against; the current model's
    # absolute Elo = anchor_elo + ΔElo(current vs anchor). The ladder starts pinned
    # to the random opponent (anchor_params=None, anchor_elo=0). We only PROMOTE the
    # current checkpoint to be the new anchor once it beats the current anchor in the
    # *informative* band of the logistic — score ≥ eval_promote_score, sustained for
    # eval_promote_patience consecutive evals (hysteresis). Promoting in-band (rather
    # than at ~100%, where score saturates and the ΔElo gap is unmeasurable) keeps
    # each rung's height a real measurement instead of a clamp-capped guess, so the
    # ladder Elo stays calibrated to absolute strength rather than inflating.
    eval_period = config.get("eval_period", 100)
    # Promote when current scores this well vs the anchor (top of the informative
    # band: ~0.85 ≈ +300 ΔElo). Higher → fewer, taller rungs but riskier saturation.
    eval_promote_score = config.get("eval_promote_score", 0.75)
    # ...and only after it holds for this many consecutive evals (noise guard).
    eval_promote_patience = config.get("eval_promote_patience", 2)
    # Periodic checkpointing: every ckpt_period cycles, overwrite ckpt_path with
    # the latest params (crash recovery / resumable inference). Disabled if no
    # ckpt_period set or no path given (e.g. --no-save).
    ckpt_period = config.get("ckpt_period")
    eval_openings = all_opening_actions(
        wenv, config, plies=config.get("eval_opening_plies", 1)
    )
    eval_fn = az.run_mcts_fn
    # Ladder state. anchor_params=None ⇒ rung 0 = random opponent, pinned at Elo 0.
    anchor_params = None
    anchor_elo = 0.0
    anchor_cycle = 0
    rung = 0
    qualify_deltas = []  # in-band ΔElo samples accumulating toward a promotion
    best_score_rand = 0.0  # high-water mark vs random, for a forgetting warning
    elo_curve = [0.0]  # current model's absolute Elo over time (anchor_elo + live Δ)
    elo_cycles = [0]

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

        # --- Strength eval: gated Elo ladder (see setup block above).
        # Always measure vs random (forgetting detector + rung-0 yardstick); on
        # higher rungs also measure vs the frozen anchor. Promote in-band.
        if eval_period and cycle_n % eval_period == 0:
            print(
                f"\n--- Strength eval at cycle {cycle_n} "
                f"({len(eval_openings)} openings x2 colors per opponent) ---"
            )
            cur_params = runner_state.model_ts.params
            ek_rand, ek_anchor = jax.random.split(jax.random.PRNGKey(1234 + cycle_n))

            # vs random: permanent baseline. On rung 0 this IS the anchor match.
            score_rand, delta_rand = evaluate_vs(
                eval_fn,
                wenv,
                config,
                cur_params,
                None,
                openings=eval_openings,
                key=ek_rand,
                label="random (Elo 0)",
            )
            # Cap the Elo credit for beating the random opponent. Random is rung
            # 0's anchor, so its ΔElo is the base every higher rung stacks on.
            # Beating random ~100% otherwise pegs ΔElo at the score→Elo clamp
            # ceiling (≈+1600), inflating the whole ladder. With the default cap
            # of 0, beating random just anchors the ladder at Elo 0 (no credit
            # for beating random); all real Elo then comes from beating past
            # selves. min() keeps the negative side, so losing to random still
            # shows negative Elo and trips the forgetting detector.
            rand_elo_cap = config.get("eval_vs_random_max_elo", 0.0)
            if rand_elo_cap is not None:
                delta_rand = min(delta_rand, float(rand_elo_cap))
            if score_rand > best_score_rand:
                best_score_rand = score_rand
            elif rung > 0 and score_rand < best_score_rand - 0.15:
                print(
                    f"  [eval] ⚠️  forgetting? vs-random score {score_rand:.3f} is "
                    f"well below high-water {best_score_rand:.3f}",
                    flush=True,
                )

            # Score/ΔElo against the *current* anchor drives the ladder.
            if anchor_params is None:  # rung 0: the anchor is random
                score_vs_anchor, delta_vs_anchor = score_rand, delta_rand
            else:
                score_vs_anchor, delta_vs_anchor = evaluate_vs(
                    eval_fn,
                    wenv,
                    config,
                    cur_params,
                    anchor_params,
                    openings=eval_openings,
                    key=ek_anchor,
                    label=f"anchor R{rung}@{anchor_cycle}",
                )

            # Absolute Elo = height of the current rung + live gap above it.
            live_elo = anchor_elo + float(delta_vs_anchor)
            online_metrics.update(
                {
                    "eval/score_vs_random": float(score_rand),
                    "eval/elo_vs_random": float(delta_rand),
                    "eval/score_vs_anchor": float(score_vs_anchor),
                    "eval/elo_vs_anchor": float(delta_vs_anchor),
                    "eval/ladder_elo": float(live_elo),
                    "eval/rung": int(rung),
                }
            )
            elo_curve.append(live_elo)
            elo_cycles.append(cycle_n)
            print(
                f"  [eval] ladder Elo: {live_elo:+.0f}  (rung {rung} @ "
                f"{anchor_elo:+.0f}, +{float(delta_vs_anchor):.0f} over anchor)",
                flush=True,
            )

            # Promotion: in-band score, sustained eval_promote_patience evals. We
            # freeze the rung height as the *mean* in-band gap over the streak,
            # which halves its variance vs a single 1-batch measurement.
            if score_vs_anchor >= eval_promote_score:
                qualify_deltas.append(float(delta_vs_anchor))
                if len(qualify_deltas) >= eval_promote_patience:
                    rung_height = float(np.mean(qualify_deltas))
                    anchor_elo += rung_height
                    anchor_params = jax.tree_util.tree_map(jnp.copy, cur_params)
                    anchor_cycle = cycle_n
                    rung += 1
                    qualify_deltas = []
                    print(
                        f"  [eval] ⬆ PROMOTE to rung {rung}: anchor now @ "
                        f"{anchor_elo:+.0f} Elo (rung height +{rung_height:.0f}, "
                        f"avg of {eval_promote_patience} evals)",
                        flush=True,
                    )
            else:
                qualify_deltas = []  # streak broken — reset hysteresis

            if len(elo_curve) >= 3:
                print(
                    f"\n── ladder Elo over cycles {elo_cycles[1]}..{elo_cycles[-1]} ──",
                    flush=True,
                )
                print(_ascii_loss_chart({"total": elo_curve}), flush=True)
                print(flush=True)

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
# Head-to-head evaluation (any game)
# =============================================================================
# Self-play loss curves don't tell you if the model is getting stronger (both
# players co-improve). These helpers give a fixed yardstick that works for every
# pgx game (incl. chess, no external engine): play one game per legal opening
# move (ttt->9, connect4->7, hex4x4->16, chess->20) head-to-head between two
# param sets and report the win/draw/loss split. Two useful opponents:
#   * None         -> uniform-random legal play  (sanity: is it learning at all?)
#   * anchor_params -> a frozen earlier snapshot  (is it STILL improving? -> ΔElo)
# Enumerating every opening covers the position space deterministically (greedy
# MCTS), so the same model gives the same scoreline every time -> low variance.
# Everything is batched on-TPU and reuses run_mcts + the env, so it's cheap.


def _legal_mask_from_state(env_state, config):
    if config["env_id"] == "chess":
        return unpack_bitmask_vmap(env_state.legal_action_bitmask)
    return env_state.legal_action_mask


def _random_legal_actions(env_state, config, key):
    """Uniform random over legal actions (batched)."""
    legal = _legal_mask_from_state(env_state, config)
    logits = jnp.where(legal, 0.0, -jnp.inf)
    g = jax.random.gumbel(key, logits.shape, dtype=logits.dtype)
    return jnp.argmax(logits + g, axis=-1).astype(jnp.int32)


def all_opening_actions(wenv, config, plies=1):
    """Every legal opening LINE of length `plies` from the initial position.

    Returns a 2-D int array of shape (num_openings, plies); row i is one forced
    move sequence. plies=1 -> one row per legal first move (ttt->9, connect4->7,
    hex4x4->16, chess->20). plies=2 -> every legal (move1, move2) pair (chess
    ->~400), which probes a wider, more balanced slice of the opening tree and
    so gives a lower-variance ladder score. Lines that terminate before `plies`
    moves are dropped (can't host a full game), keeping the result rectangular.
    """
    env_state = wenv.init_dummy_estate(batch_size=1)
    legal = np.asarray(_legal_mask_from_state(env_state, config)[0])
    lines = [[int(a)] for a in np.nonzero(legal)[0]]
    for _ in range(int(plies) - 1):
        n = len(lines)
        env_state = wenv.init_dummy_estate(batch_size=n)
        depth = len(lines[0])
        for d in range(depth):  # replay the line built so far, batched
            col = jnp.asarray([line[d] for line in lines], dtype=jnp.int32)
            env_state = wenv.step(env_state, col)
        legal_masks = np.asarray(_legal_mask_from_state(env_state, config))
        terminated = np.asarray(env_state.terminated)
        next_lines = []
        for i, line in enumerate(lines):
            if terminated[i]:
                continue  # game over before reaching `plies` moves -> drop
            for a in np.nonzero(legal_masks[i])[0]:
                next_lines.append(line + [int(a)])
        lines = next_lines
    return np.asarray(lines, dtype=np.int32)


def run_eval_match(
    run_mcts_fn,
    wenv,
    config,
    params_p0,
    params_p1,
    *,
    opening_actions,
    key,
    max_plies=None,
    num_simulations=None,
    gumbel_scale=0.0,
):
    """Play one game per forced opening, fully batched.

    `opening_actions` is an array of forced opening lines: shape (N,) for a single
    forced move each, or (N, K) for a K-move forced line. One game is played per
    row. Seat 0 plays with `params_p0`, seat 1 with `params_p1`. A param set of
    `None` means uniform-random legal play. After the forced opening moves, both
    sides play greedy MCTS. Returns (p0_wins, draws, p1_wins) as ints; games that
    don't finish within `max_plies` count as draws.
    """
    if max_plies is None:
        max_plies = config.get("eval_max_plies") or config.get("game_max_steps") or 512
    if num_simulations is None:
        num_simulations = config["mcts_num_simulations"]

    opening_actions = np.asarray(opening_actions, dtype=np.int32)
    if opening_actions.ndim == 1:  # (N,) -> (N, 1): one forced move per game
        opening_actions = opening_actions[:, None]
    num_real = len(opening_actions)

    # Pad the batch up to a multiple of the device count so data-parallel
    # sharding is happy; padded games duplicate real openings and are dropped.
    mult = jax.device_count() if config.get("enable_sharding", False) else 1
    pad = (-num_real) % mult
    batch_actions = (
        np.concatenate([opening_actions, opening_actions[:pad]])
        if pad
        else opening_actions
    )
    num_games = len(batch_actions)

    env_state = wenv.init_dummy_estate(batch_size=num_games)
    for d in range(batch_actions.shape[1]):  # play each forced opening ply
        env_state = wenv.step(env_state, jnp.asarray(batch_actions[:, d]))

    final_rewards = jnp.zeros((num_games, 2), dtype=jnp.float32)
    finished = env_state.terminated

    for _ply in range(int(max_plies)):
        newly = env_state.terminated & ~finished
        final_rewards = jnp.where(newly[:, None], env_state.rewards, final_rewards)
        finished = finished | env_state.terminated
        if bool(jnp.all(finished)):
            break

        # Live games are in lockstep (player alternates each ply), so one seat
        # moves for the whole batch this ply.
        cp = jnp.where(~finished, env_state.current_player, -1)
        seat = int(jnp.max(cp))
        params = params_p0 if seat == 0 else params_p1

        key, mk = jax.random.split(key)
        if params is None:
            actions = _random_legal_actions(env_state, config, mk)
        else:
            actions = run_mcts_fn(
                mk, env_state, params, gumbel_scale, num_games, num_simulations
            ).action
        env_state = wenv.step(env_state, actions)

    # Credit any game that terminated on the very last step.
    newly = env_state.terminated & ~finished
    final_rewards = jnp.where(newly[:, None], env_state.rewards, final_rewards)

    r0 = np.asarray(final_rewards[:num_real, 0])  # drop padded games
    p0_wins = int(np.sum(r0 > 0))
    p1_wins = int(np.sum(r0 < 0))
    draws = num_real - p0_wins - p1_wins
    return p0_wins, draws, p1_wins


def evaluate_vs(
    run_mcts_fn,
    wenv,
    config,
    cur_params,
    opp_params,
    *,
    openings,
    key,
    label,
    num_simulations=None,
):
    """Play `cur_params` vs `opp_params` over every opening, from BOTH colors.
    Prints a W/D/L line + score + ΔElo estimate, and returns the score in [0,1]."""
    k1, k2 = jax.random.split(key)
    # current as seat 0
    w0, d0, l0 = run_eval_match(
        run_mcts_fn,
        wenv,
        config,
        cur_params,
        opp_params,
        opening_actions=openings,
        key=k1,
        num_simulations=num_simulations,
    )
    # current as seat 1 (swap): current's wins are the seat-1 wins
    l1, d1, w1 = run_eval_match(
        run_mcts_fn,
        wenv,
        config,
        opp_params,
        cur_params,
        opening_actions=openings,
        key=k2,
        num_simulations=num_simulations,
    )
    wins, draws, losses = w0 + w1, d0 + d1, l0 + l1
    n = max(wins + draws + losses, 1)
    score = (wins + 0.5 * draws) / n
    s = min(max(score, 1e-4), 1 - 1e-4)
    elo = -400.0 * np.log10(1.0 / s - 1.0)
    print(
        f"  [eval] vs {label:<14} {wins:>3}W {draws:>3}D {losses:>3}L "
        f"| score={score:.3f} | ΔElo≈{elo:+.0f}",
        flush=True,
    )
    return score, elo



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
    # fmt: off
    winning = {
        4: [3, 6, 9, 12],
        5: [4, 6, 7, 8, 9, 11, 12, 13, 15, 16, 17, 18, 20],
        6: [5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 30],
        7: [6, 9, 11, 12, 13, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 29,
            30, 31, 32, 33, 35, 36, 37, 39, 42],
        8: [7, 14, 15, 17, 18, 19, 20, 21, 22, 25, 26, 27, 28, 29, 30, 31, 32, 33,
            34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 48, 49, 56],
        9: [8,9,10,11,16,17,19,20,21,22,23,24,25,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,55,56,57,58,59,60,61,63,64,69,70,71,72]
    }
    # fmt: on
    if boardsize not in winning:
        return None
    n = boardsize * boardsize
    vals = np.ones(n, dtype=np.float32)
    vals[np.array(winning[boardsize])] = -1.0
    return vals.reshape((boardsize, boardsize))


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
