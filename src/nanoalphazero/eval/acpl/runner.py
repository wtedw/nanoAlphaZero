"""Explicit-state ACPL orchestration; CPU engines stay local to each host."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import uuid

import chess.engine
import chess.pgn
import jax
import numpy as np
from jax.experimental import multihost_utils

from nanoalphazero.eval.acpl.config import resolve_acpl
from nanoalphazero.eval.acpl import AcplResult
from nanoalphazero.eval.acpl.rollout import collect_games
from nanoalphazero.eval.acpl.scoring import score_fens


@dataclass(frozen=True)
class AcplState:
    runs: int = 0
    last_cycle: int = -1
    last_output: str | None = None


def load_openings(options):
    path = Path(options.openings) if options.openings else Path(__file__).with_name("adam12.json")
    rows = json.loads(path.read_text())
    if not 1 <= options.num_games <= len(rows):
        raise ValueError(f"ACPL num_games must be between 1 and {len(rows)}")
    openings = np.asarray(rows[:options.num_games], dtype=np.int32)
    if openings.ndim != 2 or openings.shape[1] == 0:
        raise ValueError("ACPL openings must be nonempty equal-length action sequences")
    return openings, hashlib.sha256(path.read_bytes()).hexdigest()


def write_games(games, directory):
    for game in games:
        pgn = chess.pgn.Game.from_board(game.board)
        outcome = game.board.outcome()
        pgn.headers["Result"] = outcome.result() if outcome else "*"
        pgn.headers["Termination"] = "normal" if game.terminated else "ply cap"
        pgn.headers["OpeningPlies"] = str(game.opening_plies)
        (directory / f"game-{game.index:06d}.pgn").write_text(str(pgn) + "\n\n")
        (directory / f"game-{game.index:06d}.fens.json").write_text(json.dumps(game.fens) + "\n")


def analyze_games(games, options):
    """Preserve parallelism ACROSS games and sequential engine reuse WITHIN each."""
    def analyze(game):
        outcome = game.board.outcome()
        row = dict(game=game.index, plies=len(game.board.move_stack),
                   opening_plies=game.opening_plies, terminated=game.terminated,
                   white_wins=int(bool(outcome and outcome.winner is True)),
                   black_wins=int(bool(outcome and outcome.winner is False)),
                   # Legacy summary counts unfinished games as draws; also expose unfinished.
                   draws=int(not outcome or outcome.winner is None), unfinished=int(not game.terminated))
        try:
            row.update(score_fens(game.fens, options, game.opening_plies))
            row["analysis_ok"] = True
        except (TimeoutError, chess.engine.EngineTerminatedError) as error:
            row.update(analysis_ok=False, error=f"{type(error).__name__}: {error}")
        return row

    if not games:
        return []
    with ThreadPoolExecutor(max_workers=min(len(games), options.workers)) as executor:
        futures = [executor.submit(analyze, game) for game in games]
        rows = [future.result() for future in as_completed(futures)]
    return sorted(rows, key=lambda row: row["game"])


def summarize(rows):
    metrics = {f"acpl/{name}": sum(row[name] for row in rows)
               for name in ("white_wins", "black_wins", "draws", "unfinished")}
    metrics.update({"acpl/games": len(rows),
                    "acpl/analyzed_games": sum(row["analysis_ok"] for row in rows),
                    "acpl/failed_games": sum(not row["analysis_ok"] for row in rows)})
    for name in ("acpl_white", "acpl_black", "stdcpl_white", "stdcpl_black"):
        values = [row[name] for row in rows if row["analysis_ok"] and row.get(name) is not None]
        if values:
            metrics[f"acpl/{name}"] = float(np.mean(values))
    for side in ("white", "black"):
        metrics[f"acpl/scored_moves_{side}"] = sum(row.get(f"moves_{side}", 0) for row in rows)
    return metrics


def make_acpl(config, *, env, policy_fn, data_sharding=None):
    """Return eval_acpl_fn(state, cycle, params) -> (state, AcplResult), and state.

    No tracker/model-class dependencies, persistent engines, module-global mesh,
    or evaluator checkpoint state. All hosts must call with the same cycle.
    """
    options = resolve_acpl(config)
    if options.period:
        if config["env_id"] != "chess":
            raise ValueError("ACPL requires chess")
        if options.period < 0 or min(options.depth, options.workers, options.engine_timeout) <= 0 or options.engine_retries < 0:
            raise ValueError("ACPL periods/budgets must be positive and retries nonnegative")
        path = Path(options.stockfish_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"ACPL Stockfish executable not found: {path}")
        options = replace(options, stockfish_path=path)
        openings, opening_hash = load_openings(options)
        max_plies = options.max_plies if options.max_plies is not None else config["game_max_steps"]
        if max_plies <= openings.shape[1]:
            raise ValueError("ACPL max_plies must allow at least one model move after the opening")

    def eval_acpl_fn(state, cycle, params):
        if not options.period or cycle % options.period or state.last_cycle == cycle:
            return state, AcplResult()
        started = time.perf_counter()
        host, hosts = jax.process_index(), jax.process_count()
        # Fixed-length identifier broadcast preserves shared output paths across hosts.
        token = (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex).encode()
        token = np.frombuffer(token, dtype=np.uint8)
        if hosts > 1:
            token = multihost_utils.broadcast_one_to_all(token, is_source=host == 0)
        directory = Path(options.output).expanduser() / f"cycle-{cycle:06d}-{bytes(token).decode()}"
        directory.mkdir(parents=True, exist_ok=False if hosts == 1 else True)
        if host == 0:
            manifest = {
                "cycle": cycle, "options": asdict(options), "hosts": hosts,
                "opening_sha256": opening_hash,
                "engine_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "scoring": {"cp_cap": 1000, "mate_score": 10000,
                            "initial_white_cp": 15 if options.include_openings else "analysed",
                            "aggregation": "equal weight per game; mean of within-game sample stddevs"},
                "model_config": config,
            }
            (directory / "manifest.json").write_text(json.dumps(manifest, default=str, indent=2) + "\n")
        print(f"ACPL_BEGIN cycle={cycle} host={host} games={len(openings)} output={directory}", flush=True)
        games, placement = collect_games(env, policy_fn, params, openings, max_plies, data_sharding)
        rollout_seconds = time.perf_counter() - started
        write_games(games, directory)
        (directory / f"placement-{host}.json").write_text(json.dumps(placement, indent=2) + "\n")
        analysis_started = time.perf_counter()
        rows = analyze_games(games, options)
        analysis_seconds = time.perf_counter() - analysis_started
        (directory / f"scores-{host}.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
        if hosts > 1:
            # Variable-size JSON records stay on the shared filesystem. All hosts
            # read identical results only after every host finishes its shard.
            multihost_utils.sync_global_devices(f"acpl_written_{cycle}_{state.runs}")
            rows = [row for index in range(hosts)
                    for row in json.loads((directory / f"scores-{index}.json").read_text())]
        metrics = summarize(rows)
        timings = np.array([rollout_seconds, analysis_seconds, time.perf_counter() - started])
        if hosts > 1:
            timings = np.max(multihost_utils.process_allgather(timings, tiled=False), axis=0)
        metrics.update(dict(zip(("acpl/rollout_seconds", "acpl/analysis_seconds", "acpl/seconds"), map(float, timings))))
        if host == 0:
            (directory / "summary.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
            with (directory / "games.pgn").open("w") as stream:
                for pgn in sorted(directory.glob("game-*.pgn")):
                    stream.write(pgn.read_text())
            print("ACPL_RESULT " + json.dumps(metrics, allow_nan=False), flush=True)
        if hosts > 1:
            multihost_utils.sync_global_devices(f"acpl_done_{cycle}_{state.runs}")
        files = ({name: directory / name for name in ("games.pgn", "manifest.json", "summary.json")}
                 if host == 0 else {})
        return AcplState(state.runs + 1, cycle, str(directory)), AcplResult(metrics, files)

    return eval_acpl_fn, AcplState()
