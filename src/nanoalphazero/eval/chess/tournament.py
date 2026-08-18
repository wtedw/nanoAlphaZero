"""Standalone searchless-compatible tournament with batched games."""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chess
import chess.pgn
import numpy as np

from nanoalphazero.eval.chess.bayeselo import find_binary, score_pgn
from nanoalphazero.eval.chess.assets import sha256_file, verified_archive_source_sha256
from nanoalphazero.eval.chess.config import (
    config_hash,
    load_config,
    public_config,
    resolve_path,
    tournament_pairings,
)
from nanoalphazero.eval.protocol import PositionBatch
from nanoalphazero.eval.chess.stockfish import (
    StockfishAdjudicator,
    StockfishPlayer,
)


RESIDENT_V1_VERSION = "resident-v1-mctx-v1"
_PROGRESS_BAR_WIDTH = 24
_NONINTERACTIVE_PROGRESS_INTERVAL_SECONDS = 30.0


def _format_unit_progress(
    *,
    unit_index: int,
    total_units: int,
    total_games: int,
    active_games: int,
    resident_size: int | None,
    dynamic_batch_min: int,
    phase: int,
    elapsed_seconds: float,
) -> str:
    completed_games = total_games - active_games
    fraction = completed_games / total_games if total_games else 1.0
    filled = min(_PROGRESS_BAR_WIDTH, int(fraction * _PROGRESS_BAR_WIDTH))
    bar = "#" * filled + "-" * (_PROGRESS_BAR_WIDTH - filled)
    units_after = max(total_units - unit_index, 0)
    if active_games == 0:
        stage = "complete"
    elif resident_size is None:
        stage = f"{active_games} active games left"
    elif resident_size <= dynamic_batch_min:
        stage = f"final b{resident_size} stage: {active_games} games left"
    else:
        next_size = max(dynamic_batch_min, resident_size // 2)
        until_next = max(active_games - resident_size // 2, 0)
        stage = (
            f"b{resident_size} stage: {active_games} active, "
            f"{until_next} games left before b{next_size}"
        )
    return (
        f"unit {unit_index}/{total_units} [{bar}] "
        f"{completed_games}/{total_games} games ({100 * fraction:5.1f}%) | "
        f"{stage} | phase {phase} | {elapsed_seconds:.0f}s elapsed | "
        f"{units_after} units after this"
    )


class _TournamentUnitProgress:
    """Low-frequency terminal progress for one independently checkpointed unit."""

    def __init__(
        self,
        *,
        unit_index: int,
        total_units: int,
        total_games: int,
        dynamic_batch_min: int,
        stream=None,
    ):
        self.unit_index = unit_index
        self.total_units = total_units
        self.total_games = total_games
        self.dynamic_batch_min = dynamic_batch_min
        self.stream = stream if stream is not None else sys.stdout
        self.interactive = bool(getattr(self.stream, "isatty", lambda: False)())
        self.started = time.perf_counter()
        self.last_printed = float("-inf")
        self.last_resident_size: int | None = None
        self.last_phase = 0
        self.last_line_width = 0
        self.rendered = False

    def update(
        self,
        active_games: int,
        resident_size: int | None,
        phase: int,
        *,
        force: bool = False,
    ) -> None:
        self.last_resident_size = resident_size
        self.last_phase = phase
        now = time.perf_counter()
        if (
            not force
            and not self.interactive
            and now - self.last_printed < _NONINTERACTIVE_PROGRESS_INTERVAL_SECONDS
        ):
            return
        line = _format_unit_progress(
            unit_index=self.unit_index,
            total_units=self.total_units,
            total_games=self.total_games,
            active_games=active_games,
            resident_size=resident_size,
            dynamic_batch_min=self.dynamic_batch_min,
            phase=phase,
            elapsed_seconds=now - self.started,
        )
        if self.interactive:
            print(
                f"\r{line.ljust(self.last_line_width)}",
                end="",
                file=self.stream,
                flush=True,
            )
            self.last_line_width = len(line)
        else:
            print(line, file=self.stream, flush=True)
        self.last_printed = now
        self.rendered = True

    def finish(self) -> None:
        self.update(0, self.last_resident_size, self.last_phase, force=True)
        self.close()

    def close(self) -> None:
        if self.interactive and self.rendered:
            print(file=self.stream, flush=True)
            self.rendered = False


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _git_sha(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _stable_seed(*parts: object) -> int:
    digest = hashlib.sha256(":".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:4], "little")


def load_openings(path: Path) -> list[chess.Board]:
    if not path.is_file():
        raise FileNotFoundError(
            f"tournament openings file not found: {path}\n"
            "Run `uv run assets fetch` to download the standard ECO "
            "openings, or correct tournament.openings in the config."
        )
    boards = []
    with path.open() as file:
        while (game := chess.pgn.read_game(file)) is not None:
            boards.append(game.end().board())
    if not boards:
        raise ValueError(f"no PGN openings found in {path}")
    return boards


def select_openings(
    boards: list[chess.Board],
    num_games_per_pair: int,
    seed: int,
    *,
    require_playable: bool = False,
) -> tuple[list[int], list[chess.Board]]:
    count = num_games_per_pair // 2
    # DeepMind's source PGN contains one terminal endpoint: zero-based entry
    # 1978 is the C44 Sea-Cadet Mate through 10.Nd5#.  It is a valid opening
    # encyclopedia trap line but a meaningless tournament starting position.
    # Resident tournaments exclude terminal encyclopedia endpoints.
    candidates = np.asarray(
        [
            index
            for index, board in enumerate(boards)
            if not require_playable or not board.is_game_over(claim_draw=True)
        ],
        dtype=np.int64,
    )
    if count > len(candidates):
        qualifier = " playable" if require_playable else ""
        raise ValueError(
            f"requested {count} openings from a set of {len(candidates)}{qualifier} openings"
        )
    indices = np.random.default_rng(seed=seed).choice(
        candidates, size=count, replace=False
    )
    return [int(index) for index in indices], [boards[int(index)] for index in indices]






def _block_tree(value):
    if value is None:
        return value
    import jax

    return jax.tree.map(
        lambda leaf: leaf.block_until_ready()
        if hasattr(leaf, "block_until_ready")
        else leaf,
        value,
    )


def _init_env_state(env, slots, *, resident: bool = False):
    if env is None:
        return None
    import jax
    import jax.numpy as jnp

    from nanoalphazero.eval.chess.action import PgxAction

    state = env.init_dummy_estate(len(slots))
    if resident:
        from nanoalphazero import core

        state = jax.device_put(state, core.DATA_PARALLEL_SHARDING)
    replay_boards = [chess.Board() for _ in slots]
    histories = [list(slot["board"].move_stack) for slot in slots]
    max_depth = max((len(history) for history in histories), default=0)
    for depth in range(max_depth):
        mask = np.asarray([depth < len(history) for history in histories])
        actions = np.zeros(len(slots), dtype=np.int32)
        for index, active in enumerate(mask):
            if active:
                move = histories[index][depth]
                actions[index] = PgxAction.encode(move, replay_boards[index])
                replay_boards[index].push(move)
        action_array = jnp.asarray(actions)
        mask_array = jnp.asarray(mask)
        if resident:
            from nanoalphazero import core

            action_array = jax.device_put(action_array, core.DATA_PARALLEL_SHARDING)
            mask_array = jax.device_put(mask_array, core.DATA_PARALLEL_SHARDING)
        stepped = env.step(state, action_array)
        state = jax.tree.map(
            lambda new, old: jnp.where(
                mask_array.reshape((-1,) + (1,) * (new.ndim - 1)), new, old
            ),
            stepped,
            state,
        )
    # Legacy callers leave this state unsharded.  The resident-v1 scheduler places
    # it once above and preserves data-parallel residency thereafter.
    return state


def _create_first_mover_slots(
    opening_indices: list[int],
    openings: list[chess.Board],
    first_mover: str,
    other: str,
):
    """Create one row per opening with ``first_mover`` owning side-to-move."""
    slots = []
    for opening_index, opening in zip(opening_indices, openings, strict=True):
        first_is_white = opening.turn == chess.WHITE
        slots.append(
            {
                "opening_index": opening_index,
                "board": copy.deepcopy(opening),
                "white": first_mover if first_is_white else other,
                "black": other if first_is_white else first_mover,
                "active": True,
                "failure": None,
                "plies_played": 0,
            }
        )
    return slots


def _next_resident_size(active: int, minimum: int, devices: int) -> int:
    target = 1 << (max(active, minimum, 1) - 1).bit_length()
    while target % devices:
        target <<= 1
    return target


def _compact_resident_state(state, mapping, slots, minimum: int):
    """Gather live rows once, then restore data-parallel resident placement."""
    import jax
    import numpy as np

    from nanoalphazero import core

    live_positions = [
        pos for pos, original in enumerate(mapping)
        if original >= 0 and slots[original]["active"]
    ]
    if not live_positions:
        return state, mapping
    target = _next_resident_size(
        len(live_positions), minimum, max(1, jax.device_count())
    )
    if target >= len(mapping):
        return state, mapping
    # Padding duplicates the first live row, matching the surrogate board used
    # by board-based JAX players for mapping entries marked -1.
    gather = live_positions + [live_positions[0]] * (target - len(live_positions))
    compacted = jax.tree.map(lambda value: value[np.asarray(gather)], state)
    compacted = jax.device_put(compacted, core.DATA_PARALLEL_SHARDING)
    new_mapping = [mapping[pos] for pos in live_positions]
    new_mapping.extend([-1] * (target - len(live_positions)))
    return compacted, new_mapping






def _adjudicated_result(
    board: chess.Board,
    info: dict[str, Any] | None,
    threshold: int | None,
):
    if info is None or threshold is None:
        return None
    score = info["score"].relative
    raw = score.score()
    too_high = score.is_mate() or (raw is not None and abs(raw) > threshold)
    if not too_high:
        return None
    winning = score.mate() > 0 if score.is_mate() else raw > 0
    white_wins = (board.turn == chess.WHITE and winning) or (
        board.turn == chess.BLACK and not winning
    )
    return "1-0" if white_wins else "0-1"




def _games_from_slots(slots, event: str) -> list[chess.pgn.Game]:
    games = []
    today = datetime.now(timezone.utc).strftime("%Y.%m.%d")
    for slot in slots:
        game = chess.pgn.Game.from_board(slot["board"])
        game.headers["Event"] = event
        game.headers["Date"] = slot.get("event_date", today)
        game.headers["White"] = slot["white"]
        game.headers["Black"] = slot["black"]
        game.headers["Result"] = slot.get("result", "*")
        game.headers["Termination"] = slot.get("termination", "unknown")
        game.headers["OpeningIndex"] = str(slot["opening_index"])
        games.append(game)
    return games




def _resident_actions(player, env_state, mapping, slots, seed: int):
    """Return pgx labels without ever gathering the resident pgx state."""
    from nanoalphazero.eval.chess.action import PgxAction

    if hasattr(player, "play_actions"):
        return np.asarray(player.play_actions(env_state, seed=seed), dtype=np.int32)

    live_positions = [
        position
        for position, original in enumerate(mapping)
        if original >= 0 and slots[original]["active"]
    ]
    if not live_positions:
        raise RuntimeError("resident action request has no live games")
    live = [mapping[position] for position in live_positions]

    if getattr(player, "resident_live_only", False):
        boards = tuple(slots[original]["board"] for original in live)
        batch = PositionBatch(
            slot_ids=tuple(live),
            boards=boards,
            env_state=None,
            seeds=tuple(_stable_seed(seed, position) for position in live_positions),
        )
        moves = list(player.play_batch(batch))
        if len(moves) != len(live_positions):
            raise ValueError(
                f"{player.name} returned {len(moves)} moves for "
                f"{len(live_positions)} live resident rows"
            )
        encoded = [
            PgxAction.encode(move, board)
            for move, board in zip(moves, boards, strict=True)
        ]
        # The label for inactive rows is semantically ignored.  Padding rows
        # duplicate a live state, so using a live label also keeps it in the
        # valid pgx action range without asking Stockfish to search it.
        actions = np.full(len(mapping), encoded[0], dtype=np.int32)
        actions[np.asarray(live_positions)] = np.asarray(encoded, dtype=np.int32)
        return actions

    surrogate = slots[live[0]]["board"]
    boards = tuple(
        slots[original]["board"]
        if original >= 0 and slots[original]["active"]
        else surrogate
        for original in mapping
    )
    batch = PositionBatch(
        slot_ids=tuple(mapping),
        boards=boards,
        env_state=None,
        seeds=tuple(_stable_seed(seed, row) for row in range(len(mapping))),
    )
    moves = list(player.play_batch(batch))
    if len(moves) != len(mapping):
        raise ValueError(
            f"{player.name} returned {len(moves)} moves for {len(mapping)} resident rows"
        )
    return np.asarray(
        [PgxAction.encode(move, board) for move, board in zip(moves, boards, strict=True)],
        dtype=np.int32,
    )


def run_resident_v1_unit(
    player_a,
    player_b,
    adjudicator,
    env,
    opening_indices,
    openings,
    *,
    first_mover: str,
    seed: int,
    pairing_id: str,
    max_plies: int,
    dynamic_batch_min: int = 32,
    profile_stages: bool = False,
    progress=None,
):
    """Play one resident-v1 first-mover unit with a resident pgx1 state."""
    import jax
    import jax.numpy as jnp

    from nanoalphazero.eval.chess.action import PgxAction

    players = {player_a.name: player_a, player_b.name: player_b}
    if first_mover not in players:
        raise ValueError(f"unknown first mover {first_mover!r}")
    other = player_b.name if first_mover == player_a.name else player_a.name
    overall_started = time.perf_counter()
    stage_seconds = Counter()
    batch_sizes = Counter()

    stage_started = time.perf_counter()
    slots = _create_first_mover_slots(
        opening_indices, openings, first_mover, other
    )
    env_state = _init_env_state(env, slots, resident=True)
    if profile_stages:
        _block_tree(env_state)
    stage_seconds["initialize_slots_and_resident_env"] += (
        time.perf_counter() - stage_started
    )

    mapping = list(range(len(slots)))
    current_player = first_mover
    phase = 0
    total_move_seconds = Counter()
    compactions = []
    if progress is not None:
        progress.update(len(slots), len(mapping), phase, force=True)
    while any(slot["active"] for slot in slots):
        active_before = np.asarray(
            [original >= 0 and slots[original]["active"] for original in mapping],
            dtype=np.bool_,
        )
        live_positions = np.flatnonzero(active_before).tolist()
        if not live_positions:
            raise RuntimeError("active resident tournament slots lost their state mapping")
        pgx_players = np.asarray(jax.device_get(env_state.current_player))
        for position in live_positions:
            original = mapping[position]
            expected = 0 if slots[original]["board"].turn == chess.WHITE else 1
            if int(pgx_players[position]) != expected:
                raise RuntimeError(
                    "resident pgx/Python turn mismatch before search: "
                    f"phase={phase}, position={position}, original={original}, "
                    f"opening={slots[original]['opening_index']}, "
                    f"pgx_player={int(pgx_players[position])}, expected={expected}"
                )

        batch_sizes[len(mapping)] += 1
        action_seed = _stable_seed(
            seed, pairing_id, first_mover, phase, current_player
        )
        stage_started = time.perf_counter()
        actions = _resident_actions(
            players[current_player], env_state, mapping, slots, action_seed
        )
        move_elapsed = time.perf_counter() - stage_started
        if actions.shape != (len(mapping),):
            raise ValueError(
                f"{current_player} returned action shape {actions.shape}; "
                f"expected {(len(mapping),)}"
            )
        total_move_seconds[current_player] += move_elapsed
        stage_seconds["resident_player_search"] += move_elapsed

        stage_started = time.perf_counter()
        action_array = jax.device_put(jnp.asarray(actions), env.data_sharding)
        # pgx1 already makes stepping terminated rows safe.  Adjudicated and
        # padding rows may advance, but are semantically ignored and are never
        # decoded or copied into a real game's Python/PGN state.
        env_state = env.step(env_state, action_array)
        if profile_stages:
            _block_tree(env_state)
        stage_seconds["resident_step"] += time.perf_counter() - stage_started

        moved_originals = []
        stage_started = time.perf_counter()
        for position in live_positions:
            original = mapping[position]
            slot = slots[original]
            board = slot["board"]
            move = PgxAction.decode(int(actions[position]), board)
            if move not in board.legal_moves:
                raise ValueError(
                    f"{current_player} returned illegal move {move.uci()} for {board.fen()} "
                    f"at phase={phase}, position={position}, original={original}, "
                    f"opening={slot['opening_index']}"
                )
            board.push(move)
            slot["plies_played"] += 1
            moved_originals.append(original)
        stage_seconds["decode_validate_and_push_python_boards"] += (
            time.perf_counter() - stage_started
        )

        stage_started = time.perf_counter()
        if adjudicator is None:
            infos = [None] * len(moved_originals)
        else:
            infos = adjudicator.analyse_batch(
                tuple(slots[original]["board"] for original in moved_originals)
            )
        stage_seconds["stockfish_adjudication"] += time.perf_counter() - stage_started

        stage_started = time.perf_counter()
        terminated = np.asarray(jax.device_get(env_state.terminated))
        for position, original, info in zip(
            live_positions, moved_originals, infos, strict=True
        ):
            slot = slots[original]
            board = slot["board"]
            result = _adjudicated_result(
                board,
                info,
                adjudicator.threshold_cp if adjudicator is not None else None,
            )
            if result is not None:
                slot["result"] = result
                slot["termination"] = "Stockfish adjudication"
                slot["active"] = False
            elif board.is_game_over() or board.can_claim_fifty_moves() or board.is_repetition():
                slot["result"] = board.result(claim_draw=True)
                slot["termination"] = "normal"
                slot["active"] = False
            elif terminated[position]:
                slot["result"] = "*"
                slot["termination"] = "PGX/python-chess termination mismatch"
                slot["failure"] = slot["termination"]
                slot["active"] = False
            elif slot["plies_played"] >= max_plies:
                slot["result"] = "*"
                slot["termination"] = f"unscored ply cap ({max_plies})"
                slot["failure"] = slot["termination"]
                slot["active"] = False
        stage_seconds["termination_and_ply_cap_checks"] += (
            time.perf_counter() - stage_started
        )

        phase += 1
        current_player = other if current_player == first_mover else first_mover

        # Only compact after both entrants have completed a turn in the round.
        active_count = sum(slot["active"] for slot in slots)
        if (
            active_count
            and phase % 2 == 0
            and active_count <= len(mapping) // 2
        ):
            old_size = len(mapping)
            stage_started = time.perf_counter()
            env_state, mapping = _compact_resident_state(
                env_state, mapping, slots, dynamic_batch_min
            )
            if profile_stages:
                _block_tree(env_state)
            stage_seconds["dynamic_compaction"] += time.perf_counter() - stage_started
            if len(mapping) < old_size:
                compactions.append(
                    {
                        "phase": phase,
                        "active_games": active_count,
                        "from_batch_size": old_size,
                        "to_batch_size": len(mapping),
                        "padding_rows": len(mapping) - active_count,
                    }
                )
        if progress is not None and (phase % 2 == 0 or active_count == 0):
            progress.update(active_count, len(mapping), phase)

    stage_started = time.perf_counter()
    games = _games_from_slots(
        slots, f"nanoAlphaZero resident-v1 tournament: {player_a.name} vs {player_b.name}"
    )
    stage_seconds["build_output_pgn_objects"] += time.perf_counter() - stage_started
    wall_seconds = time.perf_counter() - overall_started
    measured_seconds = sum(stage_seconds.values())
    return games, {
        "scheduler_version": RESIDENT_V1_VERSION,
        "stage_timings_synchronized": profile_stages,
        "first_mover": first_mover,
        "wall_seconds": wall_seconds,
        "move_phase_seconds": dict(total_move_seconds),
        "profile_stage_seconds": dict(stage_seconds),
        "profile_unattributed_seconds": wall_seconds - measured_seconds,
        "profile_batch_size_calls": {
            str(size): count for size, count in sorted(batch_sizes.items())
        },
        "compactions": compactions,
        "move_phases": phase,
        "games": len(slots),
        "plies": sum(slot["plies_played"] for slot in slots),
        "failures": sum(slot["failure"] is not None for slot in slots),
    }


def _write_games(path: Path, games: list[chess.pgn.Game]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as file:
        for game in games:
            print(game, file=file, end="\n\n")
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(path)


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _rebuild_pgn(run_dir: Path, units: list[str]) -> Path:
    pgn_path = run_dir / "games.pgn"
    temporary = run_dir / "games.pgn.tmp"
    with temporary.open("wb") as output:
        for unit in units:
            output.write((run_dir / "units" / f"{unit}.pgn").read_bytes())
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(pgn_path)
    return pgn_path


def _result_counts(games: list[chess.pgn.Game], player_a: str) -> dict[str, int]:
    counts = Counter()
    for game in games:
        result = game.headers["Result"]
        if result == "*":
            counts["unscored"] += 1
        elif result == "1/2-1/2":
            counts["draws"] += 1
        else:
            winner = game.headers["White"] if result == "1-0" else game.headers["Black"]
            counts["wins" if winner == player_a else "losses"] += 1
    return dict(counts)


def _format_unit_report(record: dict[str, Any]) -> str:
    wall_seconds = float(record["wall_seconds"])
    games = int(record["games"])
    plies = int(record["plies"])
    lines = [
        "",
        f"completed tournament unit: {record['unit']}",
        f"  pairing: {record['player_a']} vs {record['player_b']}",
    ]
    if first_mover := record.get("first_mover"):
        lines.append(f"  first mover: {first_mover}")
    lines.extend(
        [
            f"  results from {record['player_a']}: "
            f"{json.dumps(record['results_from_a'], sort_keys=True)}",
            f"  games: {games}  plies: {plies}  failures: {record['failures']}",
            f"  wall: {wall_seconds:.3f}s  games/s: "
            f"{games / wall_seconds if wall_seconds else 0.0:.3f}  plies/s: "
            f"{plies / wall_seconds if wall_seconds else 0.0:.3f}",
        ]
    )

    synchronized = bool(record.get("stage_timings_synchronized", False))
    qualifier = "synchronized" if synchronized else "non-synchronizing estimates"
    lines.append(f"  stage timings ({qualifier}):")
    stages = record.get("profile_stage_seconds", {})
    for stage, seconds in sorted(
        stages.items(), key=lambda item: (-item[1], item[0])
    ):
        share = 100.0 * float(seconds) / wall_seconds if wall_seconds else 0.0
        lines.append(f"    {stage}: {float(seconds):.3f}s ({share:.1f}%)")
    unattributed = float(record.get("profile_unattributed_seconds", 0.0))
    share = 100.0 * unattributed / wall_seconds if wall_seconds else 0.0
    lines.append(f"    unattributed: {unattributed:.3f}s ({share:.1f}%)")
    if not synchronized:
        lines.append(
            "    note: timings preserve normal JAX/CPU overlap; set "
            "NANOAZ_PROFILE_TOURNAMENT=1 for synchronized attribution"
        )

    lines.append("  model move timings:")
    for player, seconds in record.get("move_phase_seconds", {}).items():
        lines.append(f"    {player}: {float(seconds):.3f}s")
    batch_calls = record.get("profile_batch_size_calls", {})
    if batch_calls:
        rendered = ", ".join(
            f"batch {size}: {calls} calls"
            for size, calls in sorted(
                batch_calls.items(),
                key=lambda item: int(item[0]),
                reverse=True,
            )
        )
        lines.append(f"  resident batch calls: {rendered}")
    compactions = record.get("compactions", [])
    if compactions:
        lines.append("  compactions:")
        for compaction in compactions:
            lines.append(
                f"    phase {compaction['phase']}: "
                f"{compaction['from_batch_size']} -> {compaction['to_batch_size']} "
                f"({compaction['active_games']} active, "
                f"{compaction['padding_rows']} padding)"
            )
    return "\n".join(lines)


def _print_unit_report(record: dict[str, Any]) -> None:
    print(_format_unit_report(record), flush=True)


def _build_stockfish_players(config, batch_size):
    players = {}
    for agent in config["agents"]:
        if agent["kind"] == "stockfish":
            players[agent["name"]] = StockfishPlayer(agent["name"], agent, batch_size)
    return players


def _build_neural_players(config, batch_size, env, players):
    for agent in config["agents"]:
        if agent["kind"] == "searchless":
            from nanoalphazero.eval.chess.searchless import SearchlessPlayer

            players[agent["name"]] = SearchlessPlayer(
                agent["name"], agent, batch_size
            )
        elif agent["kind"] == "kata":
            from nanoalphazero.eval.chess.mctx_player import MctxPlayer

            players[agent["name"]] = MctxPlayer(
                agent["name"],
                agent,
                batch_size,
                env,
                max_plies=int(config["tournament"].get("max_plies", 512)),
            )
    return players


def _prepare_paths(config, source, resume, output_root):
    digest = config.get("_run_config_hash", config_hash(config))
    if resume:
        run_dir = Path(resume).expanduser().resolve()
        manifest = json.loads((run_dir / "manifest.json").read_text())
        expected_version = RESIDENT_V1_VERSION
        if expected_version and manifest.get("scheduler_version") != expected_version:
            raise ValueError(
                "cannot resume an incompatible manifest with the "
                f"{expected_version} scheduler"
            )
        if manifest["config_hash"] != digest:
            raise ValueError("resume config does not match the existing run")
        return run_dir, manifest
    root = (
        Path(output_root).expanduser().resolve()
        if output_root
        else resolve_path(
            config,
            config["tournament"].get("output_root", "runs"),
        )
    )
    run_dir = root / f"{_utc_stamp()}-{digest[:10]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(source, run_dir / "config.toml")
    manifest = {
        "config_hash": digest,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed_units": [],
    }
    manifest["scheduler_version"] = RESIDENT_V1_VERSION
    _write_json(run_dir / "manifest.json", manifest)
    return run_dir, manifest


def _resolve_agent_paths(config):
    for agent in config["agents"]:
        if "checkpoint" in agent:
            agent["checkpoint"] = str(resolve_path(config, agent["checkpoint"]))
    tournament = config["tournament"]
    tournament["openings"] = str(
        resolve_path(config, tournament.get("openings", "data/eval/eco_openings.pgn"))
    )


def _configured_bayeselo_binary(config) -> str | None:
    binary = config.get("bayeselo", {}).get("binary")
    return str(resolve_path(config, binary)) if binary else None


def _dependency_versions() -> dict[str, str]:
    names = (
        "nanoalphazero",
        "jax",
        "jaxlib",
        "libtpu",
        "numpy",
        "mctx",
        "pgx1",
        "python-chess",
        "orbax-checkpoint",
    )
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _asset_hashes(config) -> dict[str, str]:
    hashes = {"openings": sha256_file(Path(config["tournament"]["openings"]))}
    for agent in config["agents"]:
        checkpoint = agent.get("checkpoint")
        if not checkpoint:
            continue
        path = Path(checkpoint)
        if path.is_file():
            hashes[f"checkpoint:{agent['name']}"] = sha256_file(path)
        else:
            hashes[f"checkpoint:{agent['name']}"] = (
                verified_archive_source_sha256(path) or "unverified-directory"
            )
    return hashes


def _maybe_log_wandb(config, run_dir: Path, summary: dict[str, Any]) -> None:
    settings = config.get("wandb", {})
    if not settings.get("enabled", False):
        return
    import wandb

    run = wandb.init(
        project=settings.get("project", "nanoAlphaZero-eval"),
        entity=settings.get("entity"),
        name=settings.get("name", run_dir.name),
        config=public_config(config),
    )
    flat = {
        "tournament/games": summary["games_in_pgn"],
        "tournament/elapsed_seconds": summary["elapsed_seconds_this_process"],
    }
    for name, rating in summary.get("bayeselo", {}).get("ratings", {}).items():
        flat[f"bayeselo/{name}/elo"] = rating["elo"]
        flat[f"bayeselo/{name}/score_pct"] = rating["score_pct"]
    run.log(flat)
    artifact = wandb.Artifact(run_dir.name, type="tournament-results")
    artifact.add_dir(str(run_dir))
    run.log_artifact(artifact)
    run.finish()


def run_tournament(config, source, *, resume=None, output_root=None, skip_bayeselo=False):
    """Run a fixed-sample resident-v1 chess tournament."""
    from nanoalphazero.eval.wandb_artifacts import materialize_agent_checkpoints

    config["_run_config_hash"] = config_hash(config)
    _resolve_agent_paths(config)

    tournament = config["tournament"]
    batch_size = int(tournament["batch_size"])
    num_games = int(tournament["num_games_per_pair"])
    seed = int(tournament.get("seed", 1))
    max_plies = int(tournament.get("max_plies", 512))
    dynamic_batch_min = int(tournament.get("dynamic_batch_min", 32))
    profile_stages = os.environ.get("NANOAZ_PROFILE_TOURNAMENT") == "1"

    # Fail on a missing/invalid opening book before downloading large models.
    openings_all = load_openings(Path(tournament["openings"]))
    opening_indices, openings = select_openings(
        openings_all, num_games, seed, require_playable=True
    )
    engine_configs = [
        agent for agent in config["agents"] if agent["kind"] == "stockfish"
    ]
    adjudication = dict(config.get("adjudication", {"enabled": False}))
    if bool(adjudication.get("enabled", False)):
        engine_configs.append(adjudication)
    for engine_config in engine_configs:
        engine_path = Path(
            engine_config.get("path", "/usr/local/bin/stockfish")
        ).expanduser()
        if not engine_path.is_file():
            raise FileNotFoundError(f"Stockfish executable not found: {engine_path}")
    materialize_agent_checkpoints(config)

    run_dir, manifest = _prepare_paths(
        config, source, resume, output_root
    )
    (run_dir / "units").mkdir(exist_ok=True)
    pgn_path = run_dir / "games.pgn"
    summary_path = run_dir / "summary.json"
    progress_path = run_dir / "progress.json"

    _write_json(run_dir / "resolved.json", public_config(config))

    pairings = tournament_pairings(config)
    expected_units = {
        f"{player_a}__{player_b}__first-{first_mover}"
        for player_a, player_b in pairings
        for first_mover in (player_a, player_b)
    }
    completed = set(manifest["completed_units"])
    if (
        expected_units.issubset(completed)
        and pgn_path.exists()
        and summary_path.exists()
    ):
        summary = json.loads(summary_path.read_text())
        if not skip_bayeselo and "bayeselo" not in summary:
            summary["bayeselo"] = score_pgn(
                pgn_path, binary=_configured_bayeselo_binary(config)
            )
            (run_dir / "bayeselo.txt").write_text(
                summary["bayeselo"]["raw_output"]
            )
            _write_json(summary_path, summary)
            _maybe_log_wandb(config, run_dir, summary)
            print(f"scored completed tournament with BayesElo: {run_dir}")
        print(f"already complete: {run_dir}")
        return run_dir, summary

    adjudication_enabled = bool(adjudication.get("enabled", False))
    pair_records = (
        json.loads(progress_path.read_text()) if progress_path.exists() else []
    )
    players: dict[str, Any] = {}
    adjudicator = None
    player_stats: dict[str, Any] = {}
    adjudicator_stats: dict[str, Any] = {"enabled": adjudication_enabled}
    started = time.perf_counter()

    try:
        # UCI processes must be created before importing JAX/libtpu.
        if adjudication_enabled:
            adjudicator = StockfishAdjudicator(adjudication, batch_size)
        players = _build_stockfish_players(config, batch_size)

        from nanoalphazero.eval.chess.mctx_player import make_pgx1_env
        import jax

        if batch_size % jax.device_count():
            raise ValueError(
                f"resident batch_size {batch_size} must be divisible by "
                f"the JAX device count {jax.device_count()}"
            )

        env = make_pgx1_env()
        players = _build_neural_players(config, batch_size, env, players)
        for player in players.values():
            if not isinstance(player, StockfishPlayer):
                player.warmup(batch_size)

        completed_unit_count = len(expected_units.intersection(completed))
        total_unit_count = len(expected_units)
        for player_a, player_b in pairings:
            for first_mover in (player_a, player_b):
                unit = f"{player_a}__{player_b}__first-{first_mover}"
                if unit in completed:
                    continue
                print(
                    f"pair {player_a} vs {player_b}: {first_mover} moves first "
                    f"({len(openings)} resident games)",
                    flush=True,
                )
                progress = _TournamentUnitProgress(
                    unit_index=completed_unit_count + 1,
                    total_units=total_unit_count,
                    total_games=len(openings),
                    dynamic_batch_min=dynamic_batch_min,
                )
                try:
                    games, timing = run_resident_v1_unit(
                        players[player_a],
                        players[player_b],
                        adjudicator,
                        env,
                        opening_indices,
                        openings,
                        first_mover=first_mover,
                        seed=seed,
                        pairing_id=f"{player_a}__{player_b}",
                        max_plies=max_plies,
                        dynamic_batch_min=dynamic_batch_min,
                        profile_stages=profile_stages,
                        progress=progress,
                    )
                except BaseException:
                    progress.close()
                    raise
                progress.finish()

                _write_games(run_dir / "units" / f"{unit}.pgn", games)
                record = {
                    "unit": unit,
                    "player_a": player_a,
                    "player_b": player_b,
                    "opening_offset": 0,
                    "results_from_a": _result_counts(games, player_a),
                    **timing,
                }
                pair_records.append(record)
                manifest["completed_units"].append(unit)
                manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
                _write_json(run_dir / "manifest.json", manifest)
                _write_json(progress_path, pair_records)
                pgn_path = _rebuild_pgn(
                    run_dir, manifest["completed_units"]
                )
                _print_unit_report(record)
                completed.add(unit)
                completed_unit_count += 1
    finally:
        player_stats = {
            name: player.stats() for name, player in players.items()
        }
        if adjudicator is not None:
            adjudicator_stats = {"enabled": True, **adjudicator.stats()}
        for player in players.values():
            player.close()
        if adjudicator is not None:
            adjudicator.close()

    total_plies = sum(record["plies"] for record in pair_records)
    game_wall_seconds = sum(
        record["wall_seconds"] for record in pair_records
    )
    game_count = (
        pgn_path.read_text().count("[Event ") if pgn_path.exists() else 0
    )
    summary = {
        "config_hash": config["_run_config_hash"],
        "scheduler_version": RESIDENT_V1_VERSION,
        "games_in_pgn": game_count,
        "expected_games": len(pairings) * num_games,
        "elapsed_seconds_this_process": time.perf_counter() - started,
        "pair_batch_wall_seconds": game_wall_seconds,
        "total_plies": total_plies,
        "games_per_second": (
            game_count / game_wall_seconds if game_wall_seconds else 0.0
        ),
        "plies_per_second": (
            total_plies / game_wall_seconds if game_wall_seconds else 0.0
        ),
        "pair_batches": pair_records,
        "player_stats": player_stats,
        "adjudicator_stats": adjudicator_stats,
        "asset_sha256": _asset_hashes(config),
        "dependency_versions": _dependency_versions(),
        "devices": (
            [str(device) for device in sys.modules["jax"].devices()]
            if "jax" in sys.modules
            else []
        ),
        "nanoalphazero_git_sha": _git_sha(
            Path(__file__).resolve().parents[4]
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    # Games and the base summary survive missing or crashing external scorers.
    _write_json(summary_path, summary)
    if not skip_bayeselo:
        summary["bayeselo"] = score_pgn(
            pgn_path, binary=_configured_bayeselo_binary(config)
        )
        (run_dir / "bayeselo.txt").write_text(
            summary["bayeselo"]["raw_output"]
        )
    _write_json(summary_path, summary)
    _maybe_log_wandb(config, run_dir, summary)
    print(f"wrote {run_dir}")
    return run_dir, summary


def tournament_main(
    config_path: str,
    *,
    resume: str | None = None,
    output_root: str | None = None,
    skip_bayeselo: bool = False,
) -> None:
    config, source = load_config(config_path)
    if not skip_bayeselo:
        try:
            find_binary(_configured_bayeselo_binary(config))
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"{error}\n"
                "The tournament has not started. Install BayesElo, or add "
                "`--skip-bayeselo` and score games.pgn later with "
                "`uv run bayeselo --pgn <games.pgn>`."
            ) from None
    run_tournament(
        config,
        source,
        resume=resume,
        output_root=output_root,
        skip_bayeselo=skip_bayeselo,
    )
