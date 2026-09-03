"""Batched Hex model-vs-MoHex match runtime."""

from __future__ import annotations

import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from nanoalphazero.eval.hex.engine import action_to_vertex
from nanoalphazero.eval.hex.perfect_play import opening_metrics


def _physical_size(real_size: int, *, sharded: bool) -> int:
    if not sharded:
        return real_size
    devices = max(1, jax.device_count())
    return real_size + (-real_size % devices)


def _record_move(history: list[dict[str, Any]], color: str, action: int, size: int):
    history.append(
        {
            "color": color,
            "action": int(action),
            "vertex": action_to_vertex(int(action), size),
        }
    )


def run_match(run_mcts_fn, wenv, config, params, engine_pool, *, seed: int = 1):
    """Play one real game for every possible P1 opening."""
    size = int(config["boardsize"])
    real_size = size * size
    if len(engine_pool.engines) != real_size:
        raise ValueError(
            f"Hex {size}x{size} requires {real_size} MoHex engines; "
            f"got {len(engine_pool.engines)}"
        )
    sharded = bool(config.get("enable_sharding", False))
    physical_size = _physical_size(real_size, sharded=sharded)
    openings = np.arange(real_size, dtype=np.int32)
    physical_openings = np.pad(openings, (0, physical_size - real_size))

    state = wenv.init_dummy_estate(physical_size)
    state = wenv.step(state, jnp.asarray(physical_openings))
    if sharded:
        from nanoalphazero import core

        state = jax.device_put(state, core.DATA_PARALLEL_SHARDING)

    engine_pool.reset(openings.tolist())
    histories: list[list[dict[str, Any]]] = [[] for _ in range(real_size)]
    for opening, history in zip(openings, histories, strict=True):
        _record_move(history, "b", int(opening), size)

    active = np.ones(real_size, dtype=np.bool_)
    records: list[dict[str, Any] | None] = [None] * real_size
    model_seconds = 0.0
    engine_seconds = 0.0
    started = time.perf_counter()
    current_player = 1

    # One opening stone is already on the board. Hex must finish within N^2 moves.
    for ply in range(1, real_size):
        if not active.any():
            break
        legal = np.asarray(jax.device_get(state.legal_action_mask))[:real_size]
        physical_actions = np.zeros(physical_size, dtype=np.int32)

        if current_player == 0:
            tic = time.perf_counter()
            output = run_mcts_fn(
                jax.random.PRNGKey(np.uint32(seed + ply)),
                state,
                params,
                0.0,
                physical_size,
            )
            physical_actions[:] = np.asarray(
                jax.device_get(output.action), dtype=np.int32
            )
            model_seconds += time.perf_counter() - tic
            real_actions = physical_actions[:real_size]
            for index in np.flatnonzero(active):
                action = int(real_actions[index])
                if not legal[index, action]:
                    raise ValueError(
                        f"model returned illegal action {action} in opening {index}"
                    )
                _record_move(histories[index], "b", action, size)
            engine_pool.play("b", real_actions.tolist(), active.tolist())
        else:
            tic = time.perf_counter()
            generated = engine_pool.generate("w", active.tolist())
            engine_seconds += time.perf_counter() - tic
            play_active = active.copy()
            real_actions = np.zeros(real_size, dtype=np.int32)
            for index in np.flatnonzero(active):
                action = generated[index]
                if action is None:
                    records[index] = _game_record(
                        index,
                        histories[index],
                        winner=0,
                        termination="mohex_resignation",
                    )
                    active[index] = False
                    play_active[index] = False
                    continue
                action = int(action)
                if not legal[index, action]:
                    raise ValueError(
                        f"MoHex returned illegal action {action} in opening {index}"
                    )
                real_actions[index] = action
                physical_actions[index] = action
                _record_move(histories[index], "w", action, size)
            engine_pool.play("w", real_actions.tolist(), play_active.tolist())

        state = wenv.step(state, jnp.asarray(physical_actions))
        terminated = np.asarray(jax.device_get(state.terminated))[:real_size]
        rewards = np.asarray(jax.device_get(state.rewards))[:real_size]
        for index in np.flatnonzero(active & terminated):
            winner = 0 if rewards[index, 0] > rewards[index, 1] else 1
            records[index] = _game_record(
                index, histories[index], winner=winner, termination="normal"
            )
            active[index] = False
        current_player = 1 - current_player

    for index in np.flatnonzero(active):
        records[index] = _game_record(
            index, histories[index], winner=None, termination="unscored_ply_cap"
        )
    completed = [record for record in records if record is not None]
    if len(completed) != real_size:
        raise RuntimeError("Hex evaluation did not produce one record per opening")
    elapsed = time.perf_counter() - started
    wins = sum(record["winner"] == 0 for record in completed)
    losses = sum(record["winner"] == 1 for record in completed)
    unscored = real_size - wins - losses
    summary = {
        "board_size": size,
        "real_games": real_size,
        "physical_batch_size": physical_size,
        "padding_rows": physical_size - real_size,
        "model_wins": wins,
        "model_losses": losses,
        "unscored": unscored,
        "model_win_rate": wins / real_size,
        "total_plies": sum(int(record["plies"]) for record in completed),
        "elapsed_seconds": elapsed,
        "model_search_seconds": model_seconds,
        "mohex_search_seconds": engine_seconds,
        **opening_metrics(size, completed),
    }
    return completed, summary


def _game_record(opening: int, moves: list[dict], *, winner, termination: str):
    return {
        "opening_action": int(opening),
        "opening_vertex": moves[0]["vertex"],
        "winner": winner,
        "model_won": winner == 0,
        "termination": termination,
        "plies": len(moves),
        "moves": list(moves),
    }


def training_metrics(summary: dict[str, Any]) -> dict[str, int | float]:
    keys = (
        "real_games",
        "model_wins",
        "model_losses",
        "unscored",
        "model_win_rate",
        "elapsed_seconds",
        "model_search_seconds",
        "mohex_search_seconds",
        "perfect_opening_wins",
        "perfect_opening_total",
        "perfect_opening_fraction",
        "perfect_play_achieved",
        "unexpected_opening_wins",
    )
    return {f"hex_eval/{key}": summary[key] for key in keys}


def print_grid(records: list[dict], board_size: int) -> None:
    cells = {
        int(record["opening_action"]): (
            "W" if record["winner"] == 0 else "L" if record["winner"] == 1 else "?"
        )
        for record in records
    }
    print(f"\n--- Hex {board_size}x{board_size} model (P1) vs MoHex ---")
    for row in range(board_size):
        print("  " + " ".join(cells[row * board_size + col] for col in range(board_size)))
