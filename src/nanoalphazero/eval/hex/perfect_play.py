"""Solved first-move outcomes for the supported Hex boards."""

from __future__ import annotations

import numpy as np


P1_WINNING_OPENINGS: dict[int, tuple[int, ...]] = {
    4: (3, 6, 9, 12),
    5: (4, 6, 7, 8, 9, 11, 12, 13, 15, 16, 17, 18, 20),
    6: (
        5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
        22, 23, 24, 25, 26, 27, 28, 30,
    ),
    7: (
        6, 9, 11, 12, 13, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25,
        26, 27, 29, 30, 31, 32, 33, 35, 36, 37, 39, 42,
    ),
    8: (
        7, 14, 15, 17, 18, 19, 20, 21, 22, 25, 26, 27, 28, 29, 30,
        31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 48,
        49, 56,
    ),
    9: (
        8, 9, 10, 11, 16, 17, 19, 20, 21, 22, 23, 24, 25, 27, 28,
        29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
        44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 55, 56, 57, 58, 59,
        60, 61, 63, 64, 69, 70, 71, 72,
    ),
}


def p1_winning_openings(board_size: int) -> tuple[int, ...] | None:
    return P1_WINNING_OPENINGS.get(int(board_size))


def perfect_play_values(board_size: int) -> np.ndarray | None:
    winning = p1_winning_openings(board_size)
    if winning is None:
        return None
    values = np.ones(board_size * board_size, dtype=np.float32)
    values[np.asarray(winning, dtype=np.int32)] = -1.0
    return values.reshape((board_size, board_size))


def opening_metrics(board_size: int, records: list[dict]) -> dict[str, int | float]:
    required = set(p1_winning_openings(board_size) or ())
    model_wins = {
        int(record["opening_action"])
        for record in records
        if record.get("winner") == 0
    }
    won = len(required & model_wins)
    total = len(required)
    return {
        "perfect_opening_wins": won,
        "perfect_opening_total": total,
        "perfect_opening_fraction": won / total if total else float("nan"),
        "perfect_play_achieved": int(bool(total) and won == total),
        "unexpected_opening_wins": len(model_wins - required),
    }

