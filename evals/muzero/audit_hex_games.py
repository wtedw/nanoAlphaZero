"""Independently check saved Hex matches with graph connectivity, without JAX.

PGX1 starts at _turn=1: player 0 north/south, player 1 west/east.
This audit uses neither PGX stepping nor model predictions to decide winners.
Run: uv run evals/muzero/audit_hex_games.py EVAL_DIRECTORY --size 7
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np


def connected(board, player, size):
    starts = (range(size) if player == 0 else range(0, size * size, size))
    frontier = [cell for cell in starts if board[cell] == player]
    seen = set(frontier)
    while frontier:
        cell = frontier.pop()
        row, column = divmod(cell, size)
        if (row if player == 0 else column) == size - 1:
            return True
        for dr, dc in ((0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1)):
            r, c = row + dr, column + dc
            if 0 <= r < size and 0 <= c < size:
                neighbor = r * size + c
                if neighbor not in seen and board[neighbor] == player:
                    seen.add(neighbor)
                    frontier.append(neighbor)
    return False


def replay(opening, actions, size):
    board = [-1] * (size * size)
    for ply, value in enumerate([*opening, *actions]):
        action = int(value)
        if action != value or not 0 <= action < len(board) or board[action] != -1:
            raise ValueError(f"Illegal pre-terminal move at ply {ply}: {action}")
        player = ply % 2
        board[action] = player
        if connected(board, player, size):
            return player, ply + 1
    raise ValueError("Saved Hex game did not reach a connected winning path")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    openings = np.load(args.directory / "openings.npy")
    summary = json.loads((args.directory / "summary.json").read_text())
    paths = sorted(args.directory.glob("*.npz"))
    if args.size < 1 or openings.ndim != 2 or openings.shape[1] != 2 or not len(openings):
        raise ValueError("Expected nonempty two-ply openings and a positive board size")
    if not summary or {path.stem for path in paths} != set(summary):
        raise ValueError("Pairing files do not match the nonempty evaluation summary")
    results = {}
    for path in paths:
        wins = [0, 0]
        lengths = []
        with np.load(path) as data:
            if (data["actions"].ndim != 2 or data["actions"].shape[0] != len(openings)
                    or data["rewards"].shape != (len(openings), 2)
                    or data["finished"].shape != (len(openings),)):
                raise ValueError(f"Game rows do not match the opening slate in {path.name}")
            for index, opening in enumerate(openings):
                winner, length = replay(opening, data["actions"][index], args.size)
                expected = np.array([1., -1.]) * (1 if winner == 0 else -1)
                if not bool(data["finished"][index]) or not np.array_equal(data["rewards"][index], expected):
                    raise ValueError(f"Connectivity/reward mismatch in {path.name}, row {index}")
                wins[winner] += 1
                lengths.append(length)
        reported = summary[path.stem]
        if (wins != [reported["wins"], reported["losses"]]
                or reported["draws"] or reported["unfinished"]):
            raise ValueError(f"Summary count mismatch for {path.name}")
        results[path.stem] = {"verified_games": len(openings), "wins_by_player": wins,
                              "mean_game_plies": float(np.mean(lengths))}
    record = {"timestamp": datetime.now(timezone.utc).isoformat(),
              "directory": str(args.directory), "board_size": args.size,
              "method": "Independent graph-connectivity replay; no simulator or model calls",
              "pairings": results}
    with args.output.open("x") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
