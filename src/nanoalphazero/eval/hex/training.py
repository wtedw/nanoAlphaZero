"""Training adapter and local observability for Hex engine evaluations."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path


SCHEMA_VERSION = 1


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _atomic_write_jsonl(path: Path, records) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    )
    temporary.replace(path)


def _append_jsonl(path: Path, value) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _opening_results(records) -> list[dict]:
    return [
        {
            "opening_action": int(record["opening_action"]),
            "opening_vertex": record["opening_vertex"],
            "result": (
                "W" if record["winner"] == 0 else "L" if record["winner"] == 1 else "?"
            ),
        }
        for record in records
    ]


class HexTrainingEvaluator:
    def __init__(self, period: int, engine_pool, output_dir, board_size: int):
        self.period = int(period)
        self.engine_pool = engine_pool
        self.output_dir = Path(output_dir)
        self.board_size = int(board_size)
        self.output_dir.mkdir(parents=True, exist_ok=False)
        _atomic_write_json(
            self.output_dir / "status.json",
            {
                "schema_version": SCHEMA_VERSION,
                "state": "waiting",
                "board_size": self.board_size,
                "games_total": self.board_size * self.board_size,
                "period": self.period,
                "timestamp": _timestamp(),
            },
        )
        print(f"Hex evaluation data: {self.output_dir}", flush=True)

    def run_if_due(self, cycle, run_mcts, env, config, params, *, train_step=None):
        if not self.period or cycle % self.period:
            return {}
        from nanoalphazero.eval.hex.runtime import (
            print_grid,
            run_match,
            training_metrics,
        )

        cycle = int(cycle)
        common = {
            "schema_version": SCHEMA_VERSION,
            "cycle": cycle,
            "train_step": None if train_step is None else int(train_step),
            "board_size": self.board_size,
            "games_total": self.board_size * self.board_size,
        }
        _atomic_write_json(
            self.output_dir / "status.json",
            {**common, "state": "running", "timestamp": _timestamp()},
        )
        try:
            records, summary = run_match(
                run_mcts,
                env,
                config,
                params,
                self.engine_pool,
                seed=1234 + cycle,
            )
        except BaseException as error:
            _atomic_write_json(
                self.output_dir / "status.json",
                {
                    **common,
                    "state": "failed",
                    "timestamp": _timestamp(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            print(
                f"HEX_EVAL_FAILED cycle={cycle} error={type(error).__name__}",
                flush=True,
            )
            raise

        result = {
            **common,
            **summary,
            "state": "complete",
            "timestamp": _timestamp(),
            "opening_results": _opening_results(records),
        }
        _atomic_write_jsonl(
            self.output_dir / f"games-cycle-{cycle:06d}.jsonl", records
        )
        _atomic_write_json(self.output_dir / "latest.json", result)
        _append_jsonl(self.output_dir / "history.jsonl", result)
        _atomic_write_json(self.output_dir / "status.json", result)
        print_grid(records, int(config["boardsize"]))
        print(
            f"HEX_EVAL_RESULT cycle={cycle} "
            f"score={summary['model_win_rate']:.6f} "
            f"wins={summary['model_wins']} losses={summary['model_losses']} "
            f"unscored={summary['unscored']}",
            flush=True,
        )
        return training_metrics(summary)
