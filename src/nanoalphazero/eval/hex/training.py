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


class MuZeroHexEvaluator:
    """Read-only MoHex matches; evaluation never changes training RNG or replay."""

    def __init__(self, algorithm, pool, period, output):
        import hashlib
        from nanoalphazero.mcts import make_muzero_policy

        self.algorithm = algorithm
        self.period = period
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.streak = 0
        self.searches = {"search": algorithm.run_mcts_fn,
                         "policy": make_muzero_policy(algorithm.env, algorithm.model)}
        self.evaluators = {
            mode: HexTrainingEvaluator(1, pool, self.output / mode, algorithm.config["boardsize"])
            for mode in self.searches
        }
        manifest = {
            "config": algorithm.config, "period": period,
            "engine": str(pool.executable), "engine_config": pool.config.read_text(),
            "engine_sha256": hashlib.sha256(pool.executable.read_bytes()).hexdigest(),
            "criterion": "All known-winning openings in both modes; zero unscored games",
            "scope": "Opening-suite coverage, not proof of perfect play",
        }
        (self.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    def run(self, cycle, state, *, force=False):
        from nanoalphazero.ckpt import save_muzero_checkpoint, muzero_checkpoint_payload

        if not force and cycle % self.period:
            return {}
        metrics, results = {}, {}
        for mode, search in self.searches.items():
            print(f"MUZERO_MOHEX_BEGIN cycle={cycle} mode={mode}", flush=True)
            scores = self.evaluators[mode].run_if_due(
                cycle, search, self.algorithm.env, self.algorithm.config, state.model_ts.params,
                train_step=int(state.model_ts.n_updates),
            )
            results[mode] = {key.removeprefix("hex_eval/"): value for key, value in scores.items()}
            metrics.update({f"mohex/{mode}/{key}": value for key, value in results[mode].items()})
        success = all(
            row["perfect_opening_total"] > 0
            and row["perfect_opening_wins"] == row["perfect_opening_total"]
            and row["unscored"] == 0
            for row in results.values()
        )
        self.streak = self.streak + 1 if success else 0
        result = {"cycle": cycle, "updates": int(state.model_ts.n_updates),
                  "modes": results, "both_modes_passed": success,
                  "consecutive_passes": self.streak}
        with (self.output / "scores.jsonl").open("a") as stream:
            stream.write(json.dumps(result) + "\n")
        print("MUZERO_MOHEX_RESULT " + json.dumps(result), flush=True)
        # Retain the exact evaluated parameters/optimizer without multi-GB buffers.
        save_muzero_checkpoint(
            self.output / f"cycle-{cycle:06d}.safetensors",
            muzero_checkpoint_payload(state, cycle, False),
            self.algorithm.config,
        )
        return {**metrics, "mohex/both_modes_passed": int(success),
                "mohex/consecutive_passes": self.streak}
