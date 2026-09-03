"""Small adapter between the host training loop and the Hex evaluator."""

from __future__ import annotations


class HexTrainingEvaluator:
    def __init__(self, period: int, engine_pool):
        self.period = int(period)
        self.engine_pool = engine_pool

    def run_if_due(self, cycle, run_mcts, env, config, params):
        if not self.period or cycle % self.period:
            return {}
        from nanoalphazero.eval.hex.runtime import (
            print_grid,
            run_match,
            training_metrics,
        )

        records, summary = run_match(
            run_mcts,
            env,
            config,
            params,
            self.engine_pool,
            seed=1234 + int(cycle),
        )
        print_grid(records, int(config["boardsize"]))
        return training_metrics(summary)

