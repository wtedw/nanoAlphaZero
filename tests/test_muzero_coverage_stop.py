"""Exercise the actual host loop after a simulated coverage decision."""

import json
import sys

from nanoalphazero.research.muzero.cli import main
from nanoalphazero.research.muzero.convergence import OpeningCoverageGate


def test_coverage_exit_runs_final_evaluation_and_writes_no_checkpoint(tmp_path, monkeypatch):
    config = tmp_path / "config.toml"
    config.write_text('''env = "ttt"
defaults = "pilot_v1"
platform = "cpu"
devices = 4
width = 8
depth = 1
selfplay_batch_size = 4
train_batch_size = 4
heldout_batch_size = 4
replay_batches = 2
cycles = 2
updates_per_cycle = 1
eval_period = 2
save_checkpoints = false
''')
    # Predicate correctness is tested separately with real evaluator-shaped
    # metrics. Here force success to exercise control flow without an engine.
    def achieved(self, metrics):
        self.streak = 3
        return True
    monkeypatch.setattr(OpeningCoverageGate, "observe", achieved)
    output = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["muzero-train", str(config), "--output", str(output), "--no-save"])
    main()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["stop_reason"] == "opening_coverage_reached"
    assert manifest["completed_cycles"] == 1 and manifest["completed_updates"] == 1
    assert manifest["opening_coverage_streak"] == 3
    assert (output / "eval-000001/summary.json").exists()
    assert (output / "heldout-cycle-000001.npz").exists()
    assert not list(output.glob("*.safetensors"))
