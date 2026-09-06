import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "muzero_progression", Path(__file__).parents[1] / "evals/muzero/run_opening_progression.py")
progression = importlib.util.module_from_spec(spec)
spec.loader.exec_module(progression)


def test_progression_requires_clean_completed_four_device_run(tmp_path):
    manifest = dict(finished="now", completed_updates=4, devices=["tpu"] * 4,
                    sharding=dict(training_local_shapes=[[2, 4, 4, 4]] * 4),
                    stop_reason="configured_cycles_complete")
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "config.json").write_text(json.dumps(dict(save_checkpoints=False)))
    row = {"cycle": 2, "runner_state/n_updates": 4}
    (tmp_path / "metrics.jsonl").write_text(json.dumps(row))
    assert progression.check_completed(tmp_path)["updates"] == 4
    with pytest.raises(RuntimeError, match="without repeated"):
        progression.check_completed(tmp_path, True)
    manifest["stop_reason"] = "opening_coverage_reached"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert progression.check_completed(tmp_path, True)["state"] == "complete"
    row["anomalies/selfplay_buffer-consume/n_dupes"] = 1
    (tmp_path / "metrics.jsonl").write_text(json.dumps(row))
    with pytest.raises(RuntimeError, match="Data/numerical"):
        progression.check_completed(tmp_path)
