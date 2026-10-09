"""Resume safety, bounded persistence, and session identity regressions."""

import sqlite3
import sys
import time
from unittest.mock import Mock

import jax.numpy as jnp
import pytest

from nanoalphazero import ckpt
from nanoalphazero.tracking import TrainingTracker, make_tracker


@pytest.mark.parametrize("run_name", [None, "existing"])
def test_legacy_metadata_allows_trackio_initialization(tmp_path, monkeypatch, run_name):
    path = tmp_path / "legacy.safetensors"
    ckpt.save_muzero_checkpoint(path, {"cycle": jnp.int32(7)}, {})
    assert ckpt.checkpoint_metadata(path) == {}
    from trackio.sqlite_storage import SQLiteStorage
    api = Mock()
    monkeypatch.setitem(sys.modules, "trackio", api)
    monkeypatch.setattr(SQLiteStorage, "get_max_step_for_run", lambda *a, **k: 7)
    tracker = make_tracker({"enable_trackio": True}, resume=path, trackio_run=run_name)
    assert api.init.call_args.kwargs.get("resume") == ("must" if run_name else None)
    assert tracker.resume_step == (7 if run_name else -1)


def test_bad_config_rejected_before_deleting_initial_state(tmp_path):
    path = tmp_path / "state.ckpt"
    initial = jnp.arange(8)
    ckpt.save_checkpoint(path, state=initial, cycle=1, config={"conv_width": 2})
    assert ckpt.inspect_checkpoint(path)["config"]["conv_width"] == 2
    with pytest.raises(ValueError, match="identical resolved configuration"):
        ckpt.restore_checkpoint(path, state=initial, config={"conv_width": 3})
    assert not initial.is_deleted()
    assert initial.tolist() == list(range(8))


def test_database_lock_respects_remaining_wait(tmp_path, monkeypatch, capsys):
    from trackio.sqlite_storage import SQLiteStorage
    path = tmp_path / "locked.db"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE metrics (run_id TEXT, step INTEGER)")
    connection.commit()
    connection.execute("BEGIN EXCLUSIVE")
    monkeypatch.setattr(SQLiteStorage, "get_project_db_path", lambda _: path)
    tracker = TrainingTracker(Mock(), {"trackio": {"project": "test", "id": "run"}}, 1)
    try:
        start = time.monotonic()
        tracker.flush(timeout=0.05)
        assert time.monotonic() - start < 0.3
        assert "continuing with checkpoint save" in capsys.readouterr().err
    finally:
        connection.close()


def test_persistence_import_failure_does_not_prevent_save(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "trackio.sqlite_storage", None)
    tracker = TrainingTracker(Mock(), {"trackio": {"project": "test", "id": "run"}}, 1)
    tracker.flush(timeout=0.1)
    assert "continuing with checkpoint save" in capsys.readouterr().err


def test_should_log_uses_resume_cutoff_not_last_logged_step():
    tracker = TrainingTracker(Mock(), last_step=5)
    assert not tracker.should_log(5)
    tracker.log({"loss": 1}, step=6)
    assert tracker.should_log(6)
    assert not TrainingTracker().should_log(6)


def test_ladder_resumes_have_separate_rating_series(monkeypatch):
    from nanoalphazero.eval import ladder
    monkeypatch.setattr(ladder, "all_opening_actions", lambda *a, **k: [0])
    monkeypatch.setattr(ladder, "evaluate_vs", lambda *a, **k: (0.6, 25.))
    series = []
    for _ in range(2):
        evaluate, state = ladder.make_ladder({"eval_period": 1}, env=None, policy_fn=None,
                                              separate_sessions=True)
        _, metrics = evaluate(state, 1, {})
        assert metrics["eval/score_vs_random"] == 0.6
        assert "eval/ladder_elo" not in metrics
        series.append({key for key in metrics if key.startswith("eval/ladder/")})
    assert len(series[0]) == len(series[1]) == 4
    assert series[0].isdisjoint(series[1])


@pytest.mark.parametrize("full,compressed", [(False, False), (False, True), (True, True)])
def test_model_only_load_skips_buffers_and_optimizer(tmp_path, full, compressed):
    import json
    from typing import NamedTuple
    import jax
    import numpy as np
    from nanoalphazero.config import get_muzero_config
    from nanoalphazero.core import make_env
    from nanoalphazero.model import make_muzero_model

    class ModelState(NamedTuple):
        params: dict
        optimizer: object

    class State(NamedTuple):
        model_ts: ModelState
        buffers: object
        rng: object

    config = get_muzero_config("ttt", conv_width=4, conv_depth=1, enable_sharding=False)
    env = make_env(config)
    config.update(game_obs_shape=list(env.obs_shape), game_num_actions=env.num_actions)
    _, variables = make_muzero_model(config, jax.random.PRNGKey(9))
    state = State(ModelState(variables["params"], jnp.ones(16)), jnp.ones(32), jax.random.PRNGKey(0))
    path = tmp_path / ("model.ckpt" if compressed else "model.safetensors")
    ckpt.save_muzero_checkpoint(path, ckpt.muzero_checkpoint_payload(state, 3, full), config,
                               compression="zstd" if compressed else "none")
    if compressed:
        root = path / json.loads((path / "LATEST").read_text())["generation"]
        manifest = json.loads((root / "manifest.json").read_text())
        for field in manifest["fields"]:
            if 'optimizer' in field['path'] or 'buffers' in field['path']:
                for shard in field['shards']:
                    for chunk in shard['chunks']:
                        (root / chunk['file']).unlink()  # model loading must never read these
    _, actual, saved = ckpt.load_model(path)
    assert saved == config
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(variables["params"]), strict=True):
        np.testing.assert_array_equal(a, b)
