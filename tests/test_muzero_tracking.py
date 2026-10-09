"""Optional tracking in the root MuZero host loop."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nanoalphazero.cli import parse_muzero_args


@pytest.fixture
def assembly():
    path = Path(__file__).resolve().parents[1] / "muzero.py"
    spec = importlib.util.spec_from_file_location("muzero_tracking", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_trackio_flags():
    assert parse_muzero_args([]).enable_trackio is None
    for flag in ("--enabled-trackio", "--enable-trackio"):
        assert parse_muzero_args([flag]).enable_trackio is True
    assert parse_muzero_args(["--no-enable-trackio"]).enable_trackio is False
    assert parse_muzero_args(["--trackio-run", "existing-run"]).trackio_run == "existing-run"


def test_disabled_tracker_needs_no_backend(monkeypatch):
    from nanoalphazero.tracking import make_tracker

    monkeypatch.setitem(sys.modules, "trackio", None)
    monkeypatch.setitem(sys.modules, "wandb", None)
    tracker = make_tracker({})
    tracker.log_training({"loss": 1.0}, step=1, tables={"unconverted": object()})
    tracker.flush()
    tracker.log({"eval/score": 1.0}, step=1)
    tracker.finish()
    assert tracker.metadata == {}


def test_real_trackio_resume_and_duplicate_cycles(tmp_path):
    """Use an isolated local database, never the user's actual Trackio project."""
    import os
    import subprocess

    subprocess.run([sys.executable, "-c", r'''
import json
import sqlite3
from pathlib import Path
import jax.numpy as jnp
from nanoalphazero.ckpt import compressed
from nanoalphazero.tracking import make_tracker
from trackio.sqlite_storage import SQLiteStorage

config = {"enable_trackio": True}
tracker = make_tracker(config)
identity = tracker.metadata["trackio"]
tracker.log({"value": 1.0}, step=1)
tracker.log({"value": 2.0}, step=2)
tracker.finish()
path = Path("checkpoint")
compressed.save(path, {"counter": jnp.int32(1)}, config, metadata=tracker.metadata)
resumed = make_tracker(config, resume=path)
assert resumed.metadata == tracker.metadata
assert resumed.last_step == 2
resumed.log({"value": 999.0}, step=2)  # Recomputed after an older checkpoint.
resumed.log({"value": 3.0}, step=3)
resumed.finish()
explicit = make_tracker(config, trackio_run=identity["name"])
assert explicit.run.id == identity["id"]
assert explicit.last_step == 3
explicit.finish()
with sqlite3.connect(SQLiteStorage.get_project_db_path(identity["project"])) as db:
    rows = db.execute("SELECT step FROM metrics WHERE run_id=? ORDER BY step", (identity["id"],)).fetchall()
assert rows == [(1,), (2,), (3,)], rows
try:
    make_tracker(config, trackio_run="does-not-exist")
except ValueError:
    pass
else:
    raise AssertionError("Missing run must not silently create a new identity")
'''], check=True, cwd=tmp_path, env={**os.environ, "TRACKIO_DIR": str(tmp_path / "trackio"),
                                   "JAX_PLATFORMS": "cpu"}, capture_output=True, text=True)


@pytest.mark.parametrize("trackio_enabled,wandb_enabled,backend", [
    (False, False, None), (True, False, "trackio"),
    (False, True, "wandb"), (True, True, "trackio"),
])
@pytest.mark.parametrize("fails", [False, True])
def test_host_tracking(assembly, monkeypatch, trackio_enabled, wandb_enabled, backend, fails):
    trackers = {name: Mock() for name in ("trackio", "wandb")}
    for name, tracker in trackers.items():
        # Disabled backends must never be imported.
        monkeypatch.setitem(sys.modules, name, tracker if name == backend else None)
    config = dict(
        enable_trackio=trackio_enabled, enable_wandb=wandb_enabled,
        replay_buffer_warmup_steps=0, cycle_n_selfplay=1, num_iters=2,
        diagnostic_period=0, eval_period=0,
    )
    state = SimpleNamespace(model_ts=SimpleNamespace(step=0, params={}))
    metrics = {"selfplay_buffer/error_count": 0, "loss": 0.5}
    compiled = Mock(return_value=(state, metrics))
    compiled.memory_analysis.return_value = None
    if fails:
        compiled.side_effect = RuntimeError("cycle failed")
    run_fn = Mock()
    run_fn.lower.return_value.compile.return_value = compiled
    algorithm = SimpleNamespace(config=config, runner_state=state, run_fn=run_fn,
                                env=None, run_mcts_fn=None, data_sharding=None)

    if fails:
        with pytest.raises(RuntimeError, match="cycle failed"):
            assembly.run_muzero(algorithm)
    else:
        assert assembly.run_muzero(algorithm) is state

    if backend is not None:
        tracker = trackers[backend]
        tracker.init.assert_called_once_with(project="nanoAlphaZero-muzero", config=config)
        run = tracker.init.return_value
        run.finish.assert_called_once_with()
        assert run.log.call_count == (0 if fails else (4 if backend == "wandb" else 2))
        if not fails:
            assert [call.kwargs["step"] for call in run.log.call_args_list] == (
                [1, 1, 2, 2] if backend == "wandb" else [1, 2]
            )
            assert run.log.call_args_list[0].args[0]["loss"] == 0.5
            if backend == "wandb":
                assert [call.kwargs["commit"] for call in run.log.call_args_list] == [False, True, False, True]


def test_missing_trackio_has_install_hint(assembly, monkeypatch):
    monkeypatch.setitem(sys.modules, "trackio", None)
    compiled = Mock()
    compiled.memory_analysis.return_value = None
    run_fn = Mock()
    run_fn.lower.return_value.compile.return_value = compiled
    algorithm = SimpleNamespace(config={"enable_trackio": True}, runner_state=None, run_fn=run_fn)
    with pytest.raises(RuntimeError, match="uv run --extra trackio"):
        assembly.run_muzero(algorithm)
    compiled.assert_not_called()


@pytest.mark.parametrize("flush_times_out", [False, True])
def test_training_log_wait_precedes_checkpoint_even_on_timeout(assembly, monkeypatch, tmp_path, capsys, flush_times_out):
    from nanoalphazero import ckpt, tracking
    from nanoalphazero.eval import acpl
    from trackio.sqlite_storage import SQLiteStorage

    events = []
    backend = Mock()
    tracker = tracking.TrainingTracker(backend, {"trackio": {
        "project": "test", "name": "test", "id": "test",
    }})
    backend.log.side_effect = lambda *args, **kwargs: events.append("log")
    monkeypatch.setattr(tracking, "_persisted_step",
                        lambda *a, **k: None if flush_times_out else 1)
    original_flush = tracker.flush

    def flush():
        events.append("flush")
        original_flush(timeout=0.02)

    tracker.flush = flush
    monkeypatch.setattr(tracking, "make_tracker", lambda *a, **k: tracker)

    def save(*args, **kwargs):
        assert events == ["log", "flush"]
        events.append("save")
        return {"seconds": 0.1}

    def evaluate(*args):
        assert events == ["log", "flush", "save"]
        raise RuntimeError("evaluation interrupted")

    monkeypatch.setattr(ckpt, "save_checkpoint", save)
    monkeypatch.setattr(acpl, "make_acpl", lambda *a, **k: (evaluate, None))
    state = SimpleNamespace(model_ts=SimpleNamespace(step=0, params={}))
    compiled = Mock(return_value=(state, {"selfplay_buffer/error_count": 0, "loss": 0.5}))
    compiled.memory_analysis.return_value = None
    run_fn = Mock()
    run_fn.lower.return_value.compile.return_value = compiled
    config = dict(replay_buffer_warmup_steps=0, cycle_n_selfplay=1, num_iters=1,
                  diagnostic_period=0, eval_period=0)
    algorithm = SimpleNamespace(config=config, runner_state=state, run_fn=run_fn,
                                env=None, run_mcts_fn=None, data_sharding=None)
    with pytest.raises(RuntimeError, match="evaluation interrupted"):
        assembly.run_muzero(algorithm, checkpoint_path=tmp_path / "state.ckpt")
    assert events == ["log", "flush", "save"]
    backend.finish.assert_called_once()
    if flush_times_out:
        assert "continuing with checkpoint save" in capsys.readouterr().err


def test_trackio_barrier_survives_sigkill(tmp_path):
    import json
    import os
    import signal
    import sqlite3
    import subprocess

    result = subprocess.run([sys.executable, "-c", r'''
import json, os, signal
from pathlib import Path
from nanoalphazero.tracking import make_tracker
tracker = make_tracker({"enable_trackio": True})
tracker.log({"loss": 0.25}, step=200, commit=False)
tracker.flush()
Path("identity.json").write_text(json.dumps(tracker.metadata["trackio"]))
os.kill(os.getpid(), signal.SIGKILL)
'''], cwd=tmp_path, env={**os.environ, "TRACKIO_DIR": str(tmp_path / "trackio"),
                        "JAX_PLATFORMS": "cpu"}, capture_output=True, text=True)
    assert result.returncode == -signal.SIGKILL, result.stderr
    identity = json.loads((tmp_path / "identity.json").read_text())
    with sqlite3.connect(tmp_path / "trackio" / f"{identity['project']}.db") as db:
        rows = db.execute("SELECT step FROM metrics WHERE run_id=?", (identity["id"],)).fetchall()
    assert rows == [(200,)]


@pytest.mark.parametrize("read_error", [False, True])
def test_trackio_wait_is_best_effort(monkeypatch, capsys, read_error):
    import nanoalphazero.tracking as tracking
    from nanoalphazero.tracking import TrainingTracker
    from trackio.sqlite_storage import SQLiteStorage

    def read(*args, **kwargs):
        if read_error:
            raise OSError("database unavailable")
        return None
    monkeypatch.setattr(tracking, "_persisted_step", read)
    tracker = TrainingTracker(None, {"trackio": {"project": "test", "name": "test", "id": "test"}}, 200)
    tracker.flush(timeout=0.01)
    assert "continuing with checkpoint save" in capsys.readouterr().err


def test_same_cycle_extra_metrics_are_logged():
    from nanoalphazero.tracking import TrainingTracker

    run = Mock()
    tracker = TrainingTracker(run, {"trackio": {}}, last_step=199)
    tracker.log({"loss": 0.25}, step=199)
    tracker.log({"loss": 0.25}, step=200, commit=False)
    tracker.log({"checkpoint/seconds": 1.0}, step=200)
    assert [call.args[0] for call in run.log.call_args_list] == [
        {"loss": 0.25}, {"checkpoint/seconds": 1.0},
    ]


@pytest.mark.parametrize("resume_step", [-1, 1])
def test_host_acpl_count_and_viewer_are_distinct(assembly, monkeypatch, tmp_path, resume_step):
    import nanoalphazero.tracking as tracking
    from nanoalphazero.eval import acpl
    from nanoalphazero.diagnostics import chess_games

    path = tmp_path / "games.pgn"
    path.write_text("1. e4 *")
    result = acpl.AcplResult({"acpl/games": 1}, {"games.pgn": path})
    monkeypatch.setattr(acpl, "make_acpl", lambda *a, **k: (lambda *a: (None, result), None))
    render = Mock(return_value="<html>games</html>")
    monkeypatch.setattr(chess_games, "games_html", render)
    api, run = Mock(), Mock()
    monkeypatch.setitem(sys.modules, "trackio", api)
    tracker = tracking.TrainingTracker(run, {"trackio": {}}, last_step=resume_step)
    monkeypatch.setattr(tracking, "make_tracker", lambda *a, **k: tracker)
    state = SimpleNamespace(model_ts=SimpleNamespace(step=0, params={}))
    compiled = Mock(return_value=(state, {"selfplay_buffer/error_count": 0, "loss": 0.5}))
    compiled.memory_analysis.return_value = None
    run_fn = Mock()
    run_fn.lower.return_value.compile.return_value = compiled
    config = dict(replay_buffer_warmup_steps=0, cycle_n_selfplay=1, num_iters=1,
                  diagnostic_period=0, eval_period=0)
    algorithm = SimpleNamespace(config=config, runner_state=state, run_fn=run_fn,
                                env=None, run_mcts_fn=None, data_sharding=None)
    assembly.run_muzero(algorithm)
    if resume_step < 1:
        logs = run.log.call_args.args[0]
        assert logs["acpl/games"] == 1
        assert logs["acpl/games_viewer"] is api.Html.return_value
        render.assert_called_once()
    else:
        render.assert_not_called()
        run.log.assert_not_called()
        api.Artifact.assert_not_called()


def test_cli_rejects_resume_config_before_assembly(assembly, monkeypatch, tmp_path):
    import jax.numpy as jnp
    from nanoalphazero import ckpt

    path = tmp_path / "checkpoint.safetensors"
    ckpt.save_muzero_checkpoint(path, {"cycle": jnp.int32(1)}, {"conv_width": 99})
    build = Mock(side_effect=AssertionError("must not allocate model or buffers"))
    monkeypatch.setattr(assembly, "make_muzero", build)
    with pytest.raises(ValueError, match="identical resolved configuration"):
        assembly.main(["--env", "ttt", "--platform", "cpu", "--ckpt.resume", str(path),
                       "--ckpt.mode", "partial"])
    build.assert_not_called()
