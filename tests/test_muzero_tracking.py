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
        diagnostic_period=0,
    )
    state = SimpleNamespace(model_ts=SimpleNamespace(step=0))
    metrics = {"selfplay_buffer/error_count": 0, "loss": 0.5}
    compiled = Mock(return_value=(state, metrics))
    compiled.memory_analysis.return_value = None
    if fails:
        compiled.side_effect = RuntimeError("cycle failed")
    run_fn = Mock()
    run_fn.lower.return_value.compile.return_value = compiled
    algorithm = SimpleNamespace(config=config, runner_state=state, run_fn=run_fn)

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
        assert run.log.call_count == (0 if fails else 2)
        if not fails:
            assert [call.kwargs["step"] for call in run.log.call_args_list] == [1, 2]
            assert run.log.call_args.args[0]["loss"] == 0.5


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
