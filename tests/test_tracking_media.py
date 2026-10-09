"""Tracking accepts arbitrary media without importing game diagnostics."""

import sys
from unittest.mock import Mock

import pytest

from nanoalphazero.tracking import TrainingTracker


@pytest.mark.parametrize("backend", ["wandb", "trackio", None])
def test_explicit_file_publication(tmp_path, monkeypatch, backend):
    wandb = Mock()
    monkeypatch.setitem(sys.modules, "wandb", wandb)
    path = tmp_path / "example.pgn"
    path.write_text("game")
    run = Mock(id="identity") if backend else None
    tracker = TrainingTracker(run, {"trackio": {}} if backend == "trackio" else {}, 3)
    tracker.log_files("games", {"example.pgn": path}, step=3)
    tracker.log_files("games", {}, step=4)
    wandb.Artifact.assert_not_called()
    tracker.log_files("games", {"example.pgn": path}, step=4, kind="evaluation-games",
                      metadata={"games": 1})
    assert path.read_text() == "game"
    if backend == "wandb":
        wandb.Artifact.assert_called_once_with(
            "games-identity", type="evaluation-games", metadata={"cycle": 4, "games": 1})
        wandb.Artifact.return_value.add_file.assert_called_once_with(str(path), name="example.pgn")
        run.log_artifact.assert_called_once_with(
            wandb.Artifact.return_value, aliases=["latest", "cycle-4"])
    else:
        wandb.Artifact.assert_not_called()


def test_disabled_tracker_does_not_convert_figures(monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)
    monkeypatch.setitem(sys.modules, "trackio", None)
    TrainingTracker().log_training({}, step=1, figures={"any/name": object()})
