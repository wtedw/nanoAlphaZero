"""Tracking accepts arbitrary media without importing game diagnostics."""

import sys
from unittest.mock import Mock

import pytest

from nanoalphazero.tracking import TrainingTracker


@pytest.mark.parametrize("backend", ["wandb", "trackio", None])
def test_explicit_file_publication(tmp_path, monkeypatch, backend):
    api = Mock()
    monkeypatch.setitem(sys.modules, backend or "wandb", api)
    monkeypatch.setitem(sys.modules, "wandb" if backend == "trackio" else "trackio", None)
    path = tmp_path / "example.pgn"
    path.write_text("game")
    run = Mock(id="identity") if backend else None
    tracker = TrainingTracker(run, {"trackio": {}} if backend == "trackio" else {}, 3)
    tracker.log_files("games", {"example.pgn": path}, step=3)
    tracker.log_files("games", {}, step=4)
    api.Artifact.assert_not_called()
    tracker.log_files("games", {"example.pgn": path}, step=4, kind="evaluation-games",
                      metadata={"games": 1})
    assert path.read_text() == "game"
    if backend is not None:
        api.Artifact.assert_called_once_with(
            "games-identity", type="evaluation-games", metadata={"cycle": 4, "games": 1})
        api.Artifact.return_value.add_file.assert_called_once_with(str(path), name="example.pgn")
        run.log_artifact.assert_called_once_with(
            api.Artifact.return_value, aliases=["latest", "cycle-4"])
    else:
        api.Artifact.assert_not_called()


@pytest.mark.parametrize("backend", ["wandb", "trackio"])
def test_html_is_logged_with_evaluation_metrics(backend, monkeypatch):
    api, run = Mock(), Mock()
    monkeypatch.setitem(sys.modules, backend, api)
    tracker = TrainingTracker(run, {"trackio": {}} if backend == "trackio" else {}, 3)
    tracker.log({}, step=3, html={"games": "old"})
    api.Html.assert_not_called()
    tracker.log({"score": 1}, step=4, html={"games": "<html>viewer</html>"})
    api.Html.assert_called_once_with("<html>viewer</html>", inject=False)
    assert run.log.call_args.args[0] == {"score": 1, "games": api.Html.return_value}


def test_disabled_tracker_does_not_convert_figures(monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)
    monkeypatch.setitem(sys.modules, "trackio", None)
    TrainingTracker().log_training({}, step=1, figures={"any/name": object()})


def test_real_trackio_file_versions_and_html(tmp_path):
    import os
    import subprocess
    import sys

    pytest.importorskip("trackio")
    subprocess.run([sys.executable, "-c", r'''
from pathlib import Path
import os
from nanoalphazero.tracking import make_tracker
from nanoalphazero.diagnostics.chess.games import games_html
from trackio.sqlite_storage import SQLiteStorage

root = Path(os.environ["TRACKIO_DIR"]).parent
path = root / "games.pgn"
original = b'[Event "test"]\r\n\r\n1. e4 e5 *\r\n'
path.write_bytes(original)
tracker = make_tracker({"enable_trackio": True})
tracker.log_files("games", {"games.pgn": path}, step=1)
path.write_bytes(b'1. d4 d5 *\n')
tracker.log_files("games", {"games.pgn": path}, step=2)
tracker.log({"acpl/games": 1}, step=2,
            html={"acpl/games_viewer": games_html(original.decode())})
for cycle, expected in [(1, original), (2, path.read_bytes())]:
    artifact = tracker.run.use_artifact(f"games-{tracker.run.id}:cycle-{cycle}")
    restored = Path(artifact.download(root / f"download-{cycle}"))
    assert (restored / "games.pgn").read_bytes() == expected
identity = tracker.metadata["trackio"]
tracker.finish()
rows = SQLiteStorage.get_logs(identity["project"], identity["name"], run_id=identity["id"])
assert len(rows) == 1 and rows[0]["step"] == 2
assert rows[0]["acpl/games"] == 1
assert rows[0]["acpl/games_viewer"]["_type"] == "trackio.html"
'''], check=True, env={**os.environ, "TRACKIO_DIR": str(tmp_path / "trackio")})
