from pathlib import Path

import pytest

from nanoalphazero.eval.chess import bayeselo
from nanoalphazero.eval.chess.bayeselo import find_binary, parse_ratings


def test_parse_ratings_table():
    output = """
Rank Name             Elo    +    - games score oppo. draws
   1 alphazero         48   36   36   200   57%   -21   21%
   2 9M              -321   44   49   200   11%   128   11%
"""
    ratings = parse_ratings(output)
    assert ratings["alphazero"] == {
        "rank": 1,
        "elo": 48,
        "plus": 36,
        "minus": 36,
        "games": 200,
        "score_pct": 57,
        "opponent_elo": -21,
        "draws_pct": 21,
    }
    assert ratings["9M"]["elo"] == -321


def test_find_binary_accepts_bayeselo_on_path(monkeypatch, tmp_path):
    path_executable = tmp_path / "path" / "bayeselo"
    path_executable.parent.mkdir()
    path_executable.touch()
    managed_executable = (
        tmp_path / "artifacts" / "bayeselo" / "BayesElo" / "bayeselo"
    )
    managed_executable.parent.mkdir(parents=True)
    managed_executable.touch()
    monkeypatch.delenv("BAYESELO_PATH", raising=False)
    monkeypatch.setattr(
        bayeselo.shutil, "which", lambda name: str(path_executable)
    )
    monkeypatch.chdir(tmp_path)

    assert find_binary() == path_executable.resolve()


def test_find_binary_rejects_nano_console_wrapper(monkeypatch, tmp_path):
    wrapper = tmp_path / "bayeselo"
    wrapper.write_text(
        "#!/usr/bin/python\nfrom nanoalphazero.cli import bayeselo_main\n"
    )
    monkeypatch.delenv("BAYESELO_PATH", raising=False)
    monkeypatch.setattr(bayeselo.shutil, "which", lambda name: str(wrapper))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.chdir(empty)

    with pytest.raises(FileNotFoundError, match="scoring wrapper"):
        find_binary()


def test_find_binary_reports_install_options(monkeypatch, tmp_path):
    monkeypatch.delenv("BAYESELO_PATH", raising=False)
    monkeypatch.setattr(bayeselo.shutil, "which", lambda name: None)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    monkeypatch.chdir(tmp_path)

    with pytest.raises(FileNotFoundError, match="assets fetch bayeselo"):
        find_binary()


def test_find_binary_accepts_managed_asset(monkeypatch, tmp_path):
    executable = tmp_path / "artifacts" / "bayeselo" / "BayesElo" / "bayeselo"
    executable.parent.mkdir(parents=True)
    executable.touch()
    monkeypatch.delenv("BAYESELO_PATH", raising=False)
    monkeypatch.setattr(bayeselo.shutil, "which", lambda name: None)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    monkeypatch.chdir(tmp_path)

    assert find_binary() == executable.resolve()
