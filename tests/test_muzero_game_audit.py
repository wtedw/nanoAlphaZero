"""Independent saved-game checker: both goals and pre-terminal legality."""

from pathlib import Path
import json
import runpy
import sys

import numpy as np
import pytest


replay = runpy.run_path(str(Path(__file__).parents[1] / "evals/muzero/audit_hex_games.py"))["replay"]
main = runpy.run_path(str(Path(__file__).parents[1] / "evals/muzero/audit_hex_games.py"))["main"]


def test_connectivity_checker_uses_both_correct_goals_and_ignores_absorbing_tail():
    assert replay([], [0, 1, 3, 2, 6, 0, 0], 3) == (0, 5)
    assert replay([3, 0], [4, 1, 8, 2], 3) == (1, 6)


@pytest.mark.parametrize("actions", [[0, 0], [0.5, 1]])
def test_connectivity_checker_rejects_invalid_moves_before_termination(actions):
    with pytest.raises(ValueError, match="Illegal pre-terminal"):
        replay([], actions, 3)


@pytest.mark.parametrize("defect", [None, "missing_pairing", "extra_game"])
def test_file_audit_requires_complete_and_aligned_exports(tmp_path, monkeypatch, defect):
    np.save(tmp_path / "openings.npy", np.array([[0, 1]]))
    summary = {"policy-vs-random": dict(wins=1, losses=0, draws=0, unfinished=0)}
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    if defect != "missing_pairing":
        rows = 2 if defect == "extra_game" else 1
        np.savez(tmp_path / "policy-vs-random.npz", actions=np.tile([3, 2, 6], (rows, 1)),
                 rewards=np.tile([1., -1.], (rows, 1)), finished=np.ones(rows, bool))
    monkeypatch.setattr(sys, "argv", ["audit_hex_games", str(tmp_path), "--size", "3",
                                     "--output", str(tmp_path / "audit.json")])
    if defect is None:
        main()
        assert json.loads((tmp_path / "audit.json").read_text())["pairings"]["policy-vs-random"]["verified_games"] == 1
    else:
        with pytest.raises(ValueError, match="Pairing files|Game rows"):
            main()
        assert not (tmp_path / "audit.json").exists()
