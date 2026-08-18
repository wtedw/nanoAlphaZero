from pathlib import Path


def test_tournament_code_is_chess_scoped_and_legacy_eval_is_absent():
    root = Path("src/nanoalphazero/eval")
    assert (root / "chess" / "tournament.py").is_file()
    assert (root / "chess" / "mctx_player.py").is_file()
    assert not (root / "tournament.py").exists()
    assert not (root / "puzzles.py").exists()
    assert not (root / "schedulers" / "resident_v2.py").exists()


def test_tournament_runtime_does_not_reference_omctx():
    files = Path("src/nanoalphazero/eval/chess").glob("*.py")
    combined = "\n".join(path.read_text() for path in files)
    assert "import omctx" not in combined
    assert "mcts_variant = \"omctx\"" not in combined
