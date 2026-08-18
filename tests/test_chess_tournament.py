from pathlib import Path

import chess
import chess.engine
import pytest

from nanoalphazero.eval.chess.action import PgxAction
from nanoalphazero.eval.chess.config import load_config
from nanoalphazero.eval import wandb_artifacts
from nanoalphazero.eval.chess.tournament import (
    _adjudicated_result,
    _consume_prefetched,
    _create_first_mover_slots,
    _next_resident_size,
    _resolve_agent_paths,
    select_openings,
    run_tournament,
)


def test_action_round_trip_for_legal_moves_and_promotions():
    boards = [chess.Board()]
    promoted = chess.Board("8/P7/8/8/8/8/7p/4K2k w - - 0 1")
    boards.append(promoted)
    for board in boards:
        for move in board.legal_moves:
            assert PgxAction.decode(PgxAction.encode(move, board), board) == move


def test_first_mover_owns_side_to_move_for_even_and_odd_openings():
    even = chess.Board()
    odd = chess.Board()
    odd.push_uci("e2e4")
    a_first = _create_first_mover_slots([10, 11], [even, odd], "A", "B")
    b_first = _create_first_mover_slots([10, 11], [even, odd], "B", "A")

    assert [(slot["white"], slot["black"]) for slot in a_first] == [
        ("A", "B"),
        ("B", "A"),
    ]
    assert [(slot["white"], slot["black"]) for slot in b_first] == [
        ("B", "A"),
        ("A", "B"),
    ]
    assert [slot["opening_index"] for slot in a_first] == [10, 11]
    assert [slot["opening_index"] for slot in b_first] == [10, 11]


def test_terminal_opening_is_not_sampled():
    playable = chess.Board()
    terminal = chess.Board()
    for move in ("f2f3", "e7e5", "g2g4", "d8h4"):
        terminal.push_uci(move)
    indices, selected = select_openings(
        [terminal, playable], 2, seed=1, require_playable=True
    )
    assert indices == [1]
    assert selected[0].fen() == playable.fen()


def test_resident_size_is_power_of_two_device_aligned():
    assert _next_resident_size(245, 32, 4) == 256
    assert _next_resident_size(30, 32, 4) == 32
    assert _next_resident_size(33, 32, 8) == 64


def test_adjudication_assigns_result_from_board_turn():
    board = chess.Board()
    positive = {"score": chess.engine.PovScore(chess.engine.Cp(1400), chess.WHITE)}
    negative = {"score": chess.engine.PovScore(chess.engine.Cp(-1400), chess.WHITE)}
    assert _adjudicated_result(board, positive, 1300) == "1-0"
    assert _adjudicated_result(board, negative, 1300) == "0-1"
    assert _adjudicated_result(board, positive, 1500) is None


def test_stockfish_paths_resolve_relative_to_config():
    config, _ = load_config("evals/tournament-chess-v4-example/config.toml")

    _resolve_agent_paths(config)

    relative = "../../artifacts/stockfish/16/stockfish"
    expected = str(
        (Path(config["_config_dir"]) / relative).resolve()
    )
    assert config["adjudication"]["path"] == expected
    stockfish = next(
        agent for agent in config["agents"] if agent["kind"] == "stockfish"
    )
    assert stockfish["path"] == expected


def test_prefetched_action_rejects_a_changed_board():
    board = chess.Board()
    slots = [{"board": board, "plies_played": 0}]
    prefetched = {
        "phase": 1,
        "player": "candidate",
        "mapping": (0,),
        "board_keys": ((0, 0, board.fen()),),
        "actions": "held action",
    }
    assert (
        _consume_prefetched(prefetched, 1, "candidate", [0], slots)
        == "held action"
    )

    board.push_uci("e2e4")
    slots[0]["plies_played"] = 1
    with pytest.raises(RuntimeError, match="stale adjudication pipeline"):
        _consume_prefetched(prefetched, 1, "candidate", [0], slots)


def test_missing_openings_fail_before_artifact_download(monkeypatch, tmp_path):
    config, source = load_config("evals/tournament-chess-v4-example/config.toml")
    config["tournament"]["openings"] = str(tmp_path / "missing.pgn")
    monkeypatch.setattr(
        wandb_artifacts,
        "materialize_agent_checkpoints",
        lambda config: pytest.fail("artifact download started before opening preflight"),
    )
    with pytest.raises(FileNotFoundError, match="uv run assets fetch"):
        run_tournament(config, source, output_root=tmp_path, skip_bayeselo=True)
