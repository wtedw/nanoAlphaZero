import chess
import numpy as np
import pytest

from nanoalphazero.eval.chess import searchless as searchless_9m


def _board(moves: tuple[str, ...] = ()) -> chess.Board:
    board = chess.Board()
    for uci in moves:
        board.push_uci(uci)
    return board


def _snapshot(board: chess.Board) -> tuple[str, tuple[chess.Move, ...], int]:
    return board.fen(), tuple(board.move_stack), len(board._stack)


CORPUS = (
    (),
    ("g1f3", "g8f6"),
    ("g1f3", "g8f6", "f3g1", "f6g8"),
    (
        "g1f3",
        "g8f6",
        "f3g1",
        "f6g8",
        "g1f3",
        "g8f6",
        "f3g1",
        "f6g8",
    ),
    ("g1f3", "g8f6", "f3g1", "f6g8", "e2e4"),
    ("e2e4", "a7a6", "e4e5", "d7d5"),
    ("e2e4", "e7e5", "g1f3", "g8f6", "f1e2", "f8e7", "e1g1"),
    ("e2e4", "d7d5", "e4d5", "d8d5"),
)


def test_released_model_registry_matches_deepmind_architectures():
    assert searchless_9m.SEARCHLESS_MODELS == {
        "9M": searchless_9m.SearchlessModelSpec(8, 256, 8, 1024),
        "136M": searchless_9m.SearchlessModelSpec(8, 1024, 8, 256),
        "270M": searchless_9m.SearchlessModelSpec(16, 1024, 8, 128),
    }


@pytest.mark.parametrize("history", CORPUS)
def test_repetition_v2_matches_v1_for_every_legal_candidate(history):
    board = _board(history)
    moves = list(board.legal_moves)
    before = _snapshot(board)

    v1 = searchless_9m.repetition_draws_v1(board, moves)
    v2 = searchless_9m.repetition_draws_v2(board, moves)

    np.testing.assert_array_equal(v2, v1)
    assert _snapshot(board) == before


@pytest.mark.parametrize(
    "fen",
    (
        "8/P7/8/8/8/8/7p/4K2k w - - 0 1",
        "7k/5Q2/7K/8/8/8/8/8 b - - 0 1",
    ),
)
def test_repetition_v2_matches_v1_for_special_and_terminal_positions(fen):
    board = chess.Board(fen)
    moves = list(board.legal_moves)

    np.testing.assert_array_equal(
        searchless_9m.repetition_draws_v2(board, moves),
        searchless_9m.repetition_draws_v1(board, moves),
    )


def test_repetition_v2_skips_all_authoritative_checks_without_repeated_history():
    board = chess.Board()
    moves = list(board.legal_moves)
    stats = searchless_9m.RepetitionCheckStats()

    draws = searchless_9m.repetition_draws_v2(board, moves, stats)

    assert not draws.any()
    assert stats.candidate_checks == len(moves)
    assert stats.prefilter_skips == len(moves)
    assert stats.authoritative_checks == 0


def test_repetition_v2_uses_authoritative_check_for_possible_claims():
    board = _board(
        (
            "g1f3",
            "g8f6",
            "f3g1",
            "f6g8",
            "g1f3",
            "g8f6",
            "f3g1",
            "f6g8",
        )
    )
    moves = list(board.legal_moves)
    stats = searchless_9m.RepetitionCheckStats()

    v2 = searchless_9m.repetition_draws_v2(board, moves, stats)
    v1 = searchless_9m.repetition_draws_v1(board, moves)

    np.testing.assert_array_equal(v2, v1)
    assert stats.authoritative_checks > 0
    assert stats.draw_overrides > 0
    assert stats.authoritative_checks + stats.prefilter_skips == len(moves)


def test_repetition_v2_restores_board_when_authoritative_check_raises(monkeypatch):
    board = _board(
        (
            "g1f3",
            "g8f6",
            "f3g1",
            "f6g8",
            "g1f3",
            "g8f6",
            "f3g1",
            "f6g8",
        )
    )
    before = _snapshot(board)

    def fail():
        raise RuntimeError("oracle failure")

    monkeypatch.setattr(board, "can_claim_threefold_repetition", fail)

    with pytest.raises(RuntimeError, match="oracle failure"):
        searchless_9m.repetition_draws_v2(board, list(board.legal_moves))
    assert _snapshot(board) == before


def test_select_moves_v2_matches_v1():
    boards = (_board(), _board(CORPUS[3]))
    move_groups = [list(board.legal_moves) for board in boards]
    row_count = sum(map(len, move_groups))
    rng = np.random.default_rng(7)
    log_probs = rng.normal(size=(row_count, 128))

    assert searchless_9m.select_moves_v2(boards, move_groups, log_probs) == (
        searchless_9m.select_moves_v1(boards, move_groups, log_probs)
    )
