"""DeepMind Searchless Chess action vocabulary and move ordering."""

from __future__ import annotations

import chess


def _compute_actions() -> dict[str, int]:
    all_moves: list[str] = []
    board = chess.BaseBoard.empty()
    for square in range(64):
        next_squares = []
        board.set_piece_at(square, chess.Piece.from_symbol("Q"))
        next_squares += board.attacks(square)
        board.set_piece_at(square, chess.Piece.from_symbol("N"))
        next_squares += board.attacks(square)
        board.remove_piece_at(square)
        for next_square in next_squares:
            all_moves.append(chess.square_name(square) + chess.square_name(next_square))
    files = list("abcdefgh")
    for rank, next_rank in [("2", "1"), ("7", "8")]:
        for index, file in enumerate(files):
            destinations = [file]
            if file > "a":
                destinations.append(files[index - 1])
            if file < "h":
                destinations.append(files[index + 1])
            for destination in destinations:
                base = f"{file}{rank}{destination}{next_rank}"
                all_moves.extend(base + piece for piece in "qrbn")
    result = {move: action for action, move in enumerate(all_moves)}
    if len(result) != 1968:
        raise AssertionError(f"expected 1968 Searchless actions, got {len(result)}")
    return result


MOVE_TO_ACTION = _compute_actions()


def ordered_legal_moves(board: chess.Board) -> list[chess.Move]:
    return sorted(board.legal_moves, key=lambda move: MOVE_TO_ACTION[move.uci()])

