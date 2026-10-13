"""PGN fidelity and chess-rule coverage for the standalone HTML viewer."""

import json
import re

import chess
import chess.pgn
import pytest

from nanoalphazero.diagnostics.chess.games import games_html


def payload(html):
    return json.loads(re.search(r'type="application/json">(.*?)</script>', html, re.S).group(1))


@pytest.mark.parametrize("fen,moves", [
    (chess.STARTING_FEN, ["e2e4", "a7a6", "e4e5", "d7d5", "e5d6"]),
    ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", ["e1g1", "e8c8"]),
    ("7k/P7/8/8/8/8/8/7K w - - 0 1", ["a7a8q"]),
    ("7k/8/8/8/8/8/8/7K b - - 0 17", ["h8g8"]),
])
def test_all_positions_match_chess_rules(fen, moves):
    board = chess.Board(fen)
    expected = [board.fen()]
    for move in moves:
        board.push_uci(move)
        expected.append(board.fen())
    pgn = str(chess.pgn.Game.from_board(board)) + "\n\n"
    data = payload(games_html(pgn))
    assert data["pgn"] == pgn
    assert data["games"][0]["pgn"] == pgn
    assert data["games"][0]["positions"] == expected
    assert len(data["pieces"]) == 12


def test_multiple_games_comments_variations_and_script_text():
    pgn = '[Event "</script><script>alert(1)</script>"]\n\n1. e4 {hello} (1. d4) e5 *\n\n1. d4 d5 *\n\n'
    html = games_html(pgn)
    data = payload(html)
    assert len(data["games"]) == 2
    assert data["pgn"] == pgn
    assert ''.join(g["pgn"] for g in data["games"]) == pgn
    assert data["games"][0]["moves"] == ["1. e4", "1... e5"]
    assert "</script><script>alert(1)</script>" not in html
    assert '<script src=' not in html


def test_empty_and_invalid_games():
    assert payload(games_html(""))["games"] == []
    with pytest.raises(ValueError, match="Cannot display game 1"):
        games_html("1. e4 e5 2. Bh6 *")
