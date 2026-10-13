import io
import json

import chess
import chess.pgn
import pytest

from nanoalphazero.diagnostics.chess.acpl import analyze_acpl, analyze_game


def game_from_moves(moves, fen=None):
    game = chess.pgn.Game()
    if fen:
        game.setup(chess.Board(fen))
    board, node = game.board(), game
    for san in moves.split():
        move = board.parse_san(san)
        node = node.add_variation(move)
        board.push(move)
    return game


def test_mate_and_opening_exclusion():
    game = game_from_moves('f3 e5 g4 Qh4#')
    row, events = analyze_game(game)
    assert (row['mate_opportunities'], row['mates_taken'], events) == (1, 1, [])
    assert row['result'] == '0-1'
    game.headers['OpeningPlies'] = '4'
    assert analyze_game(game)[0]['mate_opportunities'] == 0


def test_missed_mate():
    game = game_from_moves('f3 e5 g4 Nc6')
    row, events = analyze_game(game, game_id='example')
    assert row['missed_mates'] == 1
    assert events[0]['mating_moves'] == ['Qh4#']
    assert events[0]['ply'] == 4
    assert events[0]['eventual_result'] == 'unknown'
    game.headers['Result'] = '1-0'
    assert analyze_game(game)[1][0]['eventual_result'] == 'loss'


def test_stalemate_and_repetition():
    game = game_from_moves('Qxd6', '4k3/8/3rP3/5p2/3Q1P2/7P/7P/6RK w - - 1 45')
    row, events = analyze_game(game)
    assert row['termination'] == 'stalemate'
    assert events[0]['kind'] == 'stalemate'
    assert events[0]['eventual_result'] == 'draw'
    game = game_from_moves('Nf3 Nf6 Ng1 Ng8 Nf3 Nf6 Ng1 Ng8')
    row, _ = analyze_game(game)
    assert row['termination'] == 'threefold_repetition'
    assert row['result'] == '1/2-1/2'
    assert row['pgn_result'] == '*'


def test_directory_and_combined_pgn_agree_without_double_counting(tmp_path):
    games = [game_from_moves('f3 e5 g4 Qh4#'), game_from_moves('f3 e5 g4 Nc6')]
    combined = tmp_path / 'games.pgn'
    combined.write_text('\n\n'.join(str(g) for g in games))
    expected = analyze_acpl(combined)
    assert analyze_acpl(tmp_path).scalar_metrics == expected.scalar_metrics
    for i, game in enumerate(games):
        (tmp_path / f'game-{i:06}.pgn').write_text(str(game))
    output = analyze_acpl(tmp_path)
    assert output.scalar_metrics == expected.scalar_metrics
    assert output.scalar_metrics['acpl/behaviour/games'] == 2
    assert output.scalar_metrics['acpl/behaviour/missed_mate_fraction'] == .5
    assert output.histograms['acpl/behaviour/game_plies'] == [4, 4]
    json.dumps(output.tables, allow_nan=False)


def test_empty_and_invalid_input(tmp_path):
    path = tmp_path / 'empty.pgn'
    path.write_text('')
    with pytest.raises(ValueError, match='No games'):
        analyze_acpl(path)
    game = chess.pgn.read_game(io.StringIO('1. e4 *'))
    game.headers['OpeningPlies'] = '2'
    with pytest.raises(ValueError, match='OpeningPlies'):
        analyze_game(game)
    game.errors.append(ValueError('bad PGN'))
    with pytest.raises(ValueError, match='parse errors'):
        analyze_game(game)


def test_no_opportunities_and_empty_event_table(tmp_path):
    path = tmp_path / 'game.pgn'
    path.write_text(str(game_from_moves('e4 e5')))
    output = analyze_acpl(path)
    assert output.scalar_metrics['acpl/behaviour/missed_mate_fraction'] == 0
    assert output.tables['acpl/behaviour/events']['data'] == []
