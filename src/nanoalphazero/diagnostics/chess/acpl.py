"""Exact PGN behaviour metrics, without a model, engine, or tracking backend.

Call ``analyze_acpl(path)`` with an ACPL directory or multi-game PGN. The result
is a DiagnosticOutput: scalar metrics, per-game/event tables, and length
histograms. Opening moves are replayed for history but excluded from tactical
counts. This does not recompute centipawn loss or attribute errors to MCTS.
"""

from pathlib import Path

import chess
import chess.pgn

from nanoalphazero.diagnostics import DiagnosticOutput


def analyze_game(game: chess.pgn.Game, *, game_id: str = "0") -> tuple[dict, list[dict]]:
    """Inspect one parsed game; retain history for repetition adjudication.

    Misses count positions, not mating actions: a position with three available
    mates contributes one opportunity. Outcome counts on misses describe the
    eventual result for the player who missed, not a proven consequence.
    """
    if game.errors:
        raise ValueError(f"{game_id}: PGN parse errors: {game.errors}")
    board = game.board()
    moves = list(game.mainline_moves())
    opening = int(game.headers.get("OpeningPlies", 0))
    if not 0 <= opening <= len(moves):
        raise ValueError(f"{game_id}: OpeningPlies outside game length")
    events = []
    opportunities = taken = 0
    for index, move in enumerate(moves):
        if move not in board.legal_moves:
            raise ValueError(f"{game_id}: illegal move at ply {index + 1}")
        if index >= opening:
            mating = []
            for candidate in list(board.legal_moves):
                if not board.gives_check(candidate):
                    continue
                board.push(candidate)
                mate = board.is_checkmate()
                board.pop()
                if mate:
                    mating.append(candidate)
            opportunities += bool(mating)
            taken += move in mating
            event = dict(game=game_id, ply=index + 1,
                         side="white" if board.turn else "black", san=board.san(move),
                         uci=move.uci(), fen=board.fen())
            if mating and move not in mating:
                events.append(dict(**event, kind="missed_mate_in_one",
                                   mating_moves=[board.san(m) for m in mating]))
            board.push(move)
            if board.is_stalemate():
                events.append(dict(**event, kind="stalemate", mating_moves=[]))
        else:
            board.push(move)
    outcome = board.outcome(claim_draw=True)
    result = outcome.result() if outcome else game.headers.get("Result", "*")
    termination = outcome.termination.name.lower() if outcome else "unresolved"
    for event in events:
        event['eventual_result'] = (
            'draw' if result == '1/2-1/2' else
            'unknown' if result not in ('1-0', '0-1') else
            'win' if result == ('1-0' if event['side'] == 'white' else '0-1') else 'loss'
        )
    row = dict(game=game_id, plies=len(moves), opening_plies=opening,
               played_plies=len(moves) - opening, result=result,
               pgn_result=game.headers.get('Result', '*'), termination=termination,
               mate_opportunities=opportunities, mates_taken=taken,
               missed_mates=opportunities - taken, final_fen=board.fen())
    return row, events


def analyze_acpl(source: str | Path) -> DiagnosticOutput:
    """Read saved ACPL games and return ready-to-log metrics and ordinary tables.

    A directory uses its game-*.pgn files, or games.pgn if none exist; it never
    counts both. No files are modified. Terminal board outcomes take precedence
    over headers (ACPL repetition draws can have a '*' Result header).
    Fractions with no opportunities are zero; the denominator is also emitted.
    """
    source = Path(source)
    paths = (sorted(source.glob('game-*.pgn')) or [source / 'games.pgn']) if source.is_dir() else [source]
    rows, events = [], []
    for path in paths:
        with path.open() as stream:
            index = 0
            while (game := chess.pgn.read_game(stream)) is not None:
                game_id = path.name if index == 0 else f'{path.name}:{index}'
                row, found = analyze_game(game, game_id=game_id)
                rows.append(row); events.extend(found)
                index += 1
    if not rows:
        raise ValueError(f'No games in {source}')
    misses = [e for e in events if e['kind'] == 'missed_mate_in_one']
    opportunities = sum(r['mate_opportunities'] for r in rows)
    metrics = dict(games=len(rows), plies=sum(r['plies'] for r in rows),
                   played_plies=sum(r['played_plies'] for r in rows),
                   mean_game_plies=sum(r['plies'] for r in rows) / len(rows),
                   mate_opportunities=opportunities,
                   mates_taken=sum(r['mates_taken'] for r in rows),
                   missed_mates=len(misses), missed_mate_fraction=len(misses) / max(opportunities, 1),
                   games_with_missed_mates=sum(r['missed_mates'] > 0 for r in rows),
                   result_header_mismatches=sum(r['result'] != r['pgn_result'] for r in rows))
    for ending in ('checkmate', 'stalemate', 'threefold_repetition', 'fivefold_repetition',
                   'insufficient_material', 'fifty_moves', 'seventyfive_moves', 'unresolved'):
        metrics[f'endings/{ending}'] = sum(r['termination'] == ending for r in rows)
    for result in ('win', 'draw', 'loss', 'unknown'):
        metrics[f'missed_mates/eventual_{result}'] = sum(e['eventual_result'] == result for e in misses)
    def table(records, columns):
        return dict(columns=columns, data=[[r[c] for c in columns] for r in records])
    prefix = 'acpl/behaviour/'
    return DiagnosticOutput(
        scalar_metrics={prefix + k: float(v) for k, v in metrics.items()},
        tables={prefix + 'games': table(rows, list(rows[0])),
                prefix + 'events': table(events, ['game', 'ply', 'side', 'san', 'uci', 'fen',
                                                 'kind', 'mating_moves', 'eventual_result'])},
        histograms={prefix + 'game_plies': [r['plies'] for r in rows]},
    )
