"""Read-only PGN audit: exact mate/repetition checks and targeted engine review."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import json
from pathlib import Path

import chess
import chess.engine
import chess.pgn


VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


def material(board):
    return {name: sum(VALUES[p.piece_type] for p in board.piece_map().values() if p.color == color)
            for name, color in (("white", chess.WHITE), ("black", chess.BLACK))}


def mate_moves(board):
    result = []
    for m in board.legal_moves:
        if board.gives_check(m):
            board.push(m)
            mate = board.is_checkmate()
            board.pop()
            if mate:
                result.append(m)
    return result


def scan(path):
    with path.open() as stream:
        game = chess.pgn.read_game(stream)
    assert not game.errors, (path, game.errors)
    board = game.board()
    moves = list(game.mainline_moves())
    opening = int(game.headers.get("OpeningPlies", 0))
    events = []
    repetitions = Counter()
    repeated = 0
    opportunities = converted = 0
    for i, move in enumerate(moves):
        assert move in board.legal_moves, (path, i, move)
        key = " ".join(board.fen().split()[:4])
        repetitions[key] += 1
        repeated += repetitions[key] > 1
        if i < opening:
            board.push(move)
            continue
        mates = mate_moves(board)
        common = dict(game=path.name, ply=i+1, move_number=board.fullmove_number,
                      side="white" if board.turn else "black", san=board.san(move),
                      uci=move.uci(), fen=board.fen())
        if mates:
            opportunities += 1
            converted += move in mates
            if move not in mates:
                events.append(dict(**common, kind="missed_mate_in_one",
                                   mating_moves=[board.san(m) for m in mates]))
        board.push(move)
        if i+1 < len(moves):
            reply = moves[i+1]
            if reply.to_square == move.to_square and board.is_capture(reply):
                victim = board.piece_at(move.to_square)
                if victim and VALUES[victim.piece_type] >= 5:
                    before = board.copy(); before.pop()
                    captured = before.piece_at(move.to_square)
                    gain = VALUES[captured.piece_type] if captured else 0
                    if VALUES[victim.piece_type] - gain >= 3:
                        events.append(dict(**common, kind="major_piece_taken_next",
                                           reply=board.san(reply), material_given=VALUES[victim.piece_type]-gain))
    outcome = board.outcome(claim_draw=True)
    return dict(game=path.name, plies=len(moves),
                result=outcome.result() if outcome else game.headers.get("Result"),
                pgn_result=game.headers.get("Result"),
                termination=outcome.termination.name if outcome else "no_python_chess_outcome",
                repeated_positions=repeated, max_position_occurrences=max(repetitions.values(), default=0),
                final_material=material(board), final_fen=board.fen(),
                mate_opportunities=opportunities, mates_taken=converted, events=events)


def review(event, directory):
    # Retain repetition/halfmove history for engine adjudication.
    with (directory / event['game']).open() as stream:
        game = chess.pgn.read_game(stream)
    board = game.board()
    moves = list(game.mainline_moves())
    for move in moves[:event['ply'] - 1]:
        board.push(move)
    move = moves[event['ply'] - 1]
    assert move.uci() == event['uci'] and board.fen() == event['fen']
    with chess.engine.SimpleEngine.popen_uci("/usr/local/bin/stockfish") as engine:
        engine.configure({"Threads": 1, "Hash": 32})
        best = engine.analyse(board, chess.engine.Limit(depth=18, time=2.0))
        forced = engine.analyse(board, chess.engine.Limit(depth=18, time=2.0), root_moves=[move])
    color = board.turn
    s1, s2 = best["score"].pov(color), forced["score"].pov(color)
    return dict(**event, best_score=str(s1), played_score=str(s2),
                best_depth=best.get('depth'), played_depth=forced.get('depth'),
                score_loss=s1.score(mate_score=10000)-s2.score(mate_score=10000),
                best_pv=board.variation_san(best["pv"]), played_pv=board.variation_san(forced["pv"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scan-only", action="store_true")
    args = parser.parse_args()
    paths = sorted(args.directory.glob("game-*.pgn"))
    if not paths:
        parser.error(f"No game PGNs found in {args.directory}")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, path in enumerate(paths):
        rows.append(scan(path))
        if (i+1) % 50 == 0:
            print(f"Scanned {i+1} games", flush=True)
    events = [e for r in rows for e in r["events"]]
    summary = dict(source=str(args.directory), games=len(rows),
                   terminations=dict(Counter(r["termination"] for r in rows)),
                   mate_opportunities=sum(r["mate_opportunities"] for r in rows),
                   mates_taken=sum(r["mates_taken"] for r in rows),
                   event_counts=dict(Counter(e["kind"] for e in events)))
    report = dict(summary=summary, games=rows)
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(summary), flush=True)
    if args.scan_only:
        return
    # One missed mate per game; most severe immediate material concessions.
    chosen, seen = [], set()
    for e in events:
        if e["kind"] == "missed_mate_in_one" and e["game"] not in seen:
            chosen.append(e); seen.add(e["game"])
            if len(chosen) == 12:
                break
    sacrifices = sorted((e for e in events if e["kind"] == "major_piece_taken_next"),
                        key=lambda e: e["material_given"], reverse=True)
    chosen.extend(sacrifices[:24])
    with ThreadPoolExecutor(max_workers=4) as pool:
        report["engine_review"] = list(pool.map(partial(review, directory=args.directory), chosen))
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(f"Reviewed {len(chosen)} positions with full history, Stockfish depth 18 / 2s limit", flush=True)


if __name__ == "__main__":
    main()
