"""Targeted history-aware engine checks following the PGN behaviour scan."""

from concurrent.futures import ThreadPoolExecutor
import argparse
import json
from pathlib import Path

import chess
import chess.engine
import chess.pgn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Existing behaviour audit directory')
    args = parser.parse_args()
    output = args.output
    report = json.loads((output / "report.json").read_text())
    source = Path(report["summary"]["source"])
    candidates = [e for g in report['games'] for e in g['events']
                  if e['kind'] == 'major_piece_taken_next']
    selected = candidates[::max(1, len(candidates)//40)][:40]
    for g in report["games"]:
        if g["termination"] in ("THREEFOLD_REPETITION", "STALEMATE", "FIFTY_MOVES"):
            selected.append(dict(game=g["game"], ply=g["plies"], kind=g["termination"]))
        if g["result"] not in ("1-0", "0-1"):
            selected.extend(e for e in g["events"] if e["kind"] == "missed_mate_in_one")

    def check(e):
        game = chess.pgn.read_game((source/e["game"]).open())
        b = game.board(); moves = list(game.mainline_moves())
        for m in moves[:e["ply"]-1]:
            b.push(m)
        m = moves[e["ply"]-1]
        with chess.engine.SimpleEngine.popen_uci("/usr/local/bin/stockfish") as engine:
            engine.configure({"Threads": 1, "Hash": 32})
            limit = chess.engine.Limit(depth=18, time=1.0)
            best = engine.analyse(b, limit)
            forced = engine.analyse(b, limit, root_moves=[m])
        def pack(info):
            score = info["score"].pov(b.turn)
            return dict(score=str(score), cp=score.score(mate_score=10000),
                        depth=info.get("depth"), pv=b.variation_san(info["pv"]))
        return {**e, "played_san": b.san(m), "fen": b.fen(), "best": pack(best), "forced": pack(forced)}

    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(check, selected))
    (output/"followup.json").write_text(json.dumps(rows, indent=2))
    print(f"Reviewed {len(rows)} positions with full history", flush=True)


if __name__ == "__main__":
    main()
