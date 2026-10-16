"""Score the union of frozen search candidates once per root, without JAX."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import chess
import chess.engine


def score(record):
    board = chess.Board()
    for move in record["moves"]:
        board.push_uci(move)
    candidates = [chess.Move.from_uci(m) for m in record["candidates"]]
    assert all(m in board.legal_moves for m in candidates)
    with chess.engine.SimpleEngine.popen_uci("/usr/local/bin/stockfish") as engine:
        engine.configure({"Threads": 1, "Hash": 64})
        infos = engine.analyse(board, chess.engine.Limit(depth=18, time=20),
                               multipv=len(candidates), root_moves=candidates)
    values = {}
    for info in infos:
        value = info["score"].pov(board.turn)
        values[info["pv"][0].uci()] = dict(cp=value.score(mate_score=10000),
            mate=value.mate(), depth=info.get("depth"), nodes=info.get("nodes"))
    assert set(values) == set(record["candidates"])
    return dict(index=record["index"], scores=values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    records = json.loads((args.output / "search.json").read_text())
    grouped = {}
    for row in records:
        item = grouped.setdefault(row["index"], dict(index=row["index"], moves=row["moves"], candidates=set()))
        item["candidates"].update(c["move"] for c in row["candidates"])
    items = [{**item, "candidates": sorted(item["candidates"])} for item in grouped.values()]
    result = dict(depth_target=18, time_cap=20, threads=1, hash_mb=64,
        stockfish_sha256=hashlib.sha256(Path("/usr/local/bin/stockfish").read_bytes()).hexdigest(), records=[])
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for scored in pool.map(score, items):
            result["records"].append(scored)
            (args.output / "stockfish.json").write_text(json.dumps(result, indent=2))
            print(f"scored {len(result['records'])}/{len(items)}", flush=True)
