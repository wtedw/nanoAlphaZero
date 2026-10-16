"""Paired Stockfish scoring of frozen root decisions, without loading JAX."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import chess
import chess.engine
import numpy as np


def score_record(record):
    board = chess.Board()
    for move in record['moves']:
        board.push_uci(move)
    candidates = sorted({r['move'] for r in record['results'].values()} | {record['policy_move']})
    with chess.engine.SimpleEngine.popen_uci('/usr/local/bin/stockfish') as engine:
        engine.configure({'Threads': 1, 'Hash': 64})
        limit = chess.engine.Limit(depth=18, time=10.)
        best = engine.analyse(board, limit)
        # Include unrestricted best move, then score every candidate together.
        candidates = sorted(set(candidates) | {best['pv'][0].uci()})
        infos = engine.analyse(board, limit, multipv=len(candidates),
                               root_moves=[chess.Move.from_uci(m) for m in candidates])
        scores = {}
        for info in infos:
            cp = info['score'].pov(board.turn).score(mate_score=10_000)
            scores[info['pv'][0].uci()] = dict(cp=cp, clipped_cp=max(-1000, min(1000, cp)),
                mate=info['score'].pov(board.turn).mate(), depth=info.get('depth'),
                nodes=info.get('nodes'), pv=[m.uci() for m in info['pv']])
    best_score = max(v['clipped_cp'] for v in scores.values())
    record['stockfish'] = scores
    record['results']['policy'] = {'move': record['policy_move']}
    for result in record['results'].values():
        result['loss_cp'] = best_score - scores[result['move']]['clipped_cp']
    return record


def summary(records):
    output = {}
    variants = ['policy', 'learned', 'oracle_rules', 'real_board']
    variants += [v for v in ('real_value', 'real_policy') if v in records[0]['results']]
    for group in ['sample', 'targeted']:
        rows = [r for r in records if r['group'] == group]
        stats = {'count': len(rows)}
        for variant in variants:
            losses = np.array([r['results'][variant]['loss_cp'] for r in rows])
            stats[variant] = dict(mean_loss_cp=float(losses.mean()), median_loss_cp=float(np.median(losses)),
                                 losses_over_100=int((losses > 100).sum()))
        stats['paired'] = {}
        pairs = [('real_board', 'learned'), ('oracle_rules', 'learned'), ('real_board', 'oracle_rules')]
        pairs += [(v, 'oracle_rules') for v in ('real_value', 'real_policy') if v in variants]
        for variant, baseline in pairs:
            delta = np.array([r['results'][baseline]['loss_cp'] - r['results'][variant]['loss_cp'] for r in rows])
            rng = np.random.default_rng(0)
            means = rng.choice(delta, (10000, len(delta)), replace=True).mean(axis=1)
            stats['paired'][f'{variant}_vs_{baseline}'] = dict(
                improvement_cp=float(delta.mean()), bootstrap_95_ci=np.quantile(means, [.025, .975]).tolist(),
                better=int((delta > 0).sum()), worse=int((delta < 0).sum()), equal=int((delta == 0).sum()),
                different_move=sum(r['results'][variant]['move'] != r['results'][baseline]['move'] for r in rows))
        output[group] = stats
    depths = [s['depth'] for r in records for s in r['stockfish'].values()]
    output['scoring'] = dict(depth_target=18, time_cap_seconds=10, min_depth=min(depths),
                              max_depth=max(depths), candidates_below_depth18=sum(d < 18 for d in depths),
                              stockfish_sha256=hashlib.sha256(Path('/usr/local/bin/stockfish').read_bytes()).hexdigest())
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--additional-results', type=Path)
    args = parser.parse_args()
    records = json.loads(args.results.read_text())
    assert all({'learned', 'oracle_rules', 'real_board'} <= r.get('results', {}).keys() for r in records)
    if args.additional_results:
        extra = json.loads(args.additional_results.read_text())
        assert len(extra) == len(records)
        for row, other in zip(records, extra):
            assert all(row[k] == other[k] for k in ('game', 'ply', 'moves', 'fen', 'group'))
            assert not row['results'].keys() & other['results'].keys()
            assert {'real_value', 'real_policy'} <= other['results'].keys()
            row['results'].update(other['results'])
    scored = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for record in pool.map(score_record, records):
            scored.append(record)
            args.results.with_name('scored.json').write_text(json.dumps(scored, indent=2))
            print(f'Scored {len(scored)}/{len(records)}', flush=True)
    result = summary(scored)
    args.results.with_name('summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
