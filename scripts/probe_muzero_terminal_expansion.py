"""CPU diagnostic: widen/deepen fixed root actions without changing live search.

Width visits distinct policy-ranked children of the same successor latent.
Depth follows a single greedy chain. Neither is a full adaptive MCTS rerun.
All reported Qs use the original actor's perspective and include the first
root-action evaluation, as the production two-visit backup does.
"""
import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import argparse
import json
from pathlib import Path

import chess.pgn
import jax
import jax.numpy as jnp
import numpy as np
import pgx1

from nanoalphazero.ckpt import load_model
from nanoalphazero.core import make_env
from nanoalphazero.eval.chess.action import PgxAction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--input-report', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--depth', type=int, default=16)
    args = parser.parse_args()
    report = json.loads((args.input_report or args.audit / 'missed-mate-search.json').read_text())
    source = Path(json.loads((args.audit / 'report.json').read_text())['summary']['source'])
    model, params, config = load_model(args.checkpoint)
    assert config.get('muzero_predict_value_only'), 'This diagnostic assumes zero learned rewards'
    env = pgx1.make('chess', use_bitmask=True)
    wrapped = make_env({**config, 'enable_sharding': False})
    step = jax.jit(env.step)
    states, entries = [], []
    for row in report['rows']:
        with (source / row['game']).open() as f:
            game = chess.pgn.read_game(f)
        state, board = env.init(), game.board()
        for move in list(game.mainline_moves())[:row['ply'] - 1]:
            state = step(state, jnp.int32(PgxAction.encode(move, board)))
            board.push(move)
        assert board.fen() == row['fen']
        for candidate in row['candidates']:
            states.append(state)
            entries.append(dict(game=row['game'], ply=row['ply'], reproduced=row['reproduced'],
                                **candidate))
    roots = jax.tree.map(lambda *xs: jnp.stack(xs), *states)
    initial = jax.jit(lambda obs: model.apply({'params': params}, obs, method=model.initial))
    recurrent = jax.jit(lambda s, a: model.apply({'params': params}, s, a, method=model.recurrent))
    latent, _, _ = initial(wrapped.observe(roots, roots.current_player))
    terminal, rewards, logits, values = recurrent(latent, jnp.asarray([e['action'] for e in entries]))
    np.testing.assert_array_equal(rewards, 0)
    first_q = -np.asarray(values)
    ranked_actions = np.asarray(jax.lax.top_k(logits, args.width)[1])
    width_q, depth_q = [], [first_q]
    print(f'Prepared {len(entries)} root-action candidates', flush=True)
    for rank in range(args.width):
        _, reward, _, value = recurrent(terminal, jnp.asarray(ranked_actions[:, rank]))
        np.testing.assert_array_equal(reward, 0)
        width_q.append(np.asarray(value))
        if rank in (0, 3, 7, 15, 31):
            print(f'Width {rank + 1} complete', flush=True)
    state, policy = terminal, logits
    for depth in range(1, args.depth + 1):
        state, reward, policy, value = recurrent(state, jnp.argmax(policy, axis=-1))
        np.testing.assert_array_equal(reward, 0)
        depth_q.append((-1.) ** (depth + 1) * np.asarray(value))
    width_q, depth_q = np.asarray(width_q), np.asarray(depth_q)
    assert np.isfinite(width_q).all() and np.isfinite(depth_q).all()
    for i, entry in enumerate(entries):
        np.testing.assert_allclose(first_q[i], entry['q1'], atol=3e-5)
        np.testing.assert_allclose(width_q[0, i], entry['q2'], atol=3e-5)
        if args.depth:
            np.testing.assert_allclose(depth_q[1, i], entry['q2'], atol=3e-5)
        entry['width_average_q'] = {str(n): float((first_q[i] + width_q[:n, i].sum()) / (n + 1))
                                    for n in sorted({0, 1, args.width} | {2**k for k in range(1, 13) if 2**k <= args.width})}
        entry['depth_average_q'] = {str(n): float(depth_q[:n + 1, i].mean())
                                    for n in (0, 1, 2, 4, 8, 16) if n <= args.depth}
        entry['depth_leaf_q'] = depth_q[:, i].tolist()
        entry['width_child_q'] = width_q[:, i].tolist()
        entry['width_child_actions'] = ranked_actions[i].tolist()
    output = dict(checkpoint=str(args.checkpoint),
                  note='Controlled width/depth extensions of each fixed root action, not adaptive MCTS. n counts expansions AFTER its first successor. Uniform visit averages, not policy-weighted. Played controls need not be terminal.',
                  rows=entries)
    path = args.output or args.audit / 'terminal-expansion.json'
    path.write_text(json.dumps(output, indent=2))
    print(f'Saved {path}', flush=True)


if __name__ == '__main__':
    main()
