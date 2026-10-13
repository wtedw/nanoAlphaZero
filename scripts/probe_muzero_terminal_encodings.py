"""Compare actual terminal observations and learned terminal transitions on CPU."""
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
    args = parser.parse_args()
    source = Path(json.loads((args.audit / 'report.json').read_text())['summary']['source'])
    cases = json.loads((args.audit / 'missed-mate-search.json').read_text())['rows']
    cases += json.loads((args.audit / 'stalemate-search.json').read_text())['rows']
    env = pgx1.make('chess', use_bitmask=True)
    step = jax.jit(env.step)
    before, after, rows = [], [], []
    for case in cases:
        with (source / case['game']).open() as stream:
            game = chess.pgn.read_game(stream)
        board, state = game.board(), env.init(jax.random.PRNGKey(0))
        for move in list(game.mainline_moves())[:case['ply'] - 1]:
            state = step(state, jnp.int32(PgxAction.encode(move, board)))
            board.push(move)
        assert board.fen() == case['fen']
        assert not bool(state.terminated)
        for candidate in case['candidates']:
            mate = candidate.get('kind') == 'mate'
            if not mate and not candidate.get('stalemate', False):
                continue
            action = candidate['action']
            terminal_board = board.copy()
            terminal_board.push(PgxAction.decode(action, board))
            assert terminal_board.is_checkmate() if mate else terminal_board.is_stalemate()
            terminal_state = step(state, jnp.int32(action))
            assert bool(terminal_state.terminated)
            assert int(terminal_state.current_player) != int(state.current_player)
            before.append(state); after.append(terminal_state)
            rows.append(dict(game=case['game'], ply=case['ply'], san=candidate['san'],
                kind='mate' if mate else 'stalemate', action=action,
                terminal_fen=terminal_board.fen(), target=-1. if mate else 0.,
                previous_dynamics_value=candidate['successor_value']))
    print(f'Prepared {len(rows)} terminal transitions with full histories; no autoreset', flush=True)
    model, params, config = load_model(args.checkpoint)
    assert config['muzero_predict_value_only']
    wrapped = make_env({**config, 'enable_sharding': False})
    initial = jax.jit(lambda obs: model.apply({'params': params}, obs, method=model.initial))
    recurrent = jax.jit(lambda s, a: model.apply({'params': params}, s, a, method=model.recurrent))
    pre = jax.tree.map(lambda *xs: jnp.stack(xs), *before)
    post = jax.tree.map(lambda *xs: jnp.stack(xs), *after)
    pre_obs = wrapped.observe(pre, pre.current_player)
    post_obs = wrapped.observe(post, post.current_player)
    assert np.all(np.any(np.asarray(post_obs) != 0, axis=tuple(range(1, post_obs.ndim))))
    latent, _, _ = initial(pre_obs)
    _, reward, _, dynamics_value = recurrent(latent, jnp.array([r['action'] for r in rows]))
    _, _, encoded_value = initial(post_obs)
    dynamics_value, encoded_value = map(np.asarray, (dynamics_value, encoded_value))
    np.testing.assert_array_equal(reward, 0)
    assert np.isfinite(dynamics_value).all() and np.isfinite(encoded_value).all()
    for i, row in enumerate(rows):
        np.testing.assert_allclose(dynamics_value[i], row['previous_dynamics_value'], atol=3e-5)
        row.update(dynamics_value=float(dynamics_value[i]), encoded_value=float(encoded_value[i]),
                   dynamics_error=float(abs(dynamics_value[i] - row['target'])),
                   encoded_error=float(abs(encoded_value[i] - row['target'])))
    summary = {}
    for kind in ('mate', 'stalemate'):
        selected = [r for r in rows if r['kind'] == kind]
        summary[kind] = dict(count=len(selected),
            dynamics_mae=float(np.mean([r['dynamics_error'] for r in selected])),
            encoded_mae=float(np.mean([r['encoded_error'] for r in selected])),
            encoded_better=sum(r['encoded_error'] < r['dynamics_error'] for r in selected))
    output = dict(checkpoint=str(args.checkpoint), summary=summary, rows=rows,
                  note='Both values are from the player-to-move perspective AFTER the real move. Actual terminal boards produced by env.step with complete PGN history, not autostep; encoded using the same wrapped observation function as training. Counts are candidate transitions, including multiple mating moves per position and repeated positions.')
    (args.audit / 'terminal-encodings.json').write_text(json.dumps(output, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
