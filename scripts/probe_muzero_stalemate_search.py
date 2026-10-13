"""Replay audited stalemates and isolate terminal versus continuation errors on CPU."""

import os
os.environ["JAX_PLATFORMS"] = "cpu"

import argparse
import json
import re
from pathlib import Path

import chess.pgn
import jax
import jax.numpy as jnp
import numpy as np
import pgx1
import pgx1.chess as pc

from nanoalphazero.ckpt import load_model
from nanoalphazero.core import make_env
from nanoalphazero.eval.chess.action import PgxAction
from nanoalphazero.mcts import make_muzero_mcts, final_qtransform_completed_by_mix_value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads((args.audit / 'report.json').read_text())
    source = Path(audit['summary']['source'])
    cases = [e for e in json.loads((args.audit / 'followup.json').read_text())
             if e['kind'] == 'STALEMATE']
    env = pgx1.make('chess', use_bitmask=True)
    step = jax.jit(env.step)
    states, boards = [], []
    for case in cases:
        with (source / case['game']).open() as stream:
            game = chess.pgn.read_game(stream)
        board, state = game.board(), env.init(jax.random.PRNGKey(0))
        moves = list(game.mainline_moves())
        for move in moves[:case['ply'] - 1]:
            state = step(state, jnp.int32(PgxAction.encode(move, board)))
            board.push(move)
        assert board.fen() == case['fen']
        after = board.copy(); after.push(moves[case['ply'] - 1])
        assert after.is_stalemate()
        boards.append(board); states.append(state)
    model, params, config = load_model(args.checkpoint)
    assert config['muzero_predict_value_only']
    wrapped = make_env({**config, 'enable_sharding': False})
    batch = jax.tree.map(lambda *xs: jnp.stack(xs), *states)
    legal = jax.vmap(pc.unpack_bitmask)(batch.legal_action_bitmask)
    for i, board in enumerate(boards):
        assert set(np.flatnonzero(legal[i])) == {PgxAction.encode(m, board) for m in board.legal_moves}
    search = make_muzero_mcts(config, wrapped, model, return_diagnostics=True)
    result = jax.device_get(search(jax.random.PRNGKey(42), batch, params, 0., len(cases)))
    initial = jax.jit(lambda obs: model.apply({'params': params}, obs, method=model.initial))
    recur = jax.jit(lambda s, a: model.apply({'params': params}, s, a, method=model.recurrent))
    latent, logits, value = initial(wrapped.observe(batch, batch.current_player))
    prior = np.asarray(jax.nn.softmax(jnp.where(legal, logits, -jnp.inf)))
    candidates = []
    for i, case in enumerate(cases):
        played = PgxAction.encode(boards[i].parse_san(case['played_san']), boards[i])
        best_san = re.sub(r'^\d+\.+\s*', '', case['best']['pv']).split()[0]
        best = PgxAction.encode(boards[i].parse_san(best_san), boards[i])
        case['played_action'], case['engine_best_action'] = played, best
        candidates.extend((i, int(a)) for a in sorted(set(np.flatnonzero(result.visit_counts[i])) | {played, best}))
    parents = jnp.array([i for i, a in candidates]); actions = jnp.array([a for i, a in candidates])
    child, r1, l1, v1 = recur(latent[parents], actions)
    replies = jnp.argmax(l1, axis=-1)
    _, r2, _, v2 = recur(child, replies)
    r1, v1, r2, v2, replies = jax.device_get((r1, v1, r2, v2, replies))
    np.testing.assert_array_equal(r1, 0); np.testing.assert_array_equal(r2, 0)
    rows = []
    for i, case in enumerate(cases):
        selected = boards[i].san(PgxAction.decode(int(result.action[i]), boards[i]))
        rows.append({**case, 'selected': selected, 'reproduced': selected == case['played_san'],
                     'root_value': float(value[i]), 'candidates': []})
    for j, (i, action) in enumerate(candidates):
        move = PgxAction.decode(action, boards[i]); after = boards[i].copy(); after.push(move)
        visits = float(result.visit_counts[i, action])
        q1, q2 = float(-v1[j]), float(v2[j])
        if visits:
            np.testing.assert_allclose(result.diagnostics['backed_up_q'][i, action],
                                       (q1 + q2) / (2 + 1e-6) if visits == 2 else q1, atol=3e-5)
        rows[i]['candidates'].append(dict(action=action, san=boards[i].san(move),
            stalemate=after.is_stalemate(), engine_best=action == cases[i]['engine_best_action'],
            prior=float(prior[i, action]), prior_rank=int(np.sum(prior[i] > prior[i, action])) + 1,
            visits=visits, successor_value=float(v1[j]), continuation_value=float(v2[j]),
            q1=q1, q2=q2, q_search=float(result.diagnostics['backed_up_q'][i, action]) if visits else None,
            score=float(result.diagnostics['final_score'][i, action])))
    def finish(q):
        transformed = jax.vmap(lambda q, v, p, n: final_qtransform_completed_by_mix_value(
            q, v, p, n, value_scale=config.get('mcts_value_scale', 1.),
            maxvisit_init=config.get('mcts_maxvisit_init', 50.),
            rescale_values=config.get('mcts_rescale_values', False)))(
                jnp.asarray(q), jnp.asarray(result.diagnostics['raw_value']),
                jnp.asarray(result.diagnostics['prior_logit']), jnp.asarray(result.visit_counts))
        return np.asarray(jnp.argmax(jnp.where(legal, transformed + result.diagnostics['prior_logit'], -jnp.inf), axis=-1))
    np.testing.assert_array_equal(finish(result.diagnostics['backed_up_q']), result.action)
    for correction in ('terminal', 'continuation', 'both'):
        q = np.array(result.diagnostics['backed_up_q'], copy=True)
        for i, row in enumerate(rows):
            for c in row['candidates']:
                if not c['stalemate'] or not c['visits']:
                    continue
                first = 0. if correction in ('terminal', 'both') else c['q1']
                tail = 0. if correction in ('continuation', 'both') else c['q2']
                q[i, c['action']] = (first + tail) / (2 + 1e-6) if c['visits'] == 2 else first
        for i, action in enumerate(finish(q)):
            rows[i][f'correct_{correction}_selection'] = boards[i].san(PgxAction.decode(int(action), boards[i]))
    report = dict(checkpoint=str(args.checkpoint), rows=rows,
                  counterfactual='Correct only known stalemate values to zero; keep first-rung selection and survivors fixed. Diagnostic oracle, not a full corrected search.')
    (args.audit / 'stalemate-search.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
