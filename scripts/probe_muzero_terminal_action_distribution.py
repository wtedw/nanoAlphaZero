"""Compare uniform and policy-favored actions from identical terminal latents, on CPU."""
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import argparse
import json
from pathlib import Path
import time
import chess
import jax
import jax.numpy as jnp
import numpy as np
import pgx1
from probe_muzero_chess_terminals import cases
from nanoalphazero.ckpt import load_model
from nanoalphazero.core import make_env
from nanoalphazero.eval.chess.action import PgxAction


def describe(action, board):
    try:
        move = PgxAction.decode(int(action), board)
        piece = board.piece_at(move.from_square)
        group = "empty-source" if piece is None else "own-piece" if piece.color == board.turn else "opponent-piece"
        return dict(uci=move.uci(), source_group=group, pseudo_legal=board.is_pseudo_legal(move),
                    source_piece=str(piece), target_piece=str(board.piece_at(move.to_square)))
    except ValueError:
        return dict(uci=None, source_group="undecodable", pseudo_legal=False)


def stats(values, target):
    values = np.asarray(values)
    if not len(values):
        return {"n": 0}
    errors = np.abs(values-target)
    return dict(n=len(values), mean_value=float(values.mean()), mae=float(errors.mean()),
                median_error=float(np.median(errors)), max_error=float(errors.max()),
                within_0_1=float(np.mean(errors < .1)), within_0_2=float(np.mean(errors < .2)),
                wrong_sign=float(np.mean(values*target < 0)) if target else None,
                min_value=float(values.min()), max_value=float(values.max()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--random-actions', type=int, default=512)
    p.add_argument('--top-actions', type=int, default=64)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    model, params, cfg = load_model(args.checkpoint)
    assert cfg.get('muzero_predict_value_only', False)
    wrapped = make_env({**cfg, 'enable_sharding': False})
    env = pgx1.make('chess', use_bitmask=True)
    selected = {'fools_mate_black', 'scholars_mate_white', 'previous_missed_Qh1_mate',
                'queen_mate', 'stalemate', 'threefold_repetition', 'fifty_move_rule', 'capture_to_bare_kings'}
    examples = [e for e in cases(env) if e[0] in selected]
    initial = jax.jit(lambda obs: model.apply({'params': params}, obs, method=model.initial))
    recurrent = jax.jit(lambda s, a: model.apply({'params': params}, s, a, method=model.recurrent))
    # Fixed-size chunks avoid recompilation and large CPU activation peaks.
    @jax.jit
    def branch(latent, actions):
        _, reward, _, value = model.apply({'params': params},
            jnp.repeat(latent, actions.shape[0], axis=0), actions, method=model.recurrent)
        return reward, value
    rng = np.random.default_rng(0)
    results = []
    for name, fen, state, action, after, _ in examples:
        start = time.monotonic()
        batch = jax.tree.map(lambda x: x[None], state)
        latent, _, _ = initial(wrapped.observe(batch, batch.current_player))
        terminal, r1, logits, v1 = recurrent(latent, jnp.array([action], jnp.int32))
        logits = np.asarray(logits[0]); probs = np.asarray(jax.nn.softmax(jnp.asarray(logits)))
        target = float(after.rewards[state.current_player])
        board = chess.Board(fen); board.push(PgxAction.decode(action, board))
        uniform = rng.choice(4672, args.random_actions, replace=False)
        top = np.argsort(logits)[::-1][:args.top_actions]
        pseudo = np.array([PgxAction.encode(m, board) for m in board.pseudo_legal_moves], dtype=int)
        actions = np.unique(np.concatenate([uniform, top, pseudo]))
        predictions = {}
        for offset in range(0, len(actions), 32):
            chunk = actions[offset:offset+32]
            padded = np.pad(chunk, (0, 32-len(chunk)), mode='edge')
            rewards, values = jax.device_get(branch(terminal, jnp.asarray(padded, jnp.int32)))
            np.testing.assert_array_equal(rewards, 0.)
            assert np.isfinite(values).all()
            predictions.update(zip(map(int, chunk), map(float, values[:len(chunk)])))
        def values(ids):
            return [predictions[int(a)] for a in ids]
        descriptions = {int(a): describe(a, board) for a in actions}
        uniform_groups = {}
        for group in ['empty-source', 'own-piece', 'opponent-piece', 'undecodable']:
            ids = [a for a in uniform if descriptions[int(a)]['source_group'] == group]
            uniform_groups[group] = stats(values(ids), target)
        row = dict(case=name, target_next_value=target, terminal_value=float(v1[0]),
            uniform=stats(values(uniform), target), top64=stats(values(top), target),
            top8=stats(values(top[:8]), target), pseudo_legal=stats(values(pseudo), target),
            top64_probability_mass=float(probs[top].sum()),
            top64_conditional_policy_mae=float(np.average(np.abs(np.array(values(top))-target), weights=probs[top])),
            uniform_source_groups=uniform_groups,
            uniform_logit_error_correlation=float(np.corrcoef(logits[uniform], np.abs(np.array(values(uniform))-target))[0, 1]),
            top_actions=[dict(action=int(a), logit=float(logits[a]), probability=float(probs[a]),
                next_value=predictions[int(a)], error=abs(predictions[int(a)]-target), **descriptions[int(a)]) for a in top],
            seconds=time.monotonic()-start)
        results.append(row)
        np.savez_compressed(args.output/f'{name}.npz', actions=actions, values=values(actions),
                            uniform=uniform, top=top, pseudo_legal=pseudo, logits=logits)
        (args.output/'report.json').write_text(json.dumps(dict(checkpoint=str(args.checkpoint),
            platform=jax.default_backend(), seed=0, random_actions=args.random_actions,
            rows=results, note='Each action branches once from the identical terminal latent; no chained random trajectory.'), indent=2))
        print(json.dumps({k:row[k] for k in ['case','terminal_value','uniform','top8','top64','pseudo_legal','seconds']}), flush=True)


if __name__ == '__main__':
    main()
