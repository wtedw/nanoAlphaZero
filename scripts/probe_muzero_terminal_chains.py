"""Inspect root and recurrent policy/value predictions around checkmate on CPU."""
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import argparse
import json
from pathlib import Path

import chess
import jax
import jax.numpy as jnp
import numpy as np
import pgx1

from probe_muzero_chess_terminals import cases
from nanoalphazero.ckpt import load_model
from nanoalphazero.core import make_env
from nanoalphazero.eval.chess.action import PgxAction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    model, params, config = load_model(args.checkpoint)
    wrapped = make_env({**config, 'enable_sharding': False})
    initial = jax.jit(lambda obs: model.apply({'params': params}, obs, method=model.initial))
    recurrent = jax.jit(lambda s, a: model.apply({'params': params}, s, a, method=model.recurrent))

    def encode(state):
        batch = jax.tree.map(lambda x: x[None], state)
        return initial(wrapped.observe(batch, batch.current_player))

    arrays = {}
    def inspect(key, prediction, board):
        _, logits, value = prediction
        logits = np.asarray(logits[0])
        probabilities = np.asarray(jax.nn.softmax(jnp.asarray(logits)))
        assert np.isfinite(logits).all() and np.isfinite(value).all()
        arrays[key] = logits
        top = []
        for action in np.argsort(logits)[::-1][:5]:
            try:
                uci = PgxAction.decode(int(action), board).uci()
            except ValueError:
                uci = None
            top.append(dict(action=int(action), uci=uci, logit=float(logits[action]),
                            probability=float(probabilities[action])))
        return dict(value=float(value[0]), perspective='white' if board.turn else 'black',
                    top_policy=top)

    def chain(name, prediction, board, first_action=None):
        rows = [inspect(name + '/0', prediction, board)]
        for depth in (1, 2):
            action = first_action if depth == 1 and first_action is not None else int(jnp.argmax(prediction[1][0]))
            latent, reward, logits, value = recurrent(prediction[0], jnp.asarray([action], jnp.int32))
            # No environment transition after termination: keep the terminal board
            # only as a coordinate reference and alternate the action orientation.
            if depth == 1 and first_action is not None:
                board = board.copy()
                board.push(PgxAction.decode(action, board))
            else:
                board = board.copy()
                board.turn = not board.turn
            prediction = (latent, logits, value)
            row = inspect(name + '/' + str(depth), prediction, board)
            row.update(incoming_action=action, predicted_reward=float(reward[0]))
            rows.append(row)
        return rows

    results = []
    selected = {'fools_mate_black', 'scholars_mate_white', 'previous_missed_Qh1_mate'}
    for name, fen, state, action, after, _ in cases(pgx1.make('chess', use_bitmask=True)):
        if name not in selected:
            continue
        assert bool(after.terminated)
        board = chess.Board(fen)
        terminal_board = board.copy()
        terminal_board.push(PgxAction.decode(action, board))
        root = encode(state)
        terminal_latent, reward, logits, value = recurrent(root[0], jnp.asarray([action], jnp.int32))
        row = dict(case=name, fen=fen, mating_action=action,
                   premove=chain(name + '/premove', root, board, action),
                   terminal_dynamics=chain(name + '/terminal_dynamics',
                       (terminal_latent, logits, value), terminal_board),
                   terminal_observed=chain(name + '/terminal_observed', encode(after), terminal_board))
        results.append(row)
        print(json.dumps(row), flush=True)
    assert len(results) == 3
    (args.output / 'report.json').write_text(json.dumps(dict(checkpoint=str(args.checkpoint),
        note='Top policies use unmasked softmax. After terminal, UCI is coordinate decoding only, not a legal move or simulated board. Premove first action is the known mate; all later actions are policy argmax.',
        cases=results), indent=2))
    np.savez_compressed(args.output / 'logits.npz', **arrays)


if __name__ == '__main__':
    main()
