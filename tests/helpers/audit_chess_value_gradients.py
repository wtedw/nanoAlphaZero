"""CPU-only frozen-checkpoint per-depth outcome losses and gradient norms.

Uses recorded legal PGN trajectories, not the live training replay buffer.
"""
import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import chess
import chess.pgn
import jax
import jax.numpy as jnp
import numpy as np

from compare_chess_dynamics import replay
from nanoalphazero.ckpt.model import load_model
from nanoalphazero.core import make_env
from nanoalphazero.eval.chess.action import PgxAction

spec = importlib.util.spec_from_file_location('muzero_audit_module', 'muzero.py')
assembly = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = assembly
spec.loader.exec_module(assembly)


def main():
    base = Path('artifacts/dynamics-comparison-20261016')
    output = Path('artifacts/value-target-audit-20261017')
    manifest = json.loads((base/'expanded/manifest.json').read_text())
    records = json.loads((base/'expanded/positions.json').read_text())
    chosen, continuations, outcomes = [], [], []
    for record in records:
        if record['group'] != 'sample' or record['ply'] < 30:
            continue
        with (Path(manifest['games'])/record['game']).open() as stream:
            game = chess.pgn.read_game(stream)
        moves = list(game.mainline_moves())
        if len(moves) <= record['ply'] + 5 or game.headers['Result'] == '*':
            continue
        chosen.append(record)
        continuations.append(moves[record['ply']:record['ply']+5])
        outcomes.append({'1-0': 1., '0-1': -1., '1/2-1/2': 0.}[game.headers['Result']])
        if len(chosen) == 16:
            break
    states, boards = replay(chosen)
    model, params, config = load_model(base/'checkpoint')
    env = make_env(config)
    state = jax.tree.map(lambda *xs: jnp.stack(xs), *states)
    observations, actions, labels = [], [], []
    for depth in range(6):
        assert not np.asarray(state.terminated | state.truncated).any()
        observations.append(env.observe(state, state.current_player))
        labels.append(jnp.array([z if b.turn else -z for z, b in zip(outcomes, boards)]))
        if depth < 5:
            move = [moves[depth] for moves in continuations]
            action = jnp.array([PgxAction.encode(m, b) for m, b in zip(move, boards)])
            actions.append(action)
            state = env.step(state, action)
            for b, m in zip(boards, move):
                b.push(m)
    observations = jnp.stack(observations, 1)
    actions, labels = jnp.stack(actions, 1), jnp.stack(labels, 1)

    def predicted(p, obs, act):
        batch = SimpleNamespace(observation=obs[:, :1], action=act)
        return assembly.unroll_muzero_predictions(model, p, batch)[1]

    @jax.jit
    def measure(p, obs, act, target, depth):
        def loss(p):
            value = predicted(p, obs, act)
            return jnp.mean((value[:, depth] - target[:, depth]) ** 2)
        loss_value, gradients = jax.value_and_grad(loss)(p)
        norms = {k: jnp.sqrt(sum(jnp.sum(x*x) for x in jax.tree.leaves(g)))
                 for k, g in gradients.items()}
        return loss_value, norms

    values = np.asarray(jax.jit(predicted)(params, observations, actions))
    flat = observations.reshape((-1, *observations.shape[2:]))
    fresh = np.asarray(jax.jit(lambda p, o: model.apply({'params': p}, o, method=model.initial)[2])(
        params, flat)).reshape(values.shape)
    result = dict(checkpoint_cycle=3200, positions=chosen, sample_count=len(chosen),
        note='16 fixed PGN windows after ply 30; all six positions live. Outcome labels are played-game results. Each depth loss is normalized separately, not the production 6-position mean. No optimizer updates.',
        values=values.tolist(), fresh_values=fresh.tolist(), targets=np.asarray(labels).tolist(), depths=[])
    for depth in range(6):
        loss, norms = jax.device_get(measure(params, observations, actions, labels, jnp.int32(depth)))
        row = dict(depth=depth, imagined_mse=float(loss),
                   fresh_mse=float(np.mean((fresh[:, depth]-np.asarray(labels)[:, depth])**2)),
                   imagined_vs_fresh_mae=float(np.mean(abs(values[:, depth]-fresh[:, depth]))),
                   gradient_norms={k: float(v) for k, v in norms.items()})
        result['depths'].append(row)
        print(json.dumps(row), flush=True)
        (output/'checkpoint-gradient-audit.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
