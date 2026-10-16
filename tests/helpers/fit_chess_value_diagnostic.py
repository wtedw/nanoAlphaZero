"""Bounded frozen-data fitting diagnostic. Does not resume or alter training.

Prepare a disjoint-game dataset on CPU, then run with uv run pod on free TPUs.
Fresh Adam moments, saved learning rate, 200 updates per objective by default.
Policy targets are regenerated once from the frozen checkpoint, not recovered
historical replay targets. All losses/metrics reduce over the global batch.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument('--prepare', action='store_true')
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--checkpoint', type=Path, default=Path('artifacts/dynamics-comparison-20261016/checkpoint'))
parser.add_argument('--games', type=Path, default=Path('artifacts/chess-pod-exact-m16-n32-local-gather-20261015/acpl/cycle-002400-20261016T073159-4919f98182c24a94b7d8698006f35a78'))
parser.add_argument('--steps', type=int, default=200)
args = parser.parse_args()
os.environ['JAX_PLATFORMS'] = 'cpu' if args.prepare else 'tpu'
if not args.prepare:
    from nanoalphazero.distributed import DistributedArgs, initialize_distributed
    initialize_distributed(DistributedArgs(enabled=True))

import chess
import chess.pgn
import jax
import jax.numpy as jnp
import numpy as np
import pgx1
from jax.experimental import multihost_utils

from nanoalphazero.ckpt.model import load_model
from nanoalphazero.core import make_env, DATA_PARALLEL_SHARDING, REPLICATED_SHARDING
from nanoalphazero.buffers import MuZeroTrainingSample
from nanoalphazero.eval.chess.action import PgxAction
from nanoalphazero.mcts import make_muzero_mcts, unpack_bitmask_vmap


def prepare():
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261017)
    files = sorted(args.games.glob('game-*.pgn'))
    rng.shuffle(files)
    raw = pgx1.make('chess', use_bitmask=True)
    step, observe = jax.jit(raw.step), jax.jit(raw.observe)
    windows, observations, actions, targets, records = [], [], [], [], []
    for path in files:
        with path.open() as stream:
            game = chess.pgn.read_game(stream)
        moves = list(game.mainline_moves())
        if len(moves) < 40 or game.headers['Result'] == '*':
            continue
        root = int(rng.integers(30, min(len(moves)-5, 160)))
        outcome = {'1-0': 1., '0-1': -1., '1/2-1/2': 0.}[game.headers['Result']]
        state, board = raw.init(), game.board()
        local_states, obs, act, label = [], [], [], []
        for ply in range(root+6):
            if ply >= root:
                assert not bool(state.terminated | state.truncated)
                legal = np.asarray(unpack_bitmask_vmap(state.legal_action_bitmask[None]))[0]
                expected = np.zeros(4672, bool)
                for move in board.legal_moves:
                    expected[PgxAction.encode(move, board)] = True
                np.testing.assert_array_equal(legal, expected)
                local_states.append(state)
                obs.append(np.asarray(observe(state, state.current_player)))
                label.append(outcome if board.turn else -outcome)
            if ply < root+5:
                action = PgxAction.encode(moves[ply], board)
                if ply >= root:
                    act.append(action)
                state = step(state, jnp.int32(action))
                board.push(moves[ply])
        windows.extend(local_states)
        observations.append(obs)
        actions.append(act)
        targets.append(label)
        records.append(dict(game=path.name, ply=root, result=game.headers['Result'],
                            split='train' if len(records) < 128 else 'heldout'))
        if len(records) % 32 == 0:
            print('Prepared', len(records), flush=True)
        if len(records) == 256:
            break
    assert len(records) == 256
    states = jax.tree.map(lambda *xs: np.stack(xs), *windows)
    arrays = dict(observations=np.asarray(observations), actions=np.asarray(actions, np.int32),
                  targets=np.asarray(targets, np.float32))
    arrays.update({f'state_{i}': x for i, x in enumerate(jax.tree.leaves(states))})
    np.savez_compressed(args.output/'dataset.npz', **arrays)
    (args.output/'dataset.json').write_text(json.dumps(records, indent=2))


def run():
    spec = importlib.util.spec_from_file_location('muzero_fit_audit', 'muzero.py')
    assembly = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = assembly
    spec.loader.exec_module(assembly)
    model, frozen, cfg = load_model(args.checkpoint, sharding=REPLICATED_SHARDING)
    cfg = dict(cfg, lr_warmup_steps=0)
    env = make_env(cfg)
    with np.load(args.output/'dataset.npz') as data:
        arrays = {k: np.array(data[k]) for k in data.files}
    def put(x):
        return jax.make_array_from_callback(x.shape, DATA_PARALLEL_SHARDING, lambda index: x[index])
    arrays = {k: put(x) for k, x in arrays.items()}
    tree = jax.tree.structure(env.init_dummy_estate(1))
    states = tree.unflatten([arrays[f'state_{i}'] for i in range(tree.num_leaves)])
    search = make_muzero_mcts(cfg, env, model, DATA_PARALLEL_SHARDING)
    print('Topology', jax.process_index(), jax.process_count(), jax.device_count(), flush=True)
    policy = search(jax.random.PRNGKey(0), states, frozen, 0., 256*6)
    jax.block_until_ready(policy.action)
    batch = MuZeroTrainingSample(observation=arrays['observations'], action=arrays['actions'],
        reward=arrays['targets'], action_weights=policy.bnk_action_weights.reshape(256, 6, -1),
        k_indices=policy.bnk_k_indices.reshape(256, 6, -1),
        transition_reward=jnp.zeros((256, 5)), policy_mask=jnp.ones((256, 6), bool),
        value_mask=jnp.ones((256, 6), bool), reward_mask=jnp.zeros((256, 5), bool),
        terminal_mask=jnp.zeros((256, 6), bool))
    train = jax.tree.map(lambda x: x[:128], batch)
    heldout = jax.tree.map(lambda x: x[128:], batch)
    # Uniformly shard both halves; slicing global arrays may otherwise retain
    # only part of the mesh for an intermediate value.
    train = jax.tree.map(lambda x: jax.lax.with_sharding_constraint(x, DATA_PARALLEL_SHARDING), train)
    heldout = jax.tree.map(lambda x: jax.lax.with_sharding_constraint(x, DATA_PARALLEL_SHARDING), heldout)
    _, initial = assembly.make_muzero_train(cfg, model, {'params': frozen}, DATA_PARALLEL_SHARDING)

    @jax.jit
    def update(state, batch, policy_weight):
        def objective(params):
            _, metrics = assembly.muzero_loss(model, params, batch, remat=True, value_only=True)
            return metrics['loss_v'] + policy_weight * metrics['loss_pi']
        loss, grads = jax.value_and_grad(objective)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    @jax.jit
    def metrics(params, batch):
        logits, values, _ = assembly.unroll_muzero_predictions(model, params, batch, remat=True)
        obs = batch.observation
        flat = obs.reshape((-1, *obs.shape[2:]))
        fresh = model.apply({'params': params}, flat, method=model.initial)[2].reshape(values.shape)
        return dict(value_mse=jnp.mean((values-batch.reward)**2, axis=0),
                    fresh_mse=jnp.mean((fresh-batch.reward)**2, axis=0),
                    discrepancy=jnp.mean(abs(values-fresh), axis=0),
                    policy_kl=jnp.mean(assembly.muzero_policy_kl(logits, batch.action_weights, batch.k_indices)))

    @jax.jit
    def conflicts(params, batch):
        def components(p):
            _, m = assembly.muzero_loss(model, p, batch, remat=True, value_only=True)
            return jnp.stack([m['loss_v'], m['loss_pi']])
        gradients = jax.jacrev(components)(params)
        out = {}
        for group, g in gradients.items():
            leaves = jax.tree.leaves(g)
            vv = sum(jnp.sum(x[0]**2) for x in leaves)
            pp = sum(jnp.sum(x[1]**2) for x in leaves)
            vp = sum(jnp.sum(x[0]*x[1]) for x in leaves)
            out[group] = dict(value_norm=jnp.sqrt(vv), policy_norm=jnp.sqrt(pp),
                              cosine=vp/jnp.maximum(jnp.sqrt(vv*pp), 1e-12))
        return out

    def host(x):
        return jax.tree.map(lambda v: np.asarray(v).tolist(), x)
    report = dict(checkpoint=str(args.checkpoint), topology=dict(hosts=jax.process_count(), devices=jax.device_count()),
                  steps=args.steps, train_windows=128, heldout_windows=128, learning_rate=cfg['learning_rate'],
                  fresh_optimizer_moments=True, initial_conflicts=host(conflicts(frozen, train)), variants={})
    if jax.process_index() == 0:
        print('Initial gradient conflicts', json.dumps(report['initial_conflicts']), flush=True)
    for name, weight in [('value_only', 0.), ('policy_and_value', 1.)]:
        state = initial
        rows = []
        started = time.monotonic()
        for step in range(args.steps+1):
            if step % 25 == 0 or step == args.steps:
                row = dict(step=step, train=host(metrics(state.params, train)),
                           heldout=host(metrics(state.params, heldout)), seconds=time.monotonic()-started)
                rows.append(row)
                report['variants'][name] = rows
                if jax.process_index() == 0:
                    (args.output/'fit-results.json').write_text(json.dumps(report, indent=2))
                    print(name, json.dumps(row), flush=True)
            if step < args.steps:
                state, loss = update(state, train, jnp.float32(weight))
        report['variants'][name+'_final_conflicts'] = host(conflicts(state.params, train))
    if jax.process_index() == 0:
        (args.output/'fit-results.json').write_text(json.dumps(report, indent=2))
    multihost_utils.sync_global_devices('fit-diagnostic-finished')


if __name__ == '__main__':
    prepare() if args.prepare else run()
