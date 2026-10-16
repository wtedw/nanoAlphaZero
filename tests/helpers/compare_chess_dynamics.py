"""Frozen-checkpoint paired search ablation; CPU only, no training changes.

Three identical Gumbel searches: production learned dynamics, learned dynamics
with true rules, and fresh representation of real successor boards with true
rules. The latter pair isolates recurrent predictions from legality/endings.
Positions retain PGN history. Targeted failures are reported separately.
Optional real_value / real_policy variants replace just one recurrent output
with the fresh-board prediction, retaining exact rules and learned latents.
"""
import argparse
import json
import os
from pathlib import Path
import time

os.environ['JAX_PLATFORMS'] = 'cpu'

import chess
import chess.pgn
import jax
import jax.numpy as jnp
import numpy as np
import pgx1

from nanoalphazero.ckpt.model import load_model
from nanoalphazero.core import make_env
from nanoalphazero.eval.chess.action import PgxAction
from nanoalphazero.mcts import RootFnOutput, RecurrentFnOutput, make_muzero_mcts, unpack_bitmask_vmap
from nanoalphazero.mcts_exact import search


def legal(state):
    return unpack_bitmask_vmap(state.legal_action_bitmask)


def rule_output(parent, child, logits, value):
    """Reward belongs to the mover; value belongs to the next player."""
    done = child.terminated | child.truncated
    was_done = parent.terminated | parent.truncated
    reward = jnp.take_along_axis(child.rewards, parent.current_player[:, None], axis=1)[:, 0]
    reward = jnp.where(was_done, 0., reward)
    logits = jnp.where(legal(child), logits, jnp.finfo(logits.dtype).min)
    # Absorbing nodes have no effect beyond their incoming terminal reward.
    logits = jnp.where(done[:, None], jnp.zeros_like(logits), logits)
    return RecurrentFnOutput(reward=reward, discount=jnp.where(done, 0., -1.),
                             prior_logits=logits, value=jnp.where(done, 0., value))


def make_search(model, env, config, variant):
    def recurrent(params, key, action, embedding):
        del key
        if variant == 'learned':
            latent, _, logits, value = model.apply(
                {'params': params}, embedding, action, method=model.recurrent)
            return RecurrentFnOutput(reward=jnp.zeros_like(value), discount=-jnp.ones_like(value),
                                     prior_logits=logits, value=value), latent
        if variant in ('oracle_rules', 'real_value', 'real_policy'):
            parent, latent = embedding
            child = env.step(parent, action)
            latent, _, logits, value = model.apply(
                {'params': params}, latent, action, method=model.recurrent)
            if variant != 'oracle_rules':
                _, real_logits, real_value = model.apply(
                    {'params': params}, env.observe(child, child.current_player), method=model.initial)
                if variant == 'real_value':
                    value = real_value
                else:
                    logits = real_logits
            return rule_output(parent, child, logits, value), (child, latent)
        parent = embedding
        child = env.step(parent, action)
        _, logits, value = model.apply(
            {'params': params}, env.observe(child, child.current_player), method=model.initial)
        return rule_output(parent, child, logits, value), child

    @jax.jit
    def run(params, states, key):
        latent, logits, value = model.apply(
            {'params': params}, env.observe(states, states.current_player), method=model.initial)
        mask = legal(states)
        if config.get('exp_use_root_temperature', False):
            logits = jnp.where(mask, logits / config['exp_root_temperature'], jnp.finfo(logits.dtype).min)
        embedding = latent if variant == 'learned' else (states if variant == 'real_board' else (states, latent))
        output = search(params, key, RootFnOutput(prior_logits=logits, value=value, embedding=embedding), recurrent,
                        num_simulations=config['mcts_num_simulations'],
                        num_root_considered=config['mcts_num_root_considered'],
                        invalid_actions=~mask, gumbel_scale=0.,
                        value_scale=config['mcts_value_scale'],
                        maxvisit_init=config['mcts_maxvisit_init'],
                        rescale_values=config['mcts_rescale_values'])
        return output.action, output.action_weights, value, jnp.argmax(jnp.where(mask, logits, -jnp.inf), -1)
    return run


def dataset(directory, count, seed):
    rng = np.random.default_rng(seed)
    paths = sorted(directory.glob('game-*.pgn'))
    selected = rng.choice(len(paths), count, replace=False)
    records = []
    # One position per independently selected game; balanced early/mid/late.
    for i, index in enumerate(selected):
        path = paths[index]
        with path.open() as stream:
            game = chess.pgn.read_game(stream)
        moves = list(game.mainline_moves())
        lo, hi = ((24, 40), (40, 80), (80, 160))[i % 3]
        candidates = list(range(lo, min(hi, len(moves))))
        if not candidates:
            candidates = list(range(24, len(moves))) or list(range(len(moves)))
        ply = int(rng.choice(candidates))
        records.append(dict(group='sample', game=path.name, ply=ply,
                            moves=[m.uci() for m in moves[:ply]]))
    # Historical missed mates in this experiment's cycle-2400 PGN directory.
    for game_id, ply in [(276, 42), (383, 45), (383, 47)]:
        path = directory / f'game-{game_id:06d}.pgn'
        with path.open() as stream:
            game = chess.pgn.read_game(stream)
        records.append(dict(group='targeted', game=path.name, ply=ply,
                            moves=[m.uci() for m in list(game.mainline_moves())[:ply]]))
    return records


def replay(records):
    env = pgx1.make('chess', use_bitmask=True)
    step = jax.jit(env.step)
    states, boards = [], []
    for record in records:
        board, state = chess.Board(), env.init()
        for uci in record['moves']:
            move = chess.Move.from_uci(uci)
            state = step(state, jnp.int32(PgxAction.encode(move, board)))
            board.push(move)
        assert not bool(state.terminated | state.truncated), record
        pgx_legal = np.asarray(legal(jax.tree.map(lambda x: x[None], state)))[0]
        expected = np.zeros(4672, dtype=bool)
        for move in board.legal_moves:
            expected[PgxAction.encode(move, board)] = True
        np.testing.assert_array_equal(pgx_legal, expected)
        record['fen'] = board.fen()
        record['mate_in_one'] = []
        for move in list(board.legal_moves):
            board.push(move)
            if board.is_checkmate():
                record['mate_in_one'].append(move.uci())
            board.pop()
        states.append(state)
        boards.append(board)
    return states, boards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--games', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=48)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20261016)
    parser.add_argument('--variants', nargs='+', default=['learned', 'oracle_rules', 'real_board'],
                        choices=['learned', 'oracle_rules', 'real_board', 'real_value', 'real_policy'])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = dataset(args.games, args.count, args.seed)
    states, boards = replay(records)
    print(f'Replayed {len(records)} positions; all PGX legal masks match python-chess', flush=True)
    model, params, config = load_model(args.checkpoint)
    assert config['muzero_predict_value_only'] and config['mcts_variant'] == 'gumbel'
    manifest = dict(checkpoint=str(args.checkpoint), generation=json.loads((args.checkpoint/'LATEST').read_text()),
                    games=str(args.games), seed=args.seed, config=config, gumbel_scale=0,
                    max_depth=None, return_tree=False, devices=str(jax.devices()), variants=args.variants)
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    (args.output/'positions.json').write_text(json.dumps(records, indent=2))
    env = make_env(config)
    runners = {v: make_search(model, env, config, v) for v in args.variants}
    production = make_muzero_mcts(config, env, model)
    for start in range(0, len(records), args.batch):
        stop = min(start + args.batch, len(records))
        indices = list(range(start, stop)) + [stop - 1] * (args.batch - (stop - start))
        batch = jax.tree.map(lambda *xs: jnp.stack(xs), *[states[i] for i in indices])
        key = jax.random.PRNGKey(args.seed)
        for variant, run in runners.items():
            then = time.monotonic()
            action, weights, values, policy = jax.device_get(run(params, batch, key))
            if start == 0 and variant == 'learned':
                baseline = production(key, batch, params, 0., args.batch)
                np.testing.assert_array_equal(action, np.asarray(baseline.action))
                np.testing.assert_allclose(weights, np.asarray(baseline.action_weights), atol=2e-6, rtol=2e-6)
                print('Production action and dense target equivalence passed', flush=True)
            for j, index in enumerate(range(start, stop)):
                move = PgxAction.decode(int(action[j]), boards[index])
                assert move in boards[index].legal_moves
                records[index].setdefault('results', {})[variant] = dict(move=move.uci())
                records[index]['root_value'] = float(values[j])
                records[index]['policy_move'] = PgxAction.decode(int(policy[j]), boards[index]).uci()
            (args.output/'results.json').write_text(json.dumps(records, indent=2))
            print(f'{start}:{stop} {variant}: {time.monotonic()-then:.2f}s', flush=True)
    print('Search comparisons complete', flush=True)


if __name__ == '__main__':
    main()
