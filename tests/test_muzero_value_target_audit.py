"""Independent chess outcome oracle through collector, backfill and ring wrap."""
import chess
import jax
import jax.numpy as jnp
import numpy as np

from test_muzero_components import assembly
from nanoalphazero.buffers import get_dummy_muzero_selfplay_output, make_muzero_selfplay_buffer, gather_muzero_training_samples
from nanoalphazero.config import get_muzero_config
from nanoalphazero.core import make_env
from nanoalphazero.eval.chess.action import PgxAction
from nanoalphazero.mcts import PolicyOutput


def test_complete_chess_games_value_targets_after_wrap(assembly):
    lines = ['f2f3 e7e5 g2g4 d8h4', 'e2e4 e7e5 f1c4 b8c6 d1h5 g8f6 h5f7']
    encoded, winners = [], []
    for line in lines:
        board, actions = chess.Board(), []
        for uci in line.split():
            move = chess.Move.from_uci(uci)
            actions.append(PgxAction.encode(move, board))
            board.push(move)
        assert board.is_checkmate()
        winners.append(board.outcome().winner)
        encoded.append(actions + [0] * (8-len(actions)))
    actions = jnp.array(encoded)
    cfg = get_muzero_config('chess', enable_sharding=False,
        selfplay_batch_size=2, train_batch_size=2, selfplay_buffer_consume_size=2,
        selfplay_buffer_max_len=8, selfplay_buffer_min_len=1, muzero_unroll_steps=5,
        muzero_predict_value_only=True, muzero_train_on_terminal_states=True,
        muzero_terminal_policy='uniform', muzero_exploration_losses='policy-transition',
        exp_bnk_action_weights=False, num_exploratory_moves=1)
    env = make_env(cfg)
    cfg.update(game_obs_shape=list(env.obs_shape), game_num_actions=env.num_actions)
    def search(key, state, params, scale, count):
        action = actions[jnp.arange(2), jnp.minimum(state._step_count, 7)]
        weights = jax.nn.one_hot(action, env.num_actions)
        return PolicyOutput(action=action, action_weights=weights, visit_counts=weights)
    collect, state = assembly.make_muzero_selfplay(cfg, env, search)
    collect = jax.jit(collect)
    dummy = get_dummy_muzero_selfplay_output(cfg)
    buffer, storage = make_muzero_selfplay_buffer(cfg, dummy)
    add = jax.jit(buffer.add_backfill)
    checked = 0
    for time in range(24):
        state, row, _ = collect(jax.random.PRNGKey(time), state, {})
        storage, _ = add(storage, row, state.env_state.terminated, state.env_state.rewards)
        exp = storage.experience
        for lane in range(2):
            for column in np.flatnonzero(np.asarray(exp.is_valid_sample[lane])):
                start = jax.tree.map(lambda x: x[lane:lane+1, column:column+1], exp)
                sample, missing = gather_muzero_training_samples(exp, start, 5, env.num_actions,
                    exploration_losses='policy-transition', value_only=True, terminal_policy='uniform')
                assert int(missing) == 0
                ply = int(start.ep_step[0, 0])
                expected = [1. if ((ply+k) % 2 == 0) == winners[lane] else -1. for k in range(6)]
                np.testing.assert_array_equal(sample.reward[0], expected)
                np.testing.assert_array_equal(sample.value_mask, True)
                np.testing.assert_array_equal(sample.policy_mask, True)
                np.testing.assert_array_equal(sample.reward_mask, False)
                remaining = int(start.ep_termination_step[0, 0]) - ply
                np.testing.assert_array_equal(sample.terminal_mask[0], np.arange(6) > remaining)
                for k in range(min(5, remaining+1)):
                    assert int(sample.action[0, k]) == encoded[lane][ply+k]
                checked += 1
    assert checked > 50  # Both colors, repeated episodes and circular-buffer wrap.


def test_opening_value_coverage_for_all_thirty_switch_steps():
    # Enumerate the actual switch distribution, before survival/replay weighting.
    cfg = get_muzero_config('chess', exp_bnk_action_weights=True, mcts_num_k_actions=128)
    env = make_env(cfg)
    cfg.update(game_obs_shape=list(env.obs_shape), game_num_actions=env.num_actions)
    dummy = get_dummy_muzero_selfplay_output(cfg)
    exp = jax.tree.map(lambda x: jnp.broadcast_to(x, (30, 40, *x.shape)), dummy)
    positions = jnp.broadcast_to(jnp.arange(40), (30, 40))
    exp = exp.replace(
        row_id=jnp.broadcast_to(jnp.arange(1, 31)[:, None], (30, 40)),
        global_step_id=positions, ep_step=positions, game_id=jnp.ones((30, 40), jnp.uint32),
        ep_termination_step=jnp.full((30, 40), 39),
        is_valid_sample=jnp.ones((30, 40), bool), is_from_selfplay=jnp.ones((30, 40), bool),
        is_pending_reward_i8=jnp.zeros((30, 40), jnp.int8),
        is_exploration=positions < jnp.arange(30)[:, None])
    for root in (0, 10, 20, 29):
        starts = jax.tree.map(lambda x: x[:, root:root+1], exp)
        sample, missing = gather_muzero_training_samples(exp, starts, 5, env.num_actions,
            exploration_losses='policy-transition', value_only=True, terminal_policy='uniform')
        assert int(missing) == 0
        expected = root + np.arange(6)[None, :] >= np.arange(30)[:, None]
        np.testing.assert_array_equal(sample.value_mask, expected)
        assert int(sample.value_mask[:, 0].sum()) == root + 1
        assert int(sample.value_mask.any(axis=1).sum()) == min(root + 6, 30)
        assert np.asarray(sample.policy_mask).all()
