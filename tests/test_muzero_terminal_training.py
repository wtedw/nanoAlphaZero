"""Terminal-root collection and absorbing targets through real game boundaries."""
import chess
import jax
import jax.numpy as jnp
import numpy as np
import pgx1
import pgx1.chess as pc
import pytest

from nanoalphazero.buffers import (get_dummy_muzero_selfplay_output, make_muzero_selfplay_buffer,
    gather_muzero_training_samples, combine_observation_vmap, split_observation_vmap)
from nanoalphazero.config import get_muzero_config
from nanoalphazero.core import make_env
from nanoalphazero.diagnostics.chess.cases import from_board
from nanoalphazero.eval.chess.action import PgxAction
from nanoalphazero.mcts import PolicyOutput
from nanoalphazero.muzero_metrics import training_batch_metrics
from test_muzero_components import assembly, config, assert_trees_equal


@pytest.mark.parametrize('game', ['hex4', 'chess'])
@pytest.mark.parametrize('value_only', [False, True])
def test_terminal_roots_reset_and_ring_wrap(assembly, game, value_only):
    raw = pgx1.make('hexnoswap_4x4' if game == 'hex4' else game,
                    **({'use_bitmask': True} if game == 'chess' else {}))
    step = jax.jit(raw.step)
    if game == 'chess':
        board = chess.Board('k7/2Q5/2K5/8/8/8/8/8 w - - 0 1')
        root = from_board(raw, board)
        roots = [root, root, raw.init()]
        actions = [PgxAction.encode(board.parse_san(san), board) for san in ('Qb7#', 'Qb6')]
        actions.append(PgxAction.encode(chess.Move.from_uci('e2e4'), chess.Board()))
    else:
        state = raw.init()
        while not bool(state.terminated):
            root = state
            action = int(jnp.argmax(state.legal_action_mask))
            state = step(state, jnp.int32(action))
        roots, actions = [root, raw.init()], [action, 0]
    b, capacity, k = len(roots), 8, 3
    cfg = get_muzero_config(game, enable_sharding=False, selfplay_batch_size=b,
        train_batch_size=b, selfplay_buffer_consume_size=b, selfplay_buffer_max_len=capacity,
        selfplay_buffer_min_len=1, muzero_unroll_steps=k, muzero_train_on_terminal_states=True,
        muzero_predict_value_only=value_only, exp_bnk_action_weights=game == 'chess',
        mcts_num_k_actions=8, num_exploratory_moves=0, muzero_exploration_losses='none')
    env = make_env(cfg)
    cfg.update(game_obs_shape=list(env.obs_shape), game_num_actions=env.num_actions)
    def search(key, state, actions, scale, batch_size):
        weights = jax.nn.one_hot(actions, env.num_actions)
        sparse, indices = jax.lax.top_k(weights, 8)
        return PolicyOutput(action=actions, action_weights=weights, visit_counts=weights,
                            bnk_k_indices=indices.astype(jnp.uint16), bnk_action_weights=sparse)
    collect, sp = assembly.make_muzero_selfplay(cfg, env, search)
    collect = jax.jit(collect)
    sp = sp._replace(env_state=jax.tree.map(lambda *x: jnp.stack(x), *roots),
                     step_count=jnp.int32(capacity - 1))
    dummy = get_dummy_muzero_selfplay_output(cfg)
    buffer, storage = make_muzero_selfplay_buffer(cfg, dummy)
    storage = storage.replace(current_index=jnp.int32(capacity - 1))
    add = jax.jit(buffer.add_backfill)
    def legal_actions(state):
        legal = (jax.vmap(pc.unpack_bitmask)(state.legal_action_bitmask)
                 if game == 'chess' else state.legal_action_mask)
        return jnp.argmax(legal, axis=-1)
    first, output, _ = collect(jax.random.PRNGKey(1), sp, jnp.array(actions))
    terminal = np.asarray(first.env_state.terminated)
    assert terminal[:-1].all() and not terminal[-1]
    assert not np.asarray(output.is_terminal).any()
    expected_final = jax.tree.map(lambda *x: jnp.stack(x),
        *[step(s, jnp.int32(a)) for s, a in zip(roots, actions)])
    np.testing.assert_array_equal(env.observe(first.env_state, first.env_state.current_player),
                                  env.observe(expected_final, expected_final.current_player))
    storage, _ = add(storage, output, first.env_state.terminated, first.env_state.rewards)
    previous_targets = np.asarray(storage.experience.reward[:, -1]).copy()
    second, terminal_output, _ = collect(jax.random.PRNGKey(2), first, legal_actions(first.env_state))
    np.testing.assert_array_equal(terminal_output.is_terminal[:, 0], terminal)
    np.testing.assert_array_equal(terminal_output.transition_reward[terminal], 0)
    assert not np.asarray(second.env_state.terminated).any()
    np.testing.assert_array_equal(second.ep_step[:-1], 0)
    assert int(second.ep_step[-1]) == 2
    # Reset lanes have the initial board; they did not play the ignored action.
    initial = env.init(jax.random.split(jax.random.PRNGKey(3), b))
    np.testing.assert_array_equal(env.observe(second.env_state, second.env_state.current_player)[terminal],
                                  env.observe(initial, initial.current_player)[terminal])
    storage, _ = add(storage, terminal_output, second.env_state.terminated, second.env_state.rewards)
    np.testing.assert_array_equal(storage.experience.reward[:, -1], previous_targets)
    np.testing.assert_array_equal(storage.experience.game_id[terminal, 0], storage.experience.game_id[terminal, -1])
    assert np.asarray(storage.experience.is_valid_sample[terminal, 0]).all()
    # Insert the next game's first move; its pending root must not leak backwards.
    fresh, fresh_output, _ = collect(jax.random.PRNGKey(4), second, legal_actions(second.env_state))
    storage, _ = add(storage, fresh_output, fresh.env_state.terminated, fresh.env_state.rewards)
    assert np.asarray(storage.experience.is_pending_reward_i8[terminal, 1]).all()
    starts = jax.tree.map(lambda x: x[terminal, :1], storage.experience)
    expected = np.asarray(first.env_state.rewards)[terminal,
        np.asarray(first.env_state.current_player)[terminal]] if value_only else np.zeros(terminal.sum())
    for policy in ('none', 'uniform'):
        sample, missing = gather_muzero_training_samples(storage.experience, starts, k, env.num_actions,
            self_consistency=True, exploration_losses='none', value_only=value_only, terminal_policy=policy)
        assert int(missing) == 0
        np.testing.assert_array_equal(sample.reward, expected[:, None] * (-1.) ** np.arange(k + 1))
        np.testing.assert_array_equal(sample.value_mask, True)
        np.testing.assert_array_equal(sample.terminal_mask, True)
        np.testing.assert_array_equal(sample.policy_mask, policy == 'uniform')
        np.testing.assert_array_equal(sample.transition_reward, 0)
        np.testing.assert_array_equal(sample.reward_mask, not value_only)
        assert float(training_batch_metrics(sample)['train_batch/n_is_valid']) == terminal.sum()
        obs = (sample.observation if sample.observation is not None else
               combine_observation_vmap(sample.board_bool[:, 0], sample.board_float[:, 0])[:, None])
        expected_obs = env.observe(first.env_state, first.env_state.current_player)[terminal]
        if game == 'chess':
            # Chess deliberately quantizes float planes in buffer storage.
            expected_obs = combine_observation_vmap(*split_observation_vmap(expected_obs))
        np.testing.assert_array_equal(obs[:, 0], expected_obs)
        if sample.observation is not None:
            np.testing.assert_array_equal(sample.observation[:, 1:], 0)
        else:
            np.testing.assert_array_equal(sample.board_bool[:, 1:], 0)
            np.testing.assert_array_equal(sample.board_float[:, 1:], 0)
    # The preterminal root retains its usual targets and excludes the actual
    # terminal observation from consistency, even though it now exists in storage.
    starts = jax.tree.map(lambda x: x[terminal, -1:], storage.experience)
    sample, missing = gather_muzero_training_samples(storage.experience, starts, k, env.num_actions,
        value_only=value_only)
    assert int(missing) == 0
    np.testing.assert_array_equal(sample.policy_mask[:, 0], True)
    np.testing.assert_array_equal(sample.policy_mask[:, 1:], False)
    np.testing.assert_array_equal(sample.reward[:, 0], previous_targets[terminal])


@pytest.mark.parametrize('sparse', [False, True])
def test_uniform_policy_loss_full_action_gradients(assembly, sparse):
    from nanoalphazero.buffers import get_dummy_muzero_training_sample
    cfg = get_muzero_config('hex4', exp_bnk_action_weights=sparse,
                           mcts_num_k_actions=4, muzero_unroll_steps=2)
    cfg.update(game_obs_shape=[4, 4, 2], game_num_actions=16)
    batch = jax.tree.map(lambda x: x[None], get_dummy_muzero_training_sample(cfg))
    batch = batch.replace(terminal_mask=jnp.ones((1, 3), bool),
                          policy_mask=jnp.ones((1, 3), bool))
    class Model:
        def initial(self):
            pass
        def recurrent(self):
            pass
        def apply(self, variables, inputs, *args, method):
            logits = variables['params'][None]
            base = (jnp.zeros((1, 2)), logits, jnp.zeros(1))
            return base if method == self.initial else (base[0], jnp.zeros(1), base[1], base[2])
    logits = jnp.linspace(-2., 2., 16)
    objective = lambda p: assembly.muzero_loss(Model(), p, batch, value_only=True)[0]
    loss, grad = jax.value_and_grad(objective)(logits)
    expected = -jnp.mean(jax.nn.log_softmax(logits)) - jnp.log(16.)
    np.testing.assert_allclose(loss, expected, atol=1e-6)
    np.testing.assert_allclose(grad, jax.nn.softmax(logits) - 1 / 16, atol=1e-6)
    assert abs(float(objective(jnp.zeros(16)))) < 1e-6
    batch = batch.replace(policy_mask=jnp.zeros((1, 3), bool))
    masked_loss, masked_grad = jax.value_and_grad(objective)(logits)
    assert float(masked_loss) == 0
    np.testing.assert_array_equal(masked_grad, 0)


@pytest.mark.parametrize('sharded', [False, True])
def test_terminal_training_cycle_and_checkpoint(config, assembly, tmp_path, sharded):
    from nanoalphazero.ckpt import save_muzero_checkpoint, load_muzero_checkpoint
    cfg = {**config, 'muzero_train_on_terminal_states': True,
           'muzero_terminal_policy': 'uniform', 'muzero_predict_value_only': True,
           'exp_self_consistency': True, 'selfplay_buffer_min_len': 1}
    if sharded:
        if jax.device_count() != 4:
            pytest.skip('Requires four virtual CPU devices')
        cfg['enable_sharding'] = True
        cfg['buffer_layout'] = 'features'
        for field in ('selfplay_batch_size', 'train_batch_size', 'selfplay_buffer_consume_size',
                      'selfplay_buffer_add_batch_size', 'selfplay_buffer_sample_batch_size',
                      'replay_buffer_add_batch_size', 'replay_buffer_sample_batch_size'):
            cfg[field] = 4
    algorithm = assembly.make_muzero(cfg, jax.random.PRNGKey(0))
    state = algorithm.runner_state
    terminal_seen = False
    for _ in range(3):
        state, (metrics, _) = algorithm.run_fn(state, jnp.bool_(False))
        assert int(metrics['selfplay_buffer/error_count']) == 0
        assert all(np.isfinite(x) for x in metrics.values())
        terminal_seen |= bool(jnp.any(state.selfplay_buffer_state.experience.is_terminal))
    assert terminal_seen
    assert int(state.model_ts.n_updates) == 3
    payload = {'runner_state': state, 'cycle': jnp.int32(3)}
    path = tmp_path / 'terminal.safetensors'
    save_muzero_checkpoint(path, payload, algorithm.config)
    loaded, _ = load_muzero_checkpoint(path, payload)
    assert_trees_equal(loaded, payload)
    next_state, metrics = algorithm.run_fn(state, jnp.bool_(False))
    restored, restored_metrics = algorithm.run_fn(loaded['runner_state'], jnp.bool_(False))
    assert_trees_equal(next_state, restored)
    assert_trees_equal(metrics, restored_metrics)


def test_terminal_uniform_policy_does_not_enable_consistency(config, assembly):
    from nanoalphazero.buffers import get_dummy_muzero_training_sample
    from nanoalphazero.model import make_muzero_model
    cfg = {**config, 'exp_self_consistency': True}
    model, variables = make_muzero_model(cfg, jax.random.PRNGKey(0))
    batch = jax.tree.map(lambda x: x[None], get_dummy_muzero_training_sample(cfg))
    batch = batch.replace(terminal_mask=jnp.ones((1, 4), bool),
                          policy_mask=jnp.ones((1, 4), bool), value_mask=jnp.ones((1, 4), bool))
    loss, metrics = assembly.muzero_loss(model, variables['params'], batch, self_consistency_weight=1.)
    assert np.isfinite(loss)
    assert float(metrics['loss_consistency']) == 0
    assert float(metrics['consistency/n_valid']) == 0


def test_cli_terminal_flags():
    from nanoalphazero.cli import parse_muzero_args
    args = parse_muzero_args(['--muzero.train-on-terminal-states', '--muzero.terminal-policy', 'uniform'])
    assert args.muzero.train_on_terminal_states
    assert args.muzero.terminal_policy == 'uniform'
    defaults = parse_muzero_args([])
    assert not defaults.muzero.train_on_terminal_states
    assert defaults.muzero.terminal_policy == 'none'
