"""Persistent MuZero component regressions. Run only on an authorized runtime.

These tests are intentionally not run as part of the MacBook implementation
session: the user will move algorithm execution to the TPU environment.
"""

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nanoalphazero.buffers import (
    MuZeroSelfplayOutput,
    MuZeroTrainingSample,
    gather_muzero_training_samples,
    get_dummy_muzero_selfplay_output,
    make_muzero_selfplay_buffer,
    make_selfplay_buffer,
)
from nanoalphazero.config import get_muzero_config
from nanoalphazero.core import make_env, make_selfplay
from nanoalphazero.mcts import PolicyOutput, make_muzero_mcts
from nanoalphazero.model import make_muzero_model
from nanoalphazero.training import muzero_loss


@pytest.fixture
def config():
    cfg = get_muzero_config(
        "ttt", enable_sharding=False, selfplay_batch_size=2, train_batch_size=2,
        selfplay_buffer_consume_size=2, selfplay_buffer_max_len=16,
        replay_buffer_max_len=4, conv_width=8, conv_depth=1,
        muzero_network="vector", muzero_unroll_steps=3,
        cycle_n_selfplay=10, cycle_n_train=1, lr_warmup_steps=0,
        mcts_num_root_considered=2, mcts_num_survivors=1,
        num_exploratory_moves=0,
    )
    cfg.update(game_obs_shape=[3, 3, 2], game_num_actions=9)
    return cfg


@pytest.fixture(scope="module")
def assembly():
    path = Path(__file__).resolve().parents[1] / "muzero.py"
    spec = importlib.util.spec_from_file_location("muzero_components", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assert_trees_equal(left, right):
    a, b = jax.tree.leaves(left), jax.tree.leaves(right)
    assert len(a) == len(b)
    for actual, expected in zip(a, b, strict=True):
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)


def transition(config, t, step, *, terminal=False, reward=0.0,
               discount=-1.0, game_id=1, exploration=False):
    dummy = get_dummy_muzero_selfplay_output(config)
    b = config["selfplay_batch_size"]
    row = jax.tree.map(lambda x: jnp.broadcast_to(x, (b, 1, *x.shape)), dummy)

    def full(value, dtype):
        return jnp.full((b, 1), value, dtype)

    return row.replace(
        global_step_id=full(t, jnp.uint32), col_id=full(t % config["selfplay_buffer_max_len"], jnp.uint32),
        row_id=jnp.arange(1, b + 1, dtype=jnp.uint32)[:, None],
        game_id=full(game_id if terminal else 0, jnp.uint32),
        ep_step=full(step, jnp.int16), ep_termination_step=full(step if terminal else -1, jnp.int16),
        player=full(step % 2, jnp.int32), just_terminated=full(terminal, bool),
        action=full(step % 9, jnp.int32),
        action_weights=jnp.broadcast_to(jax.nn.one_hot(step % 9, 9), (b, 1, 9)),
        transition_reward=full(reward, jnp.float32),
        discount=full(0 if terminal else discount, jnp.float32),
        is_from_selfplay=full(True, bool), is_pending_reward_i8=full(1, jnp.int8),
        is_fresh_i8=full(0, jnp.int8), is_valid_sample=full(False, bool),
        is_exploration=full(exploration, bool),
    )


def append(buffer, state, row):
    # Compile the wrapper as production does, including its donated inner add.
    return jax.jit(buffer.add_backfill)(
        state, row, row.just_terminated[:, 0], jnp.zeros((row.action.shape[0], 2)),
    )[0]


def make_buffer(config):
    return make_muzero_selfplay_buffer(config, get_dummy_muzero_selfplay_output(config))


def test_returns_wait_for_completion_and_preserve_immediate_rewards(config):
    buffer, state = make_buffer(config)
    state = append(buffer, state, transition(config, 0, 0, discount=-0.5))
    state = append(buffer, state, transition(config, 1, 1, discount=-0.5))
    assert int(state.num_valid_consumable) == 0
    state = append(buffer, state, transition(config, 2, 2, terminal=True, reward=1))
    np.testing.assert_allclose(state.experience.reward[:, :3], [[0.25, -0.5, 1]] * 2)
    np.testing.assert_allclose(state.experience.transition_reward[:, :3], [[0, 0, 1]] * 2)
    assert int(state.num_valid_consumable) == 6
    assert int(state.missing_count) == 0


def test_same_player_discount_and_draw_are_valid(config):
    buffer, state = make_buffer(config)
    state = append(buffer, state, transition(config, 0, 0, reward=0.25, discount=0.5))
    state = append(buffer, state, transition(config, 1, 1, terminal=True, reward=0))
    np.testing.assert_allclose(state.experience.reward[:, :2], [[0.25, 0]] * 2)
    assert int(state.num_valid_consumable) == 4


def test_consume_matches_alphazero_selection_and_retains_overlapping_records(config):
    buffer, state = make_buffer(config)
    for t in range(5):
        state = append(buffer, state, transition(config, t, t, terminal=t == 4, reward=float(t == 4)))
    original, _ = make_selfplay_buffer(config, get_dummy_muzero_selfplay_output(config))
    # Independent storage because AlphaZero consume donates its input.
    copy = jax.tree.map(lambda x: jnp.array(x, copy=True), state)
    expected_after, starts, _ = original.consume(copy)
    expected_samples, missing = gather_muzero_training_samples(expected_after.experience, starts, 3)
    actual_after, actual_samples, _ = jax.jit(buffer.consume)(state)
    assert int(missing) == 0
    assert_trees_equal(actual_samples, jax.tree.map(lambda x: x[:, None], expected_samples))
    np.testing.assert_array_equal(actual_after.experience.is_fresh_i8, expected_after.experience.is_fresh_i8)
    np.testing.assert_array_equal(actual_after.experience.action, expected_after.experience.action)
    assert int(actual_after.num_valid_consumable) == 8


def test_terminal_padding_never_uses_next_game(config):
    buffer, state = make_buffer(config)
    state = append(buffer, state, transition(config, 0, 0))
    state = append(buffer, state, transition(config, 1, 1, terminal=True, reward=1, game_id=7))
    state = append(buffer, state, transition(config, 2, 0))
    starts = jax.tree.map(lambda x: x[:, 1:2], state.experience)
    samples, missing = gather_muzero_training_samples(state.experience, starts, 3)
    assert int(missing) == 0
    np.testing.assert_array_equal(samples.policy_mask, [[True, False, False, False]] * 2)
    np.testing.assert_array_equal(samples.reward, [[1, 0, 0, 0]] * 2)
    np.testing.assert_array_equal(samples.transition_reward, [[1, 0, 0]] * 2)
    assert bool(jnp.all(samples.value_mask))
    assert bool(jnp.all(samples.reward_mask))


def test_ring_wrap_and_consumed_starts_keep_chronology(config):
    config = {**config, "selfplay_buffer_max_len": 8, "selfplay_buffer_min_len": 1}
    buffer, state = make_buffer(config)
    # Fill two episodes and consume every start; the next episode crosses wrap.
    for t in range(6):
        state = append(buffer, state, transition(config, t, t % 3, terminal=t % 3 == 2,
                                                 reward=float(t % 3 == 2), game_id=t // 3 + 1))
        if t % 3 == 2:
            for _ in range(3):
                state, _, _ = jax.jit(buffer.consume)(state)
    for t in range(6, 10):
        state = append(buffer, state, transition(config, t, t - 6, terminal=t == 9,
                                                 reward=float(t == 9), game_id=3))
    starts = jax.tree.map(lambda x: x[:, 6:7], state.experience)
    samples, missing = gather_muzero_training_samples(state.experience, starts, 3)
    assert int(state.overwritten_count) == int(state.missing_count) == int(missing) == 0
    np.testing.assert_array_equal(samples.action, [[0, 1, 2]] * 2)
    np.testing.assert_array_equal(samples.reward, [[-1, 1, -1, 1]] * 2)


def test_overwrite_of_pending_game_is_reported(config):
    config = {**config, "selfplay_buffer_max_len": 4, "selfplay_buffer_min_len": 1}
    buffer, state = make_buffer(config)
    for t in range(5):
        state = append(buffer, state, transition(config, t, t))
    assert int(state.overwritten_count) == 2


def test_exploration_is_stored_but_never_selected_as_start(config):
    buffer, state = make_buffer(config)
    for t in range(3):
        state = append(buffer, state, transition(config, t, t, exploration=t == 0,
                                                 terminal=t == 2, reward=float(t == 2)))
    assert int(state.num_valid_consumable) == 4
    np.testing.assert_array_equal(state.experience.is_valid_sample[:, :3], [[False, True, True]] * 2)
    # Retain the actual action even though its start is excluded.
    np.testing.assert_array_equal(state.experience.action[:, :3], [[0, 1, 2]] * 2)


def test_selfplay_wrapper_preserves_original_actions_and_state(config, assembly):
    env = make_env(config)

    def search(key, state, params, scale, batch_size):
        counts = state.legal_action_mask.astype(jnp.float32)
        return PolicyOutput(action=jnp.argmax(counts, axis=-1),
                            action_weights=counts / jnp.sum(counts, axis=-1, keepdims=True),
                            visit_counts=counts)

    original, original_state = make_selfplay(config, env, search)
    wrapped, wrapped_state = assembly.make_muzero_selfplay(config, env, search)
    terminations = 0
    for t in range(24):
        key = jax.random.PRNGKey(t)
        original_state, expected, _ = original(key, original_state, None)
        wrapped_state, actual, _ = wrapped(key, wrapped_state, None)
        assert isinstance(actual, MuZeroSelfplayOutput)
        assert_trees_equal(original_state, wrapped_state)
        for field, value in vars(expected).items():
            assert_trees_equal(value, getattr(actual, field))
        terminations += int(jnp.sum(actual.just_terminated))
    assert terminations > config["selfplay_batch_size"]
    assert int(wrapped_state.step_count) == 24


@pytest.mark.parametrize("network", ["vector", "spatial"])
def test_model_loss_and_gradients_match_research(config, network):
    from nanoalphazero.research.muzero.model import build_model
    from nanoalphazero.research.muzero.learning import loss

    config = {**config, "muzero_network": network}
    original = build_model(dict(network=network, num_actions=9, width=8, depth=1,
                                env_id="tic_tac_toe", activation="mish", use_rvgl=True))
    model, variables = make_muzero_model(config, jax.random.PRNGKey(3))
    original_variables = original.init(jax.random.PRNGKey(3), jnp.zeros((1, 3, 3, 2)), jnp.zeros(1, jnp.int32))
    assert_trees_equal(variables, original_variables)
    batch = MuZeroTrainingSample(
        observation=jnp.ones((2, 3, 3, 2)), action=jnp.array([[0, 1, 2], [3, 4, 5]]),
        action_weights=jnp.ones((2, 4, 9)) / 9, reward=jnp.ones((2, 4)),
        transition_reward=jnp.zeros((2, 3)), policy_mask=jnp.ones((2, 4), bool),
        value_mask=jnp.ones((2, 4), bool), reward_mask=jnp.ones((2, 3), bool),
    )
    old_batch = dict(observation=batch.observation, action=batch.action,
                     policy=batch.action_weights, value=batch.reward, reward=batch.transition_reward,
                     policy_mask=batch.policy_mask, value_mask=batch.value_mask, reward_mask=batch.reward_mask)
    actual = jax.value_and_grad(lambda p: muzero_loss(model, p, batch)[0])(variables["params"])
    expected = jax.value_and_grad(lambda p: loss(original, p, old_batch)[0])(variables["params"])
    assert_trees_equal(actual, expected)


def test_search_matches_research_adapter(config):
    from nanoalphazero.research.muzero.search import latent_search

    env = make_env(config)
    model, variables = make_muzero_model(config, jax.random.PRNGKey(3))
    states = env.init(jax.random.split(jax.random.PRNGKey(1), 2))
    key = jax.random.PRNGKey(5)
    actual = make_muzero_mcts(config, env, model)(key, states, variables["params"], 1.0, 2)
    expected = latent_search(model, variables["params"], env.observe(states, states.current_player),
                             states.legal_action_mask, key, roots=2, survivors=1)
    assert_trees_equal(actual, expected)


def test_cycle_and_checkpoint_continuation(config, assembly, tmp_path):
    from nanoalphazero.checkpoint import save_muzero_checkpoint, load_muzero_checkpoint

    algorithm = assembly.make_muzero(config, jax.random.PRNGKey(0))
    state, metrics = algorithm.run_fn(algorithm.runner_state, jnp.bool_(False))
    assert int(metrics["selfplay_buffer/error_count"]) == 0
    assert int(state.model_ts.n_updates) == 1
    path = tmp_path / "muzero.safetensors"
    payload = {"runner_state": state, "cycle": jnp.int32(1)}
    save_muzero_checkpoint(path, payload, algorithm.config)
    loaded, saved_config = load_muzero_checkpoint(path, payload)
    assert saved_config == algorithm.config
    assert_trees_equal(loaded, payload)
    continued, _ = algorithm.run_fn(state, jnp.bool_(False))
    restored, _ = algorithm.run_fn(loaded["runner_state"], jnp.bool_(False))
    assert_trees_equal(continued, restored)


@pytest.mark.parametrize("error", ["overwritten_count", "missing_count", "truncated_count"])
def test_error_blocks_optimizer_updates(config, assembly, error):
    algorithm = assembly.make_muzero(config, jax.random.PRNGKey(0))
    state = algorithm.runner_state
    state = state._replace(selfplay_buffer_state=state.selfplay_buffer_state.replace(**{error: jnp.int32(1)}))
    following, metrics = algorithm.run_fn(state, jnp.bool_(False))
    assert int(metrics["selfplay_buffer/error_count"]) >= 1
    assert int(following.model_ts.n_updates) == 0
