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
    config = {**config, "game_obs_shape": list(env.obs_shape)}
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


def test_sparse_policy_loss_and_gradients_use_full_action_space():
    from nanoalphazero.training import muzero_policy_kl

    logits = jax.random.normal(jax.random.PRNGKey(1), (2, 3, 4672))
    indices = jnp.broadcast_to(jnp.arange(128, dtype=jnp.uint16), (2, 3, 128))
    target = jnp.full((2, 3, 128), 1 / 128).at[:, -1].set(0)
    dense = jnp.zeros_like(logits).at[:, :, :128].set(target)
    sparse_result = jax.value_and_grad(lambda x: jnp.sum(muzero_policy_kl(x, target, indices)))(logits)
    dense_result = jax.value_and_grad(lambda x: jnp.sum(muzero_policy_kl(x, dense)))(logits)
    assert_trees_equal(sparse_result, dense_result)
    assert float(sparse_result[1][0, 0, 4000]) > 0
    np.testing.assert_array_equal(sparse_result[1][:, -1], 0)


def test_policy_compression_matches_selected_logits():
    from nanoalphazero.mcts import compress_muzero_policy

    logits = jax.random.normal(jax.random.PRNGKey(2), (2, 4672))
    output = PolicyOutput(action=jnp.array([0, 1]), action_weights=jax.nn.softmax(logits),
                          visit_counts=jnp.zeros_like(logits))
    compressed, mass = compress_muzero_policy(output, 128)
    selected_logits, indices = jax.lax.top_k(logits, 128)
    np.testing.assert_array_equal(compressed.bnk_k_indices, indices)
    np.testing.assert_allclose(compressed.bnk_action_weights, jax.nn.softmax(selected_logits), rtol=1e-6)
    assert compressed.bnk_k_indices.dtype == jnp.uint16
    assert bool(jnp.all((mass > 0) & (mass < 1)))
    np.testing.assert_array_equal(compressed.action, output.action)
    np.testing.assert_array_equal(compressed.visit_counts, output.visit_counts)


@pytest.fixture
def chess_config():
    cfg = get_muzero_config(
        "chess", enable_sharding=False, selfplay_batch_size=2, train_batch_size=2,
        selfplay_buffer_consume_size=2, selfplay_buffer_max_len=1024,
        selfplay_buffer_min_len=1, replay_buffer_max_len=4,
        conv_width=8, conv_depth=1, muzero_unroll_steps=2,
        cycle_n_selfplay=512, cycle_n_train=1, lr_warmup_steps=0,
        mcts_num_root_considered=2, mcts_num_survivors=1,
        num_exploratory_moves=0, muzero_warmup_cycles=0,
    )
    cfg.update(game_obs_shape=[8, 8, 119], game_num_actions=4672)
    return cfg


def test_packed_sparse_sequences_keep_storage_and_full_padding_space(chess_config):
    from nanoalphazero.buffers import get_dummy_muzero_training_sample

    dummy = get_dummy_muzero_selfplay_output(chess_config)
    exp = jax.tree.map(lambda x: jnp.broadcast_to(x, (2, 4, *x.shape)), dummy)
    exp = exp.replace(
        row_id=jnp.broadcast_to(jnp.array([1, 2], jnp.uint32)[:, None], (2, 4)),
        global_step_id=jnp.broadcast_to(jnp.arange(4, dtype=jnp.uint32), (2, 4)),
        game_id=jnp.ones((2, 4), jnp.uint32), ep_step=jnp.zeros((2, 4), jnp.int16),
        ep_termination_step=jnp.zeros((2, 4), jnp.int16),
        is_pending_reward_i8=jnp.zeros((2, 4), jnp.int8),
        is_from_selfplay=jnp.ones((2, 4), bool), is_valid_sample=jnp.ones((2, 4), bool),
        action_weights=jnp.full((2, 4, 128), 1 / 128),
        k_indices=jnp.broadcast_to(jnp.arange(128, dtype=jnp.uint16), (2, 4, 128)),
    )
    starts = jax.tree.map(lambda x: x[:, :1], exp)
    sample, missing = gather_muzero_training_samples(exp, starts, 2, 4672)
    assert int(missing) == 0
    assert sample.observation is None
    assert sample.board_bool.shape == (2, 936)
    assert sample.k_indices.shape == (2, 3, 128)
    assert bool(jnp.any(sample.action[:, 1] >= 128))
    assert bool(jnp.all(sample.action < 4672))
    np.testing.assert_array_equal(sample.policy_mask, [[True, False, False]] * 2)
    np.testing.assert_array_equal(sample.action_weights[:, 1:], 0)
    expected = get_dummy_muzero_training_sample(chess_config)
    assert jax.tree.structure(sample) == jax.tree.structure(expected)
    with pytest.raises(ValueError, match="full action count"):
        gather_muzero_training_samples(exp, starts, 2)


@pytest.mark.parametrize("sharded", [False, True])
def test_chess_packed_sparse_cycle_and_checkpoint(chess_config, assembly, tmp_path, sharded):
    from nanoalphazero.buffers import combine_observation_vmap, muzero_storage_spec
    from nanoalphazero.training import run_muzero
    from nanoalphazero.checkpoint import (
        save_muzero_checkpoint, load_muzero_checkpoint, muzero_checkpoint_payload,
    )

    if sharded:
        if jax.device_count() != 4:
            pytest.skip("Four virtual CPU devices required")
        chess_config = {**chess_config, "enable_sharding": True}
        for field in ("selfplay_batch_size", "train_batch_size", "selfplay_buffer_consume_size",
                      "selfplay_buffer_add_batch_size", "selfplay_buffer_sample_batch_size",
                      "replay_buffer_add_batch_size", "replay_buffer_sample_batch_size"):
            chess_config[field] = 4
    algorithm = assembly.make_muzero(chess_config, jax.random.PRNGKey(0))
    assert algorithm.buffer_storage["replay"] == muzero_storage_spec(
        algorithm.runner_state.replay_buffer_state, 4 if sharded else 1
    )
    probe = algorithm.compression_probe(
        algorithm.runner_state.model_ts.params, algorithm.runner_state.selfplay_state.env_state,
        jax.random.PRNGKey(8),
    )
    assert int(probe["compression_probe/n_roots"]) == chess_config["selfplay_batch_size"]
    np.testing.assert_allclose(probe["compression_probe/retained_mass_mean"], 1, atol=1e-6)
    assert float(probe["compression_probe/support_overflow_fraction"]) == 0
    state = run_muzero(algorithm, num_iters=1, checkpoint_path=tmp_path / "host.safetensors")
    assert int(state.selfplay_buffer_state.overwritten_count) == 0
    assert int(state.selfplay_buffer_state.missing_count) == 0
    assert int(state.selfplay_buffer_state.truncated_count) == 0
    assert int(state.model_ts.n_updates) == 1
    replay = state.replay_buffer_state.experience
    assert replay.observation is None
    assert replay.board_bool.dtype == jnp.uint8
    assert replay.k_indices.dtype == jnp.uint16
    assert replay.action_weights.shape[-2:] == (3, 128)
    if sharded:
        assert len(replay.board_bool.addressable_shards) == 4
        assert replay.board_bool.addressable_shards[0].data.shape[0] == 1
    batch = algorithm.replay_buffer.sample(state.replay_buffer_state, jax.random.PRNGKey(3)).experience
    batch = jax.tree.map(lambda x: x[:, 0], batch)
    decoded = batch.replace(observation=combine_observation_vmap(batch.board_bool, batch.board_float),
                            board_bool=None, board_float=None)
    objective = jax.jit(jax.value_and_grad(lambda p, b: muzero_loss(algorithm.model, p, b)[0]))
    packed_result = objective(state.model_ts.params, batch)
    assert np.isfinite(float(packed_result[0]))
    assert_trees_equal(packed_result, objective(state.model_ts.params, decoded))
    for full in (False, True):
        payload = muzero_checkpoint_payload(state, 1, full)
        path = tmp_path / f"chess-{full}.safetensors"
        save_muzero_checkpoint(path, payload, algorithm.config)
        restored, config = load_muzero_checkpoint(path, payload)
        assert config == algorithm.config
        assert_trees_equal(payload, restored)
        with pytest.raises(ValueError, match="incompatible"):
            load_muzero_checkpoint(path, muzero_checkpoint_payload(state, 1, not full))
    assert (tmp_path / "chess-False.safetensors").stat().st_size < (tmp_path / "chess-True.safetensors").stat().st_size


def test_compact_resume_restarts_buffers_and_runs_warmup(config, assembly, tmp_path):
    from nanoalphazero.checkpoint import (
        save_muzero_checkpoint, muzero_checkpoint_payload,
    )
    from nanoalphazero.training import run_muzero

    config = {**config, "muzero_checkpoint_replay": False, "muzero_warmup_cycles": 1}
    algorithm = assembly.make_muzero(config, jax.random.PRNGKey(0))
    state, _ = algorithm.run_fn(algorithm.runner_state, jnp.bool_(False))
    payload = muzero_checkpoint_payload(state, 1, False)
    path = tmp_path / "compact.safetensors"
    save_muzero_checkpoint(path, payload, algorithm.config)
    fresh = assembly.make_muzero(config, jax.random.PRNGKey(0))
    restored = run_muzero(fresh, num_iters=1, resume=path)
    assert int(restored.model_ts.n_updates) == int(state.model_ts.n_updates)
    assert int(restored.selfplay_state.step_count) == config["cycle_n_selfplay"]
    assert int(restored.selfplay_buffer_state.overwritten_count) == 0
    assert bool(fresh.replay_buffer.can_sample(restored.replay_buffer_state))
    assert_trees_equal(restored.model_ts, payload["model_ts"])


def test_chess_storage_estimates_and_defaults(chess_config):
    from nanoalphazero.buffers import estimate_muzero_storage

    defaults = get_muzero_config("chess")
    assert defaults["exp_bnk_action_weights"]
    assert defaults["mcts_num_k_actions"] == 128
    assert defaults["muzero_remat_blocks"] and defaults["muzero_remat_unroll"]
    assert not defaults["muzero_checkpoint_replay"]
    assert defaults["replay_buffer_total_size"] == 4096000
    packed = estimate_muzero_storage(chess_config)
    dense_policy = estimate_muzero_storage({**chess_config, "exp_bnk_action_weights": False})
    assert packed["replay"]["bytes"] < dense_policy["replay"]["bytes"] / 10
    assert packed["selfplay"]["bytes"] < dense_policy["selfplay"]["bytes"] / 5
