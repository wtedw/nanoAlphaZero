"""Cross-check MuZero collection against real transitions and AZ backfill."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nanoalphazero.buffers import get_dummy_selfplay_output, make_selfplay_buffer
from nanoalphazero.config import get_ttt_config
from nanoalphazero.core import make_env
from nanoalphazero.research.muzero.learning import make_collect
from nanoalphazero.research.muzero.replay import sequences
from nanoalphazero.research.muzero.staging import make_staging


def test_historical_small_episode_smoke_does_not_require_staging_capacity():
    from nanoalphazero.research.muzero.cli import resolve
    config = resolve(dict(env="ttt", defaults="pilot_v1", selfplay_batch_size=8,
                          train_batch_size=8, cycles=2))
    assert config["data_pipeline"] == "episodes"
    with pytest.raises(ValueError, match="staging capacity"):
        resolve({**{k: config[k] for k in ("env", "defaults", "selfplay_batch_size", "train_batch_size", "cycles")},
                 "data_pipeline": "staged"})


class ConstantBootstrap:
    initial = "initial"

    def apply(self, variables, observation, method):
        batch = observation.shape[0]
        return jnp.zeros((batch, 1)), jnp.zeros((batch, 9)), jnp.full(batch, .25)


@pytest.mark.parametrize("cap", [3, 7, 9])
def test_collection_against_real_trajectory_and_production_backfill(cap):
    # One five-ply win and one nine-ply draw in the same batch. The smaller
    # caps exercise nonterminal bootstraps, including padding beside a cutoff.
    actions = jnp.array([[0, 3, 1, 4, 2, 0, 0, 0, 0],
                         [0, 1, 2, 4, 3, 5, 7, 6, 8]])
    config = get_ttt_config()
    env = make_env(config)
    config.update(enable_sharding=False, game_obs_shape=env.obs_shape,
                  selfplay_buffer_add_batch_size=2, selfplay_buffer_sample_batch_size=2,
                  selfplay_buffer_consume_size=18, selfplay_buffer_min_len=1,
                  selfplay_buffer_max_len=9)
    dummy = get_dummy_selfplay_output(config)
    buffer, staging = make_selfplay_buffer(config, dummy)

    def search(key, state, params, scale, batch):
        action = actions[jnp.arange(batch), jnp.minimum(state._step_count, 8)]
        policy = jax.nn.one_hot(action, 9)
        return SimpleNamespace(action=action, visit_counts=policy, action_weights=policy)

    collect_config = dict(selfplay_batch_size=2, exploration_moves=0,
                          max_steps=cap, discount=1.)
    key = jax.random.PRNGKey(71)
    episodes = jax.jit(make_collect(env, ConstantBootstrap(), search, collect_config))({}, key)
    _, init_key = jax.random.split(key)
    state = env.init(jax.random.split(init_key, 2))
    players, observations, rewards = [], [], []
    ended = np.zeros(2, bool)
    for t in range(cap):
        obs = env.observe(state, state.current_player)
        player = state.current_player
        action = actions[:, t]
        following = env.step(state, action)
        valid = ~ended
        reward = np.asarray(following.rewards)[np.arange(2), np.asarray(player)]
        rewards.append(np.where(valid, reward, 0.))
        players.append(np.asarray(player))
        observations.append(np.asarray(obs))
        sample = jax.tree.map(lambda x: jnp.broadcast_to(x, (2, 1, *x.shape)), dummy)
        sample = sample.replace(
            observation=obs[:, None], player=player[:, None], action=action[:, None],
            action_weights=jax.nn.one_hot(action, 9)[:, None],
            is_from_selfplay=jnp.asarray(valid[:, None]),
            global_step_id=jnp.full((2, 1), t, jnp.uint32),
            # Exercise the production exploration exclusion independently.
            is_exploration=jnp.full((2, 1), t < 2),
        )
        staging, _ = buffer.add_backfill(staging, sample,
                                         following.terminated & valid, following.rewards)
        state = jax.tree.map(
            lambda old, new: jnp.where(jnp.asarray(ended).reshape((2,) + (1,) * (new.ndim - 1)), old, new),
            state, following)
        ended |= np.asarray(following.terminated)

    lengths = np.minimum([5, 9], cap)
    np.testing.assert_array_equal(episodes["length"], lengths)
    np.testing.assert_array_equal(episodes["valid"], np.arange(cap)[None] < lengths[:, None])
    np.testing.assert_array_equal(episodes["reward"], np.stack(rewards, 1))
    assert not np.any(episodes["illegal"])
    for b, length in enumerate(lengths):
        for t in range(length):
            np.testing.assert_array_equal(episodes["observation"][b, t], observations[t][b])
            assert episodes["action"][b, t] == actions[b, t]
            np.testing.assert_array_equal(episodes["policy"][b, t], jax.nn.one_hot(actions[b, t], 9))
            if ended[b]:
                # Independent AZ convention: terminal reward indexed by the
                # stored player, rather than alternating a return recursively.
                expected = state.rewards[b, players[t][b]]
                assert episodes["value"][b, t] == expected
                assert staging.experience.reward[b, t] == expected
                assert bool(staging.experience.is_valid_sample[b, t]) == (t >= 2)
            else:
                expected = .25 * (1 if players[t][b] == int(state.current_player[b]) else -1)
                assert episodes["value"][b, t] == expected
                assert not staging.experience.is_valid_sample[b, t]
        # All possible real starting positions, not just one random sample.
        for start in range(length):
            one = jax.tree.map(lambda x: x[b:b + 1], episodes)
            seq = sequences(one, key, 10, jnp.array([start]))
            remaining = length - start
            np.testing.assert_array_equal(seq["reward"][0, :remaining], episodes["reward"][b, start:length])
            np.testing.assert_array_equal(seq["value"][0, :remaining], episodes["value"][b, start:length])
            np.testing.assert_array_equal(seq["policy_mask"][0], np.arange(11) < remaining)
            if ended[b]:
                assert np.all(seq["value"][0, remaining:] == 0)
                assert np.all(seq["reward"][0, remaining:] == 0)
            else:
                assert seq["value"][0, remaining] == .25
                assert not np.any(seq["value_mask"][0, remaining + 1:])
                assert not np.any(seq["reward_mask"][0, remaining:])

    expected_valid = sum(max(lengths[b] - 2, 0) for b in range(2) if ended[b])
    staging, consumed, _ = buffer.consume(staging)
    assert int(jnp.sum(consumed.is_valid_sample)) == expected_valid
    _, consumed_again, _ = buffer.consume(staging)
    assert not np.any(consumed_again.is_valid_sample)


def test_staged_consumption_is_once_only_and_replay_contains_only_real_starts(tmp_path):
    from nanoalphazero.research.muzero import checkpoint
    config = dict(selfplay_batch_size=2, staging_batches=2, max_steps=5,
                  consume_size=2, train_batch_size=32, replay_positions=16, unroll=4)
    init, add, consume, drain, replay = make_staging(config)
    real = jnp.arange(5)[None] < jnp.array([3, 5])[:, None]
    episodes = dict(observation=(jnp.arange(5)[None] + jnp.array([0, 100])[:, None])[..., None],
                    action=jnp.zeros((2, 5), jnp.int32), reward=jnp.zeros((2, 5)),
                    policy=jnp.ones((2, 5, 2)) / 2, value=jnp.zeros((2, 5)),
                    length=jnp.array([3, 5]), terminal=jnp.ones(2, bool), bootstrap=jnp.zeros(2),
                    valid=real, exploration=jnp.broadcast_to(jnp.arange(5)[None] == 0, (2, 5)))
    staged = init(jax.tree.map(lambda x: x[0], episodes))
    staged, metrics = add(staged, episodes)
    assert metrics["staging/new_eligible"] == 6
    assert metrics["staging/overwritten_fresh"] == 0
    original = staged
    observations = []
    for i in range(3):
        staged, batch = consume(staged, jax.random.PRNGKey(i))
        observations.extend(np.asarray(batch["observation"]).flatten().tolist())
        assert np.all(batch["policy_mask"][:, 0])
    assert sorted(observations) == [1, 2, 101, 102, 103, 104]
    staged, empty = consume(staged, jax.random.PRNGKey(4))
    assert not np.any(empty["value_mask"])
    replay_state = replay.init(jax.tree.map(lambda x: x[0], batch))
    staged, replay_state, key, metrics = jax.jit(drain)(original, replay_state, jax.random.PRNGKey(5))
    assert metrics["drain/inserted_positions"] == 6
    assert metrics["drain/remaining_fresh"] == 0
    np.testing.assert_array_equal(np.sort(np.asarray(replay_state.experience["observation"][:, :3]).flatten()),
                                  sorted(observations))
    index = int(replay_state.current_index)
    staged, replay_state, key, metrics = drain(staged, replay_state, key)
    assert replay_state.current_index == index
    assert metrics["drain/inserted_positions"] == 0
    sampled = replay.sample(replay_state, key).experience
    assert np.all(sampled["policy_mask"][:, 0, 0])
    path = tmp_path / "staging.safetensors"
    tree = dict(staging=staged, replay=vars(replay_state), key=key)
    checkpoint.save(path, tree, config)
    restored, _ = checkpoint.load(path, tree)
    for before, after in zip(jax.tree.leaves(tree), jax.tree.leaves(restored)):
        np.testing.assert_array_equal(before, after)
    # Never silently overwrite unconsumed starts: the caller treats this
    # explicit error metric as fatal, before any optimizer update.
    undrained = original
    undrained, _ = add(undrained, episodes)
    _, metrics = add(undrained, episodes)
    assert metrics["staging/overwritten_fresh"] == 6
