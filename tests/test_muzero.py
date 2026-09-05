"""MuZero gates: sequence semantics, signed search, gradients and persistence."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax.training.train_state import TrainState

from nanoalphazero.mcts import RecurrentFnOutput, RootFnOutput, gumbel_muzero_policy_1sh
from nanoalphazero.research.muzero import checkpoint
from nanoalphazero.research.muzero.learning import loss, make_collect, train_step
from nanoalphazero.research.muzero.model import MuZero
from nanoalphazero.research.muzero.replay import discounted_returns, make_replay, sequences
from nanoalphazero.research.muzero.search import latent_search, make_search


def episode(terminal=True):
    return dict(observation=jnp.arange(12, dtype=jnp.float32).reshape(1, 3, 4),
                action=jnp.array([[0, 1, 2]]), reward=jnp.array([[0., 0., 1.]]),
                policy=jnp.eye(3)[None], value=jnp.array([[1., -1., 1.]]),
                length=jnp.array([3]), terminal=jnp.array([terminal]), bootstrap=jnp.array([0.7]))


def test_delayed_reward_and_player_signs():
    values = discounted_returns(jnp.array([[0., 0., 1.]]), jnp.array([[-1., -1., 0.]]), jnp.array([99.]))
    np.testing.assert_allclose(values, [[1., -1., 1.]])
    # Same-player / single-agent discount and nonzero intermediate rewards.
    values = discounted_returns(jnp.array([[2., 3.]]), jnp.array([[.5, .5]]), jnp.array([4.]))
    np.testing.assert_allclose(values, [[4.5, 5.]])


def test_defaults_inherit_alphazero_and_hex5_batch_exception():
    from nanoalphazero.config import CONFIG_FACTORIES
    from nanoalphazero.research.muzero.cli import resolve
    for name in CONFIG_FACTORIES:
        config = resolve(dict(env=name, wandb=True))
        base = CONFIG_FACTORIES[name]()
        assert config["learning_rate"] == base["learning_rate"]
        assert config["selfplay_batch_size"] == base["selfplay_batch_size"]
        assert config["train_batch_size"] == (4096 if name == "hex5" else base["train_batch_size"])
        assert config["updates_per_cycle"] == base["cycle_n_train"]
        assert config["warmup_updates"] == base["lr_warmup_steps"]


def test_alignment_terminal_absorbing_and_padding():
    b = sequences(episode(), jax.random.PRNGKey(0), 5, jnp.array([1]))
    np.testing.assert_array_equal(b["observation"], [[4., 5., 6., 7.]])
    np.testing.assert_array_equal(b["action"][:, :2], [[1, 2]])
    np.testing.assert_array_equal(b["reward"], [[0., 1., 0., 0., 0.]])
    np.testing.assert_array_equal(b["value"], [[-1., 1., 0., 0., 0., 0.]])
    np.testing.assert_array_equal(b["policy_mask"], [[1, 1, 0, 0, 0, 0]])
    np.testing.assert_array_equal(b["policy"][:, :2], jnp.eye(3)[None, 1:])
    assert bool(jnp.all(b["reward_mask"]) & jnp.all(b["value_mask"]))


def test_truncated_tail_bootstraps_without_invented_rewards():
    b = sequences(episode(False), jax.random.PRNGKey(0), 4, jnp.array([2]))
    np.testing.assert_allclose(b["value"], [[1., .7, 0., 0., 0.]])
    np.testing.assert_array_equal(b["value_mask"], [[1, 1, 0, 0, 0]])
    np.testing.assert_array_equal(b["reward_mask"], [[1, 0, 0, 0]])


def controlled_search(reward, value, refutation):
    def recurrent(params, key, action, latent):
        depth = latent[:, 0]
        branch = jnp.where(depth == 0, action, latent[:, 1])
        r = jnp.where(depth == 0, reward[action], refutation[branch])
        v = jnp.where(depth == 0, value[action], 0.)
        return RecurrentFnOutput(reward=r, discount=-jnp.ones_like(r),
                                 prior_logits=jnp.zeros((len(action), 2)), value=v), jnp.stack([depth + 1, branch], -1)
    return gumbel_muzero_policy_1sh(
        None, jax.random.PRNGKey(0),
        RootFnOutput(prior_logits=jnp.zeros((1, 2)), value=jnp.zeros(1), embedding=jnp.zeros((1, 2), jnp.int32)),
        recurrent, num_root_considered=2, num_survivors=2, gumbel_scale=0.,
    )


def test_two_rung_predicted_reward_value_and_opponent_refutation():
    zeros = jnp.zeros(2)
    assert int(controlled_search(jnp.array([0., 1.]), zeros, zeros).action[0]) == 1
    assert int(controlled_search(zeros, jnp.array([0., 1.]), zeros).action[0]) == 0
    # Optimistic child value initially prefers action 0; the opponent's +1
    # refutation outweighs that estimate in the second signed backup.
    before = controlled_search(zeros, jnp.array([-.4, 0.]), zeros)
    after = controlled_search(zeros, jnp.array([-.4, 0.]), jnp.array([1., 0.]))
    assert int(before.action[0]) == 0
    assert int(after.action[0]) == 1
    assert float(after.action_weights[0, 0]) < float(before.action_weights[0, 0])


def test_latent_search_cannot_step_environment_and_masks_root(monkeypatch):
    from nanoalphazero.core import make_env
    env = make_env({"env_id": "tic_tac_toe"})
    state = env.init_dummy_estate(1)
    state = env.step(state, jnp.array([0]))
    model = MuZero(9, 16, 1)
    obs = env.observe(state, state.current_player)
    params = model.init(jax.random.PRNGKey(0), obs, jnp.array([1]))["params"]
    def forbidden(*args, **kwargs):
        raise AssertionError("Hypothetical environment transition")
    monkeypatch.setattr(env, "step", forbidden)
    monkeypatch.setattr(env, "autostep", forbidden)
    search = make_search(model, env, dict(roots=9, survivors=4, discount=1.))
    output = search(jax.random.PRNGKey(1), state, params, 0., 1)
    assert output.action_weights[0, 0] == 0
    assert output.action[0] != 0
    np.testing.assert_allclose(jnp.sum(output.action_weights, -1), 1.)


def test_gradients_all_components_and_checkpoint_roundtrip(tmp_path):
    b = sequences(episode(), jax.random.PRNGKey(0), 5, jnp.array([0]))
    model = MuZero(3, 16, 1)
    params = model.init(jax.random.PRNGKey(4), b["observation"], b["action"][:, 0])["params"]
    total, grads = jax.value_and_grad(lambda p: loss(model, p, b)[0])(params)
    assert np.isfinite(total)
    for component in ("representation", "dynamics", "prediction", "reward", "value", "policy"):
        assert np.isfinite(float(optax.global_norm(grads[component])))
        assert float(optax.global_norm(grads[component])) > 0
    state = TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(1e-3))
    state, _ = train_step(model, state, b)
    path = tmp_path / "model.safetensors"
    checkpoint.save(path, {"train": state, "key": jax.random.PRNGKey(2)}, {"width": 16})
    restored, config = checkpoint.load(path, {"train": state, "key": jax.random.PRNGKey(0)})
    assert config == {"width": 16}
    for x, y in zip(jax.tree.leaves(state), jax.tree.leaves(restored["train"])):
        np.testing.assert_array_equal(x, y)
    s1, _ = train_step(model, state, b)
    s2, _ = train_step(model, restored["train"], b)
    for x, y in zip(jax.tree.leaves(s1), jax.tree.leaves(s2)):
        np.testing.assert_array_equal(x, y)


def test_variable_length_episode_replay_wraparound():
    config = dict(selfplay_batch_size=2, train_batch_size=8, replay_batches=2)
    replay = make_replay(config)
    episodes = jax.tree.map(lambda x: jnp.repeat(x, 2, axis=0), episode())
    episodes["length"] = jnp.array([1, 3])
    state = replay.init(jax.tree.map(lambda x: x[0], episodes))
    for i in range(3):
        tagged = {**episodes, "observation": episodes["observation"] + i * 100}
        state = replay.add(state, jax.tree.map(lambda x: x[:, None], tagged))
    sampled = jax.tree.map(lambda x: x[:, 0], replay.sample(state, jax.random.PRNGKey(0)).experience)
    assert jnp.all(sampled["observation"] >= 100)
    b = sequences(sampled, jax.random.PRNGKey(1), 5)
    assert b["policy_mask"].shape == (8, 6)
    assert jnp.all(b["policy_mask"][:, 0])


def test_masked_padding_does_not_change_loss():
    b = sequences(episode(False), jax.random.PRNGKey(0), 5, jnp.array([2]))
    model = MuZero(3, 8, 0)
    params = model.init(jax.random.PRNGKey(0), b["observation"], b["action"][:, 0])["params"]
    changed = {**b, "reward": jnp.where(b["reward_mask"], b["reward"], 100.),
               "value": jnp.where(b["value_mask"], b["value"], -100.),
               "policy": jnp.where(b["policy_mask"][..., None], b["policy"], 100.)}
    np.testing.assert_allclose(loss(model, params, b)[0], loss(model, params, changed)[0])


def test_delayed_reward_gradient_reaches_representation():
    b = sequences(episode(), jax.random.PRNGKey(0), 3, jnp.array([0]))
    b = {**b, "policy_mask": jnp.zeros_like(b["policy_mask"]),
         "value_mask": jnp.zeros_like(b["value_mask"]),
         "reward_mask": jnp.array([[False, False, True]])}
    model = MuZero(3, 16, 1)
    params = model.init(jax.random.PRNGKey(0), b["observation"], b["action"][:, 0])["params"]
    grads = jax.grad(lambda p: loss(model, p, b)[0])(params)
    assert float(optax.global_norm(grads["representation"])) > 0
    assert float(optax.global_norm(grads["dynamics"])) > 0


def test_full_replay_checkpoint(tmp_path):
    replay = make_replay(dict(selfplay_batch_size=1, train_batch_size=1, replay_batches=2))
    episodes = episode()
    state = replay.init(jax.tree.map(lambda x: x[0], episodes))
    state = replay.add(state, jax.tree.map(lambda x: x[:, None], episodes))
    path = tmp_path / "replay.safetensors"
    checkpoint.save(path, vars(state), {"replay_batches": 2})
    loaded, _ = checkpoint.load(path, vars(state))
    restored = state.replace(**loaded)
    for x, y in zip(jax.tree.leaves(state), jax.tree.leaves(restored)):
        np.testing.assert_array_equal(x, y)
        assert x.shape == y.shape


def test_real_selfplay_reward_and_return_alignment():
    from nanoalphazero.core import make_env
    env = make_env({"env_id": "tic_tac_toe"})
    config = dict(selfplay_batch_size=4, roots=9, survivors=4, discount=1.,
                  exploration_moves=4, max_steps=9)
    model = MuZero(9, 8, 0)
    params = model.init(jax.random.PRNGKey(0), jnp.zeros((1, *env.obs_shape)), jnp.zeros(1, jnp.int32))["params"]
    rows = jax.jit(make_collect(env, model, make_search(model, env, config), config))(params, jax.random.PRNGKey(1))
    assert bool(jnp.all(rows["terminal"]))
    assert not bool(jnp.any(rows["illegal"]))
    for b, length in enumerate(np.asarray(rows["length"])):
        assert np.all(np.asarray(rows["reward"])[b, :length - 1] == 0.)
        r = float(rows["reward"][b, length - 1])
        for t in range(length):
            assert float(rows["value"][b, t]) == r * (-1) ** (length - 1 - t)
