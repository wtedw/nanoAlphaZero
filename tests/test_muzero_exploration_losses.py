"""Supervision modes on native Hex/chess storage, with independent mask oracles."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nanoalphazero.buffers import (get_dummy_muzero_selfplay_output,
                                 make_muzero_selfplay_buffer, gather_muzero_training_samples)
from nanoalphazero.config import get_muzero_config
from nanoalphazero.core import make_env
from nanoalphazero.cli import parse_muzero_args
from nanoalphazero.muzero_config import normalize_exploration_config
from nanoalphazero.model import make_muzero_model
from test_muzero_components import assembly


@pytest.fixture(scope="module", params=[(g, m) for g in ("hex4", "chess")
                                      for m in ("none", "policy-transition", "all")])
def storage(request):
    game, mode = request.param
    cfg = get_muzero_config(game, enable_sharding=False, selfplay_batch_size=1,
        train_batch_size=1, selfplay_buffer_consume_size=1, selfplay_buffer_max_len=8,
        selfplay_buffer_min_len=1, muzero_unroll_steps=3, exp_bnk_action_weights=False,
        exp_self_consistency=True, muzero_exploration_losses=mode,
        muzero_predict_value_only=False)
    env = make_env(cfg)
    cfg.update(game_obs_shape=list(env.obs_shape), game_num_actions=env.num_actions)
    dummy = get_dummy_muzero_selfplay_output(cfg)
    buffer, _ = make_muzero_selfplay_buffer(cfg, dummy)
    add = jax.jit(buffer.add_backfill)

    def append(state, t, step, explore, terminal=False, outcome=0., game_id=1):
        row = jax.tree.map(lambda x: jnp.broadcast_to(x, (1, 1, *x.shape)), dummy)
        def field(v, dtype):
            return jnp.full((1, 1), v, dtype)
        row = row.replace(
            row_id=field(1, jnp.uint32), col_id=field(t % 8, jnp.uint32),
            global_step_id=field(t, jnp.uint32), ep_step=field(step, jnp.int16),
            ep_termination_step=field(step if terminal else -1, jnp.int16),
            game_id=field(game_id if terminal else 0, jnp.uint32),
            player=field(step % 2, jnp.int32), just_terminated=field(terminal, bool),
            action=field(step, jnp.int32), is_exploration=field(explore, bool),
            is_from_selfplay=field(True, bool), is_pending_reward_i8=field(1, jnp.int8),
            transition_reward=field(outcome if terminal else 0., jnp.float32),
            discount=field(0. if terminal else -1., jnp.float32),
            action_weights=jax.nn.one_hot(jnp.array([[step]]), cfg["game_num_actions"]),
        )
        return add(state, row, row.just_terminated[:, 0], jnp.zeros((1, 2)))[0]
    return cfg, dummy, buffer, append


@pytest.mark.parametrize("exploration", [
    (True,)*5, (False,)*5, (True, True, False, True, True),
    (True, True, True, True, False),
], ids=["all-explore", "all-exploit", "mixed", "exploit-outside-first-window"])
@pytest.mark.parametrize("value_only", [False, True])
@pytest.mark.parametrize("outcome", [-1., 0., 1.])
@pytest.mark.parametrize("start_time", [0, 6], ids=["linear", "circular-wrap"])
def test_game_windows(storage, exploration, value_only, outcome, start_time):
    cfg, dummy, buffer, append = storage
    mode = cfg["muzero_exploration_losses"]
    state = buffer.init(dummy)
    for t in range(start_time):
        state = append(state, t, 0, False, terminal=True, game_id=9)
    for step, explore in enumerate(exploration):
        state = append(state, start_time+step, step, explore,
                       terminal=step == 4, outcome=outcome)
        if step < 4:
            columns = (start_time + np.arange(step+1)) % 8
            assert not np.any(state.experience.is_valid_sample[:, columns])
    # New-game exploit positions must never qualify the old game's windows.
    for step in range(2):
        state = append(state, start_time+5+step, step, False, game_id=2)
    columns = (start_time + np.arange(5)) % 8
    starts = jax.tree.map(lambda x: x[0, columns], state.experience)
    expected_eligible = [mode != "none" or any(not e for e in exploration[s:s+4])
                         for s in range(5)]
    np.testing.assert_array_equal(starts.is_valid_sample, expected_eligible)
    batch, missing = gather_muzero_training_samples(
        state.experience, jax.tree.map(lambda x: x[:, None], starts),
        3, cfg["game_num_actions"], self_consistency=True,
        exploration_losses=mode, value_only=value_only)
    assert int(missing) == 0
    for s, eligible in enumerate(expected_eligible):
        policy, value, reward = [], [], []
        targets = []
        for k in range(4):
            pos = s+k
            real = pos < 5
            explore = exploration[pos] if real else False
            policy.append(eligible and real and (not explore or mode != "none"))
            value.append(eligible and (not real or not explore or mode == "all"))
            targets.append(outcome * (-1.)**(4-pos)
                           if eligible and (real or value_only) else 0.)
            if k < 3:
                reward.append(eligible and not value_only and
                              (not real or not explore or mode != "none"))
                np.testing.assert_allclose(batch.transition_reward[s, k],
                                           outcome if eligible and pos == 4 else 0.)
                if eligible and real:
                    assert batch.action[s, k] == pos
        np.testing.assert_array_equal(batch.policy_mask[s], policy)
        np.testing.assert_array_equal(batch.value_mask[s], value)
        np.testing.assert_array_equal(batch.reward_mask[s], reward)
        np.testing.assert_allclose(batch.reward[s], targets)
        for k in range(4):
            if not eligible or s+k >= 5:
                assert not np.any(batch.action_weights[s, k])
                if k:
                    for name in ("observation", "board_bool", "board_float"):
                        observations = getattr(batch, name)
                        if observations is not None:
                            assert not np.any(observations[s, k])


@pytest.mark.parametrize("mode", ["none", "policy-transition", "all"])
def test_cli(mode):
    args = parse_muzero_args(["--muzero.exploration-losses", mode])
    assert args.muzero.overrides()["muzero_exploration_losses"] == mode


def test_masked_targets_cannot_change_loss_or_gradients(storage, assembly):
    cfg, dummy, buffer, append = storage
    state = buffer.init(dummy)
    for step, explore in enumerate((True, False, True)):
        state = append(state, step, step, explore, terminal=step == 2, outcome=1.)
    starts = jax.tree.map(lambda x: x[:, :1], state.experience)
    batch, missing = gather_muzero_training_samples(
        state.experience, starts, 3, cfg["game_num_actions"], self_consistency=True,
        exploration_losses=cfg["muzero_exploration_losses"])
    assert int(missing) == 0
    model, variables = make_muzero_model(
        {**cfg, "muzero_network": "vector", "conv_width": 8, "conv_depth": 1},
        jax.random.PRNGKey(0))
    objective = jax.jit(jax.value_and_grad(
        lambda params, b: assembly.muzero_loss(model, params, b, self_consistency_weight=1.),
        has_aux=True))
    expected = objective(variables["params"], batch)
    changed = batch.replace(
        action_weights=jnp.where(batch.policy_mask[..., None], batch.action_weights,
                                 jnp.roll(batch.action_weights, 1, axis=-1)),
        reward=batch.reward + jnp.where(batch.value_mask, 0., 100.),
        transition_reward=batch.transition_reward + jnp.where(batch.reward_mask, 0., 100.),
    )
    # Future observations are consistency labels, not recurrent inputs. Changing
    # masked successor observations (including padding) must not change gradients.
    for name in ("observation", "board_bool", "board_float"):
        obs = getattr(changed, name)
        if obs is not None:
            mask = batch.policy_mask.at[:, 0].set(True)
            mask = mask.reshape((*mask.shape, *((1,) * (obs.ndim-2))))
            changed = changed.replace(**{name: jnp.where(mask, obs, jnp.ones_like(obs))})
    actual = objective(variables["params"], changed)
    for a, b in zip(jax.tree.leaves(expected), jax.tree.leaves(actual)):
        np.testing.assert_allclose(a, b, atol=1e-6, rtol=1e-6)
    assert np.isfinite(float(expected[0][0]))
    assert float(expected[0][1]["consistency/n_valid"]) == int(batch.policy_mask[:, 1:].sum())
    # Unmasked labels must still contribute; this also covers padded value loss.
    active = batch.replace(reward=batch.reward + jnp.where(batch.value_mask, 100., 0.))
    assert float(objective(variables["params"], active)[0][1]["loss_v"]) > 1000


@pytest.mark.parametrize("flag", ["--exp-train-on-exploit-unroll-states",
                                  "--exp-train-on-exploration-without-value"])
def test_removed_flags(flag):
    with pytest.raises(SystemExit):
        parse_muzero_args([flag])


@pytest.mark.parametrize("exploit,transition,expected", [
    (True, False, "none"), (False, False, "all"),
    (True, True, "policy-transition"), (False, True, "policy-transition"),
])
def test_saved_flag_mapping(exploit, transition, expected):
    assert normalize_exploration_config({
        "exp_train_on_exploit_unroll_states": exploit,
        "exp_train_on_exploration_without_value": transition,
    }) == {"muzero_exploration_losses": expected}
