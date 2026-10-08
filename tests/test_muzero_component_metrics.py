"""Mask-aware training counts and event-weighted self-play statistics."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from nanoalphazero.muzero_metrics import (
    aggregate_selfplay_metrics, empty_training_metrics, selfplay_metrics,
    training_batch_metrics, training_norm_metrics,
)


def test_batch_counts_distinguish_windows_positions_and_padding():
    batch = SimpleNamespace(
        policy_mask=jnp.array([[False, True, False], [False, False, False]]),
        value_mask=jnp.array([[False, True, True], [False, False, False]]),
        reward_mask=jnp.array([[False, True], [False, False]]),
        reward=jnp.array([[1., -1., 0.], [1., -1., 0.]]),
    )
    metrics = training_batch_metrics(batch)
    expected = dict(n_is_valid=1, n_is_invalid=1, n_policy_valid=1,
                    n_value_valid=2, n_reward_valid=1, n_reward_pos=0,
                    n_reward_neg=1, n_reward_zero=1)
    assert {name.removeprefix("train_batch/"): int(value) for name, value in metrics.items()} == expected
    empty = empty_training_metrics(True)
    assert set(metrics) <= set(empty)
    assert all(float(value) == 0 for value in empty.values())
    assert "norms/current_lr" not in empty


def test_selfplay_cycle_weights_finishes_and_ignores_empty_steps():
    metrics = {
        "selfplay/p1_wins": jnp.array([1, 0, 1]),
        "selfplay/p2_wins": jnp.array([0, 0, 1]),
        "selfplay/p_just_tied": jnp.array([0, 0, 1]),
        "selfplay/ep_term_step_avg": jnp.array([3., -1., 7.]),
        "selfplay/ep_term_step_min": jnp.array([3., 1e6, 5.]),
        "selfplay/ep_term_step_max": jnp.array([3., -1., 9.]),
        "selfplay/n_legal_moves_avg": jnp.array([4., 5., 6.]),
        "selfplay/n_legal_moves_avg_mid": jnp.array([2., 0., 6.]),
        "selfplay/ep_step_min": jnp.array([0, 1, 2]),
        "selfplay/ep_step_max": jnp.array([4, 5, 6]),
        "selfplay/ep_step_std": jnp.array([1., 2., 3.]),
        "_n_finished": jnp.array([1, 0, 3]),
        "_n_mid": jnp.array([1, 0, 3]),
    }
    result = jax.jit(aggregate_selfplay_metrics)(metrics)
    assert float(result["selfplay/ep_term_step_avg"]) == 6
    assert float(result["selfplay/ep_term_step_min"]) == 3
    assert float(result["selfplay/ep_term_step_max"]) == 9
    assert float(result["selfplay/n_legal_moves_avg_mid"]) == 5
    assert int(result["selfplay/p1_wins"]) == 2
    assert int(result["selfplay/p2_wins"]) == 1
    assert int(result["selfplay/p_just_tied"]) == 1
    assert float(result["selfplay/ep_step_std"]) == 3
    metrics["_n_finished"] = jnp.zeros(3)
    metrics["_n_mid"] = jnp.zeros(3)
    result = aggregate_selfplay_metrics(metrics)
    assert all(float(result[f"selfplay/ep_term_step_{name}"]) == -1 for name in ("min", "max", "avg"))
    assert float(result["selfplay/n_legal_moves_avg_mid"]) == 0
    assert all(np.isfinite(value) for value in result.values())


def test_selection_and_norms():
    result = selfplay_metrics(
        {"selfplay/global_step": 1, "selfplay/p1_wins": 2, "selfplay-reward/unused": 3},
        SimpleNamespace(ep_step=jnp.array([10, 11, 30, 31])),
        SimpleNamespace(just_terminated=jnp.array([[True], [False], [True], [False]])),
    )
    assert set(result) == {"selfplay/p1_wins", "_n_finished", "_n_mid"}
    assert int(result["_n_finished"]) == int(result["_n_mid"]) == 2
    norms = training_norm_metrics({"w": jnp.array([3., 4.])},
                                  {"w": jnp.array([6., 8.])},
                                  {"w": jnp.array([0., 2.])})
    assert {name: float(value) for name, value in norms.items()} == {
        "norms/grad_norm": 10., "norms/param_norm": 5., "norms/update_norm": 2.,
    }
