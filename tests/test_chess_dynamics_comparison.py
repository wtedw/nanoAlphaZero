"""Perspective, terminal and masking checks for the controlled search ablation."""
import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pgx1

from nanoalphazero.diagnostics.chess.cases import terminal_cases

spec = importlib.util.spec_from_file_location(
    'compare_chess_dynamics', Path(__file__).parent / 'helpers' / 'compare_chess_dynamics.py')
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def batched(state):
    return jax.tree.map(lambda x: x[None], state)


def test_oracle_terminal_rewards_and_absorption():
    env = pgx1.make('chess', use_bitmask=True)
    for name, _, root, _, child, _ in terminal_cases(env):
        parent, successor = batched(root), batched(child)
        output = comparison.rule_output(parent, successor, jnp.ones((1, 4672)), jnp.array([.73]))
        # All named mate cases win for the mover; all other cases are draws.
        win = 'mate' in name and 'stalemate' not in name
        np.testing.assert_array_equal(output.reward, [float(win)])
        np.testing.assert_array_equal(output.discount, [0.])
        np.testing.assert_array_equal(output.value, [0.])
        assert np.isfinite(output.prior_logits).all()
        absorbed = comparison.rule_output(successor, successor, jnp.ones((1, 4672)), jnp.array([.73]))
        np.testing.assert_array_equal(absorbed.reward, [0.])
        np.testing.assert_array_equal(absorbed.discount, [0.])


def test_oracle_live_value_perspective_and_legality():
    env = pgx1.make('chess', use_bitmask=True)
    root = batched(env.init())
    output = comparison.rule_output(root, root, jnp.ones((1, 4672)), jnp.array([.73]))
    np.testing.assert_array_equal(output.reward, [0.])
    np.testing.assert_array_equal(output.discount, [-1.])
    np.testing.assert_allclose(output.value, [.73])
    mask = np.asarray(comparison.legal(root))
    assert mask.sum() == 20
    assert np.all(np.asarray(output.prior_logits)[~mask] < -1e30)
    assert np.all(np.asarray(output.prior_logits)[mask] == 1.)
