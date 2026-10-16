"""Analytic gradient oracle for the production unroll and value masks."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from test_muzero_components import assembly
from nanoalphazero.buffers import MuZeroTrainingSample


class LinearModel:
    initial = 'initial'
    recurrent = 'recurrent'

    def apply(self, variables, x, action=None, method=None):
        p = variables['params']
        if method == self.initial:
            latent = jnp.broadcast_to(p['h'], (x.shape[0],))
            return latent, jnp.zeros((x.shape[0], 2)), latent * p['f']
        latent = x + p['g']
        return latent, jnp.zeros_like(latent), jnp.zeros((x.shape[0], 2)), latent * p['f']


def test_gradient_paths_have_expected_scaling(assembly):
    batch = SimpleNamespace(observation=jnp.zeros((1, 1, 1)), action=jnp.zeros((1, 5), jnp.int32))
    params = dict(h=jnp.array(.2), g=jnp.array(.1), f=jnp.array(1.))
    def predict(p):
        return assembly.unroll_muzero_predictions(LinearModel(), p, batch)[1][0]
    jac = jax.jacrev(predict)(params)
    np.testing.assert_allclose(jac['h'], .5 ** np.arange(6))
    np.testing.assert_allclose(jac['g'], 2 * (1 - .5 ** np.arange(6)))
    np.testing.assert_allclose(jac['f'], .2 + .1 * np.arange(6), atol=1e-7)
    # Rematerialization changes memory use, not the derivative.
    remat = jax.jacrev(lambda p: assembly.unroll_muzero_predictions(
        LinearModel(), p, batch, remat=True)[1][0])(params)
    for key in jac:
        np.testing.assert_allclose(jac[key], remat[key])


def test_masked_root_does_not_block_future_value_gradients(assembly):
    batch = MuZeroTrainingSample(observation=jnp.zeros((1, 1, 1)),
        action=jnp.zeros((1, 5), jnp.int32), action_weights=jnp.zeros((1, 6, 2)),
        reward=jnp.zeros((1, 6)), transition_reward=jnp.zeros((1, 5)),
        policy_mask=jnp.zeros((1, 6), bool), reward_mask=jnp.zeros((1, 5), bool),
        value_mask=jnp.array([[False, False, False, False, False, True]]))
    params = dict(h=jnp.array(.2), g=jnp.array(.1), f=jnp.array(1.))
    def loss(p, b):
        return assembly.muzero_loss(LinearModel(), p, b, value_only=True)[0]
    value, gradient = jax.value_and_grad(loss)(params, batch)
    np.testing.assert_allclose(value, .7 ** 2, atol=1e-7)
    np.testing.assert_allclose(gradient['h'], 1.4 / 32, atol=1e-7)
    np.testing.assert_allclose(gradient['g'], 1.4 * 1.9375, atol=1e-6)
    changed = batch.replace(reward=batch.reward.at[:, :5].set(1000.))
    np.testing.assert_allclose(loss(params, changed), value)
