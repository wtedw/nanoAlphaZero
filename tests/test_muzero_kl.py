"""Gumbel target-to-prediction KL, including unroll gradient equivalence."""

import jax
import jax.numpy as jnp
import numpy as np
import optax

from nanoalphazero.research.muzero import learning
from nanoalphazero.research.muzero.model import MuZero


def test_kl_matches_az_and_handles_zero_targets_without_underflow():
    logits = jnp.array([[1., 2., -1.], [0., 0., 0.]])
    targets = jnp.array([[.75, .25, 0.], [0., 0., 1.]])
    expected = jnp.sum(jax.scipy.special.rel_entr(targets, jax.nn.softmax(logits)), -1)
    np.testing.assert_allclose(learning.policy_kl(logits, targets), expected, atol=2e-7)
    np.testing.assert_allclose(learning.policy_kl(jnp.log(jnp.array([[.25, .75]])), jnp.array([[.25, .75]])), 0., atol=2e-7)
    extreme = jnp.array([[-1000., 1000., 0.]])
    assert bool(jnp.isfinite(learning.policy_kl(extreme, targets[:1])).all())
    assert bool(jnp.isfinite(jax.grad(lambda x: learning.policy_kl(x, targets[:1]).sum())(extreme)).all())


def test_unrolled_kl_preserves_cross_entropy_gradients_and_masking(monkeypatch):
    model = MuZero(num_actions=3, width=8, depth=1)
    observation = jnp.ones((2, 4))
    params = model.init(jax.random.PRNGKey(2), observation, jnp.zeros(2, jnp.int32))["params"]
    target = jnp.broadcast_to(jnp.array([.7, .3, 0.]), (2, 3, 3))
    batch = dict(observation=observation, action=jnp.array([[0, 1], [1, 2]]),
                 policy=target, value=jnp.zeros((2, 3)), reward=jnp.zeros((2, 2)),
                 policy_mask=jnp.array([[True, True, True], [True, False, False]]),
                 value_mask=jnp.ones((2, 3), bool), reward_mask=jnp.ones((2, 2), bool))
    kl_loss, kl_grad = jax.value_and_grad(lambda p: learning.loss(model, p, batch)[0])(params)
    altered = {**batch, "policy": target.at[1, 1:].set(jnp.array([0., 0., 1.]))}
    np.testing.assert_allclose(learning.loss(model, params, altered)[0], kl_loss)
    monkeypatch.setattr(learning, "policy_kl", optax.softmax_cross_entropy)
    ce_loss, ce_grad = jax.value_and_grad(lambda p: learning.loss(model, p, batch)[0])(params)
    entropy = -(.7 * np.log(.7) + .3 * np.log(.3))
    np.testing.assert_allclose(ce_loss - kl_loss, entropy, atol=5e-7)
    for old, new in zip(jax.tree.leaves(ce_grad), jax.tree.leaves(kl_grad)):
        np.testing.assert_allclose(old, new, atol=1e-7, rtol=1e-6)
