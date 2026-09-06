"""Recomputation preserves masked MuZero losses, gradients, and updates."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from nanoalphazero.research.muzero.learning import loss, train_step
from nanoalphazero.research.muzero.spatial import SpatialMuZero


def test_recomputation_preserves_spatial_unroll_gradients_and_optimizer_update():
    model = SpatialMuZero(16, "hexnoswap_4x4", 8, 1)
    observation = jax.random.bernoulli(jax.random.PRNGKey(0), shape=(2, 4, 4, 4))
    params = model.init(jax.random.PRNGKey(1), observation, jnp.array([0, 1]))["params"]
    batch = dict(observation=observation, action=jnp.array([[0, 1, 2], [3, 4, 5]]),
                 policy=jnp.ones((2, 4, 16)) / 16,
                 value=jnp.array([[1., -1., 1., 0.], [-1., 1., 0., 0.]]),
                 reward=jnp.array([[0., 0., 1.], [0., 1., 0.]]),
                 policy_mask=jnp.array([[True, True, True, False], [True, True, False, False]]),
                 value_mask=jnp.ones((2, 4), bool), reward_mask=jnp.ones((2, 3), bool))
    outputs = []
    for remat in (False, True):
        evaluate = jax.jit(jax.value_and_grad(lambda p: loss(model, p, batch, remat=remat)[0]))
        value, gradient = evaluate(params)
        state = TrainState.create(apply_fn=model.apply, params=params, tx=optax.adamw(1e-4))
        updated, metrics = jax.jit(functools.partial(train_step, model, remat=remat))(state, batch)
        outputs.append((value, gradient, updated.params, updated.opt_state, metrics))
    for a, b in zip(jax.tree.leaves(outputs[0]), jax.tree.leaves(outputs[1])):
        np.testing.assert_allclose(a, b, atol=3e-6, rtol=3e-5)
    for component in ("representation", "dynamics", "prediction", "reward"):
        assert float(optax.global_norm(outputs[1][1][component])) > 0
