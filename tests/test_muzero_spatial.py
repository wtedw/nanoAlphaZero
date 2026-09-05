"""Production spatial component reuse retains MuZero's latent-only contract."""

import jax
import jax.numpy as jnp
import numpy as np
import optax

from nanoalphazero.research.muzero.spatial import SpatialMuZero, action_planes
from nanoalphazero.research.muzero.learning import loss
from nanoalphazero.research.muzero.search import latent_search


def test_action_encodings_are_state_independent():
    board = action_planes(jnp.array([5]), 4, 4, 16, "hexnoswap_4x4")
    assert board.shape == (1, 4, 4, 1)
    assert board[0, 1, 1, 0] == 1
    passing = action_planes(jnp.array([9]), 3, 3, 10, "go_3x3")
    assert jnp.sum(passing[..., 0]) == 0
    assert jnp.all(passing[..., 1] == 1)
    column = action_planes(jnp.array([2]), 6, 7, 7, "connect_four")
    assert jnp.all(column[0, :, 2, 0] == 1)
    chess = action_planes(jnp.array([123]), 8, 8, 4672, "chess")
    unrotated = jnp.rot90(chess, k=-1, axes=(1, 2)).reshape(1, -1)
    np.testing.assert_array_equal(unrotated, jax.nn.one_hot(jnp.array([123]), 4672))


def test_spatial_gradients_and_latent_search():
    model = SpatialMuZero(16, "hexnoswap_4x4", 8, 1)
    obs = jax.random.bernoulli(jax.random.PRNGKey(0), shape=(2, 4, 4, 4))
    params = model.init(jax.random.PRNGKey(1), obs, jnp.array([0, 1]))["params"]
    batch = {"observation": obs, "action": jnp.array([[0, 1], [2, 3]]),
             "policy": jnp.full((2, 3, 16), 1 / 16), "value": jnp.ones((2, 3)),
             "reward": jnp.array([[0., 1.], [0., 1.]]), "policy_mask": jnp.ones((2, 3), bool),
             "value_mask": jnp.ones((2, 3), bool), "reward_mask": jnp.ones((2, 2), bool)}
    total, grad = jax.jit(jax.value_and_grad(lambda p: loss(model, p, batch)[0]))(params)
    assert np.isfinite(float(total))
    for component in ("representation", "dynamics", "prediction", "reward"):
        norm = float(optax.global_norm(grad[component]))
        assert np.isfinite(norm) and norm > 0
    legal = jnp.ones((2, 16), bool).at[:, 0].set(False)
    out = jax.jit(lambda p: latent_search(model, p, obs, legal, jax.random.PRNGKey(2), roots=4, survivors=2))(params)
    assert jnp.all(out.action_weights[:, 0] == 0)
    assert jnp.all(out.action != 0)
