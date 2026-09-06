"""Held-out diagnostics distinguish exploratory roots and exclude padding."""

import jax
import jax.numpy as jnp
import numpy as np

from nanoalphazero.research.muzero.evaluation import heldout_metrics
from nanoalphazero.research.muzero.model import MuZero


def test_heldout_root_strata_exclude_truncated_boundary_and_padding():
    model = MuZero(3, 8, 1)
    observation = jnp.ones((2, 1, 4))
    params = model.init(jax.random.PRNGKey(1), observation[:, 0], jnp.zeros(2, jnp.int32))["params"]
    params = jax.tree.map(jnp.zeros_like, params)  # All predicted values are zero.
    episodes = dict(observation=observation, action=jnp.zeros((2, 1), jnp.int32),
                    reward=jnp.zeros((2, 1)), policy=jnp.ones((2, 1, 3)) / 3,
                    value=jnp.array([[.2], [-.8]]), length=jnp.ones(2, jnp.int32),
                    terminal=jnp.zeros(2, bool), bootstrap=jnp.ones(2),
                    exploration=jnp.array([[True], [False]]))
    result = heldout_metrics(model, params, episodes, jax.random.PRNGKey(2), unrolls=(3,))
    for label, expected in (("exploration", .04), ("nonexploration", .64)):
        prefix = f"heldout/unroll3/{label}_root"
        assert result[f"{prefix}_count"] == 1
        np.testing.assert_allclose(result[f"{prefix}_real_value_mse"], expected, atol=1e-7)
        np.testing.assert_allclose(result[f"{prefix}_k0_value_mse"], expected, atol=1e-7)
    np.testing.assert_allclose(result["heldout/unroll3/real_value_mse"], .34, atol=1e-7)
