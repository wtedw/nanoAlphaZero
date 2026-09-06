"""Block recomputation must preserve production parameters and MuZero updates."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax.training.train_state import TrainState

from nanoalphazero.model import KataGoTrunk
from nanoalphazero.research.muzero.checkpoint import load, save
from nanoalphazero.research.muzero.learning import train_step
from nanoalphazero.research.muzero.model import build_model
from nanoalphazero.research.muzero.remat import RematerializedTrunk


def assert_trees_close(left, right, *, exact=False):
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right)):
        np.testing.assert_allclose(a, b, atol=0 if exact else 3e-6,
                                   rtol=0 if exact else 3e-5)


@pytest.mark.parametrize("use_rvgl", [False, True])
def test_trunk_parameter_paths_masked_forward_and_gradients(use_rvgl):
    config = dict(c_trunk=8, c_mid=4, c_gpool=2, block_gpool=(False, True),
                  activation="mish", use_rvgl=use_rvgl)
    original, recomputed = KataGoTrunk(**config), RematerializedTrunk(**config)
    spatial = jax.random.normal(jax.random.PRNGKey(0), (2, 3, 3, 4))
    global_features = jnp.ones((2, 2))
    mask = jnp.ones((2, 3, 3, 1)).at[:, 0, 0].set(0)
    args = (spatial, global_features, mask)
    params = original.init(jax.random.PRNGKey(1), *args)
    assert_trees_close(params, recomputed.init(jax.random.PRNGKey(1), *args), exact=True)
    outputs = []
    for model in (original, recomputed):
        def objective(p, x):
            out, _, _ = model.apply(p, x, global_features, mask)
            return jnp.mean(out ** 2)
        outputs.append(jax.jit(jax.value_and_grad(objective, argnums=(0, 1)))(params, spatial))
    assert_trees_close(*outputs)


def test_block_remat_unrolled_update_and_checkpoint_metadata(tmp_path):
    config = dict(network="spatial", num_actions=9, env_id="hexnoswap_3x3",
                  width=8, depth=4, activation="mish", use_rvgl=True)
    original = build_model(config)
    remat_config = dict(config, remat_blocks=True, remat_unroll=True)
    recomputed = build_model(remat_config)
    observation = jax.random.bernoulli(jax.random.PRNGKey(2), shape=(2, 3, 3, 4))
    action = jnp.array([0, 1])
    params = original.init(jax.random.PRNGKey(3), observation, action)["params"]
    assert_trees_close(params, recomputed.init(jax.random.PRNGKey(3), observation, action)["params"], exact=True)
    batch = dict(observation=observation, action=jnp.array([[0, 1], [3, 4]]),
                 policy=jnp.ones((2, 3, 9)) / 9,
                 value=jnp.array([[1., -1., 0.], [-1., 0., 0.]]),
                 reward=jnp.array([[0., 1.], [1., 0.]]),
                 policy_mask=jnp.array([[True, True, False], [True, False, False]]),
                 value_mask=jnp.ones((2, 3), bool), reward_mask=jnp.ones((2, 2), bool))
    outputs = []
    for model in (original, recomputed):
        state = TrainState.create(apply_fn=model.apply, params=params, tx=optax.adamw(1e-4))
        updated, metrics = jax.jit(functools.partial(train_step, model, remat=True))(state, batch)
        outputs.append((updated.params, updated.opt_state, metrics))
    assert_trees_close(*outputs)
    for component in ("representation", "dynamics", "prediction", "reward"):
        delta = jax.tree.map(lambda a, b: a - b, params[component], outputs[1][0][component])
        assert float(optax.global_norm(delta)) > 0
    path = tmp_path / "block-remat.safetensors"
    save(path, {"params": outputs[1][0]}, remat_config)
    restored, saved_config = load(path, {"params": params})
    assert saved_config == remat_config
    loaded_model = build_model(saved_config)
    actual = loaded_model.apply(restored, observation, method=loaded_model.initial)
    expected = original.apply({"params": outputs[1][0]}, observation, method=original.initial)
    assert_trees_close(actual, expected)
