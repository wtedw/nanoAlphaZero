"""Production MuZero loss/update parity across activation checkpoint strategies."""

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from nanoalphazero.buffers import get_dummy_muzero_training_sample
from nanoalphazero.config import get_muzero_config
from nanoalphazero.model import make_muzero_model


def test_five_step_chess_checkpoint_strategies_preserve_updates():
    spec = importlib.util.spec_from_file_location(
        "checkpoint_equivalence_muzero", Path(__file__).resolve().parents[1] / "muzero.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = get_muzero_config(
        "chess", conv_width=8, conv_depth=1, katago_preset="b1c8nbt",
        game_obs_shape=(8, 8, 119), game_num_actions=4672,
        enable_sharding=False, muzero_unroll_steps=5, lr_warmup_steps=0,
    )
    dummy = get_dummy_muzero_training_sample(config)
    batch = jax.tree.map(lambda x: jnp.broadcast_to(x, (2, *x.shape)), dummy)
    keys = jax.random.split(jax.random.PRNGKey(23), 4)
    batch = batch.replace(
        observation=jax.random.normal(keys[0], (2, 1, 8, 8, 119)) * .1,
        board_bool=None, board_float=None,
        action=jax.random.randint(keys[1], (2, 5), 0, 4672),
        k_indices=jnp.broadcast_to(jnp.arange(128, dtype=jnp.uint16), (2, 6, 128)),
        action_weights=jax.nn.softmax(jax.random.normal(keys[2], (2, 6, 128))),
        reward=jax.random.uniform(keys[3], (2, 6), minval=-1, maxval=1),
        policy_mask=jnp.array([[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]], dtype=bool),
        value_mask=jnp.ones((2, 6), bool),
    )
    expected = None
    for blocks, unroll in ((True, True), (True, False), (False, False)):
        cfg = {**config, "muzero_remat_blocks": blocks, "muzero_remat_unroll": unroll}
        model, variables = make_muzero_model(cfg, jax.random.PRNGKey(5))
        train_fn, state = module.make_muzero_train(cfg, model, variables)
        loss, grads = jax.jit(jax.value_and_grad(
            lambda p: module.muzero_loss(model, p, batch, remat=unroll, value_only=True)[0]
        ))(state.params)
        compiled_train = jax.jit(train_fn)
        updated = state
        for _ in range(3):
            updated, metrics = compiled_train(updated, batch, jnp.bool_(False))
        actual = (variables["params"], loss, grads, updated.params, updated.opt_state, metrics)
        if expected is None:
            expected = jax.device_get(actual)
        else:
            assert jax.tree.structure(actual) == jax.tree.structure(expected)
            for x, y in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
                np.testing.assert_allclose(x, y, rtol=3e-4, atol=3e-6)
        assert int(updated.n_updates) == 3
