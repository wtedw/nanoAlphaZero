"""Experimental temporal consistency: targets, gradients, and persistent state."""

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nanoalphazero.buffers import (
    gather_muzero_training_samples, get_dummy_muzero_selfplay_output,
    get_dummy_muzero_training_sample, estimate_muzero_storage,
)
from nanoalphazero.config import get_muzero_config
from nanoalphazero.core import make_env
from nanoalphazero.model import make_muzero_model


@pytest.fixture(scope="module")
def runner():
    path = Path(__file__).resolve().parents[1] / "muzero.py"
    spec = importlib.util.spec_from_file_location("muzero_consistency_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config(env="hex5", **overrides):
    cfg = get_muzero_config(
        env, conv_width=8, conv_depth=1, selfplay_batch_size=4, train_batch_size=4,
        selfplay_buffer_consume_size=4, selfplay_buffer_max_len=128,
        replay_buffer_max_len=8, cycle_n_selfplay=20, cycle_n_train=1,
        lr_warmup_steps=0, replay_buffer_warmup_steps=0,
        mcts_num_root_considered=2, mcts_num_survivors=1, num_exploratory_moves=0,
        enable_sharding=False, **overrides,
    )
    env = make_env(cfg)
    return {**cfg, "game_obs_shape": list(env.obs_shape), "game_num_actions": env.num_actions}


def assert_trees_equal(a, b):
    assert jax.tree.structure(a) == jax.tree.structure(b)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        np.testing.assert_allclose(x, y, rtol=1e-6, atol=1e-6)


def test_flag_and_default_storage():
    from nanoalphazero.cli import parse_muzero_args

    assert parse_muzero_args([]).exp_self_consistency is None
    assert parse_muzero_args(["--exp-self-consistency"]).exp_self_consistency
    assert parse_muzero_args(["--no-exp-self-consistency"]).exp_self_consistency is False
    assert parse_muzero_args(["--self-consistency-weight", "0.5"]).self_consistency_weight == .5
    cfg = config()
    off = get_dummy_muzero_training_sample(cfg)
    assert off.observation.shape == (1, *cfg["game_obs_shape"])
    assert "exp_self_consistency" not in cfg
    on_cfg = {**cfg, "exp_self_consistency": True}
    on = get_dummy_muzero_training_sample(on_cfg)
    assert on.observation.shape == (6, *cfg["game_obs_shape"])
    assert on.observation.dtype == get_dummy_muzero_selfplay_output(cfg).observation.dtype
    assert off.observation.dtype == on.observation.dtype
    assert not any(name.startswith("consistency_") for name in vars(on))
    old_storage = estimate_muzero_storage(cfg)
    new_storage = estimate_muzero_storage(on_cfg)
    assert old_storage["selfplay"] == new_storage["selfplay"]
    assert old_storage["replay"]["bytes"] < new_storage["replay"]["bytes"]


@pytest.mark.parametrize("legacy_key", ["exp_train_on_exploration", "exp_train_on_exploratory_unroll_states"])
def test_checkpoint_observation_layout_version(tmp_path, legacy_key):
    import json
    from safetensors import safe_open
    from safetensors.numpy import save_file
    from nanoalphazero.checkpoint import save_muzero_checkpoint, load_muzero_checkpoint

    full = {"runner_state": {"observation": jnp.ones((2, 1, 3), jnp.bool_)}, "cycle": jnp.int32(0)}
    path = tmp_path / "full.safetensors"
    save_muzero_checkpoint(path, full, {legacy_key: False})
    with safe_open(str(path), framework="numpy") as reader:
        metadata = reader.metadata()
        assert metadata["format"] == "nanoalphazero.muzero.persistent.v4"
        tensors = {key: reader.get_tensor(key) for key in reader.keys()}
    restored, restored_config = load_muzero_checkpoint(path, full)
    assert restored_config == {"exp_train_on_exploit_unroll_states": True}
    assert_trees_equal(restored, full)
    # Even matching tensor paths must not let an old replay schema through.
    for version in (1, 2, 3):
        metadata["format"] = f"nanoalphazero.muzero.persistent.v{version}"
        save_file(tensors, str(path), metadata=metadata)
        with pytest.raises(ValueError, match="persistent.v4"):
            load_muzero_checkpoint(path, full)

    # Existing compact v1 contains no replay observations and remains readable.
    compact = {"model_ts": jnp.float32(1), "rng": jax.random.PRNGKey(0), "cycle": jnp.int32(2)}
    leaves, _ = jax.tree_util.tree_flatten_with_path(compact)
    save_file({str(i): np.asarray(value) for i, (_, value) in enumerate(leaves)}, str(path), metadata={
        "format": "nanoalphazero.muzero.compact.v1",
        "paths": json.dumps([jax.tree_util.keystr(keys) for keys, _ in leaves]),
        "key_impls": "{}", "config": json.dumps({legacy_key: True}),
    })
    restored, restored_config = load_muzero_checkpoint(path, compact)
    assert restored_config == {"exp_train_on_exploit_unroll_states": False}
    assert_trees_equal(restored, compact)


@pytest.mark.parametrize("env", ["hex5", "chess"])
def test_future_targets_wrap_and_exclude_next_game(env):
    cfg = config(env)
    dummy = get_dummy_muzero_selfplay_output(cfg)
    # Chronological steps 4..7 wrap around storage slots 4,0,1,2. Step 7 is
    # the final action; slot 3 already contains an unrelated game's board.
    times = jnp.array([[5, 6, 7, 8, 4]], jnp.uint32)
    exp = jax.tree.map(lambda x: jnp.broadcast_to(x, (1, 5, *x.shape)), dummy)
    exp = exp.replace(
        row_id=jnp.ones((1, 5), jnp.uint32), global_step_id=times,
        ep_step=(times - 4).astype(jnp.int16), ep_termination_step=jnp.full((1, 5), 3, jnp.int16),
        game_id=jnp.where(times == 8, 2, 1).astype(jnp.uint32),
        is_valid_sample=jnp.ones((1, 5), bool), is_from_selfplay=jnp.ones((1, 5), bool),
        is_pending_reward_i8=jnp.zeros((1, 5), jnp.int8),
    )
    field = "observation" if env == "hex5" else "board_bool"
    shape = getattr(exp, field).shape
    boards = jnp.broadcast_to(times.reshape((1, 5, *((1,) * (len(shape) - 2)))), shape)
    exp = exp.replace(**{field: boards.astype(jnp.uint8)})
    starts = jax.tree.map(lambda x: x[:, jnp.array([4, 0])], exp)
    # The public gather receives one start per lane; duplicate the lane's
    # selected records into two start rows to test overlapping starts.
    starts = jax.tree.map(lambda x: x.reshape((2, 1, *x.shape[2:])), starts)
    sample, missing = gather_muzero_training_samples(
        exp, starts, 5, cfg["game_num_actions"], self_consistency=True,
    )
    assert int(missing) == 0
    observations = getattr(sample, field)
    np.testing.assert_array_equal(observations[:, 0].reshape(2, -1)[:, 0], [4, 5])
    future = observations[:, 1:]
    np.testing.assert_array_equal(future.reshape(2, 5, -1)[:, :, 0], [[5, 6, 7, 0, 0], [6, 7, 0, 0, 0]])
    np.testing.assert_array_equal(sample.policy_mask[:, 1:], [[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])
    if env == "chess":
        assert sample.observation is None
        assert sample.board_float.shape[:2] == (2, 6)
    # Missing observations remain a buffer error, not a fabricated target.
    broken = exp.replace(global_step_id=times.at[0, 0].set(99))
    _, missing = gather_muzero_training_samples(
        broken, starts, 5, cfg["game_num_actions"], self_consistency=True,
    )
    assert int(missing) > 0


def test_cosine_mask_and_stop_gradient(runner):
    prediction = jnp.array([[1., 2.], [0., 0.]])
    target = jnp.array([[2., 1.], [3., 4.]])
    mask = jnp.array([True, False])
    loss, (gp, gt) = jax.value_and_grad(runner.masked_consistency_loss, argnums=(0, 1))(
        prediction, target, mask)
    assert float(loss) == pytest.approx(-.8)
    assert np.linalg.norm(gp[0]) > 0
    np.testing.assert_array_equal(gp[1], 0)
    np.testing.assert_array_equal(gt, 0)
    zero, grads = jax.value_and_grad(runner.masked_consistency_loss)(
        prediction, target, jnp.zeros(2, bool))
    assert float(zero) == 0
    np.testing.assert_array_equal(grads, 0)


@pytest.mark.parametrize("network", ["vector", "spatial"])
def test_consistency_gradients_and_disabled_equivalence(runner, network):
    cfg = config("ttt", muzero_network=network, muzero_unroll_steps=3)
    base_model, base_variables = make_muzero_model(cfg, jax.random.PRNGKey(3))
    enabled = {**cfg, "exp_self_consistency": True}
    model, variables = make_muzero_model(enabled, jax.random.PRNGKey(3))
    params = variables["params"]
    assert_trees_equal(base_variables["params"], {k: v for k, v in params.items() if k != "consistency_head"})
    sample = get_dummy_muzero_training_sample(enabled)
    batch = jax.tree.map(lambda x: jnp.broadcast_to(x, (2, *x.shape)), sample)
    batch = batch.replace(
        observation=jnp.concatenate((
            jax.random.normal(jax.random.PRNGKey(4), (2, 1, *cfg["game_obs_shape"])),
            jax.random.normal(jax.random.PRNGKey(5), (2, 3, *cfg["game_obs_shape"])),
        ), axis=1),
        action_weights=jnp.full((2, 4, 9), 1 / 9),
        policy_mask=jnp.ones((2, 4), bool), value_mask=jnp.ones((2, 4), bool),
        reward_mask=jnp.ones((2, 3), bool),
    )
    base_loss = runner.muzero_loss(base_model, base_variables["params"], batch)[0]
    off_loss = runner.muzero_loss(model, params, batch)[0]
    np.testing.assert_array_equal(base_loss, off_loss)
    loss_fn = lambda p: runner.muzero_loss(model, p, batch, self_consistency_weight=.5)
    (total, metrics), gradients = jax.value_and_grad(loss_fn, has_aux=True)(params)
    assert float(total) == pytest.approx(float(base_loss + .5 * metrics["loss_consistency"]))
    assert float(metrics["consistency/n_valid"]) == 6
    for group in ("representation", "dynamics", "consistency_head"):
        leaves = jax.tree.leaves(gradients[group])
        assert all(np.isfinite(x).all() for x in leaves)
        assert sum(float(jnp.sum(x * x)) for x in leaves) > 0
    consistency_grad = jax.grad(lambda p: loss_fn(p)[1]["loss_consistency"])(params)
    for group in ("representation", "dynamics", "consistency_head"):
        leaves = jax.tree.leaves(consistency_grad[group])
        assert all(np.isfinite(x).all() for x in leaves)
        assert sum(float(jnp.sum(x * x)) for x in leaves) > 0
    observation_grad = jax.grad(lambda obs: runner.muzero_loss(
        model, params, batch.replace(observation=batch.observation.at[:, 1:].set(obs)),
        self_consistency_weight=1.)[0]
    )(batch.observation[:, 1:])
    np.testing.assert_array_equal(observation_grad, 0)
    masked = batch.replace(policy_mask=batch.policy_mask.at[:, 1:].set(False))
    loss, metrics = runner.muzero_loss(model, params, masked, self_consistency_weight=1.)
    assert np.isfinite(loss)
    assert float(metrics["loss_consistency"]) == 0


def test_packed_future_observations_match_dense_loss(runner):
    from nanoalphazero.buffers import split_observation_vmap, combine_observation_vmap

    cfg = config("chess", exp_self_consistency=True, muzero_unroll_steps=2)
    model, variables = make_muzero_model(cfg, jax.random.PRNGKey(2))
    dummy = get_dummy_muzero_training_sample(cfg)
    batch = jax.tree.map(lambda x: jnp.broadcast_to(x, (2, *x.shape)), dummy)
    obs = jax.random.bernoulli(jax.random.PRNGKey(3), shape=(2 * 3, 8, 8, 119)).astype(jnp.float32)
    bits, floats = split_observation_vmap(obs)
    bits = bits.reshape((2, 3, *bits.shape[1:]))
    floats = floats.reshape((2, 3, *floats.shape[1:]))
    batch = batch.replace(
        board_bool=bits, board_float=floats,
        policy_mask=jnp.ones((2, 3), bool), value_mask=jnp.ones((2, 3), bool),
        reward_mask=jnp.ones((2, 2), bool),
    )
    decoded = combine_observation_vmap(bits.reshape(6, -1), floats.reshape(6, 8, 8, 2))
    dense = batch.replace(observation=decoded.reshape(2, 3, 8, 8, 119),
                          board_bool=None, board_float=None)
    loss = jax.jit(lambda b: runner.muzero_loss(
        model, variables["params"], b, remat=True, self_consistency_weight=1.0))
    packed_result = loss(batch)
    assert np.isfinite(packed_result[0])
    assert_trees_equal(packed_result, loss(dense))


@pytest.mark.parametrize("sharded", [False, True])
@pytest.mark.parametrize("layout", ["original", "features"])
def test_enabled_cycle_checkpoint_continuation(runner, tmp_path, sharded, layout):
    from nanoalphazero.checkpoint import save_muzero_checkpoint, load_muzero_checkpoint, muzero_checkpoint_payload

    cfg = config(exp_self_consistency=True, buffer_layout=layout)
    if sharded:
        if jax.device_count() != 4:
            pytest.skip("Requires four devices")
        cfg["enable_sharding"] = True
    algorithm = runner.make_muzero(cfg, jax.random.PRNGKey(0))
    state, metrics = algorithm.run_fn(algorithm.runner_state, True)
    state, metrics = algorithm.run_fn(state, False)
    assert float(metrics["selfplay_buffer/error_count"]) == 0
    assert int(state.model_ts.n_updates) == 1
    assert float(metrics["consistency/n_valid"]) > 0
    assert all(np.isfinite(v) for v in metrics.values())
    payload = muzero_checkpoint_payload(state, 2)
    path = tmp_path / "consistency.safetensors"
    save_muzero_checkpoint(path, payload, algorithm.config)
    loaded, restored_config = load_muzero_checkpoint(path, payload)
    assert restored_config["exp_self_consistency"]
    assert restored_config["buffer_layout"] == layout
    if layout == "features":
        assert state.replay_buffer_state.experience.observation.ndim == 3
    restored = jax.device_put(loaded["runner_state"], jax.tree.map(lambda x: x.sharding, state))
    if sharded:
        assert len(state.replay_buffer_state.experience.observation.addressable_shards) == 4
    expected, _ = algorithm.run_fn(state, False)
    actual, _ = algorithm.run_fn(restored, False)
    assert_trees_equal(expected, actual)
