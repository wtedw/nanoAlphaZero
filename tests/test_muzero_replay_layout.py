"""Reversible storage-layout experiment; CPU tests do not establish TPU savings."""

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nanoalphazero.buffers import get_dummy_muzero_training_sample, make_replay_buffer, make_muzero_replay_buffer
from nanoalphazero.config import get_muzero_config
from nanoalphazero.model import make_muzero_model


@pytest.fixture(scope='module')
def probe():
    path = Path(__file__).resolve().parents[1] / 'scripts/probe_muzero_replay_layout.py'
    spec = importlib.util.spec_from_file_location('replay_layout_probe', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config(consistency=True, sharded=False):
    cfg = get_muzero_config(
        'chess', enable_sharding=sharded, selfplay_batch_size=8,
        train_batch_size=8, selfplay_buffer_consume_size=8,
        replay_buffer_max_len=5, muzero_unroll_steps=5,
        conv_depth=1, conv_width=8, exp_self_consistency=consistency,
    )
    cfg.update(game_obs_shape=[8, 8, 119], game_num_actions=4672)
    return cfg


def varied_batch(dummy, length, offset):
    def values(x):
        shape = (8, length, *x.shape)
        data = jnp.arange(np.prod(shape), dtype=jnp.int32).reshape(shape) + offset
        if x.dtype == jnp.bool_:
            return data % 3 == 0
        if jnp.issubdtype(x.dtype, jnp.floating):
            return ((data % 19) / 20).astype(x.dtype)
        return (data % 127).astype(x.dtype)

    batch = jax.tree.map(values, dummy)
    return batch.replace(action_weights=jnp.full_like(batch.action_weights, 1 / 128))


def assert_equal(left, right):
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for x, y in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        assert x.dtype == y.dtype
        np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize('layout', ['features', 'matrix', 'production_features'])
@pytest.mark.parametrize('consistency', [False, True])
@pytest.mark.parametrize('sharded', [False, True])
def test_samples_rng_bytes_and_wrap_are_identical(probe, layout, consistency, sharded):
    cfg = config(consistency, sharded)
    sharding = None
    if sharded:
        assert jax.device_count() == 4, 'Run with four virtual CPU devices'
        sharding = jax.sharding.NamedSharding(
            jax.sharding.Mesh(np.array(jax.devices()), ('data',)), jax.sharding.PartitionSpec('data'))
    dummy = get_dummy_muzero_training_sample(cfg)
    baseline, base_state = make_replay_buffer(cfg, dummy, sharding)
    candidate, state = (make_muzero_replay_buffer({**cfg, 'buffer_layout': 'features'}, dummy, sharding)
                        if layout == 'production_features' else
                        probe.make_layout_replay_buffer(cfg, dummy, sharding, layout=layout))
    assert not candidate.can_sample(state)

    for step, length in enumerate((2, 3, 1, 2)):
        batch = varied_batch(dummy, length, 100 * step)
        if sharded:
            batch = jax.device_put(batch, sharding)
        base_state = baseline.add(base_state, batch)
        state = candidate.add(state, batch)
        assert_equal(base_state.current_index, state.current_index)
        assert_equal(base_state.is_full, state.is_full)
        assert_equal(baseline.can_sample(base_state), candidate.can_sample(state))
        # Entire circular storage is bit-for-bit identical under reshape, not
        # just the random subset selected by the next sample.
        restored = jax.tree.map(lambda flat, original: flat.reshape(original.shape),
                                state.experience, base_state.experience)
        assert_equal(restored, base_state.experience)
        for key in (jax.random.PRNGKey(step), jax.random.PRNGKey(step + 23)):
            expected = baseline.sample(base_state, key)
            actual = candidate.sample(state, key)
            assert_equal(actual, expected)
        if sharded:
            assert all(len(x.addressable_shards) == 4 for x in jax.tree.leaves(state.experience))
    assert all(x.ndim == (2 if layout == 'matrix' else 3) for x in jax.tree.leaves(state.experience))
    assert sum(x.nbytes for x in jax.tree.leaves(state)) == sum(x.nbytes for x in jax.tree.leaves(base_state))
    empty = candidate.init(dummy)
    assert_equal(empty.current_index, jnp.int32(0))
    assert not candidate.can_sample(empty)


@pytest.mark.parametrize('layout', ['features', 'matrix', 'production_features'])
def test_loss_and_gradients_unchanged(probe, layout):
    path = Path(__file__).resolve().parents[1] / 'muzero.py'
    spec = importlib.util.spec_from_file_location('layout_loss_runner', path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    cfg = config()
    dummy = get_dummy_muzero_training_sample(cfg)
    baseline, base_state = make_replay_buffer(cfg, dummy)
    candidate, state = (make_muzero_replay_buffer({**cfg, 'buffer_layout': 'features'}, dummy)
                        if layout == 'production_features' else
                        probe.make_layout_replay_buffer(cfg, dummy, layout=layout))
    batch = varied_batch(dummy, 2, 7)
    base_state = baseline.add(base_state, batch)
    state = candidate.add(state, batch)
    key = jax.random.PRNGKey(3)
    expected = jax.tree.map(lambda x: x[:, 0], baseline.sample(base_state, key).experience)
    actual = jax.tree.map(lambda x: x[:, 0], candidate.sample(state, key).experience)
    model, variables = make_muzero_model(cfg, jax.random.PRNGKey(0))
    objective = jax.jit(jax.value_and_grad(lambda params, sample: runner.muzero_loss(
        model, params, sample, remat=True, self_consistency_weight=1.0), has_aux=True))
    a = objective(variables['params'], actual)
    b = objective(variables['params'], expected)
    assert_equal(a, b)
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(a))
    assert any(np.any(x != 0) for x in jax.tree.leaves(a[1]))


def test_probe_refuses_busy_tpu_before_importing_runner(probe, monkeypatch):
    import subprocess
    monkeypatch.setattr(probe.Path, 'exists', lambda _: True)
    monkeypatch.setattr(probe.subprocess, 'run', lambda *args, **kwargs:
                        subprocess.CompletedProcess(args, 0, stdout='12345', stderr=''))
    with pytest.raises(RuntimeError, match='TPU occupied.*12345'):
        probe.check_tpu_ownership()


def test_buffer_layout_cli_and_default():
    from nanoalphazero.cli import parse_muzero_args

    assert get_muzero_config('hex4')['buffer_layout'] == 'original'
    assert parse_muzero_args(['--buffer-layout', 'features']).buffer_layout == 'features'
    assert parse_muzero_args(['--buffer-layout', 'original']).buffer_layout == 'original'
    with pytest.raises(SystemExit):
        parse_muzero_args(['--buffer-layout', 'matrix'])
