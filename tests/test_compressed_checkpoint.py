"""Lossless storage, failure publication, and real MuZero continuation."""

import json
import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nanoalphazero.ckpt import compressed
from nanoalphazero.ckpt import save_muzero_checkpoint, load_muzero_checkpoint


def test_public_api_and_legacy_imports():
    from nanoalphazero import ckpt, checkpoint
    initial = object()
    restored, cycle = ckpt.restore_checkpoint(None, state=initial, config={}, mode='full')
    assert restored is initial and cycle == 0
    assert checkpoint.save_muzero_checkpoint is ckpt.save_muzero_checkpoint
    assert checkpoint.load_muzero_checkpoint is ckpt.load_muzero_checkpoint


def test_partial_checkpoint_uses_compressed_storage(tmp_path):
    from types import SimpleNamespace
    from nanoalphazero import ckpt

    state = SimpleNamespace(model_ts={"params": jnp.arange(4)}, rng=jax.random.key(0))
    path = tmp_path / "partial.ckpt"
    ckpt.save_checkpoint(path, state=state, cycle=7, config={}, mode="partial")
    assert (path / "LATEST").is_file()
    expected = ckpt.muzero_checkpoint_payload(state, 7, False)
    restored, config = ckpt.load_muzero_checkpoint(path, expected)
    assert config == {}
    assert_exact(expected, restored)


@pytest.mark.parametrize("shape", [(), (0, 8), (3, 5, 11), (2, 3, 4, 9), (31,)])
def test_rectangular_blocks_cover_c_order_without_padding(shape, monkeypatch):
    monkeypatch.setattr(compressed, "CHUNK_BYTES", 32)
    original = np.arange(np.prod(shape), dtype=np.int32).reshape(shape)
    pieces = []
    for start, size in compressed._blocks(shape, original.dtype.itemsize):
        block = original[tuple(slice(a, a + n) for a, n in zip(start, size))]
        assert block.nbytes <= 32
        pieces.append(block.tobytes())
    assert b"".join(pieces) == original.tobytes()


def assert_exact(left, right):
    # Fresh algorithms reconstruct static callables (apply_fn/optimizer closures),
    # whose Python identities differ. Compare every persisted path and byte.
    assert [jax.tree_util.keystr(p) for p, _ in jax.tree_util.tree_flatten_with_path(left)[0]] == [
        jax.tree_util.keystr(p) for p, _ in jax.tree_util.tree_flatten_with_path(right)[0]]
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right)):
        assert a.sharding.is_equivalent_to(b.sharding, a.ndim), (a.sharding, b.sharding)
        if jax.dtypes.issubdtype(a.dtype, jax.dtypes.prng_key):
            a, b = jax.random.key_data(a), jax.random.key_data(b)
        assert a.dtype == b.dtype and a.shape == b.shape
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()


def test_shards_chunks_keys_and_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(compressed, "CHUNK_BYTES", 32)
    mesh = jax.sharding.Mesh(np.array(jax.devices()), ('data',))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('data'))
    tree = {'floats': jax.device_put(jnp.arange(84, dtype=jnp.float32).reshape(4, 21), sharding),
            'bf16': jnp.array([0., -0., jnp.nan, jnp.inf], jnp.bfloat16),
            'empty': jnp.empty((0,), jnp.uint16), 'rng': jax.random.key(99),
            'mask': jnp.array([True, False]), 'counter': jnp.int32(11)}
    path = tmp_path / 'state.ckpt'
    save_muzero_checkpoint(path, tree, {'test': True}, compression='zstd')
    restored, config = load_muzero_checkpoint(path, tree)
    assert_exact(tree, restored)
    descriptors = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.sharding), tree)
    restored_from_shapes, _ = load_muzero_checkpoint(path, descriptors)
    assert_exact(tree, restored_from_shapes)
    assert config == {'test': True}
    assert restored['floats'].sharding == sharding
    incompatible = dict(tree, floats=jax.device_put(tree['floats'], jax.devices()[0]))
    with pytest.raises(ValueError, match='shard geometry'):
        load_muzero_checkpoint(path, incompatible)
    save_muzero_checkpoint(path, tree, config, compression='zstd')
    assert len(list(path.glob('gen-*'))) == 1
    manifest_path = compressed._generation(path) / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    field = next(f for f in manifest['fields'] if f['shards'][0]['chunks'])
    field['shards'][0]['chunks'][0]['sha256'] = '0' * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Corrupt'):
        load_muzero_checkpoint(path, tree)


@pytest.mark.parametrize('stage', ['chunk', 'publication'])
def test_failed_save_keeps_previous(tmp_path, monkeypatch, stage):
    path = tmp_path / 'state.ckpt'
    tree = {'counter': jnp.int32(1)}
    compressed.save(path, tree, {}, metadata={"trackio": {"name": "original"}})
    previous = (path / 'LATEST').read_bytes()
    original = compressed._write

    def fail(destination, content):
        if (stage == 'chunk' and destination.suffix == '.zst') or (
            stage == 'publication' and destination.name == 'LATEST.tmp'
        ):
            raise OSError('injected disk failure')
        original(destination, content)

    monkeypatch.setattr(compressed, '_write', fail)
    with pytest.raises(OSError, match='injected'):
        compressed.save(path, {'counter': jnp.int32(2)}, {})
    assert (path / 'LATEST').read_bytes() == previous
    assert compressed.metadata(path) == {"trackio": {"name": "original"}}
    restored, _ = compressed.load(path, tree)
    assert_exact(tree, restored)
    assert len(list(path.glob('gen-*'))) == 1


@pytest.mark.parametrize('env', ['ttt', 'chess'])
def test_full_muzero_next_cycle(tmp_path, env):
    from nanoalphazero.config import get_muzero_config
    spec = importlib.util.spec_from_file_location('compressed_muzero', Path(__file__).parents[1] / 'muzero.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = get_muzero_config(
        'ttt', enable_sharding=True, selfplay_batch_size=4, train_batch_size=4,
        selfplay_buffer_consume_size=4, selfplay_buffer_max_len=16,
        replay_buffer_max_len=4, conv_width=8, conv_depth=1,
        muzero_network='vector', muzero_unroll_steps=3,
        cycle_n_selfplay=10, cycle_n_train=1, lr_warmup_steps=0,
        mcts_num_root_considered=2, mcts_num_survivors=1,
        num_exploratory_moves=0, replay_buffer_warmup_steps=10,
    )
    cycles = 3
    if env == 'chess':
        config = get_muzero_config(
            'chess', enable_sharding=True, selfplay_batch_size=4, train_batch_size=4,
            selfplay_buffer_consume_size=4, selfplay_buffer_max_len=1024,
            selfplay_buffer_min_len=1, replay_buffer_max_len=4, buffer_layout='features',
            conv_width=8, conv_depth=1, muzero_unroll_steps=2,
            cycle_n_selfplay=512, cycle_n_train=1, lr_warmup_steps=0,
            mcts_num_root_considered=2, mcts_num_survivors=1,
            num_exploratory_moves=0, replay_buffer_warmup_steps=0,
        )
        cycles = 1
    config.update(diagnostic_period=0, eval_period=0,
                  num_iters=(cycles + 1) * config['cycle_n_selfplay'])
    algorithm = module.make_muzero(config, jax.random.PRNGKey(0))
    state = algorithm.runner_state
    for _ in range(cycles):
        state, (metrics, _) = algorithm.run_fn(state, jnp.bool_(False))
        assert int(metrics['selfplay_buffer/error_count']) == 0
    assert int(state.model_ts.n_updates) == cycles
    payload = {'runner_state': state, 'cycle': jnp.int32(cycles)}
    path = tmp_path / 'full.ckpt'
    save_muzero_checkpoint(path, payload, algorithm.config, compression='zstd')
    restored, config_back = load_muzero_checkpoint(path, payload)
    assert config_back == algorithm.config
    assert_exact(payload, restored)
    from nanoalphazero.ckpt import muzero_checkpoint_payload
    compact = muzero_checkpoint_payload(state, cycles, False)
    save_muzero_checkpoint(tmp_path / 'compact.ckpt', compact, algorithm.config, compression='zstd')
    compact_back, _ = load_muzero_checkpoint(tmp_path / 'compact.ckpt', compact)
    assert_exact(compact, compact_back)
    # Same sampled batches under identical RNG, then identical actual updates.
    key = jax.random.PRNGKey(77)
    assert_exact(algorithm.replay_buffer.sample(state.replay_buffer_state, key),
                 algorithm.replay_buffer.sample(restored['runner_state'].replay_buffer_state, key))
    next_state, next_metrics = algorithm.run_fn(state, jnp.bool_(False))
    restored_state, restored_metrics = algorithm.run_fn(restored['runner_state'], jnp.bool_(False))
    assert int(next_state.model_ts.n_updates) == cycles + 1
    assert_exact(next_state, restored_state)
    assert_exact(next_metrics, restored_metrics)
    if env == 'ttt':
        from nanoalphazero import ckpt
        fresh = module.make_muzero(config, jax.random.PRNGKey(5))
        resumed, start_cycle = ckpt.restore_checkpoint(path, state=fresh.runner_state, config=fresh.config, mode='full')
        assert start_cycle == cycles
        resumed, _ = fresh.run_fn(resumed, jnp.bool_(False))
        assert_exact(next_state, resumed)
