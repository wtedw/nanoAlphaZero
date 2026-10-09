"""Tests for the standalone lossless compression experiment, not production I/O."""

import importlib.util
import json
from pathlib import Path

import jax
import ml_dtypes
import numpy as np
import pytest


spec = importlib.util.spec_from_file_location(
    "compression_probe", Path(__file__).parents[1] / "scripts/probe_muzero_checkpoint_compression.py"
)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_bit_exact_restore_and_corruption(tmp_path):
    tree = {
        "float": np.array([0., -0., np.nan, np.inf, -np.inf, 1.25], np.float32),
        "packed": np.arange(256, dtype=np.uint8),
        "bf16": np.array([0., -0., 1.5, np.nan], ml_dtypes.bfloat16),
        "scalar": np.int32(123),
        "mask": np.array([True, False]),
        "empty": np.zeros((0, 3), dtype=np.uint16),
    }
    directory = tmp_path / "snapshot"
    result = probe.snapshot(tree, directory, chunk_bytes=13)
    restored = probe.read_snapshot(directory)
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        actual = restored[jax.tree_util.keystr(path)]
        assert actual.shape == leaf.shape
        assert actual.dtype == leaf.dtype
        assert actual.tobytes() == leaf.tobytes()
    assert result["raw_bytes"] == sum(v.nbytes for v in tree.values())
    manifest = directory / "manifest.json"
    content = json.loads(manifest.read_text())
    content["fields"][0]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="Corrupt field"):
        probe.read_snapshot(directory)


@pytest.mark.parametrize('repeat', [1, 3])
def test_streaming_shards_and_partial_chunks(tmp_path, monkeypatch, repeat):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / 'scripts'))
    import probe_muzero_stream_restore as stream
    devices = jax.devices()
    tree = {
        'float': np.arange(len(devices) * 21, dtype=np.float32).reshape(len(devices), 3, 7),
        'packed': np.arange(len(devices) * 13, dtype=np.uint8).reshape(len(devices), 13),
        'bf16': np.arange(len(devices) * 5, dtype=np.float32).astype(ml_dtypes.bfloat16).reshape(len(devices), 5),
        'scalar': np.int32(123),
    }
    directory = tmp_path / 'snapshot'
    probe.snapshot(tree, directory, chunk_bytes=32)
    arrays, report = stream.restore(directory, devices, repeat=repeat)
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        expected = np.concatenate([leaf] * repeat, axis=0) if leaf.ndim else leaf
        actual = np.asarray(arrays[jax.tree_util.keystr(path)])
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        assert actual.tobytes() == expected.tobytes()
    assert report['verified']
