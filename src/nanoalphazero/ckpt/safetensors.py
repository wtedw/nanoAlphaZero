"""Existing MuZero safetensors format; compressed storage is separate."""

import json
import os
import tempfile

import jax
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file as save_safetensors_file


def _require_safetensors_path(path):
    if not path.endswith(".safetensors"):
        raise ValueError("Checkpoint path must end in .safetensors")


def save(path, state, config):
    """Save persistent game/buffer state as well as parameters and optimizer.

    This format is intentionally distinct from the bounded-episode research
    snapshots: their buffer layouts and continuation semantics are different.
    """
    from pathlib import Path
    path = Path(path)
    _require_safetensors_path(str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        # JAX trees cover Flax train states, Chex buffers, and PGX dataclasses.
        # Flax serialization alone does not register all three kinds of state.
        leaves, _ = jax.tree_util.tree_flatten_with_path(state)
        tensors, paths, key_impls = {}, [], {}
        for index, (keys, value) in enumerate(leaves):
            name = str(index)
            paths.append(jax.tree_util.keystr(keys))
            if hasattr(value, "dtype") and jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key):
                key_impls[name] = str(jax.random.key_impl(value))
                value = jax.random.key_data(value)
            tensors[name] = np.array(jax.device_get(value), copy=True, order="C")
        save_safetensors_file(tensors, temporary, metadata={
            "format": ("nanoalphazero.muzero.persistent.v4" if "runner_state" in state
                       else "nanoalphazero.muzero.compact.v1"),
            "paths": json.dumps(paths), "key_impls": json.dumps(key_impls),
            "config": json.dumps(config, sort_keys=True, allow_nan=False),
        })
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load(path, template):
    expected, tree = jax.tree_util.tree_flatten_with_path(template)
    expected_format = ("nanoalphazero.muzero.persistent.v4" if "runner_state" in template
                       else "nanoalphazero.muzero.compact.v1")
    with safe_open(str(path), framework="flax") as reader:
        meta = reader.metadata()
        if meta.get("format") != expected_format:
            raise ValueError(f"Expected {expected_format}; checkpoint mode or pipeline is incompatible")
        if json.loads(meta["paths"]) != [jax.tree_util.keystr(keys) for keys, _ in expected]:
            raise ValueError("MuZero checkpoint state structure does not match the template")
        if set(reader.keys()) != {str(i) for i in range(len(expected))}:
            raise ValueError("MuZero checkpoint has missing or unexpected tensors")
        key_impls = json.loads(meta["key_impls"])
        leaves = []
        for index, (_, example) in enumerate(expected):
            name = str(index)
            value = reader.get_tensor(name)
            if name in key_impls:
                value = jax.random.wrap_key_data(value, impl=key_impls[name])
            expected_dtype = example.dtype if hasattr(example, "dtype") else np.asarray(example).dtype
            if value.shape != np.shape(example) or value.dtype != expected_dtype:
                raise ValueError(f"MuZero checkpoint tensor {name} has an incompatible shape or dtype")
            leaves.append(value)
    config = json.loads(meta["config"])
    from nanoalphazero.muzero_config import normalize_exploration_config
    config = normalize_exploration_config(config)
    return tree.unflatten(leaves), config
