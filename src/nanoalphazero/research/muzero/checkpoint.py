"""Versioned MuZero snapshots including optimizer, replay, RNG and full config."""

import json
import os
from pathlib import Path
import tempfile

from flax import serialization
from flax.traverse_util import flatten_dict, unflatten_dict, empty_node
import jax
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

from nanoalphazero.checkpoint import _encode_path_segment, _decode_path_segment


def metadata(path):
    with safe_open(str(path), framework="flax") as reader:
        meta = reader.metadata()
    if meta.get("format") != "nanoalphazero.research.muzero.v1":
        raise ValueError("Not a MuZero v1 checkpoint")
    return json.loads(meta["config"])


def save(path, state, config):
    path = Path(path)
    if path.suffix != ".safetensors":
        raise ValueError("Expected .safetensors checkpoint")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        flat = flatten_dict(serialization.to_state_dict(state), keep_empty_nodes=True)
        tensors, empty = {}, []
        for keys, value in flat.items():
            name = "/".join(_encode_path_segment(k) for k in keys)
            if value is empty_node:
                empty.append(name)
            else:
                tensors[name] = np.array(jax.device_get(value), copy=True, order="C")
        save_file(tensors, temporary,
                  metadata={"format": "nanoalphazero.research.muzero.v1",
                            "empty_nodes": json.dumps(empty),
                            "config": json.dumps(config, sort_keys=True, allow_nan=False)})
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load(path, template):
    config = metadata(path)
    with safe_open(str(path), framework="flax") as reader:
        flat = {name: reader.get_tensor(name) for name in reader.keys()}
        flat.update({name: empty_node for name in json.loads(reader.metadata()["empty_nodes"])})
    restored = unflatten_dict({tuple(_decode_path_segment(k) for k in name.split("/")): value
                               for name, value in flat.items()})
    return serialization.from_state_dict(template, restored), config


def load_params(path):
    """Load inference parameters without allocating the saved replay buffer."""
    config = metadata(path)
    with safe_open(str(path), framework="flax") as reader:
        flat = {tuple(_decode_path_segment(k) for k in name.split("/")[2:]): reader.get_tensor(name)
                for name in reader.keys() if name.startswith("train/params/")}
    if not flat:
        raise ValueError("Checkpoint has no training parameters")
    return unflatten_dict(flat), config
