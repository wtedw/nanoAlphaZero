"""Model-only inspection of training checkpoints; never loads experience buffers."""

import json
from pathlib import Path

import jax
from safetensors import safe_open

from nanoalphazero.ckpt.inspection import inspect_checkpoint
from nanoalphazero.model import make_muzero_model


def load_model(path, *, sharding=None):
    """Return (model, params, saved_config), using the saved architecture."""
    header = inspect_checkpoint(path)
    config = header["config"]
    model, variables = make_muzero_model(config, jax.random.PRNGKey(0), sharding)
    template = variables["params"]
    prefix = "['runner_state'].model_ts.params" if header["mode"] == "full" else "['model_ts'].params"
    if Path(path).is_dir():
        from nanoalphazero.ckpt.compressed import load
        params, _ = load(path, template, field_prefix=prefix)
    else:
        expected, tree = jax.tree_util.tree_flatten_with_path(template)
        with safe_open(path, framework="numpy") as reader:
            paths = {p: str(i) for i, p in enumerate(json.loads(reader.metadata()["paths"]))}
            leaves = []
            for keys, example in expected:
                value = reader.get_tensor(paths[prefix + jax.tree_util.keystr(keys)])
                if value.shape != example.shape or value.dtype != example.dtype:
                    raise ValueError("Checkpoint model shape/dtype does not match saved configuration")
                leaves.append(jax.device_put(value, example.sharding))
        params = tree.unflatten(leaves)
    return model, params, config
