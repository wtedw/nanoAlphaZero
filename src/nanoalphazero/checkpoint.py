"""Safetensors checkpoint paths, metadata, saving, and loading."""

import json
import os
import tempfile
from typing import Any, Optional

import flax.traverse_util
import jax
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file as save_safetensors_file


# =============================================================================
# Checkpointing
# =============================================================================
def default_ckpt_path(env_name: str) -> str:
    """Default on-disk location for a saved alphazero checkpoint."""
    return os.path.join("artifacts", f"alphazero_{env_name}.safetensors")


_CHECKPOINT_FORMAT = "nanoalphazero.flax.params"
_CHECKPOINT_FORMAT_VERSION = "2"
_TREE_PATH_ENCODING = "json-pointer-segments-v1"
_CHECKPOINT_CONFIG_KEYS = (
    "katago_preset",
    "katago_activation",
    "katago_use_rvgl",
    "use_wdl",
    "env_id",
    "game_obs_shape",
    "game_num_actions",
)


def checkpoint_model_config(config: dict) -> dict:
    """Return the resolved architecture/game settings needed to load params."""
    resolved = {
        key: config[key] for key in _CHECKPOINT_CONFIG_KEYS if key in config
    }
    resolved.update(
        {
            "katago_preset": config.get(
                "katago_preset",
                f"b{config['conv_depth']}c{config['conv_width']}nbt",
            ),
            "katago_activation": config.get("katago_activation", "mish"),
            "katago_use_rvgl": config.get("katago_use_rvgl", True),
            "use_wdl": config.get("use_wdl", True),
        }
    )
    return resolved


def apply_checkpoint_model_config(config: dict, model_config: Optional[dict]) -> dict:
    """Overlay only model/game compatibility settings from a checkpoint."""
    if not model_config:
        return config
    updated = config.copy()
    updated.update(
        {
            key: value
            for key, value in model_config.items()
            if key in _CHECKPOINT_CONFIG_KEYS
        }
    )
    if updated.get("game_obs_shape") is not None:
        updated["game_obs_shape"] = tuple(updated["game_obs_shape"])
    return updated


def _encode_path_segment(segment: Any) -> str:
    return str(segment).replace("~", "~0").replace("/", "~1")


def _decode_path_segment(segment: str) -> str:
    return segment.replace("~1", "/").replace("~0", "~")


def _flatten_checkpoint_params(params) -> dict[str, np.ndarray]:
    flat = flax.traverse_util.flatten_dict(params)
    encoded = {}
    for path, value in flat.items():
        name = "/".join(_encode_path_segment(segment) for segment in path)
        if not name or name in encoded:
            raise ValueError(f"Duplicate or empty encoded parameter path: {path!r}")
        encoded[name] = np.ascontiguousarray(jax.device_get(value))
    return dict(sorted(encoded.items()))


def _unflatten_checkpoint_params(flat_params: dict[str, jax.Array]) -> dict:
    decoded = {}
    for name, value in flat_params.items():
        path = tuple(_decode_path_segment(segment) for segment in name.split("/"))
        if not name or path in decoded:
            raise ValueError(f"Duplicate or empty parameter path in checkpoint: {name!r}")
        decoded[path] = value
    return flax.traverse_util.unflatten_dict(decoded)


def _require_safetensors_path(path: str) -> None:
    if not path.endswith(".safetensors"):
        raise ValueError("Checkpoint path must end in .safetensors")


def save_checkpoint(params, config: dict, path: str) -> None:
    """Atomically save params and resolved model metadata as Safetensors."""
    _require_safetensors_path(path)
    directory = os.path.dirname(path)
    target_dir = directory or "."
    os.makedirs(target_dir, exist_ok=True)
    metadata = {
        "format": _CHECKPOINT_FORMAT,
        "format_version": _CHECKPOINT_FORMAT_VERSION,
        "tree_path_encoding": _TREE_PATH_ENCODING,
        "model_config": json.dumps(
            checkpoint_model_config(config),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
    }
    fd, temporary_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=target_dir
    )
    os.close(fd)
    try:
        save_safetensors_file(
            _flatten_checkpoint_params(params), temporary_path, metadata=metadata
        )
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
    print(f"✅ Saved model params to {path}")


def load_checkpoint(path: str):
    """Load Safetensors params and return `(params, model_config)`."""
    _require_safetensors_path(path)
    if not os.path.exists(path):
        raise SystemExit(
            f"No checkpoint found at {path}. Train a model first, or point "
            f"--load at an existing checkpoint."
        )
    with safe_open(path, framework="flax") as checkpoint:
        metadata = checkpoint.metadata() or {}
        if metadata.get("format") != _CHECKPOINT_FORMAT:
            raise ValueError(f"Unsupported checkpoint format in {path}")
        if metadata.get("format_version") != _CHECKPOINT_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported checkpoint format version "
                f"{metadata.get('format_version')!r} in {path}"
            )
        if metadata.get("tree_path_encoding") != _TREE_PATH_ENCODING:
            raise ValueError(f"Unsupported parameter path encoding in {path}")
        try:
            model_config = json.loads(metadata["model_config"])
        except (KeyError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid or missing model_config in {path}") from exc
        flat_params = {
            name: checkpoint.get_tensor(name) for name in checkpoint.keys()
        }
    params = _unflatten_checkpoint_params(flat_params)
    print(f"✅ Loaded model params from {path}")
    return params, model_config
