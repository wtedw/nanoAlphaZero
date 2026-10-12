"""Checkpoint headers and resume compatibility, without allocating training state."""

import json
from pathlib import Path


def inspect_checkpoint(path):
    if Path(path).is_dir():
        from nanoalphazero.ckpt.compressed import inspect_checkpoint as inspect
        return inspect(path)
    from safetensors import safe_open

    with safe_open(path, framework="numpy") as reader:
        header = reader.metadata()
    formats = {"nanoalphazero.muzero.compact.v1": "partial",
               "nanoalphazero.muzero.persistent.v4": "full"}
    return dict(format=header["format"], mode=formats[header["format"]],
                config=json.loads(header["config"]), metadata={})


def checkpoint_metadata(path):
    return inspect_checkpoint(path).get("metadata", {})


def validate_resume_config(path, config, mode="full"):
    """Reject incompatible resumes before model/buffer allocation or deletion."""
    if path is None:
        return
    saved = inspect_checkpoint(path)
    if saved["mode"] != mode:
        raise ValueError(f"Checkpoint mode is {saved['mode']}, requested {mode}")

    def training_config(values):
        values = dict(values)
        values.setdefault("buffer_layout", "original")
        values.setdefault("muzero_predict_value_only", False)
        for old in ("exp_train_on_exploratory_unroll_states", "exp_train_on_exploration"):
            if old in values:
                values.setdefault("exp_train_on_exploit_unroll_states", not values.pop(old))
        values.setdefault("exp_train_on_exploit_unroll_states", True)
        ignored = {"acpl", "ckpt_period", "enable_trackio", "enable_wandb"}
        # CLI preflight precedes environment construction. Once assembled,
        # compare derived dimensions too, before releasing the initial state.
        ignored.update(key for key in ("game_obs_shape", "game_num_actions") if key not in config)
        if mode == "partial":
            ignored.update(("selfplay_buffer_max_len", "buffer_layout"))
        return {key: value for key, value in values.items() if key not in ignored}

    if training_config(saved["config"]) != training_config(config):
        raise ValueError("MuZero resume requires identical resolved configuration except "
                         "selfplay_buffer_max_len and buffer_layout for partial checkpoints")
