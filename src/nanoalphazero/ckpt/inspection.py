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


def validate_resume_config(path, config, mode="full", *, fork=False):
    """Reject incompatible resumes before model/buffer allocation or deletion."""
    if path is None:
        return
    saved = inspect_checkpoint(path)
    if fork and mode != "partial":
        raise ValueError("Checkpoint forks require partial checkpoints with fresh games and buffers")
    if saved["mode"] != mode:
        raise ValueError(f"Checkpoint mode is {saved['mode']}, requested {mode}")

    def training_config(values):
        values = dict(values)
        # Old undiscounted checkpoints have the same return semantics.
        if values.get("muzero_discount") == 1:
            values.pop("muzero_discount")
        values.setdefault("buffer_layout", "original")
        values.setdefault("muzero_predict_value_only", False)
        if values.get("env_id") == "chess":
            values.setdefault("muzero_chess_action_encoding", "planes73")
        from nanoalphazero.muzero_config import normalize_exploration_config
        values = normalize_exploration_config(values)
        ignored = {"acpl", "ckpt_period", "enable_trackio", "enable_wandb"}
        if fork:
            # Explicit experiment continuation, not an exact resume. Keep the
            # architecture, optimizer, batch sizes, and target conventions fixed.
            ignored.update({"mcts_num_simulations", "mcts_num_root_considered",
                            "muzero_train_on_terminal_states"})
        # CLI preflight precedes environment construction. Once assembled,
        # compare derived dimensions too, before releasing the initial state.
        ignored.update(key for key in ("game_obs_shape", "game_num_actions") if config.get(key) is None)
        if mode == "partial":
            ignored.update(("selfplay_buffer_max_len", "buffer_layout"))
        return {key: value for key, value in values.items() if key not in ignored}

    if training_config(saved["config"]) != training_config(config):
        raise ValueError("MuZero resume requires identical resolved configuration except "
                         "selfplay_buffer_max_len and buffer_layout for partial checkpoints")
