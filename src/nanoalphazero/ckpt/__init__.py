"""MuZero whole-state checkpoint API. AlphaZero keeps checkpoint.py."""

from nanoalphazero.ckpt.inspection import checkpoint_metadata, inspect_checkpoint, validate_resume_config

from nanoalphazero.ckpt.state import (
    save_checkpoint, restore_checkpoint,
    payload as muzero_checkpoint_payload,
    save_payload as save_muzero_checkpoint,
    load_payload as load_muzero_checkpoint,
)


def load_model(*args, **kwargs):
    from nanoalphazero.ckpt.model import load_model as load
    return load(*args, **kwargs)
