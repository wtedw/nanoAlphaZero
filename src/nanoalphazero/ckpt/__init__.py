"""MuZero whole-state checkpoint API. AlphaZero keeps checkpoint.py."""

from nanoalphazero.ckpt.compressed import metadata as checkpoint_metadata

from nanoalphazero.ckpt.state import (
    save_checkpoint, restore_checkpoint,
    payload as muzero_checkpoint_payload,
    save_payload as save_muzero_checkpoint,
    load_payload as load_muzero_checkpoint,
)
