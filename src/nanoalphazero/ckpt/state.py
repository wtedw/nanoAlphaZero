"""MuZero checkpoint state semantics, format dispatch, and resume compatibility."""

from pathlib import Path

import jax
import jax.numpy as jnp


def payload(state, cycle, include_replay=True):
    """Full snapshots continue games/buffers; partial snapshots restart them."""
    if include_replay:
        return {"runner_state": state, "cycle": jnp.int32(cycle)}
    return {"model_ts": state.model_ts, "rng": state.rng, "cycle": jnp.int32(cycle)}


def save_payload(path, state, config, *, compression="none"):
    if compression == "zstd":
        from nanoalphazero.ckpt.compressed import save as write
    elif compression == "none":
        from nanoalphazero.ckpt.safetensors import save as write
    else:
        raise ValueError(f"Unknown checkpoint compression: {compression}")
    return write(path, state, config)


def load_payload(path, template):
    if Path(path).is_dir():
        from nanoalphazero.ckpt.compressed import load as read
    else:
        from nanoalphazero.ckpt.safetensors import load as read
    return read(path, template)


def save_checkpoint(path, *, state, cycle, config, mode="full", metadata=None):
    """Save one completed cycle. The caller owns cadence and evaluation ordering."""
    if mode not in ("full", "partial"):
        raise ValueError(f"Unknown checkpoint mode: {mode}")
    from nanoalphazero.ckpt.compressed import save
    return save(path, payload(state, cycle, mode == "full"), config, metadata=metadata)


def restore_checkpoint(path, *, state, config, mode="full", fork=False):
    """Return (RunnerState, start_cycle), or the initial state when path is None.

    Owns the supplied initial state, like the donating training loop. Compressed
    full restore releases its arrays before allocation; callers must not reuse it.
    Partial restore retains initial games/buffers. The host decides when to warm up.
    """
    if mode not in ("full", "partial"):
        raise ValueError(f"Unknown checkpoint mode: {mode}")
    if path is None:
        return state, 0
    from nanoalphazero.ckpt.inspection import validate_resume_config
    validate_resume_config(path, config, mode, fork=fork)
    resume = path
    include_replay = mode == "full"
    placement = jax.tree.map(lambda x: x.sharding, state)
    template = payload(state, 0, include_replay)
    if Path(resume).is_dir() and include_replay:
        # Restore directly to devices without retaining a second full set
        # of initial buffers. run_muzero owns/donates this state already.
        template = jax.tree.map(
            lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.sharding), template
        )
        for leaf in jax.tree.leaves(state):
            leaf.delete()
    saved, _ = load_payload(
        resume, template
    )
    start_cycle = int(saved["cycle"])
    if include_replay:
        state = saved["runner_state"]
    else:
        state = state._replace(model_ts=saved["model_ts"], rng=saved["rng"])
        print("Partial checkpoint restored; restarting games and warming up empty buffers", flush=True)
    if config.get("enable_sharding", False):
        # Restore the original placement, including replicated scalar counters.
        state = jax.device_put(state, placement)

    return state, start_cycle
