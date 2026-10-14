"""Bounded host transfers for logging. Every process calls these in order."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils


@partial(jax.jit, static_argnames=("limit",))
def _sample(arrays, limit):
    # Shared indices preserve alignment between provenance and validity masks.
    size = arrays["valid"].size
    count = min(size, limit)
    step, remainder = divmod(size, max(count, 1))
    slots = jnp.arange(count, dtype=jnp.int32)
    # Equivalent to slots * size // count, without the large intermediate.
    indices = slots * step + slots * remainder // max(count, 1)
    return jax.tree.map(lambda x: x.reshape(-1)[indices], arrays)


def gather_training_arrays(arrays, limit=4096):
    """Gather at most limit global samples per field, never the full buffer.

    Sampling is a global JAX operation before process_allgather. In particular,
    taking a NumPy view of a non-addressable array is not a gather.
    """
    if limit < 1:
        raise ValueError("Histogram sample limit must be positive")
    if jax.process_count() == 1:
        return jax.device_get(arrays)
    return multihost_utils.process_allgather(_sample(arrays, limit=limit), tiled=True)


def device_memory_metrics(devices):
    """Query local devices only, then aggregate pod-wide maxima on all hosts."""
    names = ("bytes_in_use", "peak_bytes_in_use", "bytes_limit")
    stats = [device.memory_stats() or {} for device in devices
             if device.process_index == jax.process_index()]
    local = np.array([max(s[name] for s in stats)
                      if stats and all(name in s for s in stats) else np.nan
                      for name in names], dtype=np.float32)
    gathered = (multihost_utils.process_allgather(local) if jax.process_count() > 1 else local)
    rows = np.asarray(gathered).reshape(-1, len(names))
    return {f"memory/{name}_max": float(rows[:, i].max()) for i, name in enumerate(names)
            if np.isfinite(rows[:, i]).all()}
