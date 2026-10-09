"""Synchronous, lossless, sharded snapshots with atomic generation publication.

Only compressed chunks are written to disk. Host staging is bounded by one
4 MiB chunk, except for runtime/codec overhead. Save between completed cycles;
do not mutate/donate the source state concurrently. Single-host checkpoints
restore with the same shard geometry (device IDs themselves may change).
"""

from contextlib import contextmanager
from functools import partial
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import uuid
import time

import jax
import jax.numpy as jnp
import numpy as np
import zstandard as zstd

CHUNK_BYTES = 4 * 1024**2
FORMAT = "nanoalphazero.muzero.zstd.v2"


def _sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write(path, content):
    with path.open("wb") as file:
        file.write(content)
        file.flush()
        os.fsync(file.fileno())


@contextmanager
def _lock(path, exclusive):
    import fcntl
    with (path / ".lock").open("a+b") as file:
        fcntl.flock(file, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        yield


def _index(index, shape):
    return [list(part.indices(size)) for part, size in zip(index, shape)]


def _generation(root):
    name = json.loads((root / "LATEST").read_text())["generation"]
    if not name.startswith("gen-") or Path(name).name != name:
        raise ValueError("Invalid checkpoint generation")
    return root / name


def _blocks(shape, itemsize):
    """C-order rectangular blocks, each bounded by the staging budget."""
    if math.prod(shape) == 0:
        return
    limit = max(1, CHUNK_BYTES // itemsize)
    if math.prod(shape) <= limit:
        yield (0,) * len(shape), tuple(shape)
        return
    suffix = 1
    axis = len(shape) - 1
    while axis > 0 and suffix * shape[axis] <= limit:
        suffix *= shape[axis]
        axis -= 1
    stride = max(1, limit // suffix)
    for prefix in itertools.product(*(range(n) for n in shape[:axis])):
        for offset in range(0, shape[axis], stride):
            start = prefix + (offset,) + (0,) * (len(shape) - axis - 1)
            size = (1,) * axis + (min(stride, shape[axis] - offset),) + tuple(shape[axis + 1:])
            yield start, size


@partial(jax.jit, static_argnums=(2,))
def _slice(array, start, shape):
    return jax.lax.dynamic_slice(array, start, shape)


@partial(jax.jit, donate_argnums=(0,))
def _insert(array, chunk, start):
    return jax.lax.dynamic_update_slice(array, chunk, start)


@partial(jax.jit, static_argnums=(1,))
def _allocate(zero, shape):
    return jnp.full(shape, zero, dtype=zero.dtype)


def save(path, payload, config, *, metadata=None):
    started = time.monotonic()
    timings = dict(transfer_seconds=0.0, compress_seconds=0.0, write_seconds=0.0)
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    _sync_directory(root.parent)
    with _lock(root, exclusive=True):
        # A previous process may have died before publishing. These are our own
        # private temporary generations, never committed checkpoints.
        for stale in root.glob(".writing-*"):
            if stale.is_dir():
                shutil.rmtree(stale)
        previous = _generation(root) if (root / "LATEST").exists() else None
        token = uuid.uuid4().hex
        temporary = root / f".writing-{token}"
        committed = root / f"gen-{token}"
        temporary.mkdir()
        compressor = zstd.ZstdCompressor(level=3)
        manifest = dict(format=FORMAT, mode="full" if "runner_state" in payload else "partial",
                        config=config, metadata=metadata or {}, fields=[], chunk_bytes=CHUNK_BYTES)
        published = False
        try:
            for field_id, (keys, value) in enumerate(jax.tree_util.tree_flatten_with_path(payload)[0]):
                key_impl = None
                if hasattr(value, "dtype") and jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key):
                    key_impl = str(jax.random.key_impl(value))
                    value = jax.random.key_data(value)
                value = jax.numpy.asarray(value)
                if not value.is_fully_addressable:
                    raise ValueError("Compressed checkpoints currently require single-host arrays")
                field = dict(path=jax.tree_util.keystr(keys), shape=list(value.shape),
                             dtype=str(value.dtype), key_impl=key_impl, shards=[])
                seen = set()
                for shard in value.addressable_shards:
                    index = _index(shard.index, value.shape)
                    identity = str(index)
                    if identity in seen:
                        continue  # Store replicated data only once.
                    seen.add(identity)
                    entry = dict(index=index, shape=list(shard.data.shape), chunks=[])
                    for block_id, (start, shape) in enumerate(_blocks(shard.data.shape, value.dtype.itemsize)):
                        tick = time.monotonic()
                        chunk = (shard.data if shape == shard.data.shape else
                                 _slice(shard.data, np.asarray(start, dtype=np.int32), shape))
                        raw = np.asarray(jax.device_get(chunk)).tobytes()
                        timings["transfer_seconds"] += time.monotonic() - tick
                        tick = time.monotonic()
                        encoded = compressor.compress(raw)
                        timings["compress_seconds"] += time.monotonic() - tick
                        filename = f"{field_id}-{len(field['shards'])}-{block_id}.zst"
                        tick = time.monotonic()
                        _write(temporary / filename, encoded)
                        timings["write_seconds"] += time.monotonic() - tick
                        entry["chunks"].append(dict(file=filename, bytes=len(raw),
                                                    start=start, shape=shape,
                                                    sha256=hashlib.sha256(raw).hexdigest()))
                    field["shards"].append(entry)
                manifest["fields"].append(field)
            _write(temporary / "manifest.json", json.dumps(manifest, allow_nan=False).encode())
            _sync_directory(temporary)
            temporary.rename(committed)
            _sync_directory(root)
            _write(root / "LATEST.tmp", json.dumps({"generation": committed.name}).encode())
            os.replace(root / "LATEST.tmp", root / "LATEST")
            published = True
            _sync_directory(root)
            if previous is not None:
                shutil.rmtree(previous)
                _sync_directory(root)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
            # If publication failed, the previous generation remains authoritative.
            if not published and committed.exists():
                shutil.rmtree(committed)
    return {**timings, "seconds": time.monotonic() - started}


def metadata(path):
    """Read host metadata without allocating model or buffer arrays."""
    root = Path(path)
    with _lock(root, exclusive=False):
        manifest = json.loads((_generation(root) / "manifest.json").read_text())
        if manifest["format"] != FORMAT:
            raise ValueError("Unsupported compressed checkpoint format")
        return manifest.get("metadata", {})


def load(path, template):
    root = Path(path)
    with _lock(root, exclusive=False):
        directory = _generation(root)
        manifest = json.loads((directory / "manifest.json").read_text())
        expected, tree = jax.tree_util.tree_flatten_with_path(template)
        mode = "full" if "runner_state" in template else "partial"
        if manifest["format"] != FORMAT or manifest["mode"] != mode:
            raise ValueError("Compressed checkpoint format or mode does not match")
        if [f["path"] for f in manifest["fields"]] != [jax.tree_util.keystr(k) for k, _ in expected]:
            raise ValueError("Checkpoint state structure does not match the template")
        decompressor = zstd.ZstdDecompressor()
        leaves = []
        for (_, example), field in zip(expected, manifest["fields"]):
            typed_key = hasattr(example, "dtype") and jax.dtypes.issubdtype(example.dtype, jax.dtypes.prng_key)
            if typed_key:
                example = (jax.eval_shape(jax.random.key_data, example)
                           if isinstance(example, jax.ShapeDtypeStruct) else jax.random.key_data(example))
            dtype = np.dtype(example.dtype if hasattr(example, "dtype") else np.asarray(example).dtype)
            shape = np.shape(example)
            if list(shape) != field["shape"] or str(dtype) != field["dtype"]:
                raise ValueError(f"Checkpoint shape/dtype mismatch: {field['path']}")
            sharding = getattr(example, "sharding", None) or jax.sharding.SingleDeviceSharding(jax.devices()[0])
            if not sharding.is_fully_addressable:
                raise ValueError("Compressed restore currently requires single-host arrays")
            local_arrays = []
            for device, index in sharding.addressable_devices_indices_map(shape).items():
                saved = next((s for s in field["shards"] if s["index"] == _index(index, shape)), None)
                if saved is None:
                    raise ValueError("Checkpoint shard geometry differs; use the original device count/sharding")
                array = None
                blocks = list(_blocks(saved["shape"], dtype.itemsize))
                if len(blocks) != len(saved["chunks"]):
                    raise ValueError("Incomplete checkpoint shard")
                for chunk, (start, block_shape) in zip(saved["chunks"], blocks):
                    if chunk["start"] != list(start) or chunk["shape"] != list(block_shape):
                        raise ValueError("Invalid checkpoint block geometry")
                    filename = chunk["file"]
                    if Path(filename).name != filename or not filename.endswith(".zst"):
                        raise ValueError("Invalid checkpoint chunk filename")
                    if chunk["bytes"] != math.prod(block_shape) * dtype.itemsize:
                        raise ValueError("Invalid checkpoint chunk size")
                    raw = decompressor.decompress((directory / filename).read_bytes(), max_output_size=chunk["bytes"])
                    if len(raw) != chunk["bytes"] or hashlib.sha256(raw).hexdigest() != chunk["sha256"]:
                        raise ValueError(f"Corrupt checkpoint chunk: {filename}")
                    host = np.frombuffer(raw, dtype=dtype).reshape(block_shape)
                    data = jax.device_put(host, device)
                    if block_shape == tuple(saved["shape"]):
                        array = data
                    else:
                        if array is None:
                            array = _allocate(jax.device_put(np.array(0, dtype=dtype), device), tuple(saved["shape"]))
                        array = _insert(array, data, jax.device_put(np.asarray(start, dtype=np.int32), device))
                    array.block_until_ready()
                if array is None:
                    array = jax.device_put(np.empty(saved["shape"], dtype=dtype), device)
                local_arrays.append(array)
            value = jax.make_array_from_single_device_arrays(shape, sharding, local_arrays)
            if typed_key:
                value = jax.random.wrap_key_data(value, impl=field["key_impl"])
            leaves.append(value)
        return tree.unflatten(leaves), manifest["config"]
