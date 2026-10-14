"""Shared-filesystem checkpoint transaction; only metadata crosses hosts.

All hosts must call save/load in the same order. Python/I/O failures are shared
before publication. A dead process is handled by JAX's distributed timeout and
the launcher's supervision; no incomplete generation becomes authoritative.
"""

from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import uuid

import jax
import numpy as np
from jax.experimental import multihost_utils
import zstandard as zstd


def topology():
    return {"processes": jax.process_count(), "devices": [
        [d.process_index, d.platform, d.device_kind] for d in jax.devices()
    ]}


def collectively(operation, phase):
    """Propagate ordinary failures on any host before the next phase."""
    if jax.process_count() == 1:
        return operation()
    result, message = None, ""
    try:
        result = operation()
    except Exception as exc:
        message = f"host {jax.process_index()} {phase}: {type(exc).__name__}: {exc}"
    status = np.zeros(4096, dtype=np.uint8)
    encoded = message.encode()[:4095]
    status[:len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)
    reports = np.asarray(multihost_utils.process_allgather(status)).reshape(-1, 4096)
    errors = [bytes(row).rstrip(b"\0").decode(errors="replace") for row in reports if row.any()]
    if errors:
        raise RuntimeError("Checkpoint failed: " + "; ".join(errors))
    return result


def _fields(directory, payload, timings):
    from nanoalphazero.ckpt import compressed as codec
    compressor = zstd.ZstdCompressor(level=3)
    fields = []
    rank = jax.process_index()
    for field_id, (keys, value) in enumerate(jax.tree_util.tree_flatten_with_path(payload)[0]):
        key_impl = None
        if hasattr(value, "dtype") and jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key):
            key_impl = str(jax.random.key_impl(value))
            value = jax.random.key_data(value)
        value = jax.numpy.asarray(value)
        # Global replicas have the same index. Elect one owning process for each
        # unique shard; local scalars (e.g. cycle) are replicated host metadata.
        owners = {}
        for device, index in value.sharding.devices_indices_map(value.shape).items():
            identity = str(codec._index(index, value.shape))
            owner = 0 if value.is_fully_addressable else device.process_index
            owners[identity] = min(owners.get(identity, owner), owner)
        field = dict(path=jax.tree_util.keystr(keys), shape=list(value.shape),
                     dtype=str(value.dtype), key_impl=key_impl, shards=[],
                     indices=sorted(owners))
        verify_local = value.is_fully_addressable and jax.process_count() > 1
        local_hashes = []
        seen = set()
        for shard in value.addressable_shards:
            index = codec._index(shard.index, value.shape)
            identity = str(index)
            write_shard = owners[identity] == rank
            if identity in seen or not (write_shard or verify_local):
                continue
            seen.add(identity)
            entry = dict(index=index, shape=list(shard.data.shape), chunks=[])
            digest = hashlib.sha256()
            for block_id, (start, shape) in enumerate(codec._blocks(shard.data.shape, value.dtype.itemsize)):
                tick = time.monotonic()
                chunk = (shard.data if shape == shard.data.shape else
                         codec._slice(shard.data, np.asarray(start, dtype=np.int32), shape))
                raw = np.asarray(jax.device_get(chunk)).tobytes()
                timings["transfer_seconds"] += time.monotonic() - tick
                if verify_local:
                    digest.update(raw)
                if not write_shard:
                    continue
                tick = time.monotonic()
                encoded = compressor.compress(raw)
                timings["compress_seconds"] += time.monotonic() - tick
                filename = f"host-{rank}-{field_id}-{len(field['shards'])}-{block_id}.zst"
                tick = time.monotonic()
                codec._write(directory / filename, encoded)
                timings["write_seconds"] += time.monotonic() - tick
                entry["chunks"].append(dict(file=filename, bytes=len(raw), start=start,
                                            shape=shape, sha256=hashlib.sha256(raw).hexdigest()))
            if verify_local:
                local_hashes.append((identity, digest.hexdigest()))
            if write_shard:
                field["shards"].append(entry)
        if verify_local:
            field["local_hashes"] = sorted(local_hashes)
        fields.append(field)
    return fields


def _merge(directory, count):
    parts = [json.loads((directory / f"host-{rank}.json").read_text()) for rank in range(count)]
    merged = parts[0]
    for part in parts[1:]:
        for key in ("config", "metadata", "topology", "mode"):
            if part[key] != merged[key]:
                raise ValueError(f"Checkpoint hosts disagree on {key}")
        if len(part["fields"]) != len(merged["fields"]):
            raise ValueError("Checkpoint hosts disagree on state structure")
        for field, other in zip(merged["fields"], part["fields"]):
            if {k: v for k, v in field.items() if k != "shards"} != {
                k: v for k, v in other.items() if k != "shards"
            }:
                raise ValueError("Checkpoint hosts disagree on field geometry or local replicated values")
            field["shards"].extend(other["shards"])
    for field in merged["fields"]:
        if sorted(str(s["index"]) for s in field["shards"]) != field.pop("indices"):
            raise ValueError("Checkpoint shards are missing or duplicated")
    return merged


def save(path, payload, config, *, metadata=None):
    from nanoalphazero.ckpt import compressed as codec
    started = time.monotonic()
    timings = dict(transfer_seconds=0.0, compress_seconds=0.0, write_seconds=0.0)
    rank = jax.process_index()
    token = np.frombuffer(uuid.uuid4().bytes, dtype=np.uint8).copy()
    token = bytes(multihost_utils.broadcast_one_to_all(token)).hex()
    root = Path(path)
    temporary, committed = root / f".writing-{token}", root / f"gen-{token}"
    previous, published = None, False
    with ExitStack() as stack:
        def prepare():
            nonlocal previous
            if rank == 0:
                root.mkdir(parents=True, exist_ok=True)
                codec._sync_directory(root.parent)
                stack.enter_context(codec._lock(root, exclusive=True))
                previous = codec._generation(root) if (root / "LATEST").exists() else None
                temporary.mkdir()
                codec._sync_directory(root)
        try:
            collectively(prepare, "prepare")

            def write_host():
                manifest = dict(format=codec.FORMAT, mode="full" if "runner_state" in payload else "partial",
                                config=config, metadata=metadata or {}, topology=topology(),
                                fields=_fields(temporary, payload, timings), chunk_bytes=codec.CHUNK_BYTES)
                codec._write(temporary / f"host-{rank}.json", json.dumps(manifest, allow_nan=False).encode())
                codec._sync_directory(temporary)
            collectively(write_host, "write shards")

            def publish():
                nonlocal published
                if rank == 0:
                    manifest = _merge(temporary, jax.process_count())
                    codec._write(temporary / "manifest.json", json.dumps(manifest, allow_nan=False).encode())
                    codec._sync_directory(temporary)
                    temporary.rename(committed)
                    codec._sync_directory(root)
                    codec._write(root / "LATEST.tmp", json.dumps({"generation": committed.name}).encode())
                    os.replace(root / "LATEST.tmp", root / "LATEST")
                    published = True
                    codec._sync_directory(root)
            collectively(publish, "publish")
            # Readers hold shared locks, so the old generation is now unreferenced.
            def cleanup():
                if rank == 0 and previous is not None:
                    shutil.rmtree(previous)
                    codec._sync_directory(root)
            collectively(cleanup, "cleanup")
        finally:
            if rank == 0:
                if temporary.exists():
                    shutil.rmtree(temporary)
                if not published and committed.exists():
                    shutil.rmtree(committed)
    return {**timings, "seconds": time.monotonic() - started}
