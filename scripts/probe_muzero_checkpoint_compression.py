"""Lossless checkpoint prototype using real self-play from a compact checkpoint.

Run from the repo root with uv. Does not resume training or update parameters.
Reconstructs fresh buffers at a smaller lane count; never claims these are the
lost historical buffers. Keeps buffer lengths, search, and model unchanged.
Writes independently compressed chunks plus a manifest, verifying every chunk
from disk bit-for-bit. Full-size estimates assume similar data across lanes.
"""

import argparse
import ctypes
import ctypes.util
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time


class Zstd:
    """Small binding to the system lossless Zstandard library (no pip changes)."""

    def __init__(self):
        self.lib = ctypes.CDLL(ctypes.util.find_library("zstd"))
        for name, arguments in {
            "ZSTD_compressBound": [ctypes.c_size_t],
            "ZSTD_compress": [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
                              ctypes.c_size_t, ctypes.c_int],
            "ZSTD_decompress": [ctypes.c_void_p, ctypes.c_size_t,
                                ctypes.c_void_p, ctypes.c_size_t],
            "ZSTD_isError": [ctypes.c_size_t],
        }.items():
            function = getattr(self.lib, name)
            function.argtypes = arguments
            function.restype = ctypes.c_size_t

    def encode(self, raw, level):
        out = ctypes.create_string_buffer(self.lib.ZSTD_compressBound(len(raw)))
        size = self.lib.ZSTD_compress(out, len(out), raw, len(raw), level)
        if self.lib.ZSTD_isError(size):
            raise RuntimeError("Zstandard compression failed")
        return out.raw[:size]

    def decode(self, encoded, size):
        out = ctypes.create_string_buffer(size)
        actual = self.lib.ZSTD_decompress(out, size, encoded, len(encoded))
        if self.lib.ZSTD_isError(actual) or actual != size:
            raise RuntimeError("Zstandard decompression failed")
        return out.raw


def shuffle(raw, itemsize, inverse=False):
    import numpy as np
    a = np.frombuffer(raw, dtype=np.uint8)
    return (a.reshape(itemsize, -1).T if inverse else
            a.reshape(-1, itemsize).T).copy().tobytes()


def snapshot(tree, directory, chunk_bytes=4 * 1024**2, lane_scale=1):
    """Benchmark codecs; persist shuffled zstd-3, then read and verify it."""
    import jax
    import numpy as np
    directory.mkdir(exist_ok=False)
    codec = Zstd()
    variants = ("gzip5", "zstd3", "zstd9", "shuffle_zstd3")
    totals = {name: dict(bytes=0, encode_seconds=0., decode_seconds=0.) for name in variants}
    manifest = dict(format="muzero-lossless-chunks-prototype-v1", fields=[])
    transfer_seconds = 0.
    for index, (path, leaf) in enumerate(jax.tree_util.tree_flatten_with_path(tree)[0]):
        started = time.perf_counter()
        array = np.asarray(jax.device_get(leaf))
        transfer_seconds += time.perf_counter() - started
        raw = array.tobytes(order="C")
        field = dict(path=jax.tree_util.keystr(path), shape=list(array.shape),
                     dtype=str(array.dtype), itemsize=array.dtype.itemsize,
                     raw_bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), chunks=[],
                     codecs={name: dict(bytes=0, encode_seconds=0., decode_seconds=0.) for name in variants})
        step = max(array.dtype.itemsize, chunk_bytes // array.dtype.itemsize * array.dtype.itemsize)
        for offset in range(0, len(raw), step):
            chunk = raw[offset:offset + step]
            for name in variants:
                started = time.perf_counter()
                payload = shuffle(chunk, array.dtype.itemsize) if name.startswith("shuffle") else chunk
                encoded = gzip.compress(payload, compresslevel=5, mtime=0) if name == "gzip5" else codec.encode(payload, 9 if name == "zstd9" else 3)
                encode_seconds = time.perf_counter() - started
                started = time.perf_counter()
                decoded = gzip.decompress(encoded) if name == "gzip5" else codec.decode(encoded, len(chunk))
                if name.startswith("shuffle"):
                    decoded = shuffle(decoded, array.dtype.itemsize, inverse=True)
                assert decoded == chunk, (field["path"], name, offset)
                decode_seconds = time.perf_counter() - started
                for stat in (totals[name], field["codecs"][name]):
                    stat["bytes"] += len(encoded)
                    stat["encode_seconds"] += encode_seconds
                    stat["decode_seconds"] += decode_seconds
                if name == "shuffle_zstd3":
                    filename = f"{index:03d}-{offset:012d}.zst"
                    with (directory / filename).open("wb") as out:
                        out.write(encoded)
                        out.flush()
                        os.fsync(out.fileno())
                    field["chunks"].append(dict(file=filename, raw_bytes=len(chunk)))
        # Independent disk read, hash the reconstructed field in streaming order.
        digest = hashlib.sha256()
        for chunk in field["chunks"]:
            restored = codec.decode((directory / chunk["file"]).read_bytes(), chunk["raw_bytes"])
            digest.update(shuffle(restored, array.dtype.itemsize, inverse=True))
        assert digest.hexdigest() == field["sha256"], field["path"]
        manifest["fields"].append(field)
        print(json.dumps({"field": field["path"], "raw_bytes": len(raw),
                          "compressed": {k: v["bytes"] for k, v in field["codecs"].items()}}), flush=True)
    raw_bytes = sum(field["raw_bytes"] for field in manifest["fields"])
    manifest.update(raw_bytes=raw_bytes, codecs=totals, transfer_seconds=transfer_seconds,
                    chunk_bytes=chunk_bytes, roundtrip="all chunks bit-exact, including disk read",
                    lane_scale=lane_scale,
                    estimated_full_bytes={k: v["bytes"] * lane_scale for k, v in totals.items()})
    temporary = directory / "manifest.tmp"
    with temporary.open("w") as out:
        json.dump(manifest, out, indent=2)
        out.flush()
        os.fsync(out.fileno())
    temporary.replace(directory / "manifest.json")
    return manifest


def read_snapshot(directory):
    """Restore host arrays keyed by tree path, verifying stored SHA-256 hashes.

    This prototype deliberately does not instantiate a training RunnerState;
    production integration must bind these paths to its versioned tree schema.
    """
    import numpy as np
    import ml_dtypes  # Registers bfloat16 with NumPy.
    del ml_dtypes
    manifest = json.loads((directory / "manifest.json").read_text())
    codec = Zstd()
    arrays = {}
    for field in manifest["fields"]:
        raw = bytearray()
        for chunk in field["chunks"]:
            decoded = codec.decode((directory / chunk["file"]).read_bytes(), chunk["raw_bytes"])
            raw.extend(shuffle(decoded, field["itemsize"], inverse=True))
        if hashlib.sha256(raw).hexdigest() != field["sha256"]:
            raise ValueError(f"Corrupt field: {field['path']}")
        arrays[field["path"]] = np.frombuffer(raw, dtype=field["dtype"]).reshape(field["shape"])
    return arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--platform", choices=("cpu", "tpu"), default="cpu")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--max-cycles", type=int, default=250)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    if args.platform == "tpu":
        owners = subprocess.run(["fuser", "/tmp/libtpu_lockfile"], capture_output=True, text=True)
        if owners.returncode == 0:
            raise SystemExit(f"TPU occupied: {owners.stdout}")
    os.environ["JAX_PLATFORMS"] = args.platform
    if args.platform == "cpu":
        os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
    import jax
    import jax.numpy as jnp
    from safetensors import safe_open
    from nanoalphazero.ckpt import load_muzero_checkpoint, muzero_checkpoint_payload

    with safe_open(str(args.checkpoint), framework="numpy") as f:
        source_config = json.loads(f.metadata()["config"])
    config = dict(source_config)
    for key in ("selfplay_batch_size", "train_batch_size", "selfplay_buffer_consume_size",
                "selfplay_buffer_add_batch_size", "selfplay_buffer_sample_batch_size",
                "replay_buffer_add_batch_size", "replay_buffer_sample_batch_size"):
        config[key] = args.batch
    spec = importlib.util.spec_from_file_location("muzero_compression_probe", Path(__file__).resolve().parents[1] / "muzero.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    algorithm = module.make_muzero(config, jax.random.PRNGKey(config["seed"]))
    state = algorithm.runner_state
    restored, _ = load_muzero_checkpoint(args.checkpoint, muzero_checkpoint_payload(state, 0, False))
    state = state._replace(model_ts=restored["model_ts"], rng=restored["rng"])
    placement = jax.tree.map(lambda x: x.sharding, algorithm.runner_state)
    state = jax.device_put(state, placement)
    updates = int(state.model_ts.n_updates)
    # Literal True eliminates the unused optimizer branch from this data collector.
    collect = jax.jit(lambda s: algorithm.run_fn(s, True), donate_argnums=(0,))
    started = time.perf_counter()
    collect = collect.lower(state).compile()
    print(json.dumps({"compile_seconds": time.perf_counter() - started}), flush=True)
    for cycle in range(1, args.max_cycles + 1):
        state, metrics = collect(state)
        metrics = {k: float(v) for k, v in jax.device_get(metrics).items()}
        assert metrics["selfplay_buffer/error_count"] == 0, metrics
        assert int(state.model_ts.n_updates) == updates
        full = bool(state.replay_buffer_state.is_full)
        if cycle % 10 == 0 or full:
            print(json.dumps({"collection_cycle": cycle, "replay_full": full,
                              "replay_index": int(state.replay_buffer_state.current_index),
                              **{k: v for k, v in metrics.items() if "buffer/" in k}}), flush=True)
        if full and cycle * config["cycle_n_selfplay"] >= 2 * config["selfplay_buffer_max_len"]:
            break
    else:
        raise RuntimeError("Replay did not fill within the bounded collection budget")
    info = dict(source_checkpoint=str(args.checkpoint), source_cycle=int(restored["cycle"]),
                source_config=source_config, collection_config=config, collection_cycles=cycle,
                optimizer_updates_during_collection=0, replay_full=full,
                provenance="Fresh self-play from saved model, NOT original lost buffers", metrics=metrics)
    (args.output / "collection.json").write_text(json.dumps(info, indent=2))
    results = {}
    for name, buffer in (("selfplay", state.selfplay_buffer_state), ("replay", state.replay_buffer_state)):
        manifest = snapshot(buffer, args.output / name,
                            lane_scale=source_config["selfplay_batch_size"] / args.batch)
        results[name] = {k: v for k, v in manifest.items() if k != "fields"}
    (args.output / "summary.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
