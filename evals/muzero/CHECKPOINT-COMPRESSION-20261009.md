# Lossless MuZero chess buffer checkpoint prototype

## Source and limits

The interrupted overnight chess run retained a **compact** checkpoint at cycle
2400 (48,000 optimizer updates). Its historical buffers were not saved and cannot
be reconstructed exactly. This experiment generated fresh self-play from that
checkpoint, with the same b8c256 model, search settings, K=5, BNK-128 policies,
packed chess observations, and disabled consistency. No optimizer updates ran.

Collection used 256 lanes instead of 4096, with the same 512 self-play slots and
1024 replay slots. After 61 cycles / 1220 steps per lane / 312,320 positions,
replay was full and self-play had crossed its ring twice. All buffer error and
drop counters were zero. This avoids measuring empty or synthetically seeded
buffers. The original replay contained targets from changing model weights;
these freshly generated targets use fixed weights, so extrapolation is only an
estimate. Each replay item remains a full unroll, not one position.

## Results

Decimal units (GB = 10^9 bytes). Raw sampled buffers: 1,881,145,374 bytes.
Chunk size: 4 MiB, preserving C-order bytes, dtype, shape, and all stale slots.
Times are single-process codec time summed over fields; they exclude disk I/O,
device transfers, model/optimizer serialization, and tree reconstruction.

| Codec | Sample size | Ratio | Encode | Decode + byte verification | Estimated 4096-lane buffer size |
|---|---:|---:|---:|---:|---:|
| gzip level 5 (Vault's default codec/level) | 293.25 MB | 6.41x | 19.96 s | 4.23 s | 4.69 GB |
| Zstandard level 3 | 290.33 MB | 6.48x | 5.31 s | 1.38 s | 4.65 GB |
| Zstandard level 9 | 273.17 MB | 6.89x | 17.24 s | 1.33 s | 4.37 GB |
| Byte shuffle + Zstandard level 3 | 279.41 MB | 6.73x | 6.16 s | 4.77 s | 4.47 GB |

The full-size raw buffers are approximately 30.10 GB. Estimates multiply sample
payload by 16; negligible replicated scalar counters are also scaled. Add model,
optimizer, environment state, manifest, and filesystem overhead for a full
checkpoint. The existing compact model/optimizer checkpoint is about 150 MB.

Zstandard-3 breakdown: self-play 339.48 MB -> 30.07 MB, replay 1541.67 MB ->
260.26 MB. Replay policy weights account for 155.14 MB compressed and action
indices 73.43 MB; together they dominate the result. Packed boards still compress:
replay board_bool 245.37 MB -> 27.95 MB. Board_float 67.11 MB -> 1.08 MB.

## Prototype and validation

`scripts/probe_muzero_checkpoint_compression.py` collects real buffers, compares
the four codecs, and persists shuffled-Zstandard chunks with a manifest. It uses
the system libzstd through ctypes; no dependency changes or production I/O changes.
All codec round trips are byte-exact. Every persisted field is independently
read back from disk and checked against its SHA-256 hash. A separate CPU process
restored all 29 self-play fields and 12 replay fields (1.88 GB total), taking
1.42 s and 7.15 s respectively, including file reads, hashes and host array assembly.
Device-to-host transfers during capture totaled about 2.03 seconds.

The focused CPU test covers signed zero, infinities, NaN, bfloat16, packed uint8,
booleans, scalars, empty arrays, unaligned chunk sizes, and hash corruption:

```bash
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \
  uv run --extra trackio pytest -q tests/test_muzero_checkpoint_compression_probe.py
```

Result: **1 passed**. `git diff --check` passed. TPU collection completed and
released its lock. Snapshot loading returns host arrays keyed by tree path;
this is not yet a production full-RunnerState restore or next-cycle parity test.

Reproduction (use a new output directory and check TPU ownership first):

```bash
uv run --extra trackio python -u scripts/probe_muzero_checkpoint_compression.py \
  --checkpoint artifacts/muzero-chess-overnight-20261008-165124/checkpoint.safetensors \
  --output artifacts/muzero-compression-NEW \
  --platform tpu --batch 256 --max-cycles 250
```

Retained local evidence: `artifacts/muzero-compression-20261009-a/` contains
collection provenance, per-field manifests, summary.json and compressed chunks.
Console log: `/tmp/muzero-compression-20261009-a.log`.

## Recommendation

Start with plain Zstandard-3: nearly gzip's size at about 3.8x the compression
speed in this measurement. Shuffle saves only another 3.8% here and makes decode
slower. Level 9 saves about 6% versus level 3 at over 3x encode time.
Full-size sequential codec time extrapolates to approximately 85 seconds to
compress and 22 seconds to decode with level 3, before I/O/transfers. These are
not measured full-size checkpoint pauses; parallelism needs a separate benchmark.

Compression changes disk/host serialization, not live TPU buffer allocation.
It is promising, but the machine's remaining ~4.6 GB disk space still cannot
safely hold an old plus replacement full checkpoint. Production integration
needs consistent whole-state snapshots, atomic publication, durability, disk
headroom, and exact next-cycle restore tests. Flashbax Vault's append-only default
cannot by itself preserve our backfilled returns and consumed-start metadata.

## Streaming device restore follow-up

`scripts/probe_muzero_stream_restore.py` restores directly to four TPU shards.
It decodes at most one 4 MiB source chunk at a time, carries partial chunks across
shard boundaries, and updates preallocated device arrays using donated
`dynamic_update_slice` calls. Transfers are synchronized to bound outstanding
host buffers. Scalars are replicated; experience arrays are partitioned on B.
No complete host field and no uncompressed disk file is created. Device arrays
remain uncompressed, as required by the normal training code.

Each source field hash is checked during reading. Every resulting device shard
is then independently read back in chunks and hashed. The final global arrays
have the original shapes/dtypes (or repeated B for the capacity stress test).

| TPU test | Restored bytes | Time including compilation and full device readback verification | Initial host RSS | Peak host RSS |
|---|---:|---:|---:|---:|
| Original 256-lane sample | 1.88 GB | 71.50 s | 4.62 GB | 4.84 GB |
| 4096-lane capacity stress | 30.10 GB | 249.52 s | 4.63 GB | 4.86 GB |

The capacity test repeats the saved 256 lanes 16 times while decompressing them
again on each pass. It verifies real full-size device allocations and bounded
host memory, **not independent 4096-lane trajectories or a new compression
ratio measurement**. Both buffers are retained together through the end of each
test. Host RSS includes the sizeable TPU/JAX runtime; the full-size run's increase
over that baseline was approximately 232 MB, not 30 GB. User-space chunk buffers
are bounded, but compiled executables and runtime allocations also contribute.

Both tests passed every byte comparison. CPU tests additionally cover cross-shard
chunks, partial tails, repeated lanes, scalar replication and bfloat16: **3 passed**.
No training process was started, and the TPU lock was released afterward.

```bash
uv run --extra trackio python -u scripts/probe_muzero_stream_restore.py \
  --input artifacts/muzero-compression-20261009-a \
  --output artifacts/muzero-stream-restore-NEW.json \
  --platform tpu --repeat 16
```

Measured reports: `artifacts/muzero-stream-restore-20261009-small.json` and
`artifacts/muzero-stream-restore-20261009-full.json`; logs under
`/tmp/muzero-stream-small.log` and `/tmp/muzero-stream-full.log`.

Flashbax lessons, from the pinned installed `vault/vault.py` and
[upstream implementation](https://github.com/instadeepai/flashbax/blob/main/flashbax/vault/vault.py):

- TensorStore/Zarr supports compressed slice-based reads/writes. Its OCDBT backing
  store is useful infrastructure for managing many chunks rather than individual
  prototype files.
- Vault's asynchronous writes offer concurrency, but our streaming path should
  bound in-flight chunks; unlimited concurrency can grow host memory.
- Vault's default `read()` materializes each requested leaf interval on the host
  with `read().result()` before `jnp.asarray`. It is not this direct-to-shards
  streaming loader. It also returns `current_index=0, is_full=True`, which is not
  an exact restoration of our original ring metadata.
- Ring-aware append intervals are useful for experience archives. They do not
  capture our later return backfills and consumption metadata by default.

This is still a buffer-I/O prototype. Binding the arrays to a versioned complete
RunnerState, atomic full-state save/load, and next-cycle equivalence remain
production integration work. The timings above deliberately include verification
and compilation; they are not claimed optimized checkpoint resume latencies.
