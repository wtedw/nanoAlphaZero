# Compressed MuZero checkpoints

The public MuZero API is `nanoalphazero.ckpt.save_checkpoint` / `restore_checkpoint`. This package owns
full/partial state handling, configuration compatibility, device placement, and
both storage formats. AlphaZero's existing model-checkpoint functions remain in
`checkpoint.py`, unchanged. Its old MuZero imports are compatibility aliases only.

```python
state, start_cycle = ckpt.restore_checkpoint(resume, state=initial_state, config=config)
ckpt.save_checkpoint(path, state=state, cycle=cycle, config=config)
```

With no resume path, `restore_checkpoint` returns the original state and cycle zero. The
host retains warmup, cycle counting, save cadence, and evaluation ordering.
`state.py` owns state semantics, `safetensors.py` the existing MuZero format,
and `compressed.py` the chunked directory format.

## Follow-ups

- Add `load_model(...)` to read model parameters, inference state, and architecture
  directly from a training checkpoint without loading optimizer or experience
  buffers. Arrays are compressed in separate chunks, so selective loading can
  skip those fields. Add `export_model(...)` for standalone inference files.
  Neither model-only API is implemented yet.

- Consider moving the host cycle counter into `RunnerState`, then returning just
  state from `restore_checkpoint`. Review shared AlphaZero constructors, checkpoint migration,
  warmup semantics, and evaluator/logging cadence first. For now, the saved `cycle`
  field and separate `start_cycle` return are unchanged.

Public modes are `full` and `partial`; `partial` includes optimizer state and RNG,
so it is not a model-only export. Mode is a checkpoint option, not algorithm config.

```bash
# Add these options to your existing MuZero training command:
--ckpt.save artifacts/chess.ckpt \
--ckpt.every 400
```

MuZero training defaults to full checkpoints: model, optimizer, RNG, cycle,
ongoing games, and both buffers. All CLI saves use lossless Zstandard compression;
there is no compression switch. Full resume skips buffer warmup. Use
`--ckpt.mode partial` only when deliberately omitting games/buffers;
partial resume rebuilds buffers and warms up.
The low-level compatibility API still supports legacy safetensors writing;
`save_checkpoint` always writes compressed directories.
Do not point compressed output at an existing safetensors file.

Resume with `--ckpt.resume artifacts/chess.ckpt`, keeping the
original training configuration and device count. Loading detects the directory
format automatically. To continue saving compressed checkpoints, also supply
`--ckpt.save ...`; resume alone does not enable saving.
Compression is an I/O option and is not part of training configuration comparison.

## Storage and consistency

Each checkpoint directory contains a `LATEST` pointer and a committed generation
with `manifest.json` plus Zstandard-3 chunks. The manifest records version, mode,
configuration, tree paths, dtypes, shapes, RNG key implementations, shard indices,
and per-chunk SHA-256 hashes. Replicated arrays are stored once.

Saving is synchronous between completed training cycles and occurs before
evaluation. The original device state is not modified. Each device shard is read
in at most 4 MiB chunks; only compressed bytes are written to disk. Loading checks
checksums and fills target device shards using bounded transfers and donated
updates. It never writes an uncompressed checkpoint file. Small leaves fit in one
host chunk; large leaves do not require a complete host array. JAX executable and
runtime memory is additional to that staging space.

Saving writes and fsyncs a private generation, then fsyncs its directory, renames
it, and atomically replaces/fsyncs `LATEST`. Only after publication is durable is
the previous generation deleted. An advisory lock serializes writers and prevents
readers racing generation deletion. A failed save before publication leaves the
previous checkpoint authoritative. Interrupted `.writing-*` directories are
removed under the writer lock on the next save. An interruption between committing
a generation and pointer publication can leave an unused `gen-*` directory; it
does not change which checkpoint loads. No automatic deletion of unknown
generation directories is performed.

Allow space for old and new compressed checkpoints simultaneously: approximately
10 GB total for two estimated 5 GB chess snapshots, plus margin. The compression
estimate came from sampled buffers; real checkpoint sizes vary.

Current scope: one host, same shard geometry on resume. There is no automatic
resharding across device counts, asynchronous save, incremental buffer archive,
or automatic restart. Local Trackio identity is saved in host metadata and reused
on resume. The loader creates new device
arrays; callers retaining a populated template temporarily retain those arrays
too. MuZero's compressed full-resume path retains shape/sharding descriptors and
releases the initial arrays before loading to avoid that duplicate allocation.

## Resume and logging

Checkpoint cadence and tracker enablement are host options, excluded from
training-config matching. Resume alone never writes checkpoints; `--ckpt.every`
may be omitted or changed. Omit `--ckpt.save` to disable every save.

With `--enable-trackio`, new checkpoints record the project, run name, and ID.
Resume uses Trackio's `resume="must"` and checks that the ID still matches. The
local Trackio database must still exist; the checkpoint does not archive it.
`--trackio-run NAME` explicitly selects an existing run. Already-persisted cycles
are not logged again if training resumes from an earlier checkpoint. Trackio
logging is asynchronous: an abrupt kill can lose the last queued metrics even
when training state was saved. W&B run continuation is not implemented here.

## Rectangular chunks (prototype v2)

The v2 compressed format stores each chunk's original rectangular shape and
position. Small shards transfer directly to the CPU; large shards are sliced
without flattening the whole array on TPU. Restore allocates the final shape and
inserts shaped chunks. Host staging stays bounded to a few chunk-sized temporary
arrays. This avoids TPU compilation of full-shard reshape/layout conversions.
Prototype v1 compressed directories are not migrated by this loader.

Saves report `checkpoint/transfer_seconds` (including slice compilation),
`checkpoint/compress_seconds`, `checkpoint/write_seconds`, and
`checkpoint/seconds` (including hashing and atomic publication). These separate
initial compilation from steady-state checkpoint overhead. The reproducible
probe is `scripts/probe_muzero_checkpoint_runtime.py`; it checks two saves,
restore, and byte-exact next-cycle continuation with Hex4's default model/buffers.

Four-device TPU measurement (2026-10-09): default Hex4 b4c64, batch 1024,
self-play length 512, replay length 2000, features layout, 203 warmup cycles.
First save 26.43 s (15.74 transfer, 6.51 compression, 3.28 writes); second save
16.56 s (5.86 transfer, 6.50 compression, 3.25 writes); restore 15.08 s.
Stored size 406,973,482 bytes. Every next-cycle state array and metric matched
exactly; optimizer updates reached 20 with zero buffer errors. This is a Hex4
measurement, not a chess performance claim. The original flattening saver stalled
for many minutes compiling reshapes with the same Hex4 model/buffer dimensions.

Full resume should be tested by comparing the next actual training cycle, not
just model parameters. Tests cover buffer/state restoration, deterministic replay
sampling, optimizer continuation, bit-exact dtypes, partial chunks, checksums,
failed publication, and saving before interrupted evaluation.

Validation on 2026-10-09: the broader CPU checkpoint/CLI/MuZero-component run
passed 57 tests. The subsequently added evaluation-interruption test passed;
the final shape-template/host-resume checks passed four tests on CPU and four
on TPU. The initial five-test TPU suite also passed packed-chess next-cycle
continuation (including full and compact saves). Next-cycle comparisons are
bit-exact within each platform; no CPU-versus-TPU numerical parity is claimed.
