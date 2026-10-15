# Launching jobs on a TPU pod

The `pod` command runs from worker 0 of an existing Google Cloud TPU pod.
It discovers the workers from VM metadata: no `podips.txt`, personal shell
aliases, or manually entered addresses are required. This first version supports
one TPU slice and a shared checkout. It does not create cloud resources.

## Getting started

Clone this repository on worker 0, then run from the repository root:

```bash
uv run pod setup
uv run pod status
uv run pod run --cpu -- python -c 'import socket; print(socket.gethostname())'
```

Prerequisites are `uv`, Git, system Python 3.8+, SSH access from worker 0 to all
workers, and passwordless sudo for NFS and read-only ownership checks. Training
uses Python 3.11+ installed by uv; the system Python only bootstraps SSH workers.
For the Ubuntu TPU VM images, install `nfs-kernel-server` on worker 0 and
`nfs-common` on workers if missing. Setup reports SSH/authentication failures by
worker; configure the pod's SSH keys using your normal Google Cloud procedure.

`setup` reuses an existing NFS export or creates an export restricted to the
discovered worker addresses. A repository inside `/nfs_share` uses that share;
otherwise it exports the repository directory itself. Override with
`uv run pod --share /your/share setup`. The checkout must be inside the share.
Workers mount it at the same absolute path. Existing unrelated mounts and
nonempty directories are never covered by a new mount.

Dependencies are installed from `uv.lock` into a separate local environment on
each host under `~/.venvs/nanoalphazero/<checkout-id>`. The environment and uv
cache stay off NFS, with `UV_LINK_MODE=copy`, matching the existing `az1`
Justfile convention. Setup is repeatable; rerun it when the lockfile changes.
Setup verifies dependency imports with the CPU backend before marking a worker
ready. Repair an interrupted installation with `uv run pod setup --reinstall`.
Install optional tracking support with `uv run pod setup --extra trackio`.
Trackio uses `wtedw/trackio`'s `extra` branch, pinned to a commit in `uv.lock`;
no sibling Trackio checkout is required.
The fork builds its dashboard during installation, so Node.js and npm must be
available on each worker's SSH `PATH` when installing this extra.

Setup logs are saved under `logs/pod/setup-*/worker-*.log`. Existing installed
environments are left intact when setting up a different checkout/worktree.

## Running commands

```bash
# Every worker runs the same command, using its local project environment.
uv run pod run --cpu -- python -c 'import os; print(os.environ["NANOAZ_POD_PROCESS_ID"])'

# Bound an experiment's runtime. No automatic restart occurs.
uv run pod run --cpu --timeout 60 -- python your_cpu_smoke.py

# TPU ownership is checked on every worker before any command starts.
uv run pod run -- python your_distributed_program.py
```

Arguments after `--` are preserved literally. Use `bash -c '...'` explicitly
if a job needs shell syntax. Standard input belongs to the launcher's liveness
checks, so job commands must be noninteractive.

Each launch gets a directory under `logs/pod/` containing `launch.json`,
`worker-0.log`, `worker-1.log`, etc., and `result.json`. The launcher prints the
directory and streams output directly to your terminal, prefixed with the
worker index. Keep this command in the foreground; Ctrl+C in this terminal
cancels the launch on every worker. No separate `tail` command is needed.
A worker failure identifies its index/IP and
log, stops the other workers in that launch, and returns a nonzero exit code.
Ctrl+C, a launch timeout, or loss of the launcher connection also stops that
launch's process groups. SSH loss has a 20-second worker heartbeat deadline.
There is no broad `pkill`, TPU lock deletion, or automatic restart.

Before launch, workers verify identical code (including dirty and untracked
nonignored files), installed dependencies, and lockfiles. A fresh file exchange
checks shared-filesystem visibility in both directions. Every 30 seconds during
a job, another check verifies that the share is still accessible. A failed check
stops the launch; it does not create an independent remount watchdog or attempt
to repair storage under a running training process. A missing mount is restored
at setup or before the next run.

`pod status` reports mounts, environment readiness, TPU ownership, and active
jobs started by this launcher without initializing JAX. Use `pod status --json`
for full reports. TPU ownership checks include root-visible processes; an
existing `grpc_tpu_worker` lock owner is reported as occupied. This launcher does
not stop system TPU services or other jobs. `--cpu` forces `JAX_PLATFORMS=cpu`
and permits CPU validation while the TPU is occupied.

## Distributed JAX startup

The launcher starts processes; the program initializes JAX before device access.
Root MuZero accepts:

```bash
uv run pod run --timeout 900 -- python muzero.py --env hex4 --platform tpu \
  --distributed.enabled --num-iters 100 --replay-buffer-warmup-steps 100 \
  --selfplay-batch-size 32 --train-batch-size 32 \
  --selfplay-buffer-consume-size 32 --selfplay-buffer-min-len 1 \
  --selfplay-buffer-max-len 256 --replay-buffer-max-len 128 \
  --conv-width 16 --conv-depth 1 --cycle-n-selfplay 10 --cycle-n-train 2 \
  --ladder-period 0 --diagnostic-period 0 --no-enable-trackio --no-enable-wandb
```

This is a bounded training smoke configuration. Batch sizes are global and
buffer time capacities do not multiply by the number of hosts. Compressed
multihost checkpoints use the shared filesystem (see below). Chess ACPL supports
multiple hosts. Ladder and MoHex evaluations still need validation; the CLI
rejects those on multiple hosts for now.
Existing AlphaZero commands do not
automatically become distributed merely by being launched on several hosts.

For small CPU topology tests the helper also accepts explicit
`--distributed.coordinator-address`, `--distributed.num-processes`, and
`--distributed.process-id`; provide all three together. The launcher exposes
`NANOAZ_POD_PROCESS_ID`, `NANOAZ_POD_PROCESS_COUNT`, `NANOAZ_POD_COORDINATOR`, and
`NANOAZ_POD_LAUNCH` to custom test programs. Normal TPU JAX startup uses its own
metadata discovery.
JAX process indices may differ from launcher worker indices on TPU. Use
`jax.process_index()` for JAX state ownership, not the launcher's log index.

## Tracking

Start one dashboard from the shared checkout with:

```bash
uv run --extra trackio trackio-server
```

This serves `artifacts/trackio` on localhost port 7862. Override with `--port`,
`--host`, `--project`, or `--data-dir`; `TRACKIO_DIR` is also respected.
On your laptop, forward the port with `ssh -N -L 7862:127.0.0.1:7862 <pod-host>`
and open <http://localhost:7862>. Run the dashboard once, without `pod run`.

Install the fork on every worker once with `uv run pod setup --extra trackio`.
Then replace `--no-enable-trackio` with `--enable-trackio` in the example above
and set `--diagnostic-period 10` (or your preferred cadence). Do not run setup
without the extra afterward: that removes the optional dependency.
Multihost Trackio defaults to `artifacts/trackio` in the shared checkout, so
the run remains visible even when JAX process 0 is another VM. An explicit
`TRACKIO_DIR` is respected; pass it to workers with
`pod run -- env TRACKIO_DIR=/shared/path python ...`. Your fork uses JSONL
fragments on NFS and imports them when the dashboard reads the project.
Explicit tracker resume also imports pending fragments before looking up the
run, then broadcasts its identity and last persisted cycle to every process.

Only JAX process 0 creates or writes the Trackio/W&B run. Every process receives
the same tracker identity and resume cutoff and enters the same collective
diagnostic schedule. Global training scalars are already reduced by JAX; they
are not summed again across hosts. Provenance histograms select at most 4096
evenly spaced global samples per field on device, keeping the validity mask
aligned, then use `multihost_utils.process_allgather(..., tiled=True)` before
NumPy conversion. Histograms therefore describe a bounded sample when a cycle
contains more than 4096 training windows. Scalar counts remain exact.

Fixed game probes run on process 0 using local replicated model parameters.
The global compression probe runs on every process. Device memory is queried
locally and aggregated across hosts. Only process 0 constructs backend media
and writes files. Tracker initialization errors are broadcast to every process.
Ladder and MoHex evaluations remain separate work.

## Checkpoints

Add these options to the same MuZero pod command on every worker:

```bash
--ckpt.save artifacts/chess-pod.ckpt --ckpt.every 100 --ckpt.mode full
```

The path must be on the shared filesystem. Saving is synchronous between cycles,
on the first cycle, every 100 cycles, and the final cycle. Full snapshots include
model parameters, optimizer, environment/games, self-play and replay buffers,
RNG, counters, cycle, resolved configuration, and tracker identity. Partial
snapshots (`--ckpt.mode partial`) omit games and buffers.

Resume using the original training options plus:

```bash
--ckpt.resume artifacts/chess-pod.ckpt --ckpt.save artifacts/chess-pod.ckpt
```

Resume does not enable saving by itself. Use the same host/device topology,
shard geometry, and resolved training configuration. Full resume skips buffer
warmup. Replicated model parameters can also be read separately on one host
through `nanoalphazero.ckpt.load_model`, without reading experience buffers.

Each host writes its unique local shards with bounded 4 MiB compression chunks;
replicated tensors are stored once. Host 0 holds the checkpoint lock and publishes
`LATEST` atomically only after every writer succeeds and shard coverage is checked.
Ordinary I/O errors propagate to all hosts. A failure before publication leaves
the previous checkpoint authoritative. A dead worker relies on JAX timeouts and
launcher supervision; an unpublished directory may remain after abrupt death.
The old published generation is removed after a successful publication, so allow
space for both generations during a save. Do not remove incomplete generations
while a save is active.

The existing resume-budget issue is unchanged: the host loop subtracts optimizer
steps from a self-play-step budget. This can run too many or too few cycles when
resuming if updates and self-play steps advance at different rates. Checkpoint
continuation tests compare the next cycle directly, independently of that budget.

CPU validation uses two actual JAX processes, including full chess state and
next-cycle equivalence. A bounded chess test also passed on four TPU hosts / 16
TPU v4 devices: full/partial snapshots, exact next-cycle continuation, and a fresh
algorithm restore. Injected write/publication failures preserved the previous
checkpoint on the same pod. This validates correctness with small buffers,
not production chess checkpoint throughput or peak memory.

```bash
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \
  uv run pytest tests/test_distributed_checkpoint.py tests/test_compressed_checkpoint.py

# Only when the pod is free; use a new shared output directory:
uv run pod run --timeout 900 -- python tests/helpers/checkpoint_smoke.py \
  --platform tpu --case chess --directory artifacts/checkpoint-chess-smoke
```

## Multihost chess ACPL

Add the same ACPL options to every worker's MuZero command:

```bash
--ladder-period 0 --acpl.period 100 \
--acpl.stockfish-path /usr/local/bin/stockfish \
--acpl.output artifacts/chess-acpl --acpl.num-games 496 \
--acpl.depth 18 --acpl.workers 8
```

Stockfish must exist on every host, and the output path must be shared. Workers
are per host; the bounded worker count above deliberately differs from the old
512-worker default. Game count is global, so 496 games on four hosts means 124
distinct games per host. Only JAX process 0 publishes the combined PGN/summary
and tracker artifacts. Evaluation pauses training synchronously.

Validated on four TPU hosts / 16 devices using a restored MuZero model: 16 games
twice (identical trajectories) and 496 games once, all analyzed without failures
or duplicate game IDs. These used depth 2 and a 28-ply cap. They establish batch,
partition, aggregation, and engine compatibility; they do not benchmark full
games at depth 18. The original AZ scorer matched all 16 smoke games to 1e-12.

To repeat a bounded inference-only check from a full or partial chess checkpoint:

```bash
uv run pod run --timeout 600 -- python scripts/smoke_acpl.py \
  --algorithm mz --checkpoint artifacts/chess-pod.ckpt --platform tpu --distributed \
  --games 16 --max-plies 28 --depth 2 --workers 2 --repeats 2 \
  --output artifacts/acpl-pod-smoke-NEW
```

## Validation

```bash
JAX_PLATFORMS=cpu uv run pytest tests/test_pod.py tests/test_distributed_startup.py tests/test_distributed_runtime.py tests/test_muzero_cli.py
```

The runtime test uses two CPU processes, distinct local array slices, and a
global JAX reduction. These tests validate launcher/startup behavior, not TPU
training performance or complete multihost training support.

For real TPU validation (after checking availability), run sequentially:

```bash
uv run pod run --timeout 240 -- python tests/helpers/pod_smoke.py --platform tpu
uv run pod run --timeout 900 -- python tests/helpers/pod_hex_smoke.py --env hex4
uv run pod run --timeout 900 -- python tests/helpers/pod_hex_smoke.py --env hex8
uv run pod run --timeout 900 -- python tests/helpers/pod_hex_smoke.py --env hex9
```

The Hex helper uses the real MuZero assembly and host loop, requires optimizer
updates and finite metrics, checks disjoint lane ownership and distinct board
states, and compares the full model/optimizer/RNG digest across hosts. These
are small-network correctness runs, not convergence or strength evaluations.
