# Hex4 → Hex5 → Hex6 replication and progression

**Latest user-directed change:** all MuZero checkpoint files and compressed
checkpoint archives were deleted on 2026-09-05, reclaiming 6,949,089,280 bytes.
The checkpoint hashes/paths below are historical provenance only. Metrics,
raw evaluations, and source snapshots remain. New runs default to no saves,
including no final save; `--no-save` forces this even against enabled configs.
The first full Hex6 process was stopped to apply the new policy and its partial
history retained. Replacement command:

```bash
uv run muzero-train evals/muzero/hex6-staged-seed1-nosave.toml --no-save --output artifacts/muzero/hex6-staged-seed1-nosave-20260905-a --hex-eval-engine-path /home/tedpsw/benzene-vanilla-cmake/build/src/mohex/mohex
```

Replacement W&B: https://wandb.ai/tdoubleu/nanoAlphaZero-muzero/runs/v9nl236a .
Active user service: `muzero-hex6-staged-seed1-nosave-20260905-a.service`.
Its resolved config verifies `save_checkpoints = false`, `checkpoint_period = 0`,
batch 8192, LR 0.0001, and the original 3500-cycle budget. The new supervisor
has no checkpoint archival step. The CPU no-save smoke completed four updates
with zero checkpoint files, overriding a config that requested saves every
cycle. Three focused defaults/checkpoint regression tests also passed.

Starting implementation: commit `8911c7e`. All three learning runs use fresh
seed-1 initialization, staged consume/drain replay, spatial h/g/f, five-step
unrolls, unchanged 16+8 two-rung search, and self-play-only targets. Production
AlphaZero defaults supply each board's model size, learning rate, batch sizes,
and optimizer-update budget; Hex5 uses the requested training batch 4096.
All runs use four TPU devices and W&B tables/Plotly monitoring.

1. **Hex4 replication:** does the current staged pipeline learn the small board
   with another seed? Compared with the earlier spatial Hex4 learning run,
   both seed and replay pipeline differ; this is not an isolated replay ablation.
   Keep b4c64, LR 0.001, train/self-play batch 1024, 5000 optimizer updates.
2. **Hex5 replication:** does the seed-0 result reproduce? Only the seed changes
   in the learning setup; current monitoring adds AZ-compatible diagnostics.
   Keep b4c128, LR 0.0001, train batch 4096, self-play batch 1024, 34000 updates.
3. **Hex6 smoke and learning:** can the same method scale to the next board
   under its AZ defaults? Keep b8c256, LR 0.0001, train/self-play batch 8192,
   105000 updates. First require a full-size four-device model/batch smoke
   completing collection, consume/drain, optimizer updates, save, and MoHex
   evaluation. The smoke reduces replay capacity and update count only.

For each run, inspect malformed/overwritten/invalid data counters, held-out
prediction errors, policy-only and search strength, and all periodic regressions.
The user requested proceeding to Hex6 after these replications. A correctness
or numerical failure changes that decision: diagnose it before launching the
next expensive run. MoHex opening conversion is reference-opponent evidence;
even 100% conversion does not establish broadly optimal play or search benefit.
Preserve all periodic results rather than selecting a favorable checkpoint.

Full replay snapshots are saved at completion, matching AZ's final-only default,
to fit available disk while preserving existing checkpoints. Every periodic
evaluation and held-out dataset is still retained. Runs are sequential, with
process/lock inspection before each TPU launch.

Commands (repository root; unique output directories):

```bash
uv run muzero-train evals/muzero/hex4-staged-seed1.toml --output artifacts/muzero/hex4-staged-seed1-20260905-a --hex-eval-engine-path /home/tedpsw/benzene-vanilla-cmake/build/src/mohex/mohex
uv run muzero-train evals/muzero/hex5-staged-seed1.toml --output artifacts/muzero/hex5-staged-seed1-20260905-a --hex-eval-engine-path /home/tedpsw/benzene-vanilla-cmake/build/src/mohex/mohex
uv run muzero-train evals/muzero/hex6-staged-smoke.toml --output artifacts/muzero/hex6-staged-smoke-20260905-a --hex-eval-engine-path /home/tedpsw/benzene-vanilla-cmake/build/src/mohex/mohex
uv run muzero-train evals/muzero/hex6-staged-seed1.toml --output artifacts/muzero/hex6-staged-seed1-20260905-a --hex-eval-engine-path /home/tedpsw/benzene-vanilla-cmake/build/src/mohex/mohex
```

Results and W&B URLs will be recorded as runs complete.

## Hex4 completed

W&B: https://wandb.ai/tdoubleu/nanoAlphaZero-muzero/runs/r7diskcw .
Finished at 12:58:56 UTC: 500 cycles / 5000 optimizer updates. Both modes
converted 4/4 known winning openings. Across both seats of the 240-opening
two-ply suite (480 games per comparison), policy versus random scored 99.58%,
search versus random 99.38%, and search versus policy 56.04%. Search's advantage
was larger earlier (91.88% at cycle 300); the final result is retained rather
than selecting that peak. Data validity checks stayed clean.

Measured final training-loop duration was 209.58 seconds, excluding initial
compilation/evaluation and replay warmup; full timestamps and allocator stats
are in the manifest and selected results JSON. The full replay checkpoint is
losslessly archived as `cycle-000500.safetensors.gz` in the run output directory:
1,177,261,438 bytes before compression, 530,536,055 after. Decompressed SHA-256:
`d3b9ee3d6d7338752cb6a48674e377a63230ff2742e0147bb7e8757e2fc43866`.
The archival check verified the entire decompressed byte stream before removing
the redundant uncompressed copy. Use `gzip -dk <checkpoint.safetensors.gz>`
before passing the restored `.safetensors` to the installed loader.

## Hex5 completed

W&B: https://wandb.ai/tdoubleu/nanoAlphaZero-muzero/runs/8mkkcpj9 .
Fresh seed-1 run started after Hex4 released the TPU and finished at
13:26:43 UTC: 1700 cycles / 34000 optimizer updates, 28,000,750 training
transitions plus replay warmup. Final conversion is **13/13 in both modes**.
Across both seats of 600 two-ply openings (1200 games per comparison), policy
versus random scored 99.92%, search versus random 99.42%, and search versus
policy 50.08%. Thus the narrow conversion result reproduces across two seeds,
but neither final checkpoint demonstrates a material search advantage.

Search conversion rose to 10/13 at cycle 600, regressed to 8/13 at 650, and
reached 13/13 by cycle 1000. Every periodic result remains retained. All
recorded anomaly maxima and nonfinite-metric counts are zero. Current-policy
held-out value error remains nonzero despite the saturated opening result.
Training-loop time was 1383.60 seconds; run time before final save was 1569.86
seconds. Median collection/drain/20-update time was 0.575 seconds, with
evaluation and host logging overhead recorded separately.

Final checkpoint: `artifacts/muzero/hex5-staged-seed1-20260905-a/cycle-001700.safetensors.gz`.
Verified decompressed SHA-256:
`d6da8cb26c322e141eec860ff147fa4e61f2db2739dd1fd09444357441379e7e`.
Archive size is 888,018,123 bytes (1,746,325,746 uncompressed).

## Hex6 smoke completed; full run launched

The smoke finished at 13:34:39 UTC, after Hex5 exited and released its TPU lock.
Full AZ Hex6 model and batch sizes were used, with two warmup batches and four
optimizer updates. It has 12,043,018 parameters; all four training shards had
shape `[2048, 6, 6, 4]`. All anomaly maxima and nonfinite-metric counts were zero.
Untrained MoHex conversion was 0/24 for both modes. The second collection/drain/
two-update cycle took 10.897 seconds before diagnostics. Peak reported allocator
usage was 1,541,408,256 bytes per device; this is not a complete HBM reservation
or compiler-memory measurement. The full update budget will take substantially
longer than Hex5; the short smoke does not determine its exact duration.

Smoke checkpoint: `artifacts/muzero/hex6-staged-smoke-20260905-a/cycle-000002.safetensors.gz`.
Verified decompressed SHA-256:
`db5b5747fe921db6c790e36e7d3ce224b9e2ddf49dc82a021abf5e116cc39208`.

Before the full run, three host timing metrics were added under `muzero/timing/`
to distinguish collection, replay insertion, and optimizer time. A four-device
CPU smoke completed four updates and saved successfully with these timers:

```bash
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 uv run muzero-train evals/muzero/smoke-cpu.toml --output artifacts/muzero/timing-cpu-smoke-20260905-a
```

This instrumentation is the only source change after `8911c7e`; model, search,
targets, replay selection, and optimizer settings are unchanged. The full run
stores the actual source snapshot and hashes. CPU timing is not TPU performance.

The full Hex6 run is supervised by
`artifacts/muzero/progression-20260905-tools/run-hex6.sh`, launched as the user
systemd service `muzero-hex6-staged-seed1-20260905-a.service`.
It executes the exact training command above, checks TPU ownership first, records
the training exit code, and on success losslessly archives the final checkpoint
and refreshes `results/progression-20260905.json`. Its log is
`/tmp/muzero-hex6-staged-seed1-20260905-a.log`; persistent run metrics and raw
evaluations are under `artifacts/muzero/hex6-staged-seed1-20260905-a/`.
Full-run learning results remain pending.

Initial full-capacity training passed: at cycle 14 / 420 updates, anomaly
counters remain zero and losses finite. Steady cycle time is about 15.4 seconds:
10.14 seconds collection, 0.41 seconds replay insertion/drain, and 4.86 seconds
for 30 optimizer updates. This implies roughly 15 hours of cycle computation
for the full budget, plus initialization and periodic evaluations; completion
time remains an estimate. See [DISK_USAGE-20260905.md](DISK_USAGE-20260905.md)
for measured storage use and projected growth. The archival supervisor now
checks free space before creating a second checkpoint copy; if insufficient,
it retains the complete original and writes an explicit skipped-archive record.

Full Hex6 W&B run:
https://wandb.ai/tdoubleu/nanoAlphaZero-muzero/runs/mb718t2r .
Resolved configuration confirms b8c256, self-play/train batch 8192, LR 0.0001,
2,048,000 replay positions, nine warmup batches, four TPUs, and 3500 cycles
with 30 optimizer updates each.

The initial detached shell did not survive tool process cleanup and exited
before creating a training output directory or initializing the TPU. The
replacement user service was verified active with trainer PID 3146862.
Launch and status commands:

```bash
systemd-run --user --unit=muzero-hex6-staged-seed1-20260905-a --property=WorkingDirectory=/home/tedpsw/nanoAlphaZero-main --property=StandardOutput=append:/tmp/muzero-hex6-staged-seed1-20260905-a.log --property=StandardError=append:/tmp/muzero-hex6-staged-seed1-20260905-a.log --setenv=PATH=/home/tedpsw/.local/bin:/usr/local/bin:/usr/bin:/bin /bin/sh /home/tedpsw/nanoAlphaZero-main/artifacts/muzero/progression-20260905-tools/run-hex6.sh
systemctl --user status muzero-hex6-staged-seed1-20260905-a.service
```
