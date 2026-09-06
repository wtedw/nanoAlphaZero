# MuZero status — 2026-09-05

September 6 pre-commit review: hardened the independent game audit against
missing pairing files, mismatched game rows and fractional actions. The MuZero
CPU suite passed 68 tests in 849.75 seconds with four virtual devices; the
updated audit checks separately passed six tests. Command:
`JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 uv run pytest tests/test_muzero*.py --ignore=tests/test_muzero_standalone.py -q`.
Standalone-export work is separate and excluded from this review/commit.
The running queue's frozen source/config hashes remained unchanged.

Latest continuation: [known-opening coverage plan](OPENING-COVERAGE-20260906.md).
Hex9's baseline and block-recomputation full-size TPU smokes passed, but
memory reservation remained high. The full-replay/full-evaluation smoke passed
at 06:01 UTC and the first 1,250-update Hex9 trial started. The new coverage runs stop after three
consecutive complete MoHex opening evaluations in both modes. Hex4/5 KL and
the earlier Hex6 CE run already have such recorded streaks. Hex7–9 remain open.

Latest work: [overnight Hex5–9 progression](OVERNIGHT-20260905.md), starting
19:39 UTC. Hex4/5 KL reruns completed. Hex6 KL reproduced the checked CE
trajectory and stopped after 352 cycles; cycle 350 converted 24/24 known
winning openings in both modes. Hex7 passed its full-size four-TPU smoke;
activation recomputation then reduced per-device allocator reservation from
30.4 to 10.9 GB. A 100-cycle / 3,000-update full-replay Hex7 learning trial
launched at 21:17 UTC with unchanged AZ batch size and learning rate. Hex8/9
small spatial CPU compatibility passed; Hex9's full-size TPU gate is next.
Hex7 completed 3,000 updates: both modes scored 100% against random on the
fixed two-ply slate, search scored 77.4% against its policy, and MoHex
known-winning-opening conversions were 2/27 search versus 0/27 policy.
This demonstrates learning and useful search, not expert-level Hex7 play.
Hex8's full-size TPU smoke passed at 23:25 UTC (four updates, both MoHex
evaluations, 14.2 GB reservation/device, no data anomalies). A 50-cycle /
2,500-update full-replay learning trial completed at 01:18 UTC: policy/random
86.2%, search/random 99.6%, search/policy 99.7%, but both lost all MoHex games
and terminal reward predictions remained weak. W&B:
https://wandb.ai/tdoubleu/nanoAlphaZero-muzero/runs/0gwa3qly.
Hex9's full-size four-TPU smoke started at 01:20 UTC.
New runs use no checkpoints, smaller independent diagnostic batches,
and an operator stop file that preserves final evaluations. Half-hour health
audits are recorded under `artifacts/muzero/night-watch-20260905/`.

The sections below retain the research history; historical checkpoint paths
are not available after the user's explicit checkpoint deletion request.

Branch: `research/muzero`, base ancestor `feature/hex-mohex-eval`.
Started with a clean tree at `deecd9fa5211f790a861a54e4094e4885e9e3d88`.
No commits, pushes, branch changes, or production core/MCTS/model edits.

## Implemented

Isolated representation, learned action-conditioned dynamics/reward, prediction,
unchanged production two-rung Gumbel search, full-episode Flashbax replay,
signed returns, unroll losses/masks, absorbing tails and truncated bootstraps.
Installed `uv run muzero-train` and `uv run muzero-eval` commands.
Full state checkpoints retain parameters, optimizer, replay, RNG and config.
W&B logs metrics without checkpoint artifact uploads. Output directories are
exclusive; checkpoints and raw evaluations are retained.

Default configurations now inherit the applicable AlphaZero configuration,
including batch sizes/LR and the explicit Hex 5 train-batch exception of 4096.
Historical smaller pilots remain labeled `pilot_v1`. Model architecture and
episode collection still differ; shared defaults do not match total resources.

## Correctness and compatibility

Focused tests cover reward indexing, delayed rewards and gradients, signed
opponent refutation, absorbing/truncated padding, root legality, simulator-free
search, masked losses, all-component gradients and exact checkpoint restoration
including a subsequent optimizer update. Full rollout-cap CPU smoke tests pass
for all 16 production environments, including train/save/load and real games
against random. Hex 5 uses a 4096-example optimizer batch in its compatibility
test. These are compatibility evidence, not strength or TPU performance claims.

The real four-device Hex 4 TPU smoke completed 64 games, 876 transitions and
four updates. Its first cycle took 9.16 s including compilation; second 0.017 s
at batch 32. Checkpoint: `artifacts/muzero/smoke-tpu-20260905-b/cycle-000002.safetensors`.

## Completed exploratory results

All scores below combine both colors over the same 240 legal two-ply openings
(480 games per comparison). This is not an optimal-play proof or independent
random sampling of the whole position space. No checkpoint was selected by score.

| Pilot, final checkpoint | Updates | Real transitions | Policy vs random | Search vs random | Search vs its policy |
|---|---:|---:|---:|---:|---:|
| seed 0, cycle 300 | 4800 | 932169 | 95.0% | 92.9% | 52.7% |
| seed 1, cycle 300 | 4800 | 919328 | 96.7% | 94.6% | 48.5% |
| seed 2, cycle 300 | 4800 | 930947 | 96.9% | 94.6% | 52.5% |
| duration seed 0, cycle 3000 | 48000 | see metrics | 99.8% | 100.0% | 53.1% |

The three short seeds establish learning against random, but not reliable
benefit from learned search. Checkpoint regressions are recorded: e.g. seed 0
search vs policy was 41.5% at cycle 100 and 48.3% at cycle 200.

CPU reference matches against `artifacts/alphazero_hex4.safetensors`:

| MuZero | Policy score vs AlphaZero | Search score vs AlphaZero |
|---|---:|---:|
| seed 0, 4800 updates | 9.8% | 15.0% |
| duration seed 0, 48000 updates | 42.1% | 44.6% |

This AlphaZero checkpoint is an unmatched strength reference, not a controlled
algorithm comparison. The CPU and TPU evaluation trajectories differ slightly
from floating-point/tie effects; raw results are preserved separately.

Duration improved reward prediction: held-out positive reward prediction is
0.974 at K=1, 0.914 at K=5 and 0.804 at K=10. Real-state value MSE against
initial-policy outcomes is still 1.15–1.24. Those outcomes are off-policy for
the final network; aggregate errors that include absorbing zeros are misleading.
This is not yet evidence of broadly strong Hex play.

## Costs and locations

The three short pilots each used 243858 parameters, about 0.062–0.068 seconds
per steady training cycle and 88–90 seconds from timed initialization/evaluation
through the final training cycle. JAX reported peak allocated memory of roughly
55–61 MiB on the most-used device; this excludes total machine memory and is
not a billing measurement. The duration pilot took about 349 s on the same
four-device TPU. Dollar cost has not been measured.

W&B runs:

- [pilot seed 0](https://wandb.ai/tdoubleu/nanoAlphaZero-muzero/runs/vajsi24m)
- [pilot seed 1](https://wandb.ai/tdoubleu/nanoAlphaZero-muzero/runs/c6gpy6bo)
- [pilot seed 2](https://wandb.ai/tdoubleu/nanoAlphaZero-muzero/runs/g3kvt8zx)
- [duration seed 0](https://wandb.ai/tdoubleu/nanoAlphaZero-muzero/runs/tv1yv1ws)
- [AlphaZero-defaults seed 0](https://wandb.ai/tdoubleu/nanoAlphaZero-muzero/runs/ylpoje5d)
- [AlphaZero-defaults seed 1](https://wandb.ai/tdoubleu/nanoAlphaZero-muzero/runs/kjy8jf3g)
- [AlphaZero-defaults seed 2 with MoHex](https://wandb.ai/tdoubleu/nanoAlphaZero-muzero/runs/tzld8hda)

Pilot checkpoints and raw evaluations:
`artifacts/muzero/hex4-vector-seed{0,1,2}-20260905-a/`.
Duration: `artifacts/muzero/hex4-duration-seed0-20260905-a/`.
Reference matches: `artifacts/muzero/hex4-seed0-reference-cpu-20260905-a/` and
`artifacts/muzero/hex4-duration-reference-cpu-20260905-a/`.
Each training run contains resolved config, manifest, metrics, source snapshot,
numbered checkpoints and raw evaluations. The first seed predates automatic
source hashes and saved held-out episodes; its source snapshot was preserved
manually while the process was active. Later runs include those automatically.

## Exact command patterns used

```bash
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \
  uv run pytest tests/test_muzero.py tests/test_muzero_compatibility.py -q

uv run muzero-train evals/muzero/hex4-baseline-seed0.toml \
  --output artifacts/muzero/hex4-vector-seed0-20260905-a
# Same command for seed1/seed2 with both occurrences of the seed changed.

uv run muzero-train evals/muzero/hex4-duration-seed0.toml \
  --output artifacts/muzero/hex4-duration-seed0-20260905-a

uv run muzero-train evals/muzero/hex4-alphazero-defaults-seed0.toml \
  --output artifacts/muzero/hex4-alphazero-defaults-seed0-20260905-a

JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \
  uv run muzero-eval artifacts/muzero/hex4-duration-seed0-20260905-a/cycle-003000.safetensors \
  --platform cpu --alphazero artifacts/alphazero_hex4.safetensors \
  --output artifacts/muzero/hex4-duration-reference-cpu-20260905-a
```

Early smoke commands used a temporary uv cache before the user requested plain
`uv run`; subsequent commands use normal uv access. The initial sandbox TPU
attempt failed to reach instance metadata, was stopped as a session-owned
process, and was retried outside the sandbox. No live lock was removed.

## Remaining work

Update after production-loop integration: all three production-defaults vector
runs completed 500 cycles / 5000 updates. Seed 2's periodic MoHex result at the
final checkpoint is 1/4 known winning openings converted by search and 0/4 by
policy-only play. A separate MoHex evaluation of the duration pilot converts
4/4 in both modes, with no unexpected wins from losing openings. This validates
that suite only. The user identifies the existing AlphaZero checkpoint as
essentially perfect play; earlier two-ply forced-opening scores are conditional
position tests, not unconstrained wins over that opponent.

The actual production `HexTrainingEvaluator` now runs periodically for both
MuZero modes, preserving raw per-opening moves, status and history. Production
value-head displays are reused through an inference adapter. W&B receives
opening-value MSE/sign accuracy, MoHex opening conversion, parameter/update
norms, LR, target balance and episode statistics.

The first detected MoHex binary under `buildmaster/` was incompatible with its
pattern files. The verified binary is
`/home/tedpsw/benzene-vanilla-cmake/build/src/mohex/mohex`; no external engine
files were modified. The MoHex four-device smoke passed, followed by a spatial
MuZero four-device smoke (64 games, 859 real transitions, four optimizer
updates, checkpoint and MoHex matches). The spatial model composes production
KataGo trunks and heads. Its long seed-0 experiment completed 500 cycles / 5000
updates under the inherited batch, LR, optimizer and search settings. Both
search and policy converted 4/4 known winning openings at the final checkpoint.
Across the forced two-ply suite, policy beat random in 100% of games and search
in 99.79%; search versus policy scored 50% when combined across seats. Thus the
large search benefit observed earlier in training did not persist at the final
checkpoint. One seed and this opening suite do not establish broad optimality.
W&B: https://wandb.ai/tdoubleu/nanoAlphaZero-muzero/runs/u3k6g3qn . Checkpoint:
`artifacts/muzero/hex4-spatial-seed0-20260905-a/cycle-000500.safetensors`. See
`hex4-spatial-seed0.toml` and RESEARCH.md for the controlled question.

Machine-readable results for all seven completed training pilots, every
evaluation checkpoint (including regressions), and checkpoint hashes are in
[results/pilots-20260905.json](results/pilots-20260905.json). Raw files and model
checkpoints remain in their original ignored artifact directories.

Broader-position evaluation and additional spatial seeds remain outstanding.
The user explicitly requested proceeding to Hex5 after the buffer audit, so
Hex5 implementation/learning validation is next; this does not strengthen the
Hex4 optimality claim. No larger-board learning or chess strength is yet claimed.

## Buffer audit and requested staging adaptation

See [BUFFER_AUDIT.md](BUFFER_AUDIT.md) for the code comparison, exact commands,
remaining collection differences and new staging design. The new path calls
production AlphaZero's **actual consume function**; production modules are
unchanged. It excludes exploration starts, retains fresh leftovers, drains
full batches of aligned unrolls into position replay, and primes replay without
optimizer updates. Historical episode-pipeline results remain preserved.

- Focused correctness/audit suite: 16 passed, including identical-trajectory
  comparisons to production backfill and malformed/duplicate-insertion guards.
- All-environment staged vector smoke: 16 passed in 337.88 seconds on CPU.
- Four-TPU staged Hex4 smoke: two warmup batches and four updates completed.
- Resuming its cycle-1 checkpoint reproduced **all 205 final tensors and
  metadata exactly**, including optimizer, replay, fresh leftovers and RNG.
  Safetensors byte headers differ in JSON ordering only.
- Hex5 four-TPU smoke launched with inherited b4c128 spatial architecture,
  self-play batch 1024, train batch 4096 and learning rate 0.0001. Output:
  `artifacts/muzero/hex5-staged-smoke-20260905-a`. Completed four updates;
  1,595,839 parameters, four local training shards of shape `(1024,5,5,4)`.
  The second collection/drain/two-update cycle took 0.217 seconds, excluding
  diagnostics/evaluation; whole cycle including those took 1.866 seconds.
  Peak device allocation was approximately 474 MB (reported allocator usage,
  not all reserved HBM). Both untrained modes converted 0/13 winning openings.
  Numeric diagnostic row counts were verified (25 opening rows, 625 reply
  rows, 600 legal replies); no images were generated.

One-move diagnostics now log **numeric W&B tables**, plus local `heads.npz`
and `tables.json`; no Matplotlib dependency or image logging remains. The
user's existing custom chart was initially absent from this checkout; it was
subsequently located in the sibling AZ repository and adapted (see METRICS.md).
The earlier local Hex4 smoke images remain preserved; no long run uploaded them.

Commands used (each ran after the preceding TPU process exited):

```bash
uv run muzero-train evals/muzero/hex5-staged-smoke.toml --output artifacts/muzero/hex5-staged-smoke-20260905-a --hex-eval-engine-path /home/tedpsw/benzene-vanilla-cmake/build/src/mohex/mohex
uv run muzero-train evals/muzero/hex5-staged-seed0.toml --output artifacts/muzero/hex5-staged-seed0-20260905-a --hex-eval-engine-path /home/tedpsw/benzene-vanilla-cmake/build/src/mohex/mohex
```

The Hex5 seed-0 command above ran at:
https://wandb.ai/tdoubleu/nanoAlphaZero-muzero/runs/wmxdk2pi . Its 82 frozen
warmup batches produced 1,776,072 real transitions, excluded 462,778 exploration
starts, inserted 1,312,768 eligible positions, and retained 526 fresh leftovers
with no overwrites. At the start of training, regular collection/drain/
20-update cycles take approximately 0.55 seconds before periodic evaluation.
Completion and strength results follow below. The first numeric tables
were logged successfully. The latest focused audit/spatial/production-eval
regression command passed 20 tests in 48.81 seconds:

```bash
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 uv run pytest tests/test_muzero_data_audit.py tests/test_muzero_spatial.py tests/test_hex_eval.py tests/test_module_layout.py -q
```

## Update: Hex5 seed 0 completed; AZ monitoring standardized

The seed-0 run above completed at 12:19:05 UTC with **1700 cycles / 34,000
optimizer updates**, 27,484,947 real training transitions (plus 1,776,072
warmup transitions), and 1366.45 seconds reported elapsed before final saving.
Final MoHex conversion is **13/13 known winning openings in both modes**.
Final policy-only versus random is 99.83%, search versus random 100%, and
search versus policy 50.08% across both seats of the forced two-ply suite.
This is one seed and narrow opening evidence; it does not establish optimal
Hex5 play or a persistent search advantage. Preserve the full periodic history.
Full checkpoints are at `artifacts/muzero/hex5-staged-seed0-20260905-a/`
(`cycle-000850.safetensors`, `cycle-001700.safetensors`).

The requested chart code was located in `/home/tedpsw/az`, not this checkout.
Its board table schema and Plotly chart behavior are now adapted into MuZero,
with a direct read-only equivalence test against the original helper. Standard
AZ chart keys are restored. No Matplotlib or raster upload is used.

See [METRICS.md](METRICS.md) for standardized AZ metric names and definitions.
New runs include self-play and reward checks, fresh/valid consume diagnostics,
replay occupancy, root-batch provenance and counters, policy entropies, WDL
head probabilities, optimizer/head norms, endgame-distance losses, opening and
sample distributions, timing and original Hex diagnostics. MuZero-only
diagnostics live under `muzero/`. Historical logs retain their original schema.
Conditional features absent from this baseline are explicitly listed in that
document rather than represented by fabricated zero metrics.

Validation:

- Final combined MuZero suite: **39 passed** in 521.45 seconds, including
  all 16 environment compatibility cases. Production Hex evaluation and module
  layout regressions: **13 passed** in 5.25 seconds. Commands:

  ```bash
  JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 uv run pytest tests/test_muzero.py tests/test_muzero_data_audit.py tests/test_muzero_metrics.py tests/test_muzero_spatial.py tests/test_muzero_compatibility.py -q
  JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 uv run pytest tests/test_hex_eval.py tests/test_module_layout.py -q
  ```

- 19 existing focused correctness/staging/spatial tests passed after adding
  diagnostic provenance; 6 chart/metric/spatial tests passed, including direct
  comparison to the original AZ helper.
- The four-TPU Hex4 monitoring smoke reproduced all **177 training/optimizer
  tensors bitwise**, despite the added monitoring.
- Full-size Hex5 four-TPU W&B monitoring smoke completed four updates, checkpoint
  save, tables/Plotly/histograms and MoHex evaluation. It emitted 258 scalar
  metrics and 4096 valid root samples, with no malformed/overwritten fresh data.
  Run: https://wandb.ai/tdoubleu/nanoAlphaZero-muzero/runs/iwxbzslo .
  Output: `artifacts/muzero/hex5-az-metrics-smoke-20260905-a`.

```bash
uv run muzero-train evals/muzero/hex5-az-metrics-smoke.toml --output artifacts/muzero/hex5-az-metrics-smoke-20260905-a --hex-eval-engine-path /home/tedpsw/benzene-vanilla-cmake/build/src/mohex/mohex
```

## Seed-1 replication and Hex6 progression

**Current experiment:** the user stopped the cross-entropy Hex6 run and requested
KL reruns of Hex4, Hex5, and Hex6. See [KL-20260905.md](KL-20260905.md) for the
loss comparison, fixed settings, tests, and live sequential queue locations.

**Checkpoint retention update:** the user requested removing all MuZero
checkpoints and disabling saves on 2026-09-05. All 49 checkpoint/archive files
under `artifacts/muzero/` were removed (6,949,089,280 allocated bytes). Historical
checkpoint locations below are no longer available; raw evaluations and metrics
are retained. Training now defaults to no checkpoint writes, including at exit.
Hex6 is restarted with `hex6-staged-seed1-nosave.toml` and `--no-save` because
the existing process had already loaded its old save policy.

The requested sequential Hex4 → Hex5 → Hex6 runs are tracked in
[PROGRESSION-20260905.md](PROGRESSION-20260905.md), with exact commands,
checkpoint archive hashes, W&B links, and limitations. Selected complete-run
histories are in `results/progression-20260905.json`. These use the committed
`8911c7e` implementation and new portable configs; local run outputs remain
outside Git.
