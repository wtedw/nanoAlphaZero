# Cycle-3600 value-scale sweep and frozen readout probes

2026-10-17 Asia/Tokyo. User requested frozen experiments, with no training
settings changes or commits. All runs were CPU-only. No production weights,
configuration, or dependencies were changed. Diagnostic readouts are disposable
fits; none was installed into the chess model.

## Main results

**The scale sweep did not identify a validated fix.** Selection chose scale zero,
which reproduces prior argmax. The best positive scale, 0.05, was nominally better
than current search on held-out positions but worse than the prior, with wide
paired confidence intervals. It fails the proposed gate of beating both.

**Some useful information is present in learned successor states but is read
poorly by the current head.** A linear probe of pooled recurrent features,
supervised by frozen fresh-board values, improved held-out within-root Stockfish
ranking from 0.306 to 0.444. The paired gain is 0.138 [0.058, 0.216]. The fresh
head achieves 0.436 on those roots. A probe of the existing value head's own
112-dimensional penultimate activations also improves ranking when supervised
by Stockfish (0.398, paired gain 0.092 [0.041, 0.144]).

These are evidence for a readout contribution, not proof that dynamics is correct
or that a head replacement fixes play. Argmax move-loss improvements are uncertain.
Earlier one-step/accumulated dynamics errors remain measured. These probes cannot
identify how training produced the mismatch or when it began.

## Checkpoint and preregistered splits

Verified the compressed scalar `cycle` is **3600**, not inferred from a filename.
Generation: `gen-d6806e5f0ce64ad0bf2710a9f266c2df`, from
`artifacts/chess-pod-exact-m16-n32-local-gather-20261015/latest.ckpt`.
A new diagnostic checkpoint directory hard-links the immutable generation and
copies its LATEST pointer; the source checkpoint is not modified.

Output root: `artifacts/latest-value-scale-20261017/`.

- Sweep selection: first 96 roots of the previous 192-root suite.
- Sweep held-out: 96 new roots from different PGN games, excluded from all 192
  earlier search-probe games. Fixed seed 2026101702; balanced early/middle/late
  ply ranges, skipping games too short for the assigned range.
- Readout train/validation: first 128 / last 64 of the previous 192 roots.
- Readout test: the same 96 new held-out games used for the sweep test.
- All splits are by game, one position per game. The readout training/validation
  sets overlap the sweep-selection suite by design; neither uses test labels
  for fitting or hyperparameter selection.

Full PGN histories were retained, and root PGX legality matches python-chess.
The model uses the saved 384x10 architecture. Search remains m=16/n=32,
root temperature 1.5, mixed-value completion, maxvisit_init=50, no Q rescaling.
Only the frozen diagnostic search call's `value_scale` argument varies.
Gumbel noise is zero for this deterministic evaluation; this is not a noisy
self-play training experiment.

## Frozen scale sweep

Scales were declared before measurement: 0, 0.05, 0.1, 0.2, 0.3, 1.0.
Selection minimizes mean candidate-relative clipped CP loss; ties prefer the
larger scale. Scale 1 is the current configuration.

| Scale | Selection mean CP loss (96 roots) |
|---|---:|
| 0 | 42.90 |
| 0.05 | 49.57 |
| 0.1 | 53.46 |
| 0.2 | 52.32 |
| 0.3 | 58.99 |
| 1 (current) | 67.31 |
| Prior only | 42.90 |

The primary selected scale is zero. After seeing this selection result, but
before examining test results, a secondary test of the best positive scale
(0.05, also selected exclusively on selection roots) was added. It is explicitly
a secondary comparison, not a replacement for the primary selection rule.

| Held-out decision rule | Mean CP loss (96 new roots) |
|---|---:|
| Prior only / scale 0 | 53.66 |
| Scale 0.05 | 60.88 |
| Current scale 1 | 66.91 |

Paired improvement, positive means better:

- Scale 0.05 over current search: **6.03 CP [-8.34, 18.47]**.
- Scale 0.05 over prior: **-7.22 CP [-24.59, 7.66]**.
- Primary scale 0 over current search: **13.25 CP [-6.46, 33.42]**.

All 96 scale-zero choices equal prior argmax. No positive scale beats both
comparators, and even the nominal differences have wide intervals. Do not change
the training scale based on this test. It does not prove a small positive scale
cannot help at other budgets/checkpoints, but the proposed validation gate failed.

## Reference scoring

For each of all 288 roots, score up to 16 highest-prior legal candidates. With
noise zero, these cover the search candidates and all selected moves; membership
is checked when scoring each search decision. Score once and reuse across arms.
Stockfish: depth 18, time cap 20 seconds per root, MultiPV restricted to candidates,
Threads=1, Hash=64 MiB, 16 scoring workers. All **3,989** candidates reached depth
18. Retain PGN history when submitting boards to Stockfish.

CP scores are from the root player's perspective, clipped to +/-1000; loss is
relative to the best tested candidate, not unrestricted optimal play. These are
single-position decision losses, not full-game ACPL. Confidence intervals use
5,000 root/game bootstrap samples. The 96-root test has limited power for rare
large blunders, which is visible in the intervals.

## Head versus dynamics: what was actually fitted

Frozen features were extracted for every candidate, with no MuZero optimizer:

- The existing value head's activated penultimate features: 112 dimensions.
- Pooled latent features: four 4x4 quadrant means plus global channel means and
  standard deviations, 2,304 dimensions.
- Full spatial latents: 24,576 dimensions, stored as float16 and converted to
  float32 for preprocessing; no spatial pooling or fitted projection.

Each view is extracted from both `g(h(s), a)` and `h(real successor)`.
A no-dynamics control concatenates pooled root features with one-hot from/to
squares (normalized for side to move) and promotion type. It is a limited
control; it does not encode exact successor chess rules.

The extracted penultimate features reproduce the original head's outputs through
its unchanged final dense layer and WDL conversion to max absolute error
**2.38e-7**, for both real and learned states. Thus the readout probe uses the
intended activations, not a mislabeled intermediate tensor.

Known terminal successors are excluded from probe fitting/evaluation: 15 removed,
3,974 live candidates retained across 128/64/96 train/validation/test roots.
No root disappears. This separates the learned value readout from exact terminal
reward substitutions. Therefore probe argmax losses below have a different
candidate universe than the scale table and must not be directly compared to it.

For each feature view, fit linear ridge and RBF kernel readouts, with RBF gamma
0.3/1/3 and ridge alpha 0.001/0.01/0.1/1/10. Feature normalization uses training
statistics only. Kernels and labels are centered within roots so across-root
material/outcome differences cannot dominate the regression objective. Test
features can be centered without test labels; centering predicted values by a
root constant does not change their ranking.

Two separate training-label conditions:

1. Stockfish CP clipped to +/-1000 and divided by 1000.
2. The unchanged fresh-board value head, used as a frozen teacher.

Hyperparameters minimize validation mean CP decision loss, separately for each
view and label condition. Test data are not used to select hyperparameters.
All 14 view/label results are retained, including poor results. These are
exploratory diagnostic comparisons, not a preregistered claim that any one
replacement head improves long-run training. No kernel/probe replaces the model.

### Held-out readout results

Spearman is averaged within roots; 82/96 roots have sufficient nonconstant rank
data. CP loss uses all 96 roots.

| Readout / training label | Mean Spearman vs Stockfish | Argmax CP loss |
|---|---:|---:|
| Original recurrent value head | 0.306 | 88.79 |
| Original fresh-board value head | 0.436 | 67.54 |
| Recurrent penultimate / Stockfish | 0.398 | 76.56 |
| Real penultimate / Stockfish | 0.476 | 62.00 |
| Recurrent pooled / fresh teacher | 0.444 | 75.86 |
| Real pooled / Stockfish | 0.489 | 59.82 |
| Recurrent full spatial / Stockfish | 0.310 | 106.90 |
| Real full spatial / Stockfish | 0.423 | 71.40 |

The recurrent pooled/fresh-teacher arm had the lowest validation CP loss among
learned-state probes (36.33); it selected a **linear** kernel, alpha 0.1.
The recurrent penultimate/Stockfish arm selected linear, alpha 0.1 as well.

For recurrent pooled/fresh teacher:

- Paired rank gain over original recurrent head: **0.138 [0.058, 0.216]**.
- Paired rank difference against original fresh head: **0.008 [-0.055, 0.070]**.
- Paired CP improvement over recurrent head: **12.93 [-25.10, 46.94]**.
- Paired CP improvement over fresh head: **-8.32 [-42.38, 21.57]**.

For recurrent penultimate/Stockfish:

- Paired rank gain over original recurrent head: **0.092 [0.041, 0.144]**.
- Paired CP improvement: **12.23 [-18.21, 38.60]**.

The no-dynamics control's CP loss is 207.28 with Stockfish labels and 212.50
with fresh labels: it does not explain the recovered candidate ranking.
The poorly performing high-dimensional probes and several teacher-label arms
also illustrate capacity/data limitations; failed probes do not prove missing
state information. Full results are in `readout-results.json` rather than only
the favorable rows above.

## Interpretation and next step

Useful action-ranking information can be recovered from frozen learned states
without updating dynamics, including from the existing head's penultimate
activations. This argues against a blanket claim that dynamics erased all useful
information, and supports a readout/supervision mismatch as part of the problem.
It does not distinguish whether earlier dynamics training produced unusual
features that the shared head reads badly, versus inadequate training of the
head itself. Both can contribute. Absolute value calibration is not established
by a root-centered ranking probe; it cannot be directly plugged into search.

Before a long-run change, the most informative next intervention would be a
bounded **head-only** calibration/distillation experiment with frozen
representation and dynamics, retaining real-board supervision to check that
root predictions are not damaged. Evaluate new held-out move choices and search,
not just ranking/MSE. If that succeeds, it identifies a readout-level remedy;
if it fails, do not infer dynamics is irreparable from these small probes.
No such intervention was run or installed here. No training setting was changed.

## Reproduction

From the multihost worktree root, commands used (steps run only after their input
files are complete):

```bash
JAX_PLATFORMS=cpu taskset -c 0-15 uv run --no-sync python -u \
  tests/helpers/sweep_latest_value_scale.py artifacts/latest-value-scale-20261017 --phase prepare
JAX_PLATFORMS=cpu taskset -c 0-15 uv run --no-sync python -u \
  tests/helpers/sweep_latest_value_scale.py artifacts/latest-value-scale-20261017 --phase selection
JAX_PLATFORMS=cpu taskset -c 16-31 uv run --no-sync python -u \
  tests/helpers/sweep_latest_value_scale.py artifacts/latest-value-scale-20261017 --phase features
taskset -c 32-63 uv run --no-sync python -u \
  tests/helpers/analyze_latest_scale.py artifacts/latest-value-scale-20261017 --phase score
uv run --no-sync python tests/helpers/analyze_latest_scale.py \
  artifacts/latest-value-scale-20261017 --phase choose
JAX_PLATFORMS=cpu taskset -c 0-15 uv run --no-sync python -u \
  tests/helpers/sweep_latest_value_scale.py artifacts/latest-value-scale-20261017 --phase test
JAX_PLATFORMS=cpu taskset -c 32-47 uv run --no-sync python -u \
  tests/helpers/sweep_latest_value_scale.py artifacts/latest-value-scale-20261017 --phase test --positive-only
uv run --no-sync python tests/helpers/analyze_latest_scale.py \
  artifacts/latest-value-scale-20261017 --phase test
OPENBLAS_NUM_THREADS=16 OMP_NUM_THREADS=16 taskset -c 16-31 uv run --no-sync python -u \
  tests/helpers/probe_latest_readouts.py artifacts/latest-value-scale-20261017
git diff --check
```

Use a new output directory for independent repeats. `prepare` expects the current
source checkpoint to be cycle 3600 and hard-links its generation; it is not a
generic arbitrary-cycle runner. All search/probe/reference commands completed.
Production action/target equivalence at scale 1, disjoint game splits, final-head
reconstruction, scale-zero/prior equivalence, Python compilation, and diff checks
passed. The reference evaluator reached its requested depth for every candidate.

Artifacts: `manifest.json`, `positions.json`, pinned `checkpoint/`, selection/test
search JSONs, `chosen-scale.json`, `heldout-scale-results.json`, `reference.json`,
`features.npz`, `feature-rows.json`, `readout-results.json`,
`readout-test-predictions.npz`, `paired-readout-comparisons.json`, and `checks.json`.
The frozen original model was never optimized. The readout helper does not even
load the model; it fits exported features and writes predictions/results only.
