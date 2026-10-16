# Replay coverage, value masks, and root/recurrent errors

2026-10-17 Asia/Tokyo. No training settings, production implementation, or
dependencies changed; no commit. This audit uses CPU synthetic production-path
tests, the actual Trackio run `fancy-breeze-4`, and the preserved cycle-3200
frozen-model diagnostic. It does not restart training or claim TPU validation.

## A concrete replay sampler defect

The pinned Flashbax implementation (`e0199d7bb232c622a19d3c28f9d6b34eb8215eab`)
uses the **same PRNG key** for the random row index and random time index in
`calculate_uniform_item_indices`. Fresh keys across optimizer updates do not
remove the within-draw correlation. The production loop does split its key
correctly between updates. Neither sampling nor draining uses model loss to
prefer easy positions.

Tested through `make_muzero_replay_buffer`, including the active `features`
layout, with 8192 rows, 1024 time slots, and batch 8192. Synthetic entries encode
their exact row/time identity. We drew 128 batches (1,048,576 entries), using
128 different keys. A control draws row and time with independent keys while
preserving the same eligible columns. No installed sampler was patched.

| Buffer state | Production distinct entries | Independent-key control |
|---|---:|---:|
| Full, write index 0 | 927,401 | 985,482 |
| Full, write index 37 | 927,401 | 985,482 |
| Half-filled, write index 512 | 8,192 | 927,626 |

For a full buffer, the implementation excludes two columns even though the
requested sequence length is one. Its time draw therefore has upper bound 1022,
not 1024. All observed row/time pairs satisfy
`(column - row) % 2 == current_index % 2`. Half the otherwise eligible pairs
are inaccessible at a fixed state. This is **not** a claim that a full buffer
collapses to 8192 entries: all 8192 rows and all 1022 eligible columns appear,
but their joint distribution is correlated. Marginal histograms miss this.

At half capacity, every draw instead satisfies
`column == (row + 2) % 512`, so only 8192 of 4,194,304 populated entries are
reachable at that fixed state. A single batch had 5,189 distinct entries, and
the first two batches shared 3,305 entries. At full capacity those numbers were
8,182 and 18. The half-capacity failure is much more severe.
Across 20 batches (the live run's updates per cycle), the full-buffer sampler
visited 160,701 distinct entries versus 162,208 in the independent control:
about 0.93% fewer. At half capacity, the same comparison is 8,192 versus 160,626.
Thus the support restriction should not be reported as a 50% loss in the number
of distinct examples per full-buffer training cycle.

The full-buffer parity constraint follows from reusing random bits in even
moduli; the half-buffer relation follows from the power-of-two moduli. Assertions
check the observed relations. A four-virtual-CPU-device sharded replay produces
identical samples to the ordinary production wrapper for the same state/key.

**Causal limit:** the live ring advances, so these are fixed-state support
restrictions, not a permanent exclusion of those physical entries. We have not
established that the half-filled state was used for optimizer updates in this
run, or that this defect explains the cycle-1500 regression. It could also affect
AlphaZero using this dependency and buffer configuration. The full-buffer
collision increase is measurable but far less dramatic than the half-filled
case. Correcting independent index randomness is a justified candidate fix;
an otherwise identical controlled comparison is still needed. Not applied here.

## Low initial loss is largely explained by draw targets

Trackio stores counts from the actual optimizer batches, including masks and
absorbing tails. For targets -1/0/+1, the zero predictor's MSE is the fraction
of supervised targets that are decisive. Values below use the ratio of
cycle-averaged counts; logged loss is an average of per-update masked means,
so their ratio is a cycle-level diagnostic, not exactly an averaged per-batch
normalized loss.

| Cycle | Logged value MSE | Zero-predictor MSE | Loss / zero baseline | Supervised slots |
|---|---:|---:|---:|---:|
| 1 | 0.0768 | 0.0759 | 1.013 | 96.5% |
| 100 | 0.1652 | 0.1708 | 0.967 | 96.9% |
| 400 | 0.1751 | 0.2087 | 0.839 | 94.7% |
| 800 | 0.3243 | 0.8914 | 0.364 | 92.5% |
| 1600 | 0.2444 | 0.8753 | 0.279 | 93.2% |
| 2400 | 0.1855 | 0.9177 | 0.202 | 94.0% |
| 3200 | 0.1690 | 0.9232 | 0.183 | 94.2% |
| 3745 | 0.1541 | 0.9299 | 0.166 | 94.4% |

Initially 92.4% of supervised targets are zero. The initial model is barely
better than guessing zero (slightly worse at cycle 1). Raw loss increases as
decisive outcomes become common, while baseline-relative error improves.
Later, loss continues improving while ACPL worsens. Thus the low initial loss
does not demonstrate good value learning, and the rise alone is not evidence
of learning failure. These are MuZero run data, not a matched AlphaZero comparison.

## Masks narrow opening supervision, not progressively the whole batch

The production mask excludes real positions before the per-game exploration
switch S (uniform 0..29), while retaining their policy/transition supervision.
For ply t below 30, direct value-label probability is `(t+1)/30`, before
survival/replay weighting. At plies 0, 10, 20, 29 this is 3.3%, 36.7%, 70%, 100%.
For a six-position live window starting at ply zero, 80% of switch choices give
no value label anywhere in that window. Recurrent labels still backpropagate
through representation when present; masking the root does not detach it.

All four existing value-target/gradient audit tests passed again, including
enumerating the 30 switches through the production sparse-chess gather and
independent chess outcome tests across circular-buffer wrap. See
[the earlier audit](muzero-value-target-audit.md) for exact tests and semantics.

There is no observed collapse in the overall supervised fraction: it increases
from 93.2% at cycle 1600 to 94.4% at 3745. Invalid training-window counts are zero
throughout the logged run. Terminal/absorbing tails are supervised and included
in this fraction; the historical counts cannot separate them from real positions.

Actual sampled-root histograms show a shift toward later game positions:
approximate mean ply 104.6 at cycle 1200, 115.0 at 1600, and 142.6 at 3200.
The fraction below ply 30 is bounded by 15.0–19.3% at 1200 and 10.7–14.6% at
3200 (bounds account for histogram bins crossing ply 30). Longer games can
produce this shift under uniform position sampling; these histograms alone
cannot attribute it to the sampler defect. Combined with the masks, early
position value supervision is scarce even while overall supervision is high.

Logged sample age averages approximately 614 self-play steps at cycle 1200 and
664 at 3200, with broad ranges, rather than concentrating entirely on recent
entries. Row-ID histograms cover all 64 bins. These are bounded samples of
4096 optimizer roots, not exhaustive replay dumps, and marginal histograms
cannot detect the joint index defect above or count duplicate board positions.
Game IDs wrap, so occupied game-ID bins must not be interpreted as unique-game
counts. At cycle 3200 the run logged 154,698 completed-start evictions, zero
pending-start evictions, zero missing/truncated errors, over 65,520 self-play
steps × 8192 lanes. This merits awareness but does not establish dominant loss
of experience or identify which positions were evicted.

## Root versus recurrent value errors

Reanalyzed the **initial frozen** cycle-3200 predictions from the preceding
fixed-data experiment; these numbers are before its diagnostic fitting updates.
The held-out half contains 128 windows from distinct recorded games, starts at
ply >=30, and contains six live positions with all value masks enabled. Labels
are recorded game outcomes, not optimal-play truth. The zero baseline is 0.9766
at every depth. The root and recurrent columns use identical windows/targets;
the fresh column evaluates each corresponding actual successor board.

| Depth | Unrolled value MSE | Fresh-board value MSE |
|---|---:|---:|
| 0 (root) | 0.4873 | 0.4873 |
| 1 | 0.5564 | 0.4311 |
| 2 | 0.5544 | 0.4666 |
| 3 | 0.5669 | 0.3966 |
| 4 | 0.5615 | 0.4199 |
| 5 | 0.5869 | 0.3743 |

Both beat zero, but recurrent predictions are worse than fresh-board predictions
at every successor depth. This survives removing exploratory and terminal-tail
positions from this diagnostic. It is not a decomposition of the live run's
0.169 training loss: the position distribution differs. The earlier held-out
sample was not newly generated, and no new model inference was necessary here.

## Reproduction and limits

```bash
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \
  taskset -c 0-15 uv run --no-sync python tests/helpers/audit_replay_coverage.py
JAX_PLATFORMS=cpu taskset -c 16-23 uv run --no-sync pytest -q \
  tests/test_muzero_value_target_audit.py tests/test_muzero_value_gradient_audit.py
git diff --check
```

Sampler and sharding assertions passed; four pytest tests passed in 23.34s.
Raw results: `artifacts/replay-coverage-audit-20261017/results.json`.
The helper reads the Trackio database in read-only mode and reuses
`artifacts/value-fit-tpu-20261017/{dataset.npz,fit-results.json}`.
Repeated execution overwrites only its diagnostic JSON.

The partial checkpoint lacks live replay, so actual board duplication,
root/recurrent training losses by terminal status, and exact population-versus-
sample coverage cannot be recovered retrospectively. Testing those requires a
bounded instrumented replay run or replay snapshot. No additional long run was
started, and no settings or sampler fix were applied. The strongest new result
is a reproducible sampling defect, not proof of the entire learning failure.
