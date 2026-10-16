# Frozen search targets, depths, candidate errors, and value persistence

2026-10-17 Asia/Tokyo. Follow-up to
[Claude's consultation](claude-recurrent-value-debug-review.md). No training
settings, weights, dependencies, or production code changed. No commit.
All computation in this experiment was CPU-only; TPUs remain free.

## Conclusions

1. Targets are often extremely sharp, and transformed Q-values dominate the
   prior-logit spread. This establishes the proposed amplification mechanism's
   prerequisites, not that sharper targets caused training to deteriorate.
2. Search selects actions with optimistic recurrent/fresh-board discrepancies.
   Independently, Stockfish ranks fresh-board values above recurrent values,
   and recurrent values above backed-up search Q. This supports noisy value
   changes being exploited by search.
3. Error is already substantial after one learned transition. Repeated dynamics
   adds further error; refreshing the representation helps at depths 2, 3, and 5
   with positive paired confidence intervals, but leaves sizeable one-step error.
4. A literal root-value-copy explanation is **not established**. Recurrent
   predictions depend on actions and change more than fresh-board predictions.
   Their changes are only weakly/moderately aligned with fresh-board changes.
   Fresh values actually correlate with the signed root estimate more strongly
   than recurrent values do. High root/recurrent correlation alone is therefore
   not evidence of a shortcut.
5. Absolute recurrent/fresh error does not significantly increase with candidate
   prior rank in this sample. This does not support the simple "lower-prior
   alternatives are where all errors occur" prediction. Prior rank is not actual
   historical action coverage, so broader coverage hypotheses remain open.

The strongest current description is **inaccurate action-dependent value changes,
some accumulated dynamics error, and search selecting optimistic errors**. The
training origin (coverage, representation, head calibration, target feedback)
and the onset near cycle 1500 remain unresolved.

## Protocol and checks

- Frozen cycle-3200 generation `gen-16c64bcee1414d8ea7c19c5ab8e120e9` under
  `artifacts/dynamics-comparison-20261016/checkpoint`.
- Exactly the prior 192 sample roots, one per selected cycle-2400 PGN game.
  Historical targeted mates are excluded. Full move histories reconstructed;
  PGX root legal masks independently match python-chess.
- Exact Gumbel m=16/n=32, saved root temperature 1.5, no depth truncation.
  Separate noise scales 0 and the saved training value 1.0; one fixed seed.
  `return_tree=True` only for this diagnostic. No production setting changed.
- Both noise conditions matched the production wrapper's action and dense
  target on the first batch. All 192 zero-noise actions match the earlier
  controlled search comparison. Parent ordering, 32 expansions, and 32 total
  root-child visits checked at every root.
- Every visited root child has a one-step recurrent value from the search tree
  and a fresh value from its actual successor board. Values are negated into
  the root player's perspective. Terminal candidates are excluded from value
  discrepancy/optimism calculations; exact terminal rewards replace fresh Q
  for reference-ranking/move-choice calculations. There are 10 and 8 terminal
  candidates in the respective noise conditions.
- Stockfish scores the union of both conditions' candidates once per root,
  preserving history, MultiPV restricted to that union. Threads=1, Hash=64 MiB,
  depth 18 with 20-second cap; 16 worker processes. **All 3,204 distinct candidate
  moves reached depth 18.** Binary SHA256 is saved. Scores use root perspective.
  CP losses are clipped to +/-1000 and measured against the best tested candidate,
  not the unrestricted best move, full-game ACPL, or a rerun of the old scoring.
- Persistence uses the existing 256 distinct-game, six-live-position windows,
  all starting at ply >=30. First 128 fit descriptive regressions; last 128 are
  held out for all reported persistence errors. No model fitting occurs.
- Confidence intervals use 5,000 bootstrap resamples of roots/windows, not
  individual candidates. Correlations omit roots with fewer than three candidates
  or constant rank inputs. Root-level statistics do not treat all moves as
  independent observations.

## Target sharpness

Actual saved Q transform: `value_scale=1`, `maxvisit_init=50`,
`rescale_values=False`. Thus Claude's suggested min-max-rescaling mechanism
does not apply here. The transform still multiplies completed Q by
`50 + maximum child visits`.

| Metric | Noise 0 | Noise 1 (configured) |
|---|---:|---:|
| Mean prior entropy, nats | 2.401 | 2.401 |
| Mean target entropy, nats | 0.813 | 0.792 |
| Mean maximum target probability | 0.732 | 0.740 |
| Roots with max probability >0.95 | 93/192 | 92/192 |
| Roots with max probability >0.99 | 78/192 | 76/192 |
| Mean KL(target || prior) | 1.350 | 1.391 |
| Mean candidate-prior logit span | 2.265 | 3.185 |
| Mean transformed-Q span over legal moves | 33.878 | 34.856 |
| Mean backed-up Q span over candidates | 0.620 | 0.638 |

The priors above include the configured root temperature. These are recomputed
targets at frozen real-board roots, not recovered training minibatch targets or
terminal-uniform targets. They are often nearly one-hot, but not uniformly so.
Top-128 retains effectively all probability mass on this suite (floating-point
roundoff around 1); BNK is not making these particular targets sharp.

## Measured search depth

Depth is root-to-new-node distance for each of 32 expansions, derived from the
returned parent pointers. This measures actual expansion depth, not model unroll
length or a configured maximum.

| Expansion depth | Noise 0 count | Noise 1 count |
|---|---:|---:|
| 1 | 2634 | 2634 |
| 2 | 2361 | 2352 |
| 3 | 818 | 815 |
| 4 | 193 | 204 |
| 5 | 63 | 64 |
| 6 | 28 | 28 |
| 7 | 20 | 20 |
| 8 | 19 | 19 |
| 9 | 7 | 7 |
| 10 | 1 | 1 |

Each column totals 6144 expansions. Mean depth is 1.860/1.864; 5.39%/5.58%
of expansions are deeper than 3. However, 80/82 of 192 roots reach beyond 3.
Search is shallow on average, but it is not just evaluating one-step children.
Interior policies and deeper dynamics cannot be dismissed from simulation count.
Mean root candidate count is 13.719; roots with few legal actions consider fewer
than 16 candidates.

## Candidate errors and selection

"Selected optimism" is `(recurrent Q - fresh Q)` for the chosen candidate minus
the mean of that discrepancy over the same root's nonterminal candidates. It is
relative optimism against another learned estimate, not an unbiased error against
true game value. Stockfish comparisons below provide a separate reference.

| Root-averaged metric | Noise 0 | Noise 1 |
|---|---:|---:|
| Mean absolute recurrent/fresh discrepancy | 0.253 | 0.252 |
| Selected optimism, one-step recurrent | 0.147 [0.110, 0.186] | 0.123 [0.088, 0.157] |
| Selected optimism, backed-up Q | 0.204 [0.163, 0.249] | 0.181 [0.141, 0.221] |
| Mean within-root Spearman(rec, fresh) | 0.394 | 0.449 |
| Median within-root std(rec)/std(fresh) | 1.335 | 1.273 |
| Spearman(abs discrepancy, prior rank) | -0.035 [-0.094, 0.022] | -0.047 [-0.111, 0.016] |

No positive prior-rank trend is demonstrated. Standard-deviation ratios show that
candidate values are not action-invariant: recurrent variation is typically
larger, not smaller, than fresh-board variation. That variation need not be useful.

Stockfish rank correlations (166 usable roots per condition, omitting constant
or insufficient rank data):

| Mean per-root Spearman versus clipped Stockfish CP | Noise 0 | Noise 1 |
|---|---:|---:|
| One-step recurrent value | 0.390 | 0.442 |
| Fresh-board successor value | 0.494 | 0.551 |
| Backed-up search Q | 0.307 | 0.385 |
| Paired fresh-minus-recurrent improvement | 0.104 [0.053, 0.151] | 0.109 [0.060, 0.155] |

Mean candidate-relative CP loss, all 192 roots:

| Decision rule within each candidate set | Noise 0 | Noise 1 |
|---|---:|---:|
| Actual Gumbel search choice | 64.17 | 68.42 |
| Highest one-step recurrent Q | 76.40 | 73.39 |
| Highest fresh-board Q | 41.10 | 44.65 |
| Highest prior among candidates | 46.48 | 43.62 |

Fresh-Q argmax improves over the actual search choice by 23.07 CP
[8.21, 38.43] / 23.77 CP [6.48, 40.61]. This is a post-hoc one-step candidate
selection comparison, not the earlier full real-board search ablation. It is
not a deployment recommendation. Noise conditions can have different candidate
sets; do not read their CP differences as a clean noise ablation.

## Does dynamics merely carry the root value forward?

For the same 128 held-out legal played windows, compare:

- **Carry:** `(-1)^depth * initial_root_value`, ignoring all actions.
- **Recurrent:** repeated learned transitions from the root representation.
- **Refreshed one-step:** encode the actual predecessor board, then apply only
  the final learned transition. No trained weights are changed.
- **Fresh:** evaluate the actual successor board directly.

MSE against recorded game outcomes (not optimal-play labels):

| Depth | Carry | Recurrent | Refreshed one-step | Fresh |
|---|---:|---:|---:|---:|
| 1 | 0.4870 | 0.5578 | 0.5578 | 0.4314 |
| 2 | 0.4870 | 0.5556 | 0.4879 | 0.4666 |
| 3 | 0.4870 | 0.5678 | 0.5340 | 0.3969 |
| 4 | 0.4870 | 0.5613 | 0.4986 | 0.4197 |
| 5 | 0.4870 | 0.5862 | 0.5036 | 0.3742 |

MSE against the corresponding fresh-board prediction:

| Depth | Carry | Recurrent | Refreshed one-step |
|---|---:|---:|---:|
| 1 | 0.0468 | 0.1105 | 0.1105 |
| 2 | 0.0284 | 0.1364 | 0.0987 |
| 3 | 0.0556 | 0.1673 | 0.1049 |
| 4 | 0.0581 | 0.1362 | 0.1108 |
| 5 | 0.0773 | 0.1673 | 0.0984 |

Paired improvement in fresh-reference MSE from refreshing is 0.0377
[0.0064, 0.0692] at depth 2; 0.0624 [0.0219, 0.1062] at depth 3;
0.0254 [-0.0187, 0.0701] at depth 4; and 0.0688 [0.0297, 0.1121] at depth 5.
Medians are near zero: the mean benefit is concentrated in a subset of windows,
not a uniform improvement for every position.

Root-persistence correlations require caution:

| Depth | Corr(recurrent, carry) | Corr(fresh, carry) | Corr(recurrent-change, fresh-change) |
|---|---:|---:|---:|
| 1 | 0.904 | 0.956 | 0.256 |
| 2 | 0.871 | 0.969 | 0.271 |
| 3 | 0.856 | 0.946 | 0.271 |
| 4 | 0.848 | 0.940 | 0.427 |
| 5 | 0.819 | 0.926 | 0.477 |

Here "change" subtracts the signed root estimate. At depth 1 the RMS recurrent
change is 0.307 versus fresh change 0.216; at depth 5 it is 0.441 versus 0.278.
The model adds substantial changes, but their alignment with fresh changes is
limited. Carry-forward alone is a better baseline than the actual recurrent
predictions in this suite; that does not mean the network literally copies root
values. Real successor values themselves strongly persist from the root.

Descriptive regressions fit on the first 128 windows and tested on the other
128 give held-out R2 of 0.815 at depth 1 and 0.661 at depth 5 when predicting
recurrent value from carry alone. Adding fresh value raises these to 0.827 and
0.737. Predictors are highly correlated, so individual coefficients cannot
identify the internal mechanism. Full coefficients and intermediate values are
saved rather than interpreted as causal proof.

## Limitations and next test

- One late checkpoint; no measurement of when these effects first appear.
- Evaluated roots come from earlier self-play evaluation games, not a saved
  dump of the training replay. Historical action coverage is not known.
- Fresh-board values and Stockfish are imperfect references. Neither provides
  the exact expectation of the training policy's eventual outcome.
- Sharp targets plus optimistic search demonstrate a plausible feedback path,
  not that policy gradients caused the value errors. No optimizer ablation yet.
- This experiment does not separate representation information loss from value
  head calibration on learned states.
- CPU held-out MSE differs from the prior TPU diagnostic by at most 0.00147
  across depths. An initial 0.0002 cross-backend assertion was too strict; it was
  replaced by explicit reporting. All new paired comparisons use CPU outputs
  from the same execution, and exact same-backend control checks pass.

The next discriminating test is a held-out calibration/readout probe on the
frozen learned successor states, compared with real successor representations,
plus small isolated policy-only/value-only update experiments if necessary.
Successful readout recovery would support a head/calibration problem. Failed
linear recovery alone would not prove missing information. Target-softening or
consistency training should follow a controlled hypothesis test, not these
correlations alone. No new training settings have been proposed as established
fixes or applied.

## Reproduction and artifacts

From the multihost worktree root:

```bash
JAX_PLATFORMS=cpu taskset -c 0-15 uv run --no-sync python -u \
  tests/helpers/probe_frozen_dynamics.py \
  --output artifacts/frozen-dynamics-probes-20261017
taskset -c 16-63 uv run --no-sync python -u \
  tests/helpers/score_frozen_candidates.py \
  artifacts/frozen-dynamics-probes-20261017 --workers 16
uv run --no-sync python tests/helpers/summarize_frozen_dynamics.py \
  artifacts/frozen-dynamics-probes-20261017
git diff --check
```

All completed successfully. Use a new output directory for independent repeats;
these helpers overwrite their own diagnostic files. Under that output directory:
`manifest.json`, `search.json`, `persistence.npz`, `stockfish.json`, `summary.json`,
and `probe.log`, `scoring.log`, `summary.log` preserve the exact evidence.
