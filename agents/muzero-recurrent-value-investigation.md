# MuZero chess: recurrent value investigation

Recorded 2026-10-17 (Asia/Tokyo). Experiment artifacts use the host UTC date,
20261016. This records completed measurements and a proposed debugging plan;
the proposed training interventions have not been run.

Follow-up: [value-target, mask, and gradient audit](muzero-value-target-audit.md)
records 335 passing tests, an opening supervision coverage gap, and frozen-model
per-depth gradients. It found no demonstrated target/indexing/disconnection bug;
the consistency-loss proposal remains untested and has not been added.

Further [TPU fitting tests and the dainty-sunset-0 Hex cross-check](muzero-value-fit-and-hex-control.md)
show that the recurrent path can fit a small fixed chess dataset, but held-out
error worsens. The strong Hex run also has imperfect recurrent opening values,
while its highest-ranked openings remain winning.

The subsequent [replay-coverage audit](muzero-replay-coverage-audit.md) finds a
reproducible row/time PRNG correlation in the pinned Flashbax sampler, explains
the initially low value loss with draw-heavy targets, and measures the later
shift toward longer-game positions. No sampler fix has been applied yet.

The [frozen search/value probes](muzero-frozen-search-value-probes.md) measure
sharp targets, actual search depths, candidate-level Stockfish rankings and
selected optimism, and one-step versus accumulated dynamics error. They do not
establish a literal root-copy shortcut or the cause of the training onset.

The [latest cycle-3600 scale/readout tests](muzero-latest-scale-and-readout-probes.md)
find no validated positive search scale on new held-out games. Frozen learned
states do contain recoverable ranking information: diagnostic readouts improve
rank correlation, while decision-loss improvements remain uncertain.

## Main finding

On the frozen cycle-3200 checkpoint, replacing learned search transitions with
real chess transitions substantially improves move selection. Most of the
improvement is recovered by replacing only the values predicted at imagined
successors with values evaluated on actual successor boards. Replacing only
the policy predictions, or supplying exact legality and endings, does little.

This implicates the recurrent value path: prediction on `g(...g(h(s), a), ...)`.
It does not yet distinguish errors in learned states from value-head calibration
on those states, nor establish why training performance turns around near cycle
1500. A fresh-board value is an alternative estimate, not ground truth.

## Controlled experiment

- Worktree: `/nfs_share/nz/.worktrees/muzero-multihost`.
- Branch: `research/muzero-multihost`; existing unrelated edits preserved.
- Run: `chess-pod-exact-m16-n32-local-gather-20261015`, Trackio `fancy-breeze-4`.
- Frozen checkpoint: cycle **3200**, **64,000 optimizer updates**, model 384x10.
- Generation: `gen-16c64bcee1414d8ea7c19c5ab8e120e9`.
- All variants use the same checkpoint parameters and root observations.
- Exact Gumbel search: **16 candidates, 32 simulations**, root temperature 1.5,
  `max_depth=None`, `return_tree=False`, Gumbel scale **0**, matching ACPL rollout.
- Learned baseline uses reward 0 and discount -1, matching production value-only
  search. Oracle variants use mover-relative terminal reward, discount 0 at
  endings, terminal value 0, and exact interior legality. Live discount is -1.
- Real-board search re-encodes each actual successor using `model.initial`.
  It uses the same MuZero network, not a separately trained AlphaZero network.
- Hybrid variants carry both the actual board and the learned latent. They use
  exact rules and replace just one recurrent output with the fresh-board output.
- Main sample: **192 games sampled without replacement**, one position per game,
  seed 20261017. Alternate zero-based ply ranges 24:40, 40:80, 80:160, with a
  shorter-game fallback. Replay retains PGN history, rather than loading FENs.
- Source games: cycle-2400 ACPL directory
  `cycle-002400-20261016T073159-4919f98182c24a94b7d8698006f35a78`.
- Three historical missed-mate roots are separate from the sampled statistics.
- An initial 48-position pilot, seed 20261016, is not pooled into the main sample.
- Computation ran on CPU while all four TPU hosts remained occupied by training.
  These measurements are not TPU performance tests or a CPU/TPU numerical parity
  claim. No training code, settings, or running processes were changed.

### Independent scoring

Stockfish evaluates all chosen moves for a position together using MultiPV,
also including the move from an unrestricted reference search. Target depth 18,
10-second cap per analysis, one thread, Hash 64. **Every candidate reached depth
18.** Scores are mover-relative, mate scores map to 10,000 cp, and scores are
clipped to +/-1000 cp before computing loss relative to the best evaluated
candidate. These fixed-position losses are **not full-game ACPL** and should not
be equated to the training dashboard's ACPL.

| Variant | Mean loss (cp) | Median loss (cp) | Positions losing >100 cp |
| --- | ---: | ---: | ---: |
| Root policy only, no search | 46.729 | 0 | 30 |
| Production learned-dynamics search | 62.234 | 6.5 | 41 |
| Learned dynamics + exact legality/endings | 62.234 | 6.5 | 41 |
| Replace recurrent policy only + exact rules | 61.000 | 8.5 | 39 |
| Replace recurrent value only + exact rules | 30.667 | 0 | 19 |
| Real-board search | 28.953 | 0 | 20 |

Paired bootstrap intervals, 10,000 resamples of positions:

- Real board vs learned: **33.281 cp improvement**, 95% interval **19.651–47.604**.
  Better on 72 positions, worse on 23, equal score on 97; 105 changed moves.
- Value replacement vs learned + rules: **31.568 cp improvement**, interval
  **17.114–46.855**. Recovers approximately 95% of the full real-board benefit.
- Policy replacement vs learned + rules: **1.234 cp improvement**, interval
  **-4.495–8.354**. No clear benefit in this sample.
- Exact rules alone change one move out of 192, with no score change.
- All search variants solve the three historical missed mates at this newer
  checkpoint. Those old failures alone do not diagnose current performance.

Selected illustrative error (not an unbiased aggregate): game 238, root ply 29,
`r2q1r1k/ppp3bp/n2p4/3Pp2n/2P2p2/2N2P2/PP3BPP/1QKR2NR b - - 2 15`.
Learned search chooses `a6b8`: learned one-step Q is **+0.6503**, fresh-board
one-step Q **-0.0940**. Real-board search chooses `d8g5`. Stockfish's pilot
comparison assigns the knight retreat 114 cp loss. Q values here are neural
outcome estimates, not centipawns or final backed-up search Q values.

## What to debug next, in order

### 1. Establish whether recurrent supervision is correct

Audit actual sampled training batches, not just the loss formula. Current
`muzero.py` computes scalar value MSE against `batch.reward`, using
`batch.value_mask`; `buffers.py` constructs the sequences, outcome targets,
exploration eligibility, and alternating terminal tails. The field name
`reward` alone does not establish that the wrong quantity is being trained.

For complete recorded games, independently reconstruct targets and compare at
every unroll offset 0..5:

- Observation/action alignment: action at offset k must produce state k+1.
- Player perspective and outcome sign for both colors, wins and draws.
- Game boundaries, terminal roots, absorbing tails, and the 512-ply cap.
- Value masks and actual coverage under `exploration_losses=policy-transition`.
- Replay gather offsets, lane identities, and behavior at ring-buffer wraparound.

Use tiny deterministic fixtures and captured real samples. If a discrepancy is
found, fix it with a regression test before evaluating learning interventions.

### 2. Measure where prediction quality deteriorates

On held-out games, compare `V(g^k(h(s), actions))` with `V(h(s_k))` and known
game outcomes for k=0..5; probe deeper than 5 separately because that is beyond
the current training unroll. Record per-depth MSE, bias, sign mistakes, saturation,
and error conditioned on checks, captures, terminal transitions, and game phase.
Outcome labels are noisy observations of the played continuation, not exact
optimal-play values; report both references rather than treating either as truth.

Distinguish played actions from alternative legal actions explored by search.
For counterfactual branches, there is no known outcome label from the original
game. Use fresh-board estimates and independent engine ranking as diagnostics.
Measure whether search preferentially selects branches with optimistic recurrent
errors. Compare mean error with the error of the branch search actually selects.

This separates a general value-fitting problem, accumulated unroll error, and a
coverage problem on actions that are rare in the training trajectories.

### 3. Inspect optimization before changing it

Split value loss and effective sample counts by unroll depth. Measure gradients
into representation, dynamics, and the shared prediction head, including their
relative contributions from root vs recurrent losses. Audit stop-gradient,
normalization, dtype, and rematerialization paths. The current recurrent input
uses `scale_gradient(latent, 0.5)`; measure its effect before removing it.

Check whether lower aggregate value loss hides worse recurrent calibration or
improvement dominated by common/easy/terminal states. Compare the scalar-MSE
objective with AlphaZero's value objective only after confirming targets and
masks; a different objective is a hypothesis, not an established bug.

### 4. Test one targeted repair at a time

Choose the intervention based on the measurements above:

| Evidence | Candidate repair to test |
| --- | --- |
| Incorrect signs, indexing, masks, or episode boundaries | Correct the data/target implementation and add exact regression fixtures. |
| Good root values but inconsistent recurrent values on legal trajectories | Add a small stop-gradient fresh-board value consistency term, retaining outcome supervision. Test teacher quality and terminal treatment explicitly. |
| Recurrent supervision or gradients are disproportionately weak | Adjust the identified loss weighting, mask, or gradient path in isolation. |
| Errors concentrate on unplayed actions selected by search | Train on a bounded set of legal counterfactual successors with appropriate bootstrap/consistency targets; do not reuse the played game's outcome as a counterfactual label. |
| Errors appear mainly beyond the training unroll | Test longer supervised unrolls or a controlled search-depth restriction, measuring quality and cost. |

Value consistency would encourage
`V(g(z_k, a_k))` toward `stop_gradient(V(h(s_{k+1})))`. It is a proposed experiment,
not a proven fix: the teacher can be wrong and must not replace outcome grounding.
Latent consistency is a separate possible intervention if representation drift
is demonstrated; do not combine both changes in the first trial.

Use identical starting parameters, data, optimizer budget, and seeds for a
baseline and each treatment. First test on a bounded fixed-data experiment,
then use fresh held-out games to avoid tuning to these 192 diagnostic roots.
Retain this paired search test as a regression check. Require improvement in
recurrent calibration and paired move quality, followed by a bounded full-game
ACPL evaluation. A lower training loss alone is insufficient.

### 5. Establish the timing of the regression

Preserve spaced model snapshots around the relevant training window in future
comparisons, subject to available storage, and repeat the same fixed-position
ablation at each checkpoint. The current experiment has only a frozen
cycle-3200 model, so it cannot show that recurrent-value error first emerges at
cycle 1500. Track this gap alongside ACPL and early/midgame legal-move counts.
Real-board search is a diagnostic reference and potential mitigation, not itself
a repair to the learned dynamics.

## Validation and artifacts

- All 195 replayed roots' legal masks match python-chess.
- First-batch learned actions and dense search targets match production
  `make_muzero_mcts` on CPU (targets checked at tolerance 2e-6).
- `tests/test_chess_dynamics_comparison.py`: **2 passed**, covering mover reward,
  terminal draws/wins, absorbing states, live discount, and legality masking.
- `git diff --check` passed. No commits or pushes were made for this experiment.
- Training was left running; repeat CPU diagnostics should avoid excessive host
  contention. TPU tests require checking ownership on every participating host.

Repository additions:

- `tests/helpers/compare_chess_dynamics.py`: frozen-checkpoint search ablations.
- `tests/helpers/score_dynamics_comparison.py`: paired Stockfish scoring/bootstrap.
- `tests/test_chess_dynamics_comparison.py`: terminal/perspective checks.

All experiment artifacts are under `artifacts/dynamics-comparison-20261016/`:

- `checkpoint/`: frozen partial checkpoint (model loading skips optimizer/buffers).
- `expanded/{manifest,positions,results,scored,summary}.json`: main experiment.
- `hybrids/{manifest,positions,results}.json`: value/policy interventions.
- `results/`: separate 48-position pilot and its scoring.
- `one_step_examples.json`: selected illustrative one-step prediction comparisons.
- `provenance.json`: checkpoint cycle, source hashes, validation, exact commands.
- `search.log`, `expanded.log`, `hybrids.log`, `expanded-scoring.log`, `tests.log`.

Artifacts are ignored and machine-local; this note preserves the numerical
findings even if those files are later unavailable. Reproduction commands are
in `provenance.json`; choose a new output directory to preserve existing evidence.
