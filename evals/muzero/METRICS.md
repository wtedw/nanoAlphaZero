# AlphaZero-compatible monitoring

The naming reference is the user's sibling `az` repository:
`src/alphazero/core.py`, `train.py`, `diagnostics/hex_diagnostics.py`, and
`diagnostics/opening_value_charts.py`. The current nanoAlphaZero training loop
is also consulted for its MoHex evaluator and timing conventions.
Chart source: AZ commit `5870684345d2f12ebb886033a6f2772663061129`, source-file
SHA-256 `9f3e6639d2c5eb76a96a0ba4b2e485b5d796c74a7124abd563f4f9a0606e2197`.

New runs declare `metric_schema = "alphazero-v1"` in resolved metadata.
`metrics.standardize` applies the same mapping to local JSON and W&B. Historical
run files and W&B histories are preserved; their old schema is not silently
rewritten. The richer replay provenance is included in new checkpoints. Older
checkpoints remain inference-loadable; resuming their old training state requires
their preserved source snapshot because the new replay payload has extra fields.

## Naming and measurement

| Family | Names and behavior |
|---|---|
| Loss | `total_loss`, `loss_pi`, `loss_v`; reward component is `muzero/loss_r` |
| Optimizer | `norms/grad_norm`, `param_norm`, `update_norm`, `update_ratio`, `update_grad_cosine`, `current_lr`, per-head update ratios, `n_updates` |
| Entropy | `entropy/target_policy`, `network_policy`, `diff_network_minus_search`; measured at sampled roots |
| WDL | Actual spatial head probabilities: `wdl/p_win_mean`, `p_loss_mean`, `p_draw_mean`; omitted for vector scalar heads |
| Loss by distance | `loss_per_step_PI/pi_term_step_00` through `09`, corresponding V and count groups; real terminal games only |
| Progress | `runner_state/n_updates`, `runner_state/train_step`, `runner_state/n_selfplay_steps`, `cycle`, `loop/loop_n` |
| Timing | `timing/cycle_seconds`, `timing/training_seconds`, `loop/loop_duration`, `loop/loop_total_duration`, `warmup/loop_duration` |
| Run summary | `stats/num_params`, `stats/warmup_duration`, `stats/total_learn_duration` |
| Self-play | Original `selfplay/ep_term_step_*`, `ep_step_*`, P1/P2 wins/ties, legal-count min/max/early/mid/late means, global step |
| Reward checks | `selfplay-reward/valid_*`, `errors/selfplay-reward/invalid_*`; only actual transitions, never padded computation |
| Search/policy disagreement | Original `selfplay-intr/kl_*`, threshold counts, and histogram; derived from the same root encoding used by search |
| Openings | `openings_explore/<action>`, `openings_exploit/<action>`, and opening histograms |
| Staging/backfill | `selfplay_buffer/n_*`, `value_diff_*`, `kl_divergence_*`, `num_peak_samples`, eviction diagnostics |
| Consume | Original `selfplay_buffer-consume/*`, `selfplay_buffer-comp/*`, duplicate/stale/not-real/pending/exploration/game-ID anomalies, last-drain histograms |
| Drain/replay | `drain/num_valid_consumable`, `drain/n_slices`, `training/n_slices_drained`, `training/rbuf-n_is_valid`, `rbuf-n_is_fresh`, `spbuf_num_consumables` |
| Batch | AZ names `train_batch/n_reward1`, `n_rewardneg1`, `n_reward0`, `n_valid`, `n_invalid`, provenance/freshness/double-check counts, unique game IDs and histograms |
| Debug | `debug/batch_checksum`, `debug/sample_rng_hash`, `debug/rbuf_current_index` |
| Hex diagnostics | `hex_perf/mse_vs_perfect`, `hex_perf/sign_accuracy`, original `model/*` and `params-logits/*` opening statistics |
| MoHex | Unchanged production `hex_eval/*`; the extra policy-only comparison is `muzero/hex_eval_policy/*` |
| MuZero extras | All held-out/unroll metrics, learned rewards, boundary bootstraps, latent charts, extra matchups, episode statistics, and additional staging counters live under `muzero/` |

The basic batch counters now count **B sampled root positions**, not the
B×(K+1) collection of unroll targets. Exact ±1/0 counts match AZ; non-outcome
bootstrap values are counted separately under `muzero/train_batch/`.
Terminal zero rewards in a no-draw game are flagged; a zero cutoff bootstrap is
legitimate. Absorbing targets and cutoff boundary targets are reported separately.
Consumed replay records retain their original fresh/provenance flags, just as
AZ's consumed copies do, while the staging flags clear immediately.

## Important semantic differences

- New runs use masked target-to-prediction **KL** for `loss_pi` across real
  unroll states, matching AZ's KL direction. Historical runs used cross-entropy.
  KL subtracts the fixed target entropy and has the same parameter gradient.
  `loss_v` remains masked scalar MSE and the total includes reward MSE. The
  root-only KL is `muzero/root_policy_kl`; per-terminal-distance policy losses
  now also use KL. AZ averages over its fixed root batch; MuZero averages over
  valid unroll states. Neither MuZero's masking nor its value objective changed.
- Spatial WDL probabilities describe the actual head parameterization. MuZero
  currently trains the scalar win-minus-loss expectation, not a WDL loss;
  the draw probability is not claimed to be calibrated.
- Collection remains synchronous. Self-play counts aggregate actual transitions
  in a whole collected episode batch; AZ ordinarily logs its last asynchronous
  step. `ep_step_*` reports each lane's last real decision index, and
  `ep_term_step_*` restricts that to terminal games. These are explicit
  collection-window differences, not identical asynchronous occupancy statistics.
- `runner_state/n_selfplay_steps` counts training collection scan steps, excluding
  warmup; `selfplay/global_step` includes warmup. Actual transitions, excluding
  padded compute, are `muzero/real_transitions`.
- Consume statistics/histograms describe the last complete consume slice in a
  drain, as in AZ. Total drain counts remain separate. A drain with no slices
  reports zero consume scalars and emits no consume histogram.
- `norms/update_norm` is now measured from the optimizer's actual update tensor,
  before adding it to parameters. LR is the schedule value used for that update.
  `update_grad_cosine` follows AZ's signed dot-product implementation; a descent
  update can have negative cosine with the gradient.
- `hex_eval/perfect_play_achieved` retains the production name, but denotes
  conversion of the small known-winning-opening suite, not broad optimality.

## Conditional and inapplicable AZ metrics

The implementation includes the applicable collection, replay, optimizer,
entropy, WDL, opening, MoHex, and distribution families above. It does not
invent numbers for features absent from this baseline:

- AZ's incoming **pending placeholder** errors (`n_rewards_not_zero`,
  `n_is_pending_reward_i8_not_set`, `n_is_fresh_i8_not_zero`) check records before
  backfill. MuZero stages completed, signed-backfilled episodes; those inputs do
  not exist. It checks malformed prefixes and fresh-slot overwrites instead.
- Muon-specific norms require a Muon optimizer; this baseline uses AdamW.
- Per-parameter gradient histograms are an optional AZ debug mode, not enabled
  in this baseline. Global and policy/value-head update metrics are present.
- Stockfish/chess-specific legality cross-checks, reference-engine metrics, and
  special chess opening categories require that evaluator. They are not emitted
  for Hex. Generic replay validity mismatch checks retain AZ's names.
- An Elo ladder requires persistent anchor matchups. Existing policy/search/random
  matches are retained under `muzero/eval/`; no anchor Elo is fabricated.

## Original table and interactive chart reuse

The board helpers from AZ are adapted into the isolated `charts.py` module,
without an external checkout dependency. A read-only test compares against the
actual sibling AZ helper when present. Columns are exactly:

`game, boardsize, action, row, col, value, ground_truth, highlight, label`.

The standard outputs are:

- `model/values_after_black_moves_table` and `model/values_after_black_moves_heatmap`;
- `model/logits_heatmap_top_k_table` and `model/logits_heatmap_top_k`.

Values are White-to-move after each Black opening; the graph preserves square
cells, reversed row axis, numeric labels, and bottom-k blue highlights. Policy
logits use the original Viridis palette and top-k red highlights. W&B receives
Tables and Plotly numeric JSON only—no Matplotlib, raster image or screenshot.
Latent one-move values and per-opening reply logits are under `muzero/model/`.

## Validation

Held-out unroll metrics retain the original mixed-distribution MSE and now
also report `exploration_root_*` and `nonexploration_root_*` under
`muzero/heldout*/unrollN/`. Each group has its sampled-root count, real-state
unroll value MSE (absorbing/padded states excluded), and root-only value MSE.
Groups are defined by whether the sampled root was exploratory. This matters
because the AZ-style replay consumer excludes exploratory starts. These
additional metrics do not alter collection RNGs, replay, or optimization.
`heldout_batch_size` controls only independent diagnostic collection; zero
inherits the full self-play batch. Hex7–9 use 512 to limit saved data.

- Original chart table/layout/palette/highlights parity test passed against AZ.
- Root-count tests demonstrate that a two-position, six-head unroll reports two
  valid roots, not twelve; provenance corruption is detected.
- First four-TPU Hex4 smoke emitted 239 metrics, saved a checkpoint, and completed
  MoHex evaluation. Its **177 training/optimizer tensors are bitwise identical**
  to the corresponding pre-standardization smoke: logging did not change learning.
- A four-TPU full-batch Hex5 W&B smoke checks the complete table/Plotly/histogram
  serialization path: **passed**, with 258 scalar metrics, 4096 valid roots,
  14 JSON media files (8 tables and 6 Plotly charts), zero raster images,
  zero malformed episodes and zero overwritten fresh slots.
  W&B: https://wandb.ai/tdoubleu/nanoAlphaZero-muzero/runs/iwxbzslo .
