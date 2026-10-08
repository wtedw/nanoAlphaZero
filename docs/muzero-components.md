# MuZero component assembly

Root `muzero.py` is the readable training entry point. `muzero-singlefile.py` is the
historical standalone research snapshot, renamed without changing its logic.
The two paths have different collection semantics. Do not compare results as
if renaming alone produced the new algorithm.

Root MuZero uses Tyro to parse the typed `MuZeroArgs` dataclass. Omitted options
inherit the selected game's defaults; boolean overrides retain `--flag` and
`--no-flag` syntax. AlphaZero's CLI remains unchanged. Run
`uv run python muzero.py --help` for the generated options and types.

## Components

MuZero definitions are appended to existing package files. AlphaZero's training
math, self-play, and search remain unchanged; `core.py` remains unchanged in its
entirety. Both host loops now use the extracted shared strength ladder.

| Location | MuZero responsibility |
| --- | --- |
| `muzero.py` | Self-play adapter, latent unroll and loss, optimizer updates, compiled cycle, host loop |
| `config.py` | Game defaults and MuZero configuration constraints |
| `model.py` | Representation, dynamics, prediction, optional rematerialization |
| `mcts.py` | Learned recurrent callback to the existing search engine |
| `buffers.py` | Extended records, completed returns, sequence gathering, storage budget |
| `muzero_metrics.py` | Component metric selection, batch counts, norms, and cycle reductions |
| `eval/hex/training.py` | Shared Hex match evaluator and MuZero evaluation adapter |
| `eval/muzero.py` | Fixed-opening head diagnostics and tracker chart adapters |
| `eval/ladder.py` | Shared explicit-state ladder against random play and frozen past selves |
| `eval/matches.py` | Shared forced openings and color-swapped batched matches |
| `checkpoint.py` | Full persistent-state snapshots |
| `cli.py` | Root script arguments; existing entry points unchanged |

`make_muzero()` returns the top-level `MuZeroAlgorithm` dataclass, exposing
`env`, `model`, `run_mcts_fn`, `selfplay_fn`, `selfplay_buffer`, `replay_buffer`,
`train_fn`, initial `runner_state`, and `run_fn`. It describes the assembly;
the host advances the separate `RunnerState` throughout training. Individual
buffer states and model train state live in that runner state.

MuZero self-play returns `(state, output, metrics)`; training returns
`(state, metrics)`, with losses and diagnostics in the metrics dictionary.
Metric helpers have no tracker dependencies, and the
host forwards the combined cycle metrics to Trackio or W&B.

### Training metrics

- Self-play wins/draws sum all events in a cycle. `selfplay/ep_term_step_*`
  describes games completed during that cycle: min/max across finishes and an
  average weighted by finish count, or -1 when none finished. As in AlphaZero,
  these are zero-based final move indices (add 1 for the number of moves).
  `selfplay/ep_step_min`, `ep_step_max`, and `ep_step_std` describe the lanes at
  cycle end. Legal-move averages cover the cycle; the midgame average is weighted
  by the number of lanes with episode steps 11 through 30, and is 0 if none qualify.
- `train_batch/n_is_valid` and `n_is_invalid` count windows with/without any
  policy-supervised position. `n_policy_valid`, `n_value_valid`, and
  `n_reward_valid` count positions supervised by the respective loss masks.
  `n_reward_pos`, `n_reward_neg`, and `n_reward_zero` count supervised completed
  value targets, not immediate rewards. Value/reward counts include absorbing
  padding; policy and consistency counts exclude it. Enabled consistency retains
  its existing `consistency/n_valid` metric.
- `norms/grad_norm` measures gradients before clipping, `norms/param_norm`
  measures parameters before the update, and `norms/update_norm` measures the
  actual optimizer update, including clipping and weight decay. Training metrics
  are averaged across optimizer batches in the cycle. Cycles without updates
  emit zero training metrics and `train/ready=0`.

No `selfplay/global_step` or `norms/current_lr` metric is added.

### Shared strength ladder

Both AlphaZero and MuZero construct the same component once:

```python
ladder_fn, ladder_state = make_ladder(
    config, env=algorithm.env, policy_fn=algorithm.run_mcts_fn,
)
ladder_state, ladder_metrics = ladder_fn(ladder_state, cycle, params)
metrics.update(ladder_metrics)
```

The factory captures openings, search, and configuration. `LadderState` contains
the frozen anchor, rung, qualifying streak, random-opponent high-water score,
and rating history. Calls return a new state; disabled/not-due calls return the
unchanged state and `{}`. Only promotion copies the current model parameters.
No evaluator state is checkpointed; a resumed host starts a new ladder.

Existing AlphaZero settings and metric names remain: `eval_period` (cycles),
`eval_opening_plies`, `eval_promote_score`, `eval_promote_patience`,
`eval_vs_random_max_elo`, and the six `eval/*` score/rating metrics. MuZero inherits
the game defaults and exposes `--ladder-period N`; use 0 to disable the ladder.
These settings are independent of MoHex and head diagnostics. Both Trackio and
W&B receive ladder metrics through the usual MuZero host logger.

Match helpers retain the original AlphaZero behavior, including both colors,
zero Gumbel exploration, game/search budgets, and treating games unfinished at
the ply cap as draws. RNG uses the existing independent `1234 + cycle` seed.
Metrics describe the evaluated rung before any promotion; qualifying deltas are
averaged to set the next rung height. Ratings are relative ladder estimates,
not externally calibrated chess Elo. AlphaZero keeps its ASCII rating chart.
Existing imports of match helpers from `training.py` and `core.py` still work.

## Data lifecycle

1. `make_muzero_selfplay` wraps `core.make_selfplay` without changing its RNG
   splitting, exploration, action selection, automatic reset, or counters.
   As in AlphaZero, initial game/exploration setup uses its existing fixed seed;
   the configured seed controls model initialization and cycle RNGs.
2. Each call appends one transition per lane. Games continue across cycles.
   A finished lane immediately receives a new environment state through the
   existing PGX auto-reset wrapper.
3. `MuZeroSelfplayOutput` extends `SelfplayOutput`. `reward` remains the return
   target; `transition_reward` stores the immediate acting-player reward.
   `discount` is signed to convert the next player's value to the actor's value.
4. At termination, `add_backfill` scans chronological transitions backwards to
   calculate returns. Starts with an exploit position in their K+1-position
   window become fresh, including starts whose root is exploratory. Finished
   game IDs and final step numbers are assigned using existing self-play output.
5. `consume` calls the existing AlphaZero consumer to select fresh starts and
   clear their flags. It then gathers K transitions and K+1 policy/value targets
   from the returned storage. Records remain available for overlapping samples.
6. Replay stores each complete `MuZeroTrainingSample` as one item. Its Flashbax
   `sample_sequence_length` stays 1. Sampling replay does not consume an item.

Chronology is checked with global step, game ID, and episode step. Circular
storage may evict pending positions or unconsumed completed starts; cumulative
counters report those drops without blocking training. Surviving starts retain
their forward transitions because eviction is chronological. Missing transitions
inside a sampled window remain sticky errors: the optimizer is skipped and the
host raises an exception.

Terminal tails train zero rewards and values with random padding actions and
no policy loss. Completed games alone supply training samples. If no replay
batch is ready, a cycle advances games without an optimizer update. Warmup also
advances games and fills buffers without optimizer updates.

`--buffer-layout features` flattens each replay field to `[B, T, F]`, keeping
its dtype and restoring its original shape when sampled. For chess with K=5,
`board_float` changes from `[B, T, 6, 8, 8, 2]` to `[B, T, 768]` in storage.
Logical bytes, sample selection, and training targets are unchanged. This is a
MuZero-only wrapper around the existing replay factory; AlphaZero and self-play
storage are unchanged. The default is `--buffer-layout original`. Matrix layout
remains a benchmark experiment. The flag currently applies only to replay.

PGX's current auto-reset discards the boundary observation on truncation.
The new path reports truncation as an error instead of inventing a bootstrap
target. Built-in environments use termination, including their game limits.
Supporting custom truncated environments requires preserving that observation.

## Learning and compatibility

The model architecture and objective follow the research implementation:
policy KL, scalar value MSE, immediate reward MSE, and recurrent gradient scale
0.5. Representation and dynamics use separate parameters. Policy/value heads
are reused across unroll steps. Search uses root legality and learned internal
transitions, with the research implementation's alternating-player assumption.

### Experimental self-consistency

`--exp-self-consistency` adds an EfficientZero-style temporal consistency loss.
It is off by default. `--self-consistency-weight` sets its coefficient (default
1.0 when enabled). This is only the auxiliary consistency objective, not a full
EfficientZero V2 implementation; search and the policy/value/reward losses stay
the same.

At unroll step k, the model compares a predicted latent with the representation
of the real observation at t+k. Both pass through a shared projection MLP; only
the predicted branch passes through an additional prediction MLP. The target
projection is stop-gradient. The loss is mean negative cosine similarity over
valid future observations, added to the existing objective with the configured
weight. `loss_consistency` approaches -1 as vectors align, so total loss can
become negative. `consistency/n_valid` reports the number of contributing pairs
per training batch, averaged over the cycle's updates.

This follows the [EfficientZero V2 consistency objective](https://github.com/Shengjiewang-Jason/EfficientZeroV2/blob/main/ez/agents/base.py)
with compact, model-width MLPs and LayerNorm instead of the reference's BatchNorm.
It introduces no running normalization state or target-network update. The
coefficient and valid-pair averaging follow this runner's loss scaling rather
than reproducing the reference experiment's hyperparameters.

Replay observation fields (`observation`, or packed `board_bool`/`board_float`)
have one position axis: length 1 without consistency, length K+1 with it.
Training reads position 0 for the root and positions 1 onward for consistency;
there are no separate consistency-observation fields. Chess stays packed until
training; dense observations retain their self-play storage dtype, including
the root, and are converted to float32 only when used by the model.
The existing chronology checks and policy masks exclude reset/absorbing tails
and cross-game targets. Terminal next observations are not stored by this runner,
so consistency is not applied to the transition ending a game. Storage estimates
include the added observations. Disabling consistency leaves model parameters
unchanged and stores only the root observation. Enabled checkpoints require the same feature flag
and coefficient on resume; this experiment is intended for a fresh run.

For example, add `--exp-self-consistency --self-consistency-weight 1.0` to the
existing Hex5 command. No additional evaluation or logging backend is required;
both Trackio and W&B receive the new scalar metrics through the usual host loop.

By default, a completed start is eligible when at least one of its K+1 real
positions is non-exploratory. Exploratory roots are retained when they lead to
exploit positions within the unroll. All-exploration windows are excluded;
terminal padding and the next game cannot make a start eligible.
Policy and value losses are masked at each exploratory position. Reward loss
is masked for the action taken at an exploratory source position. Consistency
uses the future target position's mask. Actions and observations remain intact,
so losses at later exploit positions still backpropagate through exploratory
transitions. Absorbing zero-value/reward padding keeps its existing masks.

`--exp-train-on-exploit-unroll-states` enables this masking and is on by
default. Use `--no-exp-train-on-exploit-unroll-states` to train direct losses
at exploratory positions too, including all-exploration windows. It is independent
of `--exp-self-consistency`. This flag does not change action selection. Policy
targets remain search distributions; value targets remain completed self-play
returns, including any exploratory moves. Positions still wait for completion.

### Differences from the MuZero paper

The reference is [MuZero, Appendix G: Training](https://arxiv.org/pdf/1911.08265),
which specifies board-game exceptions to the general three-head loss in the
main text. This implementation is not an exact reproduction of that setup.

| Detail | Paper's board-game setup | This implementation and implication |
| --- | --- | --- |
| Reward loss | Omitted for board games without intermediate rewards. | `muzero_loss` retains immediate-reward MSE, including terminal rewards. This adds a separate transition prediction objective; many zero targets can make average reward loss small without accurate terminal predictions. |
| Loss scaling | Appendix G describes scaling each head's loss by `1/K`. | Each head is averaged over its own unmasked positions across the batch and unroll. With all positions enabled, policy/value divide by `B*(K+1)` and reward by `B*K`. With fewer supervised positions, their individual weights increase instead of the head's contribution shrinking under a fixed denominator. |
| Replay eligibility | Board-game states are sampled uniformly. | Replay entries are sampled uniformly after eligibility filtering. By default, windows need at least one real exploit state and direct losses at exploratory positions are masked. This changes the supervised population. |
| Consistency | Not part of the original MuZero loss. | Optional `--exp-self-consistency` adds an EfficientZero-style negative cosine objective between predicted future latents and representations of actual future observations, through projection heads with a stopped target gradient. |

Terminal rewards still supply value targets even when a separate reward loss is
omitted. For board games, extending the return to termination leaves no estimated
continuation value: with discount 1 and no intermediate rewards, each value target
is the final outcome from that position's player perspective. Our completed-return
backfill follows this convention. Policy/value are supervised at the root and
each imagined position, subject to the masks described above.

For example, with K=5 and no masking, our six policy/value losses are divided by
6 per sample, rather than 5 under a literal fixed-`1/K` sum. If only two policy
positions per sample remain supervised, their sum is divided by 2 rather than 5.
Reward has five predictions because the root has no predicted incoming reward.

See `unroll_muzero_predictions` and `muzero_loss` in [muzero.py](../muzero.py),
and `make_muzero_selfplay_buffer` in
[buffers.py](../src/nanoalphazero/buffers.py). The unroll retains the paper's
0.5 gradient scaling at the dynamics input; representation and recurrent latent
states retain min/max normalization to [0, 1], with an epsilon for constant states.
K defaults to 5. These differences have not been established as improvements by
controlled strength comparisons.

Both dense and compressed BNK targets are supported. The MuZero search adapter
selects and renormalizes top-K probabilities from the existing dense search
output, avoiding its legacy BNK one-hot gather without changing the AlphaZero
search function. Chess uses 128 weights and uint16 action indices per target.
KL normalizes predictions over all 4,672 actions before gathering stored target
indices. Compression can truncate target mass; it is not identical to retaining
the dense target. `compress_muzero_policy` also returns retained probability mass
for independent inspection. Every 50 cycles (and the first and final cycles), a separate
probe logs retained mass and support overflow for up to 32 current roots. These
are snapshot diagnostics, not full-collection averages. Probe keys do not advance
the training RNG; `--diagnostic-period 0` disables the probe and memory
telemetry.

At the same interval, Hex runs inspect every possible first move, using both
the actual next-board observation and the learned latent transition. The host
logs opening policy entropy, logit summaries, real/latent value agreement,
predicted reward error, and value error against solved opening outcomes. Values
after Black's opening are from White's perspective. These are model-head
diagnostics, not MoHex match results or a proof of playing strength.

With Trackio or W&B enabled, the same probe also logs interactive Plotly opening
logit and real/latent value heatmaps, top/bottom move highlights, numeric tables
including reply logits and legality, and head-output histograms. Trackio uses
HTML figures; W&B uses Plotly objects. Without a tracker, scalars still appear in
JSON logs and no chart files are written. `--diagnostic-period 0` disables
all these probes. The fixed opening suite does not advance training RNG or
modify games, replay, or optimizer state. Historical episode/replay histograms
are not copied because they describe the old episodic collector.

Chess observations stay packed through self-play, consumption, and replay.
Only sampled training roots (and future targets when self-consistency is enabled)
decode. Unroll gathering reads required transition fields directly and never
decodes an entire replay buffer. Terminal padding
actions use the full action space, independently of target compression width.
Data-parallel placement uses existing buffer initialization and the device mesh;
TPU performance and compiled placement still need runtime verification.

There are no inherited research limits on model width, search budget, or W&B.
Established game defaults apply. Replay capacity counts sequence items, so its
memory cost is higher than AlphaZero's position replay. Size it deliberately
before a full run. `selfplay_buffer_max_len` defaults to extra headroom for
completed games waiting until the cycle drain. Chess defaults to 4,096,000
replay items and block/unroll rematerialization. Warmup uses the inherited
AlphaZero replay-warmup setting. These settings do not imply that every TPU fits.
Before allocating buffers the assembly reports their logical and per-device
bytes and checks the combined persistent requirement against available device
statistics (75% budget by default). The host compiles and reports cycle memory
before warmup. Temporary arrays and optimizer storage require additional room.

MuZero search honors AlphaZero's root-temperature option. `--num-iters` now has
AlphaZero's meaning for both defaults and explicit overrides: a fresh run executes
`num_iters // cycle_n_selfplay` training cycles (Hex5: 34,000 / 20 = 1,700).
Buffer warmup uses `replay_buffer_warmup_steps // cycle_n_selfplay` cycles, with
frozen optimizer and LR counters. `--replay-buffer-warmup-steps` overrides that
step budget; there is no MuZero-specific warmup option. Both divisions round
down, exactly as AlphaZero does. Evaluation and checkpoint periods remain cycles.
On resume, the host also follows AlphaZero's existing calculation:
`(num_iters - model_ts.step) // cycle_n_selfplay` additional cycles. The saved
model step counts optimizer updates; this inherited formula is not equivalent
to subtracting completed self-play steps when N and M differ.
Earlier MuZero runs used cycle-valued `num_iters` and `muzero_warmup_cycles`;
their configs do not satisfy the current strict resume-config check. Historical
artifacts are preserved; convert old command budgets explicitly before new runs.
MuZero still learns latent transitions and rewards and trains unroll samples,
so matching these settings does not make it identical to AlphaZero.
Hex self-play storage has a 512-slot minimum by default: the shorter Hex5 ring
hit the former overwrite guard in runtime testing. Random selection may retain a
fresh remainder across cycles. Capacity defaults remain unchanged, but ring
evictions now count dropped data instead of stopping training:
`selfplay_buffer/dropped_pending_count` counts discarded positions awaiting a
game outcome, and `selfplay_buffer/dropped_completed_count` counts discarded
completed starts not yet consumed. Both are cumulative counts across all lanes;
`selfplay_buffer/overwritten_count` is their total. Already consumed starts and
completed ineligible positions do not count as dropped samples.
Chronological writes evict a start before its successors, so surviving starts
still have complete forward windows. If an unfinished game outlives the ring,
its retained suffix receives correct returns at termination; discarded prefix
positions are not needed for those returns. Missing transitions within a sampled
window and unsupported environment truncation remain fatal errors.

Full checkpoints use `nanoalphazero.muzero.persistent.v4` and include environment
state, both buffers, optimizer, model parameters, RNG, and cycle number. They
support exact continuation. Chess defaults to compact checkpoints using
`nanoalphazero.muzero.compact.v1`: parameters, optimizer, RNG, and cycle only.
The v4 format records position-level exploration masks and window-based start
eligibility, alongside the unified observation sequence and pending-drop counter.
Older full checkpoints are rejected: v1/v2 have different state layouts, and v3
replay lacks exploration flags needed to reconstruct the new masks. Compact v1
checkpoints remain compatible because they rebuild both buffers on resume.
Compact resume starts fresh games and buffers and reruns warmup; it is not exact
trajectory continuation. Compact checkpoints allow changing
`selfplay_buffer_max_len` and `buffer_layout` when rebuilding buffers; all other
resolved settings must match. Full checkpoints require identical configuration.
Full checkpoints retain their selected layout; switching a full checkpoint's
layout requires conversion and is not implemented. Resume a features checkpoint
with `--buffer-layout features` again. Checkpoints without a layout
setting use `original`. Compact checkpoints can resume with either layout.
Under the former overwrite guard, the 552-slot chess capacity failed a real run
after warmup, despite passing short memory probes. The default capacity remains
2073 slots with N=20, K=5; monitor drop counts when reducing capacity.
A bounded TPU recovery check with b10c256, batches of 4096, replay length 128,
and consistency enabled passed 76 warmup plus 32 training cycles: 2160 self-play
steps crossed the larger ring's first wrap with zero buffer errors. The compact
checkpoint advanced from 20 to 660 optimizer updates. This is not a guarantee
against evictions in arbitrarily long runs. These TPU measurements predate the
change from fatal overwrites to logged evictions.
Historical episode-based
checkpoints cannot resume this pipeline. The host returns the final runner
state; exposed component handles can be used for evaluation independently.
Historical research diagnostic/evaluation CLIs remain available for their
original checkpoint format; they have not been switched to this format.

MuZero groups checkpoint CLI options under `--ckpt.*`:

| Option | Meaning |
| --- | --- |
| `--ckpt.save PATH` | Save to this path; omit to disable saving. |
| `--ckpt.resume PATH` | Restore this checkpoint; does not enable saving. |
| `--ckpt.every N` | Save every N training cycles; defaults to the game's checkpoint period. Zero disables periodic saves, but first/final saves remain. |
| `--ckpt.mode compact\|full` | Snapshot contents for saving and restoring; defaults to compact for chess, full otherwise. Must match the checkpoint when resuming. |

These replace the flat MuZero options `--save`, `--resume`, `--ckpt-period`, and
`--[no-]muzero-checkpoint-replay`. AlphaZero's CLI is unchanged. This grouping
preserves checkpoint formats, configuration matching, and save timing.

Run chess with `uv run python muzero.py --env chess --ckpt.save artifacts/chess.safetensors`.
Saving remains opt-in; the first cycle, periodic cycles (50 for chess), and final
cycle update that path. Add `--ckpt.mode full` for full snapshots,
`--mcts-num-k-actions N` to set target support, or `--no-exp-bnk-action-weights`
for dense targets. `--replay-buffer-total-size` counts unroll items;
`--selfplay-buffer-max-len` counts transitions per lane. An explicit
`--replay-buffer-max-len` overrides the capacity calculation.

The portable exporter now writes `muzero-singlefile.py`, never root `muzero.py`.
It exports the historical package sources and excludes appended production
components. The pre-existing standalone snapshot had already diverged from
those sources; this change preserves that snapshot's behavior rather than
silently refreshing it. Running the exporter explicitly refreshes the snapshot.

## Validation handoff

For periodic Hex4 strength checks during persistent training, use root MuZero's
MoHex adapter (the engine pool starts before JAX initializes):

```bash
uv run python muzero.py --env hex4 --platform tpu --num-iters 5000 \
  --replay-buffer-warmup-steps 20 --muzero-unroll-steps 3 --no-enable-wandb \
  --ckpt.mode compact --ckpt.every 50 \
  --hex-eval-period 50 --hex-eval-engine-path /path/to/mohex \
  --hex-eval-output artifacts/hex4-NEW/mohex \
  --ckpt.save artifacts/hex4-NEW/latest.safetensors
```

The evaluation directory must be new. Baseline, scheduled, and final evaluations
play all 16 openings against MoHex in both greedy-policy and learned-search
modes, with independent evaluation RNGs. `MUZERO_MOHEX_RESULT` prints the scores;
`scores.jsonl` retains them alongside raw games and compact evaluated checkpoints.
The Hex4 target is all four known winning openings (actions 3, 6, 9, 12) in both
modes with zero unscored games. Consecutive successes are counted; three passes
match the historical opening-suite gate. This does not prove perfect play at
every position. Compact snapshots restart games and replay on resume.
The recorded Hex strength runs used three-step unrolls. The five-step TPU
compiler failure was subsequently isolated to a cross-device gather of K+1
columns followed by slicing to K. XLA's all-reduce-scatter rewrite incorrectly
retained the extra column. Transition fields now gather exactly K columns;
state targets still gather K+1. This preserves sample semantics and avoids the
faulty compiler rewrite. The previously failing Hex4 batch-1,024 configuration
compiled and completed 100 optimizer updates at K=5 with zero buffer errors.
The strength results above have not been repeated at K=5.
The standalone [compiler reproducer](../scripts/repro_tpu_gather_slice.py)
pins the affected JAX/libtpu versions and includes an equivalent passing variant
and a four-device CPU control.

The original MacBook refactor received static checks only. The chess port adds
CPU regression coverage below. Original AlphaZero sections and all of `core.py`
retain their training semantics; the shared evaluation helpers were extracted
from the AlphaZero host without changing match or ladder rules.

On the authorized environment, first run the focused logic tests:

```bash
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \
  uv run pytest tests/test_muzero_components.py tests/test_muzero_standalone.py
```

They cover signed returns, same-player discounts, draws, terminal masks,
exploration, ring wrap, overwrite detection, identical consume selection,
self-play wrapper equivalence, research model/loss/search equivalence,
checkpoint continuation, and blocking optimizer updates on buffer errors.
Chess coverage includes a 512-step packed/sparse collection-drain-training cycle,
packed/decoded loss and gradient equivalence, dense/sparse KL gradients, full
padding action support, storage estimates, and compact/full checkpoint modes.
Then run the complete CPU suite using the repository instructions. These
checks establish logic only; they do not validate TPU performance.

After confirming no other process owns the TPU, use a small explicit run to
check compilation, stable sharding, memory, and cycle throughput before scaling
to the game defaults. Do not start that run on the MacBook.
