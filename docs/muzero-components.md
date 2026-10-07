# MuZero component assembly

Root `muzero.py` is the readable assembly. `muzero-singlefile.py` is the
historical standalone research snapshot, renamed without changing its logic.
The two paths have different collection semantics. Do not compare results as
if renaming alone produced the new algorithm.

Root MuZero uses Tyro to parse the typed `MuZeroArgs` dataclass. Omitted options
inherit the selected game's defaults; boolean overrides retain `--flag` and
`--no-flag` syntax. AlphaZero's CLI remains unchanged. Run
`uv run python muzero.py --help` for the generated options and types.

## Components

New definitions are appended to existing package files. Original AlphaZero
definitions remain unchanged; `core.py` remains unchanged in its entirety.

| Location | MuZero responsibility |
| --- | --- |
| `muzero.py` | Assemble components; show the self-play/drain/train cycle; wrap existing self-play |
| `config.py` | `get_muzero_config`, based on established game defaults |
| `model.py` | Representation, dynamics, prediction, optional rematerialization |
| `mcts.py` | Learned recurrent callback to the existing search engine |
| `buffers.py` | Extended records, completed returns, sequence gathering |
| `training.py` | Unrolled loss, optimizer factory, host loop |
| `checkpoint.py` | Full persistent-state snapshots |
| `cli.py` | Root script arguments; existing entry points unchanged |

`make_muzero()` exposes `env`, `model`, `run_mcts_fn`, `selfplay_fn`,
`selfplay_buffer`, `replay_buffer`, `train_fn`, `runner_state`, and `run_fn`.
Individual buffer states and model train state live in `runner_state`.

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
   calculate returns. Eligible non-exploration starts become fresh. Finished
   game IDs and final step numbers are assigned using existing self-play output.
5. `consume` calls the existing AlphaZero consumer to select fresh starts and
   clear their flags. It then gathers K transitions and K+1 policy/value targets
   from the returned storage. Records remain available for overlapping samples.
6. Replay stores each complete `MuZeroTrainingSample` as one item. Its Flashbax
   `sample_sequence_length` stays 1. Sampling replay does not consume an item.

Chronology is checked with global step, game ID, and episode step. Circular
storage must retain every transition needed by pending games or fresh starts.
Overwrite and missing-sequence counters are sticky; the optimizer is skipped
when any error is present, and the host raises an exception.

Terminal tails train zero rewards and values with random padding actions and
no policy loss. Completed games alone supply training samples. If no replay
batch is ready, a cycle advances games without an optimizer update. Warmup also
advances games and fills buffers without optimizer updates.

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

Both dense and compressed BNK targets are supported. The MuZero search adapter
selects and renormalizes top-K probabilities from the existing dense search
output, avoiding its legacy BNK one-hot gather without changing the AlphaZero
search function. Chess uses 128 weights and uint16 action indices per target.
KL normalizes predictions over all 4,672 actions before gathering stored target
indices. Compression can truncate target mass; it is not identical to retaining
the dense target. `compress_muzero_policy` also returns retained probability mass
for independent inspection. Every 50 cycles (and the first cycle), a separate
probe logs retained mass and support overflow for up to 32 current roots. These
are snapshot diagnostics, not full-collection averages. Probe keys do not advance
the training RNG; `--muzero-diagnostic-period 0` disables the probe and memory
telemetry.

Chess observations stay packed through self-play, consumption, and replay.
Only sampled training roots decode. Unroll gathering reads required transition
fields directly and never decodes an entire replay buffer. Terminal padding
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
hit the buffer safety guard in runtime testing. Random selection may retain a
fresh remainder across cycles; this capacity is empirically validated, not a
guarantee against overflow. Explicit capacity overrides and error checks remain.

Full checkpoints use `nanoalphazero.muzero.persistent.v1` and include environment
state, both buffers, optimizer, model parameters, RNG, and cycle number. They
support exact continuation. Chess defaults to compact checkpoints using
`nanoalphazero.muzero.compact.v1`: parameters, optimizer, RNG, and cycle only.
Compact resume starts fresh games and buffers and reruns warmup; it is not exact
trajectory continuation. Both modes require identical resolved configuration
for resume. Historical episode-based
checkpoints cannot resume this pipeline. The host returns the final runner
state; exposed component handles can be used for evaluation independently.
Historical research diagnostic/evaluation CLIs remain available for their
original checkpoint format; they have not been switched to this format.

Run chess with `uv run python muzero.py --env chess --save artifacts/chess.safetensors`.
Saving remains opt-in; the first cycle, periodic cycles (50 for chess), and final
cycle update that path. Add `--muzero-checkpoint-replay` for full snapshots,
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
  --no-muzero-checkpoint-replay --ckpt-period 50 \
  --hex-eval-period 50 --hex-eval-engine-path /path/to/mohex \
  --hex-eval-output artifacts/hex4-NEW/mohex \
  --save artifacts/hex4-NEW/latest.safetensors
```

The evaluation directory must be new. Baseline, scheduled, and final evaluations
play all 16 openings against MoHex in both greedy-policy and learned-search
modes, with independent evaluation RNGs. `MUZERO_MOHEX_RESULT` prints the scores;
`scores.jsonl` retains them alongside raw games and compact evaluated checkpoints.
The Hex4 target is all four known winning openings (actions 3, 6, 9, 12) in both
modes with zero unscored games. Consecutive successes are counted; three passes
match the historical opening-suite gate. This does not prove perfect play at
every position. Compact snapshots restart games and replay on resume.
The three-step unroll compiled on the TPU validation machine; the default
five-step unroll at batch 1,024 hit an XLA all-reduce-scatter shape error before
training. That compiler failure remains unresolved.

The original MacBook refactor received static checks only. The chess port adds
CPU regression coverage below. Original AlphaZero sections and all of `core.py`
are preserved byte-for-byte; changes to existing package files stay within their
appended MuZero sections.

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
