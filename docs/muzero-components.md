# MuZero component assembly

Root `muzero.py` is the readable assembly. `muzero-singlefile.py` is the
historical standalone research snapshot, renamed without changing its logic.
The two paths have different collection semantics. Do not compare results as
if renaming alone produced the new algorithm.

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
replay items, block/unroll rematerialization, and enough warmup cycles to advance
one game horizon. These are capacity settings, not a claim that every TPU fits.
Before allocating buffers the assembly reports their logical and per-device
bytes and checks the combined persistent requirement against available device
statistics (75% budget by default). The host compiles and reports cycle memory
before warmup. Temporary arrays and optimizer storage require additional room.

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
