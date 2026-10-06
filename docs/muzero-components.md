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

Full action targets are supported; compressed BNK targets are rejected.
Chess self-play observations retain AlphaZero's compression and are decoded
when constructing training samples. Data-parallel placement uses existing
buffer initialization and the device mesh; performance and compiled placement
still need runtime verification.

There are no inherited research limits on model width, search budget, or W&B.
Established game defaults apply. Replay capacity counts sequence items, so its
memory cost is higher than AlphaZero's position replay. Size it deliberately
before a full run. `selfplay_buffer_max_len` defaults to extra headroom for
completed games waiting until the cycle drain.

Checkpoints use `nanoalphazero.muzero.persistent.v1` and include environment
state, both buffers, optimizer, model parameters, RNG, and cycle number. They
require identical resolved configuration for resume. Historical episode-based
checkpoints cannot resume this pipeline. The host returns the final runner
state; exposed component handles can be used for evaluation independently.
Historical research diagnostic/evaluation CLIs remain available for their
original checkpoint format; they have not been switched to this format.

The portable exporter now writes `muzero-singlefile.py`, never root `muzero.py`.
It exports the historical package sources and excludes appended production
components. The pre-existing standalone snapshot had already diverged from
those sources; this change preserves that snapshot's behavior rather than
silently refreshing it. Running the exporter explicitly refreshes the snapshot.

## Validation handoff

No algorithm, model inference, training cycle, or runtime regression test was
run on the MacBook at the user's request. Static syntax checks and source
preservation checks were performed. The code is not yet runtime-validated.

On the authorized environment, first run the focused logic tests:

```bash
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \
  uv run pytest tests/test_muzero_components.py tests/test_muzero_standalone.py
```

They cover signed returns, same-player discounts, draws, terminal masks,
exploration, ring wrap, overwrite detection, identical consume selection,
self-play wrapper equivalence, research model/loss/search equivalence,
checkpoint continuation, and blocking optimizer updates on buffer errors.
Then run the complete CPU suite using the repository instructions. These
checks establish logic only; they do not validate TPU performance.

After confirming no other process owns the TPU, use a small explicit run to
check compilation, stable sharding, memory, and cycle throughput before scaling
to the game defaults. Do not start that run on the MacBook.
