# Self-play and replay audit — 2026-09-05

The original episode collector was checked against production
`buffers.make_selfplay_buffer().add_backfill` using identical real TTT
trajectories: a five-ply win and a nine-ply draw, with rollout caps 3, 7, and 9.
All terminal value targets equal the terminal reward indexed by the player
stored at each position. Independent checks cover every real starting position,
policy/action/reward alignment, cutoff bootstrap perspective, terminal padding,
and production exploration exclusion and once-only consumption. No target
corruption was found in these cases. This is focused evidence, not a proof
that every possible environment trajectory is correct.

## Comparison and resulting change

| Property | Production AlphaZero | Original MuZero episode pipeline | New MuZero staged pipeline |
|---|---|---|---|
| Collection | Persistent asynchronous game lanes, auto-reset | One complete/capped episode per lane per collection | Same simple complete-episode collector retained |
| Pending outcomes | Staged until terminal reward arrives | Signed return scan after collection | Same signed scan before staging becomes consumable |
| Value meaning | Terminal outcome for stored player | Actor-perspective discounted return, cutoff bootstrap | Same MuZero target, verified against AZ on terminal games |
| Reward for dynamics | Not retained separately as replay target | Actual actor reward for each transition | Retained separately from value |
| Exploration | Excluded from eligible positions | Included | Excluded from eligible starts |
| Freshness | Production fresh/valid flags cleared by consume | Every episode inserted once | **Production consume function reused**, with fresh/valid index view |
| Replay sample | Uniform finished position | Uniform game, then uniform real start | Uniform consumed position, carrying contiguous unroll |
| Partial drain | Keep fewer-than-K leftovers staged | Not applicable | Same; no partial/invalid batch inserted |
| Training warmup | Collect without optimizer updates | None (only LR warmup) | Collect without updates, collection count derived from AZ warmup steps |
| Capacity | Configured number of positions | Allocated episode slots including padding | Same configured position capacity; unroll payload costs more bytes |
| Checkpoint | Production state schema | Train, episode replay, RNG | Train, position replay, fresh staging leftovers, RNG |

The new path is explicitly selected with `data_pipeline = "staged"`; old
experiment configurations retain the episode path for reproducibility.
`staging.py` stores complete episodes in a small persistent circular staging
area. Its lightweight index view calls the **unchanged production consume**
function. Selected lane/episode/start indices materialize complete unroll
targets before episodes can be overwritten. The drain loop inserts only full
consume-size batches. Independent replay sampling may revisit a consumed
position; this is normal replay, distinct from duplicate insertion.

Unconsumed positions remain fresh. If a later collection would overwrite a
fresh position, training fails explicitly; it does not silently discard data.
This guard is tested. Eight staging episode batches are the default, and
`staging/overwritten_fresh` must remain zero. Completed zero-return draws and
bootstrapped cutoffs are valid; a nonzero-value guard would incorrectly reject
them. Absorbing zero rewards/values remain loss targets, while their policies
are masked. Cutoff tails use only the boundary bootstrap and mask later losses.

## Differences that still matter

- Collection still waits for the configured episode cap and computes discarded
  padded rows. It does not match AZ's asynchronous throughput or exact number
  of real transitions per cycle. Compare measured transitions and accelerator
  time, not cycle counts alone.
- The replay priming count is `ceil(AZ warmup steps / MuZero episode cap)`;
  frozen padded rows and exploration exclusions mean this is an approximate
  collection-budget match, not an identical initial replay distribution.
- Each optimizer sample has K+1 policy/value heads and K reward targets. Equal
  training batch size and optimizer update count do not imply equal supervised
  target counts or FLOPs.
- The position replay stores materialized unrolls, so its byte size exceeds
  production's single-position payload. Absorbing random actions are drawn at
  consumption and retained on subsequent replay samples.
- MuZero action/observation storage is not yet production chess's compressed
  layout. Full-scale chess memory/performance remains unvalidated.
- The `train_batch/n_reward_*` names follow AZ's outcome/value convention;
  they count value-target signs, not immediate dynamics rewards.

## Validation and commands

```bash
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 uv run pytest tests/test_muzero_data_audit.py tests/test_muzero.py -q
uv run muzero-train evals/muzero/hex4-staged-smoke.toml --output artifacts/muzero/hex4-staged-smoke-20260905-a --hex-eval-engine-path /home/tedpsw/benzene-vanilla-cmake/build/src/mohex/mohex
uv run muzero-train evals/muzero/hex4-staged-smoke.toml --output artifacts/muzero/hex4-staged-resume-20260905-a --resume artifacts/muzero/hex4-staged-smoke-20260905-a/cycle-000001.safetensors --hex-eval-engine-path /home/tedpsw/benzene-vanilla-cmake/build/src/mohex/mohex
```

The focused CPU suite passed **16 tests**. The four-TPU smoke completed two
warmup batches, two training cycles, four optimizer updates, production MoHex
evaluation, graphs, and full checkpoints. Its first training drain inserted
320 positions and retained 6 leftovers; the second inserted 288 and retained
16. No malformed episodes or overwritten fresh positions occurred. A new
staging serialization error was caught by the checkpoint unit test and fixed
before this TPU smoke. Resume and all-environment staged checks are recorded
in STATUS.md as they finish.

## One-move graphs

The existing production Hex diagnostic prints values after every possible
first move. `inspection.py` continues to call it and now saves numeric tables plus
raw `heads.npz`: empty-board policy logits, real and latent one-move values
(both White-to-move perspective), reference opening values, and reply-policy
logits after each opening. Real legality is a table field for visualization,
never supplied as an internal search legality oracle. Long runs log only W&B
tables and the original AZ Plotly graph format for dynamic custom charts. Matplotlib and image logging were removed
at the user's request before the Hex5 run; the old local Hex4 smoke PNGs remain
as historical outputs. The chart helpers were found in the sibling `az` repo
and adapted with table/layout/palette/highlight equivalence tests. See METRICS.md.
