# ACPL component

`make_acpl` returns `eval_acpl_fn` and an explicit `AcplState`. Both training
hosts call it with `(state, cycle, params)` and merge the returned scalar metrics.
It is disabled by default and does not checkpoint evaluator state.
MuZero permits changing ACPL settings on resume; training-state compatibility
checks still apply to model, optimizer, buffers, and other training settings.

The port originates in `~/az/src/alphazero/validation/chess_validation.py`.
It retains batched PGX self-play with zero Gumbel exploration, per-host game
partitioning, the seed 42 initialization/split sequence, and parallel CPU analysis
across games. Each analysis worker starts one Stockfish process and reuses it
sequentially through that game's positions. Timeouts/engine deaths retry the
whole game with a fresh process and exponential backoff. Engines and worker
threads close at the end of each evaluation. There is no W&B, NFS-root, global
mesh initialization, or custom signal-handler dependency.

`adam12.json` is an unchanged copy of the literal `ADAM12_292_OPENING_ACTIONS`
from `~/az/src/alphazero/validation/chess_opening_actions.py`. Despite its name,
the original contains 496 lines, each 24 plies long. All lines are checked for
legal python-chess replay and action round trips. A custom book uses the same
rectangular JSON array of PGX actions. Game count selects a fixed prefix.

## Scoring compatibility

Defaults retain the historical arithmetic: Stockfish depth 18, White-perspective
scores, mate mapped to +/-10,000 cp, scores clipped to +/-1,000 cp, and nonnegative
losses. The initial White score is fixed at +15 cp; each subsequent before-score
is the previous position's engine score. Forced opening moves are included.
Summary ACPL weights games equally, and summary standard deviations average the
within-game sample standard deviations (they are not pooled deviations).

`--acpl.no-include-openings` is an explicit alternate measurement: it scores only
model moves and analyses the post-opening starting board. Do not compare that
curve directly with historical book-inclusive ACPL. This is self-play scored by
Stockfish, not playing against Stockfish, and is not an Elo estimate.

The legacy outcome summary counts capped/unfinished games as draws. The port
also emits `acpl/unfinished` and writes unfinished PGNs with result `*`.
Failed engine analyses are excluded from averages, with explicit analysed/failed
counts. If all analyses fail, ACPL averages are omitted rather than reported as
zero or NaN. Per-game details retain the error and remain in the output.

## Usage

The same grouped flags work with `uv run train --env chess --no-play` and
`uv run --extra trackio python muzero.py --env chess --platform tpu`:

```text
--acpl.period 100
--acpl.stockfish-path /usr/local/bin/stockfish
--acpl.output artifacts/acpl
--acpl.num-games 496
--acpl.depth 18
--acpl.workers 512
```

Those reproduce the old book/analysis budget, not a small smoke. Evaluation
synchronously pauses training. For a bounded check use 4 games, depth 2,
2 workers, and `--acpl.max-plies 28` (24 book plies plus 4 model moves).

Each evaluation creates a fresh cycle/timestamp/UUID directory containing a
manifest, PGNs, FEN histories, per-host placement and score records, a merged PGN,
and summary. The manifest records the engine/book hashes, scoring convention,
configuration, and cycle. Per-game artifacts replace hundreds of per-game tracker
series; summary values use `acpl/*` and flow through either W&B or Trackio.

## Multihost boundary

Global game batches, host partitions, per-ply gathers/barriers, broadcast output
IDs, and final aggregation are retained. All hosts must call the function at the
same cycles with a common mesh and shared output filesystem. Multihost execution
has been validated with small search/analysis budgets on four TPU v4 hosts;
this is not a production-budget performance claim.
The caller must arrange distributed initialization, as for the training model.

## Reproducible smoke

`scripts/smoke_acpl.py` accepts a real AlphaZero checkpoint or a full/partial MuZero
checkpoint and constructs only inference components, without allocating training
buffers. It checks TPU ownership before JAX initialization. Run from repo root:

```bash
uv run --extra trackio assets fetch desert-snowball-34400
uv run --extra trackio python scripts/smoke_acpl.py \
  --algorithm az --platform cpu \
  --checkpoint artifacts/models/desert-snowball/model34400.safetensors \
  --games 4 --max-plies 28 --depth 2 --workers 2 --repeats 2 \
  --output artifacts/acpl-desert-cpu-UNIQUE
```

Use `--platform tpu` when free. For MuZero, use `--algorithm mz` and a
chess checkpoint; add `--distributed` when launching through `uv run pod run`.
The script restores architecture metadata before loading
parameters. First-call time includes compilation; subsequent calls use the same
batch shape and compiled search. These tiny runs validate integration, not strength.

### Validation on 2026-10-08

Four virtual CPU devices and four TPU v4 devices were tested with real checkpoints:
Desert Snowball `model34400` (downloaded and verified through the asset manager),
and `muzero-chess-recovery-check-20261008-010321.safetensors` (b10c256, compact v1).
Each evaluation used four games, 28 total plies, four root candidates/two
survivors, Stockfish depth 2, and two analysis workers. Two calls per process
separate the first compilation from repeated execution:

| Model / platform | First evaluation | Second evaluation | Second rollout / analysis |
| --- | ---: | ---: | ---: |
| Desert Snowball / CPU | 19.31 s | 1.41 s | 0.79 / 0.61 s |
| Desert Snowball / TPU | 25.17 s | 0.78 s | 0.16 / 0.61 s |
| MuZero / CPU | 11.75 s | 1.34 s | 0.73 / 0.61 s |
| MuZero / TPU | 18.27 s | 0.78 s | 0.16 / 0.62 s |

Setup/model restoration is excluded. All runs scored all four games with zero
analysis failures and finite metrics. Placement records confirm four addressable
TPU shards. Repeated trajectories were identical within each platform. Desert
Snowball CPU/TPU scores matched; MuZero selected some different moves across
platforms, so no cross-platform action parity is claimed.

The original `~/az` scorer was rerun over all eight first-suite TPU game records
(four per model); its means and standard deviations matched the port within
1e-12. Unit coverage includes legal replay of all 496 opening lines, color signs,
mate clipping, per-game standard deviations, engine lifetime/retries, parallel
workers, masked padded games, last-move checkmate, and partial-failure coverage.
A root MuZero CPU host cycle also emitted ACPL metrics successfully.

Evidence remains under local `artifacts/acpl-{desert,muzero}-{cpu,tpu}-20261008-*`
with console logs in `/tmp/acpl-*.log`. All smoke games reached the deliberately
short cap; this does not measure completed-game strength. The full 496-game,
depth-18 workload and multihost execution have not been benchmarked or validated.

### Multihost validation on 2026-10-15 (JST)

Four TPU v4 hosts with four devices each passed 16-game evaluations twice and a
496-game evaluation once, using a small MuZero chess model restored from a full
distributed checkpoint. Each game used 24 book plies plus four model moves and
Stockfish depth 2. The 16-game suite assigned four distinct games to each host;
the 496-game suite assigned 124. All games were analyzed, every global game ID
appeared once, and all hosts received identical aggregate metrics. Only process
0 published combined artifacts. Repeated trajectories matched exactly.

The 16-game evaluations took 16.84 s initially and 1.06 s on repeat; the 496-game
evaluation took 30.71 s, including 18.29 s rollout and 10.55 s analysis. Model
loading is excluded. All games intentionally hit the short cap; these results
are integration checks, not playing-strength results or full-game depth-18 timing.

The original scorer from `~/nfs_share/az1/src/alphazero/validation/chess_validation.py`
was run on all 16 first-suite FEN histories; all four score statistics matched to
1e-12. The inherited evaluator logic needed no rewrite. Evidence is retained in
`artifacts/acpl-pod-{16,496}-20261015-01`; launch manifests contain exact commands.
