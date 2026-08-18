# Chess tournaments

The v4 evaluator runs fixed-sample, color-balanced chess tournaments between
nanoAlphaZero Safetensors checkpoints, DeepMind Searchless Chess models, and
Stockfish. Chess-specific code lives under `nanoalphazero.eval.chess`; the
training loop remains game-agnostic.

## First-time setup

Install the locked environment:

```bash
uv sync --group dev
```

Fetch the opening book, all Searchless releases, and BayesElo. The 136M and
270M downloads are deliberately opt-in, and every model download checks free
space before transfer and verifies a model-specific SHA-256 marker.

```bash
uv run assets fetch
uv run assets fetch searchless-136m
uv run assets fetch searchless-270m
uv run assets fetch bayeselo
```

Stockfish is external. Install Stockfish 16 and either put it at
`/usr/local/bin/stockfish` or set each relevant `path` in the TOML file.
BayesElo is preflighted before engines or JAX are initialized. To deliberately
run without it, pass `--skip-bayeselo`; the resulting `games.pgn` can be scored
later:

```bash
uv run bayeselo --pgn /path/to/games.pgn
```

Pinned W&B artifacts are downloaded independently of the tournament:

```bash
uv run artifacts fetch \
  --config evals/tournament-chess-v4-example/config.toml
```

## Running and resuming

Run by evaluation name, directory, or exact config path:

```bash
uv run eval tournament-chess-v4-example
uv run eval evals/tournament-chess-v4-example/config.toml
```

Each run writes an immutable config snapshot, resolved paths, a manifest,
per-unit PGNs, combined `games.pgn`, progress records, stage timings, player
statistics, asset hashes, and `summary.json`. BayesElo ratings are pool-relative
and should only be compared within a connected tournament pool.

Resume using the run directory:

```bash
uv run eval evals/tournament-chess-v4-example/config.toml \
  --resume evals/tournament-chess-v4-example/runs/<run-id>
```

The resume boundary is one completed first-mover unit. An interrupted unit is
deterministically replayed; a manifest with another config hash or scheduler
version is rejected.

## Resident-v1 scheduling

`resident_v1` requires:

```toml
[tournament]
game = "chess"
scheduler = "resident_v1"
num_games_per_pair = 1024
batch_size = 512
dynamic_batch = true
dynamic_batch_min = 32
```

Every pairing has two units over the same openings. In the first, entrant A
owns the side to move after every opening; in the second, entrant B does. This
works for both odd- and even-ply openings and ensures each opening is played
once with each entrant as White.

The pgx1 state stays data-parallel and resident. A KataModel or Searchless turn
covers the complete current batch. Stockfish is asked only about live boards.
Normal pgx1 stepping already handles terminal rows safely, so externally ended
and padding rows are simply ignored by the Python/PGN layer. Compaction happens
only after a complete two-ply round, when live games fall to at most half the
resident size; the target is a device-aligned power of two no smaller than
`dynamic_batch_min`.

## Entrants

### nanoAlphaZero

Use one local `.safetensors` checkpoint or a pinned W&B artifact. Architecture
is read exclusively from checkpoint metadata; model overrides and pickle files
are rejected. Multiple entrants may reference the same checkpoint with
different search settings; the checkpoint and replicated parameter tree are
cached once.

The `[agents.search]` table is authoritative. V4 initially supports only the
pinned MCTX fork's `opt` backend. Every execution-affecting field is explicit,
and the root, final, and interior Q-transforms are constructed from one
validated specification. OMCTX is not a dependency of tournament play.

### Searchless Chess

Use `kind = "searchless"` and `model = "9M"`, `"136M"`, or `"270M"`.
Data-parallel inference is the default. Candidate moves follow DeepMind's
action-value protocol, including the optimized repetition prefilter whose final
decision remains python-chess's authoritative repetition check.

### Stockfish

Only one Stockfish entrant is accepted per run. `mode = "standard"` asks the
engine for its ordinary timed move. `mode = "all_moves"` reproduces the
Searchless Chess oracle: every legal root move receives the complete time
limit, then the best score is selected. `all_moves` is much more expensive.

Stockfish workers are spawned before JAX initializes libtpu. Timeout and engine
termination trigger one worker restart and retry.

## Adjudication

Adjudication is globally on or off:

```toml
[adjudication]
enabled = true
backend = "async_pool"
path = "../../artifacts/stockfish/16/stockfish"
time_limit = 0.01
threshold_cp = 1300
pool_size = 48
threads = 1
hash_mb = 16
```

When enabled, every live post-move board is analyzed. A mate score or absolute
centipawn score beyond `threshold_cp` ends the game as a decisive result; it is
not converted to a draw. Normal checkmate, stalemate, repetition, fifty-move,
and insufficient-material results come from python-chess. Reaching `max_plies`
or a pgx/python termination disagreement produces an unscored `*` failure.

`async_pool` is the default adjudication backend. It runs persistent UCI
processes concurrently on one event loop and requests only the score field used
by adjudication. `threaded_pool` retains the previous implementation for
comparative profiling; both backends apply the same adjudication rule.
