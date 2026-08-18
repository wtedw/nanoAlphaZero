# Reproducing the Searchless Chess tournament

## Purpose

This evaluation recreates the reference tournament from DeepMind's
*Grandmaster-Level Chess Without Search*. It is a four-player full round robin
between the released Searchless Chess 9M, 136M, and 270M checkpoints and the
paper's Stockfish 16 oracle.

The Tournament Elo values reported by the paper are:

| Entrant | Paper Tournament Elo |
|---|---:|
| Searchless 9M | 2025 |
| Searchless 136M | 2259 |
| Searchless 270M | 2299 |
| Stockfish 16, 50 ms oracle | 2711 |

This is primarily a one-time parity evaluation. Its purpose is to demonstrate
that our imported Searchless models, action scoring, repetition handling,
opening/color protocol, and Stockfish oracle reproduce the strength ordering
and approximate Elo spacing of the original implementation. It is not meant
to be rerun routinely.

This is a partial reproduction focused on the publicly released Searchless
Chess checkpoints and Stockfish oracle. We omit the paper's AlphaZero and
Leela Chess Zero entrants because their original checkpoints and compatible
TPU inference setups are not available in this environment. Consequently,
this run validates the reproduced entrants and their relative Elo spacing; it
is not a reproduction of the paper's complete tournament pool.

The oracle is particularly expensive. For every Stockfish turn, it gives
every legal root move a separate 50 ms search and selects the move with the
best score. It is therefore much more costly than ordinary Stockfish with a
single 50 ms budget for the entire move.

The example uses 128 games per pairing: 64 openings, each played again with
colors reversed. Four entrants produce six pairings and 768 games total.
Adjudication is disabled, so games end only through the normal chess rules or
the configured safety cap.

## Relationship to Searchless Chess

Our tournament protocol and Searchless model integration are derived from the
original `searchless_chess` repository. The important evaluation semantics are
preserved, including the released checkpoints, candidate-move scoring,
python-chess repetition decisions, ECO openings, color reversal, and
Stockfish's all-moves oracle.

The execution strategy is different for throughput:

- The original evaluator processes games sequentially.
- Our resident scheduler keeps a batch of games in pgx1 and performs batched,
  data-parallel JAX inference on a single TPU v4 host.
- Finished rows remain out of Python-side game processing, and the resident
  batch compacts once it reaches the configured halving threshold.
- Stockfish positions are distributed across a pool of independent UCI
  workers, while each worker preserves the original per-position oracle
  algorithm.

These changes parallelize independent model and engine work; they do not
change the intended move-selection or scoring protocol.

## Reference run

Run on 2026-08-18 with scheduler `resident-v1-mctx-v1` and nanoAlphaZero git
SHA `baac240dadc94ca609476b5e92245b139359bf82`:

- 768 games and 95,265 plies
- zero failed or unscored games
- zero Stockfish worker restarts
- 4,768.10 seconds wall time (79.47 minutes)
- 0.163 games/s and 20.263 plies/s

BayesElo ratings are relative to this tournament pool and therefore have an
arbitrary additive offset:

| Rank | Entrant | Elo | BayesElo + / - | Games | Score | Draws |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Stockfish 16 oracle, 50 ms per legal move | +410 | +49 / -42 | 384 | 96% | 7% |
| 2 | Searchless 270M | +3 | +27 / -27 | 384 | 52% | 16% |
| 3 | Searchless 136M | -102 | +26 / -26 | 384 | 39% | 18% |
| 4 | Searchless 9M | -311 | +30 / -32 | 384 | 12% | 14% |

For an easier comparison with the paper, the table below adds a single
constant to our pool-relative ratings so that our 9M result is anchored at the
paper's 2025 Elo. This changes only the displayed origin, not any rating gap:

| Entrant | Paper Elo | This run, aligned at 9M | Difference |
|---|---:|---:|---:|
| Searchless 9M | 2025 | 2025 | 0 |
| Searchless 136M | 2259 | 2234 | -25 |
| Searchless 270M | 2299 | 2339 | +40 |
| Stockfish 16 oracle | 2711 | 2746 | +35 |

Given the modest 128 games per pairing and the reported BayesElo uncertainty,
this reproduces the paper's ordering and approximate spacing well. It is a
parity smoke/reference run, not a higher-precision remeasurement of the
paper's ratings.

Pairwise results are from the first-named entrant's perspective:

| Pairing | Wins | Draws | Losses |
|---|---:|---:|---:|
| 9M vs 136M | 12 | 32 | 84 |
| 9M vs 270M | 9 | 17 | 102 |
| 9M vs Stockfish oracle | 0 | 3 | 125 |
| 136M vs 270M | 30 | 31 | 67 |
| 136M vs Stockfish oracle | 0 | 8 | 120 |
| 270M vs Stockfish oracle | 1 | 14 | 113 |

The raw PGN, per-unit timing records, manifest, and `summary.json` remain in
the ignored local `runs/20260818-072706-5dc47c444b/` directory.

## Run it

Fetch the optional Searchless checkpoints and BayesElo first, then run:

```bash
uv run assets fetch
uv run assets fetch searchless-136m
uv run assets fetch searchless-270m
uv run assets fetch bayeselo
uv run eval \
  evals/tournament-searchless-all-vs-stockfish16-oracle-50ms-128games-per-pair/config.toml
```

Stockfish 16 must be available at `/usr/local/bin/stockfish`, or its path in
`config.toml` must be changed.
