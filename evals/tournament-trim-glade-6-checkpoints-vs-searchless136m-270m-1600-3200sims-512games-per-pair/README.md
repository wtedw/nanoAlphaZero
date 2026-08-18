# Trim Glade 6 test-time scaling

This tournament evaluates the 10x192 Trim Glade 6 checkpoints at updates
10,400 and 20,800 against Searchless Chess 136M and 270M. Each checkpoint is
evaluated with 1,600 and 3,200 MCTS simulations over 512 games per pairing.

## Run

From the repository root, install the environment and fetch the public
reference assets:

```bash
uv sync --group dev
uv run assets fetch \
  eco-openings searchless-136m searchless-270m stockfish-16 bayeselo
```

Authenticate W&B once unless `WANDB_API_KEY` is already set:

```bash
uv run wandb login
```

Then run the tournament:

```bash
uv run eval \
  evals/tournament-trim-glade-6-checkpoints-vs-searchless136m-270m-1600-3200sims-512games-per-pair
```

The evaluation command automatically downloads the pinned Trim Glade 6 W&B
artifacts (`v12/model10400.safetensors` and `v25/model20800.safetensors`) if
they are not already cached. Public reference assets are fetched separately so
their multi-gigabyte download remains explicit.

Completed tournament units are retained under `runs/`. To continue an
interrupted run, pass its directory back with `--resume`:

```bash
uv run eval \
  evals/tournament-trim-glade-6-checkpoints-vs-searchless136m-270m-1600-3200sims-512games-per-pair \
  --resume evals/tournament-trim-glade-6-checkpoints-vs-searchless136m-270m-1600-3200sims-512games-per-pair/runs/<run-id>
```
