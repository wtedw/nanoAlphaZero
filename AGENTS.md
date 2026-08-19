# Repository Guide for Coding Agents

## Scope and ownership

- Run all commands from the repository root.
- The authoritative implementation is the Python package under
  `src/nanoalphazero/`. There is no mirrored single-file implementation or
  root-level Colab notebook on this branch.
- Core training code lives directly in `src/nanoalphazero/`. Evaluation code
  (`src/nanoalphazero/eval/`), reproducible evaluations (`evals/`), and tests
  (`tests/`) are primarily developed and maintained with coding agents.
- The custom-game Colab notebook is maintained separately on the `custom-env`
  branch. Do not recreate or synchronize it on this branch unless explicitly
  requested.
- Preserve unrelated user changes. Do not push, publish, rewrite history, or
  modify another branch unless the user explicitly asks.

## Project layout

```text
src/nanoalphazero/
  config.py       built-in game and training defaults
  core.py         fused self-play and training computation
  mcts.py         Gumbel MuZero tree search
  model.py        neural-network definitions and presets
  buffers.py      self-play and replay buffers
  training.py     host training loop, diagnostics, and logging
  checkpoint.py   safetensors checkpoint serialization
  play.py         interactive play
  cli.py          command-line entry points
  eval/chess/     chess tournament and reference-opponent tooling

evals/            evaluation configs, reports, and selected results
tests/            package and evaluation tests
docs/             detailed documentation
artifacts/        downloaded assets and local checkpoints (ignored)
data/             local datasets and openings (ignored)
```

## Setup and command entry points

Use Python 3.11 or newer. Install the project and development dependencies with:

```bash
uv sync --group dev
```

The installed commands are `train`, `eval`, `assets`, `artifacts`, and
`bayeselo`. Always invoke them through `uv run`; there is no `nanoaz` wrapper.

If a sandbox cannot write to the normal uv cache, use a temporary cache:

```bash
UV_CACHE_DIR=/tmp/nanoaz-uv-cache uv run <command>
```

## Training and play

Train using a built-in environment:

```bash
uv run train --env ttt
uv run train --env hex5
uv run train --env chess --no-play
```

Useful forms:

```bash
# Train without writing a checkpoint or opening interactive play.
uv run train --env chess --no-save --no-play

# Save somewhere other than artifacts/alphazero_<env>.safetensors.
uv run train --env chess --save artifacts/my-model.safetensors --no-play

# Load an existing checkpoint and play without training.
uv run train --env chess --play-only --load artifacts/my-model.safetensors

# Enable W&B explicitly; it is disabled by default.
uv run train --env chess --enable-wandb --no-play
```

Built-in environment names are `ttt`, `connect4`, `hex4` through `hex9`,
`chess`, and `go3` through `go9`.

Model architecture is stored in safetensors metadata. When loading a
checkpoint, preserve and apply that metadata rather than assuming the current
defaults. If changing a preset, keep `katago_preset`, `conv_depth`,
`conv_width`, activation, RVGL, and WDL settings mutually consistent.

## Tests

The complete CPU test suite needs four virtual JAX devices because
`tests/test_tpu_runtime.py` exercises a four-device resident batch:

```bash
JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=4 \
uv run pytest
```

Run a focused test module or test by name while iterating:

```bash
JAX_PLATFORMS=cpu uv run pytest tests/test_eval_config.py
JAX_PLATFORMS=cpu uv run pytest tests/test_stockfish.py -k adjudication
```

Before committing, run the relevant tests and:

```bash
git diff --check
git status --short
```

Use four-space indentation, snake_case for functions and modules, PascalCase
for classes, and absolute `nanoalphazero.*` imports.

## Chess evaluations

An evaluation is defined by `evals/<name>/config.toml`. The following target
forms are equivalent:

```bash
uv run eval <name>
uv run eval evals/<name>
uv run eval evals/<name>/config.toml
```

Fetch the public assets required by an evaluation before launching it:

```bash
uv run assets fetch \
  eco-openings searchless-270m stockfish-16 bayeselo \
  desert-snowball-34400 desert-snowball-68800

uv run assets verify \
  eco-openings searchless-270m stockfish-16 bayeselo \
  desert-snowball-34400 desert-snowball-68800
```

For configs that reference W&B artifacts, fetch the pinned versions with:

```bash
uv run artifacts fetch --config evals/<name>/config.toml
```

This may require W&B authentication. Prefer public release assets for public,
reproducible evaluations when available.

Resume an interrupted tournament using its original, unchanged config:

```bash
uv run eval <name> --resume evals/<name>/runs/<run-id>
```

Do not edit a config and then resume a run created from its old hash. To score
an existing PGN independently:

```bash
uv run bayeselo --pgn evals/<name>/runs/<run-id>/games.pgn
```

See `docs/chess-tournaments.md` for configuration details.

## Evaluation data safety

- Evaluation runs can take many hours and may be intentionally committed as
  reproducibility evidence. Never delete, replace, or regenerate an existing
  `evals/<name>/runs/` directory without explicit authorization.
- Keep raw PGNs, resolved configs, manifests, summaries, and BayesElo output
  together. They establish where a reported result came from.
- Do not commit downloaded models, credentials, W&B caches, or machine-local
  assets under `artifacts/`, `data/`, `logs/`, or `wandb/`.
- Use a new evaluation directory when changing pairings, search budgets, model
  checkpoints, or statistical design. Use a new run inside the same directory
  only when rerunning the same root config.
- A smoke evaluation must use a copied/temporary config. Do not weaken the
  committed evaluation config merely to make a quick test finish.

## TPU process safety

Training and tournament evaluation are TPU-first. Only one TPU/JAX process
should initialize libtpu at a time. Before assuming a quiet command is stalled,
remember that JAX compilation can produce no terminal output for several
minutes and inspect the process:

```bash
pgrep -af '\.venv/bin/(train|eval)|uv run (train|eval)'
```

Before diagnosing a TPU lock, inspect both the lock and its owner:

```bash
fuser -v /tmp/libtpu_lockfile 2>&1 || true
ps -fp <pid>
```

- Never remove `/tmp/libtpu_lockfile` while a live process owns it.
- Never kill a training or evaluation process merely to free the TPU. Stop it
  only with explicit user authorization or when it is conclusively a stale
  process started by the current session.
- Avoid casual `jax.devices()` probes while another TPU process is active;
  probing can initialize libtpu and contend for the lock.
- CPU tests validate logic, not TPU performance. Do not present a CPU smoke test
  as an end-to-end performance validation.

## Git and change hygiene

- Use `rg` or `rg --files` for discovery.
- Use `apply_patch` for hand-written file edits.
- Preserve dirty worktree changes that are outside the requested task.
- Stage explicit paths rather than using `git add .` when unrelated changes are
  present.
- Do not commit evaluation configs containing temporary or machine-specific
  checkpoint paths.
- Do not commit or push unless the user requests it.
