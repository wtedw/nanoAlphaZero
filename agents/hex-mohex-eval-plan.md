# Port Hex Evaluation Against MoHex

## Summary

Add a shared Hex evaluator that runs the model as P1 from every possible first
move against MoHex. Expose it through standalone reproducible evaluations and
opt-in periodic training evaluation, while retaining the current random/anchor
ladder.

Only MoHex is implemented, behind an engine interface that can later accept
KataHex.

## Configuration and Interfaces

Standalone configuration:

```toml
[hex_eval]
game = "hex"
board_size = 7
opening_suite = "all_first_moves"
model_seat = "p1"
seed = 1
output_root = "runs"

[model]
checkpoint = "/path/to/alphazero_hex7.safetensors"

[opponent]
kind = "mohex"
path = "/home/user/benzene/build/src/mohex/mohex"
config = "default"
```

- Derive the real batch size as `board_size * board_size`; 7x7 always evaluates
  49 real openings. Do not expose a batch-size setting.
- On multi-device JAX only, pad the device array to the smallest required
  multiple of `jax.device_count()`. Padding rows receive no MoHex process and
  are excluded from games, metrics, manifests, and artifacts.
- Require an explicit MoHex binary path. Accept absolute paths, `~` paths, or a
  bare executable name resolved through `PATH`.
- Do not assume MoHex lives under `artifacts/`, download it, or create a symlink.
- Ship a package-local default MoHex config with the historical uncapped solver
  settings. Users may point to another config file.

Training flags:

- `--hex-eval-period N`, default `0`
- `--hex-eval-engine mohex`
- `--hex-eval-engine-path PATH`, required when enabled
- `--hex-eval-engine-config PATH|default`

## Implementation Changes

- Add `nanoalphazero.eval.hex` with coordinate conversion, solved-opening data,
  GTP handling, match orchestration, metrics, and artifact generation.
- Run one persistent MoHex process per real opening and issue searches
  concurrently. Start the process bank before JAX initialization.
- Use deterministic model search with `gumbel_scale=0`; force each possible P1
  opening before normal play begins.
- Treat malformed responses, illegal coordinates, duplicate moves, or
  unexpected engine exits as failures. Treat MoHex resignation as a model win.
- Extract the existing solved-opening table into the shared Hex package.
- Dispatch `[hex_eval]` configs through `uv run eval`. Support `--resume` and
  `--output-root`; reject the chess-only `--skip-bayeselo` option for Hex.
- Persist copied/resolved configs, a config-hashed manifest, atomic progress
  records, ordered `games.jsonl`, `summary.json`, and `opening-grid.svg`.
- During training, run the same all-first-moves suite at the configured period,
  print the outcome grid, and merge `hex_eval/*` scalars into normal
  console/W&B metrics without changing the existing Elo ladder.
- Keep a generic `HexEngine` protocol and engine factory for a future KataHex
  adapter.

## Test Plan

- Test coordinate conversion, GTP framing/errors, `reg_genmove` replay
  semantics, executable/config resolution, concurrency, and cleanup.
- Test exactly `N^2` real games, internal padding exclusion, one result per
  opening, terminal handling, resignation, illegal moves, and solved-opening
  classification.
- Test standalone validation, Hex checkpoint compatibility, CLI dispatch,
  atomic artifacts, SVG output, and exact-config resume.
- Test that training does not start MoHex by default, runs it at configured
  cycles, preserves ladder evaluation, logs metrics, and cleans up after errors.
- Add an opt-in real-MoHex smoke test with a temporary low-cost config.
- Run focused tests, the four-device CPU suite, `git diff --check`, and
  `git status --short`.

## Assumptions

- The model is evaluated only as P1, once from each forced first move.
- JAX padding is hidden and never changes the reported game count.
- MoHex installation remains manual and outside repository asset management.
- KataHex is an extension point only and is not implemented in this change.
