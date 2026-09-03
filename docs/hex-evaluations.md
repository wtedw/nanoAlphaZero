# Hex evaluation against MoHex

Hex evaluation plays one game for every possible first move with the
nanoAlphaZero model as P1 and MoHex as P2. A 7x7 evaluation therefore has 49
real games. There is no configurable evaluation batch size. If a multi-device
JAX runtime requires divisible array shapes, the runtime adds hidden padding
rows; those rows do not start engines or appear in results.

## Installing MoHex

MoHex is part of the Benzene Hex framework and is not downloaded or managed by
this repository. Build it separately, then set `opponent.path` to the resulting
executable. The value may be an absolute path, a `~` path, or the bare name
`mohex` when its directory is on `PATH`. No `artifacts/` location or symlink is
assumed.

The bundled default configuration enables MoHex's parallel solver with four
DFPN threads and does not impose a game, node, or time limit. Set
`opponent.config` to another configuration file when a bounded search is
needed.

## Standalone evaluation

Copy `evals/hex7-vs-mohex-example/config.toml`, update the checkpoint and MoHex
paths, then run:

```bash
uv run eval evals/<name>/config.toml
```

The run directory contains the source and resolved configurations,
`manifest.json`, ordered `games.jsonl`, `summary.json`, and
`opening-grid.svg`. Resume an incomplete run with its unchanged configuration:

```bash
uv run eval evals/<name>/config.toml --resume evals/<name>/runs/<run-id>
```

## Evaluation during training

The external-engine evaluation is disabled by default and runs alongside the
existing random/anchor Elo ladder when enabled:

```bash
uv run train --env hex7 --no-play \
  --hex-eval-period 100 \
  --hex-eval-engine mohex \
  --hex-eval-engine-path /absolute/path/to/mohex
```

Use `--hex-eval-engine-config /path/to/mohex.cfg` to replace the bundled defaults.
MoHex processes are started before JAX initializes and persist for the training
run. Metrics are logged under `hex_eval/`.

The match runtime depends on a small Hex-engine interface. A KataHex adapter can
be added later without changing game scheduling or result accounting; only the
MoHex adapter is currently implemented.
