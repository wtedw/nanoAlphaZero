# Single-file MuZero

`muzero.py` at the repository root is a readable, flattened snapshot. Copy that
one file to another machine; it does not import `nanoalphazero` or require a Git
checkout. `uv` installs the dependencies declared at its top, including the
repository's pinned PGX, MCTX, and Flashbax revisions. External MoHex executables
and reference checkpoints are supplied separately.

Prerequisites: Python 3.11 or newer, `uv`, and network access on the first run to
install dependencies. Git is needed to fetch the pinned dependency repositories,
but the script itself does not need to live in a Git checkout. Run the commands
below in the directory containing `muzero.py`. Choose a fresh output directory
each time; runs are never overwritten.

```bash
# Built-in tiny staged-replay smoke: four virtual CPU devices, four updates.
uv run muzero.py train --preset smoke-cpu --output /tmp/muzero-smoke-unique

# Existing research TOML configs work; output must be a new directory.
uv run muzero.py train evals/muzero/hex4-staged-smoke.toml --output artifacts/flat-hex4-smoke

# Checkpoints are opt-in, including the final checkpoint.
uv run muzero.py train --preset smoke-cpu --save --output /tmp/muzero-saved-unique
uv run muzero.py eval /tmp/muzero-saved-unique/cycle-000002.safetensors \
  --platform cpu --opening-plies 1 --output /tmp/muzero-eval-unique
```

`--env` and `--platform` override a config or preset. Without either a config or
preset, training uses the package's resolved Hex 4 defaults; use an explicit
research TOML for long runs. The built-in presets are `smoke-cpu`, `smoke-tpu`,
and `hex4-staged-smoke`. The CPU preset deliberately exercises staged replay.
CPU execution creates four virtual devices unless `XLA_FLAGS` already specifies
a device count; an incompatible count is rejected by the training loop.

For a small spatial Hex run on CPU, save this as `hex-smoke.toml` beside the
script. The command combines these overrides with the built-in small preset:

```toml
env = "hex4"
network = "spatial"
eval_period = 0
```

```bash
uv run muzero.py train hex-smoke.toml --preset smoke-cpu --save --output /tmp/muzero-hex-unique
uv run muzero.py train --help
uv run muzero.py eval --help
```

For TPU runs, use four physical devices and ensure no other JAX process owns
the TPU. Use `--preset smoke-tpu` for the tiny interface smoke or
`--preset hex4-staged-smoke` for the spatial Hex smoke. CPU tests do not measure
TPU performance. Long runs require `wandb = true` in the TOML and prior W&B
authentication; the script generates a run name using the environment, network,
seed, and output directory name. Checkpoints are not uploaded as artifacts.

Use `--save` to retain checkpoints. `checkpoint_period = 0` saves only the final
checkpoint; a positive period also saves at that cycle interval. To resume an
interrupted run, supply the same TOML, preset, and save settings, the checkpoint,
and a new output directory:

```bash
uv run muzero.py train config.toml --save \
  --resume artifacts/previous-run/cycle-000010.safetensors --output artifacts/resumed-run
```

Resolved configuration must match the checkpoint exactly. If held-out evaluation
was enabled, keep `heldout-initial-selfplay.npz` beside the checkpoint. Each run
stores resolved configuration, a manifest, source snapshot, and JSONL metrics in
its output directory. Evaluation writes real-game results into its own directory.
To evaluate against a reference, add `--alphazero CHECKPOINT` or
`--mohex-engine-path EXECUTABLE` to the `eval` command. Periodic MoHex evaluation
during training uses `hex_eval_period` in the TOML and
`--hex-eval-engine-path EXECUTABLE` on the command line.

The script includes both vector and spatial h/g/f networks, the existing fixed
two-rung search, real environment adapters, sequence targets, staged and episode
replay, optimizer updates, checkpoint resume, diagnostics, W&B charts, and
policy/search/AlphaZero/MoHex evaluation. Opening reference tables and solver
calls remain evaluation-only. It preserves the source implementation's current
limitations; exporting it is not new evidence of playing strength or TPU speed.

The few standalone adaptations are command dispatch, optional embedded presets,
source provenance without requiring Git, copying this script into each run's
source snapshot, and materializing the default MoHex configuration when needed.
No modules are embedded as strings or reconstructed by an import loader.

The package remains authoritative. Regenerate after package edits:

```bash
uv run evals/muzero/export_flat.py
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \
  uv run pytest tests/test_muzero_standalone.py
```

Tests compare both architectures' initial/recurrent outputs, search outputs,
unrolled losses and gradients against the package. They also train copied
scripts outside Git with package imports blocked, verify staging drains and
optimizer updates, and load the resulting checkpoints through both interfaces.
The spatial training smoke enables both block and unroll activation recomputation.
The exporter retains source comments, checks duplicate function/class names,
and records source hashes. It fails if its targeted CLI adaptations no longer
match; review and test regenerated exports before using them for experiments.
