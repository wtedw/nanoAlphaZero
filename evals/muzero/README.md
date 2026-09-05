# MuZero research

Implementation: `src/nanoalphazero/research/muzero/`.
Production core, search and models are unchanged.

```bash
# Correctness tests (CPU only).
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \
  uv run pytest tests/test_muzero.py tests/test_muzero_compatibility.py

# Four-device CPU smoke, with an unused output directory.
XLA_FLAGS=--xla_force_host_platform_device_count=4 \
  uv run muzero-train evals/muzero/smoke-cpu.toml \
  --output artifacts/muzero/my-cpu-smoke

# Check TPU processes and lock ownership before TPU commands.
uv run muzero-train evals/muzero/smoke-tpu.toml \
  --output artifacts/muzero/my-tpu-smoke

# Inherit AlphaZero defaults; long runs log to W&B without artifact uploads.
uv run muzero-train evals/muzero/hex4-alphazero-defaults-seed0.toml \
  --output artifacts/muzero/my-hex4-run

# Read architecture from the checkpoint; both policy-only and learned search.
uv run muzero-eval artifacts/muzero/my-hex4-run/cycle-000500.safetensors \
  --output artifacts/muzero/my-hex4-eval \
  --alphazero artifacts/alphazero_hex4.safetensors

# Periodic production MoHex evaluator, including raw first-move game records.
uv run muzero-train evals/muzero/hex4-alphazero-mohex-seed0.toml \
  --output artifacts/muzero/my-hex4-mohex-run \
  --hex-eval-engine-path /path/to/mohex
```

`--resume CHECKPOINT` restores optimizer, replay and training RNG using the same
resolved configuration and a new output directory. Existing directories are
never overwritten. Historical pilot snapshots remain inference-loadable; their
source snapshots and resolved configs record the exact implementation used.

All inference roots come from real observations, encoded once. Search then
passes only latents and action indices to learned dynamics. The production
two-rung Gumbel implementation supplies policy targets. Sequence replay,
absorbing-state targets and recurrent optimization are MuZero-specific.

New experiments use `data_pipeline = "staged"`: production consume selects
fresh non-exploration positions from completed episode staging, then drains
contiguous MuZero unrolls into position replay. Staging leftovers are included
in checkpoints. See [BUFFER_AUDIT.md](BUFFER_AUDIT.md) for the comparison to
AlphaZero and the exact differences that remain. `hex5-staged-smoke.toml`
and `hex5-staged-seed0.toml` provide the four-device Hex5 progression with
training batch 4096. One-move diagnostics reuse AZ's numeric W&B tables and
interactive Plotly charts, not images. [METRICS.md](METRICS.md) records the
standardized names, all monitored families, and semantic differences.

See [RESEARCH.md](RESEARCH.md) for design, experiment questions and limitations,
[COMPATIBILITY.md](COMPATIBILITY.md) for environment evidence, and
[STATUS.md](STATUS.md) for commands, run locations and results. Random-opponent
scores or a small opening set are not evidence of optimal play.
