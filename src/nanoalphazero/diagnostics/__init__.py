"""Model inspection and notebook charts, independent of experiment tracking."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DiagnosticOutput:
    scalar_metrics: dict[str, float] = field(default_factory=dict)
    tables: dict[str, dict] = field(default_factory=dict)
    figures: dict[str, Any] = field(default_factory=dict)
    histograms: dict[str, Any] = field(default_factory=dict)
    files: dict[str, Path] = field(default_factory=dict)


def make_diagnostics_fn(model, env, config):
    """Return diagnose_fn(params), lazily preparing the game's fixed probes once."""
    probe = None

    def diagnose_fn(params):
        nonlocal probe
        import jax

        if jax.process_index() != 0:
            return DiagnosticOutput()
        # Inspection runs on host zero using local replicas, not global collectives.
        params = jax.tree.map(
            lambda x: x.addressable_data(0)
            if isinstance(x, jax.Array) and not x.is_fully_addressable else x,
            params,
        )
        if config.get("env_id", "").startswith("hex"):
            from nanoalphazero.diagnostics.hex import make_muzero_head_probe, hex_figures

            if probe is None:
                probe = make_muzero_head_probe(model, env, config)
            metrics, tables, histograms = probe(params)
            return DiagnosticOutput(metrics, tables, hex_figures(tables), histograms)
        return DiagnosticOutput()

    return diagnose_fn
