"""Stockfish-scored chess self-play, shared by AlphaZero and MuZero."""

from nanoalphazero.eval.acpl.config import AcplConfig
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class AcplResult:
    metrics: dict[str, float] = field(default_factory=dict)
    files: dict[str, Path] = field(default_factory=dict)


def make_acpl(*args, **kwargs):
    # CLI configuration/help must not initialize JAX.
    from nanoalphazero.eval.acpl.runner import make_acpl as factory
    return factory(*args, **kwargs)
