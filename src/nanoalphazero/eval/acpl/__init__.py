"""Stockfish-scored chess self-play, shared by AlphaZero and MuZero."""

from nanoalphazero.eval.acpl.config import AcplConfig


def make_acpl(*args, **kwargs):
    # CLI configuration/help must not initialize JAX.
    from nanoalphazero.eval.acpl.runner import make_acpl as factory
    return factory(*args, **kwargs)
