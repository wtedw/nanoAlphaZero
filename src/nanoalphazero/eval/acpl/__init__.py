"""Stockfish-scored chess self-play, shared by AlphaZero and MuZero."""

from nanoalphazero.eval.acpl.config import AcplConfig


def make_acpl(*args, **kwargs):
    # CLI configuration/help must not initialize JAX.
    from nanoalphazero.eval.acpl.runner import make_acpl as factory
    return factory(*args, **kwargs)


def acpl_files(state, metrics):
    """Local games and provenance for a completed evaluation; no publication."""
    if not metrics or state.last_output is None:
        return {}
    from pathlib import Path
    import jax

    if jax.process_index() != 0:
        return {}
    directory = Path(state.last_output)
    return {name: directory / name for name in ("games.pgn", "manifest.json", "summary.json")}
