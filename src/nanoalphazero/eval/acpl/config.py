"""Shared ACPL options; importing them does not initialize JAX or engines."""

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class AcplConfig:
    # Training cycles between evaluations; zero disables ACPL.
    period: int = 0
    # Stockfish UCI executable used for scoring, not as a playing opponent.
    stockfish_path: Path = Path("/usr/local/bin/stockfish")
    # Each evaluation creates a unique subdirectory containing PGNs and scores.
    output: Path = Path("artifacts/acpl")
    # JSON array of equal-length PGX opening-action sequences; default is Adam12.
    openings: Path | None = None
    # Prefix of the fixed book; the legacy "292" book actually contains 496 lines.
    num_games: int = 496
    # Total plies INCLUDING the forced opening; defaults to the game's cap.
    max_plies: int | None = None
    # Stockfish analysis depth for each position.
    depth: int = 18
    # As in ~/az: parallel games, one sequentially reused engine per game.
    workers: int = 512
    # UCI protocol timeout in seconds.
    engine_timeout: float = 30.0
    # Whole-game retries after an engine timeout or termination.
    engine_retries: int = 2
    # Legacy ACPL includes book moves; disable to score only model-selected moves.
    include_openings: bool = True


def resolve_acpl(config):
    """Hosts store only explicitly enabled ACPL settings in their resolved config."""
    values = config.get("acpl", {})
    return values if isinstance(values, AcplConfig) else AcplConfig(**values)


def acpl_config_dict(options):
    return {name: str(value) if isinstance(value, Path) else value
            for name, value in asdict(options).items()}


def add_acpl_arguments(parser):
    """Expose the same grouped options in AlphaZero's existing argparse CLI."""
    defaults = AcplConfig()
    group = parser.add_argument_group("ACPL chess evaluation")
    types = dict(period=int, stockfish_path=Path, output=Path, openings=Path,
                 num_games=int, max_plies=int, depth=int, workers=int,
                 engine_timeout=float, engine_retries=int)
    for name, kind in types.items():
        group.add_argument(f"--acpl.{name.replace('_', '-')}", dest=f"acpl_{name}",
                           type=kind, default=getattr(defaults, name))
    group.add_argument("--acpl.include-openings", dest="acpl_include_openings", action="store_true", default=True)
    group.add_argument("--acpl.no-include-openings", dest="acpl_include_openings", action="store_false")


def acpl_from_args(args):
    return AcplConfig(**{name: getattr(args, f"acpl_{name}") for name in AcplConfig.__dataclass_fields__})
