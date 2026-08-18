"""Command-line entry points for training and standalone evaluation."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys
import tomllib


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def parse_train_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    from nanoalphazero.config import CONFIG_FACTORIES

    parser = argparse.ArgumentParser(prog="train")
    parser.add_argument("--env", default="ttt", choices=list(CONFIG_FACTORIES))
    parser.add_argument(
        "--save",
        default=None,
        help="path to save params after training "
        "(default: artifacts/alphazero_<env>.safetensors)",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="do not save a checkpoint after training"
    )
    parser.add_argument(
        "--load",
        default=None,
        help="path to load params from (defaults to the save path "
        "when --play-only is set)",
    )
    parser.add_argument(
        "--enable-wandb",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="log training metrics and saved checkpoints to Weights & Biases",
    )
    parser.add_argument(
        "--play",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="play against the model interactively after training "
        "(on by default; pass --no-play to train only)",
    )
    parser.add_argument(
        "--play-only",
        action="store_true",
        help="skip training; load a checkpoint and play",
    )
    parser.add_argument(
        "--play-both",
        action="store_true",
        help="skip training and the model; enter moves for both players yourself",
    )
    parser.add_argument(
        "--play-as",
        type=int,
        default=1,
        choices=[1, 2],
        help="which player you are (1 = you move first)",
    )
    parser.add_argument(
        "--play-sims",
        type=int,
        default=None,
        help="MCTS simulations the model uses while playing",
    )
    return parser.parse_args(argv)


def run_play(config: dict, args: argparse.Namespace) -> None:
    from nanoalphazero.checkpoint import (
        apply_checkpoint_model_config,
        default_ckpt_path,
        load_checkpoint,
    )
    from nanoalphazero.play import play_against_model

    save_path = args.save or default_ckpt_path(args.env)
    params, model_config = load_checkpoint(args.load or save_path)
    config = apply_checkpoint_model_config(config, model_config)
    play_against_model(
        config,
        params,
        human_player=args.play_as - 1,
        num_simulations=args.play_sims,
    )


def run_play_both(config: dict, args: argparse.Namespace) -> None:
    from nanoalphazero.checkpoint import (
        apply_checkpoint_model_config,
        default_ckpt_path,
        load_checkpoint,
    )
    from nanoalphazero.play import play_both

    checkpoint = args.load or args.save or default_ckpt_path(args.env)
    if Path(checkpoint).expanduser().exists():
        params, model_config = load_checkpoint(checkpoint)
        config = apply_checkpoint_model_config(config, model_config)
    else:
        params = None
    play_both(config, params)


def train_main(argv: Sequence[str] | None = None) -> None:
    """Run training or an interactive play mode."""
    args = parse_train_args(sys.argv[1:] if argv is None else argv)

    from nanoalphazero.checkpoint import default_ckpt_path
    from nanoalphazero.config import CONFIG_FACTORIES
    from nanoalphazero.play import play_against_model
    from nanoalphazero.training import run_alphazero

    config = CONFIG_FACTORIES[args.env]()
    config["game_name"] = args.env
    config["enable_wandb"] = args.enable_wandb

    if args.play_both:
        run_play_both(config, args)
        return
    if args.play_only:
        run_play(config, args)
        return

    save_path = args.save or default_ckpt_path(args.env)
    runner_state = run_alphazero(
        config, ckpt_path=None if args.no_save else save_path
    )
    if args.play:
        play_against_model(
            config,
            runner_state.model_ts.params,
            human_player=args.play_as - 1,
            num_simulations=args.play_sims,
        )


def _resolve_eval_config(target: str | Path) -> tuple[Path, dict]:
    path = Path(target).expanduser()
    if not path.exists() and not path.is_absolute():
        colocated = REPOSITORY_ROOT / "evals" / path
        if colocated.exists():
            path = colocated
    if path.is_dir():
        path = path / "config.toml"
    if not path.is_file():
        raise SystemExit(f"evaluation config not found: {path}")
    try:
        with path.open("rb") as handle:
            config = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise SystemExit(f"invalid evaluation config {path}: {exc}") from exc
    return path.resolve(), config


def _reject_option(condition: bool, option: str, kind: str) -> None:
    if condition:
        raise SystemExit(f"{option} is not supported for {kind} evaluations")


def eval_main(argv: Sequence[str] | None = None) -> None:
    """Run an evaluation selected by a colocated TOML configuration."""
    parser = argparse.ArgumentParser(prog="eval")
    parser.add_argument(
        "target",
        help="evaluation name, directory, or config.toml",
    )
    parser.add_argument("--resume", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--skip-bayeselo", action="store_true")
    args = parser.parse_args(argv)

    config_path, config = _resolve_eval_config(args.target)
    config_arg = str(config_path)

    if "tournament" in config:
        game = str(config["tournament"].get("game", "chess"))
        if game != "chess":
            raise SystemExit(f"unsupported tournament game: {game!r}")
        from nanoalphazero.eval.chess.tournament import tournament_main

        tournament_main(
            config_arg,
            resume=args.resume,
            output_root=args.output_root,
            skip_bayeselo=args.skip_bayeselo,
        )
        return

    raise SystemExit(
        f"cannot determine evaluation kind from top-level tables in {config_path}"
    )


_ASSET_CHOICES = [
    "all",
    "bayeselo",
    "desert-snowball-34400",
    "desert-snowball-68800",
    "eco-openings",
    "stockfish-16",
    "searchless-9m",
    "searchless-136m",
    "searchless-270m",
]


def assets_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="assets")
    parser.add_argument("action", choices=["fetch", "verify"])
    parser.add_argument("targets", nargs="*", choices=_ASSET_CHOICES)
    parser.add_argument("--root", default=None)
    args = parser.parse_args(argv)

    from nanoalphazero.eval.chess.assets import assets_main as run_assets

    run_assets(args.action, args.root, args.targets)


def artifacts_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="artifacts")
    parser.add_argument("action", choices=["fetch"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--root", default=None)
    args = parser.parse_args(argv)

    from nanoalphazero.eval.wandb_artifacts import artifacts_main as run_artifacts

    run_artifacts(args.config, args.root)


def bayeselo_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="bayeselo")
    parser.add_argument("--pgn", required=True)
    parser.add_argument("--binary", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    from nanoalphazero.eval.chess.bayeselo import bayeselo_main as run_bayeselo

    run_bayeselo(args.pgn, binary=args.binary, out=args.out)
