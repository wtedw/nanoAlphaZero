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
    parser.add_argument(
        "--hex-eval-period",
        type=int,
        default=0,
        help="run model-as-P1 evaluation against MoHex every N cycles (0 disables)",
    )
    parser.add_argument(
        "--hex-eval-engine",
        choices=["mohex"],
        default="mohex",
        help="external Hex evaluation engine (currently only MoHex)",
    )
    parser.add_argument(
        "--hex-eval-engine-path",
        default=None,
        help="explicit MoHex executable path, or a bare command resolved through PATH",
    )
    parser.add_argument(
        "--hex-eval-engine-config",
        default="default",
        help="MoHex config path, or 'default' for the bundled uncapped solver config",
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

    from nanoalphazero.config import CONFIG_FACTORIES

    config = CONFIG_FACTORIES[args.env]()
    config["game_name"] = args.env
    config["enable_wandb"] = args.enable_wandb
    if args.hex_eval_period < 0:
        raise SystemExit("--hex-eval-period must be non-negative")
    config["hex_eval_period"] = args.hex_eval_period
    config["hex_eval_engine"] = args.hex_eval_engine
    config["hex_eval_engine_path"] = args.hex_eval_engine_path
    config["hex_eval_engine_config"] = args.hex_eval_engine_config

    engine_pool = None
    if args.hex_eval_period:
        if not args.env.startswith("hex"):
            raise SystemExit("--hex-eval-period is supported only for Hex")
        if not args.hex_eval_engine_path:
            raise SystemExit(
                "--hex-eval-engine-path is required when Hex evaluation is enabled"
            )
        # This module is stdlib-only. Start every external process before imports
        # below initialize JAX/libtpu.
        from nanoalphazero.eval.hex.engine import MoHexPool

        size = int(config["boardsize"])
        engine_pool = MoHexPool(
            args.hex_eval_engine_path,
            args.hex_eval_engine_config,
            size,
            size * size,
        )

    from nanoalphazero.checkpoint import default_ckpt_path
    from nanoalphazero.play import play_against_model
    from nanoalphazero.training import run_alphazero

    try:
        if args.play_both:
            run_play_both(config, args)
            return
        if args.play_only:
            run_play(config, args)
            return

        save_path = args.save or default_ckpt_path(args.env)
        runner_state = run_alphazero(
            config,
            ckpt_path=None if args.no_save else save_path,
            hex_eval_engine_pool=engine_pool,
        )
        if args.play:
            play_against_model(
                config,
                runner_state.model_ts.params,
                human_player=args.play_as - 1,
                num_simulations=args.play_sims,
            )
    finally:
        if engine_pool is not None:
            engine_pool.close()


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

    if "hex_eval" in config:
        if args.skip_bayeselo:
            raise SystemExit("--skip-bayeselo is not supported for Hex evaluations")
        from nanoalphazero.eval.hex.standalone import evaluation_main

        evaluation_main(
            config_arg,
            resume=args.resume,
            output_root=args.output_root,
        )
        return

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


# MuZero components

from dataclasses import dataclass
from typing import Literal

import tyro

from nanoalphazero.config import CONFIG_FACTORIES


MuZeroEnv = Literal[tuple(CONFIG_FACTORIES)]


@dataclass
class MuZeroArgs:
    """Persistent MuZero. Omitted overrides inherit the selected game's defaults."""

    env: MuZeroEnv = "hex4"
    platform: Literal["cpu", "tpu"] | None = None

    # Optional safetensors checkpoint; chess defaults to compact.
    save: Path | None = None

    resume: Path | None = None
    muzero_network: Literal["vector", "spatial"] | None = None
    hex_eval_period: int = 0
    hex_eval_engine_path: str | None = None
    hex_eval_engine_config: str = "default"
    hex_eval_output: Path | None = None

    # AlphaZero step budget, divided by cycle_n_selfplay into full cycles.
    num_iters: int | None = None

    # Self-play steps per lane for buffer warmup, rounded down to full cycles.
    replay_buffer_warmup_steps: int | None = None

    seed: int | None = None
    conv_width: int | None = None
    conv_depth: int | None = None
    selfplay_batch_size: int | None = None
    train_batch_size: int | None = None
    cycle_n_selfplay: int | None = None
    cycle_n_train: int | None = None
    muzero_unroll_steps: int | None = None
    selfplay_buffer_max_len: int | None = None
    selfplay_buffer_consume_size: int | None = None
    replay_buffer_max_len: int | None = None
    mcts_num_root_considered: int | None = None
    mcts_num_survivors: int | None = None
    lr_warmup_steps: int | None = None
    ckpt_period: int | None = None
    mcts_num_k_actions: int | None = None
    selfplay_buffer_min_len: int | None = None
    replay_buffer_total_size: int | None = None
    muzero_diagnostic_period: int | None = None
    learning_rate: float | None = None
    weight_decay: float | None = None
    muzero_discount: float | None = None
    muzero_memory_fraction: float | None = None
    enable_sharding: bool | None = None
    enable_wandb: bool | None = None
    muzero_remat_blocks: bool | None = None
    muzero_remat_unroll: bool | None = None
    exp_bnk_action_weights: bool | None = None
    muzero_checkpoint_replay: bool | None = None


def parse_muzero_args(argv: Sequence[str] | None = None) -> MuZeroArgs:
    """Tyro owns MuZero CLI types and flags; AlphaZero keeps its existing parser."""
    args = tyro.cli(MuZeroArgs, args=argv, config=(tyro.conf.DisallowNone,))
    if args.hex_eval_period < 0:
        raise ValueError("--hex-eval-period cannot be negative")
    if args.hex_eval_period and (
        not args.env.startswith("hex") or not args.hex_eval_engine_path or not args.hex_eval_output
    ):
        raise ValueError("Hex evaluation requires a Hex environment, --hex-eval-engine-path and --hex-eval-output")
    if args.hex_eval_period and args.hex_eval_output.exists():
        raise ValueError("--hex-eval-output must be a new directory")
    return args
