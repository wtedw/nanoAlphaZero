"""TOML schema for standalone Hex-vs-engine evaluations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tomllib
from typing import Any

def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    source = Path(path).expanduser().resolve()
    with source.open("rb") as handle:
        config = tomllib.load(handle)
    config["_config_path"] = str(source)
    config["_config_dir"] = str(source.parent)
    validate_config(config)
    return config, source


def validate_config(config: dict[str, Any]) -> None:
    settings = config.get("hex_eval")
    model = config.get("model")
    opponent = config.get("opponent")
    if not isinstance(settings, dict):
        raise ValueError("config requires a [hex_eval] table")
    if settings.get("game", "hex") != "hex":
        raise ValueError("hex_eval.game must be 'hex'")
    board_size = settings.get("board_size")
    if isinstance(board_size, bool) or board_size not in range(4, 10):
        raise ValueError("hex_eval.board_size must be an integer from 4 through 9")
    if settings.get("opening_suite", "all_first_moves") != "all_first_moves":
        raise ValueError("only opening_suite='all_first_moves' is supported")
    if settings.get("model_seat", "p1") != "p1":
        raise ValueError("only model_seat='p1' is supported")
    if not isinstance(model, dict) or not str(model.get("checkpoint", "")).endswith(
        ".safetensors"
    ):
        raise ValueError("[model] requires a .safetensors checkpoint")
    if not isinstance(opponent, dict):
        raise ValueError("config requires an [opponent] table")
    if opponent.get("kind", "mohex") != "mohex":
        raise ValueError("only opponent.kind='mohex' is implemented")
    if not str(opponent.get("path", "")).strip():
        raise ValueError("[opponent] requires an explicit path")
def public_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not key.startswith("_")}


def config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(
        public_config(config), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def resolve_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(config["_config_dir"]) / path
    return path.resolve()
