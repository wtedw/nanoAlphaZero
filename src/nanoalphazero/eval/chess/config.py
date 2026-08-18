"""TOML schema and validation for fixed-sample chess tournaments."""

from __future__ import annotations

import hashlib
import itertools
import json
import tomllib
from pathlib import Path
from typing import Any


PLAYER_KINDS = {"kata", "searchless", "stockfish"}
SEARCHLESS_MODELS = {"9M", "136M", "270M"}
STOCKFISH_MODES = {"standard", "all_moves"}
SEARCH_KEYS = {
    "mcts_variant",
    "mcts_num_simulations",
    "mcts_max_m",
    "mcts_num_k_actions",
    "mcts_visit_exponent",
    "mcts_visit_aggregator",
    "mcts_visit_exponent_interior",
    "mcts_visit_aggregator_interior",
    "mcts_maxvisit_init",
    "mcts_value_scale",
    "mcts_rescale_values",
    "mcts_use_mixed_value",
    "mcts_gumbel_scale",
    "mcts_use_opt_backward",
    "mcts_bnk_rehydrate_fields",
    "mcts_return_search_tree",
    "mcts_return_summary",
    "mcts_use_advantage_weights",
    "mcts_advantage_scale",
    "mcts_use_puct_interior",
}
REQUIRED_SEARCH_KEYS = {
    "mcts_variant",
    "mcts_num_simulations",
    "mcts_max_m",
    "mcts_num_k_actions",
    "mcts_visit_exponent",
    "mcts_visit_aggregator",
    "mcts_maxvisit_init",
    "mcts_value_scale",
    "mcts_rescale_values",
    "mcts_use_mixed_value",
    "mcts_gumbel_scale",
    "mcts_use_opt_backward",
    "mcts_bnk_rehydrate_fields",
    "mcts_return_search_tree",
    "mcts_return_summary",
    "mcts_use_advantage_weights",
    "mcts_advantage_scale",
    "mcts_use_puct_interior",
}


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    source = Path(path).expanduser().resolve()
    with source.open("rb") as handle:
        config = tomllib.load(handle)
    config["_config_path"] = str(source)
    config["_config_dir"] = str(source.parent)
    validate_config(config)
    return config, source


def _validate_search(name: str, search: Any) -> None:
    if not isinstance(search, dict):
        raise ValueError(f"Kata agent {name!r} requires [agents.search]")
    unknown = set(search) - SEARCH_KEYS
    if unknown:
        raise ValueError(f"Kata agent {name!r} has unknown search keys: {sorted(unknown)}")
    missing = REQUIRED_SEARCH_KEYS - set(search)
    if missing:
        raise ValueError(f"Kata agent {name!r} is missing search keys: {sorted(missing)}")
    if search["mcts_variant"] != "opt":
        raise ValueError("v4 tournament evaluation initially supports mcts_variant='opt' only")
    for key in ("mcts_num_simulations", "mcts_max_m", "mcts_num_k_actions"):
        value = search[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"Kata agent {name!r} {key} must be a positive integer")
    boolean_keys = {
        "mcts_rescale_values",
        "mcts_use_mixed_value",
        "mcts_use_opt_backward",
        "mcts_bnk_rehydrate_fields",
        "mcts_return_search_tree",
        "mcts_return_summary",
        "mcts_use_advantage_weights",
        "mcts_use_puct_interior",
    }
    for key in boolean_keys:
        if not isinstance(search[key], bool):
            raise ValueError(f"Kata agent {name!r} {key} must be boolean")
    numeric_keys = {
        "mcts_visit_exponent",
        "mcts_maxvisit_init",
        "mcts_value_scale",
        "mcts_gumbel_scale",
        "mcts_advantage_scale",
    }
    numeric_keys.update(
        key
        for key in ("mcts_visit_exponent_interior",)
        if key in search
    )
    for key in numeric_keys:
        if isinstance(search[key], bool) or not isinstance(search[key], (int, float)):
            raise ValueError(f"Kata agent {name!r} {key} must be numeric")
    for key in ("mcts_visit_aggregator", "mcts_visit_aggregator_interior"):
        if key in search and search[key] not in {"max", "sum", "mean", "log_sum"}:
            raise ValueError(f"Kata agent {name!r} {key} has unsupported value")
    if int(search["mcts_num_k_actions"]) > 4672:
        raise ValueError("mcts_num_k_actions cannot exceed the chess action space")
    if int(search["mcts_max_m"]) > int(search["mcts_num_k_actions"]):
        raise ValueError("mcts_max_m cannot exceed mcts_num_k_actions")
    if bool(search["mcts_rescale_values"]):
        raise ValueError(
            "mcts_rescale_values=true is incompatible with the opt final transform"
        )


def validate_config(config: dict[str, Any]) -> None:
    tournament = config.get("tournament")
    agents = config.get("agents")
    if not isinstance(tournament, dict):
        raise ValueError("config requires a [tournament] table")
    if str(tournament.get("game", "chess")) != "chess":
        raise ValueError("v4 initially supports game='chess' tournaments only")
    if tournament.get("scheduler") != "resident_v1":
        raise ValueError("tournament.scheduler must be 'resident_v1'")
    if not isinstance(agents, list) or len(agents) < 2:
        raise ValueError("config requires at least two [[agents]] tables")

    games = int(tournament.get("num_games_per_pair", 0))
    batch = int(tournament.get("batch_size", 0))
    if games <= 0 or batch <= 0 or games != 2 * batch:
        raise ValueError("resident_v1 requires num_games_per_pair == 2 * batch_size")
    if not bool(tournament.get("dynamic_batch", False)):
        raise ValueError("resident_v1 requires dynamic_batch=true")
    minimum = int(tournament.get("dynamic_batch_min", 32))
    if minimum <= 0:
        raise ValueError("dynamic_batch_min must be positive")

    names: list[str] = []
    stockfish_count = 0
    for agent in agents:
        if not isinstance(agent, dict):
            raise ValueError("each [[agents]] entry must be a table")
        name = str(agent.get("name", "")).strip()
        kind = str(agent.get("kind", ""))
        if not name:
            raise ValueError("every agent requires a non-empty name")
        if kind not in PLAYER_KINDS:
            raise ValueError(f"agent {name!r} has unsupported kind {kind!r}")
        if kind == "kata":
            checkpoint = str(agent.get("checkpoint", ""))
            artifact = str(agent.get("artifact_path", ""))
            filename = str(agent.get("artifact_filename", ""))
            if bool(checkpoint) == bool(artifact):
                raise ValueError(
                    f"Kata agent {name!r} requires exactly one of checkpoint or artifact_path"
                )
            if artifact and not filename:
                raise ValueError(f"Kata agent {name!r} artifact_path requires artifact_filename")
            if not (checkpoint or filename).endswith(".safetensors"):
                raise ValueError(f"Kata agent {name!r} requires a .safetensors checkpoint")
            if "model" in agent:
                raise ValueError(
                    f"Kata agent {name!r} must obtain architecture from checkpoint metadata"
                )
            _validate_search(name, agent.get("search"))
        elif kind == "searchless":
            model = str(agent.get("model", ""))
            if model not in SEARCHLESS_MODELS:
                raise ValueError(
                    f"Searchless agent {name!r} model must be one of {sorted(SEARCHLESS_MODELS)}"
                )
            sharding = str(agent.get("inference_sharding", "data_parallel"))
            if sharding not in {"single", "data_parallel"}:
                raise ValueError(f"Searchless agent {name!r} has invalid inference_sharding")
        else:
            stockfish_count += 1
            mode = str(agent.get("mode", "standard"))
            if mode not in STOCKFISH_MODES:
                raise ValueError(f"Stockfish agent {name!r} has invalid mode {mode!r}")
        names.append(name)

    if len(names) != len(set(names)):
        raise ValueError("agent names must be unique")
    if stockfish_count > 1:
        raise ValueError("resident_v1 supports at most one Stockfish entrant")

    configured = tournament.get("pairings")
    if configured is not None:
        if not isinstance(configured, list) or not configured:
            raise ValueError("tournament.pairings must be a non-empty array")
        known = set(names)
        seen: set[frozenset[str]] = set()
        for pair in configured:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError("each pairing must contain exactly two agent names")
            a, b = map(str, pair)
            if a == b or {a, b} - known:
                raise ValueError(f"invalid tournament pairing: {pair!r}")
            identity = frozenset((a, b))
            if identity in seen:
                raise ValueError(f"duplicate tournament pairing: {pair!r}")
            seen.add(identity)

    adjudication = config.get("adjudication", {"enabled": False})
    if not isinstance(adjudication, dict):
        raise ValueError("[adjudication] must be a table")
    if bool(adjudication.get("enabled", False)):
        if float(adjudication.get("time_limit", 0)) <= 0:
            raise ValueError("enabled adjudication requires positive time_limit")
        if int(adjudication.get("threshold_cp", 0)) <= 0:
            raise ValueError("enabled adjudication requires positive threshold_cp")


def tournament_pairings(config: dict[str, Any]) -> list[tuple[str, str]]:
    configured = config["tournament"].get("pairings")
    if configured is not None:
        return [(str(pair[0]), str(pair[1])) for pair in configured]
    names = [str(agent["name"]) for agent in config["agents"]]
    return list(itertools.combinations(names, 2))


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
