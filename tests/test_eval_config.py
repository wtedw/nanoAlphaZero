from copy import deepcopy
from pathlib import Path

import pytest

from nanoalphazero.eval.chess.config import load_config, validate_config


EXAMPLE = Path("evals/tournament-chess-v4-example/config.toml")


def test_example_config_is_valid_and_explicit():
    config, _ = load_config(EXAMPLE)
    assert config["tournament"]["scheduler"] == "resident_v1"
    assert config["tournament"]["num_games_per_pair"] == 2 * config["tournament"]["batch_size"]
    assert {agent["model"] for agent in config["agents"] if agent["kind"] == "searchless"} == {
        "9M",
        "136M",
        "270M",
    }


def test_rejects_old_scheduler_name():
    config, _ = load_config(EXAMPLE)
    config["tournament"]["scheduler"] = "all_jax"
    with pytest.raises(ValueError, match="resident_v1"):
        validate_config(config)


def test_rejects_missing_mctx_setting():
    config, _ = load_config(EXAMPLE)
    candidate = next(agent for agent in config["agents"] if agent["kind"] == "kata")
    del candidate["search"]["mcts_visit_aggregator"]
    with pytest.raises(ValueError, match="missing search keys"):
        validate_config(config)


def test_rejects_model_override_and_pickle():
    config, _ = load_config(EXAMPLE)
    candidate = next(agent for agent in config["agents"] if agent["kind"] == "kata")
    candidate["model"] = {"katago_preset": "b6c96nbt"}
    with pytest.raises(ValueError, match="checkpoint metadata"):
        validate_config(config)
    del candidate["model"]
    candidate["artifact_filename"] = "model.pkl"
    with pytest.raises(ValueError, match="safetensors"):
        validate_config(config)


def test_only_one_stockfish_entrant():
    config, _ = load_config(EXAMPLE)
    stockfish = next(agent for agent in config["agents"] if agent["kind"] == "stockfish")
    duplicate = deepcopy(stockfish)
    duplicate["name"] = "stockfish-two"
    config["agents"].append(duplicate)
    with pytest.raises(ValueError, match="at most one Stockfish"):
        validate_config(config)


def test_rejects_unknown_adjudication_backend():
    config, _ = load_config(EXAMPLE)
    config["adjudication"]["enabled"] = True
    config["adjudication"]["backend"] = "deepmind_per_ply"

    with pytest.raises(ValueError, match="unsupported adjudication backend"):
        validate_config(config)
