import json
from io import StringIO
from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from nanoalphazero import core
from nanoalphazero.config import get_hex_config
from nanoalphazero.eval.hex.config import validate_config
from nanoalphazero.eval.hex.engine import (
    DEFAULT_CONFIG,
    MoHexEngine,
    action_to_vertex,
    resolve_config,
    resolve_executable,
    vertex_to_action,
)
from nanoalphazero.eval.hex.perfect_play import opening_metrics
from nanoalphazero.eval.hex.runtime import _physical_size, run_match
from nanoalphazero.eval.hex.standalone import (
    SCHEDULER_VERSION,
    _prepare_run,
    _write_grid,
)
from nanoalphazero.eval.hex.training import HexTrainingEvaluator


def test_hex_vertex_roundtrip():
    for action in range(49):
        assert vertex_to_action(action_to_vertex(action, 7), 7) == action
    with pytest.raises(ValueError, match="out of range"):
        vertex_to_action("h1", 7)
    with pytest.raises(ValueError, match="invalid"):
        vertex_to_action("nope", 7)


def test_default_config_is_bundled_and_uncapped():
    path = resolve_config("default")
    contents = path.read_text()
    assert path == DEFAULT_CONFIG.resolve()
    assert "param_mohex use_parallel_solver 1" in contents
    assert "param_dfpn threads 4" in contents
    assert "max_games" not in contents
    assert "max_time" not in contents


def test_executable_can_be_explicit_or_resolved_from_path(tmp_path, monkeypatch):
    executable = tmp_path / "mohex"
    executable.write_text("#!/bin/sh\n")
    executable.chmod(0o755)
    assert resolve_executable(str(executable)) == executable.resolve()
    monkeypatch.setenv("PATH", str(tmp_path))
    assert resolve_executable("mohex") == executable.resolve()


def test_gtp_answer_parses_success_and_rejects_error():
    engine = object.__new__(MoHexEngine)
    engine.process = SimpleNamespace(stdout=StringIO("= b3\n\n"))
    assert engine._answer() == "b3"
    engine.process = SimpleNamespace(stdout=StringIO("? illegal move\n\n"))
    with pytest.raises(IOError, match="illegal move"):
        engine._answer()


def _hex_config(size=4):
    return {
        "hex_eval": {
            "game": "hex",
            "board_size": size,
            "opening_suite": "all_first_moves",
            "model_seat": "p1",
        },
        "model": {"checkpoint": "model.safetensors"},
        "opponent": {"kind": "mohex", "path": "/opt/mohex", "config": "default"},
    }


def test_hex_config_has_derived_batch_and_rejects_unimplemented_modes():
    config = _hex_config(7)
    validate_config(config)
    assert "batch_size" not in config["hex_eval"]
    config["hex_eval"]["model_seat"] = "p2"
    with pytest.raises(ValueError, match="model_seat"):
        validate_config(config)
    config = _hex_config(7)
    config["opponent"]["kind"] = "katahex"
    with pytest.raises(ValueError, match="only.*mohex"):
        validate_config(config)


def test_physical_padding_does_not_change_real_game_count(monkeypatch):
    monkeypatch.setattr("nanoalphazero.eval.hex.runtime.jax.device_count", lambda: 4)
    assert _physical_size(49, sharded=True) == 52
    assert _physical_size(49, sharded=False) == 49


def test_opening_metrics_require_the_correct_openings():
    records = [
        {"opening_action": action, "winner": 0 if action in {3, 6, 12, 0} else 1}
        for action in range(16)
    ]
    metrics = opening_metrics(4, records)
    assert metrics["perfect_opening_wins"] == 3
    assert metrics["perfect_opening_total"] == 4
    assert metrics["unexpected_opening_wins"] == 1
    assert metrics["perfect_play_achieved"] == 0


def test_standalone_run_manifest_and_grid_are_reproducible(tmp_path):
    source = tmp_path / "config.toml"
    source.write_text("[hex_eval]\n")
    config = _hex_config(4)
    config["_config_dir"] = str(tmp_path)
    config["_config_path"] = str(source)
    config["hex_eval"]["output_root"] = "runs"

    run_dir, manifest = _prepare_run(config, source, None, None)
    assert (run_dir / "config.toml").read_text() == source.read_text()
    assert manifest["scheduler_version"] == SCHEDULER_VERSION
    resumed_dir, resumed = _prepare_run(config, source, run_dir, None)
    assert resumed_dir == run_dir
    assert resumed == manifest

    records = [
        {
            "opening_action": action,
            "opening_vertex": action_to_vertex(action, 4),
            "winner": action % 2,
            "plies": 4,
        }
        for action in range(16)
    ]
    grid = run_dir / "opening-grid.svg"
    _write_grid(grid, records, 4)
    assert grid.read_text().startswith('<svg xmlns="http://www.w3.org/2000/svg"')
    assert "Hex 4x4: model P1 vs MoHex" in grid.read_text()


class _FakePool:
    def __init__(self, size):
        self.size = size
        self.engines = [object() for _ in range(size * size)]
        self.occupied = []

    def reset(self, openings):
        self.occupied = [{int(opening)} for opening in openings]

    def generate(self, color, active):
        assert color == "w"
        return [
            next(action for action in range(self.size * self.size) if action not in board)
            if is_active
            else -1
            for board, is_active in zip(self.occupied, active, strict=True)
        ]

    def play(self, color, actions, active):
        for board, action, is_active in zip(self.occupied, actions, active, strict=True):
            if is_active:
                assert action not in board
                board.add(int(action))


def test_match_plays_exactly_one_game_per_opening():
    config = get_hex_config(4)
    config["enable_sharding"] = False
    env = core.make_env(config)

    def first_legal(_key, state, _params, _gumbel, batch_size):
        assert batch_size == 16
        return SimpleNamespace(
            action=jnp.argmax(state.legal_action_mask, axis=-1).astype(jnp.int32)
        )

    records, summary = run_match(first_legal, env, config, {}, _FakePool(4))
    assert len(records) == 16
    assert {record["opening_action"] for record in records} == set(range(16))
    assert summary["real_games"] == 16
    assert summary["physical_batch_size"] == 16
    assert summary["padding_rows"] == 0
    assert summary["model_wins"] + summary["model_losses"] == 16
    assert all(record["moves"][0]["action"] == record["opening_action"] for record in records)


def _training_match_result():
    records = [
        {
            "opening_action": action,
            "opening_vertex": action_to_vertex(action, 4),
            "winner": action % 2,
            "model_won": action % 2 == 0,
            "termination": "normal",
            "plies": 4,
            "moves": [],
        }
        for action in range(16)
    ]
    summary = {
        "board_size": 4,
        "real_games": 16,
        "physical_batch_size": 16,
        "padding_rows": 0,
        "model_wins": 8,
        "model_losses": 8,
        "unscored": 0,
        "model_win_rate": 0.5,
        "total_plies": 64,
        "elapsed_seconds": 1.0,
        "model_search_seconds": 0.25,
        "mohex_search_seconds": 0.75,
        "perfect_opening_wins": 2,
        "perfect_opening_total": 4,
        "perfect_opening_fraction": 0.5,
        "perfect_play_achieved": 0,
        "unexpected_opening_wins": 6,
    }
    return records, summary


def test_training_eval_writes_agent_readable_observation_files(
    tmp_path, monkeypatch, capsys
):
    records, summary = _training_match_result()
    monkeypatch.setattr(
        "nanoalphazero.eval.hex.runtime.run_match",
        lambda *_args, **_kwargs: (records, summary),
    )
    output_dir = tmp_path / "hex4.hex-eval"
    evaluator = HexTrainingEvaluator(5, object(), output_dir, board_size=4)

    waiting = json.loads((output_dir / "status.json").read_text())
    assert waiting["state"] == "waiting"
    assert evaluator.run_if_due(4, None, None, {}, None) == {}
    assert not (output_dir / "latest.json").exists()

    metrics = evaluator.run_if_due(
        5,
        None,
        None,
        {"boardsize": 4},
        None,
        train_step=320,
    )
    latest = json.loads((output_dir / "latest.json").read_text())
    status = json.loads((output_dir / "status.json").read_text())
    history = [json.loads(line) for line in (output_dir / "history.jsonl").read_text().splitlines()]
    games = (output_dir / "games-cycle-000005.jsonl").read_text().splitlines()

    assert metrics["hex_eval/model_win_rate"] == 0.5
    assert latest["state"] == status["state"] == "complete"
    assert latest["cycle"] == 5
    assert latest["train_step"] == 320
    assert latest["opening_results"][0]["result"] == "W"
    assert latest["opening_results"][1]["result"] == "L"
    assert history == [latest]
    assert len(games) == 16
    assert "HEX_EVAL_RESULT cycle=5 score=0.500000" in capsys.readouterr().out


def test_training_eval_records_failure_status(tmp_path, monkeypatch):
    def fail(*_args, **_kwargs):
        raise RuntimeError("engine stopped")

    monkeypatch.setattr("nanoalphazero.eval.hex.runtime.run_match", fail)
    output_dir = tmp_path / "hex4.hex-eval"
    evaluator = HexTrainingEvaluator(1, object(), output_dir, board_size=4)

    with pytest.raises(RuntimeError, match="engine stopped"):
        evaluator.run_if_due(
            1,
            None,
            None,
            {"boardsize": 4},
            None,
            train_step=64,
        )

    status = json.loads((output_dir / "status.json").read_text())
    assert status["state"] == "failed"
    assert status["error_type"] == "RuntimeError"
    assert status["error"] == "engine stopped"
    assert not (output_dir / "latest.json").exists()
