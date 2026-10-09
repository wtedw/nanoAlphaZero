"""Opening diagnostics match the historical heads without touching training state."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import jax
import numpy as np
import pytest

from nanoalphazero.config import get_muzero_config
from nanoalphazero.core import make_env
from nanoalphazero.eval.muzero import make_muzero_head_probe
from nanoalphazero.tracking import TrainingTracker
from nanoalphazero.model import make_muzero_model


def test_diagnostic_period_uses_shared_game_default():
    from nanoalphazero.cli import parse_muzero_args
    from nanoalphazero.config import CONFIG_FACTORIES

    for env in ("ttt", "hex4", "chess", "go5"):
        config = get_muzero_config(env)
        assert config["diagnostic_period"] == CONFIG_FACTORIES[env]()["diagnostic_period"]
        assert "muzero_diagnostic_period" not in config
    assert parse_muzero_args(["--diagnostic-period", "7"]).diagnostic_period == 7
    assert get_muzero_config("hex4", diagnostic_period=0)["diagnostic_period"] == 0
    with pytest.raises(SystemExit):
        parse_muzero_args(["--muzero-diagnostic-period", "7"])


def test_head_tables_match_historical_predictions(tmp_path):
    from nanoalphazero.research.muzero.inspection import opening_head_tables

    config = get_muzero_config("hex4", conv_width=8, conv_depth=1, enable_sharding=False)
    env = make_env(config)
    config.update(game_obs_shape=list(env.obs_shape), game_num_actions=env.num_actions)
    model, variables = make_muzero_model(config, jax.random.PRNGKey(7))
    params = variables["params"]
    before = jax.tree.map(lambda x: np.asarray(x).copy(), params)
    metrics, tables, histograms = make_muzero_head_probe(model, env, config)(params)
    old_tables, old_metrics = opening_head_tables(model, params, env, config, tmp_path / "old")
    for name, table in tables.items():
        assert table["columns"] == old_tables[name]["columns"]
        for row, old_row in zip(table["data"], old_tables[name]["data"], strict=True):
            for value, expected in zip(row, old_row, strict=True):
                if isinstance(value, float):
                    assert value == pytest.approx(expected, abs=1e-5)
                else:
                    assert value == expected
    for name, expected in old_metrics.items():
        assert metrics[name] == pytest.approx(expected, abs=1e-5)
    for a, b in zip(jax.tree.leaves(before), jax.tree.leaves(params), strict=True):
        np.testing.assert_array_equal(a, b)
    assert all(np.isfinite(x).all() for x in histograms.values())
    values = histograms["model/values_after_black_moves_distribution"]
    truth = np.array([row[6] for row in tables["model/values_after_black_moves_table"]["data"]])
    assert metrics["hex_perf/mse_vs_perfect"] == pytest.approx(np.mean((values - truth) ** 2))
    reply_rows = tables["muzero/model/reply_policy_logits_table"]["data"]
    assert len(reply_rows) == 16 * 16
    assert sum(not row[-1] for row in reply_rows) == 16


@pytest.mark.parametrize("backend", ["wandb", "trackio"])
def test_tracker_chart_adapters(backend, monkeypatch):
    import sys
    from nanoalphazero.diagnostics.hex import board_table, hex_figures

    tracker = Mock()
    monkeypatch.setitem(sys.modules, backend, tracker)
    monkeypatch.setitem(sys.modules, "trackio" if backend == "wandb" else "wandb", None)
    table = board_table(np.arange(16).reshape(4, 4), highlights=[3, 6, 9, 12])
    tables = {name: table for name in (
        "model/values_after_black_moves_table", "model/logits_heatmap_top_k_table",
        "muzero/model/latent_values_after_black_moves_table",
    )}
    run = Mock()
    experiment = TrainingTracker(run, {"trackio": {}} if backend == "trackio" else {})
    experiment.log_training({"loss": 0.5}, step=3, tables=tables,
                            histograms={"distribution": np.arange(16)}, figures=hex_figures(tables))
    logs = run.log.call_args.args[0]
    assert len(logs) == 8
    assert logs["loss"] == 0.5
    assert run.log.call_args.kwargs == ({"step": 3} if backend == "trackio" else
                                       {"step": 3, "commit": False})
    wrapper = tracker.Plotly if backend == "wandb" else tracker.Html
    assert wrapper.call_count == 3
    figure = wrapper.call_args.args[0]
    np.testing.assert_array_equal(figure.data[0].z, np.arange(16).reshape(4, 4))
    assert len(figure.data[1].x) == 4


@pytest.mark.parametrize("period,expected", [(0, []), (3, [1, 3, 5])])
@pytest.mark.parametrize("game", ["hex", "chess"])
def test_host_diagnostic_cadence(period, expected, monkeypatch, game):
    if game == "hex":
        import nanoalphazero.diagnostics.hex as diagnostics
    else:
        import nanoalphazero.diagnostics.chess as diagnostics

    path = Path(__file__).resolve().parents[1] / "muzero.py"
    spec = importlib.util.spec_from_file_location("muzero_diagnostic_runner", path)
    assembly = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(assembly)
    state = SimpleNamespace(model_ts=SimpleNamespace(step=0, params={}))
    cycle = 0
    observed = []

    def run_cycle(state, warmup):
        nonlocal cycle
        cycle += 1
        return state, {"selfplay_buffer/error_count": 0, "total_loss": 1.0}

    def probe(params):
        observed.append(cycle)
        result = ({"model/init_board_value": 0.0}, {})
        return (*result, {}) if game == "hex" else result

    factory = Mock(return_value=probe)
    monkeypatch.setattr(diagnostics, "make_muzero_head_probe" if game == "hex" else "make_chess_probe", factory)
    compiled = Mock(side_effect=run_cycle)
    compiled.memory_analysis.return_value = None
    run_fn = Mock()
    run_fn.lower.return_value.compile.return_value = compiled
    algorithm = SimpleNamespace(
        config=dict(env_id=game, num_iters=5, cycle_n_selfplay=1,
                    replay_buffer_warmup_steps=0, diagnostic_period=period, eval_period=0),
        runner_state=state, run_fn=run_fn, model=None, env=None,
        data_sharding=None, compression_probe=None, run_mcts_fn=None,
    )
    assert assembly.run_muzero(algorithm) is state
    assert observed == expected
    assert factory.call_count == bool(expected)
