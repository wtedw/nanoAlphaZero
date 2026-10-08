"""Shared ladder scheduling, frozen anchors, and AlphaZero promotion semantics."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nanoalphazero.eval import ladder


def test_disabled_and_not_due_leave_state_untouched(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Disabled ladder must not initialize openings or play")
    monkeypatch.setattr(ladder, "all_opening_actions", forbidden)
    monkeypatch.setattr(ladder, "evaluate_vs", forbidden)
    ladder_fn, state = ladder.make_ladder({"eval_period": 0}, env=None, policy_fn=None)
    following, metrics = ladder_fn(state, 50, None)
    assert following is state and metrics == {}
    monkeypatch.setattr(ladder, "all_opening_actions", lambda *a, **kw: np.array([[0]]))
    ladder_fn, state = ladder.make_ladder({"eval_period": 2}, env=None, policy_fn=None)
    following, metrics = ladder_fn(state, 1, None)
    assert following is state and metrics == {}


def test_promotions_reset_streak_cap_random_and_keep_old_state(monkeypatch, capsys):
    monkeypatch.setattr(ladder, "all_opening_actions", lambda *a, **kw: np.array([[0], [1]]))
    outcomes = iter([
        (.8, 100), (.9, 200),                         # establish random baseline
        (.95, 300), (.8, 240),                       # first qualifying anchor score
        (.6, -50), (.5, 0),                          # forgetting warning, reset streak
        (.9, 300), (.8, 200), (.9, 300), (.85, 300), # promote by mean 250
    ])
    calls = []
    def match(policy, env, config, params, opponent, **kwargs):
        calls.append((opponent, kwargs["key"]))
        return next(outcomes)
    monkeypatch.setattr(ladder, "evaluate_vs", match)
    charts = []
    ladder_fn, initial = ladder.make_ladder(
        {"eval_period": 2}, env=None, policy_fn=None,
        chart_fn=lambda history: charts.append(history) or "chart",
    )
    params = {"w": jnp.array([1.])}
    first, metrics = ladder_fn(initial, 2, params)
    assert first.qualifying_deltas == (0.,)
    assert metrics["eval/elo_vs_random"] == 0
    second, metrics = ladder_fn(first, 4, params)
    assert second.rung == 1 and second.anchor_cycle == 4 and second.anchor_elo == 0
    assert metrics["eval/rung"] == 0  # report the opponent evaluated, before promotion
    assert second.anchor_params["w"] is not params["w"]
    params["w"] = jnp.array([2.])
    third, _ = ladder_fn(second, 6, params)
    assert third.qualifying_deltas == (240.,)
    fourth, metrics = ladder_fn(third, 8, params)
    assert fourth.qualifying_deltas == ()
    assert metrics["eval/elo_vs_random"] == -50
    fifth, _ = ladder_fn(fourth, 10, params)
    final, metrics = ladder_fn(fifth, 12, params)
    assert final.rung == 2 and final.anchor_elo == 250 and final.anchor_cycle == 12
    assert metrics["eval/ladder_elo"] == 300 and metrics["eval/rung"] == 1
    assert final.best_random_score == .95 and final.qualifying_deltas == ()
    assert initial.rung == 0 and initial.anchor_params is None and initial.elo_cycles == (0,)
    assert first.qualifying_deltas == (0.,) and second.anchor_params["w"][0] == 1
    assert "forgetting?" in capsys.readouterr().out
    assert len(charts) == 5
    call_index = 0
    for cycle in (2, 4, 6, 8, 10, 12):
        keys = jax.random.split(jax.random.PRNGKey(1234 + cycle))
        np.testing.assert_array_equal(calls[call_index][1], keys[0])
        assert calls[call_index][0] is None
        call_index += 1
        if cycle > 4:
            np.testing.assert_array_equal(calls[call_index][1], keys[1])
            assert calls[call_index][0] is second.anchor_params
            call_index += 1


@pytest.mark.parametrize("cap,expected", [(None, 300.), (100., 100.), (0., 0.)])
def test_random_cap_configuration(monkeypatch, cap, expected):
    monkeypatch.setattr(ladder, "all_opening_actions", lambda *a, **kw: np.array([[0]]))
    monkeypatch.setattr(ladder, "evaluate_vs", lambda *a, **kw: (.9, 300.))
    ladder_fn, state = ladder.make_ladder(
        {"eval_period": 1, "eval_promote_patience": 1, "eval_vs_random_max_elo": cap},
        env=None, policy_fn=None,
    )
    state, metrics = ladder_fn(state, 1, {"w": jnp.array([1.])})
    assert state.anchor_elo == metrics["eval/ladder_elo"] == expected


@pytest.mark.parametrize("algorithm", ["alphazero", "muzero"])
def test_ladder_accepts_both_search_adapters(algorithm):
    from nanoalphazero.config import get_muzero_config
    from nanoalphazero.core import make_alphazero

    cfg = get_muzero_config(
        "ttt", enable_sharding=True, selfplay_batch_size=4, train_batch_size=4,
        selfplay_buffer_consume_size=4, selfplay_buffer_max_len=16,
        replay_buffer_max_len=4, conv_width=8, conv_depth=1, muzero_network="vector",
        muzero_unroll_steps=3, mcts_num_root_considered=2, mcts_num_survivors=1,
        eval_period=1, eval_promote_score=0., eval_promote_patience=1,
    )
    if algorithm == "alphazero":
        component = make_alphazero(cfg, jax.random.PRNGKey(0))
    else:
        spec = importlib.util.spec_from_file_location("muzero_ladder_test", Path(__file__).parents[1] / "muzero.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        component = module.make_muzero(cfg, jax.random.PRNGKey(0))
    ladder_fn, state = ladder.make_ladder(cfg, env=component.env, policy_fn=component.run_mcts_fn)
    params = component.runner_state.model_ts.params
    state, metrics = ladder_fn(state, 1, params)
    assert state.rung == 1
    assert 0 <= metrics["eval/score_vs_random"] <= 1
    state, metrics = ladder_fn(state, 2, params)
    assert metrics["eval/score_vs_anchor"] == .5  # identical frozen/current models, both colors
    assert all(np.isfinite(value) for value in metrics.values())


def test_alphazero_host_logs_shared_ladder(monkeypatch, tmp_path):
    from nanoalphazero import training
    from nanoalphazero.config import get_muzero_config

    cfg = get_muzero_config(
        "ttt", enable_sharding=True, selfplay_batch_size=4, train_batch_size=4,
        selfplay_buffer_consume_size=4, selfplay_buffer_max_len=16,
        replay_buffer_max_len=4, conv_width=8, conv_depth=1, muzero_unroll_steps=3,
        mcts_num_root_considered=2, mcts_num_survivors=1,
        cycle_n_selfplay=10, cycle_n_train=1, num_iters=10,
        replay_buffer_warmup_steps=0, eval_period=1,
    )
    logs = []
    tracker = SimpleNamespace(summary={}, log=lambda values, **kw: logs.append(values), finish=lambda: None)
    monkeypatch.setattr(training, "_init_wandb", lambda config: tracker)
    monkeypatch.setattr(training, "_start_run_logfile", lambda config: str(tmp_path / "train.log"))
    monkeypatch.setattr(training, "_run_ttt_diagnostics", lambda *args: None)
    state = training.run_alphazero(cfg)
    assert int(state.model_ts.n_updates) == 1
    assert len(logs) == 1
    assert logs[0]["eval/rung"] == 0
    assert 0 <= logs[0]["eval/score_vs_random"] <= 1
    assert logs[0]["eval/score_vs_anchor"] == logs[0]["eval/score_vs_random"]
