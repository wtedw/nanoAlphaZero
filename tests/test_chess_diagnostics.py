"""Chess probe semantics, search invariance, and tracker media boundaries."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nanoalphazero.diagnostics import DiagnosticOutput, make_diagnostics_fn
from nanoalphazero.tracking import TrainingTracker
from nanoalphazero.diagnostics.chess import (
    SEARCH, VALUES, REPLIES, chess_figures, make_chess_probe, opening_suite,
)
from nanoalphazero.mcts import RootFnOutput, RecurrentFnOutput, gumbel_muzero_policy_1sh


@pytest.mark.parametrize("rescale", [False, True])
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("bnk", [False, True])
def test_search_diagnostics_leave_decision_and_targets_unchanged(rescale, mixed, bnk):
    def recurrent(params, keys, actions, latent):
        return RecurrentFnOutput(
            reward=actions * .01, discount=jnp.full(actions.shape, -.9),
            value=actions * .1, prior_logits=jnp.tile(jnp.arange(6.), (actions.size, 1))), latent

    root = RootFnOutput(prior_logits=jnp.array([[0., 1., 2., 3., 4., 5.]]),
                        value=jnp.array([.4]), embedding=jnp.zeros((1, 2)))
    invalid = jnp.array([[False, False, False, False, False, True]])
    def search(diagnostic):
        return jax.jit(lambda: gumbel_muzero_policy_1sh(
            {}, jax.random.PRNGKey(3), root, recurrent, num_root_considered=3,
            num_survivors=1, invalid_actions=invalid, rescale_values=rescale,
            use_mixed_value=mixed, use_bnk=bnk, num_k_actions=3,
            return_diagnostics=diagnostic))()
    normal, diagnostic = search(False), search(True)
    assert normal.diagnostics is None
    for name in ("action", "action_weights", "visit_counts"):
        np.testing.assert_array_equal(getattr(normal, name), getattr(diagnostic, name))
    for name in ("bnk_k_indices", "bnk_action_weights"):
        if bnk:
            np.testing.assert_array_equal(getattr(normal, name), getattr(diagnostic, name))
        else:
            assert getattr(normal, name) is getattr(diagnostic, name) is None
    d = diagnostic.diagnostics
    legal = ~np.asarray(invalid)
    combined = np.asarray(d["prior_logit"] + d["transformed_q"])
    # The existing mask helper subtracts the row maximum for stability.
    combined = combined - combined.max(axis=-1, keepdims=True)
    np.testing.assert_allclose(np.asarray(d["search_logit"])[legal], combined[legal], atol=2e-6)
    np.testing.assert_allclose(d["final_score"], d["gumbel"] + d["prior_logit"] + d["transformed_q"],
                               rtol=1e-6, atol=2e-6)
    np.testing.assert_allclose(diagnostic.action_weights, jax.nn.softmax(d["search_logit"]))
    visited = np.asarray(diagnostic.visit_counts) > 0
    np.testing.assert_allclose(np.asarray(d["completed_q"])[visited], np.asarray(d["backed_up_q"])[visited])
    baseline = d["mixed_value"] if mixed else d["raw_value"]
    np.testing.assert_allclose(d["advantage"], d["completed_q"] - baseline[:, None])
    for action in np.flatnonzero(visited[0]):
        assert float(d["child_value"][0, action]) == pytest.approx(action * .1)
    assert float(diagnostic.action_weights[0, -1]) == 0


def test_diagnostic_search_preserves_rng_dependent_multiple_roots():
    def recurrent(params, keys, actions, latent):
        noise = jax.vmap(jax.random.uniform)(keys)
        return RecurrentFnOutput(
            reward=noise * .1, discount=jnp.full(actions.shape, -.9),
            value=actions * .1 + noise, prior_logits=latent[:, :1] + jnp.arange(6.)), latent + noise[:, None]

    root = RootFnOutput(prior_logits=jnp.arange(18.).reshape(3, 6),
                        value=jnp.array([.1, .4, -.2]), embedding=jnp.zeros((3, 2)))
    def search(seed, diagnostics):
        return jax.jit(lambda: gumbel_muzero_policy_1sh(
            {}, jax.random.PRNGKey(seed), root, recurrent, num_root_considered=4,
            num_survivors=2, use_bnk=True, num_k_actions=3, return_diagnostics=diagnostics))()
    normal, diagnostic = search(4, False), search(4, True)
    for name in ("action", "action_weights", "visit_counts", "bnk_k_indices", "bnk_action_weights"):
        np.testing.assert_array_equal(getattr(normal, name), getattr(diagnostic, name))
    assert not np.array_equal(normal.action_weights, search(5, False).action_weights)


@pytest.fixture(scope="module")
def chess_probe_result():
    from nanoalphazero.config import get_muzero_config
    from nanoalphazero.core import make_env
    from nanoalphazero.model import make_muzero_model

    config = get_muzero_config("chess", enable_sharding=False, conv_width=4, conv_depth=1,
                              mcts_num_root_considered=4, mcts_num_survivors=2)
    env = make_env(config)
    config.update(game_obs_shape=list(env.obs_shape), game_num_actions=env.num_actions)
    model, variables = make_muzero_model(config, jax.random.PRNGKey(1))
    probe = make_chess_probe(model, env, config)
    result = probe(variables["params"])
    assert result == probe(variables["params"])
    return result


def test_chess_opening_tables(chess_probe_result):
    metrics, tables = chess_probe_result
    assert set(metrics) == {"model/init_board_value_chess"}
    assert len(opening_suite()) == 20
    assert len(tables[SEARCH]["data"]) == len(tables[VALUES]["data"]) == 20
    assert len(tables[REPLIES]["data"]) == 400
    assert sum(r[-1] for r in tables[SEARCH]["data"]) == 1
    json.dumps(tables, allow_nan=False)
    for row in tables[VALUES]["data"]:
        assert row[4] == pytest.approx(row[3] - row[2])
        assert row[6] == 0  # no opening move ends chess
        assert all(0 <= v <= 1 for v in row[7:])
    for _, opening in opening_suite():
        replies = [r for r in tables[REPLIES]["data"] if r[0] == opening]
        assert sum(r[5] for r in replies) == pytest.approx(1)
        assert sum(r[6] for r in replies) == pytest.approx(1)
    columns = tables[SEARCH]["columns"]
    for row in tables[SEARCH]["data"]:
        if row[columns.index("visits")] == 0:
            assert row[columns.index("child_value")] is None
            assert row[columns.index("backed_up_q")] is None


def test_notebook_figures_without_tracking(chess_probe_result, monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)
    monkeypatch.setitem(sys.modules, "trackio", None)
    figures = chess_figures(chess_probe_result[1])
    assert set(figures) == {SEARCH, VALUES, REPLIES}
    for figure in figures.values():
        assert "plotly" in figure.to_html(include_plotlyjs="cdn")


@pytest.mark.parametrize("backend,wrapper", [("wandb", "Plotly"), ("trackio", "Html")])
def test_only_three_tables_three_interactive_charts(chess_probe_result, monkeypatch, backend, wrapper):
    tracker = SimpleNamespace(Table=Mock(side_effect=lambda **kw: kw))
    setattr(tracker, wrapper, Mock(side_effect=lambda fig: fig))
    monkeypatch.setitem(sys.modules, backend, tracker)
    tables = chess_probe_result[1]
    run = Mock()
    logger = TrainingTracker(run, {"trackio": {}} if backend == "trackio" else {})
    logger.log_training({}, step=1, tables={f"{k}_table": v for k, v in tables.items()},
                        figures=chess_figures(tables))
    logs = run.log.call_args.args[0]
    assert len(logs) == 6
    assert tracker.Table.call_count == getattr(tracker, wrapper).call_count == 3
    for name in (SEARCH, VALUES, REPLIES):
        assert logs[name].to_json()
    assert len(logs[SEARCH].layout.updatemenus[0].buttons) == 11
    assert len(logs[REPLIES].layout.updatemenus[0].buttons) == 20


def test_unsupported_game_is_lazy_noop():
    diagnose = make_diagnostics_fn(None, None, {"env_id": "ttt"})
    assert diagnose(None) == DiagnosticOutput()


def test_host_probe_uses_local_replica_and_skips_other_hosts(monkeypatch):
    import nanoalphazero.diagnostics.chess as module

    parameter = Mock(spec=jax.Array)
    parameter.is_fully_addressable = False
    parameter.addressable_data.return_value = "local-replica"
    probe = Mock(return_value=({}, {}))
    factory = Mock(return_value=probe)
    monkeypatch.setattr(module, "make_chess_probe", factory)
    diagnose = make_diagnostics_fn(None, None, {"env_id": "chess"})
    monkeypatch.setattr(jax, "process_index", lambda: 1)
    assert diagnose({"weight": parameter}) == DiagnosticOutput()
    factory.assert_not_called()
    monkeypatch.setattr(jax, "process_index", lambda: 0)
    assert diagnose({"weight": parameter}) == DiagnosticOutput()
    probe.assert_called_once_with({"weight": "local-replica"})
