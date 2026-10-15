"""Terminal suite, backup arithmetic, and lazy diagnostic dispatch."""
import json
from unittest.mock import Mock

import jax
import numpy as np
import pgx1
import pytest

from nanoalphazero.diagnostics import make_diagnostics_fn
from nanoalphazero.diagnostics.chess.cases import terminal_cases
from nanoalphazero.diagnostics.chess.terminals import terminal_metrics


def test_terminal_probe_needs_only_default_backend(monkeypatch):
    from nanoalphazero.config import get_muzero_config
    from nanoalphazero.core import make_env
    from nanoalphazero.model import make_muzero_model
    from nanoalphazero.diagnostics.chess.terminals import make_chess_terminal_probe

    config = get_muzero_config('chess', enable_sharding=False, conv_width=4,
                              conv_depth=1, mcts_num_root_considered=2,
                              mcts_num_survivors=1)
    env = make_env(config)
    config.update(game_obs_shape=list(env.obs_shape), game_num_actions=env.num_actions)
    model, variables = make_muzero_model(config, jax.random.PRNGKey(1))
    local_devices = jax.local_devices

    def default_backend_only(*args, **kwargs):
        if kwargs.get('backend') is not None:
            raise RuntimeError('Additional backend unavailable')
        return local_devices(*args, **kwargs)

    monkeypatch.setattr(jax, 'local_devices', default_backend_only)
    probe = make_chess_terminal_probe(model, env, config, uniform_actions=2, top_actions=2)
    metrics, tables = probe(variables['params'])
    assert all(np.isfinite(value) for value in metrics.values())
    assert len(tables['chess/terminal']['data']) == 13


def test_terminal_suite_real_outcomes():
    cases = terminal_cases(pgx1.make('chess', use_bitmask=True))
    assert len(cases) == 13
    assert len({c[0] for c in cases}) == 13
    for name, _, root, _, following, _ in cases:
        assert not bool(root.terminated)
        assert bool(following.terminated)
        expected = 1 if 'mate' in name and 'stalemate' not in name else 0
        assert float(following.rewards[root.current_player]) == expected


@pytest.mark.parametrize('value_only', [False, True])
def test_correct_absorbing_predictions_have_zero_errors(value_only):
    outcomes = np.array([1., 0., -1.])
    zero = np.zeros(3)
    terminal = -outcomes if value_only else zero
    p = dict(root_value=outcomes, observed_terminal_value=zero,
             terminal_value=terminal, reward=zero if value_only else outcomes,
             greedy_value=-terminal, greedy_reward=zero, greedy2_value=terminal,
             greedy2_reward=zero, terminal_policy_entropy=zero,
             uniform_value=np.tile(-terminal, (32, 1)), uniform_reward=np.zeros((32, 3)),
             top_policy_value=np.tile(-terminal, (8, 1)), top_policy_reward=np.zeros((8, 3)))
    metrics = terminal_metrics(outcomes, value_only, p)
    for name, values in metrics.items():
        if name.endswith(('_error', '_mae')):
            np.testing.assert_array_equal(values, 0)
    for key in ('q_one_step', 'q_two_step', 'q_three_step'):
        np.testing.assert_array_equal(metrics[key], outcomes)
    p['greedy_value'] = p['greedy_value'] + .4
    p['greedy_reward'] = np.full(3, .1)
    metrics = terminal_metrics(outcomes, value_only, p)
    np.testing.assert_allclose(metrics['q_two_step_error'], .3)
    np.testing.assert_allclose(metrics['q_three_step_error'], .1)


def test_dispatch_is_lazy_and_merges_terminal_metrics(monkeypatch):
    import nanoalphazero.diagnostics.chess as opening
    import nanoalphazero.diagnostics.chess.terminals as terminal

    factory = Mock(return_value=Mock(return_value=({'chess/terminal/test': .25},
        {'chess/terminal': {'columns': ['x'], 'data': [[1]]}})))
    monkeypatch.setattr(terminal, 'make_chess_terminal_probe', factory)
    monkeypatch.setattr(opening, 'make_chess_probe', lambda *a: lambda p: ({'opening': 1.}, {}))
    monkeypatch.setattr(opening, 'chess_figures', lambda tables: {})
    diagnose = make_diagnostics_fn(None, None, {'env_id': 'chess', 'algorithm': 'muzero'})
    factory.assert_not_called()
    output = diagnose({})
    assert output.scalar_metrics == {'opening': 1., 'chess/terminal/test': .25}
    assert 'chess/terminal_table' in output.tables
    json.dumps(output.tables, allow_nan=False)
    diagnose({})
    factory.assert_called_once()
    monkeypatch.setattr(jax, 'process_index', lambda: 1)
    assert diagnose({}).scalar_metrics == {}
    assert factory.return_value.call_count == 2
