"""Metric definitions, original AZ chart parity, and no raster logging."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from nanoalphazero.research.muzero.charts import board_table, board_figure, wandb_board_logs
from nanoalphazero.research.muzero.metrics import batch_metrics, standardize


def test_root_counts_do_not_count_training_unroll_or_reject_bootstrap():
    batch = dict(value=np.array([[1.] * 6, [.25] * 6]), policy_mask=np.ones((2, 6), bool),
                 value_mask=np.ones((2, 6), bool), sample_info=dict(
                     is_fresh_i8=np.ones(2, np.int8), is_from_selfplay=np.ones(2, bool),
                     is_pending_reward_i8=np.zeros(2, np.int8), is_exploration=np.zeros(2, bool),
                     terminal=np.array([True, False]), ep_step=np.array([3, 5]),
                     ep_termination_step=np.array([7, 10]), game_id=np.array([10, 11]), row_id=np.array([1, 2])))
    metrics = batch_metrics(batch, False)
    assert metrics["train_batch/n_valid"] == 2
    assert metrics["train_batch/n_reward1"] == 1
    assert metrics["muzero/train_batch/n_real_targets"] == 12
    assert metrics["muzero/train_batch/n_non_outcome_values"] == 1
    assert metrics["train_batch/n_is_invalid_envforbidsdraws_doublecheck"] == 0
    batch["sample_info"]["is_fresh_i8"][0] = 0
    assert batch_metrics(batch, False)["anomalies/train_batch/n_not_is_fresh_i8_but_valid"] == 1


def test_canonical_names_and_extra_namespace():
    metrics = standardize(dict(loss=3., policy_loss=2., value_loss=.9, reward_loss=.1, grad_norm=2.,
                               updates=20, seconds=.5, **{"heldout/unroll5/real_value_mse": .2,
                                                       "hex_eval/model_wins": 4}))
    assert metrics == {"total_loss": 3., "loss_pi": 2., "loss_v": .9, "muzero/loss_r": .1,
                       "norms/grad_norm": 2., "runner_state/n_updates": 20, "timing/cycle_seconds": .5,
                       "muzero/heldout/unroll5/real_value_mse": .2, "hex_eval/model_wins": 4}
    with pytest.raises(ValueError, match="collision"):
        standardize({"loss": 1., "total_loss": 2.})


def test_table_and_dynamic_chart_never_use_images(monkeypatch):
    import wandb
    def forbidden(*args, **kwargs):
        raise AssertionError("Raster logging is forbidden")
    monkeypatch.setattr(wandb, "Image", forbidden)
    table = board_table(np.array([[1., 2.], [3., 4.]]), highlights=[2])
    assert table["data"][2] == ["hex", 2, 2, 1, 0, 3., None, True, "r1c0"]
    logs = wandb_board_logs({"model/logits_heatmap_top_k_table": table})
    assert isinstance(logs["model/logits_heatmap_top_k"], wandb.Plotly)
    assert isinstance(logs["model/logits_heatmap_top_k_table"], wandb.Table)
    figure = board_figure(table, "test", True)
    assert figure.layout.yaxis.scaleanchor == "x"
    assert figure.layout.yaxis.autorange == "reversed"
    assert list(figure.data[1].x) == [0] and list(figure.data[1].y) == [1]


def test_chart_matches_original_az_helper_when_checkout_available():
    source = Path(__file__).resolve().parents[2] / "az/src/alphazero/diagnostics/opening_value_charts.py"
    if not source.exists():
        pytest.skip("Optional read-only comparison requires the sibling az checkout")
    spec = importlib.util.spec_from_file_location("az_chart_reference", source)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    values = np.array([[.2, -.4], [.8, -.1]], dtype=np.float32)
    truth = np.array([[1., -1.], [1., -1.]], dtype=np.float32)
    table = board_table(values, truth, [1, 3])
    assert table["columns"] == original.OPENING_VALUE_COLUMNS
    assert table["data"] == original.board_value_rows("hex", 2, values, truth, [1, 3])
    expected = original.board_value_heatmap_figure("test", values, [1, 3])
    actual = board_figure(table, "test")
    assert actual.layout.to_plotly_json() == expected.layout.to_plotly_json()
    np.testing.assert_array_equal(actual.data[0].z, expected.data[0].z)
    assert actual.data[0].colorscale == expected.data[0].colorscale
    assert actual.data[1].to_plotly_json() == expected.data[1].to_plotly_json()
