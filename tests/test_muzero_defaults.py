import pytest

from nanoalphazero.config import get_hex_config
from nanoalphazero.research.muzero.cli import resolve


@pytest.mark.parametrize("size, expected_updates", [(7, 150000), (8, 500000), (9, 2000000)])
def test_default_cycle_budget_uses_production_selfplay_step_semantics(size, expected_updates):
    original = get_hex_config(size)
    config = resolve({"env": f"hex{size}", "network": "spatial", "wandb": True})
    assert config["cycles"] == original["num_iters"] // original["cycle_n_selfplay"]
    assert config["cycles"] * config["updates_per_cycle"] == expected_updates
    assert config["train_batch_size"] == original["train_batch_size"]
    assert config["learning_rate"] == original["learning_rate"]


def test_explicit_pilot_budget_is_preserved():
    config = resolve({"env": "hex8", "network": "spatial", "cycles": 50, "wandb": True})
    assert config["cycles"] * config["updates_per_cycle"] == 2500
