"""MuZero-specific CLI settings; omitted options inherit game defaults."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class MuZeroConfig:
    network: Literal["vector", "spatial"] | None = None
    chess_action_encoding: Literal["from_to8", "planes73"] | None = None
    unroll_steps: int | None = None
    # No reward head or reward loss; retain outcome values beyond termination.
    predict_value_only: bool = True
    # Losses at real exploratory positions; terminal padding is unaffected.
    exploration_losses: Literal["none", "policy-transition", "all"] = "policy-transition"
    # Retain the final board as a completed root; reset on the following call.
    train_on_terminal_states: bool = False
    # Policy supervision on terminal and absorbing positions, over all actions.
    terminal_policy: Literal["none", "uniform"] = "none"
    memory_fraction: float | None = None
    remat_blocks: bool | None = None
    remat_unroll: bool | None = None

    def overrides(self):
        return {f"muzero_{name}": value for name, value in vars(self).items()
                if value is not None}


def normalize_exploration_config(config):
    """Translate saved experimental flags without changing their supervision."""
    config = dict(config)
    for old in ("exp_train_on_exploratory_unroll_states", "exp_train_on_exploration"):
        if old in config:
            config.setdefault("exp_train_on_exploit_unroll_states", not config.pop(old))
    exploit_only = config.pop("exp_train_on_exploit_unroll_states", True)
    without_value = config.pop("exp_train_on_exploration_without_value", False)
    config.setdefault("muzero_exploration_losses",
                      "policy-transition" if without_value else "none" if exploit_only else "all")
    return config
