"""Explicit experimental continuation must retain state without weakening resume."""
from collections import namedtuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nanoalphazero import ckpt
from nanoalphazero.cli import parse_muzero_args


def test_partial_fork_retains_optimizer_rng_cycle_and_new_buffers(tmp_path):
    State = namedtuple("State", "model_ts rng buffer")
    old = State({"params": jnp.arange(4.), "momentum": jnp.ones(4),
                 "step": jnp.int32(72000)}, jax.random.PRNGKey(123), jnp.ones(8))
    initial = State(jax.tree.map(jnp.zeros_like, old.model_ts),
                    jax.random.PRNGKey(0), jnp.zeros(8))
    source = tmp_path / "source"
    config = dict(mcts_num_simulations=32, mcts_num_root_considered=16,
                  muzero_train_on_terminal_states=True, conv_width=384)
    ckpt.save_checkpoint(source, state=old, cycle=3600, config=config, mode="partial")
    changed = dict(config, mcts_num_simulations=64, mcts_num_root_considered=32,
                   muzero_train_on_terminal_states=False)
    with pytest.raises(ValueError, match="identical"):
        ckpt.restore_checkpoint(source, state=initial, config=changed, mode="partial")
    restored, cycle = ckpt.restore_checkpoint(source, state=initial, config=changed,
                                              mode="partial", fork=True)
    assert cycle == 3600
    for a, b in zip(jax.tree.leaves((old.model_ts, old.rng)),
                    jax.tree.leaves((restored.model_ts, restored.rng))):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(restored.buffer, initial.buffer)
    assert ckpt.inspect_checkpoint(source)["config"] == config
    for key, value in [("conv_width", 256), ("learning_rate", .001),
                       ("muzero_terminal_policy", "none")]:
        with pytest.raises(ValueError, match="identical"):
            ckpt.validate_resume_config(source, {**changed, key: value}, "partial", fork=True)
    with pytest.raises(ValueError, match="partial checkpoints"):
        ckpt.validate_resume_config(source, changed, "full", fork=True)


@pytest.mark.parametrize("extra", [
    ["--ckpt.resume", "old"], ["--trackio-run", "existing"], ["--ckpt.save", "old"],
])
def test_fork_cannot_overwrite_or_reuse_run(extra):
    with pytest.raises(ValueError):
        parse_muzero_args(["--ckpt.fork", "old", *extra])


def test_unresolved_dimensions_checked_after_assembly(monkeypatch):
    from nanoalphazero.ckpt import inspection
    saved = dict(game_obs_shape=[8, 8, 119], game_num_actions=4672)
    monkeypatch.setattr(inspection, "inspect_checkpoint",
                        lambda _: dict(mode="partial", config=saved))
    inspection.validate_resume_config("source", dict(game_obs_shape=None,
                                                       game_num_actions=None), "partial")
    with pytest.raises(ValueError, match="identical"):
        inspection.validate_resume_config("source", {**saved, "game_num_actions": 9}, "partial")
