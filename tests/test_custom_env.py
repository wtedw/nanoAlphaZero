import dataclasses

import jax
import jax.numpy as jnp
import pytest

from nanoalphazero import core
from nanoalphazero.config import get_ttt_config
from nanoalphazero.play import make_play


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class TinyState:
    current_player: jax.Array
    board: jax.Array
    rewards: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    legal_action_mask: jax.Array
    step_count: jax.Array

    def replace(self, **kwargs):
        return dataclasses.replace(self, **kwargs)


class TinyEnv:
    num_actions = 4
    num_players = 2

    def init(self, key=None):
        del key
        return TinyState(
            current_player=jnp.int32(0),
            board=jnp.zeros((2, 2), dtype=jnp.int8),
            rewards=jnp.zeros((2,), dtype=jnp.float32),
            terminated=jnp.bool_(False),
            truncated=jnp.bool_(False),
            legal_action_mask=jnp.ones((4,), dtype=jnp.bool_),
            step_count=jnp.int32(0),
        )

    def step(self, state, action, key=None):
        del key
        piece = (state.current_player + 1).astype(state.board.dtype)
        board = state.board.at[action // 2, action % 2].set(piece)
        legal = state.legal_action_mask.at[action].set(False)
        step_count = state.step_count + 1
        terminated = step_count == 2
        rewards = jnp.where(
            terminated,
            jnp.array([1.0, -1.0], dtype=jnp.float32),
            jnp.zeros((2,), dtype=jnp.float32),
        )
        return state.replace(
            current_player=1 - state.current_player,
            board=board,
            rewards=rewards,
            terminated=terminated,
            legal_action_mask=legal,
            step_count=step_count,
        )

    def observe(self, state, player_id=None):
        if player_id is None:
            player_id = state.current_player
        mine = state.board == (player_id + 1)
        theirs = state.board == (2 - player_id)
        return jnp.stack((mine, theirs), axis=-1)


def test_make_env_wraps_injected_env_without_using_pgx_registry(monkeypatch):
    def unexpected_make(*args, **kwargs):
        raise AssertionError("pgx1.make should not resolve an injected environment")

    monkeypatch.setattr(core.pgx1, "make", unexpected_make)
    wrapped = core.make_env({"env_id": "tiny"}, custom_env=TinyEnv())

    state = wrapped.init(jax.random.split(jax.random.PRNGKey(0), 2))
    observation = wrapped.observe(state, state.current_player)
    next_state = wrapped.step(state, jnp.array([0, 1], dtype=jnp.int32))
    auto_state = wrapped.autostep(
        state,
        jnp.array([0, 1], dtype=jnp.int32),
        jax.random.split(jax.random.PRNGKey(1), 2),
    )

    assert wrapped.obs_shape == (2, 2, 2)
    assert wrapped.num_actions == 4
    assert observation.shape == (2, 2, 2, 2)
    assert next_state.rewards.shape == (2, 2)
    assert auto_state.rewards.shape == (2, 2)


def test_make_env_returns_an_injected_wrapped_env_unchanged():
    wrapped = core.make_env({"env_id": "tiny"}, custom_env=TinyEnv())

    assert core.make_env({"env_id": "tiny"}, custom_env=wrapped) is wrapped


def test_custom_env_validation_reports_missing_state_contract():
    class InvalidEnv(TinyEnv):
        def init(self, key=None):
            del key
            return object()

    with pytest.raises(TypeError, match="current_player.*legal_action_mask"):
        core.make_env({"env_id": "invalid"}, custom_env=InvalidEnv())


def test_make_play_accepts_custom_env_and_derives_dimensions():
    config = get_ttt_config()
    config.update(
        env_id="tiny",
        katago_preset="b1c8nbt",
        conv_depth=1,
        conv_width=8,
        mcts_max_m=4,
        mcts_num_root_considered=4,
        mcts_num_survivors=2,
        mcts_num_k_actions=4,
    )

    wrapped, _, _, _, resolved = make_play(config, custom_env=TinyEnv())

    assert wrapped.obs_shape == (2, 2, 2)
    assert resolved["game_obs_shape"] == (2, 2, 2)
    assert resolved["game_num_actions"] == 4
