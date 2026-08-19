# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
# ---

# %% [markdown]
# # nanoAlphaZero — custom-game Colab
#
# This notebook is a quick demonstration of nanoAlphaZero's game-agnostic logic.
#
# Suppose you want to train AlphaZero on a completely new game like Tic-Tac-Toe but 4x4 (3 in a row wins).
#
# Here's what you do
#
# 1. Create a PGX-styled env (ask an LLM)
# 2. Paste it below
# 3. Run the cells
#

# %% [markdown]
# ## Install

# %%
# Preserve Colab's JAX/libtpu pair, install runtime dependencies, then
# install the split package from the exact implementation commit used here.
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import jax

PACKAGE_REF = "93aed2b4d11c946385c5e0e9afd19407f249a75f"
print("JAX:", jax.__version__)
print("Devices:", jax.devices())
if not jax.devices() or jax.devices()[0].platform != "tpu":
    raise RuntimeError("No TPU found. Select a TPU runtime and restart the session.")

constraints = Path("/tmp/colab-jax-constraints.txt")
constraints.write_text(
    f"jax=={version('jax')}\n"
    f"jaxlib=={version('jaxlib')}\n"
)

runtime_dependencies = [
    "pgx1 @ git+https://github.com/wtedw/pgx1.git@fa313c84338d93ab96fc02bc7c658364bf43098f",
    "mctx @ git+https://github.com/wtedw/mctx.git@6cf1a39",
    "flashbax @ git+https://github.com/instadeepai/flashbax.git@e0199d7bb232c622a19d3c28f9d6b34eb8215eab",
    "flax==0.10.1",
    "optax==0.2.7",
    "chex==0.1.91",
    "safetensors==0.8.0",
    "wandb==0.21.0",
]
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--constraint", str(constraints), *runtime_dependencies],
    check=True,
)
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--no-deps",
        f"nanoalphazero @ git+https://github.com/wtedw/nanoAlphaZero.git@{PACKAGE_REF}",
    ],
    check=True,
)
print("Installed nanoAlphaZero package ref:", PACKAGE_REF[:8])


# %% [markdown]
# ## Imports
#

# %%
import time

import jax
import jax.numpy as jnp
import numpy as np

import nanoalphazero.core as az_core
import nanoalphazero.play as az_play
from nanoalphazero.checkpoint import load_checkpoint, save_checkpoint
from nanoalphazero.config import get_ttt_config
from nanoalphazero.core import make_alphazero


# %% [markdown]
# ## Custom game env
#

# %% [markdown]
# All envs follow the API design from [Pgx](https://github.com/sotetsuk/pgx/).
#
# Roll your own env or ask some LLM to adapt an existing one. We adapt the TicTacToe environment from my personal [pgx1 repo](https://github.com/wtedw/pgx1/blob/main/pgx1/tic_tac_toe.py)

# %%
# Copyright 2023 The Pgx Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generalized N x N Tic-Tac-Toe used as the custom-environment example."""

import dataclasses
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
from jax import Array, lax

TRUE = jnp.bool_(True)
FALSE = jnp.bool_(False)


class GameState(NamedTuple):
    color: Array
    board: Array
    winner: Array


def _make_winning_lines(n: int, k: int) -> Array:
    """Return shape [num_lines, k] containing flattened board indices."""
    lines = []
    for r in range(n):
        for c in range(n - k + 1):
            lines.append([r * n + c + i for i in range(k)])
    for r in range(n - k + 1):
        for c in range(n):
            lines.append([(r + i) * n + c for i in range(k)])
    for r in range(n - k + 1):
        for c in range(n - k + 1):
            lines.append([(r + i) * n + c + i for i in range(k)])
    for r in range(n - k + 1):
        for c in range(k - 1, n):
            lines.append([(r + i) * n + c - i for i in range(k)])
    return jnp.asarray(lines, dtype=jnp.int32)


class Game:
    def __init__(self, n: int, k: int = 3):
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if k > n:
            raise ValueError(f"k ({k}) cannot be larger than n ({n})")
        self.n = n
        self.k = k
        self._winning_lines = _make_winning_lines(n, k)

    def init(self) -> GameState:
        return GameState(
            color=jnp.int32(0),
            board=-jnp.ones(self.n * self.n, dtype=jnp.int32),
            winner=jnp.int32(-1),
        )

    def step(self, state: GameState, action: Array) -> GameState:
        board = state.board.at[action].set(state.color)
        won = (board[self._winning_lines] == state.color).all(axis=1).any()
        winner = lax.select(won, state.color, jnp.int32(-1))
        return state._replace(
            board=board,
            color=(state.color + 1) % 2,
            winner=winner,
        )

    def observe(self, state: GameState, color: Optional[Array] = None) -> Array:
        if color is None:
            color = state.color
        grid = state.board.reshape((self.n, self.n))
        return jnp.stack(
            [
                grid == color,
                grid == (1 - color),
                jnp.full((self.n, self.n), color, dtype=jnp.bool_),
                jnp.ones((self.n, self.n), dtype=jnp.bool_),
            ],
            axis=-1,
        )

    def legal_action_mask(self, state: GameState) -> Array:
        return state.board < 0

    def is_terminal(self, state: GameState) -> Array:
        return (state.winner >= 0) | jnp.all(state.board != -1)

    def rewards(self, state: GameState) -> Array:
        return lax.select(
            state.winner >= 0,
            jnp.float32([-1, -1]).at[state.winner].set(1),
            jnp.zeros(2, dtype=jnp.float32),
        )


def _field(factory):
    return dataclasses.field(default_factory=factory)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class State:
    current_player: Array = _field(lambda: jnp.int32(0))
    observation: Array = _field(
        lambda: jnp.zeros((1, 1, 4), dtype=jnp.bool_)
    )
    rewards: Array = _field(lambda: jnp.zeros(2, dtype=jnp.float32))
    terminated: Array = _field(lambda: FALSE)
    truncated: Array = _field(lambda: FALSE)
    legal_action_mask: Array = _field(lambda: jnp.ones(1, dtype=jnp.bool_))
    _step_count: Array = _field(lambda: jnp.int32(0))
    _x: GameState = _field(
        lambda: GameState(
            color=jnp.int32(0),
            board=-jnp.ones(1, dtype=jnp.int32),
            winner=jnp.int32(-1),
        )
    )

    def replace(self, **kwargs) -> "State":
        return dataclasses.replace(self, **kwargs)

    @property
    def env_id(self) -> str:
        return "custom_tic_tac_toe"


class TicTacToeGeneral:
    """N x N Tic-Tac-Toe with k consecutive pieces required to win."""

    def __init__(self, n: int = 3, k: int = 3):
        self.n = n
        self.k = k
        self._game = Game(n=n, k=k)

    def init(self, key: Optional[Array] = None) -> State:
        del key
        x = self._game.init()
        return State(
            current_player=jnp.int32(0),
            observation=self._game.observe(x),
            legal_action_mask=self._game.legal_action_mask(x),
            _x=x,
        )

    def step(
        self,
        state: State,
        action: Array,
        key: Optional[Array] = None,
    ) -> State:
        del key
        is_illegal = ~self._check_legality(state, action)
        current_player = state.current_player
        state = lax.cond(
            state.terminated | state.truncated,
            lambda: state.replace(rewards=jnp.zeros_like(state.rewards)),
            lambda: self._step(
                state.replace(_step_count=state._step_count + 1), action
            ),
        )
        state = lax.cond(
            is_illegal,
            lambda: self._step_with_illegal_action(state, current_player),
            lambda: state,
        )
        return lax.cond(
            state.terminated,
            lambda: state.replace(
                legal_action_mask=jnp.ones_like(state.legal_action_mask)
            ),
            lambda: state,
        )

    def observe(
        self,
        state: State,
        player_id: Optional[Array] = None,
    ) -> Array:
        if player_id is None:
            player_id = state.current_player
        curr_color = state._x.color
        my_color = lax.select(
            player_id == state.current_player,
            curr_color,
            1 - curr_color,
        )
        return lax.stop_gradient(self._game.observe(state._x, my_color))

    def _step(self, state: State, action: Array) -> State:
        x = self._game.step(state._x, action)
        state = state.replace(
            current_player=(state.current_player + 1) % 2,
            _x=x,
        )
        terminated = self._game.is_terminal(x)
        rewards = self._game.rewards(x)
        rewards = lax.select(
            state.current_player != x.color,
            jnp.flip(rewards),
            rewards,
        )
        rewards = lax.select(
            terminated,
            rewards,
            jnp.zeros(2, dtype=jnp.float32),
        )
        return state.replace(
            observation=self.observe(state, state.current_player),
            legal_action_mask=self._game.legal_action_mask(x),
            rewards=rewards,
            terminated=terminated,
        )

    def _check_legality(self, state: State, action: Array) -> Array:
        mask_i32 = state.legal_action_mask.astype(jnp.int32)
        one_hot_a = jax.nn.one_hot(
            action,
            mask_i32.shape[0],
            dtype=jnp.int32,
        )
        return jnp.dot(one_hot_a, mask_i32).astype(jnp.bool_)

    def _step_with_illegal_action(self, state: State, loser: Array) -> State:
        rewards = jnp.where(
            jnp.arange(2) == loser,
            -1.0,
            1.0,
        ).astype(jnp.float32)
        return state.replace(rewards=rewards, terminated=TRUE)

    @property
    def id(self) -> str:
        return "custom_tic_tac_toe"

    @property
    def version(self) -> str:
        return "v0"

    @property
    def num_players(self) -> int:
        return 2

    @property
    def num_actions(self) -> int:
        return self.n * self.n


# %% [markdown]
# ## Configure the custom game
#
# nanoAlphaZero accepts any compatible PGX-style environment through `custom_env=`.
#

# %%
BOARD_SIZE = 4
WIN_LENGTH = 3
CUSTOM_ENV_ID = f"custom_ttt_{BOARD_SIZE}x{BOARD_SIZE}_k{WIN_LENGTH}"
CUSTOM_ENV = TicTacToeGeneral(n=BOARD_SIZE, k=WIN_LENGTH)


def custom_config():
    n, k = BOARD_SIZE, WIN_LENGTH
    if n < 4 or not 1 <= k <= n:
        raise ValueError("Require BOARD_SIZE >= 4 and 1 <= WIN_LENGTH <= BOARD_SIZE")
    game_max_steps = n * n
    root_actions = min(16, game_max_steps)
    survivors = max(1, root_actions // 2)
    # Use the basic ttt config
    config = get_ttt_config()
    batch_size = config["selfplay_batch_size"]
    selfplay_buffer_len = game_max_steps + 10
    replay_buffer_len = config["replay_buffer_total_size"] // batch_size
    warmup_steps = selfplay_buffer_len + replay_buffer_len
    config.update(
        env_id=CUSTOM_ENV_ID,
        game_name=CUSTOM_ENV_ID,
        boardsize=n,
        game_max_steps=game_max_steps,
        game_obs_shape=None,
        game_num_actions=None,
        num_iters=2500,
        num_exploratory_moves=max(1, game_max_steps // 2),
        mcts_num_simulations=root_actions + survivors,  # this field is unused at the moment
        mcts_max_m=root_actions,
        mcts_num_root_considered=8,
        mcts_num_survivors=4,
        mcts_num_k_actions=game_max_steps,
        lr_warmup_steps=warmup_steps,
        replay_buffer_warmup_steps=warmup_steps,
        selfplay_buffer_min_len=game_max_steps,
        selfplay_buffer_max_len=game_max_steps,
        enable_wandb=False,
    )
    return config


CONFIG = custom_config()


# %% [markdown]
# ## Smoke test the env
#

# %%
wenv = az_core.make_env(CONFIG, custom_env=CUSTOM_ENV)
keys = jax.random.split(jax.random.PRNGKey(0), 2)
state = wenv.init(keys)
observation = wenv.observe(state, state.current_player)
actions = jnp.argmax(state.legal_action_mask, axis=1).astype(jnp.int32)
step_keys = jax.random.split(jax.random.PRNGKey(1), 2)
next_state = wenv.autostep(state, actions, step_keys)

assert observation.shape == (2, BOARD_SIZE, BOARD_SIZE, 4)
assert state.legal_action_mask.shape == (2, BOARD_SIZE * BOARD_SIZE)
assert next_state.rewards.shape == (2, 2)
print("Custom package environment OK:", wenv)


# %% [markdown]
# ## Empty-board diagnostics
#
# This reports the model's P1-to-move value, W/D/L probabilities, and N×N policy
# logits. A confident forced P1 win should approach value `+1` and win probability
# `1`. The training loop calls this periodically without rebuilding the model.
#

# %%
def print_initial_evaluation(az, runner_state):
    state = az.env.init_dummy_estate(batch_size=1)
    obs = az.env.observe(state, state.current_player)
    logits, value, wdl_logits = runner_state.model_ts.apply_fn(
        {"params": runner_state.model_ts.params},
        obs,
        state.legal_action_mask,
        deterministic=True,
        return_wdl_logits=True,
    )
    board_size = az.config["boardsize"]
    logits = np.asarray(logits[0]).reshape((board_size, board_size))
    wdl = np.asarray(jax.nn.softmax(wdl_logits[0]))
    print(
        f"P1 initial value={float(value[0]):+.3f} | "
        f"win={wdl[0]:.3f} draw={wdl[2]:.3f} loss={wdl[1]:.3f}"
    )
    print("Policy logits:")
    for row in logits:
        print("  " + "  ".join(f"{x:+.2f}" for x in row))


# %% [markdown]
# ## AlphaZero training loop
#
# `az.run_fn` is just one compiled cycle containing the three expensive phases:
#
# 1. self-play,
# 2. drain (move completed-games into replay buffer),
# 3. train
#

# %%
SAVE_PATH = f"artifacts/alphazero_{CUSTOM_ENV_ID}.safetensors"
MAX_TRAINING_CYCLES = None  # e.g. 5 for a short trial
DIAGNOSTIC_PERIOD = CONFIG.get("diagnostic_period", 100)

rng = jax.random.PRNGKey(42)
rng, build_key = jax.random.split(rng)
az = make_alphazero(CONFIG, build_key, custom_env=CUSTOM_ENV)
runner_state = az.runner_state

# 1. Fill replay memory while the optimizer is frozen.
warmup_cycles = (
    az.config["replay_buffer_warmup_steps"] // az.config["cycle_n_selfplay"]
)
print(f"Warmup: {warmup_cycles} cycles")
print("(this will take some time as JAX JIT compiles the first call)")
for cycle in range(warmup_cycles):
    run_fn_started = time.time()
    runner_state, _ = az.run_fn(runner_state, jnp.array(True))
    runner_state.model_ts.step.block_until_ready()
    run_fn_seconds = time.time() - run_fn_started
    print(
        f"  warmup {cycle + 1}/{warmup_cycles} | "
        f"run_fn={run_fn_seconds:.2f}s"
    )

# 2. Alternate self-play, replay drain, and gradient training.
training_cycles = az.config["num_iters"] // az.config["cycle_n_selfplay"]
if MAX_TRAINING_CYCLES is not None:
    training_cycles = min(training_cycles, MAX_TRAINING_CYCLES)

print(
    f"Training: {training_cycles} cycles "
    f"({az.config['num_iters']} total iterations)"
)
for cycle in range(1, training_cycles + 1):
    run_fn_started = time.time()
    runner_state, (scalar_metrics, _) = az.run_fn(
        runner_state, jnp.array(False)
    )
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), runner_state)
    run_fn_seconds = time.time() - run_fn_started
    last = jax.tree_util.tree_map(lambda x: float(x[-1]), scalar_metrics)
    print(
        f"cycle {cycle}/{training_cycles} | run_fn={run_fn_seconds:.2f}s | "
        f"loss={last['total_loss']:.4f} "
        f"value={last['loss_v']:.4f} policy={last['loss_pi']:.4f}"
    )

    if cycle == 1 or (
        DIAGNOSTIC_PERIOD and cycle % DIAGNOSTIC_PERIOD == 0
    ):
        print_initial_evaluation(az, runner_state)

save_checkpoint(runner_state.model_ts.params, az.config, SAVE_PATH)
print("Saved:", SAVE_PATH)


# %% [markdown]
# How do we know our model did well?
#
# For 3x3 k=3 games (Tic-Tac-Toe), the game-theoretic result for P1 is a draw, but for 4x4 k=3, P1 has a guaranteed win. There are two signs our model understands this
# 1. our model views P1 board with value=+1.000
# 2. its logits are concentrated in the center (optimal strategy).
#
# We can now pat ourselves on the back and call it a day

# %% [markdown]
# ## Optional: play against the model
#
# Pass the custom environment again when constructing an interactive play setup.
#

# %%
# az_play.play_against_model(
#     CONFIG,
#     runner_state.model_ts.params,
#     custom_env=CUSTOM_ENV,
#     human_player=0,
#     num_simulations=None,
# )

# In a fresh session, recreate CUSTOM_ENV, then load and play:
# params, model_config = load_checkpoint(SAVE_PATH)
# az_play.play_against_model(
#     CONFIG, params, custom_env=CUSTOM_ENV, human_player=0
# )
