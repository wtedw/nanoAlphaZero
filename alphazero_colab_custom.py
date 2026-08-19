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
# Suppose you want to train AlphaZero on a new game such as the 4,4,4 member of
# the M,N,K family: a 4×4 board where four in a row wins.
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

PACKAGE_REF = "e14e431079510334c1ad8c0543a4caf80cbd4417"
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
from nanoalphazero.config import get_hex_config
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

"""Generalized M,N,K game used as the custom-environment example."""

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


def _make_winning_lines(m: int, n: int, k: int) -> Array:
    """Return shape [num_lines, k] containing flattened board indices."""
    lines = []
    for r in range(m):
        for c in range(n - k + 1):
            lines.append([r * n + c + i for i in range(k)])
    for r in range(m - k + 1):
        for c in range(n):
            lines.append([(r + i) * n + c for i in range(k)])
    for r in range(m - k + 1):
        for c in range(n - k + 1):
            lines.append([(r + i) * n + c + i for i in range(k)])
    for r in range(m - k + 1):
        for c in range(k - 1, n):
            lines.append([(r + i) * n + c - i for i in range(k)])
    return jnp.asarray(lines, dtype=jnp.int32)


class Game:
    def __init__(self, m: int, n: int, k: int):
        if m < 1:
            raise ValueError(f"m must be >= 1, got {m}")
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if k > max(m, n):
            raise ValueError(
                f"k ({k}) cannot be larger than both m ({m}) and n ({n})"
            )
        self.m = m
        self.n = n
        self.k = k
        self._winning_lines = _make_winning_lines(m, n, k)

    def init(self) -> GameState:
        return GameState(
            color=jnp.int32(0),
            board=-jnp.ones(self.m * self.n, dtype=jnp.int32),
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
        grid = state.board.reshape((self.m, self.n))
        return jnp.stack(
            [
                grid == color,
                grid == (1 - color),
                jnp.full((self.m, self.n), color, dtype=jnp.bool_),
                jnp.ones((self.m, self.n), dtype=jnp.bool_),
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
        return "custom_mnk"


class MNKGame:
    """M rows by N columns with K consecutive pieces required to win."""

    def __init__(self, m: int, n: int, k: int):
        self.m = m
        self.n = n
        self.k = k
        self._game = Game(m=m, n=n, k=k)

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
        return f"mnk_{self.m}x{self.n}_k{self.k}"

    @property
    def max_steps(self) -> int:
        return self.m * self.n

    @property
    def allows_draws(self) -> bool:
        return True

    @property
    def version(self) -> str:
        return "v0"

    @property
    def num_players(self) -> int:
        return 2

    @property
    def num_actions(self) -> int:
        return self.m * self.n


# %% [markdown]
# ## Configure the custom game
#
# nanoAlphaZero accepts any compatible PGX-style environment through `custom_env=`.
# Choose a built-in config with an action space similar to your custom game. The
# config is a training preset; nanoAlphaZero replaces game-specific facts from the
# live environment. Hex 4×4 has 16 actions and Hex 5×5 has 25, so this 4×4 game
# uses Hex 4×4. A 5×5, k=4 Tic-Tac-Toe game should use the Hex 5×5 config.
#

# %%
M = 4  # rows
N = 4  # columns
K = 3  # consecutive marks needed to win
CUSTOM_ENV = MNKGame(m=M, n=N, k=K)
CUSTOM_ENV_ID = CUSTOM_ENV.id

# Pick a training preset, then override only intentional experiment choices.
CONFIG = get_hex_config(board_size=4)  # Hex 4×4 also has 16 actions.
CONFIG["num_iters"] = 1000  # We can probably train faster than Hex 4x4.


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

assert observation.shape == (2, M, N, 4)
assert state.legal_action_mask.shape == (2, M * N)
assert next_state.rewards.shape == (2, 2)
print("Custom package environment OK:", wenv)


# %% [markdown]
# ## Empty-board diagnostics
#
# This reports the model's side-to-move value, W/D/L probabilities, and M×N
# policy logits. The training loop calls it periodically without rebuilding the
# model.
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
    board_shape = az.env.obs_shape[:2]
    logits = np.asarray(logits[0]).reshape(board_shape)
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
# Watch the loss curves and periodic empty-board value/policy diagnostics above.
# For a rigorous strength measurement, evaluate checkpoints against a perfect
# M,N,K solver or a fixed baseline using both starting seats.

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
