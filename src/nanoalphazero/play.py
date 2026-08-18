"""Interactive local play and single-game inference helpers."""

import re
import time

import jax
import jax.numpy as jnp
import numpy as np

from nanoalphazero.buffers import unpack_bitmask_vmap
from nanoalphazero.core import make_env
from nanoalphazero.mcts import make_mcts
from nanoalphazero.model import make_model


def make_play(config):
    """Build a single-game (batch=1) inference setup: env, model, MCTS fn.

    Sharding is disabled so a batch of one plays cleanly on a single device.
    """
    config = config.copy()
    config["enable_sharding"] = False

    wenv = make_env(config)
    config["game_obs_shape"] = wenv.obs_shape
    config["game_num_actions"] = wenv.num_actions

    rng = jax.random.PRNGKey(0)
    model, model_state = make_model(config, rng, sharding=None)

    play_cfg = config.copy()
    play_cfg["selfplay_batch_size"] = 1
    play_cfg["mcts_bnk_rehydrate_fields"] = True
    run_mcts_fn = make_mcts(play_cfg, wenv, model)

    return wenv, model, model_state, run_mcts_fn, config


def _p0p1_board(env_state, wenv):
    """Fixed-perspective board: +1 = player 0's stones, -1 = player 1's."""
    obs = np.asarray(wenv.observe(env_state, env_state.current_player))[0].astype(
        np.int8
    )
    board = obs[:, :, 0] - obs[:, :, 1]
    if int(env_state.current_player[0]) == 1:
        board = -board
    return board


def _sym(v, empty="."):
    return "X" if v > 0 else ("O" if v < 0 else empty)


def _print_board(env_state, wenv, env_id):
    board = _p0p1_board(env_state, wenv)
    H, W = board.shape
    print()
    if env_id == "connect_four":
        for r in range(H):
            print("   " + "  ".join(_sym(board[r, c]) for c in range(W)))
        print("   " + "  ".join(str(c + 1) for c in range(W)))
    elif env_id.startswith("hex"):
        # Columns are letters (a, b, ...), rows are 1-indexed numbers.
        # Each row is shifted half a cell to the right to form the rhombus.
        col_letters = " ".join(chr(ord("a") + c) for c in range(W))
        print("     " + col_letters)
        for r in range(H):
            cells = " ".join(_sym(board[r, c]) for c in range(W))
            print(f"  {r + 1:>2} " + " " * r + cells)
    elif env_id.startswith("go"):
        # Columns are letters (a, b, ...), rows are 1-indexed numbers.
        col_letters = " ".join(chr(ord("a") + c) for c in range(W))
        print("     " + col_letters)
        for r in range(H):
            cells = " ".join(_sym(board[r, c]) for c in range(W))
            print(f"  {r + 1:>2} " + cells)
    else:  # tic_tac_toe and other square grids
        for r in range(H):
            print(
                "   "
                + " ".join(
                    _sym(board[r, c], empty=str(r * W + c + 1)) for c in range(W)
                )
            )
    print()


def _parse_move(raw: str, env_id: str, H: int, W: int, legal):
    """Parse a human move string into a legal action index, or None."""
    raw = raw.strip().lower()
    try:
        if env_id == "connect_four":
            # input is a 1-indexed column number
            action = int(raw) - 1
        elif env_id.startswith("go"):
            if raw in ("pass", "p"):
                action = W * W  # pass is the last action index (size**2)
            else:
                m = re.fullmatch(r"([a-z])\s*(\d+)", raw)
                if m:  # algebraic, e.g. "a1" (column letter + row number)
                    col = ord(m.group(1)) - ord("a")
                    row = int(m.group(2)) - 1
                    action = row * W + col
                else:
                    parts = raw.split()
                    if len(parts) == 2:  # "row col", 1-indexed
                        action = (int(parts[0]) - 1) * W + (int(parts[1]) - 1)
                    else:  # single 1-indexed cell number
                        action = int(raw) - 1
        elif env_id.startswith("hex"):
            m = re.fullmatch(r"([a-z])\s*(\d+)", raw)
            if m:  # algebraic, e.g. "a1" / "c3" (column letter + row number)
                col = ord(m.group(1)) - ord("a")
                row = int(m.group(2)) - 1
                action = row * W + col
            else:
                parts = raw.split()
                if len(parts) == 2:  # "row col", 1-indexed
                    action = (int(parts[0]) - 1) * W + (int(parts[1]) - 1)
                else:  # single 1-indexed cell number
                    action = int(raw) - 1
        else:
            parts = raw.split()
            if len(parts) == 2:  # "row col", 1-indexed
                action = (int(parts[0]) - 1) * W + (int(parts[1]) - 1)
            else:  # single 1-indexed cell number
                action = int(raw) - 1
    except (ValueError, TypeError):
        return None
    n_actions = legal.shape[0]
    if 0 <= action < n_actions and bool(legal[action]):
        return action
    return None


def _action_to_str(action: int, env_id: str, W: int) -> str:
    """Human-readable description of a model's chosen action."""
    if env_id == "connect_four":
        return f"column {action + 1}"
    if env_id.startswith("go") and action == W * W:
        return "pass"
    row, col = action // W, action % W
    if env_id.startswith("hex"):
        return f"{chr(ord('a') + col)}{row + 1}"
    if env_id.startswith("go"):
        return f"{chr(ord('a') + col)}{row + 1}"
    return f"cell {action + 1} (row {row + 1}, col {col + 1})"


def _print_model_eval(model, params, wenv, state, env_id, W):
    """Print the raw network output (policy logits/probs + value head) for the
    current position, from the side-to-move's (i.e. the model's) point of view.
    """
    obs = wenv.observe(state, state.current_player)
    legal = state.legal_action_mask
    logits, value = model.apply({"params": params}, obs, legal)
    logits = np.asarray(logits[0], dtype=np.float32)
    probs = np.asarray(jax.nn.softmax(jnp.asarray(logits)), dtype=np.float32)
    value = float(np.asarray(value[0]))
    legal_np = np.asarray(legal[0])

    print(f"  value head: {value:+.3f}  (current player's perspective)")
    print("  policy (legal actions, by probability):")
    legal_idxs = sorted(np.where(legal_np)[0], key=lambda i: -probs[i])
    for i in legal_idxs:
        bar = "#" * int(round(probs[i] * 20))
        print(
            f"    {_action_to_str(int(i), env_id, W):>22}  "
            f"logit={logits[i]:+7.3f}  p={probs[i]:5.3f}  {bar}"
        )


def play_against_model(config, params=None, *, human_player=0, num_simulations=None):
    """Play an interactive terminal game against the trained model.

    `human_player` is 0 (you move first) or 1 (model moves first).
    Commands during your turn: `undo`, `restart`, `quit`.
    """
    env_id = config["env_id"]
    if env_id == "chess":
        print("Playing against chess model currently not supported.")
        return

    wenv, model, model_state, run_mcts_fn, config = make_play(config)
    H, W = wenv.obs_shape[0], wenv.obs_shape[1]

    if params is None:
        params = model_state["params"]
        print("⚠️  No params provided — playing against an UNTRAINED model.")
    params = jax.device_put(params)

    if num_simulations is None:
        num_simulations = config["mcts_num_simulations"]

    human_sym = "X" if human_player == 0 else "O"
    print("\n" + "=" * 50)
    print(f"Playing {env_id}.  You are '{human_sym}' (player {human_player + 1}).")
    if env_id == "connect_four":
        print("Enter a column number (1-7) to drop your piece.")
    elif env_id.startswith("hex"):
        print("Enter a cell as a column letter + row number, e.g. 'a1' or 'c3'.")
    elif env_id.startswith("go"):
        print("Enter a cell as a column letter + row number, e.g. 'a1'; or 'pass'.")
    else:
        print("Enter the number shown in an empty cell (or 'row col').")
    print("Commands: undo, restart, quit")
    print("=" * 50)
    rng = jax.random.PRNGKey(int(time.time()))
    rng, env_rng = jax.random.split(rng)
    initial_state = wenv.init(jax.random.split(env_rng, 1))
    state = initial_state
    history = [initial_state]

    while not bool(state.terminated[0]):
        _print_board(state, wenv, env_id)
        current_player = int(state.current_player[0])

        if current_player == human_player:
            _print_model_eval(model, params, wenv, state, env_id, W)
            legal = np.asarray(state.legal_action_mask[0])
            while True:
                raw = input(f"Your move ({human_sym}): ").strip().lower()
                if raw in ("quit", "q", "exit"):
                    print("Bye.")
                    return
                if raw == "undo":
                    # step back to the human's previous turn (drop human + model plies)
                    if len(history) >= 3:
                        history = history[:-2]
                    elif len(history) >= 2:
                        history = history[:-1]
                    else:
                        print("Nothing to undo.")
                    state = history[-1]
                    break
                if raw == "restart":
                    state = initial_state
                    history = [initial_state]
                    print("Game restarted.")
                    break
                action = _parse_move(raw, env_id, H, W, legal)
                if action is None:
                    print("  Illegal or unparseable move, try again.")
                    continue
                state = wenv.step(state, jnp.array([action], dtype=jnp.int32))
                history.append(state)
                break
        else:
            print("Model is thinking...")
            _print_model_eval(model, params, wenv, state, env_id, W)
            rng, mcts_rng = jax.random.split(rng)
            policy_output = run_mcts_fn(
                mcts_rng,
                state,
                params,
                gumbel_scale=0.0,
                batch_size=1,
                num_simulations=num_simulations,
            )
            action = int(policy_output.action[0])
            print(f"Model plays {_action_to_str(action, env_id, W)}.")
            state = wenv.step(state, jnp.array([action], dtype=jnp.int32))
            history.append(state)

    # --- Game over ---
    _print_board(state, wenv, env_id)
    rewards = np.asarray(state.rewards[0])
    print("=" * 50)
    print("--- GAME OVER ---")
    human_reward = rewards[human_player]
    if human_reward > 0:
        print("You win! 🎉")
    elif human_reward < 0:
        print("Model wins. 🤖")
    else:
        print("Draw.")
    print(f"Rewards [P1, P2] = {rewards.tolist()}")
    print("=" * 50)


def play_both(config, params=None):
    """Interactive terminal game where you enter moves for BOTH players.

    You control player 1 and player 2 yourself; the model never moves, but its
    value head and policy are shown for each position.
    Commands during a turn: `undo`, `restart`, `quit`.
    """
    env_id = config["env_id"]
    if env_id == "chess":
        print("Playing both sides for chess is currently not supported.")
        return

    wenv, model, model_state, run_mcts_fn, config = make_play(config)
    H, W = wenv.obs_shape[0], wenv.obs_shape[1]

    if params is None:
        params = model_state["params"]
        print("⚠️  No params provided — showing eval from an UNTRAINED model.")
    params = jax.device_put(params)

    print("\n" + "=" * 50)
    print(f"Playing {env_id}. You control both players.")
    print("Player 1 is 'X', player 2 is 'O'.")
    if env_id == "connect_four":
        print("Enter a column number (1-7) to drop your piece.")
    elif env_id.startswith("hex"):
        print("Enter a cell as a column letter + row number, e.g. 'a1' or 'c3'.")
    elif env_id.startswith("go"):
        print("Enter a cell as a column letter + row number, e.g. 'a1'; or 'pass'.")
    else:
        print("Enter the number shown in an empty cell (or 'row col').")
    print("Commands: undo, restart, quit")
    print("=" * 50)

    rng = jax.random.PRNGKey(int(time.time()))
    rng, env_rng = jax.random.split(rng)
    initial_state = wenv.init(jax.random.split(env_rng, 1))
    state = initial_state
    history = [initial_state]

    while not bool(state.terminated[0]):
        _print_board(state, wenv, env_id)
        current_player = int(state.current_player[0])
        sym = "X" if current_player == 0 else "O"
        _print_model_eval(model, params, wenv, state, env_id, W)
        legal = np.asarray(state.legal_action_mask[0])
        while True:
            raw = input(f"Player {current_player + 1} ({sym}) move: ").strip().lower()
            if raw in ("quit", "q", "exit"):
                print("Bye.")
                return
            if raw == "undo":
                # each ply is one human move, so step back a single state
                if len(history) >= 2:
                    history = history[:-1]
                else:
                    print("Nothing to undo.")
                state = history[-1]
                break
            if raw == "restart":
                state = initial_state
                history = [initial_state]
                print("Game restarted.")
                break
            action = _parse_move(raw, env_id, H, W, legal)
            if action is None:
                print("  Illegal or unparseable move, try again.")
                continue
            state = wenv.step(state, jnp.array([action], dtype=jnp.int32))
            history.append(state)
            break

    # --- Game over ---
    _print_board(state, wenv, env_id)
    rewards = np.asarray(state.rewards[0])
    print("=" * 50)
    print("--- GAME OVER ---")
    if rewards[0] > 0:
        print("Player 1 (X) wins! 🎉")
    elif rewards[1] > 0:
        print("Player 2 (O) wins! 🎉")
    else:
        print("Draw.")
    print(f"Rewards [P1, P2] = {rewards.tolist()}")
    print("=" * 50)

