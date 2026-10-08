"""Shared batched matches and forced openings for AlphaZero and MuZero."""

import jax
import jax.numpy as jnp
import numpy as np

from nanoalphazero.buffers import unpack_bitmask_vmap


def _legal_mask_from_state(env_state, config):
    if config["env_id"] == "chess":
        return unpack_bitmask_vmap(env_state.legal_action_bitmask)
    return env_state.legal_action_mask


def _random_legal_actions(env_state, config, key):
    """Uniform random over legal actions (batched)."""
    legal = _legal_mask_from_state(env_state, config)
    logits = jnp.where(legal, 0.0, -jnp.inf)
    g = jax.random.gumbel(key, logits.shape, dtype=logits.dtype)
    return jnp.argmax(logits + g, axis=-1).astype(jnp.int32)


def all_opening_actions(wenv, config, plies=1):
    """Every legal opening LINE of length `plies` from the initial position.

    Returns a 2-D int array of shape (num_openings, plies); row i is one forced
    move sequence. plies=1 -> one row per legal first move (ttt->9, connect4->7,
    hex4x4->16, chess->20). plies=2 -> every legal (move1, move2) pair (chess
    ->~400), which probes a wider, more balanced slice of the opening tree and
    so gives a lower-variance ladder score. Lines that terminate before `plies`
    moves are dropped (can't host a full game), keeping the result rectangular.
    """
    env_state = wenv.init_dummy_estate(batch_size=1)
    legal = np.asarray(_legal_mask_from_state(env_state, config)[0])
    lines = [[int(a)] for a in np.nonzero(legal)[0]]
    for _ in range(int(plies) - 1):
        n = len(lines)
        env_state = wenv.init_dummy_estate(batch_size=n)
        depth = len(lines[0])
        for d in range(depth):  # replay the line built so far, batched
            col = jnp.asarray([line[d] for line in lines], dtype=jnp.int32)
            env_state = wenv.step(env_state, col)
        legal_masks = np.asarray(_legal_mask_from_state(env_state, config))
        terminated = np.asarray(env_state.terminated)
        next_lines = []
        for i, line in enumerate(lines):
            if terminated[i]:
                continue  # game over before reaching `plies` moves -> drop
            for a in np.nonzero(legal_masks[i])[0]:
                next_lines.append(line + [int(a)])
        lines = next_lines
    return np.asarray(lines, dtype=np.int32)


def run_eval_match(
    run_mcts_fn,
    wenv,
    config,
    params_p0,
    params_p1,
    *,
    opening_actions,
    key,
    max_plies=None,
    num_simulations=None,
    gumbel_scale=0.0,
):
    """Play one game per forced opening, fully batched.

    `opening_actions` is an array of forced opening lines: shape (N,) for a single
    forced move each, or (N, K) for a K-move forced line. One game is played per
    row. Seat 0 plays with `params_p0`, seat 1 with `params_p1`. A param set of
    `None` means uniform-random legal play. After the forced opening moves, both
    sides play greedy MCTS. Returns (p0_wins, draws, p1_wins) as ints; games that
    don't finish within `max_plies` count as draws.
    """
    if max_plies is None:
        max_plies = config.get("eval_max_plies") or config.get("game_max_steps") or 512
    if num_simulations is None:
        num_simulations = config["mcts_num_simulations"]

    opening_actions = np.asarray(opening_actions, dtype=np.int32)
    if opening_actions.ndim == 1:  # (N,) -> (N, 1): one forced move per game
        opening_actions = opening_actions[:, None]
    num_real = len(opening_actions)

    # Pad the batch up to a multiple of the device count so data-parallel
    # sharding is happy; padded games duplicate real openings and are dropped.
    mult = jax.device_count() if config.get("enable_sharding", False) else 1
    pad = (-num_real) % mult
    batch_actions = (
        np.concatenate([opening_actions, opening_actions[:pad]])
        if pad
        else opening_actions
    )
    num_games = len(batch_actions)

    env_state = wenv.init_dummy_estate(batch_size=num_games)
    for d in range(batch_actions.shape[1]):  # play each forced opening ply
        env_state = wenv.step(env_state, jnp.asarray(batch_actions[:, d]))

    final_rewards = jnp.zeros((num_games, 2), dtype=jnp.float32)
    finished = env_state.terminated

    for _ply in range(int(max_plies)):
        newly = env_state.terminated & ~finished
        final_rewards = jnp.where(newly[:, None], env_state.rewards, final_rewards)
        finished = finished | env_state.terminated
        if bool(jnp.all(finished)):
            break

        # Live games are in lockstep (player alternates each ply), so one seat
        # moves for the whole batch this ply.
        cp = jnp.where(~finished, env_state.current_player, -1)
        seat = int(jnp.max(cp))
        params = params_p0 if seat == 0 else params_p1

        key, mk = jax.random.split(key)
        if params is None:
            actions = _random_legal_actions(env_state, config, mk)
        else:
            actions = run_mcts_fn(
                mk, env_state, params, gumbel_scale, num_games, num_simulations
            ).action
        env_state = wenv.step(env_state, actions)

    # Credit any game that terminated on the very last step.
    newly = env_state.terminated & ~finished
    final_rewards = jnp.where(newly[:, None], env_state.rewards, final_rewards)

    r0 = np.asarray(final_rewards[:num_real, 0])  # drop padded games
    p0_wins = int(np.sum(r0 > 0))
    p1_wins = int(np.sum(r0 < 0))
    draws = num_real - p0_wins - p1_wins
    return p0_wins, draws, p1_wins


def evaluate_vs(
    run_mcts_fn,
    wenv,
    config,
    cur_params,
    opp_params,
    *,
    openings,
    key,
    label,
    num_simulations=None,
):
    """Play `cur_params` vs `opp_params` over every opening, from BOTH colors.
    Prints a W/D/L line + score + ΔElo estimate, and returns the score in [0,1]."""
    k1, k2 = jax.random.split(key)
    # current as seat 0
    w0, d0, l0 = run_eval_match(
        run_mcts_fn,
        wenv,
        config,
        cur_params,
        opp_params,
        opening_actions=openings,
        key=k1,
        num_simulations=num_simulations,
    )
    # current as seat 1 (swap): current's wins are the seat-1 wins
    l1, d1, w1 = run_eval_match(
        run_mcts_fn,
        wenv,
        config,
        opp_params,
        cur_params,
        opening_actions=openings,
        key=k2,
        num_simulations=num_simulations,
    )
    wins, draws, losses = w0 + w1, d0 + d1, l0 + l1
    n = max(wins + draws + losses, 1)
    score = (wins + 0.5 * draws) / n
    s = min(max(score, 1e-4), 1 - 1e-4)
    elo = -400.0 * np.log10(1.0 / s - 1.0)
    print(
        f"  [eval] vs {label:<14} {wins:>3}W {draws:>3}D {losses:>3}L "
        f"| score={score:.3f} | ΔElo≈{elo:+.0f}",
        flush=True,
    )
    return score, elo

