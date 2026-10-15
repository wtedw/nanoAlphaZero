"""Batched PGX self-play with host-local PGN/FEN recording.

Retains the original global batch, host partition, and per-ply multihost gather
structure. Multihost validation covers 16 and 496 global games on four TPU hosts.
"""

from dataclasses import dataclass

import chess
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils

from nanoalphazero.eval.chess.action import PgxAction


@dataclass
class RecordedGame:
    index: int
    board: object
    fens: list
    opening_plies: int
    terminated: bool = False


def collect_games(env, policy_fn, params, openings, max_plies, data_sharding=None):
    """Return local games and placement metadata; never auto-reset finished lanes."""
    host, hosts = jax.process_index(), jax.process_count()
    real_count = len(openings)
    devices = data_sharding.mesh.size if data_sharding is not None else 1
    multiple = max(devices, hosts)
    padded_count = ((real_count + multiple - 1) // multiple) * multiple
    # Padding supports small single-host smoke suites; padded games are never scored.
    actions = np.asarray(openings, dtype=np.int32)
    actions = actions[np.arange(padded_count) % real_count]
    local_count = padded_count // hosts
    offset = host * local_count
    games = [RecordedGame(i, chess.Board(), [chess.Board().fen()], actions.shape[1])
             for i in range(offset, min(offset + local_count, real_count))]
    active = [True] * len(games)

    rng, env_rng = jax.random.split(jax.random.PRNGKey(42))
    state = env.init(jax.random.split(env_rng, padded_count))
    if data_sharding is not None:
        state = jax.device_put(state, data_sharding)

    def record(played, terminated):
        for local, game in enumerate(games):
            if not active[local]:
                continue
            move = PgxAction.decode(int(played[game.index]), game.board)
            if move not in game.board.legal_moves:
                raise ValueError(f"PGX/python-chess action mismatch in game {game.index}: {move}")
            game.board.push(move)
            game.fens.append(game.board.fen())
            game.terminated = bool(terminated[game.index])
            active[local] = not game.terminated

    def host_array(value):
        return np.asarray(multihost_utils.process_allgather(value, tiled=True) if hosts > 1 else value)

    for ply in range(actions.shape[1]):
        played = jnp.asarray(actions[:, ply])
        state = env.step(state, played)
        record(actions[:, ply], host_array(state.terminated))
    jax.block_until_ready(state)
    placement = {
        "global_games": real_count, "padded_games": padded_count,
        "hosts": hosts, "host": host,
        "addressable_shards": len(state.current_player.addressable_shards),
        "devices": sorted(str(shard.device) for shard in state.current_player.addressable_shards),
    }
    for ply in range(actions.shape[1], max_plies):
        if bool(jnp.all(state.terminated)):
            break
        rng, key = jax.random.split(rng)
        played = policy_fn(key, state, params, 0., padded_count).action
        state = env.step(state, played)
        terminated, played_host = host_array(state.terminated), host_array(played)
        if hosts > 1:
            multihost_utils.sync_global_devices(f"acpl_pre_record_{ply}")
        record(played_host, terminated)
        if hosts > 1:
            multihost_utils.sync_global_devices(f"acpl_post_record_{ply}")
    return games, placement
