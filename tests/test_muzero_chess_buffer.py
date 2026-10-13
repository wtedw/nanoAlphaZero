"""Real chess stepping with controlled search outputs and an independent target oracle."""
import importlib.util
from pathlib import Path

import chess
import jax
import jax.numpy as jnp
import numpy as np
import pgx1
import pgx1.chess as pc
import pytest

from nanoalphazero.buffers import (
    gather_muzero_training_samples, get_dummy_muzero_selfplay_output,
    make_muzero_selfplay_buffer, split_observation_vmap,
)
from nanoalphazero.config import get_muzero_config
from nanoalphazero.core import make_env
from nanoalphazero.eval.chess.action import PgxAction
from nanoalphazero.mcts import PolicyOutput


@pytest.fixture(scope="module")
def probe():
    path = Path(__file__).resolve().parents[1] / "scripts/probe_muzero_chess_terminals.py"
    spec = importlib.util.spec_from_file_location("chess_terminal_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("sparse,exploit_only,explore_policy,capacity", [
    (False, True, False, 11),
    (True, True, False, 11),
    (True, True, True, 11),
    (True, False, False, 11),
    (True, True, False, 7),
])
def test_chess_targets_through_reset_wrap_and_consume(probe, sparse, exploit_only, explore_policy, capacity):
    b, k = 9, 5
    cfg = get_muzero_config("chess", enable_sharding=False, selfplay_batch_size=b,
        train_batch_size=b, selfplay_buffer_consume_size=b,
        selfplay_buffer_max_len=capacity, selfplay_buffer_min_len=1,
        replay_buffer_max_len=2, muzero_unroll_steps=k, muzero_predict_value_only=False,
        exp_bnk_action_weights=sparse, mcts_num_k_actions=8,
        num_exploratory_moves=3, muzero_exploration_losses=(
            "policy-transition" if explore_policy else "none" if exploit_only else "all"))
    wrapped = make_env(cfg)
    cfg.update(game_obs_shape=list(wrapped.obs_shape), game_num_actions=4672)
    env = pgx1.make("chess", use_bitmask=True)
    boards = [chess.Board() for _ in range(b)]
    states = [env.init() for _ in range(b)]
    specials = {
        3: ("k7/2Q5/2K5/8/8/8/8/8 w - - 0 1", "c7b7"),
        4: ("k7/2Q5/2K5/8/8/8/8/8 w - - 0 1", "c7b6"),
        5: ("7k/8/8/8/8/8/Q7/K7 w - - 99 80", "a2a3"),
        6: ("7k/8/8/8/8/8/1r6/K7 w - - 0 1", "a1b2"),
        7: (chess.Board("k7/2Q5/2K5/8/8/8/8/8 w - - 0 1").mirror().fen(), "c2b2"),
    }
    for lane, (fen, _) in specials.items():
        boards[lane] = chess.Board(fen)
        states[lane] = probe.from_board(env, boards[lane])
    states[8] = states[8].replace(_x=states[8]._x._replace(step_count=jnp.int32(511)), _step_count=jnp.int32(511))
    plans = ["f2f3 e7e5 g2g4 d8h4".split(),
             "e2e4 e7e5 f1c4 b8c6 d1h5 g8f6 h5f7".split(),
             "g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1 f6g8".split()]

    def search(key, state, params, scale, batch_size):
        actions, confidence = params
        legal = jax.vmap(pc.unpack_bitmask)(state.legal_action_bitmask)
        selected = jax.nn.one_hot(actions, 4672)
        others = legal & ~selected.astype(bool)
        weights = selected * confidence[:, None] + others * (
            (1-confidence) / jnp.maximum(others.sum(-1), 1))[:, None]
        weights = weights / weights.sum(-1, keepdims=True)
        sparse_weights, indices = jax.lax.top_k(weights, 8)
        return PolicyOutput(action=actions, action_weights=weights, visit_counts=selected,
            bnk_k_indices=indices.astype(jnp.uint16),
            bnk_action_weights=sparse_weights / sparse_weights.sum(-1, keepdims=True))

    sp, selfplay = probe.make_muzero_selfplay(cfg, wrapped, search)
    selfplay = selfplay._replace(env_state=jax.tree.map(lambda *x: jnp.stack(x), *states))
    sp = jax.jit(sp)
    buffer, storage = make_muzero_selfplay_buffer(cfg, get_dummy_muzero_selfplay_output(cfg))
    add = jax.jit(buffer.add_backfill)
    consume = jax.jit(buffer.consume)
    @jax.jit
    def gather_all(experience):
        starts = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:])[:, None], experience)
        return gather_muzero_training_samples(experience, starts, k, 4672,
            self_consistency=True, exploration_losses=(
                "policy-transition" if explore_policy else "none" if exploit_only else "all"))
    records, episodes, moves = {}, np.zeros(b, int), np.zeros(b, int)
    finished_count = checked = consumed = 0

    def check_samples(samples, oracle_keys):
        nonlocal checked
        samples = jax.device_get(samples)
        for i, (lane, step) in enumerate(oracle_keys):
            record = records[lane, step]
            remaining = record["end"] - step
            for offset in range(k+1):
                real = offset <= remaining
                source = records[lane, step+offset] if real else None
                explore = source["explore"] if real else False
                value_allowed = not explore or not (exploit_only or explore_policy)
                policy_allowed = explore_policy or value_allowed
                np.testing.assert_allclose(samples.reward[i, offset], source["return"] if real else 0, atol=1e-6)
                assert samples.policy_mask[i, offset] == (real and policy_allowed)
                assert samples.value_mask[i, offset] == (not real or value_allowed)
                np.testing.assert_allclose(samples.action_weights[i, offset], source["policy"] if real else 0, atol=1e-6)
                if sparse:
                    np.testing.assert_array_equal(samples.k_indices[i, offset], source["indices"] if real else 0)
                if samples.board_bool.shape[1] > 1 or offset == 0:
                    np.testing.assert_array_equal(samples.board_bool[i, offset], source["board_bool"] if real else 0)
                    np.testing.assert_array_equal(samples.board_float[i, offset], source["board_float"] if real else 0)
                if offset < k:
                    if real:
                        assert samples.action[i, offset] == source["action"]
                    np.testing.assert_allclose(samples.transition_reward[i, offset], source["immediate"] if real else 0)
                    assert samples.reward_mask[i, offset] == (not real or policy_allowed)
            checked += 1

    for t in range(25):
        actions, expected_done, immediate = [], [], []
        for lane, board in enumerate(boards):
            special = t == 0 and lane in specials
            uci = specials[lane][1] if special else ("e2e4" if t == 0 and lane == 8 else plans[lane % 3][moves[lane]])
            move = chess.Move.from_uci(uci)
            assert board.is_valid() and move in board.legal_moves
            actions.append(PgxAction.encode(move, board))
        actions = jnp.asarray(actions, jnp.int32)
        confidence = jnp.full(b, .55 + .01*t)
        expected_policy = jax.device_get(search(None, selfplay.env_state, (actions, confidence), 0, b))
        obs = wrapped.observe(selfplay.env_state, selfplay.env_state.current_player)
        packed, floats = jax.device_get(split_observation_vmap(obs))
        for lane, board in enumerate(boards):
            actor = board.turn
            board.push(PgxAction.decode(int(actions[lane]), board))
            done = (board.is_checkmate() or board.is_stalemate() or board.is_insufficient_material()
                    or board.is_repetition(3) or board.halfmove_clock >= 100 or (lane == 8 and t == 0))
            reward = float(board.is_checkmate())
            expected_done.append(done); immediate.append(reward)
            records[lane, t] = dict(action=int(actions[lane]), actor=actor, episode=episodes[lane],
                explore=moves[lane] < 2 and not (t == 0 and lane >= 3),
                immediate=reward, board_bool=packed[lane], board_float=floats[lane],
                policy=expected_policy.bnk_action_weights[lane] if sparse else expected_policy.action_weights[lane],
                indices=expected_policy.bnk_k_indices[lane])
            if done:
                for (r_lane, r_step), record in records.items():
                    if r_lane == lane and record["episode"] == episodes[lane]:
                        record.update(end=t, **{"return": reward * (1 if record["actor"] == actor else -1)})
                boards[lane] = chess.Board()
                moves[lane] = 0; episodes[lane] += 1; finished_count += 1
            else:
                moves[lane] += 1
        selfplay = selfplay._replace(switch_step=jnp.where((t == 0) & (jnp.arange(b) >= 3), 0, 2))
        selfplay, output, _ = sp(jax.random.PRNGKey(t), selfplay, (actions, confidence))
        np.testing.assert_array_equal(output.just_terminated[:, 0], expected_done)
        np.testing.assert_allclose(output.transition_reward[:, 0], immediate)
        storage, _ = add(storage, output, selfplay.env_state.terminated, selfplay.env_state.rewards)
        exp = jax.device_get(storage.experience)
        lane_ids, cols = np.nonzero(exp.is_valid_sample)
        if len(cols):
            samples, missing = gather_all(storage.experience)
            assert int(missing) == 0
            samples = jax.tree.map(lambda x: x[lane_ids * capacity + cols], jax.device_get(samples))
            check_samples(samples, [(int(lane), int(exp.global_step_id[lane, col])) for lane, col in zip(lane_ids, cols)])
        if int(storage.num_valid_consumable) >= b:
            storage, batch, _ = consume(storage)
            batch = jax.tree.map(lambda x: x[:, 0], batch)
            keys = list(zip(np.asarray(batch.row_id).astype(int)-1, np.asarray(batch.global_step_id).astype(int)))
            check_samples(batch, keys)
            consumed += b
    assert finished_count >= 30 and checked > 100 and consumed > 50
    assert int(storage.missing_count) == 0
    if capacity == 7:
        assert int(storage.dropped_pending_count) > 0
    print(f"chess oracle: games={finished_count}, checked_windows={checked}, consumed={consumed}, sparse={sparse}, capacity={capacity}, pending_evictions={int(storage.dropped_pending_count)}")
