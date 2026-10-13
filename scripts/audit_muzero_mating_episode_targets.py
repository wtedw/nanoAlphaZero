"""Replay full mating games through production self-play/backfill/sample gathering."""
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chess
import jax
import jax.numpy as jnp
import numpy as np
from muzero import make_muzero_selfplay
from nanoalphazero.buffers import get_dummy_muzero_selfplay_output, make_muzero_selfplay_buffer, gather_muzero_training_samples
from nanoalphazero.config import get_muzero_config
from nanoalphazero.core import make_env
from nanoalphazero.eval.chess.action import PgxAction
from nanoalphazero.mcts import PolicyOutput


def audit():
    cfg = get_muzero_config("chess", enable_sharding=False, selfplay_batch_size=2,
        selfplay_buffer_consume_size=2, train_batch_size=2, selfplay_buffer_max_len=16,
        selfplay_buffer_min_len=1, replay_buffer_max_len=2, muzero_unroll_steps=5,
        muzero_predict_value_only=True, muzero_exploration_losses="policy-transition", mcts_num_k_actions=8)
    env = make_env(cfg)
    cfg.update(game_obs_shape=list(env.obs_shape), game_num_actions=env.num_actions)
    def scripted_search(key, state, actions, scale, batch_size):
        weights = jax.nn.one_hot(actions, env.num_actions)
        sparse, indices = jax.lax.top_k(weights, 8)
        return PolicyOutput(action=actions, action_weights=weights, visit_counts=weights,
                            bnk_k_indices=indices.astype(jnp.uint16), bnk_action_weights=sparse)
    sp, initial = make_muzero_selfplay(cfg, env, scripted_search)
    sp = jax.jit(sp)
    dummy = get_dummy_muzero_selfplay_output(cfg)
    buffer, _ = make_muzero_selfplay_buffer(cfg, dummy)
    add = jax.jit(buffer.add_backfill)
    plans = ["f2f3 e7e5 g2g4 d8h4".split(), "e2e4 e7e5 f1c4 b8c6 d1h5 g8f6 h5f7".split()]
    reports = []
    for explore in (False, True):
        state, storage = initial, buffer.init(dummy)
        boards, records = [chess.Board(), chess.Board()], [[], []]
        for t in range(13):
            moves = [chess.Move.from_uci(p[t]) if t < len(p) else next(iter(boards[b].legal_moves))
                     for b, p in enumerate(plans)]
            actions = jnp.array([PgxAction.encode(m, b) for m, b in zip(moves, boards)], jnp.int32)
            state = state._replace(switch_step=jnp.full_like(state.switch_step, 100 if explore else 0))
            state, output, _ = sp(jax.random.PRNGKey(t), state, actions)
            for b, (move, board) in enumerate(zip(moves, boards)):
                board.push(move)
                if t < len(plans[b]):
                    assert bool(output.is_exploration[b, 0]) == explore
                    assert bool(output.just_terminated[b, 0]) == board.is_checkmate()
                    assert float(output.transition_reward[b, 0]) == float(board.is_checkmate())
                    records[b].append(dict(ply=t+1, move=move.uci(),
                        environment_reward=float(output.transition_reward[b, 0]),
                        discount=float(output.discount[b, 0])))
                if board.is_game_over():
                    boards[b] = chess.Board()
            storage, _ = add(storage, output, state.env_state.terminated, state.env_state.rewards)
        lanes = np.array([b for b, p in enumerate(plans) for _ in p])
        cols = np.array([t for p in plans for t in range(len(p))])
        starts = jax.tree.map(lambda x: x[lanes, cols][:, None], storage.experience)
        batch, missing = gather_muzero_training_samples(storage.experience, starts, 5,
            env.num_actions, value_only=True, exploration_losses="policy-transition")
        assert int(missing) == 0 and bool(jnp.all(starts.is_valid_sample))
        samples = [[], []]
        for i, (b, t) in enumerate(zip(lanes, cols)):
            length = len(plans[b])
            expected = np.array([(-1.)**(length-1-t-k) for k in range(6)])
            real = np.arange(6) < length-t
            np.testing.assert_array_equal(batch.reward[i], expected)
            np.testing.assert_array_equal(batch.policy_mask[i], real)
            np.testing.assert_array_equal(batch.value_mask[i], ~real if explore else np.ones(6, bool))
            np.testing.assert_array_equal(batch.reward_mask[i], False)
            stored = float(storage.experience.reward[b, t])
            assert stored == expected[0]
            records[b][t]["completed_return"] = stored
            samples[b].append(dict(root_ply=int(t)+1, values=batch.reward[i].tolist(),
                value_mask=batch.value_mask[i].tolist(), policy_mask=batch.policy_mask[i].tolist(),
                immediate_rewards=batch.transition_reward[i].tolist(), reward_mask=batch.reward_mask[i].tolist()))
        for b, name in enumerate(("fools_mate_black", "scholars_mate_white")):
            reports.append(dict(case=name, all_exploration=explore, records=records[b], samples=samples[b]))
    return dict(checks="passed", platform=jax.default_backend(), config=cfg, episodes=reports,
                note="Scripted legal actions replace search only; production self-play, buffer backfill and gathering are used.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({"checks": report["checks"], "episodes": len(report["episodes"]), "output": str(args.output)}))
