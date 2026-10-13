"""CPU-only terminal-head, target, and 1sh-backup audit. Never loads buffers."""

import argparse
import json
import os
from pathlib import Path
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chess
import chess.pgn
import jax
import jax.numpy as jnp
import numpy as np
import pgx1
import pgx1.chess as pc

from nanoalphazero.buffers import get_dummy_muzero_selfplay_output, make_muzero_selfplay_buffer, gather_muzero_training_samples
from nanoalphazero.ckpt import load_model
from nanoalphazero.core import make_env
from nanoalphazero.eval.chess.action import PgxAction
from nanoalphazero.mcts import PolicyOutput, RootFnOutput, RecurrentFnOutput, gumbel_muzero_policy_1sh, make_muzero_mcts
from muzero import make_muzero_selfplay


def from_board(env, board):
    """FEN probes have only current-board history; replay probes retain all history."""
    pieces = np.zeros(64, np.int8)
    for sq, piece in board.piece_map().items():
        rank = chess.square_rank(sq) if board.turn else 7 - chess.square_rank(sq)
        pieces[chess.square_file(sq) * 8 + rank] = piece.piece_type * (1 if piece.color == board.turn else -1)
    rights = jnp.asarray([[board.has_queenside_castling_rights(c), board.has_kingside_castling_rights(c)]
                          for c in (board.turn, not board.turn)])
    assert board.ep_square is None
    x = pc.GameState()._replace(
        board=jnp.asarray(pieces), color=jnp.int16(not board.turn), castling_rights=rights,
        halfmove_count=jnp.int16(board.halfmove_clock), fullmove_count=jnp.int16(board.fullmove_number),
        board_history=jnp.zeros((8, 64), jnp.int8).at[0].set(pieces),
        hash_history=jnp.zeros_like(pc.GameState().hash_history),
    )
    x = x._replace(hash_history=x.hash_history.at[0].set(pc._zobrist_hash(x)))
    legal = pc._legal_action_result(x.board.astype(jnp.int32), x.en_passant.astype(jnp.int32), rights)
    outcome = env._game._outcome(x, legal.mask.any(), legal.in_check)
    return env.init().replace(_x=x, current_player=x.color.astype(jnp.int32),
        terminated=outcome.terminated, rewards=outcome.rewards,
        legal_action_bitmask=pc.pack_mask(legal.mask), observation=env._game.observe(x, x.color))


def cases(env):
    step = jax.jit(env.step)
    examples = []

    def add(name, board, state, move, expected):
        move = chess.Move.from_uci(move)
        assert board.is_valid() and move in board.legal_moves, (name, board.fen(), move)
        legal = np.flatnonzero(np.asarray(pc.unpack_bitmask(state.legal_action_bitmask)))
        assert set(legal) == {PgxAction.encode(m, board) for m in board.legal_moves}, name
        action = PgxAction.encode(move, board)
        after = step(state, jnp.int32(action))
        final = board.copy(); final.push(move)
        assert bool(after.terminated) == expected, (name, final.fen(), bool(after.terminated))
        examples.append((name, board.fen(), state, action, after, str(final.outcome(claim_draw=True))))

    for name, moves, final, terminal in [
        ("fools_mate_black", "f2f3 e7e5 g2g4", "d8h4", True),
        ("scholars_mate_white", "e2e4 e7e5 f1c4 b8c6 d1h5 g8f6", "h5f7", True),
        ("threefold_repetition", "g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1", "f6g8", True),
        ("opening_e4", "", "e2e4", False),
        ("ordinary_capture", "e2e4 d7d5", "e4d5", False),
    ]:
        board, state = chess.Board(), env.init()
        for uci in moves.split():
            move = chess.Move.from_uci(uci)
            state = step(state, jnp.int32(PgxAction.encode(move, board)))
            board.push(move)
        add(name, board, state, final, terminal)
    for name, fen, move in [
        ("queen_mate", "k7/2Q5/2K5/8/8/8/8/8 w - - 0 1", "c7b7"),
        ("stalemate", "k7/2Q5/2K5/8/8/8/8/8 w - - 0 1", "c7b6"),
        ("fifty_move_rule", "7k/8/8/8/8/8/Q7/K7 w - - 99 80", "a2a3"),
        ("capture_to_bare_kings", "7k/8/8/8/8/8/1r6/K7 w - - 0 1", "a1b2"),
    ]:
        board = chess.Board(fen)
        add(name, board, from_board(env, board), move, True)
        mirrored = board.mirror()
        m = chess.Move.from_uci(move)
        add(name + "_mirrored", mirrored, from_board(env, mirrored),
            chess.Move(chess.square_mirror(m.from_square), chess.square_mirror(m.to_square)).uci(), True)
    board, state = chess.Board(), env.init()
    state = state.replace(_x=state._x._replace(step_count=jnp.int32(511)), _step_count=jnp.int32(511))
    add("512_ply_cap", board, state, "e2e4", True)
    # The earlier cycle-8400 missed mate, replayed with its actual board history.
    audit = Path("artifacts/chess-pgn-audit-cycle8400/report.json")
    if audit.exists():
        source = Path(json.loads(audit.read_text())["summary"]["source"])
        with (source / "game-000221.pgn").open() as stream:
            game = chess.pgn.read_game(stream)
        board, state = game.board(), env.init()
        for move in list(game.mainline_moves())[:47]:
            state = step(state, jnp.int32(PgxAction.encode(move, board)))
            board.push(move)
        add("previous_missed_Qh1_mate", board, state, board.parse_san("Qh1#").uci(), True)
    return examples


def oracle():
    results = {}
    for name, reward, bad_value in [("win_absorbing", 1., False), ("draw_absorbing", 0., False),
                                     ("win_double_counted", 1., True)]:
        def recurrent(params, key, action, depth):
            r = jnp.where(depth == 0, reward, 0.)
            v = jnp.where((depth % 2) == 0, -1., 1.) if bad_value else jnp.zeros_like(r)
            return RecurrentFnOutput(reward=r, discount=-jnp.ones_like(r),
                                     prior_logits=jnp.zeros((depth.shape[0], 1)), value=v), depth + 1
        result = gumbel_muzero_policy_1sh(None, jax.random.PRNGKey(0),
            RootFnOutput(prior_logits=jnp.zeros((1, 1)), value=jnp.zeros(1), embedding=jnp.zeros(1, jnp.int32)),
            recurrent, num_root_considered=1, num_survivors=1, gumbel_scale=0., return_diagnostics=True)
        q = float(result.diagnostics["backed_up_q"][0, 0])
        np.testing.assert_allclose(q, reward + float(bad_value), atol=2e-6)
        results[name] = q
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    generation_path = args.checkpoint / "LATEST"
    before = generation_path.read_text() if args.checkpoint.is_dir() else None
    model, params, saved_config = load_model(args.checkpoint)
    assert before is None or generation_path.read_text() == before, "Checkpoint changed during model loading; retry"
    print("Loaded model on CPU", flush=True)
    env = pgx1.make("chess", use_bitmask=True)
    examples = cases(env)
    roots = jax.tree.map(lambda *x: jnp.stack(x), *[e[2] for e in examples])
    actual = jax.tree.map(lambda *x: jnp.stack(x), *[e[4] for e in examples])
    actions = jnp.asarray([e[3] for e in examples], jnp.int32)
    cfg = {**saved_config, "enable_sharding": False, "selfplay_batch_size": len(examples),
           "selfplay_buffer_add_batch_size": len(examples),
           "selfplay_buffer_consume_size": len(examples), "selfplay_buffer_max_len": 8,
           "selfplay_buffer_min_len": 1, "num_exploratory_moves": 0}
    wrapped = make_env(cfg)
    observation = wrapped.observe(roots, roots.current_player)
    final_observation = wrapped.observe(actual, actual.current_player)

    @jax.jit
    def predict(params):
        latent, _, value = model.apply({"params": params}, observation, method=model.initial)
        _, _, observed_value = model.apply({"params": params}, final_observation, method=model.initial)
        latent, r1, logits, v1 = model.apply({"params": params}, latent, actions, method=model.recurrent)
        # 1sh selects its second action by the successor's largest prior.
        _, r2, _, v2 = model.apply({"params": params}, latent, jnp.argmax(logits, axis=-1),
                                    method=model.recurrent)
        tails = []
        for k in range(4):
            # Match absorbing training: arbitrary actions after termination.
            arbitrary = jax.random.randint(jax.random.PRNGKey(k), actions.shape, 0, 4672)
            latent, r, _, v = model.apply({"params": params}, latent, arbitrary, method=model.recurrent)
            tails.append(jnp.stack([r, v], axis=-1))
        return value, observed_value, r1, v1, jnp.stack(tails, axis=1), jnp.stack([r2, v2], axis=-1)
    predictions = jax.device_get(predict(params))
    print("Predicted terminal heads", flush=True)
    assert all(np.isfinite(x).all() for x in predictions)

    def fixed_search(key, env_state, params, scale, batch_size):
        legal = jax.vmap(pc.unpack_bitmask)(env_state.legal_action_bitmask)
        chosen = jnp.where(legal[jnp.arange(batch_size), actions], actions, jnp.argmax(legal, axis=-1))
        weights = jax.nn.one_hot(chosen, 4672)
        sparse_weights, indices = jax.lax.top_k(weights, 128)
        return PolicyOutput(action=chosen, action_weights=weights, visit_counts=weights,
                            bnk_k_indices=indices.astype(jnp.uint16), bnk_action_weights=sparse_weights)
    sp_fn, sp_state = make_muzero_selfplay(cfg, wrapped, fixed_search)
    sp_state = sp_state._replace(env_state=roots, switch_step=jnp.zeros_like(sp_state.switch_step))
    following, output, _ = jax.jit(sp_fn)(jax.random.PRNGKey(7), sp_state, params)
    buffer, state = make_muzero_selfplay_buffer(cfg, get_dummy_muzero_selfplay_output(cfg))
    state, _ = jax.jit(buffer.add_backfill)(state, output, following.env_state.terminated, following.env_state.rewards)
    starts = jax.tree.map(lambda x: x[:, :1], state.experience)
    value_only = saved_config.get("muzero_predict_value_only", False)
    from nanoalphazero.muzero_config import normalize_exploration_config
    mode = normalize_exploration_config(saved_config)["muzero_exploration_losses"]
    targets, missing = gather_muzero_training_samples(
        state.experience, starts, 5, 4672, value_only=value_only, exploration_losses=mode)
    assert int(missing) == 0
    terminal = np.asarray(actual.terminated)
    actual_rewards = actual.rewards[jnp.arange(len(examples)), roots.current_player]
    np.testing.assert_array_equal(targets.reward[terminal, 0], actual_rewards[terminal])
    np.testing.assert_array_equal(targets.transition_reward[terminal, 0], actual_rewards[terminal])
    np.testing.assert_array_equal(output.discount[:, 0][terminal], 0)
    expected_tail = (np.asarray(actual_rewards)[terminal, None] * (-1.)**np.arange(1, 6)
                     if value_only else np.zeros((int(terminal.sum()), 5)))
    np.testing.assert_array_equal(targets.reward[terminal, 1:], expected_tail)
    np.testing.assert_array_equal(targets.value_mask[terminal, 1:], True)
    np.testing.assert_array_equal(targets.transition_reward[terminal, 1:], 0)
    np.testing.assert_array_equal(targets.reward_mask[terminal, 1:], not value_only)
    print("Verified actual self-play and absorbing targets", flush=True)
    # auto_reset immediately supplies a new initial board while preserving the
    # finished game's reward/done flags. Subsequent steps play that new game.
    np.testing.assert_array_equal(following.env_state._x.board[terminal],
        np.broadcast_to(np.asarray(env.init()._x.board), (int(terminal.sum()), 64)))
    for step_index in range(6):
        following, next_output, _ = jax.jit(sp_fn)(jax.random.PRNGKey(20 + step_index), following, params)
        state, _ = jax.jit(buffer.add_backfill)(state, next_output,
            following.env_state.terminated, following.env_state.rewards)
    after_reset, missing_after = gather_muzero_training_samples(
        state.experience, starts, 5, 4672, self_consistency=True,
        value_only=value_only, exploration_losses=mode)
    assert int(missing_after) == 0
    for field in ("reward", "transition_reward", "action", "policy_mask", "value_mask", "reward_mask"):
        np.testing.assert_array_equal(np.asarray(getattr(after_reset, field))[terminal],
                                      np.asarray(getattr(targets, field))[terminal])
    np.testing.assert_array_equal(after_reset.board_bool[terminal, 1:], 0)
    np.testing.assert_array_equal(after_reset.board_float[terminal, 1:], 0)
    print("Verified six autosteps into new games do not leak into terminal windows", flush=True)
    search = make_muzero_mcts(cfg, wrapped, model, return_diagnostics=True)
    results = []
    for scale in (0., 1.):
        results.append(jax.device_get(search(jax.random.PRNGKey(0), roots, params, scale, len(examples))))
    rows = []
    for i, (name, fen, root, action, after, outcome) in enumerate(examples):
        reward = float(after.rewards[root.current_player])
        row = dict(case=name, fen=fen, action=int(action), terminal=bool(after.terminated), python_outcome=outcome,
                   actual_reward=reward, root_value=float(predictions[0][i]),
                   observed_successor_value=float(predictions[1][i]), predicted_reward=float(predictions[2][i]),
                   latent_successor_value=float(predictions[3][i]), q_one_step=float(predictions[2][i]-predictions[3][i]),
                   arbitrary_tail_reward_value=predictions[4][i].tolist(),
                   greedy_tail_reward_value=predictions[5][i].tolist(),
                   q_two_step=float(predictions[2][i]-predictions[5][i, 0]+predictions[5][i, 1]),
                   value_targets=np.asarray(targets.reward[i]).tolist(), value_mask=np.asarray(targets.value_mask[i]).tolist(),
                   reward_targets=np.asarray(targets.transition_reward[i]).tolist())
        for scale, result in zip((0, 1), results):
            visits = int(result.visit_counts[i, action])
            expected_q = (row["q_one_step"] + row["q_two_step"]) / (2 + 1e-6) if visits == 2 else row["q_one_step"]
            if visits:
                np.testing.assert_allclose(result.diagnostics["backed_up_q"][i, action], expected_q, atol=2e-5)
            row[f"search_gumbel_{scale}"] = dict(
                visits=float(result.visit_counts[i, action]), chosen=int(result.action[i]) == action,
                q=float(result.diagnostics["backed_up_q"][i, action]),
                weight=float(result.action_weights[i, action]))
        rows.append(row)
    report = dict(checkpoint=str(args.checkpoint), generation=before, platform=jax.default_backend(),
                  checks={"terminal_targets": "passed", "six_autosteps_no_cross_game_leak": "passed",
                          "searched_action_backups_match_manual_arithmetic": "passed"},
                  config=saved_config, oracle=oracle(), rows=rows,
                  caveats=["FEN cases have no earlier observation history; replay cases retain it.",
                           "Observed terminal root values are not directly supervised like imagined absorbing values.",
                           "Tail actions are arbitrary; nonterminal control tails are not expected to be zero.",
                           "Unvisited search action Q is a placeholder; inspect visits before interpreting Q."])
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"cases":len(rows), "oracle":report["oracle"], "report":str(args.output / "report.json")}), flush=True)


if __name__ == "__main__":
    main()
