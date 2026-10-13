"""Fixed chess transitions for notebook and training diagnostics, without assets.

Replay cases retain observation history. Synthetic FEN cases have only the
current board in their history and should be interpreted separately.
"""
import chess
import jax
import jax.numpy as jnp
import numpy as np
import pgx1.chess as pc

from nanoalphazero.eval.chess.action import PgxAction


def from_board(env, board):
    pieces = np.zeros(64, np.int8)
    for sq, piece in board.piece_map().items():
        rank = chess.square_rank(sq) if board.turn else 7 - chess.square_rank(sq)
        pieces[chess.square_file(sq) * 8 + rank] = piece.piece_type * (1 if piece.color == board.turn else -1)
    rights = jnp.asarray([[board.has_queenside_castling_rights(c), board.has_kingside_castling_rights(c)]
                          for c in (board.turn, not board.turn)])
    assert board.ep_square is None
    state = pc.GameState()._replace(
        board=jnp.asarray(pieces), color=jnp.int16(not board.turn), castling_rights=rights,
        halfmove_count=jnp.int16(board.halfmove_clock), fullmove_count=jnp.int16(board.fullmove_number),
        board_history=jnp.zeros((8, 64), jnp.int8).at[0].set(pieces),
        hash_history=jnp.zeros_like(pc.GameState().hash_history))
    state = state._replace(hash_history=state.hash_history.at[0].set(pc._zobrist_hash(state)))
    legal = pc._legal_action_result(state.board.astype(jnp.int32), state.en_passant.astype(jnp.int32), rights)
    outcome = env._game._outcome(state, legal.mask.any(), legal.in_check)
    return env.init().replace(_x=state, current_player=state.color.astype(jnp.int32),
        terminated=outcome.terminated, rewards=outcome.rewards,
        legal_action_bitmask=pc.pack_mask(legal.mask), observation=env._game.observe(state, state.color))


def terminal_cases(env):
    """Return (name, fen, root, final_action, successor, outcome) tuples."""
    step = jax.jit(env.step)
    result = []

    def add(name, board, state, uci):
        move = chess.Move.from_uci(uci)
        assert move in board.legal_moves, name
        action = PgxAction.encode(move, board)
        after = step(state, jnp.int32(action))
        assert bool(after.terminated), name
        final = board.copy()
        final.push(move)
        result.append((name, board.fen(), state, action, after, str(final.outcome(claim_draw=True))))

    for name, moves, last in [
        ("fools_mate_black", "f2f3 e7e5 g2g4", "d8h4"),
        ("scholars_mate_white", "e2e4 e7e5 f1c4 b8c6 d1h5 g8f6", "h5f7"),
        ("threefold_repetition", "g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1", "f6g8"),
        ("previous_missed_Qh1_mate",
         "e2e4 e7e6 d2d4 d7d5 b1d2 c7c5 e4d5 e6d5 g1f3 b8c6 f1b5 f8d6 "
         "d4c5 d6c5 e1g1 g8e7 c2c3 e8g8 d2b3 c5b6 f1e1 e7f5 a2a4 h7h6 "
         "a4a5 b6c7 b5e2 a7a6 a1a2 d5d4 c3d4 c8e6 h2h3 e6d5 a2a1 f8e8 "
         "b3c5 f5h4 b2b3 a8b8 c1d2 d5f3 e2f3 d8d6 f3c6 d6h2 g1f1", "h2h1"),
    ]:
        board, state = chess.Board(), env.init()
        for uci in moves.split():
            move = chess.Move.from_uci(uci)
            state = step(state, jnp.int32(PgxAction.encode(move, board)))
            board.push(move)
        add(name, board, state, last)
    for name, fen, uci in [
        ("queen_mate", "k7/2Q5/2K5/8/8/8/8/8 w - - 0 1", "c7b7"),
        ("stalemate", "k7/2Q5/2K5/8/8/8/8/8 w - - 0 1", "c7b6"),
        ("fifty_move_rule", "7k/8/8/8/8/8/Q7/K7 w - - 99 80", "a2a3"),
        ("capture_to_bare_kings", "7k/8/8/8/8/8/1r6/K7 w - - 0 1", "a1b2"),
    ]:
        board = chess.Board(fen)
        add(name, board, from_board(env, board), uci)
        mirrored, move = board.mirror(), chess.Move.from_uci(uci)
        add(name + "_mirrored", mirrored, from_board(env, mirrored),
            chess.Move(chess.square_mirror(move.from_square), chess.square_mirror(move.to_square)).uci())
    state = env.init()
    state = state.replace(_x=state._x._replace(step_count=jnp.int32(511)), _step_count=jnp.int32(511))
    add("512_ply_cap", chess.Board(), state, "e2e4")
    return result
