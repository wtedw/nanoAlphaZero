"""Chess dynamics geometry, promotion conventions, and checkpoint architecture."""
from collections import namedtuple

import chess
import jax
import jax.numpy as jnp
import numpy as np
import pgx1.chess as pc
import pytest

from nanoalphazero.ckpt import load_model, validate_resume_config
from nanoalphazero.ckpt.state import save_payload
from nanoalphazero.cli import parse_muzero_args
from nanoalphazero.config import get_muzero_config
from nanoalphazero.eval.chess.action import PgxAction
from nanoalphazero.model import action_planes, chess_action_planes, make_muzero_model


def test_all_action_geometry_matches_environment_decoder():
    actions = jnp.arange(4672)
    encoded = np.asarray(jax.jit(chess_action_planes)(actions))
    decoded = jax.device_get(jax.jit(jax.vmap(pc.Action._from_label))(actions))
    assert encoded.shape == (4672, 8, 8, 8)
    assert np.isin(encoded, [0, 1]).all()
    for a in range(4672):
        source = int(decoded.from_[a])
        target = int(decoded.to[a])
        if a % 73 < 9:
            # PGX additionally enforces the promotion rank. Our flag deliberately
            # answers only whether the action's geometric destination is on-board.
            file = source // 8 + (0, 1, -1)[a % 73 % 3]
            rank = source % 8 + 1
            target = file * 8 + rank if 0 <= file < 8 and rank < 8 else -1
        assert encoded[a, 7-source % 8, source // 8, 0] == 1
        assert encoded[a, :, :, 0].sum() == 1
        assert encoded[a, :, :, 1].sum() == (target >= 0)
        assert np.all(encoded[a, :, :, 2] == (target >= 0))
        if target >= 0:
            assert encoded[a, 7-target % 8, target // 8, 1] == 1
        np.testing.assert_array_equal(encoded[a, :, :, 3:].sum(-1), 1)
        np.testing.assert_array_equal(encoded[a, :, :, 3:],
                                      np.broadcast_to(encoded[a, 0, 0, 3:], (8, 8, 5)))


@pytest.mark.parametrize("turn", [chess.WHITE, chess.BLACK])
def test_moves_and_promotions_in_mover_coordinates(turn):
    board = chess.Board()
    board.turn = turn
    # Coordinates start in White's perspective and are mirrored for Black.
    for uci, promotion in [("e2e4", 4), ("g1f3", 4), ("e1g1", 4), ("e1c1", 4),
                           ("e5d6", 4), ("a7a8q", 0), ("b7a8q", 0),
                           ("a7a8n", 1), ("b7a8b", 2), ("g7h8r", 3),
                           ("a7a8", 0), ("a1a8", 4)]:
        move = chess.Move.from_uci(uci)
        if turn == chess.BLACK:
            move = chess.Move(chess.square_mirror(move.from_square),
                              chess.square_mirror(move.to_square), promotion=move.promotion)
        action = PgxAction.encode(move, board)
        planes = np.asarray(chess_action_planes(jnp.array([action])))[0]
        source = PgxAction._orient(move.from_square, board)
        target = PgxAction._orient(move.to_square, board)
        assert planes[7-chess.square_rank(source), chess.square_file(source), 0] == 1
        assert planes[7-chess.square_rank(target), chess.square_file(target), 1] == 1
        assert np.all(planes[..., 3+promotion] == 1)


def test_queen_flag_is_conditional_and_not_a_piece_lookup():
    pawn = chess.Board("7k/P7/8/8/8/8/8/K7 w - - 0 1")
    rook = chess.Board("7k/R7/8/8/8/8/8/K7 w - - 0 1")
    a = PgxAction.encode(chess.Move.from_uci("a7a8q"), pawn)
    b = PgxAction.encode(chess.Move.from_uci("a7a8"), rook)
    assert a == b
    assert np.all(np.asarray(chess_action_planes(jnp.array([a])))[..., 3] == 1)


@pytest.mark.parametrize("encoding", [None, "planes73", "from_to8"])
def test_model_roundtrip_and_gradients(tmp_path, encoding):
    cfg = get_muzero_config("chess", conv_width=8, conv_depth=1, muzero_remat_blocks=False)
    cfg.update(game_obs_shape=[8, 8, 119], game_num_actions=4672)
    if encoding is None:
        del cfg["muzero_chess_action_encoding"]
    else:
        cfg["muzero_chess_action_encoding"] = encoding
    model, variables = make_muzero_model(cfg, jax.random.PRNGKey(4))
    effective = encoding or "planes73"
    assert model.chess_action_encoding == effective
    planes = action_planes(jnp.array([123]), 8, 8, 4672, "chess", chess_encoding=effective)
    assert planes.shape[-1] == (8 if effective == "from_to8" else 73)
    params = variables["params"]
    obs = jax.random.normal(jax.random.PRNGKey(5), (1, 8, 8, 119))
    action = jnp.array([123])
    expected = model.apply({"params": params}, obs, action)
    gradient = jax.grad(lambda p: sum(jnp.mean(x*x) for x in model.apply({"params": p}, obs, action)))(params)
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(gradient))
    assert any(np.any(np.asarray(x) != 0) for x in jax.tree.leaves(gradient["dynamics"]))
    path = tmp_path / "model.safetensors"
    state = {"model_ts": namedtuple("ModelState", "params")(params), "cycle": jnp.int32(0)}
    save_payload(path, state, cfg)
    restored, restored_params, saved = load_model(path)
    actual = restored.apply({"params": restored_params}, obs, action)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_array_equal(a, b)
    explicit = {**cfg, "muzero_chess_action_encoding": effective}
    validate_resume_config(path, explicit, "partial")
    if effective == "planes73":
        with pytest.raises(ValueError, match="identical resolved configuration"):
            validate_resume_config(path, {**cfg, "muzero_chess_action_encoding": "from_to8"}, "partial")
        from nanoalphazero.research.muzero.spatial import action_planes as old_planes
        np.testing.assert_array_equal(planes, old_planes(action, 8, 8, 4672, "chess"))


def test_cli_and_defaults():
    assert get_muzero_config("chess")["muzero_chess_action_encoding"] == "from_to8"
    assert "muzero_chess_action_encoding" not in get_muzero_config("hex4")
    assert parse_muzero_args(["--muzero.chess-action-encoding", "planes73"]).muzero.chess_action_encoding == "planes73"
