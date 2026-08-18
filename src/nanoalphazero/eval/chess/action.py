"""python-chess to PGX action conversion used by tournament replay."""

from __future__ import annotations

import chess


class PgxAction:
    @staticmethod
    def encode(move: chess.Move, board: chess.Board) -> int:
        frm_py = PgxAction._orient(move.from_square, board)
        to_py = PgxAction._orient(move.to_square, board)
        frm = PgxAction._rank_to_file_major(frm_py)
        to = PgxAction._rank_to_file_major(to_py)
        promo_map = {chess.ROOK: 0, chess.BISHOP: 1, chess.KNIGHT: 2}
        if move.promotion in promo_map:
            rank_delta = to % 8 - frm % 8
            file_delta = to // 8 - frm // 8
            if rank_delta != 1 or file_delta not in (-1, 0, 1):
                raise ValueError(f"invalid oriented underpromotion {move.uci()}")
            direction = {0: 0, 1: 1, -1: 2}[file_delta]
            return frm * 73 + promo_map[move.promotion] * 3 + direction
        return frm * 73 + PgxAction._move_plane(frm, to)

    @staticmethod
    def _move_plane(frm: int, to: int) -> int:
        rank_delta = to % 8 - frm % 8
        file_delta = to // 8 - frm // 8
        knight_planes = {
            (-1, -2): 65,
            (1, -2): 66,
            (-2, -1): 67,
            (2, -1): 68,
            (-1, 2): 69,
            (1, 2): 70,
            (-2, 1): 71,
            (2, 1): 72,
        }
        if (rank_delta, file_delta) in knight_planes:
            return knight_planes[(rank_delta, file_delta)]
        distance = max(abs(rank_delta), abs(file_delta))
        if not 1 <= distance <= 7:
            raise ValueError(f"invalid PGX move delta {(rank_delta, file_delta)}")
        if rank_delta not in (0, -distance, distance) or file_delta not in (
            0,
            -distance,
            distance,
        ):
            raise ValueError(f"invalid PGX move delta {(rank_delta, file_delta)}")
        direction = {
            (-distance, 0): 0,
            (distance, 0): 1,
            (0, -distance): 2,
            (0, distance): 3,
            (-distance, -distance): 4,
            (distance, distance): 5,
            (distance, -distance): 6,
            (-distance, distance): 7,
        }.get((rank_delta, file_delta))
        if direction is None:
            raise ValueError(f"invalid PGX move delta {(rank_delta, file_delta)}")
        offset = 7 - distance if direction % 2 == 0 else distance - 1
        return 9 + direction * 7 + offset

    @staticmethod
    def decode(label: int, board: chess.Board) -> chess.Move:
        white, black, maybe_white, maybe_black = _decode_table()
        if board.turn == chess.WHITE:
            move, maybe_queen = white[label], maybe_white[label]
        else:
            move, maybe_queen = black[label], maybe_black[label]
        if move is None:
            raise ValueError(f"invalid PGX action label {label}")
        if maybe_queen and board.piece_type_at(move.from_square) == chess.PAWN:
            return chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)
        return move

    @staticmethod
    def _orient(square: int, board: chess.Board) -> int:
        return chess.square_mirror(square) if board.turn == chess.BLACK else square

    @staticmethod
    def _rank_to_file_major(index: int) -> int:
        return (index % 8) * 8 + index // 8


_DECODE_TABLE = None


def _build_decode_table():
    promotions = {0: chess.ROOK, 1: chess.BISHOP, 2: chess.KNIGHT}
    white, black, white_queen, black_queen = [], [], [], []
    for label in range(4672):
        frm = label // 73
        plane = label % 73
        rank, file = frm % 8, frm // 8
        under = plane // 3 if plane < 9 else -1
        if plane < 9:
            direction = plane % 3
            rank_delta = 1
            file_delta = (0, 1, -1)[direction]
            valid = rank == 6
        elif plane < 65:
            encoded = plane - 9
            direction = encoded // 7
            offset = encoded % 7
            distance = 7 - offset if direction % 2 == 0 else offset + 1
            rank_sign, file_sign = (
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),
                (-1, -1),
                (1, 1),
                (1, -1),
                (-1, 1),
            )[direction]
            rank_delta = rank_sign * distance
            file_delta = file_sign * distance
            valid = True
        else:
            rank_delta, file_delta = (
                (-1, -2),
                (1, -2),
                (-2, -1),
                (2, -1),
                (-1, 2),
                (1, 2),
                (-2, 1),
                (2, 1),
            )[plane - 65]
            valid = True
        to_rank = rank + rank_delta
        to_file = file + file_delta
        valid = valid and 0 <= to_rank < 8 and 0 <= to_file < 8
        if not valid:
            to = -1
        else:
            to = to_file * 8 + to_rank
        frm = (frm % 8) * 8 + frm // 8
        to = (to % 8) * 8 + to // 8
        promotion = promotions[under] if under >= 0 else None
        maybe_queen = under < 0 and to // 8 in (0, 7)
        white.append(chess.Move(frm, to, promotion=promotion))
        black.append(
            chess.Move(
                chess.square_mirror(frm),
                chess.square_mirror(to),
                promotion=promotion,
            )
        )
        white_queen.append(maybe_queen)
        black_queen.append(maybe_queen)
    return white, black, white_queen, black_queen


def _decode_table():
    global _DECODE_TABLE
    if _DECODE_TABLE is None:
        _DECODE_TABLE = _build_decode_table()
    return _DECODE_TABLE

