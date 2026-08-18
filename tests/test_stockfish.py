import asyncio
from concurrent.futures import ThreadPoolExecutor

import chess
import chess.engine

from nanoalphazero.eval.chess.searchless_actions import ordered_legal_moves
from nanoalphazero.eval.chess.stockfish import AsyncStockfishPool, StockfishPool


def test_stockfish_pool_processes_oversized_batches_in_chunks():
    pool = object.__new__(StockfishPool)
    pool._engines = [10, 20]
    pool._executor = ThreadPoolExecutor(max_workers=2)
    pool._seconds = 0.0
    pool._calls = 0
    try:
        results = pool._map(lambda engine, board: engine + board, (1, 2, 3, 4, 5))
    finally:
        pool._executor.shutdown()
    assert results == [11, 22, 13, 24, 15]
    assert pool._calls == 5


def test_stockfish_all_moves_matches_deepmind_forced_root_selection():
    class FakeEngine:
        def __init__(self):
            self.root_moves = []

        def analyse(self, board, *, limit, root_moves):
            del limit
            move = root_moves[0]
            self.root_moves.append(move.uci())
            cp = 100 if move == chess.Move.from_uci("e2e4") else 0
            return {"score": chess.engine.PovScore(chess.engine.Cp(cp), board.turn)}

    engine = FakeEngine()
    pool = object.__new__(StockfishPool)
    pool.time_limit = 0.05
    pool.mode = "all_moves"
    pool._engines = [engine]
    pool._executor = ThreadPoolExecutor(max_workers=1)
    pool._seconds = 0.0
    pool._calls = 0
    try:
        moves = pool.play_batch((chess.Board(),))
    finally:
        pool._executor.shutdown()

    assert moves == [chess.Move.from_uci("e2e4")]
    assert engine.root_moves == [
        move.uci() for move in ordered_legal_moves(chess.Board())
    ]


def test_stockfish_pool_restarts_and_retries_timed_out_worker():
    pool = object.__new__(StockfishPool)
    pool._engines = ["stale"]
    pool._restarts = 0

    def restart(index):
        pool._engines[index] = "fresh"
        pool._restarts += 1

    pool._restart_engine = restart

    def call(engine, board):
        if engine == "stale":
            raise TimeoutError
        return engine, board

    assert pool._call_with_recovery(0, call, "position") == ("fresh", "position")
    assert pool._restarts == 1


def test_async_adjudicator_chunks_positions_and_requests_only_scores():
    class FakeEngine:
        def __init__(self, value):
            self.value = value
            self.info_masks = []

        async def analyse(self, board, limit, *, info):
            del limit
            self.info_masks.append(info)
            return {"score": self.value + board}

    engines = [FakeEngine(10), FakeEngine(20)]
    pool = object.__new__(AsyncStockfishPool)
    pool._pool_size = 2
    pool._engines = engines
    pool.time_limit = 0.01
    pool.protocol_timeout = 1.0

    results = asyncio.run(pool._analyse_batch((1, 2, 3, 4, 5)))

    assert results == [
        {"score": 11},
        {"score": 22},
        {"score": 13},
        {"score": 24},
        {"score": 15},
    ]
    assert engines[0].info_masks == [chess.engine.INFO_SCORE] * 3
    assert engines[1].info_masks == [chess.engine.INFO_SCORE] * 2
