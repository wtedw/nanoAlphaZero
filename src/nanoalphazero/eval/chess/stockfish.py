"""Parallel UCI player and adjudicator pools."""

from __future__ import annotations

import asyncio
import hashlib
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import chess
import chess.engine

from nanoalphazero.eval.protocol import PositionBatch
from nanoalphazero.eval.chess.searchless_actions import ordered_legal_moves


STOCKFISH_MODES = {"standard", "all_moves"}
ADJUDICATOR_BACKENDS = {"async_pool", "threaded_pool"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class StockfishPool:
    def __init__(
        self,
        path: str,
        *,
        pool_size: int,
        time_limit: float,
        mode: str = "standard",
        elo: int | None = None,
        threads: int = 1,
        hash_mb: int = 16,
        protocol_timeout: float = 60.0,
    ):
        executable = Path(path).expanduser().resolve()
        if not executable.is_file():
            raise FileNotFoundError(f"Stockfish executable not found: {executable}")
        self.path = executable
        self.time_limit = float(time_limit)
        if mode not in STOCKFISH_MODES:
            raise ValueError(
                f"unsupported Stockfish mode {mode!r}; "
                f"expected one of {sorted(STOCKFISH_MODES)}"
            )
        self.mode = mode
        self.elo = elo
        self.protocol_timeout = float(protocol_timeout)
        self._options: dict[str, Any] = {"Threads": threads, "Hash": hash_mb}
        if elo is not None:
            self._options.update({"UCI_LimitStrength": True, "UCI_Elo": elo})
        self._executor = ThreadPoolExecutor(max_workers=pool_size)
        self._engines: list[chess.engine.SimpleEngine] = []
        self._seconds = 0.0
        self._calls = 0
        self._restarts = 0
        try:
            for _ in range(pool_size):
                self._engines.append(self._open_engine())
        except BaseException:
            self.close()
            raise

    def _open_engine(self) -> chess.engine.SimpleEngine:
        engine = chess.engine.SimpleEngine.popen_uci(
            str(self.path), timeout=self.protocol_timeout
        )
        engine.configure(self._options)
        return engine

    def _restart_engine(self, index: int) -> None:
        try:
            self._engines[index].close()
        except Exception:
            pass
        self._engines[index] = self._open_engine()
        self._restarts += 1

    def _call_with_recovery(self, index: int, fn, board):
        try:
            return fn(self._engines[index], board)
        except (TimeoutError, chess.engine.EngineTerminatedError):
            self._restart_engine(index)
            return fn(self._engines[index], board)

    @property
    def pool_size(self) -> int:
        return len(self._engines)

    def _map(self, fn, boards: tuple[chess.Board, ...]):
        started = time.perf_counter()
        results = []
        for offset in range(0, len(boards), self.pool_size):
            chunk = boards[offset : offset + self.pool_size]
            futures = [
                self._executor.submit(self._call_with_recovery, index, fn, board)
                for index, board in enumerate(chunk)
            ]
            results.extend(future.result() for future in futures)
        self._seconds += time.perf_counter() - started
        self._calls += len(boards)
        return results

    def play_batch(self, boards: tuple[chess.Board, ...]) -> list[chess.Move]:
        limit = chess.engine.Limit(time=self.time_limit)
        if self.mode == "standard":
            return self._map(
                lambda engine, board: engine.play(board, limit).move, boards
            )

        # Match DeepMind's Searchless Chess oracle: force each legal move at
        # the root, give every move the complete time limit, and select the
        # best score from the side-to-move perspective.  The worker pool only
        # parallelizes independent boards; one worker follows the published
        # algorithm serially for all legal moves on its assigned board.
        def play_all_moves(engine, board):
            scored_moves = []
            for move in ordered_legal_moves(board):
                info = engine.analyse(board, limit=limit, root_moves=[move])
                scored_moves.append((move, info["score"].relative))
            if not scored_moves:
                raise ValueError("Stockfish received a position with no legal moves")
            return sorted(
                scored_moves, key=lambda move_and_score: move_and_score[1], reverse=True
            )[0][0]

        return self._map(play_all_moves, boards)

    def analyse_batch(self, boards: tuple[chess.Board, ...]):
        limit = chess.engine.Limit(time=self.time_limit)
        return self._map(
            lambda engine, board: engine.analyse(
                board, limit, info=chess.engine.INFO_SCORE
            ),
            boards,
        )

    def stats(self) -> dict[str, Any]:
        return {
            "calls": self._calls,
            "batch_wall_seconds": self._seconds,
            "mean_batch_wall_seconds_per_position": (
                self._seconds / self._calls if self._calls else 0.0
            ),
            "pool_size": self.pool_size,
            "mode": self.mode,
            "time_limit": self.time_limit,
            "protocol_timeout": self.protocol_timeout,
            "engine_restarts": self._restarts,
            "elo": self.elo,
            "engine_path": str(self.path),
            "engine_sha256": _sha256(self.path),
            "engine_id": dict(self._engines[0].id) if self._engines else {},
        }

    def close(self) -> None:
        for engine in self._engines:
            try:
                engine.close()
            except Exception:
                pass
        self._engines.clear()
        self._executor.shutdown(wait=True, cancel_futures=True)


class AsyncStockfishPool:
    """Persistent UCI protocols sharing one event loop for batched analysis."""

    def __init__(
        self,
        path: str,
        *,
        pool_size: int,
        time_limit: float,
        threads: int = 1,
        hash_mb: int = 16,
        protocol_timeout: float = 60.0,
    ):
        executable = Path(path).expanduser().resolve()
        if not executable.is_file():
            raise FileNotFoundError(f"Stockfish executable not found: {executable}")
        self.path = executable
        self.time_limit = float(time_limit)
        self.protocol_timeout = float(protocol_timeout)
        self._pool_size = int(pool_size)
        if self._pool_size <= 0:
            raise ValueError("Stockfish pool_size must be positive")
        self._options: dict[str, Any] = {"Threads": threads, "Hash": hash_mb}
        self._transports: list[asyncio.SubprocessTransport] = []
        self._engines: list[chess.engine.UciProtocol] = []
        self._seconds = 0.0
        self._calls = 0
        self._restarts = 0
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="stockfish-adjudicator",
            daemon=True,
        )
        self._thread.start()
        try:
            self._submit(self._open_all(), timeout=2 * self.protocol_timeout)
        except BaseException:
            self.close()
            raise

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _submit(self, awaitable, *, timeout: float):
        future = asyncio.run_coroutine_threadsafe(awaitable, self._loop)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            future.cancel()
            raise

    async def _open_engine(self):
        transport, engine = await asyncio.wait_for(
            chess.engine.popen_uci(str(self.path)),
            timeout=self.protocol_timeout,
        )
        try:
            await asyncio.wait_for(
                engine.configure(self._options), timeout=self.protocol_timeout
            )
        except BaseException:
            transport.close()
            raise
        return transport, engine

    async def _open_all(self) -> None:
        opened = await asyncio.gather(
            *(self._open_engine() for _ in range(self._pool_size))
        )
        self._transports = [transport for transport, _ in opened]
        self._engines = [engine for _, engine in opened]

    async def _restart_engine(self, index: int) -> None:
        try:
            await asyncio.wait_for(
                self._engines[index].quit(), timeout=self.protocol_timeout
            )
        except Exception:
            pass
        self._transports[index].close()
        transport, engine = await self._open_engine()
        self._transports[index] = transport
        self._engines[index] = engine
        self._restarts += 1

    async def _analyse_one(self, index: int, board: chess.Board):
        limit = chess.engine.Limit(time=self.time_limit)

        async def analyse():
            return await asyncio.wait_for(
                self._engines[index].analyse(
                    board, limit, info=chess.engine.INFO_SCORE
                ),
                timeout=self.protocol_timeout,
            )

        try:
            return await analyse()
        except (TimeoutError, chess.engine.EngineTerminatedError):
            await self._restart_engine(index)
            return await analyse()

    async def _analyse_batch(self, boards: tuple[chess.Board, ...]):
        results = []
        for offset in range(0, len(boards), self.pool_size):
            chunk = boards[offset : offset + self.pool_size]
            results.extend(
                await asyncio.gather(
                    *(
                        self._analyse_one(index, board)
                        for index, board in enumerate(chunk)
                    )
                )
            )
        return results

    @property
    def pool_size(self) -> int:
        return self._pool_size

    def analyse_batch(self, boards: tuple[chess.Board, ...]):
        started = time.perf_counter()
        chunks = max(1, math.ceil(len(boards) / self.pool_size))
        results = self._submit(
            self._analyse_batch(boards),
            timeout=chunks * (4 * self.protocol_timeout + 2 * self.time_limit),
        )
        self._seconds += time.perf_counter() - started
        self._calls += len(boards)
        return results

    def stats(self) -> dict[str, Any]:
        return {
            "calls": self._calls,
            "batch_wall_seconds": self._seconds,
            "mean_batch_wall_seconds_per_position": (
                self._seconds / self._calls if self._calls else 0.0
            ),
            "pool_size": self.pool_size,
            "mode": "standard",
            "time_limit": self.time_limit,
            "protocol_timeout": self.protocol_timeout,
            "engine_restarts": self._restarts,
            "elo": None,
            "engine_path": str(self.path),
            "engine_sha256": _sha256(self.path),
            "engine_id": dict(self._engines[0].id) if self._engines else {},
        }

    async def _close_all(self) -> None:
        async def close_one(engine, transport):
            try:
                await asyncio.wait_for(engine.quit(), timeout=self.protocol_timeout)
            except Exception:
                transport.close()

        await asyncio.gather(
            *(
                close_one(engine, transport)
                for engine, transport in zip(
                    self._engines, self._transports, strict=True
                )
            ),
            return_exceptions=True,
        )
        self._engines.clear()
        self._transports.clear()

    def close(self) -> None:
        if self._loop.is_running():
            try:
                self._submit(
                    self._close_all(), timeout=self.protocol_timeout + 1.0
                )
            finally:
                self._loop.call_soon_threadsafe(self._loop.stop)
                self._thread.join(timeout=self.protocol_timeout)
        if not self._loop.is_closed():
            self._loop.close()


class StockfishPlayer:
    # Resident JAX tournaments should ask Stockfish only about live Python
    # boards.  Inactive and padding pgx rows are stepped with ignored labels.
    resident_live_only = True

    def __init__(self, name: str, config: dict[str, Any], batch_size: int):
        self.name = name
        pool_size = int(config.get("pool_size") or min(batch_size, os.cpu_count() or 4))
        self._pool = StockfishPool(
            str(config.get("path", "/usr/local/bin/stockfish")),
            pool_size=pool_size,
            time_limit=float(config.get("time_limit", 0.05)),
            mode=str(config.get("mode", "standard")),
            elo=int(config["elo"]) if "elo" in config else None,
            threads=int(config.get("threads", 1)),
            hash_mb=int(config.get("hash_mb", 16)),
            protocol_timeout=float(config.get("protocol_timeout", 60.0)),
        )

    def warmup(self, batch_size: int) -> None:
        del batch_size

    def play_batch(self, batch: PositionBatch):
        return self._pool.play_batch(batch.boards)

    def stats(self):
        return self._pool.stats()

    def close(self) -> None:
        self._pool.close()


class StockfishAdjudicator:
    def __init__(self, config: dict[str, Any], batch_size: int):
        pool_size = int(
            config.get("pool_size") or min(batch_size, 48, os.cpu_count() or 4)
        )
        self.threshold_cp = int(config.get("threshold_cp", 1300))
        self.backend = str(config.get("backend", "async_pool"))
        if self.backend not in ADJUDICATOR_BACKENDS:
            raise ValueError(
                f"unsupported adjudicator backend {self.backend!r}; "
                f"expected one of {sorted(ADJUDICATOR_BACKENDS)}"
            )
        pool_class = (
            AsyncStockfishPool if self.backend == "async_pool" else StockfishPool
        )
        self._pool = pool_class(
            str(config.get("path", "/usr/local/bin/stockfish")),
            pool_size=pool_size,
            time_limit=float(config.get("time_limit", 0.01)),
            threads=int(config.get("threads", 1)),
            hash_mb=int(config.get("hash_mb", 16)),
            protocol_timeout=float(config.get("protocol_timeout", 60.0)),
        )

    def analyse_batch(self, boards: tuple[chess.Board, ...]):
        return self._pool.analyse_batch(boards)

    def stats(self):
        return {"backend": self.backend, **self._pool.stats()}

    def close(self) -> None:
        self._pool.close()
