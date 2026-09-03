"""Pure-stdlib MoHex GTP process management.

This module intentionally does not import JAX.  Callers can create the engine
bank before libtpu is initialized.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Protocol, Sequence


DEFAULT_CONFIG = Path(__file__).with_name("mohex.cfg")


def action_to_vertex(action: int, board_size: int) -> str:
    if action < 0 or action >= board_size * board_size:
        raise ValueError(f"Hex action out of range: {action}")
    row, column = divmod(int(action), int(board_size))
    return f"{chr(ord('a') + column)}{row + 1}"


def vertex_to_action(vertex: str, board_size: int) -> int:
    value = vertex.strip().lower()
    if len(value) < 2 or not value[0].isalpha():
        raise ValueError(f"invalid Hex vertex from engine: {vertex!r}")
    column = ord(value[0]) - ord("a")
    try:
        row = int(value[1:]) - 1
    except ValueError as exc:
        raise ValueError(f"invalid Hex vertex from engine: {vertex!r}") from exc
    if not (0 <= row < board_size and 0 <= column < board_size):
        raise ValueError(f"Hex vertex out of range: {vertex!r}")
    return row * board_size + column


def resolve_executable(value: str) -> Path:
    expanded = Path(value).expanduser()
    if expanded.is_absolute() or expanded.parent != Path("."):
        path = expanded.resolve()
    else:
        found = shutil.which(value)
        if found is None:
            raise FileNotFoundError(f"MoHex executable not found: {value}")
        path = Path(found).resolve()
    if not path.is_file() or not path.stat().st_mode & 0o111:
        raise FileNotFoundError(f"MoHex executable not found or not executable: {path}")
    return path


def resolve_config(value: str | None) -> Path:
    if value in (None, "", "default"):
        return DEFAULT_CONFIG.resolve()
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"MoHex config not found: {path}")
    return path


class HexEngine(Protocol):
    def clear(self) -> None: ...
    def play(self, color: str, action: int) -> None: ...
    def begin_genmove(self, color: str) -> None: ...
    def finish_genmove(self) -> int | None: ...
    def close(self) -> None: ...


class MoHexEngine:
    def __init__(self, executable: Path, config: Path, board_size: int, seed: int):
        self.board_size = int(board_size)
        # Never pipe stderr: solver diagnostics can otherwise fill the pipe while
        # the synchronous GTP reader waits on stdout. Keep it for crash reports.
        self.stderr = tempfile.TemporaryFile(mode="w+t")
        self.process = subprocess.Popen(
            [
                str(executable),
                "--quiet",
                "--use-logfile=0",
                f"--config={config}",
                f"--seed={int(seed)}",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr,
            text=True,
            bufsize=1,
        )
        try:
            self.query(f"boardsize {self.board_size}")
        except BaseException:
            self.close()
            raise

    def _write(self, command: str) -> None:
        if self.process.poll() is not None or self.process.stdin is None:
            raise IOError(f"MoHex exited before command {command!r}")
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def _answer(self) -> str:
        if self.process.stdout is None:
            raise IOError("MoHex stdout is unavailable")
        lines: list[str] = []
        while True:
            line = self.process.stdout.readline()
            if line == "":
                self.stderr.seek(0)
                stderr = self.stderr.read()
                raise IOError(
                    f"MoHex exited unexpectedly with code {self.process.poll()}: {stderr.strip()}"
                )
            if line in ("\n", "\r\n"):
                break
            lines.append(line.rstrip("\r\n"))
        if not lines:
            raise IOError("MoHex returned an empty GTP response")
        marker = lines[0][:1]
        payload = "\n".join([lines[0][1:].lstrip(), *lines[1:]]).strip()
        if marker != "=":
            raise IOError(f"MoHex GTP error: {payload or lines[0]}")
        return payload

    def query(self, command: str) -> str:
        self._write(command)
        return self._answer()

    def clear(self) -> None:
        self.query("clear_board")

    def play(self, color: str, action: int) -> None:
        self.query(f"play {color} {action_to_vertex(action, self.board_size)}")

    def begin_genmove(self, color: str) -> None:
        # Benzene's reg_genmove does not update the board.  The match runner
        # validates and then sends the returned move through play().
        self._write(f"reg_genmove {color}")

    def finish_genmove(self) -> int | None:
        response = self._answer().strip().lower()
        if response == "resign":
            return None
        if response == "pass":
            raise ValueError("MoHex returned pass in a no-pass Hex game")
        return vertex_to_action(response, self.board_size)

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self._write("quit")
                self.process.wait(timeout=2)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        for stream in (self.process.stdin, self.process.stdout):
            if stream is not None:
                stream.close()
        self.stderr.close()


class MoHexPool:
    """One engine per real opening; padded JAX rows never reach this pool."""

    kind = "mohex"

    def __init__(
        self,
        path: str,
        config: str | None,
        board_size: int,
        count: int,
        *,
        seed: int = 1,
        engine_factory=MoHexEngine,
    ):
        self.executable = resolve_executable(path)
        self.config = resolve_config(config)
        self.board_size = int(board_size)
        self.engines: list[HexEngine] = []
        try:
            for index in range(int(count)):
                self.engines.append(
                    engine_factory(
                        self.executable, self.config, self.board_size, seed + index
                    )
                )
        except BaseException:
            self.close()
            raise

    def reset(self, openings: Sequence[int]) -> None:
        if len(openings) != len(self.engines):
            raise ValueError("opening count does not match the MoHex engine bank")
        for engine, opening in zip(self.engines, openings, strict=True):
            engine.clear()
            engine.play("b", int(opening))

    def play(self, color: str, actions: Sequence[int], active: Sequence[bool]) -> None:
        for engine, action, is_active in zip(
            self.engines, actions, active, strict=True
        ):
            if is_active:
                engine.play(color, int(action))

    def generate(self, color: str, active: Sequence[bool]) -> list[int | None]:
        selected = [
            (index, engine)
            for index, (engine, is_active) in enumerate(
                zip(self.engines, active, strict=True)
            )
            if is_active
        ]
        for _, engine in selected:
            engine.begin_genmove(color)
        output: list[int | None] = [-1] * len(self.engines)
        with ThreadPoolExecutor(max_workers=max(1, len(selected))) as executor:
            futures = {
                executor.submit(engine.finish_genmove): index
                for index, engine in selected
            }
            for future, index in futures.items():
                output[index] = future.result()
        return output

    def close(self) -> None:
        for engine in self.engines:
            engine.close()
        self.engines.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
