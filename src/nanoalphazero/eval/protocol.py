"""Public interfaces shared by the tournament and player implementations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import chess


@dataclass(frozen=True)
class PositionBatch:
    """One player's live positions in stable tournament-slot order."""

    slot_ids: tuple[int, ...]
    boards: tuple[chess.Board, ...]
    env_state: Any | None
    seeds: tuple[int, ...]


class BatchedPlayer(Protocol):
    """A tournament entrant that selects one legal move per live position."""

    name: str

    def warmup(self, batch_size: int) -> None: ...

    def play_batch(self, batch: PositionBatch) -> Sequence[chess.Move]: ...

    def stats(self) -> Mapping[str, Any]: ...

    def close(self) -> None: ...

