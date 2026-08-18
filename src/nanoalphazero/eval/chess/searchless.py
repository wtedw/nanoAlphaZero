# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0. Modified for standalone,
# cross-game batched inference in nanoAlphaZero.
"""DeepMind Searchless Chess action-value players."""

from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import chess
import jax
import numpy as np
import orbax.checkpoint as ocp

from nanoalphazero.eval.protocol import PositionBatch
from nanoalphazero.eval.chess.searchless_actions import MOVE_TO_ACTION, ordered_legal_moves
from nanoalphazero.eval.chess.searchless_transformer import (
    PositionalEncodings,
    TransformerConfig,
    build_transformer_predictor,
)


_CHARACTERS = list("0123456789abcdefghpnrkqPBNRQKw.")
_CHARACTERS_INDEX = {letter: index for index, letter in enumerate(_CHARACTERS)}
_SPACES_CHARACTERS = frozenset("12345678")
SEQUENCE_LENGTH = 77
REPETITION_CHECK_VERSIONS = frozenset({"v1", "v2"})
INFERENCE_SHARDINGS = frozenset({"single", "data_parallel"})


@dataclass(frozen=True)
class SearchlessModelSpec:
    num_layers: int
    embedding_dim: int
    num_heads: int
    default_inference_batch_size: int


# Matches DeepMind's released action-value model registry. Keeping 270M here
# allows explicit checkpoints without making the large archive a default asset.
SEARCHLESS_MODELS = {
    "9M": SearchlessModelSpec(8, 256, 8, 1024),
    "136M": SearchlessModelSpec(8, 1024, 8, 256),
    "270M": SearchlessModelSpec(16, 1024, 8, 128),
}


def tokenize(fen: str) -> np.ndarray:
    board, side, castling, en_passant, halfmoves, fullmoves = fen.split(" ")
    board = side + board.replace("/", "")
    indices: list[int] = []
    for character in board:
        if character in _SPACES_CHARACTERS:
            indices.extend(int(character) * [_CHARACTERS_INDEX["."]])
        else:
            indices.append(_CHARACTERS_INDEX[character])
    if castling == "-":
        indices.extend(4 * [_CHARACTERS_INDEX["."]])
    else:
        indices.extend(_CHARACTERS_INDEX[character] for character in castling)
        indices.extend((4 - len(castling)) * [_CHARACTERS_INDEX["."]])
    if en_passant == "-":
        indices.extend(2 * [_CHARACTERS_INDEX["."]])
    else:
        indices.extend(_CHARACTERS_INDEX[character] for character in en_passant)
    halfmoves += "." * (3 - len(halfmoves))
    fullmoves += "." * (3 - len(fullmoves))
    indices.extend(_CHARACTERS_INDEX[character] for character in halfmoves)
    indices.extend(_CHARACTERS_INDEX[character] for character in fullmoves)
    if len(indices) != SEQUENCE_LENGTH:
        raise ValueError(f"tokenized FEN has {len(indices)} tokens, expected 77: {fen}")
    return np.asarray(indices, dtype=np.uint8)


RETURN_BUCKET_VALUES = (
    np.linspace(0.0, 1.0, 129)[:-1] + np.linspace(0.0, 1.0, 129)[1:]
) / 2


@dataclass
class RepetitionCheckStats:
    candidate_checks: int = 0
    authoritative_checks: int = 0
    prefilter_skips: int = 0
    draw_overrides: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "candidate_checks": self.candidate_checks,
            "authoritative_checks": self.authoritative_checks,
            "prefilter_skips": self.prefilter_skips,
            "draw_overrides": self.draw_overrides,
        }


# --- Repetition checking: v1 is the reference, v2 is v1 plus a prefilter. ---
#
# `repetition_check_version` picks between two chains of paired functions:
#
#     select_moves_v{1,2} -> repetition_draws_v{1,2}
#                         -> update_scores_with_repetitions_v{1,2}
#
# Only the last pair differs in logic. V1 is DeepMind's original: push every
# legal move and ask python-chess whether the opponent could then claim a draw.
# V2 first computes the position's repeated reversible-history keys once, skips
# the moves that provably cannot produce a claim, and runs the *identical* v1
# check on the ones that survive. Same decisions, fewer push/claim cycles; the
# skipped fraction is reported as `prefilter_skips`. The two wrapper pairs only
# thread the chosen variant through and record stats, so v2 is a speed
# optimization and never a rule change (tests/test_searchless.py pins that).


def update_scores_with_repetitions_v1(
    board: chess.Board,
    scores: np.ndarray,
) -> None:
    """DeepMind's original repetition helper, retained in source-shaped form."""
    sorted_legal_moves = ordered_legal_moves(board)
    for i, move in enumerate(sorted_legal_moves):
        board.push(move)
        # If the move results in a draw, associate 50% win prob to it.
        if board.is_fivefold_repetition() or board.can_claim_threefold_repetition():
            scores[i] = 0.5
        board.pop()


# --- V2 optimization: the original decision loop remains visible above. ---


def update_scores_with_repetitions_v2(
    board: chess.Board,
    scores: np.ndarray,
) -> int:
    """Runs the original helper only where cached history permits a claim."""
    sorted_legal_moves = ordered_legal_moves(board)
    repeated_keys = _repeated_reversible_history_keys(board)
    authoritative_checks = 0
    for i, move in enumerate(sorted_legal_moves):
        if not _might_claim_draw_after_move(board, move, repeated_keys):
            continue
        authoritative_checks += 1
        board.push(move)
        try:
            # The authoritative decision remains identical to v1.
            if board.is_fivefold_repetition() or board.can_claim_threefold_repetition():
                scores[i] = 0.5
        finally:
            board.pop()
    return authoritative_checks


# --- V2 prefilter implementation details. ---


def _repeated_reversible_history_keys(board: chess.Board) -> frozenset[Any]:
    """Finds keys occurring twice in python-chess's claimable history window."""
    transpositions = Counter((board._transposition_key(),))
    switchyard: list[chess.Move] = []
    try:
        while board.move_stack:
            move = board.pop()
            switchyard.append(move)
            if board.is_irreversible(move):
                break
            transpositions.update((board._transposition_key(),))
    finally:
        while switchyard:
            board.push(switchyard.pop())
    return frozenset(key for key, count in transpositions.items() if count >= 2)


def _might_claim_draw_after_move(
    board: chess.Board,
    move: chess.Move,
    repeated_keys: frozenset[Any],
) -> bool:
    """Conservatively detects candidates that require the original check."""
    if not repeated_keys or board.is_irreversible(move):
        return False
    board.push(move)
    try:
        if board._transposition_key() in repeated_keys:
            return True
        for reply in board.generate_legal_moves():
            board.push(reply)
            try:
                if board._transposition_key() in repeated_keys:
                    return True
            finally:
                board.pop()
        return False
    finally:
        board.pop()


def repetition_draws_v1(
    board: chess.Board,
    moves: Sequence[chess.Move],
    stats: RepetitionCheckStats | None = None,
) -> np.ndarray:
    """Returns the candidates changed by the original DeepMind helper."""
    scores = np.full(len(moves), np.nan)
    update_scores_with_repetitions_v1(board, scores)
    draws = scores == 0.5
    if stats is not None:
        stats.candidate_checks += len(moves)
        stats.authoritative_checks += len(moves)
        stats.draw_overrides += int(draws.sum())
    return draws


def repetition_draws_v2(
    board: chess.Board,
    moves: Sequence[chess.Move],
    stats: RepetitionCheckStats | None = None,
) -> np.ndarray:
    """Prefilters impossible claims and uses v1 as the oracle for possible ones."""
    scores = np.full(len(moves), np.nan)
    authoritative_checks = update_scores_with_repetitions_v2(board, scores)
    draws = scores == 0.5
    if stats is not None:
        stats.candidate_checks += len(moves)
        stats.authoritative_checks += authoritative_checks
        stats.prefilter_skips += len(moves) - authoritative_checks
        stats.draw_overrides += int(draws.sum())
    return draws


def _select_moves(
    boards: Sequence[chess.Board],
    move_groups: Sequence[Sequence[chess.Move]],
    log_probs: np.ndarray,
    repetition_fn,
    stats: RepetitionCheckStats | None,
) -> list[chess.Move]:
    selected = []
    offset = 0
    for board, moves in zip(boards, move_groups, strict=True):
        scores = np.exp(log_probs[offset : offset + len(moves)]) @ RETURN_BUCKET_VALUES
        scores[repetition_fn(board, moves, stats)] = 0.5
        selected.append(moves[int(np.argmax(scores))])
        offset += len(moves)
    return selected


def select_moves_v1(
    boards: Sequence[chess.Board],
    move_groups: Sequence[Sequence[chess.Move]],
    log_probs: np.ndarray,
    stats: RepetitionCheckStats | None = None,
) -> list[chess.Move]:
    """Selects moves with the original per-candidate repetition checks."""
    return _select_moves(boards, move_groups, log_probs, repetition_draws_v1, stats)


def select_moves_v2(
    boards: Sequence[chess.Board],
    move_groups: Sequence[Sequence[chess.Move]],
    log_probs: np.ndarray,
    stats: RepetitionCheckStats | None = None,
) -> list[chess.Move]:
    """Selects moves with conservative cached repetition prefiltering."""
    return _select_moves(boards, move_groups, log_probs, repetition_draws_v2, stats)


_MOVE_SELECTORS = {"v1": select_moves_v1, "v2": select_moves_v2}


def _restore_params(checkpoint_root: Path, initial_params, model_name: str = "9M"):
    checkpoint_root = checkpoint_root.expanduser().resolve()
    checkpoint_path = checkpoint_root / "6400000" / "params"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"missing Searchless {model_name} params: {checkpoint_path}; run "
            f"`uv run assets fetch searchless-{model_name.lower()}`"
        )
    restore_args = ocp.checkpoint_utils.construct_restore_args(initial_params)
    checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
    return checkpointer.restore(checkpoint_path, restore_args=restore_args)


class SearchlessPlayer:
    def __init__(self, name: str, config: dict[str, Any], batch_size: int):
        from nanoalphazero import core

        self.name = name
        self._model_name = str(config.get("model", "9M"))
        if self._model_name not in SEARCHLESS_MODELS:
            raise ValueError(
                f"unsupported Searchless model {self._model_name!r}; expected one of "
                f"{sorted(SEARCHLESS_MODELS)}"
            )
        spec = SEARCHLESS_MODELS[self._model_name]
        self._inference_batch_size = int(
            config.get("inference_batch_size", spec.default_inference_batch_size)
        )
        if self._inference_batch_size <= 0:
            raise ValueError("inference_batch_size must be positive")
        self._inference_sharding = str(
            config.get("inference_sharding", "data_parallel")
        )
        if self._inference_sharding not in INFERENCE_SHARDINGS:
            raise ValueError(
                "inference_sharding must be one of "
                f"{sorted(INFERENCE_SHARDINGS)}"
            )
        if (
            self._inference_sharding == "data_parallel"
            and self._inference_batch_size % jax.device_count()
        ):
            raise ValueError(
                "data-parallel inference_batch_size must be divisible by the "
                f"JAX device count ({jax.device_count()})"
            )
        self._repetition_check_version = str(
            config.get("repetition_check_version", "v2")
        )
        if self._repetition_check_version not in REPETITION_CHECK_VERSIONS:
            raise ValueError(
                "repetition_check_version must be one of "
                f"{sorted(REPETITION_CHECK_VERSIONS)}"
            )
        self._select_moves = _MOVE_SELECTORS[self._repetition_check_version]
        model_config = TransformerConfig(
            vocab_size=len(MOVE_TO_ACTION),
            output_size=128,
            pos_encodings=PositionalEncodings.LEARNED,
            max_sequence_length=SEQUENCE_LENGTH + 2,
            num_heads=spec.num_heads,
            num_layers=spec.num_layers,
            embedding_dim=spec.embedding_dim,
            apply_post_ln=True,
            apply_qk_layernorm=False,
            use_causal_mask=False,
        )
        self._predictor = build_transformer_predictor(model_config)
        initial = self._predictor.initial_params(
            rng=jax.random.PRNGKey(1),
            targets=np.ones((1, 1), dtype=np.uint32),
        )
        self._params = _restore_params(
            Path(config["checkpoint"]).expanduser(), initial, self._model_name
        )

        def predict(params, rows):
            return self._predictor.predict(params, None, rows)[:, -1]

        if self._inference_sharding == "data_parallel":
            self._params = jax.device_put(self._params, core.REPLICATED_SHARDING)
            self._predict = jax.jit(
                predict,
                in_shardings=(
                    core.REPLICATED_SHARDING,
                    core.DATA_PARALLEL_SHARDING,
                ),
                out_shardings=core.DATA_PARALLEL_SHARDING,
            )
        else:
            self._predict = jax.jit(predict)
        self._inference_devices: tuple[str, ...] = ()
        self._seconds = 0.0
        self._warmup_seconds = 0.0
        self._positions = 0
        self._model_rows = 0
        self._calls = 0
        self._repetition_stats = RepetitionCheckStats()

    def _predict_rows(self, rows: np.ndarray) -> np.ndarray:
        outputs = []
        size = self._inference_batch_size
        for offset in range(0, len(rows), size):
            chunk = rows[offset : offset + size]
            real = len(chunk)
            if real < size:
                chunk = np.concatenate([chunk, np.repeat(chunk[-1:], size - real, axis=0)])
            prediction = self._predict(self._params, chunk)
            self._inference_devices = tuple(
                sorted(str(device) for device in prediction.devices())
            )
            outputs.append(np.asarray(jax.device_get(prediction))[:real])
        return np.concatenate(outputs, axis=0)

    def warmup(self, batch_size: int) -> None:
        started = time.perf_counter()
        boards = tuple(chess.Board() for _ in range(max(1, batch_size)))
        self.play_batch(PositionBatch(tuple(range(len(boards))), boards, None, (0,) * len(boards)))
        self._warmup_seconds = time.perf_counter() - started
        self._seconds = 0.0
        self._positions = self._model_rows = self._calls = 0
        self._repetition_stats = RepetitionCheckStats()

    def play_batch(self, batch: PositionBatch) -> list[chess.Move]:
        started = time.perf_counter()
        move_groups = [ordered_legal_moves(board) for board in batch.boards]
        rows = []
        for board, moves in zip(batch.boards, move_groups, strict=True):
            fen = tokenize(board.fen()).astype(np.int32)
            for move in moves:
                rows.append(
                    np.concatenate(
                        [fen, [MOVE_TO_ACTION[move.uci()], 0]]
                    ).astype(np.int32)
                )
        if not rows:
            raise ValueError("SearchlessPlayer received a position with no legal moves")
        log_probs = self._predict_rows(np.stack(rows))
        selected = self._select_moves(
            batch.boards, move_groups, log_probs, self._repetition_stats
        )
        self._seconds += time.perf_counter() - started
        self._positions += len(batch.boards)
        self._model_rows += len(rows)
        self._calls += 1
        return selected

    def stats(self) -> dict[str, Any]:
        return {
            "batch_calls": self._calls,
            "positions": self._positions,
            "model_rows": self._model_rows,
            "batch_wall_seconds": self._seconds,
            "warmup_seconds": self._warmup_seconds,
            "positions_per_second": self._positions / self._seconds if self._seconds else 0.0,
            "inference_batch_size": self._inference_batch_size,
            "inference_sharding": self._inference_sharding,
            "inference_devices": self._inference_devices,
            "model": self._model_name,
            "repetition_check_version": self._repetition_check_version,
            "repetition_checks": self._repetition_stats.as_dict(),
        }

    def close(self) -> None:
        pass
