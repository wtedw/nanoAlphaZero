"""Legacy ~/az CPL arithmetic and per-game engine lifetime, without JAX/W&B."""

import statistics
import time

import chess
import chess.engine


def centipawn_losses(fens, evaluations, initial_cp=15):
    """White-perspective scores; clip to +/-1000 cp and floor losses at zero."""
    losses = {"white": [], "black": []}
    before = initial_cp
    clip = lambda value: max(-1000, min(1000, value if value is not None else 0))
    for fen, after in zip(fens[:-1], evaluations, strict=True):
        white = chess.Board(fen).turn == chess.WHITE
        loss = clip(before) - clip(after) if white else clip(after) - clip(before)
        losses["white" if white else "black"].append(max(0, loss))
        before = after
    result = {}
    for side, values in losses.items():
        result[f"acpl_{side}"] = statistics.mean(values) if values else None
        result[f"stdcpl_{side}"] = statistics.stdev(values) if len(values) > 1 else 0.
        result[f"moves_{side}"] = len(values)
    return result


def score_fens(fens, options, opening_plies=0):
    """One Stockfish process per game, retained across all positions in order.

    Retry the complete game with a fresh process, just as the original code did.
    Excluding book moves is optional; it evaluates the post-book starting board
    instead of using the legacy +15 cp initial-position baseline.
    """
    start = 0 if options.include_openings else opening_plies
    fens = fens[start:]
    if len(fens) <= 1:
        return centipawn_losses(fens, [], 0)
    for attempt in range(options.engine_retries + 1):
        try:
            with chess.engine.SimpleEngine.popen_uci(
                str(options.stockfish_path), timeout=options.engine_timeout,
            ) as engine:
                def analyse(fen):
                    info = engine.analyse(chess.Board(fen), chess.engine.Limit(depth=options.depth))
                    return info["score"].white().score(mate_score=10_000)
                before = 15 if options.include_openings else analyse(fens[0])
                # Sequential analysis intentionally preserves per-game engine hash reuse.
                return centipawn_losses(fens, [analyse(fen) for fen in fens[1:]], before)
        except (TimeoutError, chess.engine.EngineTerminatedError):
            if attempt == options.engine_retries:
                raise
            time.sleep(2 ** attempt)
