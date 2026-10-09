"""ACPL arithmetic, legacy opening book, engine retry lifecycle, and scheduling."""

import json
from pathlib import Path

import chess
import chess.engine
import jax
import numpy as np
import pytest

from nanoalphazero.eval.acpl import AcplConfig, make_acpl
from nanoalphazero.eval.acpl import scoring, runner
from nanoalphazero.eval.acpl.config import acpl_config_dict, acpl_from_args
from nanoalphazero.eval.chess.action import PgxAction


def fens(moves=("e2e4", "e7e5", "g1f3", "b8c6")):
    board = chess.Board()
    result = [board.fen()]
    for move in moves:
        board.push_uci(move)
        result.append(board.fen())
    return result


def test_legacy_scoring_signs_clipping_and_game_stddev():
    result = scoring.centipawn_losses(fens(), [-100, 50, 200, -1200])
    assert result["acpl_white"] == 57.5
    assert result["acpl_black"] == 75
    assert result["stdcpl_white"] == pytest.approx(115 / np.sqrt(2))
    assert result["stdcpl_black"] == pytest.approx(150 / np.sqrt(2))
    result = scoring.centipawn_losses(fens(), [10000, 20000, -10000, -20000])
    assert result["acpl_white"] == 1000
    assert result["acpl_black"] == 0


def test_engine_is_reused_per_game_and_closed_on_retry(monkeypatch):
    opened, closed, depths, sleeps = [], [], [], []
    class Engine:
        def __init__(self):
            self.index = len(opened)
            opened.append(self)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            closed.append(self)
        def analyse(self, board, limit):
            if self.index == 0:
                raise TimeoutError("transient")
            depths.append(limit.depth)
            return {"score": chess.engine.PovScore(chess.engine.Cp(0), chess.WHITE)}
    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", lambda *a, **kw: Engine())
    monkeypatch.setattr(scoring.time, "sleep", sleeps.append)
    result = scoring.score_fens(fens(), AcplConfig(depth=3, engine_retries=1))
    assert len(opened) == len(closed) == 2 and sleeps == [1]
    assert depths == [3] * 4  # same successful engine across every game position
    assert result["acpl_white"] == 7.5


def test_excluding_book_analyses_starting_position(monkeypatch):
    seen = []
    class Engine:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def analyse(self, board, limit):
            seen.append(board.fen())
            return {"score": chess.engine.PovScore(chess.engine.Cp(100), chess.WHITE)}
    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", lambda *a, **kw: Engine())
    positions = fens()
    result = scoring.score_fens(positions, AcplConfig(include_openings=False), opening_plies=2)
    assert seen == positions[2:]
    assert result["moves_white"] == result["moves_black"] == 1
    assert result["acpl_white"] == result["acpl_black"] == 0


def test_bundled_openings_replay_legally():
    rows, _ = runner.load_openings(AcplConfig())
    assert rows.shape == (496, 24)
    for row in rows:
        board = chess.Board()
        for action in row:
            move = PgxAction.decode(int(action), board)
            assert move in board.legal_moves
            assert PgxAction.encode(move, board) == action
            board.push(move)


def test_disabled_factory_has_no_engine_or_opening_dependencies(monkeypatch):
    monkeypatch.setattr(runner, "load_openings", lambda *a: pytest.fail("disabled"))
    eval_acpl_fn, state = make_acpl({"env_id": "hex", "acpl": {"stockfish_path": "/missing"}}, env=None, policy_fn=None)
    following, metrics = eval_acpl_fn(state, 10, None)
    assert following is state and metrics.metrics == {} and metrics.files == {}


def test_failed_games_do_not_bias_average_or_hide_coverage():
    base = dict(white_wins=0, black_wins=0, draws=1, unfinished=1)
    good = {**base, "analysis_ok": True, "acpl_white": 30., "acpl_black": 20.,
            "stdcpl_white": 4., "stdcpl_black": 2., "moves_white": 2, "moves_black": 3}
    bad = {**base, "analysis_ok": False}
    metrics = runner.summarize([good, bad])
    assert metrics["acpl/acpl_white"] == 30
    assert metrics["acpl/analyzed_games"] == metrics["acpl/failed_games"] == 1
    assert metrics["acpl/scored_moves_white"] == 2
    assert "acpl/acpl_white" not in runner.summarize([bad])
    json.dumps(runner.summarize([bad]), allow_nan=False)


def test_both_cli_parsers_share_acpl_options():
    from nanoalphazero.cli import parse_train_args, parse_muzero_args
    argv = ["--env", "chess", "--acpl.period", "4", "--acpl.num-games", "8",
            "--acpl.depth", "3", "--acpl.no-include-openings"]
    assert acpl_config_dict(acpl_from_args(parse_train_args(argv))) == acpl_config_dict(parse_muzero_args(argv).acpl)


def test_recording_padding_artifacts_and_schedule(tmp_path, monkeypatch):
    from nanoalphazero.config import get_chess_config
    from nanoalphazero.core import make_env, DATA_PARALLEL_SHARDING
    from nanoalphazero.mcts import PolicyOutput
    from nanoalphazero.buffers import unpack_bitmask_vmap
    import jax.numpy as jnp

    config = get_chess_config()
    options = AcplConfig(period=2, num_games=1, max_plies=26, output=tmp_path,
                         stockfish_path=tmp_path / "engine", workers=1)
    options.stockfish_path.touch()
    config["acpl"] = acpl_config_dict(options)
    env = make_env(config)
    def policy(key, state, params, scale, batch):
        legal = unpack_bitmask_vmap(state.legal_action_bitmask)
        return PolicyOutput(action=jnp.argmax(legal, axis=-1), action_weights=legal, visit_counts=legal)
    monkeypatch.setattr(runner, "score_fens", lambda positions, *a: scoring.centipawn_losses(positions, [0] * (len(positions) - 1)))
    eval_acpl_fn, state = make_acpl(config, env=env, policy_fn=policy, data_sharding=DATA_PARALLEL_SHARDING)
    skipped, metrics = eval_acpl_fn(state, 1, {})
    assert skipped is state and metrics.metrics == {} and metrics.files == {}
    following, result = eval_acpl_fn(state, 2, {})
    metrics = result.metrics
    assert set(result.files) == {"games.pgn", "manifest.json", "summary.json"}
    assert all(p.is_file() for p in result.files.values())
    assert state.runs == 0 and following.runs == 1
    assert metrics["acpl/games"] == metrics["acpl/analyzed_games"] == 1
    path = Path(following.last_output)
    assert len(list(path.glob("game-*.pgn"))) == 1
    assert json.loads((path / "placement-0.json").read_text())["padded_games"] == 4
    assert json.loads((path / "scores-0.json").read_text())[0]["plies"] == 26
    assert len(json.loads((path / "game-000000.fens.json").read_text())) == 27
    assert (path / "games.pgn").read_text().count('[Result "*"]') == 1
    repeated, metrics = eval_acpl_fn(following, 2, {})
    assert repeated is following and metrics.metrics == {} and metrics.files == {}


def test_last_move_checkmate_is_recorded_without_reset():
    from nanoalphazero.config import get_chess_config
    from nanoalphazero.core import make_env, DATA_PARALLEL_SHARDING
    from nanoalphazero.eval.acpl.rollout import collect_games
    from nanoalphazero.mcts import PolicyOutput
    import jax.numpy as jnp

    board, opening = chess.Board(), []
    for uci in ("f2f3", "e7e5", "g2g4"):
        move = chess.Move.from_uci(uci)
        opening.append(PgxAction.encode(move, board))
        board.push(move)
    mate = PgxAction.encode(chess.Move.from_uci("d8h4"), board)
    def policy(key, state, params, scale, batch):
        return PolicyOutput(action=jnp.full(batch, mate), action_weights=jnp.zeros((batch, 4672)), visit_counts=jnp.zeros((batch, 4672)))
    games, placement = collect_games(make_env(get_chess_config()), policy, {}, [opening], 8, DATA_PARALLEL_SHARDING)
    assert len(games) == 1 and placement["padded_games"] == 4
    assert games[0].terminated and games[0].board.is_checkmate()
    assert games[0].board.result() == "0-1" and len(games[0].fens) == 5


def test_analysis_remains_parallel_across_games(monkeypatch):
    from types import SimpleNamespace
    import threading

    barrier = threading.Barrier(2)
    threads = set()
    def score(positions, options, opening):
        threads.add(threading.get_ident())
        barrier.wait(timeout=5)
        return scoring.centipawn_losses(positions, [0] * (len(positions) - 1))
    monkeypatch.setattr(runner, "score_fens", score)
    games = [SimpleNamespace(index=i, board=chess.Board(), fens=fens(), opening_plies=2, terminated=False) for i in range(2)]
    rows = runner.analyze_games(games, AcplConfig(workers=2))
    assert len(threads) == 2 and all(row["analysis_ok"] for row in rows)
