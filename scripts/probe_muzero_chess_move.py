"""Inspect a saved MuZero search using the complete history of a PGN."""

import argparse
import json
import os
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"

import chess
import chess.engine
import chess.pgn
import jax
import jax.numpy as jnp
import numpy as np
import pgx1
import pgx1.chess as pc

from nanoalphazero.ckpt import load_model
from nanoalphazero.core import make_env
from nanoalphazero.eval.chess.action import PgxAction
from nanoalphazero.mcts import make_muzero_mcts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pgn", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ply", type=int, default=88, help="Number of plies before search")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    game = chess.pgn.read_game(args.pgn.open())
    moves = list(game.mainline_moves())
    board = game.board()
    env = pgx1.make("chess", use_bitmask=True)
    state = env.init(jax.random.PRNGKey(0))
    step = jax.jit(env.step)
    for move in moves[:args.ply]:
        state = step(state, jnp.int32(PgxAction.encode(move, board)))
        board.push(move)
    legal = np.flatnonzero(np.asarray(pc.unpack_bitmask(state.legal_action_bitmask)))
    assert set(legal) == {PgxAction.encode(m, board) for m in board.legal_moves}
    assert not bool(state.terminated)
    played = PgxAction.encode(moves[args.ply], board)
    model, params, config = load_model(args.checkpoint)
    print("Loaded model and replayed history", flush=True)
    wrapped = make_env({**config, "enable_sharding": False})
    batch = jax.tree.map(lambda x: x[None], state)
    search = make_muzero_mcts(config, wrapped, model, return_diagnostics=True)
    result = jax.device_get(search(jax.random.PRNGKey(42), batch, params, 0., 1))
    print("Reproduced search", flush=True)

    def describe(action, position):
        try:
            move = PgxAction.decode(int(action), position)
            legal_move = move in position.legal_moves
            return dict(action=int(action), uci=move.uci(), legal=legal_move,
                        san=position.san(move) if legal_move else None)
        except ValueError:
            return dict(action=int(action), uci=None, legal=False, san=None)

    initial = jax.jit(lambda obs: model.apply({"params": params}, obs, method=model.initial))
    recurrent = jax.jit(lambda s, a: model.apply({"params": params}, s, a, method=model.recurrent))
    latent, logits, value = initial(wrapped.observe(batch, batch.current_player))
    child, reward, child_logits, child_value = recurrent(
        jnp.repeat(latent, len(legal), axis=0), jnp.asarray(legal))
    responses = jnp.argmax(child_logits, axis=-1)
    _, reply_reward, _, reply_value = recurrent(child, responses)
    actual = jax.vmap(env.step)(jax.tree.map(lambda x: jnp.repeat(x[None], len(legal), axis=0), state),
                               jnp.asarray(legal))
    _, actual_logits, actual_value = initial(wrapped.observe(actual, actual.current_player))
    raw_prior = np.asarray(jax.nn.softmax(jnp.where(
        jax.vmap(pc.unpack_bitmask)(batch.legal_action_bitmask), logits, -jnp.inf)))[0]
    rows = []
    for i, a in enumerate(legal):
        after = board.copy(); after.push(PgxAction.decode(int(a), board))
        r1, v1 = float(reward[i]), float(child_value[i])
        r2, v2 = float(reply_reward[i]), float(reply_value[i])
        row = {**describe(a, board), "played": int(a) == played, "prior": float(raw_prior[a]),
               "visits": float(result.visit_counts[0, a]),
               "reward": r1, "latent_opponent_value": v1, "one_step_q": r1-v1,
               "actual_opponent_value": float(actual_value[i]),
               "imagined_reply": describe(responses[i], after),
               "reply_reward": r2, "reply_value": v2, "two_step_q": r1-r2+v2,
               "diagnostics": {k: float(v[0, a]) for k, v in result.diagnostics.items() if v.ndim == 2}}
        if int(a) == played:
            reply = moves[args.ply + 1]
            reply_action = PgxAction.encode(reply, after)
            _, forced_r, _, forced_v = recurrent(child[i:i+1], jnp.asarray([reply_action]))
            actual_after = jax.tree.map(lambda x: x[i], actual)
            after_reply = step(actual_after, jnp.int32(reply_action))
            ab = jax.tree.map(lambda x: x[None], after_reply)
            _, _, real_v2 = initial(wrapped.observe(ab, ab.current_player))
            row["actual_played_reply"] = {**describe(reply_action, after),
                "latent_policy_probability": float(jax.nn.softmax(child_logits[i])[reply_action]),
                "latent_policy_rank": int(jnp.sum(child_logits[i] > child_logits[i, reply_action])) + 1,
                "observed_policy_probability": float(jax.nn.softmax(actual_logits[i])[reply_action]),
                "observed_policy_rank": int(jnp.sum(actual_logits[i] > actual_logits[i, reply_action])) + 1,
                "reward": float(forced_r[0]), "latent_value": float(forced_v[0]),
                "observed_value": float(real_v2[0]), "two_step_q": r1-float(forced_r[0])+float(forced_v[0])}
            row["latent_reply_top5"] = [
                {**describe(a2, after), "probability": float(jax.nn.softmax(child_logits[i])[a2])}
                for a2 in np.argsort(np.asarray(child_logits[i]))[-5:][::-1]]
        rows.append(row)
    rows.sort(key=lambda r: r["diagnostics"]["final_score"], reverse=True)
    report = dict(checkpoint=str(args.checkpoint), ply=args.ply, fen=board.fen(),
                  encoding=config.get("muzero_chess_action_encoding", "planes73"),
                  selected=describe(result.action[0], board), recorded=describe(played, board),
                  root_value=float(value[0]), actions=rows)
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k != "actions"}), flush=True)
    print(json.dumps(next(r for r in rows if r["played"]), indent=2), flush=True)
    with chess.engine.SimpleEngine.popen_uci("/usr/local/bin/stockfish") as engine:
        analysis = engine.analyse(board, chess.engine.Limit(depth=18), multipv=5)
        report["stockfish"] = [{"score_white": str(x["score"].white()),
                                "pv": board.variation_san(x["pv"])} for x in analysis]
        forced = engine.analyse(board, chess.engine.Limit(depth=18), root_moves=[moves[args.ply]])
        report["stockfish_played"] = dict(score_white=str(forced["score"].white()),
                                           pv=board.variation_san(forced["pv"]))
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k.startswith("stockfish")}), flush=True)


if __name__ == "__main__":
    main()
