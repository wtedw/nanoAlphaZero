"""Inspect policy logits after the cycle-8400 Qh1# missed-mate example."""

import os
os.environ["JAX_PLATFORMS"] = "cpu"

import json
from pathlib import Path
import chess
import chess.pgn
import jax
import jax.numpy as jnp
import numpy as np
import pgx1

from nanoalphazero.ckpt import load_model
from nanoalphazero.core import make_env
from nanoalphazero.eval.chess.action import PgxAction


def main():
    out = Path("artifacts/chess-pgn-audit-cycle8400")
    source = Path(json.loads((out/"report.json").read_text())["summary"]["source"])
    game = chess.pgn.read_game((source/"game-000221.pgn").open())
    board = game.board()
    env = pgx1.make("chess", use_bitmask=True)
    state = env.init(jax.random.PRNGKey(0))
    step = jax.jit(env.step)
    for move in list(game.mainline_moves())[:47]:
        state = step(state, jnp.int32(PgxAction.encode(move,board)))
        board.push(move)
    mate = board.parse_san("Qh1#")
    action = PgxAction.encode(mate,board)
    after = board.copy(); after.push(mate)
    actual = step(state,jnp.int32(action))
    assert after.is_checkmate() and bool(actual.terminated)
    model,params,config = load_model(out/"model-cycle8400.ckpt")
    wrapped = make_env({**config,"enable_sharding":False})
    initial = jax.jit(lambda obs: model.apply({"params":params},obs,method=model.initial))
    recurrent = jax.jit(lambda s,a: model.apply({"params":params},s,a,method=model.recurrent))
    def observe(s):
        b=jax.tree.map(lambda x:x[None],s)
        return wrapped.observe(b,b.current_player)
    latent,_,_ = initial(observe(state))
    terminal_latent,reward,logits,value = recurrent(latent,jnp.asarray([action]))
    _,observed_logits,observed_value = initial(observe(actual))
    chosen=int(jnp.argmax(logits[0]))
    _,r2,_,v2=recurrent(terminal_latent,jnp.asarray([chosen]))
    def inspect(values):
        values=np.asarray(values[0]); probs=np.asarray(jax.nn.softmax(jnp.asarray(values)))
        rows=[]
        for a in np.argsort(values)[-12:][::-1]:
            row=dict(action=int(a),logit=float(values[a]),relative_logit=float(values[a]-values.max()),probability=float(probs[a]))
            try:
                m=PgxAction.decode(int(a),after)
                row.update(uci=m.uci(),legal=m in after.legal_moves,pseudo_legal=after.is_pseudo_legal(m),
                    source_piece=str(after.piece_at(m.from_square)),target_piece=str(after.piece_at(m.to_square)),
                    blockers=[chess.square_name(sq) for sq in chess.SquareSet(chess.between(m.from_square,m.to_square)&after.occupied)])
            except ValueError:
                row.update(uci=None,legal=False,pseudo_legal=False)
            rows.append(row)
        return dict(top=rows,entropy_nats=float(-np.sum(probs*np.log(np.maximum(probs,1e-30)))),
                    pseudo_legal_mass=float(sum(probs[PgxAction.encode(m,after)] for m in after.pseudo_legal_moves)))
    report=dict(checkpoint_cycle=8400,fen_before=board.fen(),fen_after=after.fen(),
        checkmate=after.is_checkmate(),reward=float(reward[0]),terminal_latent_value=float(value[0]),
        post_terminal_reward=float(r2[0]),post_terminal_value=float(v2[0]),
        latent_policy=inspect(logits),observed_terminal_policy=inspect(observed_logits),
        observed_terminal_value=float(observed_value[0]))
    (out/"terminal-policy.json").write_text(json.dumps(report,indent=2))
    np.savez_compressed(out/"terminal-policy-logits.npz",latent_logits=np.asarray(logits),observed_logits=np.asarray(observed_logits))
    print(json.dumps(report,indent=2),flush=True)


if __name__ == "__main__":
    main()
