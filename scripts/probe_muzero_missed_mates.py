"""Replay all missed mates from a PGN audit with the corresponding saved model."""

import os
os.environ["JAX_PLATFORMS"] = "cpu"

import json
import argparse
from pathlib import Path

import chess
import chess.pgn
import jax
import jax.numpy as jnp
import numpy as np
import pgx1
import pgx1.chess as pc

from nanoalphazero.ckpt import load_model
from nanoalphazero.core import make_env
from nanoalphazero.eval.chess.action import PgxAction
from nanoalphazero.mcts import make_muzero_mcts, final_qtransform_completed_by_mix_value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    args = parser.parse_args()
    out = args.audit
    audit = json.loads((out/"report.json").read_text())
    source = Path(audit["summary"]["source"])
    env = pgx1.make("chess", use_bitmask=True)
    step = jax.jit(env.step)
    cases, states, boards, candidates = [], [], [], []
    for game in audit["games"]:
        events = {e["ply"]: e for e in game["events"] if e["kind"] == "missed_mate_in_one"}
        if not events:
            continue
        pgn = chess.pgn.read_game((source/game["game"]).open())
        board, state = pgn.board(), env.init(jax.random.PRNGKey(0))
        for ply, move in enumerate(pgn.mainline_moves(), 1):
            if ply in events:
                e = events[ply]
                idx = len(cases)
                cases.append(e); states.append(state); boards.append(board.copy())
                candidates.append((idx, "played", move))
                for san in e["mating_moves"]:
                    mate = board.parse_san(san)
                    after = board.copy(); after.push(mate)
                    assert after.is_checkmate()
                    candidates.append((idx, "mate", mate))
            state = step(state, jnp.int32(PgxAction.encode(move, board)))
            board.push(move)
    model, params, config = load_model(args.checkpoint)
    wrapped = make_env({**config, "enable_sharding": False})
    batch = jax.tree.map(lambda *xs: jnp.stack(xs), *states)
    legal = jax.vmap(pc.unpack_bitmask)(batch.legal_action_bitmask)
    for i, b in enumerate(boards):
        assert set(np.flatnonzero(np.asarray(legal[i]))) == {PgxAction.encode(m,b) for m in b.legal_moves}
    print(f"Replayed {len(cases)} positions with complete PGN histories", flush=True)
    search = make_muzero_mcts(config, wrapped, model, return_diagnostics=True)
    result = jax.device_get(search(jax.random.PRNGKey(42), batch, params, 0., len(cases)))
    print("Search completed", flush=True)
    initial = jax.jit(lambda obs: model.apply({"params":params}, obs, method=model.initial))
    recur = jax.jit(lambda s,a: model.apply({"params":params}, s,a, method=model.recurrent))
    latent, logits, value = initial(wrapped.observe(batch, batch.current_player))
    parents = jnp.asarray([i for i,_,_ in candidates])
    actions = jnp.asarray([PgxAction.encode(m,boards[i]) for i,_,m in candidates])
    child, r1, l1, v1 = recur(latent[parents], actions)
    reply = jnp.argmax(l1, axis=-1)
    _, r2, _, v2 = recur(child,reply)
    arrays = jax.device_get((r1,v1,r2,v2,reply))
    prior = np.asarray(jax.nn.softmax(jnp.where(legal,logits,-jnp.inf)))
    rows = [{**e,"root_value":float(value[i]), "selected":boards[i].san(PgxAction.decode(int(result.action[i]),boards[i])),
             "reproduced":int(result.action[i])==PgxAction.encode(boards[i].parse_san(e["san"]),boards[i]),
             "candidates":[]} for i,e in enumerate(cases)]
    for j,(i,kind,move) in enumerate(candidates):
        a=int(actions[j]);reward,terminal_v,tail_r,tail_v,reply_action=[x[j] for x in arrays]
        after=boards[i].copy();after.push(move)
        try:
            reply_uci=PgxAction.decode(int(reply_action),after).uci()
        except ValueError:
            reply_uci=None
        q1=float(reward-terminal_v);q2=float(reward-tail_r+tail_v)
        visits=float(result.visit_counts[i,a])
        expected=(q1+q2)/(2+1e-6) if visits==2 else q1
        if visits:
            np.testing.assert_allclose(result.diagnostics["backed_up_q"][i,a],expected,atol=3e-5)
        rows[i]["candidates"].append(dict(kind=kind,san=boards[i].san(move),action=a,
            prior=float(prior[i,a]),prior_rank=int(np.sum(prior[i]>prior[i,a]))+1,
            visits=visits,reward=float(reward),successor_value=float(terminal_v),
            post_terminal_reward=float(tail_r),post_terminal_value=float(tail_v),
            imagined_reply_uci=reply_uci,q1=q1,q2=q2,
            q_search=float(result.diagnostics["backed_up_q"][i,a]) if visits else None,
            score=float(result.diagnostics["final_score"][i,a]),
            prior_logit=float(result.diagnostics["prior_logit"][i,a])))
    # Keep the first rung and survivors fixed. Correct only the continuation
    # past known mates: value-only needs next V=+1, reward mode needs r=V=0.
    corrected_q = np.array(result.diagnostics["backed_up_q"], copy=True)
    for i, row in enumerate(rows):
        for c in row["candidates"]:
            if c["kind"] == "mate" and c["visits"] == 2:
                target_q2 = 1. if config.get('muzero_predict_value_only', False) else c['reward']
                corrected_q[i,c["action"]] = (c["q1"]+target_q2)/(2+1e-6)
    def finish(q):
        transformed = jax.vmap(lambda q,v,p,n: final_qtransform_completed_by_mix_value(
            q,v,p,n,value_scale=config.get("mcts_value_scale",1.),
            maxvisit_init=config.get("mcts_maxvisit_init",50.),
            rescale_values=config.get("mcts_rescale_values",False)))(
                jnp.asarray(q), jnp.asarray(result.diagnostics["raw_value"]),
                jnp.asarray(result.diagnostics["prior_logit"]),jnp.asarray(result.visit_counts))
        scores = transformed + result.diagnostics["prior_logit"]
        return np.asarray(jnp.argmax(jnp.where(legal,scores,-jnp.inf),axis=-1))
    np.testing.assert_array_equal(finish(result.diagnostics["backed_up_q"]),result.action)
    corrected_actions=finish(corrected_q)
    for i,row in enumerate(rows):
        a=int(corrected_actions[i])
        row["corrected_continuation_selection"]=boards[i].san(PgxAction.decode(a,boards[i]))
        row["corrected_continuation_selects_mate"]=any(c["kind"]=="mate" and c["action"]==a for c in row["candidates"])
    report=dict(checkpoint=str(args.checkpoint), positions=len(rows),reproduced=sum(x["reproduced"] for x in rows),
                counterfactual="Correct absorbing continuation one step beyond known mate, unchanged first-rung predictions and survivors; diagnostic oracle, not a deployable search fix",
                rows=rows)
    (out/"missed-mate-search.json").write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)


if __name__ == "__main__":
    main()
