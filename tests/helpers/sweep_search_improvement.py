"""Frozen CPU search sweep, selection/validation/untouched-game confirmation.

Standalone diagnostic: never changes production search or training configuration.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import functools
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

os.environ["JAX_PLATFORMS"] = "cpu"
import numpy as np

BASE = Path("artifacts/latest-value-scale-20261017")


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def prepare(out):
    import chess.pgn
    old = json.loads((BASE / "positions.json").read_text())
    rng = np.random.default_rng(2026101719)
    order = rng.permutation(192)
    screen = [old[i] for i in order[:32]]
    validation = [old[i] for i in order[32:96]]
    excluded = {r["game"] for r in old}
    # Earlier hand-picked mate probes were outside the random 192-game sample.
    targeted = json.loads(Path("artifacts/dynamics-comparison-20261016/expanded/positions.json").read_text())
    excluded.update(r["game"] for r in targeted)
    games = Path(json.loads(Path("artifacts/dynamics-comparison-20261016/expanded/manifest.json").read_text())["games"])
    paths = [p for p in sorted(games.glob("game-*.pgn")) if p.name not in excluded]
    test = []
    for i in rng.permutation(len(paths)):
        path = paths[i]
        with path.open() as stream:
            game = chess.pgn.read_game(stream)
        moves = list(game.mainline_moves())
        lo, hi = ((24, 40), (40, 80), (80, 160))[len(test) % 3]
        candidates = list(range(lo, min(hi, len(moves))))
        if not candidates:
            continue
        ply = int(rng.choice(candidates))
        test.append(dict(index=288+len(test), game=path.name, ply=ply,
                         moves=[m.uci() for m in moves[:ply]]))
        if len(test) == 96:
            break
    assert len(test) == 96 and not excluded & {r["game"] for r in test}
    grid = [dict(m=m, n=n, scale=s, exponent=1.) for m in (2, 4, 8, 16, 32, 64)
            for n in (4, 8, 16, 32, 64, 128) if n >= m for s in (.01, .03, .1, .3, 1.)]
    for name, rows in (("screen", screen), ("validation", validation), ("test", test)):
        save(out / f"{name}-positions.json", rows)
    save(out / "screen-configs.json", grid)
    save(out / "manifest.json", dict(checkpoint=str(BASE / "checkpoint"), cycle=3600,
        source_generation=json.loads((BASE / "manifest.json").read_text())["generation"],
        seed=2026101719, screen_count=32, validation_count=64, test_count=96,
        grid=grid, gumbel_scale=0., rescale_values=False, maxvisit_init=50.,
        branch="wtedw/mctx:2026mu", branch_commit="6cf1a3918fca41569d43b773fefdc5fba70af506",
        plan="Rank linear screen by mean paired CP gain over prior; validate top four. "
             "If best validation mean gain <= 0 or its paired CI includes 0, run sqrt screen and "
             "validate its top four. Choose one validation winner, freeze it before fresh test. "
             "Always include old m16/n32/scale1 and running m32/n64/scale1 controls. "
             "Use same legal-move Stockfish reference for all settings at each root. "
             "Screen/validation are exploratory; only fresh-game test confirms winner."))


def label(c):
    return f"m{c['m']}-n{c['n']}-s{c['scale']:g}-e{c['exponent']:g}"


def controls():
    return [dict(m=16,n=32,scale=1.,exponent=1.), dict(m=32,n=64,scale=1.,exponent=1.)]


def worker(out, stage, split):
    import jax
    import jax.numpy as jnp
    from compare_chess_dynamics import replay, legal
    from nanoalphazero.ckpt.model import load_model
    from nanoalphazero.core import make_env
    from nanoalphazero.eval.chess.action import PgxAction
    from nanoalphazero.mcts import RootFnOutput, RecurrentFnOutput, make_muzero_mcts
    from nanoalphazero.mcts_exact import gumbel_muzero_policy_node
    from nanoalphazero._vendor.mctx import qtransforms as qt
    from mctx._src.qtransforms import qtransform_completed_by_mix_value as fork_transform

    assert jax.default_backend() == "cpu"
    rank = int(os.environ.get("NANOAZ_POD_PROCESS_ID", "0"))
    count = int(os.environ.get("NANOAZ_POD_PROCESS_COUNT", "1"))
    configs = json.loads((out / f"{stage}-configs.json").read_text())
    shapes = list(dict.fromkeys((c["m"],c["n"]) for c in configs))
    configs = [c for c in configs if shapes.index((c["m"],c["n"])) % count == rank]
    if not configs:
        return
    records = json.loads((out / f"{split}-positions.json").read_text())
    states, boards = replay(records)
    model, params, cfg = load_model(BASE / "checkpoint")
    env = make_env(cfg)

    @jax.jit
    def initial(p, state):
        h, logits, value = model.apply({"params":p}, env.observe(state,state.current_player),method=model.initial)
        if cfg.get("exp_use_root_temperature",False):
            logits = logits / cfg["exp_root_temperature"]
        return RootFnOutput(prior_logits=logits,value=value,embedding=h), legal(state)

    def recurrent(p, key, action, h):
        h, _, logits, value = model.apply({"params":p}, h, action, method=model.recurrent)
        return RecurrentFnOutput(reward=jnp.zeros_like(value), discount=-jnp.ones_like(value),
                                  prior_logits=logits,value=value), h

    def transform(tree, node, scale, exponent):
        # Same mixed completion and multiplication order as 2026mu; alter only
        # visit exponent. Keep max aggregator, no min/max normalization.
        visits = tree.children_visits[node]
        q = tree.qvalues(node)
        mixed = qt._compute_mixed_value(tree.raw_values[node], q, visits,
                                        jax.nn.softmax(tree.children_prior_logits[node]))
        completed = qt._complete_qvalues(q, visit_counts=visits, value=mixed)
        return ((50. + jnp.max(visits)) ** exponent * scale) * completed

    @functools.partial(jax.jit, static_argnames=("m", "n"))
    def run(p, root, mask, scale, exponent, *, m, n):
        return gumbel_muzero_policy_node(p, jax.random.PRNGKey(2026101719),root,recurrent,n,
            invalid_actions=~mask, max_num_considered_actions=m, gumbel_scale=0.,
            qtransform=functools.partial(transform,scale=scale,exponent=exponent),return_tree=False)

    batches = []
    for start in range(0,len(records),4):
        state = jax.tree.map(lambda *x:jnp.stack(x),*states[start:start+4])
        root, mask = initial(params,state)
        batches.append((state,root,mask))
    # Check production baseline and fork formula on an actual search summary.
    state, root, mask = batches[0]
    check = run(params,root,mask,1.,1.,m=16,n=32)
    prod = make_muzero_mcts(cfg,env,model)(jax.random.PRNGKey(2026101719),state,params,0.,4)
    np.testing.assert_array_equal(check.action, prod.action)
    np.testing.assert_allclose(check.action_weights, prod.action_weights, atol=2e-6, rtol=2e-6)
    for exponent in (1.,.5):
        ours = jax.vmap(lambda tree:transform(tree,0,.3,exponent))(check.search_tree)
        fork = jax.vmap(lambda tree:fork_transform(tree,0,value_scale=.3,maxvisit_init=50.,
            rescale_values=False,visit_exponent=exponent))(check.search_tree)
        np.testing.assert_array_equal(ours,fork)
    save(out / f"{stage}-checks-worker-{rank}.json",dict(backend=jax.default_backend(),
        devices=[str(d) for d in jax.devices()],production_match=True,fork_formula_match=True))
    for c in configs:
        destination = out / stage / f"{label(c)}.json"
        if destination.exists():
            continue
        rows = []
        started = time.monotonic()
        for batch_index,(_,root,mask) in enumerate(batches):
            result = run(params,root,mask,c["scale"],c["exponent"],m=c["m"],n=c["n"])
            actions, weights, priors = jax.device_get((result.action,result.action_weights,
                jnp.argmax(jnp.where(mask,root.prior_logits,-jnp.inf),axis=-1)))
            for b in range(len(actions)):
                i = batch_index*4+b
                rows.append(dict(index=records[i]["index"],game=records[i]["game"],
                    move=PgxAction.decode(int(actions[b]),boards[i]).uci(),
                    prior=PgxAction.decode(int(priors[b]),boards[i]).uci(),
                    entropy=float(-np.sum(weights[b]*np.log(np.maximum(weights[b],1e-30)))),
                    max_probability=float(weights[b].max())))
        save(destination,dict(config=c,rows=rows,seconds=time.monotonic()-started))
        print(json.dumps(dict(worker=rank,stage=stage,config=c,seconds=time.monotonic()-started)),flush=True)


def score_position(record):
    import chess
    import chess.engine
    board = chess.Board()
    for move in record["moves"]:
        board.push_uci(move)
    with chess.engine.SimpleEngine.popen_uci("/usr/local/bin/stockfish") as engine:
        engine.configure({"Threads":1,"Hash":64})
        infos = engine.analyse(board,chess.engine.Limit(depth=18),multipv=board.legal_moves.count())
    scores = {i["pv"][0].uci():dict(cp=i["score"].pov(board.turn).score(mate_score=10000),
                                      depth=i["depth"]) for i in infos}
    assert len(scores) == board.legal_moves.count() and min(s["depth"] for s in scores.values()) >= 18
    return dict(index=record["index"],scores=scores)


def reference(out, split):
    dest = out / f"{split}-reference.json"
    if dest.exists():
        return
    rows = json.loads((out / f"{split}-positions.json").read_text())
    results = []
    with ProcessPoolExecutor(max_workers=16) as pool:
        for result in pool.map(score_position,rows):
            results.append(result)
            save(out / f"{split}-reference-partial.json",results)
            print(f"Stockfish {split}: {len(results)}/{len(rows)}",flush=True)
    save(dest,results)


def analyze(out, stage, split):
    from summarize_frozen_dynamics import estimate
    refs = {r["index"]:r["scores"] for r in json.loads((out/f"{split}-reference.json").read_text())}
    configs = json.loads((out/f"{stage}-configs.json").read_text())
    summaries = []
    for c in configs:
        rows = json.loads((out/stage/f"{label(c)}.json").read_text())["rows"]
        assert len(rows) == len(refs) and {r["index"] for r in rows} == refs.keys()
        gains, losses, prior_losses = [], [], []
        for r in rows:
            scores = {move:float(np.clip(v["cp"],-1000,1000)) for move,v in refs[r["index"]].items()}
            gains.append(scores[r["move"]]-scores[r["prior"]])
            losses.append(max(scores.values())-scores[r["move"]])
            prior_losses.append(max(scores.values())-scores[r["prior"]])
        summaries.append(dict(config=c,gain_vs_prior=estimate(gains),loss=estimate(losses),
            prior_loss=estimate(prior_losses),changed=sum(r["move"]!=r["prior"] for r in rows),
            improved=sum(g>0 for g in gains),worsened=sum(g<0 for g in gains),gains=gains))
    summaries.sort(key=lambda s:-s["gain_vs_prior"]["mean"])
    save(out/f"{stage}-summary.json",summaries)
    print(json.dumps(dict(stage=stage,best=summaries[0]),indent=2),flush=True)
    return summaries


def unique(configs):
    return list({label(c):c for c in configs}.values())


def orchestrate(out):
    def stage(name, split):
        # Independent CPU processes, 24 logical CPUs per worker; libtpu is never
        # initialized. The separate training launch remains supervised in tmux.
        cmd = ["uv","run","--no-sync","pod","run","--cpu","--","taskset","-c","24-47",
               "python","-u",__file__,str(out),"worker","--stage",name,"--split",split]
        with (out/f"{name}-launcher.log").open("w") as log:
            subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
        reference(out,split)
        return analyze(out,name,split)
    ranked = stage("screen","screen")
    save(out/"validation-configs.json",unique([r["config"] for r in ranked[:4]]+controls()))
    ranked = stage("validation","validation")
    if ranked[0]["gain_vs_prior"]["ci95"][0] <= 0:
        grid = json.loads((out/"screen-configs.json").read_text())
        save(out/"sqrt-screen-configs.json",[{**c,"exponent":.5} for c in grid])
        sqrt_ranked = stage("sqrt-screen","screen")
        save(out/"sqrt-validation-configs.json",unique([r["config"] for r in sqrt_ranked[:4]]+controls()))
        ranked += stage("sqrt-validation","validation")
        ranked.sort(key=lambda r:-r["gain_vs_prior"]["mean"])
    winner = ranked[0]["config"]
    save(out/"chosen-before-test.json",dict(config=winner,validation=ranked[0]))
    save(out/"test-configs.json",unique([winner]+controls()))
    test = stage("test","test")
    save(out/"COMPLETE.json",dict(winner=winner,test=test,
         stockfish_sha256=hashlib.sha256(Path("/usr/local/bin/stockfish").read_bytes()).hexdigest()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output",type=Path)
    parser.add_argument("phase",choices=["prepare","worker","orchestrate","reference","analyze"])
    parser.add_argument("--stage",default="screen")
    parser.add_argument("--split",default="screen")
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if args.phase=="prepare": prepare(args.output)
    elif args.phase=="worker": worker(args.output,args.stage,args.split)
    elif args.phase=="reference": reference(args.output,args.split)
    elif args.phase=="analyze": analyze(args.output,args.stage,args.split)
    else: orchestrate(args.output)
