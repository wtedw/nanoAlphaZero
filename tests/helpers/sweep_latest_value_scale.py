"""CPU-only frozen scale selection, held-out evaluation, and feature extraction."""
import argparse
import functools
import json
import os
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
import chess
import chess.pgn
import jax
import jax.numpy as jnp
import numpy as np
import zstandard

from compare_chess_dynamics import replay, legal
from nanoalphazero.ckpt.model import load_model
from nanoalphazero.core import make_env
from nanoalphazero.eval.chess.action import PgxAction
from nanoalphazero.mcts import RootFnOutput, RecurrentFnOutput, make_muzero_mcts
from nanoalphazero.mcts_exact import search
from nanoalphazero.model import _ACTS


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def prepare(output):
    source = Path("artifacts/chess-pod-exact-m16-n32-local-gather-20261015/latest.ckpt")
    pointer = json.loads((source / "LATEST").read_text())
    generation = source / pointer["generation"]
    manifest = json.loads((generation / "manifest.json").read_text())
    field = next(f for f in manifest["fields"] if f["path"] == "['cycle']")
    chunk = field["shards"][0]["chunks"][0]
    cycle = int(np.frombuffer(zstandard.ZstdDecompressor().decompress(
        (generation / chunk["file"]).read_bytes()), dtype=np.int32)[0])
    assert cycle == 3600
    snapshot = output / "checkpoint"
    target = snapshot / pointer["generation"]
    target.mkdir(parents=True, exist_ok=True)
    for file in generation.iterdir():
        if file.is_file() and not (target / file.name).exists():
            os.link(file, target / file.name)
    save(snapshot / "LATEST", pointer)
    base = Path("artifacts/dynamics-comparison-20261016/expanded")
    old = [r for r in json.loads((base / "positions.json").read_text()) if r["group"] == "sample"]
    assert len(old) == 192
    games = Path(json.loads((base / "manifest.json").read_text())["games"])
    excluded = {r["game"] for r in old}
    paths = [p for p in sorted(games.glob("game-*.pgn")) if p.name not in excluded]
    rng = np.random.default_rng(2026101702)
    new = []
    for i in rng.permutation(len(paths)):
        path = paths[i]
        with path.open() as stream:
            game = chess.pgn.read_game(stream)
        moves = list(game.mainline_moves())
        lo, hi = ((24, 40), (40, 80), (80, 160))[len(new) % 3]
        candidates = list(range(lo, min(hi, len(moves))))
        if not candidates:
            continue
        ply = int(rng.choice(candidates))
        new.append(dict(game=path.name, ply=ply, moves=[m.uci() for m in moves[:ply]], group="heldout"))
        if len(new) == 96:
            break
    assert len(new) == 96 and not excluded & {r["game"] for r in new}
    rows = old + new
    for i, row in enumerate(rows):
        row["index"] = i
        row["probe_split"] = "train" if i < 128 else ("validation" if i < 192 else "test")
    save(output / "positions.json", rows)
    save(output / "manifest.json", dict(cycle=cycle, generation=pointer,
        checkpoint=str(snapshot), source=str(source), scales=[0., .05, .1, .2, .3, 1.],
        sweep_selection_indices=list(range(96)), heldout_indices=list(range(192,288)),
        probe_train_indices=list(range(128)), probe_validation_indices=list(range(128,192)),
        gumbel_scale=0., seed=2026101702, config=manifest["config"],
        selection_rule="Minimum mean clipped candidate-relative CP loss on selection roots; tie prefers larger scale. Holdout never selects scale."))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--phase", choices=["prepare", "selection", "test", "features"], required=True)
    parser.add_argument("--positive-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.phase == "prepare":
        prepare(args.output)
        print("Pinned cycle 3600; declared disjoint selection/test games and probe splits", flush=True)
        return
    manifest = json.loads((args.output / "manifest.json").read_text())
    records = json.loads((args.output / "positions.json").read_text())
    if args.phase == "selection":
        records = records[:96]
        scales = manifest["scales"]
    elif args.phase == "test":
        records = records[192:]
        choice = json.loads((args.output / "chosen-scale.json").read_text())
        chosen = choice["value_scale"]
        scales = [choice["best_positive_scale"]] if args.positive_only else sorted(set([chosen, 1.]))
    else:
        scales = []
    states, boards = replay(records)
    model, params, cfg = load_model(args.output / "checkpoint")
    env = make_env(cfg)

    def recurrent(p, key, action, latent):
        del key
        latent, _, logits, value = model.apply({"params": p}, latent, action, method=model.recurrent)
        return RecurrentFnOutput(reward=jnp.zeros_like(value), discount=-jnp.ones_like(value),
                                  prior_logits=logits, value=value), latent

    @jax.jit
    def run(p, state, key, scale):
        latent, logits, value = model.apply({"params": p}, env.observe(state, state.current_player), method=model.initial)
        mask = legal(state)
        if cfg.get("exp_use_root_temperature", False):
            logits = logits / cfg["exp_root_temperature"]
        result = search(p, key, RootFnOutput(prior_logits=logits, value=value, embedding=latent), recurrent,
            num_simulations=32, num_root_considered=16, invalid_actions=~mask, gumbel_scale=0.,
            value_scale=scale, maxvisit_init=cfg["mcts_maxvisit_init"], rescale_values=cfg["mcts_rescale_values"])
        return result.action, jnp.argmax(jnp.where(mask,logits,-jnp.inf),-1), result.action_weights

    if args.phase != "features":
        output = []
        production = make_muzero_mcts(cfg, env, model)
        for start in range(0,len(records),4):
            chunk = states[start:start+4]
            batch = jax.tree.map(lambda *x: jnp.stack(x), *chunk)
            key = jax.random.fold_in(jax.random.PRNGKey(2026101702),start)
            rows = [dict(**r, results={}) for r in records[start:start+4]]
            for scale in scales:
                actions, priors, weights = jax.device_get(run(params,batch,key,scale))
                if start == 0 and scale == 1.:
                    baseline = production(key,batch,params,0.,4)
                    np.testing.assert_array_equal(actions,baseline.action)
                    np.testing.assert_allclose(weights,baseline.action_weights,atol=2e-6,rtol=2e-6)
                for b,row in enumerate(rows):
                    row["policy_move"] = PgxAction.decode(int(priors[b]),boards[start+b]).uci()
                    row["results"][str(scale)] = dict(move=PgxAction.decode(int(actions[b]),boards[start+b]).uci(),
                        entropy=float(-np.sum(weights[b]*np.log(np.maximum(weights[b],1e-30)))),
                        max_probability=float(weights[b].max()))
            output.extend(rows)
            prefix = "positive-test" if args.positive_only else args.phase
            save(args.output / f"{prefix}-search.json",output)
            print(f"{args.phase} {start+4}/{len(records)}",flush=True)
        return

    def read_penultimate(intermediates):
        # Captured Dense_0 is preactivation; match the frozen ValueHead activation.
        return _ACTS[cfg["katago_activation"]](intermediates["prediction"]["ValueHead_0"]["Dense_0"]["__call__"][0])

    def capture(module, method):
        return method == "__call__" and module.name == "Dense_0" and "ValueHead_0" in module.scope.path

    def pooled(latent):
        # Retain coarse spatial structure, plus channel mean/std; no fitted projection.
        tiles = latent.reshape(latent.shape[0],2,4,2,4,latent.shape[-1]).mean((2,4)).reshape(latent.shape[0],-1)
        return jnp.concatenate([tiles,latent.mean((1,2)),latent.std((1,2))],-1)

    @jax.jit
    def initial(p,state):
        return model.apply({"params":p},env.observe(state,state.current_player),method=model.initial)

    @jax.jit
    def features(p,state,actions):
        latent,_,root_value = initial(p,state)
        (rec,_,_,rv), ri = model.apply({"params":p},latent,actions,method=model.recurrent,
            capture_intermediates=capture, mutable=["intermediates"])
        child=env.step(state,actions)
        (real,_,fv), fi = model.apply({"params":p},env.observe(child,child.current_player),method=model.initial,
            capture_intermediates=capture, mutable=["intermediates"])
        return dict(rec_pen=read_penultimate(ri["intermediates"]),real_pen=read_penultimate(fi["intermediates"]),
            rec_pool=pooled(rec),real_pool=pooled(real),root_pool=pooled(latent),
            rec_spatial=rec.reshape(rec.shape[0],-1).astype(jnp.float16),
            real_spatial=real.reshape(real.shape[0],-1).astype(jnp.float16),
            rec_q=-rv,fresh_q=-fv,root_value=root_value,terminal=child.terminated | child.truncated)

    output, arrays = [], []
    for index,(record,state,board) in enumerate(zip(records,states,boards)):
        batch=jax.tree.map(lambda x:x[None],state)
        _,logits,_=jax.device_get(initial(params,batch))
        mask=np.asarray(legal(batch))[0]
        actions=np.flatnonzero(mask)[np.argsort(-logits[0,mask])[:16]]
        padded=np.pad(actions,(0,16-len(actions)),mode="edge")
        batch=jax.tree.map(lambda x:jnp.repeat(x[None],16,axis=0),state)
        values=jax.device_get(features(params,batch,jnp.array(padded)))
        arrays.append({k:v[:len(actions)] for k,v in values.items()})
        for j,action in enumerate(actions):
            move=PgxAction.decode(int(action),board)
            output.append(dict(index=index,game=record["game"],split=record["probe_split"],
                moves=record["moves"],move=move.uci(),action=int(action),prior_rank=j+1,
                from_square=move.from_square if board.turn else move.from_square^56,
                to_square=move.to_square if board.turn else move.to_square^56,
                promotion=move.promotion or 0))
        if index % 16 == 15:
            print(f"features {index+1}/{len(records)}",flush=True)
    np.savez_compressed(args.output / "features.npz",**{k:np.concatenate([a[k] for a in arrays]) for k in arrays[0]})
    save(args.output / "feature-rows.json",output)
    print("Feature extraction complete",flush=True)


if __name__ == "__main__":
    main()
