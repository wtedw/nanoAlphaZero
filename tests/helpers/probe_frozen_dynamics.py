"""Frozen CPU search/candidate/persistence probes; no training or settings edits."""
import argparse
import functools
import json
import os
from pathlib import Path
import time

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np

from compare_chess_dynamics import replay, legal
from nanoalphazero.ckpt.model import load_model
from nanoalphazero.core import make_env
from nanoalphazero.eval.chess.action import PgxAction
from nanoalphazero.mcts import RootFnOutput, RecurrentFnOutput, make_muzero_mcts
from nanoalphazero.mcts_exact import gumbel_muzero_policy_node
from nanoalphazero._vendor.mctx import qtransforms as qt


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=192)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    base = Path("artifacts/dynamics-comparison-20261016")
    records = [r for r in json.loads((base / "expanded/positions.json").read_text())
               if r["group"] == "sample"][:args.count]
    states, boards = replay(records)
    model, params, cfg = load_model(base / "checkpoint")
    env = make_env(cfg)
    transform = functools.partial(qt.qtransform_completed_by_mix_value,
        value_scale=cfg["mcts_value_scale"], maxvisit_init=cfg["mcts_maxvisit_init"],
        rescale_values=cfg["mcts_rescale_values"])
    production = make_muzero_mcts(cfg, env, model)

    def recurrent(p, key, action, latent):
        del key
        latent, _, logits, value = model.apply({"params": p}, latent, action, method=model.recurrent)
        return RecurrentFnOutput(reward=jnp.zeros_like(value), discount=-jnp.ones_like(value),
            prior_logits=logits, value=value), latent

    @jax.jit
    def run(p, state, key, noise):
        latent, logits, value = model.apply({"params": p}, env.observe(state, state.current_player),
                                             method=model.initial)
        mask = legal(state)
        if cfg.get("exp_use_root_temperature", False):
            logits = logits / cfg["exp_root_temperature"]
        root = RootFnOutput(prior_logits=logits, value=value, embedding=latent)
        output = gumbel_muzero_policy_node(p, key, root, recurrent,
            num_simulations=cfg["mcts_num_simulations"], invalid_actions=~mask,
            max_num_considered_actions=cfg["mcts_num_root_considered"],
            gumbel_scale=noise, max_depth=None, qtransform=transform, return_tree=True)
        tree = output.search_tree
        transformed = jax.vmap(transform, in_axes=(0, None))(tree, 0)
        return dict(action=output.action, weights=output.action_weights, root_value=value,
            parents=tree.parents, actions=tree.action_from_parent, raw_values=tree.raw_values,
            visits=tree.node_visits, prior_logits=logits, legal=mask,
            root_q=jax.vmap(lambda t: t.qvalues(0))(tree), transformed=transformed)

    @jax.jit
    def fresh(p, state, actions):
        child = env.step(state, actions)
        value = model.apply({"params": p}, env.observe(child, child.current_player),
                             method=model.initial)[2]
        reward = jnp.take_along_axis(child.rewards, state.current_player[:, None], axis=1)[:, 0]
        return value, child.terminated | child.truncated, reward

    manifest = dict(checkpoint=str(base / "checkpoint"),
        generation=json.loads((base / "checkpoint/LATEST").read_text()), config=cfg,
        root_count=len(records), batch=4, noise_scales=[0., cfg["mcts_gumbel_scale"]],
        max_depth=None, return_tree=True, platform="cpu", seed=20261017)
    save(args.output / "manifest.json", manifest)
    output = []
    for start in range(0, len(records), 4):
        now = time.monotonic()
        indices = list(range(start, min(start + 4, len(records))))
        indices += [indices[-1]] * (4 - len(indices))
        batch = jax.tree.map(lambda *x: jnp.stack(x), *[states[i] for i in indices])
        key = jax.random.fold_in(jax.random.PRNGKey(20261017), start)
        for noise in (0., cfg["mcts_gumbel_scale"]):
            result = jax.device_get(run(params, batch, key, noise))
            if start == 0:
                expected = production(key, batch, params, noise, 4)
                np.testing.assert_array_equal(result["action"], expected.action)
                np.testing.assert_allclose(result["weights"], expected.action_weights, rtol=2e-6, atol=2e-6)
            for b, index in enumerate(indices[:min(4, len(records)-start)]):
                row = records[index]
                parents = result["parents"][b]
                depths = np.zeros(len(parents), int)
                for n in range(1, len(parents)):
                    assert 0 <= parents[n] < n
                    depths[n] = depths[parents[n]] + 1
                nodes = np.flatnonzero(parents == 0)
                actions = result["actions"][b, nodes]
                assert np.asarray(result["legal"][b])[actions].all()
                padded_actions = np.pad(actions, (0, 16-len(actions)), mode="edge")
                children = jax.tree.map(lambda x: jnp.repeat(x[None], 16, axis=0), states[index])
                fv, done, rewards = jax.device_get(fresh(params, children, jnp.array(padded_actions)))
                valid = result["legal"][b]
                weights = result["weights"][b, valid]
                logits = result["prior_logits"][b, valid]
                probs = np.exp(logits-logits.max()); probs /= probs.sum()
                rank = np.argsort(np.argsort(-result["prior_logits"][b])) + 1
                # Legal rank, excluding any unmasked illegal logits.
                ranked_actions = np.flatnonzero(valid)[np.argsort(-logits)]
                ranks = {int(a): i+1 for i,a in enumerate(ranked_actions)}
                candidates = []
                for j, (node, action) in enumerate(zip(nodes, actions)):
                    candidates.append(dict(action=int(action), move=PgxAction.decode(int(action), boards[index]).uci(),
                        prior_rank=ranks[int(action)], recurrent_q=float(-result["raw_values"][b, node]),
                        fresh_q=float(-fv[j]), terminal=bool(done[j]), terminal_reward=float(rewards[j]),
                        backed_up_q=float(result["root_q"][b, action]),
                        visits=int(result["visits"][b, node]), selected=bool(action == result["action"][b]),
                        target_probability=float(result["weights"][b, action])))
                output.append(dict(index=index, game=row["game"], ply=row["ply"], moves=row["moves"],
                    fen=row["fen"], noise_scale=noise, root_value=float(result["root_value"][b]),
                    selected_action=int(result["action"][b]), candidate_count=len(candidates), candidates=candidates,
                    depths=depths[1:].tolist(), parents=parents.tolist(),
                    depth_counts=np.bincount(depths[1:], minlength=33).tolist(),
                    target_entropy=float(-np.sum(weights*np.log(np.maximum(weights, 1e-30)))),
                    target_max=float(weights.max()), prior_entropy=float(-np.sum(probs*np.log(np.maximum(probs,1e-30)))),
                    target_kl_prior=float(np.sum(weights*(np.log(np.maximum(weights,1e-30))-np.log(np.maximum(probs,1e-30))))),
                    prior_span=float(np.ptp(logits)), candidate_prior_span=float(np.ptp(result["prior_logits"][b, actions])),
                    transformed_span=float(np.ptp(result["transformed"][b, valid])),
                    candidate_q_span=float(np.ptp(result["root_q"][b, actions])),
                    retained_top128=float(np.sort(weights)[-128:].sum())))
        save(args.output / "search.json", output)
        print(f"search {min(start+4,len(records))}/{len(records)} {time.monotonic()-now:.1f}s", flush=True)

    # Preserve per-window values rather than just means, for independent analysis.
    data = np.load("artifacts/value-fit-tpu-20261017/dataset.npz")
    observations, actions = data["observations"], data["actions"]

    @jax.jit
    def sequence(p, obs, act):
        latent, _, root_value = model.apply({"params": p}, obs[:, 0], method=model.initial)
        rolled, fresh_values, single = [root_value], [root_value], [root_value]
        for k in range(5):
            latent, _, _, value = model.apply({"params": p}, latent, act[:, k], method=model.recurrent)
            real_latent, _, real_value = model.apply({"params": p}, obs[:, k], method=model.initial)
            _, _, _, one = model.apply({"params": p}, real_latent, act[:, k], method=model.recurrent)
            _, _, next_value = model.apply({"params": p}, obs[:, k+1], method=model.initial)
            rolled.append(value); fresh_values.append(next_value); single.append(one)
        return jnp.stack(rolled,1), jnp.stack(fresh_values,1), jnp.stack(single,1)

    chunks = []
    for start in range(0, len(actions), 8):
        chunks.append(jax.device_get(sequence(params, observations[start:start+8], actions[start:start+8])))
        print(f"persistence {start+8}/{len(actions)}", flush=True)
    np.savez_compressed(args.output / "persistence.npz",
        recurrent=np.concatenate([x[0] for x in chunks]), fresh=np.concatenate([x[1] for x in chunks]),
        one_step=np.concatenate([x[2] for x in chunks]), targets=data["targets"])
    print("Frozen probes completed; production equivalence passed at both noise scales", flush=True)


if __name__ == "__main__":
    main()
