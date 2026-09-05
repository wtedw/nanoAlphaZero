"""Evaluation-only exact Hex 4 decisions on varied held-out midgame positions."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from nanoalphazero.research.muzero.search import make_search


def summarize(records):
    winning = [r for r in records if r["root_value"] == 1]
    result = {"positions": len(records), "winning_positions": len(winning)}
    for mode in ("policy", "search"):
        result[f"{mode}_winning_preserved"] = sum(r[f"{mode}_action_value"] == 1 for r in winning)
        result[f"{mode}_winning_accuracy"] = result[f"{mode}_winning_preserved"] / max(len(winning), 1)
        result[f"{mode}_action_value_mean"] = float(np.mean([r[f"{mode}_action_value"] for r in records]))
        result[f"{mode}_q_mse"] = float(np.mean([(r[f"{mode}_q"] - r[f"{mode}_action_value"]) ** 2 for r in records]))
        result[f"{mode}_reward_mse"] = float(np.mean([(r[f"{mode}_predicted_reward"] - r[f"{mode}_true_reward"]) ** 2 for r in records]))
    result["search_fixes_policy"] = sum(r["search_action_value"] > r["policy_action_value"] for r in winning)
    result["search_breaks_policy"] = sum(r["search_action_value"] < r["policy_action_value"] for r in winning)
    result["predicted_q_gain"] = float(np.mean([r["search_q"] - r["policy_q"] for r in records]))
    result["true_action_value_gain"] = float(np.mean([r["search_action_value"] - r["policy_action_value"] for r in records]))
    result["root_value_mse"] = float(np.mean([(r["predicted_root_value"] - r["root_value"]) ** 2 for r in records]))
    return result


def evaluate_decisions(model, params, env, config, pool, output, positions=256, seed=840017):
    """No solver results leave this evaluation routine for training or search.

    Uses explicit DFPN proofs, not MoHex's move-selection strength. Restrict
    this baseline to Hex 4; larger-board exact-solve cost is not assumed safe.
    """
    if config["env"] != "hex4":
        raise ValueError("Exact midgame decision evaluation is currently Hex 4 only")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    batch = len(pool.engines)
    rng = np.random.default_rng(seed)
    records, seen = [], set()
    search = make_search(model, env, config)
    policy = make_search(model, env, config, policy_only=True)
    initial = jax.jit(lambda obs: model.apply({"params": params}, obs, method=model.initial))
    recurrent = jax.jit(lambda latent, action: model.apply({"params": params}, latent, action, method=model.recurrent))

    def solve(engine, history):
        engine.clear()
        for ply, action in enumerate(history):
            engine.play("b" if ply % 2 == 0 else "w", int(action))
        response = engine.query(f"dfpn-solve-state {'b' if len(history) % 2 == 0 else 'w'}").strip().lower()
        if response not in ("black", "white"):
            raise RuntimeError(f"DFPN did not prove a winner: {response!r}")
        return 0 if response == "black" else 1

    with ThreadPoolExecutor(max_workers=batch) as workers:
        for group in range(10000):
            if len(records) >= positions:
                break
            depth = 2 + group % 9
            histories = np.stack([rng.permutation(16)[:depth] for _ in range(batch)])
            state = env.init_dummy_estate(batch)
            for ply in range(depth):
                state = env.step(state, jnp.asarray(histories[:, ply]))
            observation = env.observe(state, state.current_player)
            observed = np.asarray(observation)
            active = []
            for i in range(batch):
                identity = observed[i].tobytes()
                if bool(state.terminated[i]) or identity in seen:
                    continue
                if len(records) + len(active) >= positions:
                    break
                seen.add(identity)
                active.append(i)
            if not active:
                continue
            actor = depth % 2
            root_futures = {i: workers.submit(solve, pool.engines[i], histories[i]) for i in active}
            roots = {i: future.result() for i, future in root_futures.items()}
            latent, _, values = initial(observation)
            actions = {"policy": policy(jax.random.PRNGKey(0), state, params, 0., batch).action,
                       "search": search(jax.random.PRNGKey(0), state, params, 0., batch).action}
            rows = {i: {"history": histories[i].tolist(), "depth": depth, "actor": actor,
                        "root_value": 1 if roots[i] == actor else -1,
                        "predicted_root_value": float(values[i])} for i in active}
            for mode, action in actions.items():
                if not bool(jnp.all(state.legal_action_mask[jnp.arange(batch), action])):
                    raise RuntimeError("Illegal real decision-evaluation action")
                _, reward, _, next_value = recurrent(latent, action)
                following = env.step(state, action)
                pending = {i: workers.submit(solve, pool.engines[i], [*histories[i], int(action[i])])
                           for i in active if not bool(following.terminated[i])}
                for i in active:
                    actual_reward = float(following.rewards[i, actor])
                    if bool(following.terminated[i]):
                        action_value = int(np.sign(actual_reward))
                    else:
                        action_value = 1 if pending[i].result() == actor else -1
                    if roots[i] != actor and action_value == 1:
                        raise RuntimeError("Inconsistent DFPN proofs: a lost position has a winning action")
                    rows[i].update({f"{mode}_action": int(action[i]), f"{mode}_action_value": action_value,
                                    f"{mode}_q": float(reward[i] - config["discount"] * next_value[i]),
                                    f"{mode}_predicted_reward": float(reward[i]),
                                    f"{mode}_true_reward": actual_reward})
            with (output / "positions.jsonl").open("a") as stream:
                for row in rows.values():
                    records.append(row)
                    stream.write(json.dumps(row) + "\n")
        else:
            raise RuntimeError("Failed to generate enough unique nonterminal positions")
    result = summarize(records)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "manifest.json").write_text(json.dumps({"seed": seed, "positions": positions,
        "depths": list(range(2, 11)), "solver_command": "dfpn-solve-state",
        "source": "uniform random legal prefixes, deduplicated by observation",
        "scope": "Hex 4 only; exact action values on these sampled states"}, indent=2) + "\n")
    return result
