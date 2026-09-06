"""Real-environment matches and held-out unroll diagnostics."""

import functools
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from nanoalphazero.research.muzero.learning import loss, unroll_predictions
from nanoalphazero.research.muzero.replay import sequences
from nanoalphazero.research.muzero.search import make_search, legal_actions


def make_evaluator(model, env, config, reference=None):
    from nanoalphazero.training import all_opening_actions
    search = make_search(model, env, config)
    policy = make_search(model, env, config, policy_only=True)

    mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ("data",))
    data = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    @functools.partial(jax.jit, static_argnums=(3, 4),
                       in_shardings=(replicated, data, replicated), out_shardings=data)
    def match(params, openings, key, mode0, mode1):
        n = openings.shape[0]
        state = env.init_dummy_estate(n)
        for d in range(openings.shape[1]):
            state = env.step(state, openings[:, d])

        def step(carry, _):
            state, rewards, finished, key = carry
            key, ak = jax.random.split(key)
            random = jnp.argmax(jnp.where(legal_actions(state), jax.random.gumbel(ak, (n, env.num_actions)), -jnp.inf), -1)
            actions = {"random": random}
            for mode in set((mode0, mode1)) - {"random"}:
                if mode == "alphazero":
                    fn, reference_params = reference
                    actions[mode] = fn(ak, state, reference_params, 0., n).action
                else:
                    fn = search if mode == "search" else policy
                    actions[mode] = fn(ak, state, params, 0., n).action
            action = jnp.where(state.current_player == 0, actions[mode0], actions[mode1]).astype(jnp.int32)
            illegal = ~finished & ~legal_actions(state)[jnp.arange(n), action]
            next_state = env.step(state, action)
            newly = ~finished & next_state.terminated
            rewards = jnp.where(newly[:, None], next_state.rewards, rewards)
            return (next_state, rewards, finished | next_state.terminated, key), (action, illegal)

        (_, rewards, finished, _), (actions, illegal) = jax.lax.scan(
            step, (state, state.rewards, state.terminated, key), None, length=config["max_steps"]
        )
        return {"rewards": rewards, "finished": finished, "actions": actions.T, "illegal": illegal.T}

    def evaluate(params, key, directory, opening_plies=2):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        openings = all_opening_actions(env, config, plies=opening_plies)
        # Hex/TTT opening counts are multiples of four at depth two. General
        # environments may need padding; discard duplicated rows when scoring.
        count = len(openings)
        pad = (-count) % config["devices"]
        padded = np.concatenate([openings, np.resize(openings, (pad, opening_plies))])
        np.save(directory / "openings.npy", openings)
        scores = {}
        pairings = [("policy", "random"), ("random", "policy"),
                             ("search", "random"), ("random", "search"),
                             ("search", "policy"), ("policy", "search")]
        if reference is not None:
            pairings += [("policy", "alphazero"), ("alphazero", "policy"),
                         ("search", "alphazero"), ("alphazero", "search")]
        for mode0, mode1 in pairings:
            key, mk = jax.random.split(key)
            started = time.monotonic()
            out = jax.device_get(match(params, jnp.asarray(padded), mk, mode0, mode1))
            if np.any(out["illegal"]):
                raise RuntimeError("Illegal evaluation action")
            label = f"{mode0}-vs-{mode1}"
            np.savez_compressed(directory / f"{label}.npz", **out)
            r = out["rewards"][:count, 0]
            scores[label] = dict(wins=int(np.sum(r > 0)), draws=int(np.sum(r == 0)),
                                 losses=int(np.sum(r < 0)), unfinished=int(np.sum(~out["finished"][:count])),
                                 seconds=time.monotonic() - started)
        (directory / "summary.json").write_text(json.dumps(scores, indent=2) + "\n")
        return scores
    return evaluate


@functools.lru_cache(maxsize=16)
def _heldout_function(model):
    # Keep one callable alive across monitoring events. Recreating jitted
    # partials here otherwise recompiles each unroll shape at every evaluation.
    @jax.jit
    def evaluate(params, batch):
        _, metrics = loss(model, params, batch)
        return metrics, unroll_predictions(model, params, batch)
    return evaluate


def heldout_metrics(model, params, episodes, key, unrolls=(1, 3, 5, 10)):
    result = {}
    for unroll in unrolls:
        batch = sequences(episodes, key, unroll)
        metrics, (_, values, rewards) = _heldout_function(model)(params, batch)
        result.update({f"heldout/unroll{unroll}/{k}": float(v) for k, v in metrics.items()})
        # Report real states separately: absorbing zeros otherwise make longer
        # unroll averages look deceptively easier.
        def masked_mean(x, mask):
            return float(jnp.sum(jnp.where(mask, x, 0.)) / jnp.maximum(jnp.sum(mask), 1))
        result[f"heldout/unroll{unroll}/real_value_mse"] = masked_mean((values - batch["value"]) ** 2, batch["policy_mask"])
        # Production-style replay excludes exploratory starts. Keep the mixed
        # held-out metric, but expose this distribution difference explicitly.
        # These groups are defined by the sampled root, not by each later ply.
        exploration = batch["sample_info"]["is_exploration"]
        for label, roots in (("exploration", exploration), ("nonexploration", ~exploration)):
            mask = batch["policy_mask"] & roots[:, None]
            prefix = f"heldout/unroll{unroll}/{label}_root"
            result[f"{prefix}_count"] = int(jnp.sum(roots))
            result[f"{prefix}_real_value_mse"] = masked_mean((values - batch["value"]) ** 2, mask)
            result[f"{prefix}_k0_value_mse"] = masked_mean(
                (values[:, 0] - batch["value"][:, 0]) ** 2, roots)
        positive = batch["reward_mask"] & (batch["reward"] > 0)
        result[f"heldout/unroll{unroll}/nonzero_reward_count"] = int(jnp.sum(positive))
        result[f"heldout/unroll{unroll}/positive_reward_prediction"] = masked_mean(rewards, positive)
        result[f"heldout/unroll{unroll}/real_reward_mse"] = masked_mean((rewards - batch["reward"]) ** 2, batch["policy_mask"][:, :-1])
        for k in range(unroll + 1):
            result[f"heldout/unroll{unroll}/k{k}_real_value_mse"] = masked_mean((values[:, k] - batch["value"][:, k]) ** 2, batch["policy_mask"][:, k])
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate a MuZero checkpoint in real games")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alphazero", type=Path)
    parser.add_argument("--mohex-engine-path")
    parser.add_argument("--mohex-engine-config", default="default")
    parser.add_argument("--opening-plies", type=int, default=2)
    parser.add_argument("--platform", choices=("cpu", "tpu"), default="tpu")
    args = parser.parse_args()
    pool = None
    if args.mohex_engine_path:
        from nanoalphazero.research.muzero.checkpoint import metadata
        from nanoalphazero.eval.hex.engine import MoHexPool
        config = metadata(args.checkpoint)
        if not config["env"].startswith("hex"):
            raise ValueError("MoHex requires a Hex checkpoint")
        size = int(config["env"][3:])
        pool = MoHexPool(args.mohex_engine_path, args.mohex_engine_config, size, size * size)
    try:
        _run_evaluation(args, pool)
    finally:
        if pool is not None:
            pool.close()


def _run_evaluation(args, pool):
    # JAX has been imported but device initialization has not occurred here.
    jax.config.update("jax_platforms", args.platform)
    from nanoalphazero.core import make_env
    from nanoalphazero.research.muzero.checkpoint import load_params
    from nanoalphazero.research.muzero.model import build_model
    params, config = load_params(args.checkpoint)
    if len(jax.devices()) != 4:
        raise ValueError("Evaluation requires four devices")
    env = make_env(config)
    if list(env.obs_shape) != config["obs_shape"] or env.num_actions != config["num_actions"]:
        raise ValueError("Environment/checkpoint interface mismatch")
    model = build_model(config)
    reference = None
    if args.alphazero:
        from nanoalphazero.checkpoint import load_checkpoint, apply_checkpoint_model_config
        from nanoalphazero.config import CONFIG_FACTORIES
        from nanoalphazero.model import make_model
        from nanoalphazero.mcts import make_mcts
        ref_params, metadata = load_checkpoint(str(args.alphazero))
        ref_config = apply_checkpoint_model_config(CONFIG_FACTORIES[config["env"]](), metadata)
        if ref_config["env_id"] != config["env_id"]:
            raise ValueError("AlphaZero checkpoint environment mismatch")
        ref_config.update(game_obs_shape=env.obs_shape, game_num_actions=env.num_actions, enable_sharding=False)
        ref_model, _ = make_model(ref_config, jax.random.PRNGKey(0))
        reference = make_mcts(ref_config, env, ref_model), ref_params
    scores = make_evaluator(model, env, config, reference)(
        params, jax.random.PRNGKey(config["seed"] + 2000000), args.output, args.opening_plies
    )
    import hashlib
    manifest = {"checkpoint": str(args.checkpoint), "config": config,
                "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                "alphazero": str(args.alphazero) if args.alphazero else None,
                "matched": "same MuZero parameters for policy/search; AlphaZero is an unmatched strength reference",
                "platform": args.platform, "opening_plies": args.opening_plies}
    if args.alphazero:
        manifest["alphazero_sha256"] = hashlib.sha256(args.alphazero.read_bytes()).hexdigest()
    if pool is not None:
        from nanoalphazero.eval.hex.training import HexTrainingEvaluator
        from nanoalphazero.config import CONFIG_FACTORIES
        inspection_config = {**CONFIG_FACTORIES[config["env"]](), "enable_sharding": True}
        manifest["mohex"] = {"executable": str(pool.executable), "config": pool.config.read_text(),
                              "sha256": hashlib.sha256(pool.executable.read_bytes()).hexdigest()}
        for mode in ("search", "policy"):
            evaluator = HexTrainingEvaluator(1, pool, args.output / f"mohex-{mode}",
                                              board_size=inspection_config["boardsize"])
            fn = make_search(model, env, config, policy_only=mode == "policy")
            scores[f"mohex-{mode}"] = evaluator.run_if_due(1, fn, env, inspection_config, params)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(scores, indent=2))
