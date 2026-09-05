"""Compose production value diagnostics with MuZero initial inference."""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np


def opening_head_tables(model, params, env, config, output):
    """Log small numeric tables for dynamic charts, never rendered images."""
    import json
    from nanoalphazero.eval.hex.perfect_play import perfect_play_values

    size = config["boardsize"]
    count = size * size
    output.mkdir(parents=True, exist_ok=False)
    root = env.init_dummy_estate(count)
    latent, logits, initial_value = model.apply({"params": params}, env.observe(root, root.current_player), method=model.initial)
    following = env.step(root, jnp.arange(count))
    _, after_logits, after_value = model.apply(
        {"params": params}, env.observe(following, following.current_player), method=model.initial)
    _, reward, latent_logits, latent_value = model.apply(
        {"params": params}, latent, jnp.arange(count), method=model.recurrent)
    arrays = dict(empty_policy_logits=np.asarray(logits[0]), after_policy_logits=np.asarray(after_logits),
                  after_value=np.asarray(after_value), latent_policy_logits=np.asarray(latent_logits),
                  latent_value=np.asarray(latent_value), predicted_reward=np.asarray(reward),
                  after_legal_mask=np.asarray(following.legal_action_mask))
    np.savez_compressed(output / "heads.npz", **arrays)
    truth = perfect_play_values(size).reshape(-1)
    from nanoalphazero.research.muzero.charts import board_table
    reply_rows = [[opening, action // size, action % size,
                   float(arrays["after_policy_logits"][opening, action]),
                   float(arrays["latent_policy_logits"][opening, action]),
                   bool(arrays["after_legal_mask"][opening, action])]
                  for opening in range(count) for action in range(count)]
    tables = {
        "model/values_after_black_moves_table": board_table(
            arrays["after_value"].reshape(size, size), truth.reshape(size, size), np.argsort(arrays["after_value"])[:size]),
        "model/logits_heatmap_top_k_table": board_table(
            arrays["empty_policy_logits"].reshape(size, size), highlights=np.argsort(-arrays["empty_policy_logits"])[:size]),
        "muzero/model/latent_values_after_black_moves_table": board_table(
            arrays["latent_value"].reshape(size, size), truth.reshape(size, size), np.argsort(arrays["latent_value"])[:size]),
        "muzero/model/reply_policy_logits_table": dict(
            columns=["opening", "row", "col", "policy_logit", "latent_policy_logit", "legal"], data=reply_rows),
    }
    (output / "tables.json").write_text(json.dumps(tables, allow_nan=False) + "\n")
    import jax
    probabilities = jax.nn.softmax(logits[0])
    entropy = float(-jnp.sum(probabilities * jnp.log(probabilities + 1e-8)))
    value_probs = jax.nn.softmax(after_value)
    metrics = {"model/init_board_value": float(initial_value[0]),
               "model/value_head_empty_board_entropy": float(-jnp.sum(value_probs * jnp.log(value_probs + 1e-8))),
               "model/policy_head_empty_board_entropy": entropy,
               "params-logits/init_logits_norm": float(jnp.linalg.norm(logits[0])),
               "params-logits/init_logits_std": float(jnp.std(logits[0])),
               "params-logits/init_logits_max": float(jnp.max(logits[0])),
               "params-logits/init_logits_min": float(jnp.min(logits[0])),
               "params-logits/init_logits_entropy": entropy,
               "params-logits/init_logits_entropy_its": entropy / np.log(2),
               "params-logits/num_moves_over_5pct": int(jnp.sum(probabilities > .05))}
    return tables, metrics


def inspect_position_values(model, params, env, config):
    from nanoalphazero.training import (
        _run_ttt_diagnostics, _run_hex_diagnostics,
        _run_connect4_diagnostics, _run_go_diagnostics,
    )

    def apply(variables, observation, legal, deterministic=True):
        _, logits, value = model.apply(variables, observation, method=model.initial)
        return logits, value

    state = SimpleNamespace(params=params, apply_fn=apply)
    env_id = config["env_id"]
    if env_id.startswith("hex"):
        _run_hex_diagnostics(state, env, config)
        from nanoalphazero.eval.hex.perfect_play import perfect_play_values
        size = config["boardsize"]
        root = env.init_dummy_estate(size * size)
        following = env.step(root, jnp.arange(size * size))
        _, _, value = model.apply({"params": params}, env.observe(following, following.current_player), method=model.initial)
        truth = perfect_play_values(size).reshape(-1)
        return {"hex_perf/mse_vs_perfect": float(jnp.mean((value - truth) ** 2)),
                "hex_perf/sign_accuracy": float(np.mean(np.sign(np.asarray(value)) == np.sign(truth)))}
    if env_id == "tic_tac_toe":
        _run_ttt_diagnostics(state, env, config)
    elif env_id == "connect_four":
        _run_connect4_diagnostics(state, env, config)
    elif env_id.startswith("go_"):
        _run_go_diagnostics(state, env, config)
    return {}
