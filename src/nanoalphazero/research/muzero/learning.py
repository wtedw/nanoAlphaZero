"""Self-play episode generation and end-to-end unrolled optimization."""

import jax
import jax.numpy as jnp
import optax

from nanoalphazero.research.muzero.model import scale_gradient
from nanoalphazero.research.muzero.replay import discounted_returns
from nanoalphazero.research.muzero.search import legal_actions


def make_collect(env, model, search, config):
    batch = config["selfplay_batch_size"]

    def collect(params, key):
        key, init_key = jax.random.split(key)
        state = env.init(jax.random.split(init_key, batch))
        if config.get("exploration_mode", "fixed") == "random_switch":
            key, exploration_key = jax.random.split(key)
            switch = jax.random.randint(exploration_key, (batch,), 0,
                                        max(config["exploration_moves"], 1))
        else:
            switch = config["exploration_moves"]

        def step(carry, t):
            state, ended, key = carry
            key, sk, ek, ak = jax.random.split(key, 4)
            obs = env.observe(state, state.current_player)
            policy = search(sk, state, params, 1.0, batch)
            # Preserve production's visit-count exploration convention.
            logits = jnp.log(jnp.maximum(policy.visit_counts, 1e-30))
            logits = jnp.where(legal_actions(state), logits, -1e30)
            sampled = jax.random.categorical(ak, logits)
            action = jnp.where(t < switch, sampled, policy.action).astype(jnp.int32)
            next_state = env.step(state, action, jax.random.split(ek, batch))
            reward = next_state.rewards[jnp.arange(batch), state.current_player]
            reward = jnp.where(ended, 0.0, reward)
            sign = jnp.where(state.current_player == next_state.current_player, 1.0, -1.0)
            discount = jnp.where(ended | next_state.terminated, 0.0, sign * config["discount"])
            row = {"observation": obs, "action": action,
                   "player": state.current_player,
                   "next_legal_count": jnp.sum(legal_actions(next_state), axis=-1),
                   "raw_policy_logits": getattr(policy, "raw_policy_logits", jnp.log(jnp.maximum(policy.action_weights, 1e-30))),
                   "predicted_value": getattr(policy, "predicted_value", jnp.zeros(batch)),
                   "reward": reward, "discount": discount,
                   "policy": policy.action_weights, "valid": ~ended,
                   "exploration": (~ended) & (t < switch),
                   "illegal": ~ended & ~legal_actions(state)[jnp.arange(batch), action]}
            ended = ended | next_state.terminated | next_state.truncated
            # Freeze completed states, so truncation bootstrap remains at the
            # actual boundary and terminal rewards cannot leak into padding.
            next_state = jax.tree.map(
                lambda old, new: jnp.where(carry[1].reshape((batch,) + (1,) * (new.ndim - 1)), old, new),
                state, next_state,
            )
            return (next_state, ended, key), row

        (state, _, _), rows = jax.lax.scan(
            step, (state, jnp.zeros(batch, bool), key), jnp.arange(config["max_steps"])
        )
        rows = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), rows)
        _, _, bootstrap = model.apply(
            {"params": params}, env.observe(state, state.current_player), method=model.initial
        )
        bootstrap = jnp.where(state.terminated, 0.0, bootstrap)
        # Padded rows must pass the boundary bootstrap backwards unchanged.
        discounts = jnp.where(rows["valid"], rows["discount"], 1.0)
        values = discounted_returns(rows["reward"], discounts, bootstrap)
        return {**rows, "value": values, "bootstrap": bootstrap,
                "length": jnp.sum(rows["valid"], axis=1), "terminal": state.terminated,
                "final_rewards": state.rewards, "game_id": jnp.arange(batch, dtype=jnp.uint32) + 1,
                "row_id": jnp.arange(batch, dtype=jnp.int32) + 1}
    return collect


def unroll_predictions(model, params, batch, with_wdl=False):
    wdl = None
    if with_wdl and hasattr(model, "initial_with_wdl"):
        latent, logits, value, wdl = model.apply({"params": params}, batch["observation"], method=model.initial_with_wdl)
    else:
        latent, logits, value = model.apply({"params": params}, batch["observation"], method=model.initial)
    policies, values, rewards = [logits], [value], []
    for k in range(batch["action"].shape[1]):
        latent, reward, logits, value = model.apply(
            {"params": params}, scale_gradient(latent, 0.5), batch["action"][:, k],
            method=model.recurrent,
        )
        policies.append(logits)
        values.append(value)
        rewards.append(reward)
    policies = jnp.stack(policies, axis=1)
    values = jnp.stack(values, axis=1)
    rewards = jnp.stack(rewards, axis=1)
    return (policies, values, rewards, wdl) if with_wdl else (policies, values, rewards)


def loss(model, params, batch):
    policies, values, rewards, wdl = unroll_predictions(model, params, batch, with_wdl=True)

    def mean_masked(x, mask):
        return jnp.sum(jnp.where(mask, x, 0.0)) / jnp.maximum(jnp.sum(mask), 1)

    pi = mean_masked(optax.softmax_cross_entropy(policies, batch["policy"]), batch["policy_mask"])
    v = mean_masked((values - batch["value"]) ** 2, batch["value_mask"])
    r = mean_masked((rewards - batch["reward"]) ** 2, batch["reward_mask"])
    root_mask = batch["policy_mask"][:, 0]
    target_entropy = mean_masked(jnp.sum(jax.scipy.special.entr(batch["policy"][:, 0]), -1), root_mask)
    network_entropy = mean_masked(jnp.sum(jax.scipy.special.entr(jax.nn.softmax(policies[:, 0])), -1), root_mask)
    extra = {"entropy/target_policy": target_entropy, "entropy/network_policy": network_entropy,
             "entropy/diff_network_minus_search": network_entropy - target_entropy,
             "muzero/root_policy_kl": mean_masked(optax.softmax_cross_entropy(policies[:, 0], batch["policy"][:, 0]), root_mask) - target_entropy}
    if wdl is not None:
        for i, name in enumerate(("win", "loss", "draw")):
            extra[f"wdl/p_{name}_mean"] = mean_masked(wdl[:, i], root_mask)
    if "sample_info" in batch:
        info = batch["sample_info"]
        for step in range(10):
            mask = root_mask & info["terminal"] & (info["ep_termination_step"] - info["ep_step"] == step)
            suffix = f"term_step_{step:02d}"
            extra[f"loss_per_step_PI/pi_{suffix}"] = mean_masked(
                optax.softmax_cross_entropy(policies[:, 0], batch["policy"][:, 0]), mask)
            extra[f"loss_per_step_V/v_{suffix}"] = mean_masked((values[:, 0] - batch["value"][:, 0]) ** 2, mask)
            extra[f"loss_per_step_count/count_{suffix}"] = jnp.sum(mask)
    return pi + v + r, {"policy_loss": pi, "value_loss": v, "reward_loss": r, **extra}


def train_step(model, state, batch):
    (total, metrics), grads = jax.value_and_grad(loss, argnums=1, has_aux=True)(model, state.params, batch)
    updates, opt_state = state.tx.update(grads, state.opt_state, state.params)
    updated = state.replace(step=state.step + 1, params=optax.apply_updates(state.params, updates), opt_state=opt_state)
    grad_norm, param_norm, update_norm = map(optax.global_norm, (grads, state.params, updates))
    dot = sum(jnp.vdot(g, u) for g, u in zip(jax.tree.leaves(grads), jax.tree.leaves(updates)))
    from flax.traverse_util import flatten_dict
    flat = flatten_dict(updates)
    head_metrics = {}
    for name in ("policy", "value"):
        leaves = [v for path, v in flat.items() if name in "/".join(path).lower() and path[0] != "reward"]
        if leaves:
            head_metrics[f"norms/{name}_head_update_ratio"] = optax.global_norm(leaves) / jnp.maximum(param_norm, 1e-12)
    return updated, {
        **metrics, "loss": total, "grad_norm": grad_norm,
        "norms/param_norm": param_norm, "norms/update_norm": update_norm,
        "norms/update_ratio": update_norm / jnp.maximum(param_norm, 1e-12),
        "norms/update_grad_cosine": dot / (grad_norm * update_norm + 1e-12),
        **head_metrics,
    }
