"""MuZero component metrics and cycle reductions; no tracker dependencies."""

import jax.numpy as jnp
import optax


BATCH_COUNTS = (
    "n_is_valid", "n_is_invalid", "n_policy_valid", "n_value_valid",
    "n_reward_valid", "n_value_pos", "n_value_neg", "n_value_zero",
)
NORMS = ("grad_norm", "param_norm", "update_norm")
OUTCOMES = ("selfplay/p1_wins", "selfplay/p2_wins", "selfplay/p_just_tied")


def training_batch_metrics(batch):
    """Count usable windows and supervised positions, including absorbing V/R tails."""
    valid = jnp.any(batch.policy_mask, axis=1)
    counts = dict(
        n_is_valid=jnp.sum(valid), n_is_invalid=jnp.sum(~valid),
        n_policy_valid=jnp.sum(batch.policy_mask),
        n_value_valid=jnp.sum(batch.value_mask),
        n_reward_valid=jnp.sum(batch.reward_mask),
        n_value_pos=jnp.sum(batch.value_mask & (batch.reward > 0)),
        n_value_neg=jnp.sum(batch.value_mask & (batch.reward < 0)),
        n_value_zero=jnp.sum(batch.value_mask & (batch.reward == 0)),
    )
    return {f"train_batch/{name}": value.astype(jnp.float32) for name, value in counts.items()}


def training_norm_metrics(params, grads, updates):
    """Pre-clipping gradients, pre-update parameters, and optimizer update norms."""
    return {
        "norms/grad_norm": optax.tree.norm(grads),
        "norms/param_norm": optax.tree.norm(params),
        "norms/update_norm": optax.tree.norm(updates),
    }


def empty_training_metrics(self_consistency=False):
    """Same scalar schema as training, for cycles without optimizer updates."""
    names = ["total_loss", "loss_pi", "loss_v", "loss_r"]
    names.extend(f"train_batch/{name}" for name in BATCH_COUNTS)
    names.extend(f"norms/{name}" for name in NORMS)
    if self_consistency:
        names.extend(("loss_consistency", "consistency/n_valid"))
    return {name: jnp.float32(0) for name in names}


def selfplay_metrics(scalars, state, output):
    """Select AlphaZero statistics and retain counts for weighted reductions."""
    metrics = {name: value for name, value in scalars.items()
               if name.startswith("selfplay/") and name != "selfplay/global_step"}
    metrics["_n_finished"] = jnp.sum(output.just_terminated)
    metrics["_n_mid"] = jnp.sum((state.ep_step > 10) & (state.ep_step <= 30))
    return metrics


def aggregate_selfplay_metrics(metrics):
    """Sum events, weight episode/legal averages, and report final lane progress."""
    finished = metrics["_n_finished"]
    count = jnp.sum(finished)

    def weighted_mean(values, weights):
        return jnp.sum(jnp.where(weights > 0, values * weights, 0)) / jnp.maximum(jnp.sum(weights), 1)

    result = {name: jnp.sum(metrics[name]) for name in OUTCOMES}
    for suffix, reduction, sentinel in (("min", jnp.min, jnp.inf), ("max", jnp.max, -jnp.inf)):
        name = f"selfplay/ep_term_step_{suffix}"
        result[name] = jnp.where(count > 0, reduction(jnp.where(finished > 0, metrics[name], sentinel)), -1.)
    result["selfplay/ep_term_step_avg"] = jnp.where(
        count > 0, weighted_mean(metrics["selfplay/ep_term_step_avg"], finished), -1.)
    result["selfplay/n_legal_moves_avg"] = jnp.mean(metrics["selfplay/n_legal_moves_avg"])
    result["selfplay/n_legal_moves_avg_mid"] = weighted_mean(
        metrics["selfplay/n_legal_moves_avg_mid"], metrics["_n_mid"])
    for suffix in ("min", "max", "std"):
        name = f"selfplay/ep_step_{suffix}"
        result[name] = metrics[name][-1]
    return result
