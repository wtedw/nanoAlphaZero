"""Episode replay and aligned unrolls, separate from AlphaZero outcome records."""

import flashbax
import jax
import jax.numpy as jnp


def discounted_returns(rewards, discounts, bootstrap):
    """[B,T] actor rewards; signed discounts map next-player values to actors."""
    def backup(value, transition):
        reward, discount = transition
        value = reward + discount * value
        return value, value
    _, values = jax.lax.scan(backup, bootstrap, (rewards.T, discounts.T), reverse=True)
    return values.T


def make_replay(config):
    # A Flashbax time slot contains an entire episode. Sampling cannot cross
    # games, circular-buffer wraparound, or independently reset batch lanes.
    return flashbax.make_trajectory_buffer(
        add_batch_size=config["selfplay_batch_size"],
        sample_batch_size=config["train_batch_size"],
        sample_sequence_length=1, period=1, min_length_time_axis=1,
        max_length_time_axis=config["replay_batches"],
    )


def sequences(episodes, key, unroll, starts=None):
    """Uniform game then uniform real position; pad terminal tails only.

    reward[:, k] trains g(s[k], a[k]); policy/value[:, k] train f(s[k]).
    Truncated tails have a bootstrap value at their boundary, but no invented
    rewards/policies beyond it. Terminal tails train absorbing zero values and
    rewards with random actions and no policy loss.
    """
    batch, length = episodes["action"].shape
    key, action_key = jax.random.split(key)
    if starts is None:
        starts = jax.random.randint(key, (batch,), 0, episodes["length"])
    indices = starts[:, None] + jnp.arange(unroll + 1)
    rows = jnp.arange(batch)[:, None]
    clipped = jnp.minimum(indices, length - 1)
    real = indices < episodes["length"][:, None]
    boundary = indices == episodes["length"][:, None]
    terminal = episodes["terminal"][:, None]
    value = jnp.where(real, episodes["value"][rows, clipped], 0.0)
    value = jnp.where(boundary & ~terminal, episodes["bootstrap"][:, None], value)
    random_actions = jax.random.randint(
        action_key, (batch, unroll), 0, episodes["policy"].shape[-1]
    )
    result = {
        "observation": episodes["observation"][jnp.arange(batch), starts],
        "action": jnp.where(real[:, :-1], episodes["action"][rows, clipped][:, :-1], random_actions),
        "reward": jnp.where(real[:, :-1], episodes["reward"][rows, clipped][:, :-1], 0.0),
        "policy": episodes["policy"][rows, clipped],
        "value": value,
        "policy_mask": real,
        "value_mask": real | boundary | terminal,
        "reward_mask": real[:, :-1] | terminal,
    }
    result["sample_info"] = {
        "ep_step": starts, "ep_termination_step": episodes["length"] - 1,
        "terminal": episodes["terminal"], "action": episodes["action"][jnp.arange(batch), starts],
        "is_from_selfplay": jnp.ones(batch, bool), "is_pending_reward_i8": jnp.zeros(batch, jnp.int8),
        "is_fresh_i8": jnp.ones(batch, jnp.int8),
        "is_exploration": episodes.get("exploration", jnp.zeros((batch, length), bool))[jnp.arange(batch), starts],
        "game_id": episodes.get("game_id", jnp.zeros(batch, jnp.uint32)),
        "row_id": episodes.get("row_id", jnp.arange(batch, dtype=jnp.int32) + 1),
    }
    return result
