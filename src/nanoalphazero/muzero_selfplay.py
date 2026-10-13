"""Optional terminal-root collection; ordinary AlphaZero collection is unchanged."""

from dataclasses import replace

import jax
import jax.numpy as jnp


def with_delayed_reset(env):
    """Keep a final board for one call, then reset without applying its action.

    Match the existing autoreset wrapper's step/reset key split. A completed
    lane resets independently, while other lanes continue making real moves.
    """
    def step_one(state, action, key):
        step_key, reset_key = jax.random.split(key)
        return jax.lax.cond(
            state.terminated | state.truncated,
            lambda: jax.tree.map(lambda x: x[0], env.init(reset_key[None])),
            lambda: jax.tree.map(lambda x: x[0], env.step(
                jax.tree.map(lambda x: x[None], state), action[None], step_key[None])),
        )
    return replace(env, autostep=jax.jit(jax.vmap(step_one)))


def collect_terminal_roots(selfplay_fn, config):
    """Mark reset calls as completed terminal roots, never reset transitions.

    The preceding real move already backfilled the episode. This root owns
    the final outcome from the terminal player's perspective, no new reward.
    ep_termination_step=ep_step-1 means the whole unroll is absorbing padding.
    """
    def collect(rng, state, params):
        terminal = state.env_state.terminated | state.env_state.truncated
        following, output, metrics = selfplay_fn(rng, state, params)
        if config.get('muzero_predict_value_only', False):
            target = state.env_state.rewards[
                jnp.arange(terminal.shape[0]), state.env_state.current_player
            ]
        else:
            target = jnp.zeros_like(state.env_state.rewards[:, 0])
        mask = terminal[:, None]
        output = output.replace(
            is_terminal=mask,
            reward=jnp.where(mask, target[:, None], output.reward),
            transition_reward=jnp.where(mask, 0., output.transition_reward),
            discount=jnp.where(mask, 0., output.discount),
            is_pending_reward_i8=jnp.where(mask, 0, output.is_pending_reward_i8).astype(jnp.int8),
            is_fresh_i8=jnp.where(mask, 1, output.is_fresh_i8).astype(jnp.int8),
            is_valid_sample=output.is_valid_sample | mask,
            is_exploration=output.is_exploration & ~mask,
            ep_termination_step=jnp.where(mask, output.ep_step - 1, output.ep_termination_step),
        )
        following = following._replace(ep_step=jnp.where(
            terminal, 0, jnp.where(output.just_terminated[:, 0], state.ep_step + 1, following.ep_step)))
        # Keep progress statistics aligned with the collector's corrected counters.
        metrics = dict(metrics)
        next_env = following.env_state
        legal_count = (jnp.sum(jax.lax.population_count(next_env.legal_action_bitmask), axis=-1)
                       if getattr(next_env, 'legal_action_bitmask', None) is not None else
                       jnp.sum(next_env.legal_action_mask, axis=-1))
        legal_count = jnp.where(next_env.terminated | next_env.truncated, 0, legal_count)
        mid = (following.ep_step > 10) & (following.ep_step <= 30) & ~next_env.terminated
        metrics.update({'selfplay/ep_step_min': jnp.min(following.ep_step),
                        'selfplay/ep_step_max': jnp.max(following.ep_step),
                        'selfplay/ep_step_std': jnp.std(following.ep_step.astype(jnp.float32)),
                        'selfplay/n_legal_moves_avg': jnp.mean(legal_count),
                        'selfplay/n_legal_moves_avg_mid': jnp.sum(jnp.where(mid, legal_count, 0)) / jnp.maximum(jnp.sum(mid), 1),
                        '_n_mid': jnp.sum(mid)})
        return following, output, metrics
    return collect
