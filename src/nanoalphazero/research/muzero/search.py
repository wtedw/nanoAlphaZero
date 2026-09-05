"""Adapters for the unchanged production two-rung search."""

import functools
import chex

import jax
import jax.numpy as jnp

from nanoalphazero.buffers import unpack_bitmask_vmap
from nanoalphazero.mcts import PolicyOutput, RecurrentFnOutput, RootFnOutput, gumbel_muzero_policy_1sh


@chex.dataclass(frozen=True)
class ResearchPolicyOutput(PolicyOutput):
    raw_policy_logits: object = None
    predicted_value: object = None


def legal_actions(state):
    if hasattr(state, "legal_action_bitmask"):
        return unpack_bitmask_vmap(state.legal_action_bitmask)
    return state.legal_action_mask


def latent_search(model, params, observation, legal, key, *, roots, survivors,
                  gumbel_scale=1.0, discount=1.0, alternating=True,
                  root_temperature=1., value_scale=1., maxvisit_init=50., rescale_values=False,
                  diagnostics=False):
    """No environment object or true future state enters this function."""
    latent, logits, value = model.apply({"params": params}, observation, method=model.initial)

    def recurrent(params, keys, action, latent):
        latent, reward, logits, value = model.apply(
            {"params": params}, latent, action, method=model.recurrent
        )
        signed_discount = -discount if alternating else discount
        return RecurrentFnOutput(
            reward=reward, discount=jnp.full_like(value, signed_discount),
            prior_logits=logits, value=value,
        ), latent

    output = gumbel_muzero_policy_1sh(
        params, key, RootFnOutput(prior_logits=logits / root_temperature, value=value, embedding=latent),
        recurrent, num_root_considered=roots, num_survivors=survivors,
        invalid_actions=~legal, gumbel_scale=gumbel_scale,
        value_scale=value_scale, maxvisit_init=maxvisit_init, rescale_values=rescale_values,
    )
    if diagnostics:
        return ResearchPolicyOutput(**vars(output), raw_policy_logits=logits, predicted_value=value)
    return output


def make_search(model, env, config, policy_only=False):
    @functools.partial(jax.jit, static_argnums=(4, 5))
    def search(key, state, params, gumbel_scale, batch_size, num_simulations=None):
        obs = env.observe(state, state.current_player)
        legal = legal_actions(state)
        if policy_only:
            from nanoalphazero.mcts import PolicyOutput
            _, logits, _ = model.apply({"params": params}, obs, method=model.initial)
            logits = jnp.where(legal, logits, jnp.finfo(logits.dtype).min)
            return PolicyOutput(action=jnp.argmax(logits, -1),
                                action_weights=jax.nn.softmax(logits))
        return latent_search(
            model, params, obs, legal, key, roots=config["roots"],
            survivors=config["survivors"], gumbel_scale=gumbel_scale,
            discount=config["discount"],
            root_temperature=config.get("root_temperature", 1.),
            value_scale=config.get("value_scale", 1.), maxvisit_init=config.get("maxvisit_init", 50.),
            rescale_values=config.get("rescale_values", False),
            diagnostics=True,
        )
    return search
