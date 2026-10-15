"""Drop-in replacement for nanoalphazero.mcts.make_muzero_mcts using exact node2 search.

    from maz_integration import make_exact_muzero_mcts
    run_mcts_fn = make_exact_muzero_mcts(config, env, model, data_sharding)

Returns maz's PolicyOutput (action, action_weights, visit_counts, BNK fields),
so self-play's exploration sampler and BNK compression work unchanged. The
search is full sequential Gumbel MuZero with config["mcts_num_simulations"]
simulations and config["mcts_num_root_considered"] considered root actions,
bitwise equal to upstream mctx.gumbel_muzero_policy (per device under
sharding). mcts_num_survivors is unused: sequential halving derives the rungs.
"""
import functools

import jax
import jax.numpy as jnp

import mctx_ref
from mctx_ref._src import qtransforms as QT

import exact_gumbel as eg
from nanoalphazero.buffers import unpack_bitmask_vmap
from nanoalphazero.mcts import PolicyOutput, compress_muzero_policy


def make_exact_muzero_mcts(config, wenv, model, data_sharding=None):
    qtransform = functools.partial(
        QT.qtransform_completed_by_mix_value,
        value_scale=config.get("mcts_value_scale", 1.0),
        maxvisit_init=config.get("mcts_maxvisit_init", 50.0),
        rescale_values=config.get("mcts_rescale_values", False))

    def recurrent(params, rng, action, latent):
        latent, reward, logits, value = model.apply(
            {"params": params}, latent, action, method=model.recurrent)
        # Built-in games alternate players in hypothetical search.
        if config.get("muzero_predict_value_only", False):
            reward = jnp.zeros_like(reward)
        return mctx_ref.RecurrentFnOutput(
            reward=reward, discount=jnp.full_like(value, -1.0),
            prior_logits=logits, value=value), latent

    @functools.partial(jax.jit, static_argnums=(4, 5))
    def run_mcts(key, state, params, gumbel_scale, batch_size, num_simulations=None):
        n = num_simulations or config["mcts_num_simulations"]
        observation = wenv.observe(state, state.current_player)
        legal = (unpack_bitmask_vmap(state.legal_action_bitmask)
                 if getattr(state, "legal_action_bitmask", None) is not None
                 else state.legal_action_mask)
        latent, logits, value = model.apply({"params": params}, observation, method=model.initial)
        if config.get("exp_use_root_temperature", False):
            logits = jnp.where(legal, logits / config.get("exp_root_temperature", 1.3),
                               jnp.finfo(logits.dtype).min)
        root = mctx_ref.RootFnOutput(prior_logits=logits, value=value, embedding=latent)
        invalid = (~legal).astype(jnp.int32)
        kwargs = dict(qtransform=qtransform,
                      max_num_considered_actions=config["mcts_num_root_considered"],
                      gumbel_scale=gumbel_scale)
        if data_sharding is not None and len(data_sharding.mesh.devices.flat) > 1:
            out, visits = eg.gumbel_muzero_policy_node2_sharded(
                data_sharding.mesh, data_sharding.spec[0], params, key, root, recurrent, n,
                invalid, **kwargs)
        else:
            out, visits = eg.gumbel_muzero_policy_node2_with_visits(
                params, key, root, recurrent, n, invalid, return_tree=False, **kwargs)
        output = PolicyOutput(action=out.action, action_weights=out.action_weights,
                              visit_counts=visits)
        if config.get("exp_bnk_action_weights", False):
            output, _ = compress_muzero_policy(output, config["mcts_num_k_actions"])
        return output

    return run_mcts
