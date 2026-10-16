"""Bounded distributed chess comparison with pinned upstream mctx on PYTHONPATH."""
import json
import os

os.environ['JAX_PLATFORMS'] = 'tpu'
from nanoalphazero.distributed import DistributedArgs, initialize_distributed
initialize_distributed(DistributedArgs(enabled=True))

import functools
import jax
import jax.numpy as jnp
import mctx_ref
from mctx_ref._src.qtransforms import qtransform_completed_by_mix_value
from nanoalphazero.config import get_muzero_config
from nanoalphazero.core import make_env, DATA_PARALLEL_SHARDING, REPLICATED_SHARDING
from nanoalphazero.model import make_muzero_model
from nanoalphazero.mcts import RootFnOutput, RecurrentFnOutput, make_muzero_mcts, compress_muzero_policy
from nanoalphazero.mcts_exact import gumbel_muzero_policy_node

config = get_muzero_config('chess', conv_width=8, conv_depth=1,
    mcts_variant='gumbel', mcts_num_simulations=32, mcts_num_root_considered=16,
    muzero_chess_action_encoding='from_to8')
env = make_env(config)
config.update(game_obs_shape=list(env.obs_shape), game_num_actions=env.num_actions)
model, variables = make_muzero_model(config, jax.random.PRNGKey(0), REPLICATED_SHARDING)
state = jax.device_put(env.init_dummy_estate(16), DATA_PARALLEL_SHARDING)
observation = env.observe(state, state.current_player)
latent, logits, value = model.apply(variables, observation, method=model.initial)
from nanoalphazero.mcts import unpack_bitmask_vmap
invalid = ~unpack_bitmask_vmap(state.legal_action_bitmask)
if config.get('exp_use_root_temperature', False):
    logits = jnp.where(~invalid, logits / config.get('exp_root_temperature', 1.3),
                       jnp.finfo(logits.dtype).min)
root = RootFnOutput(prior_logits=logits, value=value, embedding=latent)

def recurrent(params, key, action, latent):
    latent, reward, logits, value = model.apply({'params': params}, latent, action, method=model.recurrent)
    return RecurrentFnOutput(reward=reward, discount=jnp.full_like(value, -1),
                             prior_logits=logits, value=value), latent

kw = dict(num_simulations=32, max_num_considered_actions=16,
    qtransform=functools.partial(qtransform_completed_by_mix_value,
        value_scale=config['mcts_value_scale'], maxvisit_init=config['mcts_maxvisit_init'],
        rescale_values=config['mcts_rescale_values']))
params = variables['params']
key = jax.random.PRNGKey(9)
reference = jax.jit(lambda p, r, mask: mctx_ref.gumbel_muzero_policy(
    p, key, r, recurrent, invalid_actions=mask, **kw))(params, root, invalid)
actual = jax.jit(lambda p, r, mask: gumbel_muzero_policy_node(
    p, key, r, recurrent, invalid_actions=mask, **kw))(params, root, invalid)
assert bool(jnp.all(actual.action == reference.action))
for a, b in zip(jax.tree.leaves(actual.search_tree), jax.tree.leaves(reference.search_tree), strict=True):
    assert bool(jnp.all(a == b)), 'Search tree differs from upstream'
assert bool(jnp.allclose(actual.action_weights, reference.action_weights, atol=2e-7, rtol=2e-6))
output = make_muzero_mcts(config, env, model, DATA_PARALLEL_SHARDING,
                          return_diagnostics=True)(key, state, params, 1., 16)
assert bool(jnp.all(output.action == reference.action))
assert bool(jnp.all(output.visit_counts == reference.search_tree.summary().visit_counts))
assert bool(jnp.allclose(output.action_weights, reference.action_weights, atol=2e-7, rtol=2e-6))
packed, mass = compress_muzero_policy(output, 128)
assert bool(jnp.all(packed.bnk_k_indices == output.bnk_k_indices))
assert bool(jnp.all(packed.bnk_action_weights == output.bnk_action_weights))
assert bool(jnp.all(output.visit_counts.sum(-1) == 32))
print(json.dumps(dict(rank=jax.process_index(), hosts=jax.process_count(),
    devices=jax.device_count(), actions_equal=True, tree_equal=True,
    max_policy_error=float(jnp.max(jnp.abs(actual.action_weights-reference.action_weights))),
    min_retained_mass=float(jnp.min(mass)), bnk_equal=True)), flush=True)
