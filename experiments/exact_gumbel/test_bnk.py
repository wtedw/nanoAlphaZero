"""compress_action_weights matches maz's compress_muzero_policy on exact-search output."""
import functools
import jax, jax.numpy as jnp, numpy as np
import mctx_ref
from mctx_ref._src import qtransforms as QT
import exact_gumbel as eg
import test_exact as te
from nanoalphazero.mcts import PolicyOutput, compress_muzero_policy

B, A, n, K = 64, 40, 24, 8
params, recurrent = te.make_model(A)
k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
root = mctx_ref.RootFnOutput(prior_logits=jax.random.normal(k1, (B, A)) * 2,
                             value=jnp.tanh(jax.random.normal(k2, (B,))),
                             embedding=jax.random.normal(k3, (B, 16)))
invalid = (jax.random.uniform(k3, (B, A)) < 0.5).astype(jnp.int32).at[:, 0].set(0)
qt = functools.partial(QT.qtransform_completed_by_mix_value, value_scale=1.0, maxvisit_init=50.0, rescale_values=False)
out = jax.jit(lambda k: eg.gumbel_muzero_policy_node2(params, k, root, recurrent, n, invalid, qtransform=qt,
                                                     max_num_considered_actions=8, return_tree=False))(jax.random.PRNGKey(1))
idx, w, mass = jax.jit(eg.compress_action_weights, static_argnums=1)(out.action_weights, K)
ref, ref_mass = jax.jit(compress_muzero_policy, static_argnums=1)(
    PolicyOutput(action=out.action, action_weights=out.action_weights), K)
assert np.array_equal(idx, ref.bnk_k_indices) and idx.dtype == ref.bnk_k_indices.dtype
assert np.array_equal(w, ref.bnk_action_weights) and np.array_equal(mass, ref_mass)
print("bnk compression matches maz: ok; mean retained mass", float(np.mean(mass)))
