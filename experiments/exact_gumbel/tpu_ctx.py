"""Is upstream's own action_weights reproducible when its final step is compiled separately?"""
import functools, json
import jax, jax.numpy as jnp, numpy as np
import mctx_ref
from mctx_ref._src import qtransforms as QT, seq_halving as SH, action_selection as AS
from mctx_ref._src.policies import _mask_invalid_actions
from nanoalphazero.config import get_muzero_config
from nanoalphazero.model import make_muzero_model

cfg = get_muzero_config("chess", muzero_remat_blocks=False, muzero_remat_unroll=False)
cfg.update(game_obs_shape=[8, 8, 119], game_num_actions=4672)
model, variables = make_muzero_model(cfg, jax.random.PRNGKey(0))
params = variables["params"]
B, A, n, m = 512, 4672, 24, 8
key = jax.random.PRNGKey(3)
obs = jax.random.normal(key, (B, 8, 8, 119))
perm = jax.vmap(lambda k: jax.random.permutation(k, A))(jax.random.split(key, B))
invalid = (jnp.argsort(perm, -1) >= 32).astype(jnp.int32)
qt = functools.partial(QT.qtransform_completed_by_mix_value, value_scale=1.0, maxvisit_init=50.0, rescale_values=False)

def recurrent(params, rng, action, latent):
    latent, reward, logits, value = model.apply({"params": params}, latent, action, method=model.recurrent)
    return mctx_ref.RecurrentFnOutput(reward=jnp.zeros_like(value), discount=jnp.full_like(value, -1.0),
                                      prior_logits=logits, value=value), latent

@jax.jit
def full(params, obs, key):
    latent, logits, value = model.apply({"params": params}, obs, method=model.initial)
    root = mctx_ref.RootFnOutput(prior_logits=logits, value=value, embedding=latent)
    out = mctx_ref.gumbel_muzero_policy(params, key, root, recurrent, n, invalid, qtransform=qt, max_num_considered_actions=m)
    return out, _mask_invalid_actions(logits, invalid)

@jax.jit
def final_only(tree, masked_logits):
    gumbel = tree.extra_data.root_gumbel
    summary = tree.summary()
    cq = jax.vmap(qt, in_axes=[0, None])(tree, tree.ROOT_INDEX)
    return jax.nn.softmax(_mask_invalid_actions(masked_logits + cq, invalid))

out, masked = full(params, obs, key)
w2 = final_only(out.search_tree, masked)
print(json.dumps(dict(upstream_vs_upstream_final_recompiled_equal=bool(np.array_equal(out.action_weights, w2)),
                      max_abs_diff=float(np.max(np.abs(np.asarray(out.action_weights) - np.asarray(w2)))))))
