"""On TPU: compare node variant vs upstream mctx with the production chess network."""
import functools, json, sys
import jax, jax.numpy as jnp, numpy as np
import mctx_ref
from mctx_ref._src import qtransforms as QT
import exact_gumbel as eg
from nanoalphazero.config import get_muzero_config
from nanoalphazero.model import make_muzero_model

env = sys.argv[1] if len(sys.argv) > 1 else "chess"
shapes = {"chess": ((8, 8, 119), 4672), "hex9": ((9, 9, 4), 81)}
obs_shape, A = shapes[env]
cfg = get_muzero_config(env, muzero_remat_blocks=False, muzero_remat_unroll=False)
cfg.update(game_obs_shape=list(obs_shape), game_num_actions=A)
model, variables = make_muzero_model(cfg, jax.random.PRNGKey(0))
params = variables["params"]
B = 512

def recurrent(params, rng, action, latent):
    latent, reward, logits, value = model.apply({"params": params}, latent, action, method=model.recurrent)
    return mctx_ref.RecurrentFnOutput(reward=jnp.zeros_like(value), discount=jnp.full_like(value, -1.0),
                                      prior_logits=logits, value=value), latent

for n, m, rescale, vs, legal in [(12, 8, False, 1.0, 32), (24, 8, False, 1.0, 32), (48, 16, False, 1.0, 40),
                                 (24, 8, True, 0.1, 32), (32, 16, False, 1.0, 3)]:
    key = jax.random.PRNGKey(n + m)
    obs = jax.random.normal(key, (B, *obs_shape))
    perm = jax.vmap(lambda k: jax.random.permutation(k, A))(jax.random.split(key, B))
    nlegal = jnp.clip(jax.random.poisson(key, legal, (B,)), 0, A)
    invalid = (jnp.argsort(perm, -1) >= nlegal[:, None]).astype(jnp.int32)
    qt = functools.partial(QT.qtransform_completed_by_mix_value, value_scale=vs, maxvisit_init=50.0, rescale_values=rescale)

    def run(policy, params, obs, invalid, key):
        latent, logits, value = model.apply({"params": params}, obs, method=model.initial)
        root = mctx_ref.RootFnOutput(prior_logits=logits, value=value, embedding=latent)
        return policy(params, key, root, recurrent, n, invalid, qtransform=qt, max_num_considered_actions=m)

    ref = jax.jit(functools.partial(run, mctx_ref.gumbel_muzero_policy))(params, obs, invalid, key)
    out = jax.jit(functools.partial(run, getattr(eg, sys.argv[2] if len(sys.argv) > 2 else "gumbel_muzero_policy_node")))(params, obs, invalid, key)
    fields = ["node_visits", "node_values", "raw_values", "children_index", "children_visits",
              "children_values", "children_prior_logits", "embeddings"]
    res = dict(n=n, m=m, rescale=rescale, value_scale=vs,
               action=bool(np.array_equal(ref.action, out.action)),
               action_weights=bool(np.array_equal(ref.action_weights, out.action_weights)),
               max_abs_weight_diff=float(np.max(np.abs(np.asarray(ref.action_weights) - np.asarray(out.action_weights)))))
    res.update({f: bool(np.array_equal(getattr(ref.search_tree, f), getattr(out.search_tree, f))) for f in fields})
    print(json.dumps(res), flush=True)
