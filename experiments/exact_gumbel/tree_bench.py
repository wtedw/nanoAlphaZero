"""Tree-only overhead of gumbel_muzero_policy_node options (trivial network, chess sizes)."""
import functools, json, sys, time
import jax, jax.numpy as jnp
import mctx_ref
from mctx_ref._src import qtransforms as QT
import exact_gumbel as eg

B, A = int(sys.argv[2]) if len(sys.argv) > 2 else 2048, 4672
LAT = (8, 8, 256)
key = jax.random.PRNGKey(0)
w = jax.random.normal(key, (256, 4)) * 0.1

def recurrent(params, rng, action, latent):
    lat = latent * 0.999 + 0.001
    feat = jnp.mean(lat, axis=(1, 2)) @ params
    logits = feat[:, :1] * jnp.arange(A, dtype=jnp.float32)[None] * 1e-3 + jnp.sin(action[:, None] + jnp.arange(A)[None])
    value = jnp.tanh(feat[:, 1])
    return mctx_ref.RecurrentFnOutput(reward=jnp.zeros_like(value), discount=-jnp.ones_like(value),
                                      prior_logits=logits, value=value), lat

latent = jax.random.normal(key, (B, *LAT)); logits = jax.random.normal(key, (B, A))
perm = jax.vmap(lambda k: jax.random.permutation(k, A))(jax.random.split(key, B))
invalid = (jnp.argsort(perm, -1) >= 32).astype(jnp.int32)
root = mctx_ref.RootFnOutput(prior_logits=logits, value=jnp.zeros(B), embedding=latent)
qt = functools.partial(QT.qtransform_completed_by_mix_value, value_scale=1.0, maxvisit_init=50.0, rescale_values=False)
configs = json.loads(sys.argv[3]) if len(sys.argv) > 3 else [{}, {"hoist_root": True}, {"edge_rows": "scatter"}, {"hoist_root": True, "edge_rows": "scatter"}]
for n in [int(x) for x in sys.argv[1].split(",")]:
    for kw in [dict(c) for c in configs]:
        fn = kw.pop("fn", "gumbel_muzero_policy_node")
        f = jax.jit(lambda p, k: getattr(eg, fn)(p, k, root, recurrent, n, invalid, qtransform=qt,
                                                  max_num_considered_actions=8, return_tree=False, **kw).action_weights)
        c = f.lower(w, key).compile()
        jax.block_until_ready(c(w, key)); ts = []
        for _ in range(3):
            t0 = time.perf_counter(); jax.block_until_ready(c(w, key)); ts.append(time.perf_counter() - t0)
        print(json.dumps(dict(n=n, B=B, fn=fn, **kw, ms=round(1000 * min(ts), 1))), flush=True)
