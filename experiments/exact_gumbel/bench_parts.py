"""Tree-overhead profile: upstream search with a trivial network at chess sizes."""
import functools, json, sys, time
import jax, jax.numpy as jnp, numpy as np
import mctx_ref
from mctx_ref._src import search as S, action_selection as AS, qtransforms as QT, tree as T
import exact_gumbel as eg

B, A, n = 2048, 4672, int(sys.argv[1]) if len(sys.argv) > 1 else 24
LAT = (8, 8, 256)
key = jax.random.PRNGKey(0)
w = jax.random.normal(key, (256, 4)) * 0.1

def recurrent(params, rng, action, latent):
    # Trivial "network": cheap, but returns full-size latents and logits.
    lat = latent * 0.999 + 0.001
    feat = jnp.mean(lat, axis=(1, 2)) @ params          # [B, 4]
    logits = feat[:, :1] * jnp.arange(A, dtype=jnp.float32)[None] * 1e-3 + jnp.sin(action[:, None] + jnp.arange(A)[None])
    value = jnp.tanh(feat[:, 1])
    return mctx_ref.RecurrentFnOutput(reward=jnp.zeros_like(value), discount=-jnp.ones_like(value),
                                      prior_logits=logits, value=value), lat

latent = jax.random.normal(key, (B, *LAT))
logits = jax.random.normal(key, (B, A))
value = jnp.zeros(B)
perm = jax.vmap(lambda k: jax.random.permutation(k, A))(jax.random.split(key, B))
invalid = (jnp.argsort(perm, -1) >= 32).astype(jnp.int32)
root = mctx_ref.RootFnOutput(prior_logits=logits, value=value, embedding=latent)
qt = functools.partial(QT.qtransform_completed_by_mix_value, value_scale=1.0, maxvisit_init=50.0, rescale_values=False)

def timeit(name, f, *args):
    c = jax.jit(f).lower(*args).compile()
    jax.block_until_ready(c(*args))
    ts = []
    for _ in range(3):
        t0 = time.perf_counter(); jax.block_until_ready(c(*args)); ts.append(time.perf_counter() - t0)
    print(json.dumps(dict(part=name, n=n, ms=round(1000 * min(ts), 1))), flush=True)

full = lambda p, k: mctx_ref.gumbel_muzero_policy(p, k, root, recurrent, n, invalid, qtransform=qt, max_num_considered_actions=8)
pass
timeit("node_full_trivial_net", lambda p, k: eg.gumbel_muzero_policy_node(p, k, root, recurrent, n, invalid, qtransform=qt, max_num_considered_actions=8, return_tree=False).action, w, key)
pass  # timeit("flat_full_trivial_net", lambda p, k: eg.gumbel_muzero_policy_flat(p, k, root, recurrent, n, invalid, qtransform=qt, max_num_considered_actions=8, return_tree=False).action, w, key)
timeit("fast_full_trivial_net", lambda p, k: eg.gumbel_muzero_policy_fast(p, k, root, recurrent, n, invalid, qtransform=qt, max_num_considered_actions=8).action, w, key)
pass

sys.exit(0)
tree = jax.jit(lambda p, k: full(p, k).search_tree)(w, key)
asel = AS.switching_action_selection_wrapper(
    root_action_selection_fn=functools.partial(AS.gumbel_muzero_root_action_selection, num_simulations=n, max_num_considered_actions=8, qtransform=qt),
    interior_action_selection_fn=functools.partial(AS.gumbel_muzero_interior_action_selection, qtransform=qt))
rows = jnp.arange(B)
timeit("simulate", lambda t, k: S.simulate(jax.random.split(k, B), t, asel, n), tree, key)
par = jnp.zeros(B, jnp.int32); act = jnp.zeros(B, jnp.int32); nxt = jnp.full(B, n, jnp.int32)
timeit("expand(trivial net)", lambda t, k: S.expand(w, k, t, recurrent, par, act, nxt).node_values, tree, key)
timeit("backward_upstream", lambda t: S.backward(t, nxt).node_values, tree)
bw = eg._make_backward_fast(n + 1)
timeit("backward_fast", lambda t: bw(t, nxt).node_values, tree)
timeit("qtransform_root", lambda t: jax.vmap(qt, in_axes=[0, None])(t, 0), tree)
