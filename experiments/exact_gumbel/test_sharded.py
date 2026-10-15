"""Sharded node2 (shard_map over 4 devices) equals upstream on the global batch; visits match."""
import functools, itertools, json, sys
import jax, jax.numpy as jnp, numpy as np
import mctx_ref
from mctx_ref._src import qtransforms as QT
import exact_gumbel as eg
import test_exact as te

mesh = jax.sharding.Mesh(np.array(jax.devices()), ("data",))
fails = total = 0
for (n, m), rescale, seed in itertools.product([(12, 8), (32, 16), (7, 16)], [False, True], [0, 1]):
    B, A = 64, 12
    params, recurrent = te.make_model(A, seed=seed)
    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(seed + 100), 4)
    emb = jax.random.normal(k1, (B, 16)); logits = jax.random.normal(k2, (B, A)) * 2
    value = jnp.tanh(jax.random.normal(k3, (B,)))
    nlegal = jax.random.randint(k4, (B,), 0, A + 1).at[:4].set(jnp.array([0, 1, 2, A]))
    perm = jax.vmap(lambda k: jax.random.permutation(k, A))(jax.random.split(k4, B))
    invalid = (jnp.argsort(perm, -1) >= nlegal[:, None]).astype(jnp.int32)
    root = mctx_ref.RootFnOutput(prior_logits=logits, value=value, embedding=emb)
    qt = functools.partial(QT.qtransform_completed_by_mix_value, value_scale=1.0, maxvisit_init=50.0, rescale_values=rescale)
    rk = jax.random.PRNGKey(seed + 7)
    kw = dict(qtransform=qt, max_num_considered_actions=m)
    ref = jax.jit(lambda rk: mctx_ref.gumbel_muzero_policy(params, rk, root, recurrent, n, invalid, None, **kw))(rk)
    out, visits = jax.jit(lambda rk: eg.gumbel_muzero_policy_node2_sharded(
        mesh, "data", params, rk, root, recurrent, n, invalid, **kw))(rk)
    total += 1
    ok = (np.array_equal(ref.action, out.action) and np.array_equal(ref.action_weights, out.action_weights)
          and np.array_equal(ref.search_tree.children_visits[:, 0], visits))
    if not ok:
        fails += 1; print("FAIL", n, m, rescale, seed, flush=True)
print(json.dumps(dict(devices=len(jax.devices()), configs=total, failures=fails)))
sys.exit(1 if fails else 0)
