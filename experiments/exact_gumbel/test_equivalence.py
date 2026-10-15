import functools, itertools, sys
import jax, numpy as np
import mctx_ref
from mctx_ref._src import qtransforms as QT
import exact_gumbel as eg
import test_exact as te
import jax.numpy as jnp

VARIANT = sys.argv[1] if len(sys.argv) > 1 else "gumbel_muzero_policy_node"
fails = 0
for (n, m), rescale, vs, md, seed in itertools.product(
        [(12, 8), (24, 8), (32, 16), (16, 4), (64, 16), (7, 16)], [False, True], [1.0, 0.1], [None, 3], [0, 1, 2]):
    B, A = 64, 12
    params, recurrent = te.make_model(A, seed=seed)
    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(seed + 100), 4)
    emb = jax.random.normal(k1, (B, 16)); logits = jax.random.normal(k2, (B, A)) * 2
    value = jnp.tanh(jax.random.normal(k3, (B,)))
    nlegal = jax.random.randint(k4, (B,), 0, A + 1).at[:4].set(jnp.array([0, 1, 2, A]))
    perm = jax.vmap(lambda k: jax.random.permutation(k, A))(jax.random.split(k4, B))
    invalid = (jnp.argsort(perm, -1) >= nlegal[:, None]).astype(jnp.int32)
    root = mctx_ref.RootFnOutput(prior_logits=logits, value=value, embedding=emb)
    qt = functools.partial(QT.qtransform_completed_by_mix_value, value_scale=vs, maxvisit_init=50.0, rescale_values=rescale)
    rk = jax.random.PRNGKey(seed + 7)
    kw = dict(qtransform=qt, max_num_considered_actions=m)
    ref = jax.jit(lambda rk: mctx_ref.gumbel_muzero_policy(params, rk, root, recurrent, n, invalid, md, **kw))(rk)
    out = jax.jit(lambda rk: getattr(eg, VARIANT)(params, rk, root, recurrent, n, invalid, md, **kw))(rk)
    same = all(np.array_equal(x, y) for x, y in zip(jax.tree.leaves(ref), jax.tree.leaves(out)))
    if not same:
        fails += 1
        print("FAIL", n, m, rescale, vs, md, seed, flush=True)
print("configs:", 6 * 2 * 2 * 2 * 3, "failures:", fails)
sys.exit(1 if fails else 0)
