"""Bitwise comparison of exact_gumbel against upstream mctx.gumbel_muzero_policy."""
import functools
import itertools
import sys

import jax
import jax.numpy as jnp
import numpy as np

import mctx_ref
from mctx_ref._src import qtransforms as QT
import exact_gumbel as eg


def make_model(num_actions, latent=16, seed=0):
    k = jax.random.split(jax.random.PRNGKey(seed), 4)
    w = dict(
        g=jax.random.normal(k[0], (latent + num_actions, latent)) / 3,
        p=jax.random.normal(k[1], (latent, num_actions)) * 2,
        v=jax.random.normal(k[2], (latent,)) / 2,
        r=jax.random.normal(k[3], (latent,)) / 2,
    )

    def recurrent(params, rng, action, emb):
        x = jnp.concatenate([emb, jax.nn.one_hot(action, num_actions)], -1)
        h = jnp.tanh(x @ params["g"])
        out = mctx_ref.RecurrentFnOutput(
            reward=jnp.tanh(h @ params["r"]) * 0.3,
            discount=jnp.full(action.shape, -1.0),
            prior_logits=h @ params["p"], value=jnp.tanh(h @ params["v"]))
        return out, h
    return w, recurrent


def compare(B=64, A=12, n=24, m=8, value_scale=1.0, rescale=False, max_depth=None,
            seed=0, round_batched=False, gumbel_scale=1.0):
    params, recurrent = make_model(A, seed=seed)
    key = jax.random.PRNGKey(seed + 100)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    emb = jax.random.normal(k1, (B, 16))
    logits = jax.random.normal(k2, (B, A)) * 2
    value = jnp.tanh(jax.random.normal(k3, (B,)))
    # Varied legal counts, including 0, 1 and 2 legal actions.
    nlegal = jax.random.randint(k4, (B,), 0, A + 1).at[:4].set(jnp.array([0, 1, 2, A]))
    perm = jax.vmap(lambda k: jax.random.permutation(k, A))(jax.random.split(k4, B))
    invalid = (jnp.argsort(perm, -1) >= nlegal[:, None]).astype(jnp.int32)
    root = mctx_ref.RootFnOutput(prior_logits=logits, value=value, embedding=emb)
    qt = functools.partial(QT.qtransform_completed_by_mix_value, value_scale=value_scale,
                           maxvisit_init=50.0, rescale_values=rescale)
    rk = jax.random.PRNGKey(seed + 7)
    ref = jax.jit(lambda rk: mctx_ref.gumbel_muzero_policy(
        params, rk, root, recurrent, n, invalid, max_depth, qtransform=qt,
        max_num_considered_actions=m, gumbel_scale=gumbel_scale))(rk)
    out, info = jax.jit(lambda rk: eg.gumbel_muzero_policy_exact(
        params, rk, root, recurrent, n, invalid, max_depth, qtransform=qt,
        max_num_considered_actions=m, gumbel_scale=gumbel_scale,
        round_batched=round_batched))(rk)
    tr, tc = ref.search_tree, info["tree"]
    checks = {
        "action": np.array_equal(ref.action, out.action),
        "action_weights": np.array_equal(ref.action_weights, out.action_weights),
        "node_visits": np.array_equal(tr.node_visits, tc.visits),
        "node_values": np.array_equal(tr.node_values, tc.values),
        "children_index": np.array_equal(tr.children_index, tc.children_index),
        "root_visits": np.array_equal(tr.children_visits[:, 0], info["root_visits"]),
        "embeddings": np.array_equal(tr.embeddings, tc.embeddings),
    }
    mism = int(np.sum(info["mismatch"])) if "mismatch" in info else 0
    ok = all(checks.values()) and mism == 0
    bad = [k for k, v in checks.items() if not v]
    return ok, bad, mism, info.get("recurrent_calls")


if __name__ == "__main__":
    fails = 0
    for rb, (n, m), rescale, vs, md, seed in itertools.product(
            [False, True], [(12, 8), (24, 8), (32, 16), (16, 4), (64, 16), (7, 16)],
            [False, True], [1.0, 0.1], [None, 3], [0, 1]):
        ok, bad, mism, calls = compare(n=n, m=m, rescale=rescale, value_scale=vs,
                                       max_depth=md, seed=seed, round_batched=rb)
        tag = f"rb={rb} n={n} m={m} rescale={rescale} vs={vs} md={md} seed={seed}"
        if not ok:
            fails += 1
            print("FAIL", tag, bad, "mismatch_rows", mism, flush=True)
        elif rb and seed == 0 and vs == 1.0 and md is None and not rescale:
            print("ok  ", tag, "recurrent calls", calls, "vs", n, flush=True)
    print("failures:", fails)
    sys.exit(1 if fails else 0)
