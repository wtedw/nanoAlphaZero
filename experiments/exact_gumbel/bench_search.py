"""Single-device timing of MuZero search variants with the production chess network."""
import argparse
import functools
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

import mctx  # installed fork (6cf1a39)
import mctx_ref
from mctx_ref._src import qtransforms as QT

import exact_gumbel as eg
from nanoalphazero.config import get_muzero_config
from nanoalphazero.mcts import RootFnOutput as MazRoot, gumbel_muzero_policy_1sh, RecurrentFnOutput as MazRec
from nanoalphazero.model import make_muzero_model

p = argparse.ArgumentParser()
p.add_argument("--env", default="chess")
p.add_argument("--batch", type=int, default=2048)
p.add_argument("--sims", default="12,24,48")
p.add_argument("--m", type=int, default=8)
p.add_argument("--variants", default="net,1sh,upstream,fork,exact,exact_rb")
p.add_argument("--reps", type=int, default=5)
p.add_argument("--legal", type=int, default=32)
p.add_argument("--k", default="128,32")
a = p.parse_args()

dev = jax.devices()[0]
cfg = get_muzero_config(a.env, muzero_remat_blocks=False, muzero_remat_unroll=False)
obs_shape = {"chess": (8, 8, 119), "hex9": (9, 9, 4), "go9": (9, 9, 17)}[a.env]
num_actions = {"chess": 4672, "hex9": 81, "go9": 82}[a.env]
cfg.update(game_obs_shape=list(obs_shape), game_num_actions=num_actions)
model, variables = make_muzero_model(cfg, jax.random.PRNGKey(0))
params = jax.device_put(variables["params"], dev)
B, A = a.batch, num_actions

key = jax.random.PRNGKey(1)
obs = jax.device_put(jax.random.normal(key, (B, *obs_shape)), dev)
perm = jax.vmap(lambda k: jax.random.permutation(k, A))(jax.random.split(key, B))
nlegal = jnp.clip(jax.random.poisson(key, a.legal, (B,)), 1, A)
invalid = jax.device_put((jnp.argsort(perm, -1) >= nlegal[:, None]).astype(jnp.int32), dev)


def recurrent(params, rng, action, latent):
    latent, reward, logits, value = model.apply({"params": params}, latent, action,
                                                method=model.recurrent)
    return mctx_ref.RecurrentFnOutput(reward=jnp.zeros_like(value),
                                      discount=jnp.full_like(value, -1.0),
                                      prior_logits=logits, value=value), latent


def recurrent_fork(params, rng, action, latent):
    o, l = recurrent(params, rng, action, latent)
    return mctx.RecurrentFnOutput(reward=o.reward, discount=o.discount,
                                  prior_logits=o.prior_logits, value=o.value), l


def recurrent_maz(params, keys, action, latent):
    o, l = recurrent(params, None, action, latent)
    return MazRec(reward=o.reward, discount=o.discount, prior_logits=o.prior_logits,
                  value=o.value), l


def root_of(params, obs):
    latent, logits, value = model.apply({"params": params}, obs, method=model.initial)
    return latent, logits, value


qt = functools.partial(QT.qtransform_completed_by_mix_value, value_scale=1.0,
                       maxvisit_init=50.0, rescale_values=False)


def make(variant, n):
    m = a.m
    if variant == "net":
        # Lower bound: root + n sequential recurrent calls at batch B, no tree.
        def f(params, obs, invalid, key):
            latent, logits, value = root_of(params, obs)
            action = jnp.argmax(logits, -1)
            def body(i, c):
                lat, acc = c
                o, lat = recurrent(params, None, action, lat)
                return lat, acc + o.value
            lat, acc = jax.lax.fori_loop(0, n, body, (latent, jnp.zeros_like(value)))
            return acc
        return f
    if variant == "1sh":
        def f(params, obs, invalid, key):
            latent, logits, value = root_of(params, obs)
            out = gumbel_muzero_policy_1sh(
                params, key, MazRoot(prior_logits=logits, value=value, embedding=latent),
                recurrent_maz, num_root_considered=m, num_survivors=m // 2,
                invalid_actions=invalid.astype(bool), gumbel_scale=1.0, value_scale=1.0,
                maxvisit_init=50.0, rescale_values=False)
            return out.action, out.action_weights
        return f
    if variant == "upstream":
        def f(params, obs, invalid, key):
            latent, logits, value = root_of(params, obs)
            out = mctx_ref.gumbel_muzero_policy(
                params, key, mctx_ref.RootFnOutput(prior_logits=logits, value=value, embedding=latent),
                recurrent, n, invalid, qtransform=qt, max_num_considered_actions=m)
            return out.action, out.action_weights
        return f
    if variant == "fork":
        fqt = functools.partial(mctx.qtransform_completed_by_mix_value, value_scale=1.0,
                                maxvisit_init=50.0, rescale_values=False)
        def f(params, obs, invalid, key):
            latent, logits, value = root_of(params, obs)
            out = mctx.gumbel_muzero_policy(
                params, key, mctx.RootFnOutput(prior_logits=logits, value=value, embedding=latent),
                recurrent_fork, n, invalid, qtransform=fqt, max_num_considered_actions=m)
            return out.action, out.action_weights
        return f
    if variant.startswith("opt"):
        K = int(variant[3:])
        fqt = functools.partial(mctx.qtransform_completed_by_mix_value, value_scale=1.0,
                                maxvisit_init=50.0, rescale_values=False)
        fin = functools.partial(mctx.qtransform_completed_by_mix_value, value_scale=1.0,
                                maxvisit_init=50.0, rescale_values=False, return_extras=True)
        def f(params, obs, invalid, key):
            latent, logits, value = root_of(params, obs)
            out = mctx.gumbel_muzero_policy_opt(
                params, key, mctx.RootFnOutput(prior_logits=logits, value=value, embedding=latent),
                recurrent_fork, K, n, invalid, qtransform=fqt, final_qtransform=fin,
                max_num_considered_actions=m)
            return out.action, out.action_weights
        return f
    if variant == "node2":
        def f(params, obs, invalid, key):
            latent, logits, value = root_of(params, obs)
            out = eg.gumbel_muzero_policy_node2(
                params, key, mctx_ref.RootFnOutput(prior_logits=logits, value=value, embedding=latent),
                recurrent, n, invalid, qtransform=qt, max_num_considered_actions=m, return_tree=False)
            return out.action, out.action_weights
        return f
    if variant == "node_tree":
        def f(params, obs, invalid, key):
            latent, logits, value = root_of(params, obs)
            out = eg.gumbel_muzero_policy_node(
                params, key, mctx_ref.RootFnOutput(prior_logits=logits, value=value, embedding=latent),
                recurrent, n, invalid, qtransform=qt, max_num_considered_actions=m)  # default return_tree=True
            return out.action, out.action_weights, out.search_tree
        return f
    if variant == "node_depth":
        def f(params, obs, invalid, key):
            latent, logits, value = root_of(params, obs)
            out = eg.gumbel_muzero_policy_node(
                params, key, mctx_ref.RootFnOutput(prior_logits=logits, value=value, embedding=latent),
                recurrent, n, invalid, max_depth=n - 1, qtransform=qt, max_num_considered_actions=m,
                return_tree=False)
            return out.action, out.action_weights
        return f
    if variant == "node":
        def f(params, obs, invalid, key):
            latent, logits, value = root_of(params, obs)
            out = eg.gumbel_muzero_policy_node(
                params, key, mctx_ref.RootFnOutput(prior_logits=logits, value=value, embedding=latent),
                recurrent, n, invalid, qtransform=qt, max_num_considered_actions=m, return_tree=False)
            return out.action, out.action_weights
        return f
    if variant == "fastbw":
        def f(params, obs, invalid, key):
            latent, logits, value = root_of(params, obs)
            out = eg.gumbel_muzero_policy_fastbw(
                params, key, mctx_ref.RootFnOutput(prior_logits=logits, value=value, embedding=latent),
                recurrent, n, invalid, qtransform=qt, max_num_considered_actions=m)
            return out.action, out.action_weights
        return f
    if variant in ("exact", "exact_rb"):
        def f(params, obs, invalid, key):
            latent, logits, value = root_of(params, obs)
            out, info = eg.gumbel_muzero_policy_exact(
                params, key, mctx_ref.RootFnOutput(prior_logits=logits, value=value, embedding=latent),
                recurrent, n, invalid, qtransform=qt, max_num_considered_actions=m,
                round_batched=variant == "exact_rb")
            return out.action, out.action_weights
        return f
    raise ValueError(variant)


results = []
for n in [int(x) for x in a.sims.split(",")]:
    for variant in a.variants.split(","):
        if variant == "1sh" and n != a.sims.split(",")[0] and n != int(a.sims.split(",")[0]):
            continue
        f = jax.jit(make(variant, n))
        try:
            t0 = time.perf_counter()
            compiled = f.lower(params, obs, invalid, key).compile()
            ctime = time.perf_counter() - t0
            mem = compiled.memory_analysis()
            temp = getattr(mem, "temp_size_in_bytes", 0) / 2**30
            out = compiled(params, obs, invalid, key)
            jax.block_until_ready(out)
            times = []
            for r in range(a.reps):
                t0 = time.perf_counter()
                jax.block_until_ready(compiled(params, obs, invalid, jax.random.PRNGKey(r)))
                times.append(time.perf_counter() - t0)
            row = dict(variant=variant, n=n if variant != "1sh" else f"{a.m}+{a.m//2}",
                       ms=round(1000 * float(np.median(times)), 2), compile_s=round(ctime, 1),
                       temp_gib=round(temp, 2))
        except Exception as e:  # noqa: BLE001
            row = dict(variant=variant, n=n, error=type(e).__name__ + ": " + str(e)[:200])
        print(json.dumps(row), flush=True)
        results.append(row)
