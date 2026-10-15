"""Search time with the global batch sharded over all local devices (maz data parallelism)."""
import functools, json, sys, time
import jax, jax.numpy as jnp, numpy as np
import mctx_ref
from mctx_ref._src import qtransforms as QT
import exact_gumbel as eg
from nanoalphazero.config import get_muzero_config
from nanoalphazero.mcts import RootFnOutput as MazRoot, gumbel_muzero_policy_1sh, RecurrentFnOutput as MazRec
from nanoalphazero.model import make_muzero_model

B = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
sims = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "12,32").split(",")]
mesh = jax.sharding.Mesh(np.array(jax.devices()), ("data",))
data = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))
repl = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
cfg = get_muzero_config("chess", muzero_remat_blocks=False, muzero_remat_unroll=False)
cfg.update(game_obs_shape=[8, 8, 119], game_num_actions=4672)
model, variables = make_muzero_model(cfg, jax.random.PRNGKey(0))
params = jax.device_put(variables["params"], repl)
A = 4672
key = jax.random.PRNGKey(1)
obs = jax.device_put(jax.random.normal(key, (B, 8, 8, 119)), data)
perm = jax.vmap(lambda k: jax.random.permutation(k, A))(jax.random.split(key, B))
invalid = jax.device_put((jnp.argsort(perm, -1) >= 32).astype(jnp.int32), data)
qt = functools.partial(QT.qtransform_completed_by_mix_value, value_scale=1.0, maxvisit_init=50.0, rescale_values=False)

def recurrent(params, rng, action, latent):
    latent, reward, logits, value = model.apply({"params": params}, latent, action, method=model.recurrent)
    return mctx_ref.RecurrentFnOutput(reward=jnp.zeros_like(value), discount=jnp.full_like(value, -1.0),
                                      prior_logits=logits, value=value), latent

def make(variant, n):
    def f(params, obs, invalid, key):
        latent, logits, value = model.apply({"params": params}, obs, method=model.initial)
        if variant == "1sh":
            out = gumbel_muzero_policy_1sh(
                params, key, MazRoot(prior_logits=logits, value=value, embedding=latent),
                lambda p, k, a, l: (lambda o: (MazRec(reward=o[0].reward, discount=o[0].discount,
                    prior_logits=o[0].prior_logits, value=o[0].value), o[1]))(recurrent(p, None, a, l)),
                num_root_considered=8, num_survivors=4, invalid_actions=invalid.astype(bool),
                gumbel_scale=1.0, value_scale=1.0, maxvisit_init=50.0, rescale_values=False)
            return out.action, out.action_weights
        r = mctx_ref.RootFnOutput(prior_logits=logits, value=value, embedding=latent)
        if variant == "node2_shard_map":
            out, _ = eg.gumbel_muzero_policy_node2_sharded(
                mesh, "data", params, key, r, recurrent, n, invalid, qtransform=qt,
                max_num_considered_actions=8)
        else:
            out = eg.gumbel_muzero_policy_node2(params, key, r, recurrent, n, invalid, qtransform=qt,
                                                max_num_considered_actions=8, return_tree=False)
        return out.action, out.action_weights
    return jax.jit(f, out_shardings=(data, data))

for n in sims:
    for variant in (["1sh"] if n == sims[0] else []) + ["node2", "node2_shard_map"]:
        c = make(variant, n).lower(params, obs, invalid, key).compile()
        hlo = c.as_text()
        jax.block_until_ready(c(params, obs, invalid, key)); ts = []
        for r in range(3):
            t0 = time.perf_counter(); jax.block_until_ready(c(params, obs, invalid, jax.random.PRNGKey(r))); ts.append(time.perf_counter() - t0)
        print(json.dumps(dict(variant=variant, n=n if variant != "1sh" else "8+4", global_batch=B,
                              devices=len(jax.devices()), ms=round(1000 * min(ts), 1),
                              all_gathers=hlo.count("all-gather"), all_reduces=hlo.count("all-reduce"),
                              temp_gib_per_device=round(c.memory_analysis().temp_size_in_bytes / 2**30, 2))), flush=True)
