"""TPU: sharded node2 vs unsharded, and batch-size dependence of upstream and node2."""
import functools, json
import jax, jax.numpy as jnp, numpy as np
import mctx_ref
from mctx_ref._src import qtransforms as QT
import exact_gumbel as eg
from nanoalphazero.config import get_muzero_config
from nanoalphazero.model import make_muzero_model
mesh = jax.sharding.Mesh(np.array(jax.devices()), ("data",))
P = jax.sharding.PartitionSpec
data = jax.sharding.NamedSharding(mesh, P("data"))
cfg = get_muzero_config("chess", muzero_remat_blocks=False, muzero_remat_unroll=False)
cfg.update(game_obs_shape=[8, 8, 119], game_num_actions=4672)
model, variables = make_muzero_model(cfg, jax.random.PRNGKey(0))
params = variables["params"]; B, A, n, m = 1024, 4672, 12, 8
key = jax.random.PRNGKey(20)
obs = jax.random.normal(key, (B, 8, 8, 119))
perm = jax.vmap(lambda k: jax.random.permutation(k, A))(jax.random.split(key, B))
invalid = (jnp.argsort(perm, -1) >= 32).astype(jnp.int32)
qt = functools.partial(QT.qtransform_completed_by_mix_value, value_scale=1.0, maxvisit_init=50.0, rescale_values=False)
def recurrent(params, rng, action, latent):
    latent, reward, logits, value = model.apply({"params": params}, latent, action, method=model.recurrent)
    return mctx_ref.RecurrentFnOutput(reward=jnp.zeros_like(value), discount=jnp.full_like(value, -1.0),
                                      prior_logits=logits, value=value), latent
lat, lg, v = jax.jit(lambda p, o: model.apply({"params": p}, o, method=model.initial))(params, obs)
root = mctx_ref.RootFnOutput(prior_logits=lg, value=v, embedding=lat)
a = jax.jit(lambda r, i, k: eg.gumbel_muzero_policy_node2(params, k, r, recurrent, n, i, qtransform=qt, max_num_considered_actions=m, return_tree=False))(root, invalid, key)
# 1-device mesh shard_map (should equal unsharded exactly)
mesh1 = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("data",))
b1, _ = jax.jit(lambda r, i, k: eg.gumbel_muzero_policy_node2_sharded(mesh1, "data", params, k, r, recurrent, n, i, qtransform=qt, max_num_considered_actions=m))(root, invalid, key)
b4, _ = jax.jit(lambda r, i, k: eg.gumbel_muzero_policy_node2_sharded(mesh, "data", params, k, r, recurrent, n, i, qtransform=qt, max_num_considered_actions=m))(jax.device_put(root, data), jax.device_put(invalid, data), key)
for name, b in [("1dev", b1), ("4dev", b4)]:
    diff = np.any(np.asarray(a.action_weights) != np.asarray(b.action_weights), -1)
    print(name, "rows differing:", int(diff.sum()), "per shard:", [int(diff[i*256:(i+1)*256].sum()) for i in range(4)])
# Local-batch invariance of the search: rows 0..255 searched alone vs inside 1024.
sub = jax.tree.map(lambda x: x[:256], root)
key2 = jax.random.PRNGKey(5)
_, _, g = eg._prepare_root(key2, root, invalid, 1.0)
r_full = eg._prepare_root(key2, root, invalid, 1.0)[0]
f = jax.jit(lambda r, i, g: eg.gumbel_muzero_policy_node2_with_visits(params, key2, r, recurrent, n, i, qtransform=qt, max_num_considered_actions=m, return_tree=False, _prepared=g))
full, vf = f(r_full, invalid, g)
part, vp = f(jax.tree.map(lambda x: x[:256], r_full), invalid[:256], g[:256])
print("rows 0..255 alone vs in 1024 (node2): weights equal", bool(np.array_equal(full.action_weights[:256], part.action_weights)), "visits equal", bool(np.array_equal(vf[:256], vp)))
fu = jax.jit(lambda r, i, k: mctx_ref.gumbel_muzero_policy(params, k, r, recurrent, n, i, qtransform=qt, max_num_considered_actions=m))
u_full = fu(root, invalid, key); u_part = fu(sub, invalid[:256], key)
print("upstream rows 0..255 alone vs in 1024: tree visits equal", bool(np.array_equal(u_full.search_tree.children_visits[:256, 0], u_part.search_tree.children_visits[:, 0])))
# recurrent inside a loop context at batch 256 vs 1024
emb = r_full.embedding
loopf = jax.jit(lambda l, a: jax.lax.fori_loop(0, 3, lambda i, c: model.apply({"params": params}, c, a, method=model.recurrent)[0], l))
o1 = loopf(emb, jnp.zeros(1024, jnp.int32)); o2 = loopf(emb[:256], jnp.zeros(256, jnp.int32))
print("recurrent x3 in fori_loop, 256 vs 1024 rows equal:", bool(np.array_equal(o1[:256], o2)))
