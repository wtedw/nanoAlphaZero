"""On TPU (all local chips): shard_map node2 vs upstream mctx on the global batch, chess network."""
import functools, json
import jax, jax.numpy as jnp, numpy as np
import mctx_ref
from mctx_ref._src import qtransforms as QT
import exact_gumbel as eg
from nanoalphazero.config import get_muzero_config
from nanoalphazero.model import make_muzero_model

mesh = jax.sharding.Mesh(np.array(jax.devices()), ("data",))
data = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))
cfg = get_muzero_config("chess", muzero_remat_blocks=False, muzero_remat_unroll=False)
cfg.update(game_obs_shape=[8, 8, 119], game_num_actions=4672)
model, variables = make_muzero_model(cfg, jax.random.PRNGKey(0))
params = variables["params"]
B, A = 1024, 4672

def recurrent(params, rng, action, latent):
    latent, reward, logits, value = model.apply({"params": params}, latent, action, method=model.recurrent)
    return mctx_ref.RecurrentFnOutput(reward=jnp.zeros_like(value), discount=jnp.full_like(value, -1.0),
                                      prior_logits=logits, value=value), latent

for n, m, rescale, legal in [(12, 8, False, 32), (32, 16, False, 3), (24, 8, True, 32)]:
    key = jax.random.PRNGKey(n + m)
    obs = jax.random.normal(key, (B, 8, 8, 119))
    perm = jax.vmap(lambda k: jax.random.permutation(k, A))(jax.random.split(key, B))
    nlegal = jnp.clip(jax.random.poisson(key, legal, (B,)), 0, A)
    invalid = (jnp.argsort(perm, -1) >= nlegal[:, None]).astype(jnp.int32)
    qt = functools.partial(QT.qtransform_completed_by_mix_value, value_scale=1.0, maxvisit_init=50.0, rescale_values=rescale)

    def root_of(params, obs):
        latent, logits, value = model.apply({"params": params}, obs, method=model.initial)
        return mctx_ref.RootFnOutput(prior_logits=logits, value=value, embedding=latent)

    # The root network's output depends on how its batch is sharded on TPU (a
    # model property, the same for any search), so both searches get one root.
    root = jax.jit(root_of)(params, obs)
    # Upstream's own output depends on the batch size on TPU, so the reference
    # is upstream run per device on that device's local batch, with the same
    # (globally drawn) Gumbel noise and masked prior as the sharded search.
    D = len(jax.devices()); L = B // D
    masked, _, gumbel = eg._prepare_root(key, root, invalid, 1.0)
    def upstream_local(p, r, i, g):
        # gumbel_muzero_policy re-masks (a no-op only on already-masked valid rows)
        # and draws its own noise, so call its search with ours instead.
        return eg.upstream_with_noise(p, key, r, recurrent, n, i, g, qtransform=qt, max_num_considered_actions=m)
    refs = [jax.jit(upstream_local)(params, jax.tree.map(lambda x: x[d*L:(d+1)*L], masked),
                                    invalid[d*L:(d+1)*L], gumbel[d*L:(d+1)*L]) for d in range(D)]
    class ref:  # concatenated per-device references
        action = np.concatenate([np.asarray(r.action) for r in refs])
        action_weights = np.concatenate([np.asarray(r.action_weights) for r in refs])
        class search_tree:
            children_visits = np.concatenate([np.asarray(r.search_tree.children_visits) for r in refs])
    r_s, i_s = jax.device_put(root, data), jax.device_put(invalid, data)
    out, visits = jax.jit(lambda p, r, i, k: eg.gumbel_muzero_policy_node2_sharded(
        mesh, "data", p, k, r, recurrent, n, i, qtransform=qt, max_num_considered_actions=m))(params, r_s, i_s, key)
    print(json.dumps(dict(n=n, m=m, rescale=rescale, devices=len(jax.devices()),
        action=bool(np.array_equal(ref.action, out.action)),
        action_weights=bool(np.array_equal(ref.action_weights, out.action_weights)),
        root_visits=bool(np.array_equal(ref.search_tree.children_visits[:, 0], visits)),
        max_abs_weight_diff=float(np.max(np.abs(np.asarray(ref.action_weights) - np.asarray(out.action_weights)))))), flush=True)
