"""make_exact_muzero_mcts on a real maz env/model equals upstream mctx (CPU; 1 or 4 devices)."""
import functools, json, sys
import jax, jax.numpy as jnp, numpy as np
import mctx_ref
from mctx_ref._src import qtransforms as QT
import exact_gumbel as eg
from maz_integration import make_exact_muzero_mcts
from nanoalphazero.config import get_muzero_config
from nanoalphazero.core import make_env
from nanoalphazero.model import make_muzero_model

env_name = sys.argv[1] if len(sys.argv) > 1 else "hex5"
B = 32
cfg = get_muzero_config(env_name, conv_width=16, conv_depth=2, muzero_remat_blocks=False,
                        muzero_remat_unroll=False, enable_sharding=False, mcts_num_simulations=16,
                        exp_bnk_action_weights=True)
wenv = make_env(cfg)
cfg.update(game_obs_shape=list(wenv.obs_shape), game_num_actions=wenv.num_actions)
model, variables = make_muzero_model(cfg, jax.random.PRNGKey(0))
params = variables["params"]
state = wenv.init(jax.random.split(jax.random.PRNGKey(1), B))
rng = np.random.default_rng(0)
for _ in range(6):  # advance to varied positions
    legal = np.asarray(state.legal_action_mask)
    act = np.array([rng.choice(np.nonzero(l)[0]) if l.any() else 0 for l in legal])
    state = wenv.step(state, jnp.asarray(act, jnp.int32))
mesh = jax.sharding.Mesh(np.array(jax.devices()), ("x",))
sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("x")) if len(jax.devices()) > 1 else None
run = make_exact_muzero_mcts(cfg, wenv, model, sharding)
key = jax.random.PRNGKey(3)
out = run(key, state, params, 1.0, B)

# Reference: upstream on the same root, unsharded.
qt = functools.partial(QT.qtransform_completed_by_mix_value, value_scale=cfg["mcts_value_scale"],
                       maxvisit_init=cfg["mcts_maxvisit_init"], rescale_values=cfg["mcts_rescale_values"])
def recurrent(params, rng, action, latent):
    latent, reward, logits, value = model.apply({"params": params}, latent, action, method=model.recurrent)
    if cfg.get("muzero_predict_value_only", False):
        reward = jnp.zeros_like(reward)
    return mctx_ref.RecurrentFnOutput(reward=reward, discount=jnp.full_like(value, -1.0),
                                      prior_logits=logits, value=value), latent
def root_of(state, params):
    obs = wenv.observe(state, state.current_player)
    latent, logits, value = model.apply({"params": params}, obs, method=model.initial)
    if cfg.get("exp_use_root_temperature", False):  # as make_muzero_mcts does
        logits = jnp.where(state.legal_action_mask, logits / cfg.get("exp_root_temperature", 1.3),
                           jnp.finfo(logits.dtype).min)
    return mctx_ref.RootFnOutput(prior_logits=logits, value=value, embedding=latent)

invalid = (~state.legal_action_mask).astype(jnp.int32)
root = jax.jit(root_of)(state, params)
D = len(jax.devices()); L = B // D
masked, rest_key, gumbel = eg._prepare_root(key, root, invalid, 1.0)
# Upstream run on each device's local batch (its output depends on batch size),
# with the globally drawn noise; for one device this is plain upstream.
refs = [jax.jit(lambda r, i, g: eg.upstream_with_noise(
            params, rest_key, r, recurrent, cfg["mcts_num_simulations"], i, g, qtransform=qt,
            max_num_considered_actions=cfg["mcts_num_root_considered"]))(
            jax.tree.map(lambda x: x[d*L:(d+1)*L], masked), invalid[d*L:(d+1)*L], gumbel[d*L:(d+1)*L])
        for d in range(D)]
class ref:
    action = np.concatenate([np.asarray(r.action) for r in refs])
    action_weights = np.concatenate([np.asarray(r.action_weights) for r in refs])
    class search_tree:
        children_visits = np.concatenate([np.asarray(r.search_tree.children_visits) for r in refs])
if D == 1:  # also check against the public upstream entry point
    pub = jax.jit(lambda k, r: mctx_ref.gumbel_muzero_policy(
        params, k, r, recurrent, cfg["mcts_num_simulations"], invalid, qtransform=qt,
        max_num_considered_actions=cfg["mcts_num_root_considered"]))(key, root)
    assert np.array_equal(pub.action_weights, ref.action_weights) and np.array_equal(pub.action, ref.action)
res = dict(env=env_name, devices=len(jax.devices()),
           action=bool(np.array_equal(ref.action, out.action)),
           action_weights=bool(np.array_equal(ref.action_weights, out.action_weights)),
           visit_counts=bool(np.array_equal(ref.search_tree.children_visits[:, 0], out.visit_counts)),
           bnk_fields=out.bnk_k_indices is not None and out.bnk_action_weights is not None,
           max_weight_diff=float(np.max(np.abs(np.asarray(ref.action_weights) - np.asarray(out.action_weights)))))
print(json.dumps(res))
sys.exit(0 if all(v for k, v in res.items() if k not in ("env", "devices", "max_weight_diff")) else 1)
