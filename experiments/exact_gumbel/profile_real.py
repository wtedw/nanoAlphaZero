"""Per-op TPU breakdown of node2 with the production chess network (one chip)."""
import collections, functools, glob, gzip, json, os, re, sys, tempfile
import jax, jax.numpy as jnp, numpy as np
import mctx_ref
from mctx_ref._src import qtransforms as QT
import exact_gumbel as eg
from nanoalphazero.config import get_muzero_config
from nanoalphazero.model import make_muzero_model
n = int(sys.argv[1]) if len(sys.argv) > 1 else 32
B, A = 2048, 4672
cfg = get_muzero_config("chess", muzero_remat_blocks=False, muzero_remat_unroll=False)
cfg.update(game_obs_shape=[8, 8, 119], game_num_actions=A)
model, variables = make_muzero_model(cfg, jax.random.PRNGKey(0))
params = variables["params"]
key = jax.random.PRNGKey(1)
obs = jax.random.normal(key, (B, 8, 8, 119))
perm = jax.vmap(lambda k: jax.random.permutation(k, A))(jax.random.split(key, B))
invalid = (jnp.argsort(perm, -1) >= 32).astype(jnp.int32)
qt = functools.partial(QT.qtransform_completed_by_mix_value, value_scale=1.0, maxvisit_init=50.0, rescale_values=False)
def recurrent(params, rng, action, latent):
    latent, reward, logits, value = model.apply({"params": params}, latent, action, method=model.recurrent)
    return mctx_ref.RecurrentFnOutput(reward=jnp.zeros_like(value), discount=jnp.full_like(value, -1.0),
                                      prior_logits=logits, value=value), latent
def f(params, obs, invalid, key):
    latent, logits, value = model.apply({"params": params}, obs, method=model.initial)
    out = eg.gumbel_muzero_policy_node2(params, key, mctx_ref.RootFnOutput(prior_logits=logits, value=value, embedding=latent),
                                        recurrent, n, invalid, qtransform=qt, max_num_considered_actions=8, return_tree=False)
    return out.action_weights
c = jax.jit(f).lower(params, obs, invalid, key).compile(); jax.block_until_ready(c(params, obs, invalid, key))
d = tempfile.mkdtemp()
with jax.profiler.trace(d, create_perfetto_trace=True):
    jax.block_until_ready(c(params, obs, invalid, key))
events = json.load(gzip.open(glob.glob(os.path.join(d, "**", "*.trace.json.gz"), recursive=True)[0]))
events = events["traceEvents"] if isinstance(events, dict) else events
pids = {e["pid"]: e["args"]["name"] for e in events if e.get("ph") == "M" and e.get("name") == "process_name"}
tpu = {p for p, nm in pids.items() if "TPU:0" in nm}
tids = {(e["pid"], e["tid"]): e["args"]["name"] for e in events if e.get("ph") == "M" and e.get("name") == "thread_name"}
hlo = c.as_text()
def category(name):
    m = re.search(r"\n\s*(ROOT )?%?" + re.escape(name) + r" = [^\n]*?op_name=\"([^\"]*)\"", hlo)
    op = m.group(2) if m else ""
    if "apply" in op or "Conv" in op or "conv" in op or "Dense" in op or "Head" in op or "Trunk" in op or "Block" in op:
        return "network"
    return "search/other"
agg = collections.Counter(); cat = collections.Counter(); top = collections.Counter()
for e in events:
    if e.get("ph") == "X" and e["pid"] in tpu and "XLA Ops" in tids.get((e["pid"], e["tid"]), ""):
        if e["name"].startswith(("while", "conditional", "call")):
            continue
        cat[category(e["name"])] += e["dur"]
        if category(e["name"]) != "network":
            top[e["name"]] += e["dur"]
print({k: round(v / 1000, 1) for k, v in cat.items()}, "ms (leaf ops)")
for name, us in top.most_common(12):
    m = re.search(r"\n\s*(ROOT )?%?" + re.escape(name) + r" = ([^\n]{0,150})", hlo)
    print(f"{us/1000:7.1f} ms {name}: {m.group(2) if m else ''}")
