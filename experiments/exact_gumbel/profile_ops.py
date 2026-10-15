"""Per-op TPU time breakdown of the search (trivial network), from a perfetto trace."""
import collections, functools, glob, gzip, json, os, sys, tempfile
import jax
sys.argv, ARGS = sys.argv[:1] + sys.argv[1:2], sys.argv
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tree_bench.py")).read().split("configs = ")[0])
n = int(ARGS[1]); kw = json.loads(ARGS[2]) if len(ARGS) > 2 else {}
fn = kw.pop("fn", "gumbel_muzero_policy_node")
f = jax.jit(lambda p, k: getattr(eg, fn)(p, k, root, recurrent, n, invalid, qtransform=qt,
                                          max_num_considered_actions=8, return_tree=False, **kw).action_weights)
c = f.lower(w, key).compile(); jax.block_until_ready(c(w, key))
d = tempfile.mkdtemp()
with jax.profiler.trace(d, create_perfetto_trace=True):
    jax.block_until_ready(c(w, key))
path = glob.glob(os.path.join(d, "**", "*.trace.json.gz"), recursive=True)[0]
ev = json.load(gzip.open(path))
events = ev["traceEvents"] if isinstance(ev, dict) else ev
pids = {e["pid"]: e["args"]["name"] for e in events if e.get("ph") == "M" and e.get("name") == "process_name"}
tpu = {p for p, nm in pids.items() if "TPU:0" in nm}
tids = {(e["pid"], e["tid"]): e["args"]["name"] for e in events if e.get("ph") == "M" and e.get("name") == "thread_name"}
agg = collections.Counter(); cnt = collections.Counter(); total = 0
for e in events:
    if e.get("ph") == "X" and e["pid"] in tpu and "XLA Ops" in tids.get((e["pid"], e["tid"]), ""):
        name = e["name"]
        agg[name] += e["dur"]; cnt[name] += 1; total += e["dur"]
print(f"total XLA op time {total/1000:.1f} ms")
for name, us in agg.most_common(25):
    print(f"{us/1000:8.1f} ms  {cnt[name]:6d}x  {name[:110]}")
hlo = c.as_text()
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "node_hlo.txt"), "w").write(hlo)
import re
for name, us in agg.most_common(40):
    if name.startswith("while"):
        continue
    m = re.search(r"\n\s*(ROOT )?%?" + re.escape(name) + r" = ([^\n]{0,220})", hlo)
    if m:
        print(f"{us/1000:7.1f} ms {name}: {m.group(2)[:200]}")
