import functools, re, sys, collections
import jax, jax.numpy as jnp
ARGS = list(sys.argv)
sys.argv = ARGS[:2]; exec(open(__file__.replace("hlo_copies.py", "bench_parts.py")).read().split("def timeit")[0])
sys.argv = ARGS
full = lambda p, k: mctx_ref.gumbel_muzero_policy(p, k, root, recurrent, n, invalid, qtransform=qt, max_num_considered_actions=8)
import exact_gumbel as eg
if len(sys.argv) > 2: full = lambda p, k: getattr(eg, sys.argv[2])(p, k, root, recurrent, n, invalid, qtransform=qt, max_num_considered_actions=8)
f = jax.jit(lambda p, k: full(p, k).action)
txt = f.lower(w, key).compile().as_text()
open(__file__.replace("hlo_copies.py", "upstream_hlo.txt"), "w").write(txt)
big = collections.Counter()
for line in txt.splitlines():
    m = re.search(r"= (\w+)\[([\d,]+)\][^ ]* (copy|select|dynamic-update-slice|scatter|fusion|copy-start)\(", line)
    if m:
        dims = [int(x) for x in m.group(2).split(",")]
        size = 1
        for d in dims: size *= d
        if size >= 2048 * 13 * 4672 // 2:
            big[(m.group(3), m.group(1), m.group(2))] += 1
for k, v in big.most_common(30):
    print(v, k)
