"""Time maz's real MuZero self-play steps (chess, all chips) with 1sh vs the exact drop-in."""
import json, sys, time
import jax
sys.path.insert(0, "/home/tedpsw/maz-muzero-review")
import muzero
import nanoalphazero.mcts as maz_mcts
from maz_integration import make_exact_muzero_mcts
from nanoalphazero.config import get_muzero_config

variant = sys.argv[1]            # "1sh" or "exact"
assert variant in ("1sh", "exact"), variant
sims = int(sys.argv[2]) if len(sys.argv) > 2 else 12
original = maz_mcts.make_muzero_mcts
if variant == "exact":
    maz_mcts.make_muzero_mcts = lambda config, env, model, data_sharding=None, **kw: \
        make_exact_muzero_mcts(config, env, model, data_sharding)
cfg = get_muzero_config("chess", enable_sharding=True, mcts_num_simulations=sims,
                        replay_buffer_total_size=1 << 16)
algo = muzero.make_muzero(cfg, jax.random.PRNGKey(0))
cfg = algo.config
st = algo.runner_state
sp, params = st.selfplay_state, st.model_ts.params
step = jax.jit(algo.selfplay_fn)
key = jax.random.PRNGKey(1)
t0 = time.perf_counter(); sp, out, _ = step(key, sp, params); jax.block_until_ready(out); compile_s = time.perf_counter() - t0
n = cfg["cycle_n_selfplay"]
t0 = time.perf_counter()
for i in range(n):
    sp, out, _ = step(jax.random.PRNGKey(i + 2), sp, params)
jax.block_until_ready(out)
dt = time.perf_counter() - t0
print(json.dumps(dict(search=variant if variant == "1sh" else f"exact {sims} sims",
                      steps=n, batch=cfg["selfplay_batch_size"], devices=len(jax.devices()),
                      ms_per_step=round(1000 * dt / n, 1), seconds_per_cycle_selfplay=round(dt, 2),
                      first_call_s=round(compile_s, 1))), flush=True)
