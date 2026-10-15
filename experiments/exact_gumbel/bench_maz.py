"""Self-play search call on real chess states, 4 chips: maz 1sh make_muzero_mcts vs make_exact_muzero_mcts."""
import json, sys, time
import jax, jax.numpy as jnp, numpy as np
from maz_integration import make_exact_muzero_mcts
from nanoalphazero.config import get_muzero_config
from nanoalphazero.core import make_env, DATA_PARALLEL_SHARDING
from nanoalphazero.mcts import make_muzero_mcts
from nanoalphazero.model import make_muzero_model

sims = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "12,32").split(",")]
cfg = get_muzero_config("chess", muzero_remat_blocks=False, muzero_remat_unroll=False)
wenv = make_env(cfg)
cfg.update(game_obs_shape=list(wenv.obs_shape), game_num_actions=wenv.num_actions)
B = cfg["selfplay_batch_size"]
model, variables = make_muzero_model(cfg, jax.random.PRNGKey(0))
params = variables["params"]
state = wenv.init(jax.random.split(jax.random.PRNGKey(1), B))
state = jax.device_put(state, DATA_PARALLEL_SHARDING)
key = jax.random.PRNGKey(2)

def timeit(name, fn, n=None):
    out = fn(key, state, params, 1.0, B); jax.block_until_ready(out)
    ts = []
    for r in range(3):
        t0 = time.perf_counter(); jax.block_until_ready(fn(jax.random.PRNGKey(r), state, params, 1.0, B)); ts.append(time.perf_counter() - t0)
    print(json.dumps(dict(search=name, sims=n, global_batch=B, devices=len(jax.devices()),
                          ms=round(1000 * min(ts), 1))), flush=True)

timeit("maz 1sh (8+4)", make_muzero_mcts(cfg, wenv, model, DATA_PARALLEL_SHARDING), "8+4")
for n in sims:
    c = dict(cfg, mcts_num_simulations=n)
    timeit("exact node2 (shard_map)", make_exact_muzero_mcts(c, wenv, model, DATA_PARALLEL_SHARDING), n)
