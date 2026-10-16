"""Bounded pod search benchmark; uses production model/search and BNK outputs."""
import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import time

parser = argparse.ArgumentParser()
parser.add_argument('--batches', default='8192,16384,32768')
parser.add_argument('--repeats', type=int, default=5)
parser.add_argument('--variants', default='1sh,gumbel12,gumbel32')
parser.add_argument('--hlo-dir')
args = parser.parse_args()
os.environ['JAX_PLATFORMS'] = 'tpu'
from nanoalphazero.distributed import DistributedArgs, initialize_distributed
initialize_distributed(DistributedArgs(enabled=True))

import jax
import jax.numpy as jnp
from nanoalphazero.config import get_muzero_config
from nanoalphazero.core import make_env, DATA_PARALLEL_SHARDING, REPLICATED_SHARDING
from nanoalphazero.model import make_muzero_model
from nanoalphazero.mcts import make_muzero_mcts

config = get_muzero_config('chess', conv_width=384, conv_depth=10,
    muzero_chess_action_encoding='from_to8', muzero_predict_value_only=True,
    exp_bnk_action_weights=True, mcts_num_k_actions=128)
env = make_env(config)
config.update(game_obs_shape=list(env.obs_shape), game_num_actions=env.num_actions)
model, variables = make_muzero_model(config, jax.random.PRNGKey(0), REPLICATED_SHARDING)
params = variables['params']
for batch in map(int, args.batches.split(',')):
    state = jax.device_put(env.init_dummy_estate(batch), DATA_PARALLEL_SHARDING)
    for variant, simulations, considered in [('1sh', 12, 8), ('gumbel', 12, 8), ('gumbel', 32, 16), ('net', 32, 16)]:
        label = variant if variant == '1sh' else f'{variant}{simulations}'
        if label not in args.variants.split(','):
            continue
        cfg = dict(config, mcts_variant=variant, mcts_num_simulations=simulations,
                   mcts_num_root_considered=considered, mcts_num_survivors=4)
        search = (None if variant == 'net' else
                  make_muzero_mcts(cfg, env, model, DATA_PARALLEL_SHARDING))

        @jax.jit
        def run(key, state, params):
            if variant == 'net':
                observation = env.observe(state, state.current_player)
                latent, logits, value = model.apply({'params': params}, observation, method=model.initial)
                def step(_, carry):
                    latent, action, total = carry
                    latent, reward, logits, value = model.apply(
                        {'params': params}, latent, action, method=model.recurrent)
                    return latent, jnp.argmax(logits, axis=-1), total + value
                return jax.lax.fori_loop(0, simulations, step,
                    (latent, jnp.argmax(logits, axis=-1), value))
            output = search(key, state, params, 1., batch)
            # Match the compact outputs consumed by production self-play.
            return output.action, output.bnk_action_weights, output.bnk_k_indices

        key = jax.device_put(jax.random.PRNGKey(1), REPLICATED_SHARDING)
        start = time.perf_counter()
        executable = run.lower(key, state, params).compile()
        compile_seconds = time.perf_counter() - start
        memory = executable.memory_analysis()
        if args.hlo_dir and jax.process_index() == 0:
            destination = Path(args.hlo_dir)
            destination.mkdir(parents=True, exist_ok=True)
            (destination / f'{label}-{batch}.hlo').write_text(executable.as_text())
        times = []
        for repeat in range(3 + args.repeats):
            start = time.perf_counter()
            result = executable(key, state, params)
            jax.block_until_ready(result)
            seconds = time.perf_counter() - start
            if repeat >= 3:
                times.append(seconds)
        print(json.dumps(dict(rank=jax.process_index(), devices=jax.device_count(),
            batch=batch, per_device=batch // jax.device_count(), variant=variant,
            simulations=simulations, considered=considered, max_depth=None,
            return_tree=False, compile_seconds=compile_seconds,
            seconds=times, median_seconds=statistics.median(times),
            positions_per_second=batch / statistics.median(times),
            temp_bytes=memory.temp_size_in_bytes,
            argument_bytes=memory.argument_size_in_bytes,
            output_bytes=memory.output_size_in_bytes)), flush=True)
        del result, executable, run, search
        jax.clear_caches()
        gc.collect()
    del state
