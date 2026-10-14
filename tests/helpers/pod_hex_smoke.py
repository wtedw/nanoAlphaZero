"""Bounded MuZero training with explicit distributed state assertions.

uv run pod run --timeout 900 -- python tests/helpers/pod_hex_smoke.py --env hex4
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=("hex4", "hex8", "hex9"), required=True)
    parser.add_argument("--platform", choices=("cpu", "tpu"), default="tpu")
    parser.add_argument("--single-host", action="store_true")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--depth", type=int, default=1)
    args = parser.parse_args()
    os.environ["JAX_PLATFORMS"] = args.platform
    from nanoalphazero.distributed import DistributedArgs, initialize_distributed

    if not args.single_host:
        initialize_distributed(DistributedArgs(enabled=True))
    import jax
    import numpy as np
    from jax.experimental import multihost_utils
    from nanoalphazero.config import get_muzero_config

    path = Path(__file__).resolve().parents[2] / "muzero.py"
    spec = importlib.util.spec_from_file_location("muzero_smoke", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    config = get_muzero_config(
        args.env, num_iters=args.steps, replay_buffer_warmup_steps=100,
        selfplay_batch_size=args.batch, train_batch_size=args.batch,
        selfplay_buffer_consume_size=args.batch, selfplay_buffer_min_len=1,
        selfplay_buffer_max_len=256, replay_buffer_max_len=128,
        conv_width=args.width, conv_depth=args.depth,
        cycle_n_selfplay=10, cycle_n_train=2, diagnostic_period=0,
        eval_period=0, enable_trackio=False, enable_wandb=False,
    )
    algorithm = module.make_muzero(config, jax.random.PRNGKey(config["seed"]))

    def digest(tree):
        result = hashlib.sha256()
        for leaf in jax.tree.leaves(tree):
            if isinstance(leaf, jax.Array):
                assert leaf.is_fully_replicated, leaf.sharding
            result.update(np.asarray(leaf).tobytes())
        return result.digest()

    initial = digest(algorithm.runner_state.model_ts.params)
    state = module.run_muzero(algorithm)
    updates = int(state.model_ts.n_updates)
    assert updates == (args.steps // 10) * 2, updates
    assert digest(state.model_ts.params) != initial, "Parameters did not change"
    synchronized = digest((state.model_ts.params, state.model_ts.opt_state, state.rng))
    hashes = np.asarray(multihost_utils.process_allgather(np.frombuffer(synchronized, dtype=np.uint8)))
    assert np.all(hashes == hashes.reshape(-1, 32)[0]), "Model/optimizer/RNG differ across hosts"
    # PGX1 keeps the observation field as a placeholder; render current boards.
    observations = algorithm.env.observe(
        state.selfplay_state.env_state, state.selfplay_state.env_state.current_player,
    )
    lanes = sorted({i for shard in observations.addressable_shards
                    for i in range(*shard.index[0].indices(args.batch))})
    lane_mask = np.zeros(args.batch, dtype=np.int32)
    lane_mask[lanes] = 1
    ownership = np.asarray(multihost_utils.process_allgather(lane_mask)).reshape(-1, args.batch)
    assert np.all(ownership.sum(axis=0) == 1), "Self-play lanes duplicated or missing"
    local_boards = np.concatenate([np.asarray(s.data) for s in observations.addressable_shards])
    board_hash = hashlib.sha256(local_boards.tobytes()).digest()
    board_hashes = np.asarray(multihost_utils.process_allgather(np.frombuffer(board_hash, dtype=np.uint8)))
    if jax.process_count() > 1:
        assert len({bytes(row) for row in board_hashes.reshape(-1, 32)}) > 1, "All hosts have identical boards"
    print("MUZERO_POD_SMOKE " + json.dumps({
        "env": args.env, "process": jax.process_index(), "hosts": jax.process_count(),
        "devices": jax.device_count(), "local_devices": jax.local_device_count(),
        "global_batch": args.batch, "local_lanes": lanes, "updates": updates,
        "selfplay_steps": int(state.selfplay_state.step_count),
        "state_sha256": synchronized.hex(), "local_boards_sha256": board_hash.hex(),
        "selfplay_time_capacity": state.selfplay_buffer_state.experience.observation.shape[1],
        "replay_time_capacity": state.replay_buffer_state.experience.observation.shape[1],
    }), flush=True)
    if not args.single_host:
        jax.distributed.shutdown()


if __name__ == "__main__":
    main()
