"""Bounded ACPL smoke using AZ or full/partial MuZero checkpoint parameters.

Run through uv from the repository root. CPU is the default; TPU use checks the
lock owner before importing JAX. Repeats show first-call compilation vs steady
evaluation time. This is an integration check, not a strength measurement.
"""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("az", "mz"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--platform", choices=("cpu", "tpu"), default="cpu")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--max-plies", type=int, default=28)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stockfish", type=Path, default=Path("/usr/local/bin/stockfish"))
    args = parser.parse_args()
    if args.platform == "tpu":
        owners = subprocess.run(["fuser", "/tmp/libtpu_lockfile"], capture_output=True, text=True)
        if owners.returncode == 0:
            raise SystemExit(f"TPU lock is owned; no initialization attempted: {owners.stdout.strip()}")
    os.environ["JAX_PLATFORMS"] = args.platform
    if args.platform == "cpu":
        os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

    from nanoalphazero.distributed import DistributedArgs, initialize_distributed
    initialize_distributed(DistributedArgs(enabled=args.distributed))

    import jax
    import numpy as np
    from nanoalphazero.config import get_chess_config
    from nanoalphazero.core import make_env, DATA_PARALLEL_SHARDING, REPLICATED_SHARDING
    from nanoalphazero.model import make_model
    from nanoalphazero.mcts import make_mcts, make_muzero_mcts
    from nanoalphazero.checkpoint import load_checkpoint, apply_checkpoint_model_config
    from nanoalphazero.ckpt import load_model
    from nanoalphazero.eval.acpl import make_acpl, AcplConfig
    from nanoalphazero.eval.acpl.config import acpl_config_dict

    started = time.perf_counter()
    if args.algorithm == "az":
        params, metadata = load_checkpoint(str(args.checkpoint))
        config = apply_checkpoint_model_config(get_chess_config(), metadata)
    else:
        model, params, config = load_model(args.checkpoint, sharding=REPLICATED_SHARDING)
    config = {**config, "enable_sharding": True,
              "mcts_num_root_considered": 4, "mcts_num_survivors": 2}
    env = make_env(config)
    config.update(game_obs_shape=tuple(env.obs_shape), game_num_actions=env.num_actions)
    if args.algorithm == "az":
        model, _ = make_model(config, jax.random.PRNGKey(0), sharding=REPLICATED_SHARDING)
        params = jax.device_put(params, REPLICATED_SHARDING)
        policy_fn = make_mcts(config, env, model, DATA_PARALLEL_SHARDING)
    else:
        policy_fn = make_muzero_mcts(config, env, model, DATA_PARALLEL_SHARDING)
    config["acpl"] = acpl_config_dict(AcplConfig(
        period=1, stockfish_path=args.stockfish, output=args.output,
        num_games=args.games, max_plies=args.max_plies, depth=args.depth, workers=args.workers,
    ))
    config["acpl_checkpoint"] = str(args.checkpoint)
    eval_acpl_fn, acpl_state = make_acpl(config, env=env, policy_fn=policy_fn, data_sharding=DATA_PARALLEL_SHARDING)
    print(json.dumps({"setup_seconds": time.perf_counter() - started,
                      "devices": [str(d) for d in jax.devices()], "checkpoint": str(args.checkpoint)}), flush=True)
    previous_games = None
    for cycle in range(1, args.repeats + 1):
        acpl_state, result = eval_acpl_fn(acpl_state, cycle, params)
        metrics = result.metrics
        assert metrics["acpl/games"] == args.games and metrics["acpl/failed_games"] == 0
        assert metrics["acpl/analyzed_games"] == args.games
        assert all(np.isfinite(value) for value in metrics.values())
        assert bool(result.files) == (jax.process_index() == 0)
        directory = Path(acpl_state.last_output)
        records = [row for host in range(jax.process_count())
                   for row in json.loads((directory / f"scores-{host}.json").read_text())]
        assert sorted(row["game"] for row in records) == list(range(args.games))
        recorded = {p.name: p.read_bytes() for p in directory.glob("game-*.fens.json")}
        assert len(recorded) == args.games
        if previous_games is not None:
            assert recorded == previous_games, "Repeated evaluation changed deterministic trajectories"
        previous_games = recorded
        print(json.dumps({"cycle": cycle, "state": asdict(acpl_state), **metrics}), flush=True)


if __name__ == "__main__":
    main()
