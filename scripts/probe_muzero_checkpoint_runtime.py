"""Measure full Hex4 save/restore and verify the next cycle on four devices."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from nanoalphazero import ckpt
from nanoalphazero.ckpt.compressed import _blocks, _slice
from nanoalphazero.config import get_muzero_config


def digest(tree):
    result = []
    for path, value in jax.tree_util.tree_flatten_with_path(tree)[0]:
        hashes = []
        for shard in value.addressable_shards:
            hasher = hashlib.sha256()
            for start, shape in _blocks(shard.data.shape, value.dtype.itemsize):
                block = (shard.data if shape == shard.data.shape else
                         _slice(shard.data, np.asarray(start, dtype=np.int32), shape))
                hasher.update(np.asarray(jax.device_get(block)).tobytes())
            hashes.append(hasher.hexdigest())
        result.append((jax.tree_util.keystr(path), str(value.dtype), value.shape, hashes))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location("muzero_probe", Path(__file__).parents[1] / "muzero.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = get_muzero_config("hex4", buffer_layout="features", diagnostic_period=0, eval_period=0)
    algorithm = module.make_muzero(config, jax.random.PRNGKey(0))
    state = algorithm.runner_state
    tick = time.monotonic()
    cycle = algorithm.run_fn.lower(state, jnp.bool_(True)).compile()
    report = {"devices": len(jax.devices()), "compile_seconds": time.monotonic() - tick}
    print(json.dumps(report), flush=True)
    for _ in range(config["replay_buffer_warmup_steps"] // config["cycle_n_selfplay"]):
        state, _ = cycle(state, jnp.bool_(True))
    state, metrics = cycle(state, jnp.bool_(False))
    jax.block_until_ready(state)
    assert float(metrics["selfplay_buffer/error_count"]) == 0
    assert int(state.model_ts.n_updates) > 0
    path = args.output / "state.ckpt"
    for name in ("first_save", "second_save"):
        report[name] = ckpt.save_checkpoint(path, state=state, cycle=1, config=algorithm.config)
        print(json.dumps({name: report[name]}), flush=True)
    report["stored_bytes"] = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    expected, expected_metrics = cycle(state, jnp.bool_(False))
    expected_hash = digest(expected)
    expected_metrics = jax.device_get(expected_metrics)
    # Restore owns/deletes these arrays; no duplicate full state remains on TPU.
    tick = time.monotonic()
    restored, start = ckpt.restore_checkpoint(path, state=expected, config=algorithm.config)
    report["restore_seconds"] = time.monotonic() - tick
    print(json.dumps({"restore_seconds": report["restore_seconds"]}), flush=True)
    assert start == 1
    actual, actual_metrics = cycle(restored, jnp.bool_(False))
    assert digest(actual) == expected_hash
    for key, value in expected_metrics.items():
        np.testing.assert_array_equal(value, np.asarray(actual_metrics[key]))
    report.update(next_cycle_exact=True, updates=int(actual.model_ts.n_updates),
                  errors=float(actual_metrics["selfplay_buffer/error_count"]))
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
