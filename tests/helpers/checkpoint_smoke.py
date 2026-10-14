"""Actual multihost checkpoint I/O and deterministic MuZero continuation.

CPU: launched by test_distributed_checkpoint.py. TPU: pod run this script with
--platform tpu --directory <new shared test directory> --case chess.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", default="cpu")
    parser.add_argument("--rank", type=int)
    parser.add_argument("--port")
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--case", choices=("storage", "ttt", "chess"), required=True)
    args = parser.parse_args()
    os.environ["JAX_PLATFORMS"] = args.platform
    if args.platform == "cpu":
        cpus = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, cpus[args.rank * 2:args.rank * 2 + 2] or cpus[:2])
    from nanoalphazero.distributed import DistributedArgs, initialize_distributed
    explicit = {} if args.port is None else dict(
        coordinator_address=f"127.0.0.1:{args.port}", num_processes=2, process_id=args.rank)
    initialize_distributed(DistributedArgs(enabled=True, **explicit))
    import jax
    import jax.numpy as jnp
    import numpy as np
    from nanoalphazero import ckpt
    from nanoalphazero.ckpt import compressed
    from nanoalphazero.ckpt.distributed import collectively
    rank = jax.process_index()
    path = args.directory / "state.ckpt"

    def exact(left, right):
        a, ta = jax.tree.flatten(left)
        b, tb = jax.tree.flatten(right)
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert x.shape == y.shape and x.dtype == y.dtype
            assert x.sharding.is_equivalent_to(y.sharding, x.ndim)
            if jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key):
                x, y = jax.random.key_data(x), jax.random.key_data(y)
            for sx, sy in zip(x.addressable_shards, y.addressable_shards):
                assert sx.index == sy.index
                assert np.asarray(sx.data).tobytes() == np.asarray(sy.data).tobytes()

    if args.case == "storage":
        compressed.CHUNK_BYTES = 32
        mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ("data",))
        sharded = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))
        replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        value = jax.make_array_from_callback((4, 21), sharded,
            lambda index: np.arange(84, dtype=np.float32).reshape(4, 21)[index])
        def replicate(value):
            host = np.asarray(value)
            return jax.make_array_from_callback(host.shape, replicated, lambda i: host[i])
        tree = {"runner_state": {"buffer": value,
                "rng": jax.random.wrap_key_data(replicate(jax.random.key_data(jax.random.key(7)))),
                "counter": replicate(jnp.int32(9)),
                "empty": jax.make_array_from_callback((4, 0), sharded, lambda i: np.empty((4, 0), np.float32)[i]),
                "bf16": replicate(jnp.array([0., -0., jnp.nan, jnp.inf], jnp.bfloat16))},
                "cycle": jnp.int32(3)}
        compressed.save(path, tree, {}, metadata={"trackio": {"id": "resume-me"}})
        back, _ = compressed.load(path, tree)
        collectively(lambda: exact(tree, back), "exact restore")
        manifest = compressed.inspect_checkpoint(path)
        for field in manifest["fields"]:
            assert len(field["shards"]) == (jax.device_count() if field["path"].endswith("['buffer']") or
                                           field["path"].endswith("['empty']") else 1)
            assert all(chunk["bytes"] <= 32 for shard in field["shards"] for chunk in shard["chunks"])
        previous = (path / "LATEST").read_bytes()
        try:
            compressed.save(path, dict(tree, cycle=jnp.int32(rank)), {})
        except RuntimeError as exc:
            assert "local replicated values" in str(exc)
        else:
            raise AssertionError("Divergent host-local counter was accepted")
        assert (path / "LATEST").read_bytes() == previous
        write = compressed._write
        for stage in ("chunk", "publication"):
            def fail(destination, content):
                if (stage == "chunk" and rank == 1 and destination.suffix == ".zst") or (
                    stage == "publication" and rank == 0 and destination.name == "LATEST.tmp"
                ):
                    raise OSError("injected failure")
                return write(destination, content)
            compressed._write = fail
            try:
                compressed.save(path, tree, {})
            except RuntimeError as exc:
                assert "injected failure" in str(exc)
            else:
                raise AssertionError("Failure was not propagated")
            finally:
                compressed._write = write
            assert (path / "LATEST").read_bytes() == previous
            back, _ = compressed.load(path, tree)
            collectively(lambda: exact(tree, back), "restore previous")
        compressed.save(path, tree, {})
        assert len(list(path.glob("gen-*"))) == 1
        # Reject a changed host/device topology before allocating restored arrays.
        manifest_path = compressed._generation(path) / "manifest.json"
        original_manifest = manifest_path.read_text()
        def wrong_topology():
            if rank == 0:
                manifest = json.loads(original_manifest)
                manifest["topology"]["processes"] += 1
                manifest_path.write_text(json.dumps(manifest))
        collectively(wrong_topology, "inject topology mismatch")
        try:
            compressed.load(path, tree)
        except RuntimeError as exc:
            assert "topology differs" in str(exc)
        else:
            raise AssertionError("Topology mismatch was accepted")
        collectively(lambda: manifest_path.write_text(original_manifest) if rank == 0 else None,
                     "restore manifest")
        # A bad shard must fail collectively, even though only one host reads it.
        manifest = compressed.inspect_checkpoint(path)
        generation = compressed._generation(path)
        def corrupt():
            if rank == 0:
                chunk = next(f for f in manifest["fields"] if f["path"].endswith("['buffer']"))["shards"][-1]["chunks"][0]
                (generation / chunk["file"]).write_bytes(b"corrupt")
        collectively(corrupt, "inject corruption")
        try:
            compressed.load(path, tree)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Corruption was not propagated")
    else:
        from nanoalphazero.config import get_muzero_config
        spec = importlib.util.spec_from_file_location("checkpoint_muzero", Path(__file__).resolve().parents[2] / "muzero.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        batch = max(4, jax.device_count())
        config = get_muzero_config(args.case, enable_sharding=True,
            selfplay_batch_size=batch, train_batch_size=batch, selfplay_buffer_consume_size=batch,
            selfplay_buffer_max_len=1024 if args.case == "chess" else 32,
            selfplay_buffer_min_len=1, replay_buffer_max_len=4, buffer_layout="features",
            conv_width=8, conv_depth=1, muzero_unroll_steps=2,
            cycle_n_selfplay=512 if args.case == "chess" else 10, cycle_n_train=1,
            mcts_num_root_considered=2, mcts_num_survivors=1, num_exploratory_moves=0,
            replay_buffer_warmup_steps=0, lr_warmup_steps=0,
            diagnostic_period=0, eval_period=0, enable_trackio=False, enable_wandb=False)
        algorithm = module.make_muzero(config, jax.random.PRNGKey(0))
        state = algorithm.runner_state
        cycles = 1 if args.case == "chess" else 3
        for _ in range(cycles):
            state, (metrics, _) = algorithm.run_fn(state, jnp.bool_(False))
            assert int(metrics["selfplay_buffer/error_count"]) == 0
        assert int(state.model_ts.n_updates) == cycles
        ckpt.save_checkpoint(path, state=state, cycle=cycles, config=algorithm.config,
                             metadata={"trackio": {"id": "same-run"}})
        restored, _ = compressed.load(path, ckpt.muzero_checkpoint_payload(state, cycles))
        collectively(lambda: exact(state, restored["runner_state"]), "all state bytes")
        next_state, next_metrics = algorithm.run_fn(state, jnp.bool_(False))
        restored_next, restored_metrics = algorithm.run_fn(restored["runner_state"], jnp.bool_(False))
        collectively(lambda: exact(next_state, restored_next), "next state")
        collectively(lambda: exact(next_metrics, restored_metrics), "next metrics")
        fresh = module.make_muzero(config, jax.random.PRNGKey(99))
        resumed, start = ckpt.restore_checkpoint(path, state=fresh.runner_state, config=fresh.config)
        assert start == cycles
        resumed, metrics = fresh.run_fn(resumed, jnp.bool_(False))
        collectively(lambda: exact(next_state, resumed), "fresh algorithm resume")
        assert ckpt.checkpoint_metadata(path) == {"trackio": {"id": "same-run"}}
        partial_path = args.directory / "partial.ckpt"
        ckpt.save_checkpoint(partial_path, state=resumed, cycle=cycles + 1,
                             config=fresh.config, mode="partial")
        partial = ckpt.muzero_checkpoint_payload(resumed, cycles + 1, False)
        partial_back, _ = compressed.load(partial_path, partial)
        collectively(lambda: exact(partial, partial_back), "partial restore")
    print("CHECKPOINT_SMOKE " + json.dumps(dict(case=args.case, host=rank,
        hosts=jax.process_count(), devices=jax.device_count(), status="passed")), flush=True)
    jax.distributed.shutdown()


if __name__ == "__main__":
    main()
