"""Bounded collective smoke; use pod run --cpu for CPU, or --platform tpu."""

import argparse
import json
import os

from nanoalphazero.distributed import DistributedArgs, initialize_distributed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=19451)
    parser.add_argument("--platform", choices=("cpu", "tpu"), default="cpu")
    args = parser.parse_args()
    if args.platform == "cpu" and os.environ.get("JAX_PLATFORMS") != "cpu":
        raise RuntimeError("This smoke test requires pod run --cpu")
    rank = int(os.environ["NANOAZ_POD_PROCESS_ID"])
    hosts = int(os.environ["NANOAZ_POD_PROCESS_COUNT"])
    cpus = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, cpus[:4])
    os.environ["JAX_PLATFORMS"] = args.platform
    if args.platform == "cpu":
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"
        os.environ["JAX_CPU_COLLECTIVES_IMPLEMENTATION"] = "gloo"
        initialize_distributed(DistributedArgs(
            True, os.environ["NANOAZ_POD_COORDINATOR"] + ":" + str(args.port), hosts, rank, 120,
        ))
    else:
        initialize_distributed(DistributedArgs(enabled=True))
    import jax
    import jax.numpy as jnp
    import numpy as np

    mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ("data",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))
    width = 2 * jax.local_device_count()
    local = np.arange(rank * width, (rank + 1) * width, dtype=np.float32)
    global_array = jax.make_array_from_process_local_data(sharding, local, global_shape=(hosts * width,))
    total = float(jax.jit(jnp.sum)(global_array))
    assert total == sum(range(hosts * width))
    assert jax.process_count() == hosts
    print(json.dumps({"process": jax.process_index(), "processes": hosts,
                      "devices": jax.device_count(), "local_devices": jax.local_device_count(),
                      "lanes": local.tolist(), "global_sum": total}), flush=True)
    jax.distributed.shutdown()


if __name__ == "__main__":
    main()
