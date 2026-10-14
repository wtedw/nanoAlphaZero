"""Two CPU processes exercise actual JAX discovery and a global reduction."""

import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import pytest


@pytest.mark.skipif(sys.platform != "linux", reason="Linux CPU affinity and distributed CPU runtime")
def test_two_process_global_array(tmp_path):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    script = """
import json, os, sys
from pathlib import Path
rank, port, output = int(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
cpus = sorted(os.sched_getaffinity(0))
os.sched_setaffinity(0, cpus[rank * 2:rank * 2 + 2] or cpus[:2])
from nanoalphazero.distributed import DistributedArgs, initialize_distributed
initialize_distributed(DistributedArgs(True, '127.0.0.1:' + port, 2, rank, 30))
import jax
import jax.numpy as jnp
import numpy as np
mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ('data',))
sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('data'))
local = np.arange(rank * 2, rank * 2 + 2, dtype=np.float32)
array = jax.make_array_from_process_local_data(sharding, local, global_shape=(4,))
total = jax.jit(lambda x: jnp.sum(x))(array)
from nanoalphazero.tracking_arrays import gather_training_arrays, device_memory_metrics
sample = gather_training_arrays({'valid': array % 2 == 0, 'row_id': array,
                                 'col_id': array + 10}, limit=2)
np.testing.assert_array_equal(sample['row_id'], [0, 2])
np.testing.assert_array_equal(sample['col_id'], [10, 12])
np.testing.assert_array_equal(sample['valid'], [True, True])
for limit, expected in [(1, [0]), (100, [0, 1, 2, 3])]:
    gathered = gather_training_arrays({'valid': array >= 0, 'row_id': array}, limit=limit)
    np.testing.assert_array_equal(gathered['row_id'], expected)
from types import SimpleNamespace
memory = device_memory_metrics([SimpleNamespace(process_index=rank,
    memory_stats=lambda: {'bytes_in_use': rank + 1, 'bytes_limit': 100})])
assert memory == {'memory/bytes_in_use_max': 2., 'memory/bytes_limit_max': 100.}
from nanoalphazero import tracking
def local_tracker(*args, **kwargs):
    assert rank == 0, 'Backend created on non-primary host'
    run = SimpleNamespace(log=lambda *a, **k: None, finish=lambda: None)
    return tracking.TrainingTracker(run, {'trackio': {'id': 'shared'}}, last_step=3)
tracking._make_local_tracker = local_tracker
tracker = tracking.make_tracker({'enable_trackio': True})
assert tracker.metadata == {'trackio': {'id': 'shared'}}
assert not tracker.should_log(3) and tracker.should_log(4)
assert (tracker.run is not None) == (rank == 0)
tracker.log({'scalar': 6.}, step=4)
tracker.finish()
def broken_tracker(*args, **kwargs):
    raise ValueError('test backend failure')
tracking._make_local_tracker = broken_tracker
try:
    tracking.make_tracker({'enable_trackio': True})
except RuntimeError as exc:
    assert 'test backend failure' in str(exc)
else:
    raise AssertionError('Backend error was not broadcast')
output.write_text(json.dumps({'rank': jax.process_index(), 'hosts': jax.process_count(),
                             'devices': jax.device_count(), 'local': local.tolist(), 'sum': float(total)}))
jax.distributed.shutdown()
"""
    env = {key: value for key, value in os.environ.items() if "proxy" not in key.lower()}
    env.update(JAX_PLATFORMS="cpu", XLA_FLAGS="--xla_force_host_platform_device_count=1",
               JAX_CPU_COLLECTIVES_IMPLEMENTATION="gloo")
    children = []
    logs = []
    try:
        for rank in range(2):
            log = (tmp_path / f"worker-{rank}.log").open("w")
            logs.append(log)
            children.append(subprocess.Popen(
                [sys.executable, "-c", script, str(rank), str(port), str(tmp_path / f"{rank}.json")],
                env=env, stdout=log, stderr=log,
            ))
        for rank, child in enumerate(children):
            assert child.wait(timeout=60) == 0, (tmp_path / f"worker-{rank}.log").read_text()
        reports = [json.loads((tmp_path / f"{rank}.json").read_text()) for rank in range(2)]
        assert [report["rank"] for report in reports] == [0, 1]
        assert [report["local"] for report in reports] == [[0, 1], [2, 3]]
        assert all(report["hosts"] == 2 and report["devices"] == 2 and report["sum"] == 6 for report in reports)
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait()
        for log in logs:
            log.close()
