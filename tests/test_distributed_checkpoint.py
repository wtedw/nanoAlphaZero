"""Two real CPU processes, shared checkpoint storage, no TPU initialization."""

import os
from pathlib import Path
import socket
import subprocess
import sys

import pytest


@pytest.mark.skipif(sys.platform != "linux", reason="Linux CPU affinity")
@pytest.mark.parametrize("case", ["storage", "ttt", "chess"])
def test_distributed_checkpoint(tmp_path, case):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    env = {k: v for k, v in os.environ.items() if "proxy" not in k.lower()}
    env.update(JAX_PLATFORMS="cpu", XLA_FLAGS="--xla_force_host_platform_device_count=2",
               JAX_CPU_COLLECTIVES_IMPLEMENTATION="gloo")
    script = Path(__file__).parent / "helpers" / "checkpoint_smoke.py"
    children, logs = [], []
    try:
        for rank in range(2):
            log = (tmp_path / f"worker-{rank}.log").open("w")
            logs.append(log)
            children.append(subprocess.Popen([sys.executable, str(script), "--rank", str(rank),
                "--port", str(port), "--directory", str(tmp_path), "--case", case],
                env=env, stdout=log, stderr=log))
        for rank, child in enumerate(children):
            assert child.wait(timeout=240) == 0, (tmp_path / f"worker-{rank}.log").read_text()
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait()
        for log in logs:
            log.close()
