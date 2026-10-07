# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "jax==0.8.1",
#   "jaxlib==0.8.1",
#   "libtpu==0.0.30; sys_platform == 'linux'",
#   "numpy==2.3.5",
# ]
# ///
"""Reproduce a TPU XLA all-reduce-scatter shape-mismatch compiler failure.

Before running, ensure no other process owns the TPU. Do not remove live locks:
    fuser -v /tmp/libtpu_lockfile /dev/accel{0,1,2,3}

Run on a machine with four local TPU devices (no project installation needed):
    JAX_PLATFORMS=tpu uv run --no-project scripts/repro_tpu_gather_slice.py

Optional compiler dumps (use a new directory):
    XLA_FLAGS='--xla_dump_to=/tmp/gather-slice-hlo --xla_dump_hlo_as_text' \
      JAX_PLATFORMS=tpu uv run --no-project scripts/repro_tpu_gather_slice.py

CPU control, which is expected to pass both variants:
    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \
      uv run --no-project scripts/repro_tpu_gather_slice.py

Observed on four TPU v4 devices with the pinned versions above:
  x[rows[:, None], cols][:, :5] fails during compilation. XLA tries to replace
  an f32[256,5] dynamic-slice with an f32[256,6] all-reduce-scatter fusion.
  x[rows[:, None], cols[:, :5]] compiles and matches NumPy exactly.

Both expressions should return identical float32 arrays of shape (1024, 5).
The source array is only 323,584 bytes. No model or training code is involved.
This isolates a compiler rewrite failure, not an observed out-of-memory error.

Exit status: 1 = bug reproduced; 0 = both variants pass; 2 = another failure.
JSON output includes versions, device descriptions, error text and checks.
"""

import importlib.metadata
import json
import os
import platform

# Backend selection must happen before importing JAX. CPU control can override it.
os.environ.setdefault("JAX_PLATFORMS", "tpu")

import jax
import numpy as np


def main():
    versions = {}
    for package in ("jax", "jaxlib", "libtpu", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    print(json.dumps({"python": platform.python_version(), "versions": versions,
                      "JAX_PLATFORMS": os.environ["JAX_PLATFORMS"],
                      "XLA_FLAGS": os.environ.get("XLA_FLAGS", "")}), flush=True)
    devices = jax.local_devices()
    print(json.dumps({"devices": [{"id": d.id, "platform": d.platform,
                                   "kind": d.device_kind} for d in devices]}), flush=True)
    if len(devices) != 4:
        print("This reproducer requires exactly four local devices.", flush=True)
        return 2

    mesh = jax.sharding.Mesh(np.array(devices), ("data",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))
    source = np.arange(1024 * 79, dtype=np.float32).reshape(1024, 79)
    # Reverse the rows so every device gathers data from a different device.
    rows = np.arange(1024, dtype=np.int32)[::-1].copy()
    cols = (np.arange(1024, dtype=np.int32)[:, None] + np.arange(6, dtype=np.int32)) % 79
    expected = source[rows[:, None], cols[:, :5]]
    inputs = tuple(jax.device_put(x, sharding) for x in (source, rows, cols))
    results = {}
    for name, fn in (
        ("gather_then_slice", lambda x, r, c: x[r[:, None], c][:, :5]),
        ("gather_exact_width", lambda x, r, c: x[r[:, None], c[:, :5]]),
    ):
        phase = "compile"
        try:
            compiled = jax.jit(fn, in_shardings=(sharding,) * 3,
                               out_shardings=sharding).lower(*inputs).compile()
            phase = "execute_and_compare"
            actual = np.asarray(compiled(*inputs))
            np.testing.assert_array_equal(actual, expected)
            results[name] = {"passed": True, "shape": list(actual.shape),
                             "matches_numpy": True}
        except Exception as exc:
            results[name] = {"passed": False, "phase": phase,
                             "exception": type(exc).__name__, "error": str(exc)}
        print(json.dumps({name: results[name]}), flush=True)

    failing = results["gather_then_slice"]
    working = results["gather_exact_width"]
    if (failing.get("phase") == "compile" and working["passed"]
            and "all-reduce-scatter" in failing.get("error", "")
            and "f32[256,5]" in failing["error"] and "f32[256,6]" in failing["error"]):
        print("BUG REPRODUCED: gather-then-slice fails; equivalent exact-width gather passes.")
        return 1
    if failing["passed"] and working["passed"]:
        print("NOT REPRODUCED: both variants compile and match NumPy.")
        return 0
    print("UNEXPECTED FAILURE: inspect the errors above.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
