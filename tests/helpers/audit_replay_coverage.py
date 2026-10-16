"""Read-only historical metrics and synthetic production replay sampling audit.

Run with JAX_PLATFORMS=cpu. Does not load or modify training state.
"""
import json
from pathlib import Path
import sqlite3

import jax
import jax.numpy as jnp
import numpy as np

from nanoalphazero.buffers import make_muzero_replay_buffer


def sampler_probe():
    cfg = dict(replay_buffer_add_batch_size=8192,
               replay_buffer_sample_batch_size=8192,
               replay_buffer_min_len=1, replay_buffer_max_len=1024,
               buffer_layout="features", enable_sharding=False)
    buffer, state = make_muzero_replay_buffer(cfg, jnp.int32(0))
    ids = jnp.arange(8192 * 1024, dtype=jnp.int32).reshape(8192, 1024, 1)
    keys = jax.random.split(jax.random.PRNGKey(20261017), 128)
    sample = jax.jit(jax.vmap(buffer.sample, in_axes=(None, 0)))
    reports = []
    for index, full in [(0, True), (37, True), (512, False)]:
        current = state.replace(experience=ids, current_index=jnp.int32(index),
                                is_full=jnp.bool_(full))
        picked = np.asarray(sample(current, keys).experience).reshape(128, 8192)
        rows, cols = picked // 1024, picked % 1024
        time_size = 1024 if full else index
        offset = (cols - rows % time_size) % time_size
        # Independent draws are a diagnostic control, not a production change.
        row_key, time_key = jax.random.split(jax.random.PRNGKey(42))
        independent = np.asarray(jax.random.randint(row_key, picked.shape, 0, 8192)) * 1024
        eligible_times = time_size - 2 if full else time_size
        control_times = np.asarray(jax.random.randint(time_key, picked.shape, 0, eligible_times))
        control_times = (control_times + index + 2) % time_size
        independent += control_times
        expected_offsets = np.arange(index % 2, 1024, 2) if full else np.array([2])
        np.testing.assert_array_equal(np.unique(offset), expected_offsets)
        if not full:
            assert np.unique(picked).size == 8192
        reports.append(dict(current_index=index, full=full,
            draws=picked.size, unique_items=int(np.unique(picked).size),
            unique_rows=int(np.unique(rows).size), unique_columns=int(np.unique(cols).size),
            row_time_offset_count=int(np.unique(offset).size),
            row_time_offset_first=int(np.unique(offset)[0]),
            first_batch_unique=int(np.unique(picked[0]).size),
            twenty_batches_unique=int(np.unique(picked[:20]).size),
            independent_twenty_batches_unique=int(np.unique(independent[:20]).size),
            first_two_batch_overlap=int(np.intersect1d(picked[0], picked[1]).size),
            independent_unique_items=int(np.unique(independent).size)))
    # Verify the same production path with a globally sharded synthetic buffer.
    if jax.device_count() > 1:
        mesh = jax.sharding.Mesh(np.array(jax.devices()), ("data",))
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))
        sharded, sharded_state = make_muzero_replay_buffer(
            dict(cfg, enable_sharding=True), jnp.int32(0), sharding)
        sharded_state = sharded_state.replace(experience=jax.device_put(ids, sharding),
            current_index=jnp.int32(37), is_full=jnp.bool_(True))
        ordinary = state.replace(experience=ids, current_index=jnp.int32(37), is_full=jnp.bool_(True))
        np.testing.assert_array_equal(np.asarray(sharded.sample(sharded_state, keys[0]).experience),
                                      np.asarray(buffer.sample(ordinary, keys[0]).experience))
    return dict(cases=reports, devices=jax.device_count(), sharded_match=jax.device_count() > 1)


def historical_metrics():
    connection = sqlite3.connect("file:artifacts/trackio/nanoAlphaZero-muzero.db?mode=ro", uri=True)
    records = []
    for step, raw in connection.execute(
        "SELECT step, metrics FROM metrics WHERE run_name=? ORDER BY step", ("fancy-breeze-4",)
    ):
        m = json.loads(raw)
        if "train_batch/n_value_valid" not in m:
            continue
        count = m["train_batch/n_value_valid"]
        decisive = m["train_batch/n_value_pos"] + m["train_batch/n_value_neg"]
        zero_mse = decisive / count if count else None
        out = dict(cycle=step, value_mse=m["loss_v"], zero_mse=zero_mse,
                   relative_mse=m["loss_v"] / zero_mse if zero_mse else None,
                   supervised_fraction=count / (6 * m["train_batch/n_is_valid"]),
                   invalid=m["train_batch/n_is_invalid"])
        for key in ("ep_step", "sample_age", "sampled_row_ids", "sampled_game_ids"):
            h = m.get(f"train_batch/{key}_distribution")
            if h:
                edges, counts = np.array(h["bins"]), np.array(h["values"])
                centers = (edges[:-1] + edges[1:]) / 2
                out[key] = dict(n=int(counts.sum()), occupied_bins=int((counts > 0).sum()),
                    mean_approx=float(np.average(centers, weights=counts)),
                    minimum=float(edges[0]), maximum=float(edges[-1]))
                if key == "ep_step":
                    out[key]["fraction_below_30_lower"] = float(counts[edges[1:] <= 30].sum()/counts.sum())
                    out[key]["fraction_below_30_upper"] = float(counts[edges[:-1] < 30].sum()/counts.sum())
        records.append(out)
    return records


def frozen_errors():
    path = Path("artifacts/value-fit-tpu-20261017")
    fit = json.loads((path / "fit-results.json").read_text())
    targets = np.load(path / "dataset.npz")["targets"]
    result = {}
    for name, section in [("train", slice(0, 128)), ("heldout", slice(128, 256))]:
        metrics = fit["variants"]["value_only"][0][name]
        result[name] = dict(**metrics, zero_mse=np.mean(targets[section] ** 2, axis=0).tolist())
    return result


if __name__ == "__main__":
    output = Path("artifacts/replay-coverage-audit-20261017")
    output.mkdir(exist_ok=True)
    report = dict(sampler=sampler_probe(), historical=historical_metrics(), frozen=frozen_errors())
    (output / "results.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({**report, "historical": [r for r in report["historical"]
          if r["cycle"] in (1, 100, 400, 800, 1200, 1600, 2400, 3200, 3745)]}, indent=2))
