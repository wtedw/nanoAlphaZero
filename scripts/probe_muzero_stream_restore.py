"""Restore compressed buffer chunks to devices without a full host array.

Consumes the previous compression probe's manifests. --repeat 16 repeats the
256-lane sample to stress the real 4096-lane allocation, NOT compression quality.
No uncompressed disk files are created. Production checkpoint code is unchanged.
"""

import argparse
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import time

from probe_muzero_checkpoint_compression import Zstd, shuffle


def rss_bytes():
    return int(Path('/proc/self/statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')


def restore(directory, devices, repeat=1, verify=True):
    import jax
    import jax.numpy as jnp
    import numpy as np
    import ml_dtypes
    del ml_dtypes
    manifest = json.loads((directory / 'manifest.json').read_text())
    mesh = jax.sharding.Mesh(np.array(devices), ('data',))
    codec = Zstd()
    arrays, reports = {}, []
    started = time.perf_counter()
    baseline_rss = rss_bytes()
    peak_sampled_rss = baseline_rss
    for field in manifest['fields']:
        shape = tuple(field['shape'])
        dtype = np.dtype(field['dtype'])
        sharded = bool(shape)
        copies = repeat if sharded else 1
        final_shape = (shape[0] * copies, *shape[1:]) if sharded else ()
        if sharded and final_shape[0] % len(devices):
            raise ValueError('First axis must divide evenly across devices')
        count = math.prod(final_shape)
        shard_count = count // len(devices) if sharded else count
        shard_devices = devices if sharded else devices[:1]
        chunk_count = max(1, min(manifest['chunk_bytes'] // dtype.itemsize, shard_count))

        @partial(jax.jit, donate_argnums=(0,))
        def put_chunk(target, values, offset):
            return jax.lax.dynamic_update_slice(target, values, (offset,))

        @jax.jit
        def get_chunk(target, offset):
            return jax.lax.dynamic_slice(target, (offset,), (chunk_count,))

        # Allocate on device by broadcasting a scalar, never a shard-sized host zero array.
        shards = [jax.jit(lambda x: jnp.full((shard_count,), x, dtype=dtype))(
            jax.device_put(np.array(0, dtype=dtype), device)) for device in shard_devices]
        expected = [hashlib.sha256() for _ in shard_devices]
        # A chunk can straddle a device boundary. Carry at most one chunk's bytes.
        device_index, written = 0, 0
        pending = bytearray()
        for copy in range(copies):
            source_hash = hashlib.sha256()
            for chunk in field['chunks']:
                raw = shuffle(codec.decode((directory / chunk['file']).read_bytes(),
                                           chunk['raw_bytes']), dtype.itemsize, inverse=True)
                source_hash.update(raw)
                pending.extend(raw)
                while pending:
                    take = min(chunk_count, shard_count - written)
                    nbytes = take * dtype.itemsize
                    if len(pending) < nbytes:
                        break
                    payload = bytes(pending[:nbytes])
                    del pending[:nbytes]
                    expected[device_index].update(payload)
                    values = np.frombuffer(payload, dtype=dtype)
                    # Tail uses its real length: avoid dynamic_update_slice clamping
                    # a padded write backward over data at the end of the shard.
                    shards[device_index] = put_chunk(
                        shards[device_index], jax.device_put(values, shard_devices[device_index]),
                        jax.device_put(np.int32(written), shard_devices[device_index]))
                    shards[device_index].block_until_ready()  # Bound transfer queue / host ownership.
                    written += take
                    if written == shard_count:
                        device_index += 1
                        written = 0
                        if device_index == len(shard_devices):
                            break
                peak_sampled_rss = max(peak_sampled_rss, rss_bytes())
            if source_hash.hexdigest() != field['sha256']:
                raise ValueError(f"Corrupt field: {field['path']}")
        if pending or (count and device_index != len(shard_devices)):
            raise ValueError('Incomplete restore')
        if verify:
            for index, shard in enumerate(shards):
                actual = hashlib.sha256()
                for offset in range(0, shard_count, chunk_count):
                    take = min(chunk_count, shard_count - offset)
                    # Fixed-size slices clamp at the end; select the tail explicitly.
                    start = min(offset, shard_count - chunk_count)
                    host = np.asarray(jax.device_get(get_chunk(shard, np.int32(start))))
                    actual.update(host[-take:].tobytes() if take < chunk_count else host.tobytes())
                if actual.digest() != expected[index].digest():
                    raise ValueError(f"Device verification failed: {field['path']} shard {index}")
        if sharded:
            local_shape = (final_shape[0] // len(devices), *final_shape[1:])
            local_arrays = [shard.reshape(local_shape) for shard in shards]
            sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('data'))
            array = jax.make_array_from_single_device_arrays(final_shape, sharding, local_arrays)
        else:
            sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
            array = jax.device_put(shards[0].reshape(()), sharding)
        array.block_until_ready()
        arrays[field['path']] = array
        reports.append(dict(path=field['path'], shape=final_shape, bytes=count * dtype.itemsize,
                            shards=len(array.addressable_shards)))
        print(json.dumps({'restored': field['path'], 'bytes': count * dtype.itemsize,
                          'rss_bytes': rss_bytes()}), flush=True)
    return arrays, dict(seconds_including_device_verification=time.perf_counter() - started,
                        raw_bytes=sum(f['bytes'] for f in reports), fields=reports,
                        baseline_rss_bytes=baseline_rss, sampled_peak_rss_bytes=peak_sampled_rss,
                        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                        verified=verify, repeat=repeat)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--platform', choices=('cpu', 'tpu'), default='cpu')
    parser.add_argument('--repeat', type=int, default=1)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Use a new output report path')
    if args.platform == 'tpu':
        owners = subprocess.run(['fuser', '/tmp/libtpu_lockfile'], capture_output=True, text=True)
        if owners.returncode == 0:
            raise SystemExit(f'TPU occupied: {owners.stdout}')
    os.environ['JAX_PLATFORMS'] = args.platform
    if args.platform == 'cpu':
        os.environ.setdefault('XLA_FLAGS', '--xla_force_host_platform_device_count=4')
    import jax
    devices = jax.devices()
    retained, reports = {}, {}
    for name in ('selfplay', 'replay'):
        retained[name], reports[name] = restore(args.input / name, devices, args.repeat)
    reports['devices'] = [str(d) for d in devices]
    reports['device_memory'] = [d.memory_stats() for d in devices]
    args.output.write_text(json.dumps(reports, indent=2))
    print(json.dumps({k: {key: value for key, value in v.items() if key != 'fields'}
                      for k, v in reports.items() if k in ('selfplay', 'replay')}), flush=True)


if __name__ == '__main__':
    main()
