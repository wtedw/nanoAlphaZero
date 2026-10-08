"""Run isolated MuZero replay-layout comparisons, retaining logs and JSON reports.

Run from the repository root through uv. No JAX import in this orchestrator.
All variants use original self-play storage; only replay storage is transformed.
Unknown arguments are forwarded to probe_muzero_replay_layout.py (see --help).

Quick CPU control:
  uv run --extra trackio python scripts/benchmark_muzero_layouts.py \
    --platform cpu --capacities 4 --operations add sample roundtrip \
    --batch 8 --iterations 3 --warmup-iterations 1

After the TPU is free, chess replay add/sample/decoding measurements:
  uv run --extra trackio python scripts/benchmark_muzero_layouts.py \
    --platform tpu --capacities 128 512 --repeats 3 \
    --operations add sample roundtrip --batch 4096 --iterations 30

Full chess cycles, synthetic replay seeded to guarantee backward passes:
  uv run --extra trackio python scripts/benchmark_muzero_layouts.py \
    --platform tpu --modes cycle --capacities 128 512 --repeats 3 \
    --batch 4096 --depth 10 --width 256 --selfplay-capacity 512 \
    --iterations 10 --warmup-iterations 2

Real-game control (no seeded replay, enough steps to cross a 512-position ring):
  uv run --extra trackio python scripts/benchmark_muzero_layouts.py \
    --platform tpu --modes cycle --capacities 128 --no-seed-replay \
    --batch 4096 --depth 10 --width 256 --selfplay-capacity 512 \
    --selfplay-steps 20 --train-steps 20 --iterations 30 --warmup-iterations 2

Append --dry-run to print commands without importing JAX or starting children.
--dump-hlo retains XLA dumps per job; these can be very large. Raw stderr retains
driver padding/OOM reports, which are not interchangeable with memory_analysis.
No checkpointing, tracker writes, TPU lock removal, or background waiting.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import statistics
import subprocess
import sys
import time
import uuid

from probe_muzero_replay_layout import check_tpu_ownership


def summarize_job(directory, returncode):
    events = []
    for line in (directory / 'stdout.jsonl').read_text().splitlines():
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict) and 'stage' in item:
            events.append(item)
    by_stage = {item['stage']: item for item in events}
    executions = [item for item in events if item['stage'] == 'executed']
    measured = [item for item in executions if item.get('measured')]
    durations = sorted(item['device_call_seconds'] for item in measured)
    stderr = (directory / 'stderr.log').read_text(errors='replace')
    status = 'passed' if returncode == 0 and 'passed' in by_stage else 'failed'
    if status == 'failed' and any(s in stderr.lower() for s in ('out of memory', 'resource_exhausted', 'hbm capacity')):
        status = 'oom'
    memory_rows = [device for item in events for device in item.get('memory', [])]
    memory_keys = {key for row in memory_rows for key, value in row.items() if isinstance(value, (int, float))}
    storage = by_stage.get('storage', {})
    result = {
        'status': status, 'returncode': returncode, 'last_stage': events[-1]['stage'] if events else None,
        'compile_seconds': by_stage.get('compiled', {}).get('seconds'),
        'initialization_seconds': storage.get('initialization_seconds'),
        'compiled_memory_bytes': by_stage.get('compiled', {}).get('compiled_memory'),
        'replay_logical_bytes': sum(field['logical_bytes'] for field in storage['fields'].values()) if storage else None,
        'first_call_seconds': executions[0]['device_call_seconds'] if executions else None,
        'measured_calls': len(durations),
        'steady_median_seconds': statistics.median(durations) if durations else None,
        'steady_p95_seconds': durations[max(0, (95 * len(durations) + 99) // 100 - 1)] if durations else None,
        'allocator_max_across_devices_and_phases': {key: max(row.get(key, 0) for row in memory_rows) for key in sorted(memory_keys)},
        'last_metrics': executions[-1]['metrics'] if executions else {},
        'measured_memory': [item['memory'] for item in measured],
        'measured_seconds': [item['device_call_seconds'] for item in measured],
    }
    if measured:
        before = executions[len(executions) - len(measured) - 1]['metrics'] if len(executions) > len(measured) else {}
        result['measured_optimizer_updates'] = result['last_metrics'].get('runner_state/n_updates', 0) - before.get('runner_state/n_updates', 0)
        result['measured_drain_batches'] = sum(item['metrics'].get('drain/n_slices', 0) for item in measured)
        elapsed = sum(durations)
        config = by_stage.get('start', {}).get('config', {})
        if 'cycle_n_selfplay' in config and by_stage['start'].get('mode') == 'cycle':
            result['selfplay_positions_per_second'] = len(measured) * config['cycle_n_selfplay'] * config['selfplay_batch_size'] / elapsed
            result['optimizer_updates_per_second'] = result['measured_optimizer_updates'] / elapsed
        for counter in ('bytes_in_use', 'bytes_reserved'):
            first = [row[counter] for row in measured[0]['memory'] if counter in row]
            last = [row[counter] for row in measured[-1]['memory'] if counter in row]
            if first and last:
                result[f'{counter}_last_minus_first_max_device'] = max(last) - max(first)
    return result


def compare_results(results):
    """Within each repeat/configuration, report comparisons against original."""
    comparisons = []
    for result in results:
        reference = next((row for row in results if row['layout'] == 'original' and
                          all(row[key] == result[key] for key in ('capacity', 'mode', 'operation', 'repeat'))), None)
        comparable = reference is not None and reference['status'] == result['status'] == 'passed'
        row = {'name': result['name'], 'status': result['status'],
               'reference_status': reference['status'] if reference else 'not_run',
               'speedup_vs_original': (reference['steady_median_seconds'] / result['steady_median_seconds']) if comparable else None,
               'same_logical_bytes': (reference['replay_logical_bytes'] == result['replay_logical_bytes']) if comparable else None}
        if comparable:
            keys = ('total_loss', 'checksum', 'runner_state/n_updates', 'selfplay_buffer/error_count',
                    'selfplay_buffer/dropped_pending_count', 'selfplay_buffer/dropped_completed_count')
            row['last_metric_matches'] = {key: math.isclose(reference['last_metrics'][key], result['last_metrics'][key],
                                                          rel_tol=1e-5, abs_tol=1e-6)
                                          for key in keys if key in reference['last_metrics'] and key in result['last_metrics']}
        comparisons.append(row)
    return comparisons


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--platform', choices=('cpu', 'tpu'), default='cpu')
    parser.add_argument('--layouts', nargs='+', choices=('original', 'features', 'matrix'), default=['original', 'features', 'matrix'])
    parser.add_argument('--capacities', nargs='+', type=int, default=[16])
    parser.add_argument('--modes', nargs='+', choices=('replay', 'cycle'), default=['replay'])
    parser.add_argument('--operations', nargs='+', choices=('add', 'sample', 'roundtrip'), default=['roundtrip'])
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--dump-hlo', action='store_true')
    args, forwarded = parser.parse_known_args(argv)
    if min([args.repeats, *args.capacities]) < 1:
        parser.error('repeats and capacities must be positive')
    if any(token.split('=')[0] in ('--layout', '--capacity', '--mode', '--operation') for token in forwarded):
        parser.error('use the plural sweep arguments for layout/capacity/mode/operation')
    root = Path(__file__).resolve().parents[1]
    probe = root / 'scripts/probe_muzero_replay_layout.py'
    jobs = []
    for repeat in range(args.repeats):
        layouts = args.layouts[repeat % len(args.layouts):] + args.layouts[:repeat % len(args.layouts)]
        for capacity in args.capacities:
            for mode in args.modes:
                for operation in args.operations if mode == 'replay' else ['roundtrip']:
                    for layout in layouts:
                        name = f'{mode}-{operation}-{layout}-t{capacity}-r{repeat}'
                        command = [sys.executable, str(probe), '--platform', args.platform, '--layout', layout,
                                   '--capacity', str(capacity), '--mode', mode, '--operation', operation, *forwarded]
                        jobs.append({'name': name, 'command': command, 'layout': layout, 'capacity': capacity,
                                     'mode': mode, 'operation': operation, 'repeat': repeat})
    if args.dry_run:
        for job in jobs:
            print(shlex.join(job['command']))
        return
    if args.platform == 'tpu':
        check_tpu_ownership()
    output = args.output or root / 'artifacts' / ('muzero-layout-' + datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6])
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(probe, output / probe.name)
    shutil.copy2(__file__, output / Path(__file__).name)
    manifest = {'jobs': jobs, 'platform': args.platform,
                'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                'tracked_diff_sha256': hashlib.sha256(subprocess.check_output(['git', 'diff', 'HEAD'], cwd=root)).hexdigest(),
                'compilation_cache': 'disabled', 'selfplay_layout': 'original',
                'limitations': ['CPU runs do not measure TPU padding or throughput.',
                    'Logical bytes, compiler estimates and allocator counters are different measurements.',
                    'One executable is compiled per process and reused; initialization also compiles helpers.',
                    'Synthetic seeded replay measures execution, not learning; inspect actual drain counts.',
                    'Steady timing excludes compilation and configured warmup calls, but includes dispatch and synchronization.']}
    environment = os.environ.copy()
    environment['JAX_PLATFORMS'] = args.platform
    environment['JAX_ENABLE_COMPILATION_CACHE'] = 'false'
    if args.platform == 'cpu' and '--xla_force_host_platform_device_count' not in environment.get('XLA_FLAGS', ''):
        environment['XLA_FLAGS'] = environment.get('XLA_FLAGS', '') + ' --xla_force_host_platform_device_count=4'
    manifest['xla_flags'] = environment.get('XLA_FLAGS', '')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    results = []
    for job in jobs:
        if args.platform == 'tpu':
            check_tpu_ownership()
        directory = output / job['name']
        directory.mkdir()
        job_env = environment.copy()
        if args.dump_hlo:
            job_env['XLA_FLAGS'] = job_env.get('XLA_FLAGS', '') + ' --xla_dump_to=' + str(directory / 'hlo')
        print(f"Running {job['name']}; logs: {directory}", flush=True)
        started = time.monotonic()
        with (directory / 'stdout.jsonl').open('w') as stdout, (directory / 'stderr.log').open('w') as stderr:
            process = subprocess.run(job['command'], cwd=root, env=job_env, stdout=stdout, stderr=stderr)
        result = {**job, **summarize_job(directory, process.returncode), 'wall_seconds': time.monotonic() - started}
        results.append(result)
        (output / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
        (output / 'comparisons.json').write_text(json.dumps(compare_results(results), indent=2) + '\n')
        print(f"{job['name']}: {result['status']}, median={result['steady_median_seconds']} s", flush=True)
        # OOM candidates may fail; continue comparisons. Never contend with a new owner.
    print(f'Results: {output / "summary.json"}', flush=True)
    return 0 if all(result['status'] == 'passed' for result in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
