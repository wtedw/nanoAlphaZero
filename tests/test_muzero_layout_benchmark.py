"""Report parsing and orchestration checks; no accelerator initialization."""

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def benchmark(monkeypatch):
    directory = Path(__file__).resolve().parents[1] / 'scripts'
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location('layout_benchmark', directory / 'benchmark_muzero_layouts.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summary_separates_warmup_compilation_and_runtime(benchmark, tmp_path):
    events = [
        {'stage': 'storage', 'fields': {'x': {'logical_bytes': 128}}, 'initialization_seconds': 2},
        {'stage': 'compiled', 'seconds': 20, 'compiled_memory': {'temp_size_in_bytes': 4096}},
    ]
    for i, duration in enumerate((10, 2, 4)):
        events.append({'stage': 'executed', 'device_call_seconds': duration, 'measured': i > 0,
                       'metrics': {'runner_state/n_updates': (i+1)*3, 'drain/n_slices': 2},
                       'memory': [{'bytes_in_use': 100+i, 'bytes_reserved': 200}]})
    events.append({'stage': 'passed'})
    (tmp_path / 'stdout.jsonl').write_text('not JSON\n' + '\n'.join(map(json.dumps, events)))
    (tmp_path / 'stderr.log').write_text('')
    summary = benchmark.summarize_job(tmp_path, 0)
    assert summary['status'] == 'passed'
    assert summary['first_call_seconds'] == 10
    assert summary['steady_median_seconds'] == 3
    assert summary['steady_p95_seconds'] == 4
    assert summary['compile_seconds'] == 20
    assert summary['replay_logical_bytes'] == 128
    assert summary['measured_optimizer_updates'] == 6
    assert summary['measured_drain_batches'] == 4
    assert summary['bytes_in_use_last_minus_first_max_device'] == 1


def test_failed_compile_does_not_report_success_or_zero_storage(benchmark, tmp_path):
    (tmp_path / 'stdout.jsonl').write_text('{"stage": "compiling"}\n')
    (tmp_path / 'stderr.log').write_text('RESOURCE_EXHAUSTED: exceeded HBM capacity')
    summary = benchmark.summarize_job(tmp_path, -6)
    assert summary['status'] == 'oom'
    assert summary['steady_median_seconds'] is None
    assert summary['replay_logical_bytes'] is None


def test_dry_run_launches_nothing_and_rotates_layouts(benchmark, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(benchmark.subprocess, 'run', lambda *a, **kw: pytest.fail('launched child'))
    monkeypatch.setattr(benchmark, 'check_tpu_ownership', lambda: pytest.fail('probed TPU'))
    output = tmp_path / 'unused'
    benchmark.main(['--dry-run', '--platform', 'tpu', '--output', str(output),
                    '--capacities', '128', '--repeats', '2', '--batch', '4096'])
    commands = capsys.readouterr().out.splitlines()
    assert len(commands) == 6
    assert '--layout original' in commands[0]
    assert '--layout features' in commands[3]
    assert all('--batch 4096' in command for command in commands)
    assert not output.exists()


def test_comparison_reports_speed_and_metric_mismatch(benchmark):
    base = {'name': 'original', 'layout': 'original', 'capacity': 4, 'mode': 'cycle',
            'operation': 'roundtrip', 'repeat': 0, 'status': 'passed', 'steady_median_seconds': 4,
            'replay_logical_bytes': 100, 'last_metrics': {'total_loss': 1.0}}
    candidate = {**base, 'name': 'matrix', 'layout': 'matrix', 'steady_median_seconds': 2,
                 'last_metrics': {'total_loss': 2.0}}
    comparison = benchmark.compare_results([base, candidate])[1]
    assert comparison['speedup_vs_original'] == 2
    assert comparison['same_logical_bytes']
    assert not comparison['last_metric_matches']['total_loss']
