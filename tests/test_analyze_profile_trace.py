import gzip
import json
import sys
import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location(
    "analyze_profile_trace",
    Path(__file__).resolve().parents[1] / "scripts" / "analyze_profile_trace.py",
)
analyze_profile_trace = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyze_profile_trace)


@pytest.mark.parametrize("mode", ["mcts", "selfplay", "cycle"])
def test_streams_and_summarizes_trace_across_small_chunks(tmp_path, monkeypatch, capsys, mode):
    trace_path = tmp_path / "run.trace.json.gz"
    events = [
        {"ph": "M", "pid": 1, "name": "process_name", "args": {"name": "/device:TPU:0"}},
        {"ph": "M", "pid": 1, "tid": 2, "name": "thread_name", "args": {"name": "XLA Ops"}},
        {"ph": "X", "pid": 1, "tid": 2, "name": "fusion.1", "ts": 10, "dur": 1_000},
        {"ph": "X", "pid": 1, "tid": 2, "name": "while.1", "ts": 1_010, "dur": 2_000},
    ]
    with gzip.open(trace_path, "wt", encoding="utf-8") as trace:
        json.dump({"displayTimeUnit": "ns", "traceEvents": events}, trace)

    streamed = list(analyze_profile_trace.iter_trace_events(str(trace_path), chunk_size=7))
    assert streamed == events

    monkeypatch.setattr(
        sys,
        "argv",
        ["analyze_profile_trace.py", str(trace_path), "--mode", mode, "--top", "2",
         "--json", str(tmp_path / "analysis.json")],
    )
    analyze_profile_trace.main()
    output = capsys.readouterr().out
    assert f"{mode} profile:" in output
    assert "device span: 3.000 ms/call" in output
    assert "control flow" in output
    assert "fusion.1" in output
    report = json.loads((tmp_path / "analysis.json").read_text())
    assert report["devices"]["/device:TPU:0"]["covered_ms"] == 3
    assert report["devices"]["/device:TPU:0"]["coverage_percent"] == 100


def test_coverage_does_not_double_count_nested_or_overlapping_events():
    assert analyze_profile_trace.covered_time([(0, 10), (2, 4), (8, 15), (20, 25)]) == 20
