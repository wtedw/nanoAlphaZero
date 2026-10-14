"""Summarize a JAX profiler trace without launching TensorBoard.

The Chrome trace JSON is parsed incrementally because MCTS traces can expand
to hundreds of megabytes even when the ``.json.gz`` file is relatively small.
"""

from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import os
import re
from collections.abc import Iterator
from typing import Any


TRACE_GLOB = "plugins/profile/*/*.trace.json.gz"


def find_trace(path: str) -> str:
    if os.path.isfile(path):
        return path
    paths = glob.glob(os.path.join(path, TRACE_GLOB))
    if not paths:
        paths = glob.glob(os.path.join(path, "**", "*.trace.json.gz"), recursive=True)
    if not paths:
        raise SystemExit(f"no *.trace.json.gz found under {path}")
    return max(paths, key=os.path.getmtime)


def iter_trace_events(path: str, chunk_size: int = 1 << 20) -> Iterator[dict[str, Any]]:
    """Yield objects from the top-level ``traceEvents`` array incrementally."""
    decoder = json.JSONDecoder()
    marker = re.compile(r'"traceEvents"\s*:\s*\[')
    with gzip.open(path, "rt", encoding="utf-8") as trace:
        buffer = ""
        while True:
            match = marker.search(buffer)
            if match is not None:
                buffer = buffer[match.end() :]
                break
            chunk = trace.read(chunk_size)
            if not chunk:
                raise ValueError(f"traceEvents array not found in {path}")
            buffer += chunk
            if marker.search(buffer) is None and len(buffer) > 128:
                buffer = buffer[-128:]
        position = 0
        eof = False
        while True:
            while True:
                while position < len(buffer) and buffer[position] in " \t\r\n,":
                    position += 1
                if position < len(buffer):
                    break
                if eof:
                    raise ValueError(f"unterminated traceEvents array in {path}")
                buffer = trace.read(chunk_size)
                position = 0
                eof = not buffer

            if buffer[position] == "]":
                return

            try:
                event, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                chunk = trace.read(chunk_size)
                if not chunk:
                    raise ValueError(f"invalid event JSON near byte buffer offset {position} in {path}")
                buffer = buffer[position:] + chunk
                position = 0
                continue

            if isinstance(event, dict):
                yield event
            position = end
            if position > chunk_size:
                buffer = buffer[position:]
                position = 0


def op_family(name: str) -> str:
    lowered = name.lower()
    if any(token in lowered for token in ("all-reduce", "all-gather", "reduce-scatter", "collective-permute", "send", "recv")):
        return "collectives"
    if any(token in lowered for token in ("convolution", "custom-call", "dot", "gemm", "matmul")):
        return "matrix / convolution"
    if lowered.startswith(("while", "conditional")) or "while." in lowered:
        return "control flow"
    if "fusion" in lowered:
        return "fusions"
    if "reduce" in lowered:
        return "reductions"
    if any(token in lowered for token in ("dynamic-slice", "dynamic-update-slice", "gather", "scatter")):
        return "indexing / updates"
    if any(token in lowered for token in ("copy", "transpose", "reshape", "bitcast", "pad")):
        return "copies / transforms"
    return "other XLA ops"


def covered_time(intervals):
    """Union duration, avoiding double-counting nested/overlapping XLA events."""
    total, end = 0.0, float("-inf")
    for start, stop in sorted(intervals):
        total += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", nargs="?", default="/tmp/tensorboard")
    parser.add_argument("--mode", choices=("mcts", "selfplay", "cycle", "envstep"), help="profile mode shown in the heading")
    parser.add_argument("--steps", type=int, default=1, help="profiled calls represented by the trace")
    parser.add_argument("--top", type=int, default=20, help="number of individual XLA ops to print")
    parser.add_argument("--json", help="Write device timing and operation breakdown as JSON")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be at least 1")

    path = find_trace(args.trace)
    processes: dict[int, str] = {}
    threads: dict[tuple[int, int], str] = {}
    lane_ops: dict[tuple[int, int], collections.Counter[str]] = collections.defaultdict(collections.Counter)
    lane_bounds: dict[tuple[int, int], list[float]] = {}
    lane_intervals = collections.defaultdict(list)
    event_count = 0

    for event in iter_trace_events(path):
        phase = event.get("ph")
        pid = event.get("pid")
        tid = event.get("tid")
        if phase == "M" and event.get("name") == "process_name":
            processes[pid] = event.get("args", {}).get("name", "")
        elif phase == "M" and event.get("name") == "thread_name":
            threads[(pid, tid)] = event.get("args", {}).get("name", "")
        elif phase == "X" and isinstance(event.get("dur"), (int, float)):
            lane = (pid, tid)
            duration = event["dur"]
            lane_ops[lane][event.get("name", "<unnamed>")] += duration
            start = event.get("ts", 0.0)
            end = start + duration
            lane_intervals[lane].append((start, end))
            bounds = lane_bounds.setdefault(lane, [start, end])
            bounds[0] = min(bounds[0], start)
            bounds[1] = max(bounds[1], end)
            event_count += 1

    per_device: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    device_bounds: dict[str, list[float]] = {}
    device_intervals = collections.defaultdict(list)
    for lane, ops in lane_ops.items():
        process_name = processes.get(lane[0], "")
        if not process_name.startswith("/device:") or threads.get(lane) != "XLA Ops":
            continue
        per_device[process_name].update(ops)
        device_intervals[process_name].extend(lane_intervals[lane])
        start, end = lane_bounds[lane]
        bounds = device_bounds.setdefault(process_name, [start, end])
        bounds[0] = min(bounds[0], start)
        bounds[1] = max(bounds[1], end)

    if not per_device:
        raise SystemExit(f"no device XLA Ops duration events found in {path}")

    device = max(per_device, key=lambda name: sum(per_device[name].values()))
    ops = per_device[device]
    total = sum(ops.values())
    span = device_bounds[device][1] - device_bounds[device][0]
    divisor = 1e3 * args.steps
    global_span = max(b[1] for b in device_bounds.values()) - min(b[0] for b in device_bounds.values())
    device_report = {
        name: {"op_ms": sum(per_device[name].values()) / divisor,
               "span_ms": (device_bounds[name][1] - device_bounds[name][0]) / divisor,
               "covered_ms": covered_time(device_intervals[name]) / divisor,
               "coverage_percent": 100 * covered_time(device_intervals[name]) / global_span}
        for name in sorted(per_device)
    }
    heading = f"{args.mode} profile" if args.mode else "profile"

    print(f"{heading}: {path}")
    print(f"device: {device} (busiest of {sorted(per_device)})")
    print(f"XLA op time: {total / divisor:.3f} ms/call (sum of XLA Ops durations)")
    print(f"device span: {span / divisor:.3f} ms/call (first op start to last op end)")
    print("Operation sums include overlapping events and control-flow containers; percentages are not exclusive wall time.")
    print(f"duration events parsed: {event_count:,}")

    if len(per_device) > 1:
        print("\nper-device XLA op time:")
        for name in sorted(per_device):
            device_total = sum(per_device[name].values())
            print(f"  {name:<14} {device_total / divisor:10.3f} ms/call")
        print("\nper-device XLA timeline coverage (not FLOP utilization):")
        for name, row in device_report.items():
            print(f"  {name:<14} {row['covered_ms']:10.3f} ms/call  {row['coverage_percent']:5.1f}% of shared device window")

    families: collections.Counter[str] = collections.Counter()
    for name, duration in ops.items():
        families[op_family(name)] += duration
    width = max(map(len, families))
    print("\nmajor op families:")
    for family, duration in families.most_common():
        print(f"  {family:<{width}}  {duration / divisor:10.3f} ms/call  {100 * duration / total:5.1f}%")

    print(f"\ntop {args.top} individual XLA ops:")
    for name, duration in ops.most_common(args.top):
        print(f"  {duration / divisor:10.3f} ms/call  {100 * duration / total:5.1f}%  {name}")
    if args.json:
        with open(args.json, "w") as output:
            json.dump({"devices": device_report, "busiest_device": device,
                       "families": {name: {"ms": duration / divisor, "percent": 100 * duration / total}
                                    for name, duration in families.items()}}, output, indent=2)


if __name__ == "__main__":
    main()
