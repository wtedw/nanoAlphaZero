"""Read-only experiment health snapshots; never initializes JAX or a TPU."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import subprocess
import time


def inspect_run(directory, owner_command=""):
    directory = Path(directory)
    if not directory.exists():
        return {"run": directory.name, "state": "not_started"}
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    config_path = directory / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    records, warmups, malformed = [], [], 0
    metric_path = directory / "metrics.jsonl"
    for path, destination in ((directory / "warmup.jsonl", warmups), (metric_path, records)):
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            try:
                destination.append(json.loads(line))
            except json.JSONDecodeError:
                # A writer may be midway through appending its final record.
                malformed += 1
    last = records[-1] if records else {}
    evaluations = [r for r in records if "hex_eval/perfect_opening_wins" in r]
    evaluated = evaluations[-1] if evaluations else {}
    pairings = [r for r in records if "muzero/eval/search-vs-policy/seat0_score" in r]
    paired = pairings[-1] if pairings else {}
    all_records = warmups + records
    abnormal = {}
    for record in all_records:
        for key, value in record.items():
            if key.startswith(("anomalies/", "errors/")) and key != "anomalies/selfplay_buffer/eviction_n_is_from_selfplay" and value:
                abnormal[key] = max(abnormal.get(key, 0), value)
    # Audit fresh-position conservation across warmup AND optimizer cycles.
    # Replay evictions are intentional; this checks the consume/drain boundary.
    previous = None if "--resume" in manifest.get("command", []) and not warmups else 0
    flow_checks, flow_failures = 0, []
    for record in all_records:
        fields = ("muzero/staging/new_eligible", "muzero/drain/inserted_positions",
                  "muzero/drain/remaining_fresh")
        if not all(k in record for k in fields):
            continue
        eligible, inserted, remaining = (record[k] for k in fields)
        if previous is not None:
            flow_checks += 1
            if previous + eligible != inserted + remaining:
                flow_failures.append({"cycle": record.get("cycle"),
                                      "warmup": record.get("warmup/loop_n"),
                                      "incoming": previous + eligible,
                                      "accounted": inserted + remaining})
        previous = remaining
    operator_path = directory / "operator-status.json"
    operator = json.loads(operator_path.read_text()) if operator_path.exists() else {}
    state = "complete" if manifest.get("finished") else "running" if directory.name in owner_command else "inactive"
    state = operator.get("status", state)
    result = {
        "run": directory.name, "env": config.get("env"), "state": state,
        "stop_reason": manifest.get("stop_reason"),
        "cycle": last.get("cycle", 0), "planned_cycles": config.get("cycles"),
        "updates": last.get("runner_state/n_updates", 0),
        "training_transitions": sum(r.get("muzero/real_transitions", 0) for r in records),
        "warmup_batches": len(warmups),
        "warmup_transitions": sum(r.get("muzero/real_transitions", 0) for r in warmups),
        "replay_flow_checks": flow_checks, "replay_flow_failures": flow_failures,
        "wandb_url": manifest.get("wandb_url"), "save_checkpoints": config.get("save_checkpoints"),
        "checkpoint_files": len(list(directory.glob("*.safetensors"))) + len(list(directory.glob("*.safetensors.gz"))),
        "anomalies": abnormal,
        "nonfinite_metrics": sum(isinstance(v, float) and not math.isfinite(v) for r in all_records for v in r.values()),
        "incomplete_metric_records": malformed,
        "metrics_age_seconds": round(time.time() - metric_path.stat().st_mtime) if metric_path.exists() else None,
        "last_evaluation_cycle": evaluated.get("cycle"),
        "last_self_match_cycle": paired.get("cycle"),
        "search_opening_wins": evaluated.get("hex_eval/perfect_opening_wins"),
        "policy_opening_wins": evaluated.get("muzero/hex_eval_policy/perfect_opening_wins"),
        "winning_openings": evaluated.get("hex_eval/perfect_opening_total"),
        "opening_coverage_streak": last.get("muzero/opening_coverage_streak"),
        "opening_coverage_required": config.get("opening_coverage_streak", 0),
        "latest_metrics": {k: v for k, v in last.items() if k in (
            "loss_pi", "loss_v", "muzero/loss_r", "total_loss", "norms/grad_norm",
            "muzero/timing/collection_seconds", "muzero/timing/replay_insert_seconds",
            "muzero/timing/optimizer_seconds", "muzero/selfplay/truncated_fraction")},
    }
    for a, b in (("policy", "random"), ("search", "random"), ("search", "policy")):
        direct = paired.get(f"muzero/eval/{a}-vs-{b}/seat0_score")
        reverse = paired.get(f"muzero/eval/{b}-vs-{a}/seat0_score")
        if direct is not None and reverse is not None:
            result[f"{a}_vs_{b}_balanced"] = (direct + 1 - reverse) / 2
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    owner = subprocess.run(["fuser", "/tmp/libtpu_lockfile"], capture_output=True, text=True)
    pids = [int(x) for x in owner.stdout.split() if x.isdigit()]
    commands = []
    for pid in pids:
        try:
            commands.append(Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode())
        except FileNotFoundError:
            pass
    now = datetime.now(timezone.utc)
    args.output.mkdir(parents=True, exist_ok=True)
    snapshot = {"timestamp": now.isoformat(), "tpu_owner_pids": pids,
                "disk_free_bytes": shutil.disk_usage(args.output).free,
                "runs": [inspect_run(path, "\n".join(commands)) for path in args.runs]}
    encoded = json.dumps(snapshot, indent=2) + "\n"
    (args.output / (now.strftime("%Y%m%dT%H%M%S") + ".json")).write_text(encoded)
    temporary = args.output / "latest.tmp"
    temporary.write_text(encoded)
    temporary.replace(args.output / "latest.json")
    with (args.output / "history.jsonl").open("a") as stream:
        stream.write(json.dumps(snapshot) + "\n")
    print(json.dumps(snapshot), flush=True)


if __name__ == "__main__":
    main()
