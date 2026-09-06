"""Run an explicit, sequential MuZero plan with completion and data gates.

Run through uv run. This host supervisor never imports JAX or owns the TPU.
Each child must finish and release the TPU before the next starts.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
import tomllib

from nanoalphazero.research.muzero.status import inspect_run


def check_completed(directory, require_coverage=False):
    manifest = json.loads((directory / "manifest.json").read_text())
    config = json.loads((directory / "config.json").read_text())
    audit = inspect_run(directory)
    if not manifest.get("finished") or audit["state"] != "complete":
        raise RuntimeError("Training did not finish normally")
    if (not audit["updates"] or audit["updates"] != manifest.get("completed_updates")
            or len(manifest.get("devices", [])) != 4
            or len(manifest.get("sharding", {}).get("training_local_shapes", [])) != 4):
        raise RuntimeError("Four-device optimizer completion gate failed")
    if (audit["anomalies"] or audit["nonfinite_metrics"] or audit["replay_flow_failures"]
            or audit["incomplete_metric_records"]):
        raise RuntimeError(f"Data/numerical completion gate failed: {audit}")
    if config["save_checkpoints"] or audit["checkpoint_files"]:
        raise RuntimeError("No-checkpoint gate failed")
    if manifest.get("stop_reason") == "operator_stop_file":
        raise RuntimeError("Operator stopped this phase; queue will not advance")
    if require_coverage and manifest.get("stop_reason") != "opening_coverage_reached":
        raise RuntimeError("Budget ended without repeated opening coverage; inspect before continuing")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hex-eval-engine-path", type=Path, required=True)
    args = parser.parse_args()
    plan = tomllib.loads(args.plan.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "plan.toml").write_text(args.plan.read_text())
    source = Path("src/nanoalphazero/research/muzero")

    def hashes():
        return {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [Path(__file__), args.plan, *source.glob("*.py"),
                          *[Path(s["config"]) for s in plan["phase"]]]}

    frozen = hashes()
    state = dict(started=datetime.now(timezone.utc).isoformat(), completed=[], source_sha256=frozen)

    def record(**fields):
        state.update(fields, updated=datetime.now(timezone.utc).isoformat())
        temporary = args.output / "status.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(args.output / "status.json")

    try:
        if "prerequisite" in plan:
            prerequisite = Path(plan["prerequisite"])
            deadline = time.monotonic() + 7200
            record(status="waiting_for_smoke", prerequisite=str(prerequisite))
            while True:
                path = prerequisite / "manifest.json"
                manifest = json.loads(path.read_text()) if path.exists() else {}
                if manifest.get("finished"):
                    state["smoke_audit"] = check_completed(prerequisite)
                    memory = manifest.get("memory_stats", [])
                    if len(memory) != 4 or any(
                        m.get("peak_bytes_reserved", float("inf")) > plan["smoke_max_reserved_bytes"]
                        for m in memory
                    ):
                        raise RuntimeError("Smoke memory margin gate failed; inspect before full replay")
                    # Manifest writing precedes process teardown. Allow this
                    # same smoke to release its lock before phase admission.
                    for _ in range(60):
                        owner = subprocess.run(["fuser", "/tmp/libtpu_lockfile"], capture_output=True)
                        if owner.returncode != 0:
                            break
                        time.sleep(1)
                    break
                if time.monotonic() > deadline:
                    raise RuntimeError("Smoke completion wait exceeded two hours")
                owner = subprocess.run(["fuser", "/tmp/libtpu_lockfile"], capture_output=True, text=True)
                if owner.returncode != 0:
                    raise RuntimeError("Smoke has no live TPU owner and no completion manifest")
                for pid in owner.stdout.split():
                    command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
                    if prerequisite.name not in command:
                        raise RuntimeError("TPU owner is not the prerequisite smoke")
                time.sleep(10)
        for phase in plan["phase"]:
            if hashes() != frozen:
                raise RuntimeError("Source/config changed during queue; refusing to mix experiments")
            if subprocess.run(["fuser", "/tmp/libtpu_lockfile"], capture_output=True).returncode == 0:
                raise RuntimeError("TPU has a live owner; queue will not compete for it")
            directory = Path(phase["output"])
            if directory.exists():
                raise RuntimeError(f"Refusing to replace existing run: {directory}")
            log = args.output / f'{phase["name"]}.log'
            command = ["uv", "run", "muzero-train", phase["config"], "--no-save",
                       "--output", str(directory), "--hex-eval-engine-path", str(args.hex_eval_engine_path),
                       "--stop-file", str(args.output / f'stop-{phase["name"]}')]
            record(status="running", phase=phase["name"], output=str(directory), command=command, log=str(log))
            with log.open("x") as stream:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'{phase["name"]} exited {result.returncode}; inspect {log}')
            audit = check_completed(directory, phase.get("require_coverage", False))
            state["completed"].append(audit)
            record(status="between_runs")
        record(status="complete")
    except Exception as error:
        record(status="failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
