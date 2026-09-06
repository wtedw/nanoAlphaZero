"""Read-only CE/KL trajectory comparison; uses no JAX or accelerator runtime.

Run with: uv run evals/muzero/compare_trajectories.py OLD_RUN NEW_RUN
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import zipfile


FIELDS = ("loss_v", "muzero/loss_r", "norms/grad_norm", "norms/param_norm",
          "norms/update_norm", "muzero/real_transitions")


def records(path):
    result = {}
    for line in (path / "metrics.jsonl").read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # A live writer may not have finished its last record.
        result[row["cycle"]] = row
    return result


def compare(old, new):
    before, after = records(old), records(new)
    common = sorted(before.keys() & after.keys())
    mismatches = []
    for cycle in common:
        for field in FIELDS:
            if field not in before[cycle] or field not in after[cycle]:
                mismatches.append({"cycle": cycle, "field": field, "reason": "missing"})
            elif before[cycle][field] != after[cycle][field]:
                mismatches.append({"cycle": cycle, "field": field,
                                   "old": before[cycle][field], "new": after[cycle][field]})
    checked, different, missing = 0, [], []
    # Compare only fully logged cycles and the initial evaluation. Reading
    # ZIP members compares actual .npy payloads, not ZIP timestamps/compression.
    for path in sorted(new.glob("eval-*/*.npz")):
        cycle = int(path.parent.name.removeprefix("eval-"))
        if cycle != 0 and cycle not in common:
            continue
        relative = path.relative_to(new)
        previous = old / relative
        if not previous.exists():
            missing.append(str(relative))
            continue
        with zipfile.ZipFile(previous) as a, zipfile.ZipFile(path) as b:
            if set(a.namelist()) != set(b.namelist()):
                different.append(str(relative) + ": member names")
            for member in sorted(set(a.namelist()) & set(b.namelist())):
                checked += 1
                if a.read(member) != b.read(member):
                    different.append(f"{relative}:{member}")
    return {"timestamp": datetime.now(timezone.utc).isoformat(),
            "old": str(old), "new": str(new), "old_cycles": len(before),
            "new_cycles": len(after), "common_cycles": len(common),
            "last_common_cycle": common[-1] if common else None,
            "fields": FIELDS, "scalar_mismatches": mismatches,
            "evaluation_arrays_checked": checked,
            "different_evaluation_arrays": different, "missing_evaluation_files": missing,
            "interpretation": "Exact equality checks on selected training scalars and saved real-game arrays; not independent seeds."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old", type=Path)
    parser.add_argument("new", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare(args.old, args.new)
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
