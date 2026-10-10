"""Summarize preserved ACPL records and fixed-sample absorbing probes."""
import json
from pathlib import Path
import statistics

output = Path("artifacts/chess-absorbing-history-20261010")
output.mkdir(parents=True, exist_ok=True)
rows = []
for name in ("muzero-chess-overnight-20261008-165124", "muzero-chess-resume-20261009-100905"):
    base = Path("artifacts") / name
    training = {}
    for line in (base / "train.log").open():
        try:
            row = json.loads(line)
        except (ValueError, TypeError):
            continue
        if "cycle" in row and "loss_v" in row:
            training[row["cycle"]] = row
    for summary in sorted((base / "acpl").glob("*/summary.json")):
        evaluation = json.loads(summary.read_text())
        manifest = json.loads((summary.parent / "manifest.json").read_text())
        cycle = manifest["cycle"]
        recent = [training[c] for c in range(cycle - 99, cycle + 1) if c in training]
        row = dict(cycle=cycle, acpl=(evaluation["acpl/acpl_white"] + evaluation["acpl/acpl_black"])/2,
                   summary=str(summary), games=evaluation["acpl/games"],
                   training_window=len(recent), metrics={})
        for field in ("loss_v", "loss_r", "loss_pi", "norms/grad_norm", "norms/param_norm",
                      "selfplay/ep_term_step_avg", "selfplay_buffer/error_count", "selfplay_buffer/missing_count"):
            values = [x[field] for x in recent if field in x]
            if values:
                row["metrics"][field] = statistics.mean(values)
        rows.append(row)
probes = []
for cycle, directory in ((2400, output / "cycle-2400"),
                         (9200, Path("artifacts/chess-absorbing-probe-20261010")),
                         (9600, output / "latest")):
    p = directory / "report.json"
    if p.exists():
        probes.append(dict(cycle=cycle, report=str(p), result=json.loads(p.read_text())))
report = dict(acpl=rows, probes=probes,
              limitations=["Cycles 400/800 from this lineage were overwritten.",
                           "All absorbing comparisons use the cycle-9200 replay sample, not each model's own replay.",
                           "ACPL measures changing self-play trajectories; this is not a fixed-position strength test."])
(output / "history.json").write_text(json.dumps(report, indent=2) + "\n")
for r in rows:
    print(r["cycle"], round(r["acpl"], 2), {k: round(v, 5) for k, v in r["metrics"].items()})

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    cycles = [r["cycle"] for r in rows]
    axes[0].plot(cycles, [r["acpl"] for r in rows], marker="o")
    axes[0].set_ylabel("ACPL (mean of White/Black)")
    axes[0].set_ylim(35, 110)
    axes[0].annotate("Cycle 400: 233 (above plot range)", (400, 105), fontsize=9)
    axes[0].set_title("Chess evaluation history; lower ACPL is better")
    axes[1].plot(cycles, [r["metrics"].get("loss_v", float("nan")) for r in rows], marker="o", label="Value MSE")
    axes[1].plot(cycles, [r["metrics"].get("loss_pi", float("nan")) for r in rows], marker="o", label="Policy KL")
    axes[1].set_ylabel("Training loss (preceding 100 cycles)")
    axes[1].set_xlabel("Training cycle")
    axes[1].legend()
    for ax in axes:
        ax.axvline(2400, color="gray", linestyle="--", label="Resume with fresh buffers")
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output / "history.png", dpi=150)
