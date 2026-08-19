# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "matplotlib==3.10.5",
#   "numpy==2.3.2",
# ]
# ///

"""Rebuild the Desert Snowball test-time scaling table and figure."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import tomllib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import blended_transform_factory


HERE = Path(__file__).resolve().parent
AGENT_RE = re.compile(r"desert-snowball-model(?P<checkpoint>\d+)-(?P<sims>\d+)sims")
HEADER_RE = re.compile(r'^\[([^ ]+) "(.*)"\]$')
PLOT = {
    "title": "Chess Elo vs. test-time compute",
    "x_label": "MCTS simulations per move",
    "y_label": "Elo difference vs. 270M",
    "baseline": 0.0,
    "baseline_label": "270M transformer · 2895 Lichess Blitz*",
    "caption": "*Reported Lichess rating; points show relative Elo over 512 games each.",
    "series": {
        34400: {"label": "~24h", "color": "#2f78c4"},
        68800: {"label": "~48h", "color": "#1f6b34"},
    },
}


@dataclass(frozen=True)
class Result:
    checkpoint: int
    simulations: int
    games: int
    wins: int
    draws: int
    losses: int
    score: float
    score_elo: float
    elo_ci_low: float
    elo_ci_high: float
    bayeselo_relative: int
    source_run: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pgn_headers(path: Path) -> list[dict[str, str]]:
    games: list[dict[str, str]] = []
    headers: dict[str, str] = {}
    with path.open(encoding="utf-8") as pgn:
        for raw_line in pgn:
            line = raw_line.rstrip("\n")
            match = HEADER_RE.match(line)
            if match:
                headers[match.group(1)] = match.group(2)
            elif not line and headers:
                games.append(headers)
                headers = {}
    if headers:
        games.append(headers)
    return games


def score_for(headers: dict[str, str], agent: str) -> float:
    result = headers["Result"]
    if result == "1/2-1/2":
        return 0.5
    if result == "1-0":
        return 1.0 if headers["White"] == agent else 0.0
    if result == "0-1":
        return 1.0 if headers["Black"] == agent else 0.0
    raise ValueError(f"unsupported PGN result: {result!r}")


def elo_from_score(score: np.ndarray | float) -> np.ndarray | float:
    clipped = np.clip(score, 1e-12, 1.0 - 1e-12)
    return 400.0 * np.log10(clipped / (1.0 - clipped))


def paired_bootstrap_ci(
    cluster_scores: np.ndarray,
    *,
    samples: int,
    confidence: float,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values: list[np.ndarray] = []
    batch_size = 2_000
    for start in range(0, samples, batch_size):
        count = min(batch_size, samples - start)
        indices = rng.integers(
            0,
            len(cluster_scores),
            size=(count, len(cluster_scores)),
            dtype=np.int16,
        )
        values.append(elo_from_score(cluster_scores[indices].mean(axis=1)))
    bootstrap_elos = np.concatenate(values)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(bootstrap_elos, [alpha, 1.0 - alpha])
    return float(low), float(high)


def analyze_run(run: dict[str, object], settings: dict[str, object]) -> list[Result]:
    run_dir = (HERE / str(run["path"])).resolve()
    games_path = run_dir / "games.pgn"
    summary_path = run_dir / "summary.json"

    for path, expected in (
        (games_path, run["games_sha256"]),
        (summary_path, run["summary_sha256"]),
    ):
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch for {path}: {actual} != {expected}")

    summary = json.loads(summary_path.read_text())
    if summary["config_hash"] != run["config_hash"]:
        raise ValueError(f"config hash mismatch in {summary_path}")
    if summary["games_in_pgn"] != summary["expected_games"]:
        raise ValueError(f"incomplete tournament in {summary_path}")
    if any(batch["failures"] for batch in summary["pair_batches"]):
        raise ValueError(f"game failures recorded in {summary_path}")

    games = pgn_headers(games_path)
    if len(games) != summary["games_in_pgn"]:
        raise ValueError(f"PGN count mismatch in {games_path}")

    agents = sorted(
        {
            player
            for game in games
            for player in (game["White"], game["Black"])
            if AGENT_RE.fullmatch(player)
        }
    )
    output: list[Result] = []
    for agent in agents:
        match = AGENT_RE.fullmatch(agent)
        assert match is not None
        checkpoint = int(match.group("checkpoint"))
        simulations = int(match.group("sims"))
        agent_games = [game for game in games if agent in (game["White"], game["Black"])]

        scores_by_opening: defaultdict[str, list[float]] = defaultdict(list)
        colors_by_opening: defaultdict[str, set[str]] = defaultdict(set)
        outcomes: Counter[str] = Counter()
        for game in agent_games:
            score = score_for(game, agent)
            scores_by_opening[game["OpeningIndex"]].append(score)
            colors_by_opening[game["OpeningIndex"]].add(
                "white" if game["White"] == agent else "black"
            )
            outcomes["win" if score == 1.0 else "loss" if score == 0.0 else "draw"] += 1

        if len(agent_games) != 512 or len(scores_by_opening) != 256:
            raise ValueError(f"expected 512 games over 256 openings for {agent}")
        if any(len(scores) != 2 for scores in scores_by_opening.values()):
            raise ValueError(f"openings are not paired for {agent}")
        if any(colors != {"white", "black"} for colors in colors_by_opening.values()):
            raise ValueError(f"openings are not color-reversed for {agent}")

        summary_counts = Counter()
        for batch in summary["pair_batches"]:
            if batch["player_a"] == agent:
                for key, value in batch["results_from_a"].items():
                    singular = {"wins": "win", "draws": "draw", "losses": "loss"}[key]
                    summary_counts[singular] += value
        if outcomes != summary_counts:
            raise ValueError(f"PGN and summary results disagree for {agent}")

        cluster_scores = np.asarray(
            [sum(scores) / 2.0 for _, scores in sorted(scores_by_opening.items())],
            dtype=np.float64,
        )
        score = float(cluster_scores.mean())
        score_elo = float(elo_from_score(score))
        ci_low, ci_high = paired_bootstrap_ci(
            cluster_scores,
            samples=int(settings["bootstrap_samples"]),
            confidence=float(settings["confidence_level"]),
            seed=int(settings["bootstrap_seed"]) + checkpoint * 10 + simulations,
        )

        ratings = summary["bayeselo"]["ratings"]
        relative_bayeselo = int(ratings[agent]["elo"] - ratings[str(settings["opponent"])]["elo"])
        output.append(
            Result(
                checkpoint=checkpoint,
                simulations=simulations,
                games=len(agent_games),
                wins=outcomes["win"],
                draws=outcomes["draw"],
                losses=outcomes["loss"],
                score=score,
                score_elo=score_elo,
                elo_ci_low=ci_low,
                elo_ci_high=ci_high,
                bayeselo_relative=relative_bayeselo,
                source_run=str(run["path"]),
            )
        )
    return output


def write_csv(results: list[Result]) -> None:
    fields = [
        "checkpoint",
        "simulations",
        "games",
        "wins",
        "draws",
        "losses",
        "score_pct",
        "score_elo",
        "elo_ci_low",
        "elo_ci_high",
        "bayeselo_relative",
        "source_run",
    ]
    with (HERE / "results.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "checkpoint": f"model{result.checkpoint}",
                    "simulations": result.simulations,
                    "games": result.games,
                    "wins": result.wins,
                    "draws": result.draws,
                    "losses": result.losses,
                    "score_pct": f"{100.0 * result.score:.2f}",
                    "score_elo": f"{result.score_elo:.2f}",
                    "elo_ci_low": f"{result.elo_ci_low:.2f}",
                    "elo_ci_high": f"{result.elo_ci_high:.2f}",
                    "bayeselo_relative": result.bayeselo_relative,
                    "source_run": result.source_run,
                }
            )


def write_figure(results: list[Result]) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "nanoalphazero-desert-snowball-scaling-v1",
        }
    )
    fig, ax = plt.subplots(figsize=(9.5, 6), dpi=110, constrained_layout=False)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.105, right=0.97, top=0.88, bottom=0.19)

    baseline = float(PLOT["baseline"])
    ymax = max(result.score_elo for result in results)
    ylim_top = ymax * 1.15
    ylim_bottom = -ylim_top * 0.42
    ax.set_ylim(ylim_bottom, ylim_top)
    ax.set_facecolor("#f2f7f2")
    ax.axhspan(ylim_bottom, baseline, color="white", zorder=0)
    ax.axhspan(baseline, ylim_top, color="#eef5ee", zorder=0)
    ax.set_axisbelow(True)
    ax.grid(True, color="#ffffff", linewidth=1.0)

    series = PLOT["series"]
    for checkpoint in (34400, 68800):
        rows = sorted(
            (result for result in results if result.checkpoint == checkpoint),
            key=lambda result: result.simulations,
        )
        x = np.asarray([result.simulations for result in rows])
        y = np.asarray([result.score_elo for result in rows])
        style = series[checkpoint]
        ax.plot(
            x,
            y,
            marker="o",
            markersize=8,
            markerfacecolor=style["color"],
            markeredgecolor="white",
            markeredgewidth=1.2,
            linewidth=2.2,
            solid_joinstyle="round",
            solid_capstyle="round",
            color=style["color"],
            label=style["label"],
            zorder=3,
        )

    baseline_line = ax.axhline(
        baseline,
        color="#d2601a",
        linestyle="--",
        linewidth=1.8,
        zorder=2,
    )
    baseline_line.set_dashes((7, 5))
    label_transform = blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(
        0.995,
        baseline + (ylim_top - ylim_bottom) * 0.018,
        PLOT["baseline_label"],
        transform=label_transform,
        ha="right",
        va="bottom",
        fontsize=11,
        fontstyle="italic",
        color="#d2601a",
        zorder=4,
    )

    ax.set_xscale("log", base=2)
    ax.set_xticks([400, 800, 1600, 3200], labels=["400", "800", "1,600", "3,200"])
    ax.set_xlim(320, 10_000)
    ax.set_xlabel(PLOT["x_label"], fontsize=13, color="#333333", labelpad=10)
    ax.set_ylabel(PLOT["y_label"], fontsize=13, color="#333333", labelpad=10)
    ax.set_title(
        PLOT["title"],
        fontsize=15,
        fontweight="bold",
        color="#1a1a2e",
        pad=15,
    )
    ax.tick_params(axis="both", which="both", labelsize=11, colors="#555555", length=0)
    for spine_name in ("left", "bottom"):
        ax.spines[spine_name].set_color("#cccccc")
        ax.spines[spine_name].set_linewidth(0.8)
    legend = ax.legend(
        title="Training time",
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="#dddddd",
        framealpha=0.95,
        borderpad=0.8,
        labelspacing=0.6,
        fontsize=12,
    )
    legend.get_title().set_fontsize(11)
    legend.get_title().set_color("#666666")
    fig.text(
        0.1,
        0.03,
        PLOT["caption"],
        ha="left",
        fontsize=9.5,
        fontstyle="italic",
        color="#777777",
    )

    output_path = HERE / "test-time-scaling.svg"
    fig.savefig(output_path, metadata={"Date": None})
    plt.close(fig)
    # Matplotlib emits spaces before line breaks in path data. Strip them so
    # generated-file whitespace checks stay useful.
    output_path.write_text(
        "\n".join(line.rstrip() for line in output_path.read_text().splitlines()) + "\n"
    )


def main() -> None:
    settings = tomllib.loads((HERE / "manifest.toml").read_text())
    results = sorted(
        (
            result
            for run in settings["runs"]
            for result in analyze_run(run, settings)
        ),
        key=lambda result: (result.checkpoint, result.simulations),
    )
    expected = {(checkpoint, sims) for checkpoint in (34400, 68800) for sims in (400, 800, 1600, 3200)}
    actual = {(result.checkpoint, result.simulations) for result in results}
    if actual != expected:
        raise ValueError(f"incomplete scaling sweep: {actual}")
    write_csv(results)
    write_figure(results)
    print(f"wrote {HERE / 'results.csv'}")
    print(f"wrote {HERE / 'test-time-scaling.svg'}")


if __name__ == "__main__":
    main()
