"""Non-interactive BayesElo scoring for tournament PGNs."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


def _is_nano_cli_wrapper(path: Path) -> bool:
    """Reject Nano's own `bayeselo` entry point when locating the C++ engine."""
    try:
        prefix = path.read_bytes()[:4096]
    except OSError:
        return False
    return b"nanoalphazero.cli" in prefix and b"bayeselo_main" in prefix


def find_binary(explicit: str | None = None) -> Path:
    candidates = [
        explicit,
        os.environ.get("BAYESELO_PATH"),
        shutil.which("bayeselo"),
        str(Path.cwd() / "artifacts" / "bayeselo" / "BayesElo" / "bayeselo"),
        str(Path.home() / "BayesElo" / "BayesElo" / "bayeselo"),
        str(Path.home() / "BayesElo" / "bayeselo"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.is_file() and not _is_nano_cli_wrapper(path):
            return path.resolve()
    raise FileNotFoundError(
        "BayesElo binary not found. Run `uv run assets fetch bayeselo`, put "
        "the native BayesElo engine on PATH, set BAYESELO_PATH, or configure "
        "bayeselo.binary. Nano's `uv run bayeselo` command is a scoring wrapper, "
        "not the native engine."
    )


def parse_ratings(output: str) -> dict[str, dict[str, int]]:
    ratings: dict[str, dict[str, int]] = {}
    pattern = re.compile(
        r"\s*(\d+)\s+(\S+)\s+(-?\d+)\s+(\d+)\s+(\d+)\s+"
        r"(\d+)\s+(\d+)%\s+(-?\d+)\s+(\d+)%"
    )
    for line in output.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        ratings[match.group(2)] = {
            "rank": int(match.group(1)),
            "elo": int(match.group(3)),
            "plus": int(match.group(4)),
            "minus": int(match.group(5)),
            "games": int(match.group(6)),
            "score_pct": int(match.group(7)),
            "opponent_elo": int(match.group(8)),
            "draws_pct": int(match.group(9)),
        }
    return ratings


def score_pgn(pgn: str | Path, *, binary: str | None = None) -> dict[str, Any]:
    pgn_path = Path(pgn).expanduser().resolve()
    executable = find_binary(binary)
    commands = (
        f"readpgn {pgn_path}\n"
        "elo\n"
        "mm 1 1\n"
        "exactdist\n"
        "ratings\n"
        "advantage\n"
        "drawelo\n"
        "los\n"
        "x\n"
        "x\n"
    )
    process = subprocess.run(
        [str(executable)],
        input=commands,
        capture_output=True,
        text=True,
        cwd=executable.parent,
        check=False,
    )
    if process.returncode:
        raise RuntimeError(
            f"BayesElo exited {process.returncode}:\n{process.stderr}\n{process.stdout}"
        )
    return {
        "binary": str(executable),
        "pgn": str(pgn_path),
        "ratings": parse_ratings(process.stdout),
        "elo_is_pool_relative": True,
        "raw_output": process.stdout,
    }


def bayeselo_main(
    pgn: str, *, binary: str | None = None, out: str | None = None
) -> None:
    result = score_pgn(pgn, binary=binary)
    output = Path(out).expanduser().resolve() if out else Path(pgn).with_suffix(".elo.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    for name, rating in sorted(
        result["ratings"].items(), key=lambda item: -item[1]["elo"]
    ):
        print(
            f"{name:>24} {rating['elo']:>5} "
            f"+{rating['plus']}/-{rating['minus']} ({rating['games']} games)"
        )
    print(f"wrote {output}")

