# Observing Hex evaluation during training

This repository exposes periodic model-vs-MoHex results as local,
machine-readable files. Use these files as the source of truth when monitoring
an active or completed training run. W&B is optional and is not required for
agent access.

## Start a run with Hex evaluation

Hex evaluation is disabled unless `--hex-eval-period` is nonzero. The MoHex
executable must be supplied explicitly:

```bash
uv run train --env hex4 --no-play \
  --hex-eval-period 100 \
  --hex-eval-engine-path /absolute/path/to/mohex
```

Training prints both of these paths during startup:

```text
Logging this run to logs/hex4_YYYYMMDD-HHMMSS.txt
Hex evaluation data: logs/hex4_YYYYMMDD-HHMMSS.hex-eval
```

The shared timestamp identifies the text log and its evaluation-data directory.
Do not assume W&B is enabled or accessible.

## Find the current observation directory

When the startup output is available, use the exact path printed after
`Hex evaluation data:`. Otherwise, list the directories newest first:

```bash
ls -dt logs/hex*.hex-eval
```

If several training jobs may be active, correlate the timestamp with the
matching `logs/hex*.txt` file or process command. Do not blindly treat the
newest directory as the intended run.

## Read current state

```bash
jq . logs/hex4_YYYYMMDD-HHMMSS.hex-eval/status.json
```

`state` has one of four meanings:

- `waiting`: training has not reached the first scheduled evaluation cycle.
- `running`: a scheduled MoHex match is in progress.
- `complete`: the latest scheduled evaluation completed successfully.
- `failed`: the evaluation raised an error. Inspect `error_type` and `error`.

`status.json` is atomically replaced, so it is safe to read while training is
running. Do not interpret absence of `latest.json` as a score of zero; it means
that no evaluation has completed yet.

## Read the latest score

```bash
jq '{
  cycle,
  train_step,
  score: .model_win_rate,
  model_wins,
  model_losses,
  unscored,
  perfect_opening_fraction
}' logs/hex4_YYYYMMDD-HHMMSS.hex-eval/latest.json
```

A valid completed evaluation should have:

```text
state == "complete"
real_games == board_size * board_size
model_wins + model_losses + unscored == real_games
unscored == 0
```

The primary strength metric is `model_win_rate`. On Hex 4 it changes in steps
of 1/16 because the suite plays one game for each first move. Also inspect
`opening_results` to see which specific openings changed between evaluations.

## Read progress over time

`history.jsonl` contains one complete result per scheduled evaluation cycle:

```bash
jq -c '{cycle, train_step, score: .model_win_rate, model_wins, model_losses}' \
  logs/hex4_YYYYMMDD-HHMMSS.hex-eval/history.jsonl
```

Read only the newest result with:

```bash
tail -n 1 logs/hex4_YYYYMMDD-HHMMSS.hex-eval/history.jsonl | jq .
```

The file is append-only during a run. Compare rows with identical board size,
opening suite, model seat, MoHex binary, and MoHex configuration. A higher win
rate under the same evaluation setup indicates progress; never compare scores
from different opponent budgets as if they were the same benchmark.

## Inspect individual games

Each completed cycle has a full move record:

```text
games-cycle-000100.jsonl
games-cycle-000200.jsonl
```

Each line is one opening and records its winner, termination, ply count, and
ordered moves. Use these files to diagnose an opening that changed from `W` to
`L`, an unscored game, or an engine resignation.

## Terminal and text-log signal

Every successful evaluation emits one stable line to both the terminal and the
training text log:

```text
HEX_EVAL_RESULT cycle=100 score=0.375000 wins=6 losses=10 unscored=0
```

Find all such results with:

```bash
rg 'HEX_EVAL_RESULT' logs/hex4_YYYYMMDD-HHMMSS.txt
```

A failure similarly emits `HEX_EVAL_FAILED`; use `status.json` for its detailed
error. Prefer the JSON files over parsing ordinary human-readable training
output.

## Data safety

These files may be read while training is active. Do not edit, truncate,
delete, or regenerate an observation directory. Do not start a second JAX/TPU
process merely to inspect results; reading the JSON and JSONL files does not
initialize JAX or contend for the TPU.
