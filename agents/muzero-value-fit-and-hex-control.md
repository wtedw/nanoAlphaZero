# Recurrent value fitting and Hex cross-check

Recorded 2026-10-17 (Japan time), following the
[target/mask/gradient audit](muzero-value-target-audit.md). No production loss,
training setting, or consistency loss was changed. Nothing was committed.

## Chess fitting experiment

The user authorized interrupting the active chess run. It was stopped through
its tmux launch with Ctrl-C at logged cycle 3745. All four workers exited cleanly;
the latest saved partial checkpoint is cycle 3600. It does not contain replay
buffers or live environments. The experiment instead used the already frozen
cycle-3200 checkpoint to preserve comparability with the search comparison.

`tests/helpers/fit_chess_value_diagnostic.py` prepares 256 windows from distinct
recorded games: 128 training and 128 held-out. Each window has six live positions
starting at ply 30 or later. Actions and legal masks were checked against
python-chess. Value labels are recorded game outcomes in the current player's
perspective, not optimal-play labels. Policy targets were regenerated using the
frozen model with exact Gumbel m=16, n=32, zero Gumbel noise, then BNK128 compressed;
these are not recovered historical replay targets.

Two branches start from identical weights and fresh AdamW moments, at learning
rate 1e-4 without warmup, and fit the whole 128-window training set for 200 updates.
They use the production unroll/loss and rematerialization. Only the diagnostic
objective differs: value alone versus policy plus value. No fitted model replaces
the running model or its checkpoint. Hardware: four TPU v4 hosts, 16 devices.

Mean value MSE across six depths:

| Objective | Initial train | Final train | Initial held-out | Final held-out |
|---|---:|---:|---:|---:|
| Value only | 0.645276 | 0.000000914 | 0.552242 | 0.775234 |
| Policy + value | 0.645276 | 0.001020 | 0.552242 | 0.629106 |

Initial policy/value gradient cosine was +0.0404 in dynamics and -0.0150
in representation: no strong initial conflict. Both branches can fit this data.
This rules out an inability to fit these targets, not all optimization problems.
Held-out deterioration is expected to be possible in this deliberately tiny,
repeated fixed-data experiment; it does not prove the online replay run overfits.
Fresh successor observations do not receive extra direct value supervision in
this experiment. The endpoint above is fixed at update 200, not the best update.

Commands (from the multihost worktree root):

```bash
JAX_PLATFORMS=cpu taskset -c 0-15 uv run --no-sync python -u \
  tests/helpers/fit_chess_value_diagnostic.py --prepare \
  --output artifacts/value-fit-tpu-20261017
uv run --no-sync pod run --timeout 900 -- python -u \
  tests/helpers/fit_chess_value_diagnostic.py \
  --output artifacts/value-fit-tpu-20261017 --steps 200
```

Both completed. Launcher `logs/pod/20261016T153817-1418b876c5b9/result.json`
reports exit 0 for every worker. Raw data, metrics, gradients and interruption
record are in `artifacts/value-fit-tpu-20261017/`; results are `fit-results.json`.
These paths are machine-local ignored artifacts. Use a new output directory to
preserve them when repeating the experiment.

## Historical strong Hex run: dainty-sunset-0

Found in `artifacts/trackio/nanoAlphaZero-muzero.db`, run ID
`8b0b25ecd048403b94cbb3dc6eec7edf`, with metrics through cycle 1433.
Its launcher is `logs/pod/20261014T020520-ef0a528ba2fa`. The launch has no
checkpoint destination, saved configuration has no checkpoint period, and
Trackio has no model artifacts for the run. Therefore its weights could not be
loaded for a new controlled search comparison.

However, it logged all 81 first-move opening values every 100 cycles. The existing
`diagnostics/hex.py` probe compares `f(h(real successor))` with
`f(g(h(empty board), opening action))`, both from White's perspective, against
solved opening values. A value of -1 denotes a winning opening for Black.
This is direct evidence from the trained model, not from the new small control.

| Cycle | Fresh-board MSE | Recurrent MSE | Fresh correct signs / 81 | Recurrent correct signs / 81 |
|---|---:|---:|---:|---:|
| 400 | 0.3100 | 0.2851 | 74 | 78 |
| 800 | 0.3437 | 0.4041 | 70 | 69 |
| 1200 | 0.4193 | 0.3459 | 66 | 71 |
| 1300 | 0.3137 | 0.1629 | 70 | 77 |
| 1400 | 0.1726 | 0.3328 | 75 | 71 |

At cycle 1400, latent/fresh value MSE is 0.12187, while logged training value
loss is 0.004199. Recurrent opening accuracy is nonmonotonic even in this run.
At every recorded cycle from 100 through 1400, the highest-policy opening,
best fresh-value opening, and best recurrent-value opening all have winning
Black ground truth. Those three rankings are not the actual MCTS choice and
do not prove perfect whole-game play. They show how substantial value errors
can coexist with a correct important decision. Hex success does not establish
that its recurrent values are accurate everywhere.

This run used a 256x32 model, 1sh 16+8 search, terminal-policy `none`, 40 exploratory
moves, and dense 81-action targets. These differ from the chess run and prevent
attributing game differences to a single factor. Its launch disabled MoHex
evaluation. Extracted arrays and summaries are preserved in
`artifacts/value-fit-tpu-20261017/dainty-sunset-opening-audit.json`.

## Short fresh Hex control and next discriminating experiment

A separate 128x4 Hex9 control completed 200 cycles / 4000 updates on all 16 devices,
with exact m=16/n=32 search, batch 512, terminal-policy uniform, no tracking or
external evaluation. It saved a partial checkpoint under
`artifacts/hex9-recurrent-control-20261017/latest.ckpt`. Launcher
`logs/pod/20261016T154600-0d6dafc3b910/result.json` reports all workers exit 0.
The exact command is retained in that launch's manifest. This is a short execution
control, not a strength-matched comparison with the mature Hex model. A CPU probe
of its saved model uses the same existing opening diagnostic; output is
`artifacts/hex9-recurrent-control-20261017/opening-probe.json`.
The probe completed: fresh-board opening MSE 1.01857, recurrent MSE 1.06297,
fresh sign accuracy 33/81, and latent/fresh MSE 0.01405. This small control has
not learned the solved opening values; agreement between its two value paths
does not imply accuracy. Its short training also logged 26 overwritten completed
buffer entries, so it should not be treated as a clean mature-strength baseline.

Together with the earlier controlled chess search comparison, the evidence
points to recurrent value generalization/calibration affecting chess decisions,
without a demonstrated sign/index/mask bug or disconnected dynamics gradients.
The cause of the cycle-1500 onset remains unresolved. Before adding consistency,
measure value errors on search-selected versus ordinary played successors using
the same fixed held-out positions across early and late checkpoints. This can
separate search exploiting optimistic errors from broad predictive deterioration.
Use actual replay data when available to test supervision coverage and target
noise; this fitting dataset is only a proxy. Keep real-board value as a comparator,
not as assumed ground truth.

The long chess run remains stopped. After both bounded TPU jobs completed,
`uv run --no-sync pod status` confirmed all four hosts free with zero active jobs.
