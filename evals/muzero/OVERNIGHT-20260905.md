# MuZero Hex5–9 overnight work

Requested work window: 2026-09-05 19:39 UTC through at least 2026-09-06
03:39 UTC. The user requested autonomous implementation, experiments, and
status checks every 30 minutes. No experiment checkpoints are written.

## Starting evidence and next decisions

Hex5 KL seed1 completed 1,700 cycles / 34,000 updates. Search and policy
each converted 13/13 known-winning first moves against MoHex. Balanced search
versus policy score was 50.08%; policy/random 99.92%, search/random 99.42%.
This is narrow first-move coverage, not proof of optimal Hex5 play.
W&B: https://wandb.ai/tdoubleu/nanoAlphaZero-muzero/runs/r7l4o6pv .
Raw data: `artifacts/muzero/hex5-kl-seed1-20260905-a/`.

Hex6 KL seed1 was active at the beginning of this window. Its preceding CE
seed1 run reached 896 cycles / 26,880 updates before the user's KL request.
That earlier run first reached 24/24 search conversions at cycle 300 and
24/24 in both modes at 350, with later regressions to 23/24. Search versus
policy approached 50%; no durable search advantage is established.

Read-only comparison found identical scalar training trajectories (value
loss, reward loss, gradient/parameter/update norms, and transition counts)
for all 500 Hex4 cycles, all 1,700 Hex5 cycles, and the first 112 Hex6 cycles.
All saved evaluation arrays matched too: 264/264 Hex4, 840/840 Hex5, and
72/72 Hex6 arrays at the comparison cutoff. KL subtracts the fixed target
entropy from CE and has the same gradient. These are replications, not
independent seeds or evidence of a strength improvement from changing loss
reporting.

Reproducible comparison:

```sh
uv run evals/muzero/compare_trajectories.py artifacts/muzero/hex6-staged-seed1-nosave-20260905-a artifacts/muzero/hex6-kl-seed1-20260905-a
```

The 20:04 audit extended the identical Hex6 scalar prefix to 191 cycles and
96 evaluation arrays. Complete Hex4/5 comparison JSON and the Hex6 prefix
audit are retained under `artifacts/muzero/night-watch-20260905/`.

**Hex6 replication gate, declared before reaching it:** inspect through
cycle 350, preserving every intermediate result and checking the corresponding
CE prefix. If replay remains valid and the trajectory still reproduces CE,
end this redundant replication and advance to larger boards. If it diverges
or develops anomalies, diagnose before advancing. A controlled early stop
does not complete the configured 105,000-update budget.

**Hex7–9 question:** does the existing implementation collect valid games,
drain contiguous replay, update the full-size model on four TPUs, and then
show learning on successively larger boards? First run each board's separate
two-cycle smoke. Keep model width/depth, train/self-play batches, learning
rate, unroll length, and 16+8 two-rung search at AlphaZero-derived defaults.
Smoke runs only reduce replay capacity, warmup, and optimizer-update count.
Do not interpret four smoke updates as playing strength. Measure costs before
choosing the overnight learning allocation. Initial long-run configs retained
the full default update budget; measured-cost budget decisions below supersede
that initial plan and explicitly record each shorter validation budget.
Finite losses alone will not justify a positive strength claim. Compare
policy/search real play, MoHex results, and held-out multi-step prediction.

## Diagnostic storage and operator controls

New `heldout_batch_size=512` on Hex7–9 changes only independent diagnostic
collection. Hex7–9 self-play and training batches stay 8,192. Hex5 uses
self-play batch 1,024 and the user's required training batch 4,096.
The default zero resolves to the self-play batch for backward compatibility.
Two four-virtual-device CPU CLI runs, with diagnostic batches 8 versus 4,
produced exactly equal training losses, gradient/parameter/update norms, and
transition counts across two cycles / four updates. Saved observation batch
shapes changed as intended. Outputs:
`artifacts/muzero/heldout-{full,small}-cpu-smoke-20260905-a/`.

`muzero-train --stop-file PATH` lets future runs finish the current optimizer
cycle, perform final diagnostics and evaluations, and exit without a
checkpoint. A real CPU Hex4 CLI smoke with evaluation period three and an
existing stop request completed one cycle / one update, ran both MoHex
evaluations at cycle one, and recorded `stop_reason=operator_stop_file`.
Output: `artifacts/muzero/operator-stop-cpu-smoke-20260905-a/`.
The already-running Hex6 process predates this option.

Three additional small spatial Hex7–9 compatibility tests passed on CPU
(102.76 seconds): full-cap legal self-play, production consume/drain,
training, temporary test checkpoint loading, and real-game evaluation.
These do not validate full-size TPU performance.

The focused regression suite passed 27 tests in 104.27 seconds, including
replay/data audits, signed search, all-component gradients, KL, checkpoint
round trips, metrics, and comparison with the original AZ table chart helper:

```sh
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 uv run pytest tests/test_muzero.py tests/test_muzero_spatial.py tests/test_muzero_kl.py tests/test_muzero_data_audit.py tests/test_muzero_metrics.py tests/test_muzero_status.py -q
```

CPU abstract-shape preflight (no full arrays allocated, no TPU initialized):

| Board | MuZero parameters | Staging logical bytes, global | Replay logical bytes, global | Updates/cycle |
|---|---:|---:|---:|---:|
| Hex7 | 23,344,247 | 1,992,622,089 | 3,035,136,005 | 30 |
| Hex8 | 23,398,502 | 3,357,081,609 | 3,895,296,005 | 50 |
| Hex9 | 46,105,431 | 5,331,288,073 | 4,870,144,005 | 25 |

These are logical array payload sizes, **not peak TPU memory estimates**:
compiled temporaries, activation storage, optimizer arrays, allocator behavior,
and actual sharding must be measured on device. Both MuZero towers inherit
the AZ depth/width, so total parameter counts are not matched to one AZ tower.
Source and raw results: `artifacts/muzero/night-watch-20260905/shape-profile.py`
and `shape-profile.json`. Reproduce using
`JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 uv run artifacts/muzero/night-watch-20260905/shape-profile.py`.

Evaluation-only MoHex first-reply latency probe, default configuration and
seed one: Hex7 corner/central opening 0.30/0.17 seconds; Hex8 openings 0/32
5.80/8.98 seconds; Hex9 corner/center 6.61/8.97 seconds. The supplied config
does not override the engine's built-in `max_time=10`, `max_games=99999999`,
or roughly 2 GB maximum memory per engine. Thus "uncapped" historical config
comments mean no additional cap in that file; actual engine search still has
the built-in time budget. Larger-board scores are approximate opponent
strength, not exact optimal-play certificates. The parallel bank can also
have different CPU contention from this one-engine timing probe.

Based on that cost, **before launching** the larger runs, smoke MoHex cadence
was set to cycle two only and long-run cadence to every 50 cycles, aligning
with AZ monitoring and avoiding repeated expensive smoke evaluations. Final
operator stops still force evaluation. No training hyperparameters changed.
Probe source/results are `mohex-latency.py` and `mohex-latency.jsonl` in the
night-watch directory. No engine replies entered training.

Read-only examination of saved current-policy held-out episodes found that
exploratory positions dominate representation-root value error:

| Run/cycle | All real states MSE | Exploratory states MSE | Nonexploratory states MSE |
|---|---:|---:|---:|
| Hex4 / 500 | 0.4847 | 1.1849 | 0.08272 |
| Hex5 / 1700 | 0.3129 | 1.0391 | 0.006274 |
| Hex6 / 250 | 0.2730 | 0.7989 | 0.01558 |

These numbers use every real state in the saved episodes and its actual
outcome, not latent-unroll predictions or a claim about optimal state values.
Replay excludes exploratory starts, while the old held-out scalar mixes them
in. Future unroll diagnostics preserve that mixed scalar and add explicit
sampled-root exploration strata/counts, including root-only and real-state
unroll MSE. A controlled CPU test passed (7.02 seconds) with unequal target
errors and truncated padding, checking that these groups exclude boundary
and padding states. No training behavior changed. Source and raw analysis:
`heldout-root-strata.py` and `heldout-root-strata.json` in the night-watch
directory. The current Hex6 process predates these additional metrics.

## Monitoring

`uv run muzero-status RUN... --output artifacts/muzero/night-watch-20260905`
reads files and lock ownership without importing JAX. It records historical
anomalies, nonfinite metrics, completion/progress, latest evaluations, disk
free space, and checkpoint count. Two focused tests passed.

User timer `muzero-night-watch-20260905.timer` has 16 explicit audit times,
from September 5 20:09 UTC every 30 minutes through September 6 03:39 UTC.
Snapshots, `latest.json`, and `history.jsonl` live in the directory above.
The agent also inspects results, takes action, and posts chat status; the
read-only timer is supporting evidence rather than a substitute for work.

## Reproduction commands

CPU compatibility:

```sh
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 uv run pytest tests/test_muzero_compatibility.py -k spatial -q
JAX_PLATFORMS=cpu uv run pytest tests/test_muzero_status.py -q
```

Planned TPU smoke/training (one owner at a time; substitute board 7, 8, 9):

```sh
uv run muzero-train evals/muzero/hex7-staged-smoke.toml --no-save --output artifacts/muzero/hex7-staged-smoke-20260905-a --hex-eval-engine-path /home/tedpsw/benzene-vanilla-cmake/build/src/mohex/mohex
uv run muzero-train evals/muzero/hex7-kl-night-seed1.toml --no-save --stop-file artifacts/muzero/hex7-kl-night-seed1-20260905.stop --output artifacts/muzero/hex7-kl-night-seed1-20260905-a --hex-eval-engine-path /home/tedpsw/benzene-vanilla-cmake/build/src/mohex/mohex
```

Commands above are planned until their completion is recorded below. Full
resolved configs, source snapshots, W&B URLs, device/memory metadata, raw
evaluations, and actual update counts remain beside each run's metrics.

## Half-hour observations

**20:09 UTC:** audit timer fired successfully. Hex6 cycle 207 / 6,210 updates;
latest evaluation at cycle 200: search 23/24 known-winning openings, policy
17/24, balanced search/policy score 51.98%. No recorded buffer anomalies or
nonfinite metrics, zero checkpoint files, 13.39 GiB filesystem space free.
Hex6 W&B: https://wandb.ai/tdoubleu/nanoAlphaZero-muzero/runs/m9e547j0 .
Hex7–9 full-size TPU runs have not started. Their CPU spatial compatibility
gates and reproducible configs are ready.

**20:39 UTC:** Hex6 cycle 316 / 9,480 updates. Cycle-300 evaluation: search
24/24, policy 23/24 known-winning openings; balanced search/policy 51.15%,
policy/random 97.50%, search/random 97.22%. Historical anomaly/nonfinite
counts remain zero; no checkpoints, 13.24 GiB free. The predeclared cycle-350
replication gate is next, followed by the full-size Hex7 TPU smoke.

**20:49 UTC, Hex6 gate and transition:** cycle 350 converted 24/24 in both
modes. Balanced search/policy score was 50.79%. The final comparison covered
352 fully logged cycles / 10,560 updates, with no mismatches in the six
checked scalar fields or any of 192 saved evaluation arrays. All historical
data-health checks passed. The session-owned KL supervisor was stopped;
`fuser` confirmed no TPU lock owner before the next launch. The Hex6 output's
`operator-status.json` records the deliberate early stop, and
`hex6-final-comparison.json` preserves the final replication audit. About
109 minutes elapsed for this four-TPU run, including initialization and
evaluations; the configured 105,000-update budget was not completed.

Hex7 full-size smoke launched at 20:49 UTC with the command listed above,
under `muzero-hex7-staged-smoke-20260905-a.service`. Its log is
`artifacts/muzero/night-watch-20260905/hex7-smoke.log`. The question is now
whether the 23.3M-parameter model and batch 8,192 fit and complete actual
four-device collection, consume/drain, replay, updates, and evaluation.

**21:00 UTC, Hex7 baseline smoke passed:** finished 20:59:51 UTC. Four
updates, 23,344,247 parameters, train batch 8,192 with four local observation
shards `[2048, 7, 7, 4]`. Collection/replay/training arrays carry the expected
data sharding. Both MoHex modes completed all 49 games (zero model wins,
as expected at four cold-start updates). Finite nonzero gradients, no data
anomalies or checkpoint files. Steady collection took 35.86 seconds; two
optimizer updates took 0.938 seconds. Initial optimizer compilation plus two
updates took 187.82 seconds. Total smoke startup/training/evaluation was about
ten minutes. Per-device peak allocator reservation reached 30.42 GB against
a reported 33.01 GB device limit, leaving little margin for full replay and
larger boards; this allocator measurement is not a live-tensor byte count.

**Next controlled memory experiment:** `hex7-remat-smoke.toml` differs from
the successful baseline smoke only by `remat_unroll=true`. Recompute initial
and recurrent activations during reverse-mode differentiation using JAX's
installed `checkpoint` implementation. This writes no checkpoint files and
does not change architecture, batch size, targets, loss, or optimizer settings.
Question: does recomputation materially reduce peak memory at acceptable
optimizer cost? If it does, use it for the larger full-replay runs; if not,
investigate the actual memory bottleneck. CPU tests comparing masked spatial
unroll losses, gradients, all updated parameter/optimizer arrays, plus KL and
held-out diagnostics passed (four tests, 43.44 seconds; numeric tolerance
3e-6 absolute / 3e-5 relative). Full-device memory/cost evidence is still needed.

**21:09 UTC:** baseline Hex7 smoke is complete and the controlled recomputation
smoke is active after two completed warmup batches. Its first warmup collected
the same 346,323 transitions as the baseline. No checkpoints; 13.14 GiB free.
Command: `uv run muzero-train evals/muzero/hex7-remat-smoke.toml --no-save --output artifacts/muzero/hex7-remat-smoke-20260905-a --hex-eval-engine-path /home/tedpsw/benzene-vanilla-cmake/build/src/mohex/mohex`.
Service: `muzero-hex7-remat-smoke-20260905-a.service`; log:
`artifacts/muzero/night-watch-20260905/hex7-remat-smoke.log`.

**21:15 UTC, recomputation smoke passed:** completed 21:14:36 UTC. Peak
allocator reservation decreased from 30,424,694,784 to 10,885,038,080 bytes
per device (about 64% lower). The second cycle's two optimizer updates took
1.187 seconds versus 0.938 baseline, about 27% more optimizer time; collection
remained 35.86 seconds. Both modes completed all MoHex games, with zero wins
after four updates. CPU baseline regression tests also passed (19 tests,
93.03 seconds). The first-cycle loss differed by 1.2e-7 and gradient norm by
8.9e-7; subsequent self-play can amplify floating-point differences, so this
is not a claim of bitwise-identical training trajectories. The algorithm,
architecture, logical batches, and optimization objective are unchanged.

The measured memory margin justifies enabling `remat_unroll` on Hex7–9.
Full-size Hex8/9 smokes will still gate their own training. No production
network or search module changed, and recomputation writes no checkpoint files.

**21:17 UTC, Hex7 learning trial launched:** the measured-cost first trial is
100 cycles / 3,000 optimizer updates. This explicit budget now appears in
`hex7-kl-night-seed1.toml`; it is not the full budget. The adapter then resolved
199,980 updates; this was corrected later because production AZ implies 150,000
updates (see the September 6 opening-coverage report).
Full replay capacity 2,048,000, training/self-play batch 8,192, learning rate
1e-4, width/depth 256/16, 30 updates/cycle, five-step unroll, and 16+8 search
remain fixed. Diagnostic batches are 512; only the computation strategy and
total trial length differ from the initial full-budget plan.

Question: does this model learn real playing strength and improve held-out
prediction in the first 3,000 updates? Compare initialization, cycle 50, and
cycle 100 policy/random and search/random games; record both MoHex modes,
search/policy comparison, and exploration-stratified unroll diagnostics.
Next decision: data/numerical faults block progression for diagnosis;
otherwise assess actual learning and retain any negative result before the
next board's feasibility/learning experiment. This trial is not enough to
claim strong or optimal Hex7 solely from a falling loss.

Service: `muzero-hex7-kl-night-seed1-20260905-a.service`; log:
`artifacts/muzero/night-watch-20260905/hex7-training.log`. The CLI command is
the training command in the reproduction section above. W&B:
https://wandb.ai/tdoubleu/nanoAlphaZero-muzero/runs/m6tsiyac .

Hex7 initialization evaluation completed before replay warmup: policy/random
50.38%, search/random 47.81%, search/policy 46.77%, balanced over both seats
and all 2,352 legal two-ply openings (4,704 games per comparison). All games
finished and real actions were legal. Preserve this baseline when interpreting
subsequent scores; it is not a trained-model strength result.

**21:31 UTC, replay-flow audit:** fresh-position conservation passed all
1,782 Hex5 warmup/training cycles, all 361 Hex6 warmup/training cycles, and
all seven completed Hex7 warmup cycles. Each check verifies previous fresh
positions plus new eligible positions equals inserted positions plus pending
fresh positions. This complements the identity/duplicate tests and does not
replace them. Warmup anomalies and nonfinite metrics are now included in
the recurring status audit too. Four status tests passed, including an
injected lost-position failure; raw audit:
`artifacts/muzero/night-watch-20260905/flow-audit/`.

**21:39 UTC:** full-replay Hex7 cycle 4 / 120 updates; steady cycles about
54.4 seconds. All 11 warmup/training flow checks passed, no buffer anomalies
or nonfinite metrics, zero checkpoint files, 13.21 GiB free. The first
trained-model strength evaluation is scheduled for cycle 50.

**22:09 UTC:** Hex7 cycle 36 / 1,080 updates; all 43 replay-flow checks pass.
No reward-error counters, buffer anomalies, nonfinite metrics, or checkpoint
files; 13.18 GiB free. Policy loss has declined much faster than value loss.
No playing-strength improvement is claimed before the cycle-50 evaluation.

**22:23 UTC, first Hex7 strength result:** cycle 50 / 1,500 updates produced
policy/random 74.55%, search/random 76.04%, and search/policy 56.36%, compared
with initialization 50.38%, 47.81%, and 46.77%. These paired comparisons
share the same MuZero parameters and opening slate; search spends additional
inference, and these are not compute-matched AZ comparisons. Both modes
converted 0/27 known-winning openings against MoHex: learning is demonstrated,
but strong reference-opponent play is not.

Current-policy held-out nonexploratory-root real-state value MSE was 0.7136,
0.7283, 0.7413, and 0.7653 at unroll lengths 1, 3, 5, and 10. The corresponding
positive terminal-reward predictions were 0.2867, 0.2504, 0.2249, and 0.2105
(target +1); reward learning is still incomplete and longer unrolls degrade.
Mixed exploration/nonexploration value MSE was 0.7839–0.8291. All data checks
passed. Continue the predeclared trial to cycle 100 without tuning against
this intermediate result. Raw games and held-out episodes are under the run's
`eval-000050/`, `mohex-*/`, and `heldout-cycle-000050.npz` paths.

Value labels in the frozen initialization dataset are outcomes under its old
self-play policy, not optimal values or outcomes under the current policy.
Keep that distinction when comparing frozen and current-policy diagnostics;
environment reward targets remain tied to the recorded transitions in both.

**22:39 UTC:** Hex7 cycle 66 / 1,980 updates; all 73 flow checks passed,
no recorded reward/data/numerical errors, zero checkpoints, 13.15 GiB free.
Latest evaluated scores remain those from cycle 50. Complete the fixed
100-cycle trial and then advance to the full-size Hex8 TPU smoke.

**23:09 UTC:** Hex7 cycle 99 / 2,970 updates; all 106 flow checks passed,
no recorded data/numerical errors, zero checkpoints, 13.12 GiB free. Value
and reward losses improved substantially after the midpoint. Wait for the
final real-game evaluation before assessing the completed trial.

**23:11 UTC, Hex7 trial completed:** finished 23:10:50 UTC at cycle 100 /
3,000 updates. Policy and search each won all 4,704 games against random
across both seats of the fixed two-ply opening slate. Search/policy score was
77.38% (up from 56.36% at cycle 50). Against MoHex, search converted 2/27
known-winning openings and policy 0/27. Retain both facts: search now improves
real decisions substantially over its policy, but expert-level Hex7 strength
is not demonstrated and random-opponent saturation is not optimality.

Current-policy held-out unroll-five real-state value MSE was 0.5896 overall
and 0.2713 for nonexploratory roots; positive terminal-reward prediction rose
to 0.8556 and real reward MSE fell to 0.005406. Seven warmup batches collected
2,424,447 transitions; 100 training cycles collected 28,818,755 more. All
107 flow checks passed, with no recorded reward/data/numerical errors.
The full-replay run retained a 10.89 GB peak allocator reservation per device;
peak live-byte statistics were about 4.98 GB, a separate measurement. No
checkpoints were saved. Total start-to-finish wall time was about 113 minutes
on four TPUs, including initialization, warmup, compilation, and evaluation.
Raw final audit: `artifacts/muzero/night-watch-20260905/hex7-final-audit/`.

**23:12 UTC, Hex8 smoke launched:** the Hex7 service completed and `fuser`
confirmed the TPU lock had no owner before launching
`muzero-hex8-staged-smoke-20260905-a.service`. Command:
`uv run muzero-train evals/muzero/hex8-staged-smoke.toml --no-save --output artifacts/muzero/hex8-staged-smoke-20260905-a --hex-eval-engine-path /home/tedpsw/benzene-vanilla-cmake/build/src/mohex/mohex`.
Log: `artifacts/muzero/night-watch-20260905/hex8-smoke.log`. This tests the full
23.4M-parameter model, batch 8,192, and recomputed unrolls on the next board;
its memory, real-data integrity, optimizer, and evaluation results gate training.

**23:19 UTC, independent game audit:** a separate graph-connectivity checker
replayed every saved move prefix in all 14,112 Hex7 final-evaluation games.
It calls neither PGX stepping nor a model to determine winners. Every legal
pre-terminal prefix, winning path, saved reward vector, completion flag, and
summary count agreed. This independently confirms the random-opponent scores
and search/policy advantage. Source: `evals/muzero/audit_hex_games.py`;
raw result: `artifacts/muzero/night-watch-20260905/hex7-connectivity-audit.json`.
Two checker tests passed (0.10 seconds), covering both goal directions and
illegal pre-terminal moves. The first checker draft used reversed goals;
inspection of PGX1's `_turn=1` initialization corrected that audit-only error
before the final check. No training/evaluation implementation changed.

Reproduce:
`JAX_PLATFORMS=cpu uv run evals/muzero/audit_hex_games.py artifacts/muzero/hex7-kl-night-seed1-20260905-a/eval-000100 --size 7 --output /tmp/hex7-connectivity-audit.json`.

**23:26 UTC, Hex8 smoke passed and learning trial launched:** the full-size
four-TPU smoke completed two cycles/four updates at 23:25:26. All four
fresh-position conservation checks passed; no anomalies or nonfinite metrics,
no truncated episodes, no checkpoint files. Training shards were [2048,8,8,4].
Peak reservation was 14,193,688,576 bytes/device. Steady collection took
60.94 seconds and two optimizer updates 1.54 seconds. Both 64-game MoHex
evaluations completed with zero wins, as expected before meaningful training.
Audit: `artifacts/muzero/night-watch-20260905/hex8-smoke-final-audit/`.

The next question is whether Hex7's learning and actual search advantage carry
to Hex8. Only board size and the original AZ board-specific defaults change;
representation/dynamics, KL, unroll 5, rematerialization, seed, and 16+8 search
remain fixed. This first trial explicitly caps training at 50 cycles/2,500
updates (versus AZ's configured 500,000 optimizer updates), retaining full replay, batch 8,192,
LR 0.0001 and b16c256. Initial/final balanced real games, held-out predictions
and MoHex results determine whether to extend learning or diagnose failure.
This is not a matched-budget AZ comparison. No checkpoint is saved.

Service: `muzero-hex8-kl-night-seed1-20260905-a.service`.
Command: `uv run muzero-train evals/muzero/hex8-kl-night-seed1.toml --no-save --output artifacts/muzero/hex8-kl-night-seed1-20260905-a --hex-eval-engine-path /home/tedpsw/benzene-vanilla-cmake/build/src/mohex/mohex --stop-file artifacts/muzero/night-watch-20260905/stop-hex8`.
Log: `artifacts/muzero/night-watch-20260905/hex8-training.log`.

Hex8 W&B: https://wandb.ai/tdoubleu/nanoAlphaZero-muzero/runs/0gwa3qly.

**23:28 UTC, earlier-board independent audits:** the same graph checker
verified every saved move/reward in Hex5's final 3,600 evaluation games and
Hex6 cycle 350's 7,560 games. All summaries agreed, with no illegal moves
before termination. Raw results are `hex5-connectivity-audit.json` and
`hex6-connectivity-audit.json` under the night-watch directory. This validates
the recorded games, not optimality or independence of the fixed opening slate.

**00:09 UTC audit:** Hex8 reached 550 updates, with all 17 warmup/training
fresh-position checks passing and no nonfinite metrics or malformed/truncated
episodes. Steady cycles take 100.16 seconds (38.34 optimizer, 60.94 collection,
0.88 insertion), plus roughly 3.5 seconds host/logging overhead. Initial
balanced scores over 8,064 games/comparison were policy/random 49.901%,
search/random 47.421%, search/policy 45.685%. Final evaluation is still pending;
falling training losses alone are not strength evidence.

**01:20 UTC, Hex8 trial completed:** all 50 cycles/2,500 updates completed at
01:18:52. Training collected 21,922,662 transitions plus 2,735,362 warmup
transitions. All 56 conservation checks passed; no numerical/data anomalies
or checkpoint files. Wall time including initialization/evaluation was about
112 minutes on four TPUs. Peak reservation remained 14.19 GB/device; reported
peak live allocation was 5.83 GB/device.

Final balanced scores over 8,064 games/comparison:
policy/random **86.2103%**, search/random **99.6156%**, search/policy **99.6900%**.
Search won 4,030/4,032 as player 0 and 4,009/4,032 as player 1 against policy.
Both modes still lost all 64 MoHex games (0/32 known-winning conversions).
This is substantial improvement against these opponents, not expert strength.

Fresh held-out nonexploratory-root value MSE at unrolls 1/3/5/10 was
0.6204/0.6302/0.6416/0.6737. Positive terminal reward predictions were only
0.0984/0.0616/0.0547/0.0630 (target +1): reward learning is still weak.
The low aggregate reward MSE (0.0242 at unroll 5) mostly reflects plentiful
zero rewards and must not be called a successful reward model. Further Hex8
training and repeated seeds remain necessary; this short run exhausted its
predeclared budget. Raw evaluations, diagnostics and resolved config remain in
`artifacts/muzero/hex8-kl-night-seed1-20260905-a/`.

**01:20 UTC, Hex9 full-size smoke launched:** Hex8 completed and released the
TPU lock. The next gate tests the original AZ b32c256 model and batch 8,192,
with the same unroll recomputation and search. Only smoke replay and update
budget are reduced. Success requires collection, insertion, four optimizer
updates, legal MoHex evaluation, finite metrics and four-device sharding;
measured memory/cost determine the next learning budget.
Command: `uv run muzero-train evals/muzero/hex9-staged-smoke.toml --no-save --output artifacts/muzero/hex9-staged-smoke-20260905-a --hex-eval-engine-path /home/tedpsw/benzene-vanilla-cmake/build/src/mohex/mohex`.
Service: `muzero-hex9-staged-smoke-20260905-a.service`.
Log: `artifacts/muzero/night-watch-20260905/hex9-smoke.log`.

**01:23 UTC independent Hex8 audit:** all 24,192 final-evaluation games passed
the separate graph-connectivity replay, including legality before termination,
winner/reward agreement and all summary counts. Raw result:
`artifacts/muzero/night-watch-20260905/hex8-connectivity-audit.json`.

**01:50 UTC, Hex9 baseline smoke passed with tight memory:** the b32c256
46,105,431-parameter model completed four optimizer updates and both 81-game
MoHex evaluations (all losses). All four replay conservation checks passed;
no malformed/truncated games, numerical anomalies or checkpoint files.
Training shards were [2048,9,9,4]. Smoke wall time was 27.4 minutes. Steady
collection took 189.93 seconds, insertion 2.06 seconds, two optimizer updates
4.12 seconds. Peak reservation was **29,980,786,688 bytes/device**, leaving
little room for the full replay and larger evaluation batches.

Controlled next change: recompute individual production nested residual blocks
during backward, in addition to whole-unroll recomputation. The research-only
adapter reuses production block/layer implementations and preserves parameter
names, initialization, architecture, batch and optimizer. This is justified by
measured reservation, not a search/model redesign. CPU parameter, masked
forward/gradient, unrolled optimizer and checkpoint-metadata parity tests gate
a repeat smoke. The repeat uses `hex9-block-remat-smoke.toml`, differing only
in `remat_blocks=true`; observed memory and update time determine whether the
tradeoff is suitable for full replay training. No activation recomputation
operation writes a checkpoint file.
