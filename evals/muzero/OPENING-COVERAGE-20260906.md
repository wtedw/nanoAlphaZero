# Repeated known-winning-opening coverage

The user requested finishing the Hex9 work and getting all Hex variants to win
their known-winning openings. Existing Hex4/5/6 runs have each reached complete
coverage in both policy and learned-search modes. Hex6 subsequently regressed
in the CE run; those results remain recorded. Hex7's first trial reached 2/27
search and 0/27 policy, Hex8 0/32 both. Neither demonstrated expert strength.

Read-only historical streak audit: Hex4 KL first reached three consecutive
complete evaluations at cycle 500 (final streak 3); Hex5 at cycle 1,050
(final streak 16). Hex6 KL stopped after one complete evaluation, while its
earlier CE run first reached three at cycle 650 and ended with a streak of
three, despite intervening regressions. The KL replay replicated the checked
CE trajectory exactly; the CE record is identified explicitly rather than
presented as a new KL result. Raw full evaluation-by-evaluation audit:
`artifacts/muzero/night-watch-20260905/earlier-opening-streaks.json`.

The new stopping gate requires **three consecutive scheduled MoHex evaluations**
with every known-winning opening converted in **both modes**, and zero unscored
games. A regression resets the streak. A cycle without evaluation does not
advance it. This is an opening-suite criterion, not proof of optimal play;
MoHex is an approximate opponent on larger boards. Separate seeds remain
necessary for general learning conclusions.

Plan: finish the first 1,250-update Hex9 learning trial, then run Hex7, Hex8,
and Hex9 toward the repeated coverage criterion. Each training process exits
before the next starts. These are fresh self-play runs because checkpoint
saving remains disabled. The earlier pilot weights cannot be resumed.

Coverage configs inherit the original AZ model, training/self-play batches,
learning rate, replay capacity and maximum update budget. The ceiling is
150,000 updates for Hex7, 500,000 for Hex8, and 2,000,000 for Hex9; these are
ceilings, not budgets claimed to be necessary or already completed. Three
successful evaluations stop a run early. At the observed Hex9 collection cost,
longer-board coverage can take much longer than one night. No curriculum,
search redesign, solved-position training data or extra model loss is added.

`opening-progression-20260906.toml` specifies the queue. Run it with:

```sh
uv run evals/muzero/run_opening_progression.py evals/muzero/opening-progression-20260906.toml --output artifacts/muzero/opening-progression-20260906-a --hex-eval-engine-path /path/to/mohex
```

The supervisor freezes source/config hashes, refuses existing output directories
or a live TPU owner, preserves logs/manifests and stops on process/data errors,
operator stop requests or exhaustion without the required coverage. It does not
silently declare success from a clean process exit. Per-phase stop files are
named `stop-<phase name>` in the supervisor output directory; the training loop
performs final evaluations before exiting. No process/lock is forcibly removed.

The Hex9 repeat smoke tests block-level activation recomputation, prompted by
29.98 GB/device reservation with reduced smoke replay. The research adapter
composes unchanged production blocks and preserves parameter names. CPU tests
passed for exact initialization, masked global/local-pooling forward/gradients
with RVGL enabled and disabled, recurrent optimizer updates, and checkpoint
metadata/loading (three tests, 55.88 seconds). Gate/supervisor/status tests also
passed (seven tests, 0.03 seconds). Full TPU evidence is required before launch.
An integrated four-virtual-device CPU test also exercised coverage-triggered
exit after one optimizer update, forced final held-out/real-game evaluation,
completion metadata and no checkpoint output (40.87 seconds). The predicate
was injected for this control-flow test; the test is not playing-strength data.

At continuation 04:29 UTC, the repeat smoke was launched in
`artifacts/muzero/hex9-block-remat-smoke-20260906-a/`. There was no TPU training
between the first smoke ending 01:49 UTC and that launch. Half-hour file audits
ran through 03:39; they do not constitute training progress during the gap.
The audit timer has now been extended for the continued work.

**04:48 UTC:** the host-only supervisor started as
`muzero-opening-progression-20260906-a.service`, waiting for the running Hex9
smoke. It requires successful four-device completion and peak reservation
below 24 GB/device before full replay training. This leaves explicit headroom
relative to the first smoke's nearly 30 GB. It will not initialize another TPU
process while the smoke owns the lock. Its live status and per-phase logs are
under `artifacts/muzero/opening-progression-20260906-a/`; supervisor log:
`artifacts/muzero/night-watch-20260905/opening-supervisor.log`.
The launched command used the MoHex executable at
`/home/tedpsw/benzene-vanilla-cmake/build/src/mohex/mohex`.

**04:59 UTC negative memory result:** the block-rematerialization smoke
completed all four updates and both 81-game MoHex evaluations, with no data
or numerical errors. Reservation was 28,849,815,552 bytes/device, only 3.8%
below the original 29,980,786,688. Two steady optimizer updates took 4.56
seconds versus 4.12 (10.6% overhead); collection was 189.07 versus 189.93
seconds. First-cycle loss differed by 2.4e-7. Subsequent self-play diverged
slightly from rounding, so bitwise trajectory equivalence is not claimed.
The supervisor failed its predeclared 24 GB margin gate and launched no
training. This was an admission-gate failure, not a TPU out-of-memory error.

**05:01 UTC full-workload memory gate:** a new smoke now allocates the full
AZ replay capacity and runs initial/final full two-ply evaluations and held-out
diagnostics. This directly tests the learning configuration's memory footprint
instead of relying on a conservative reservation estimate. Batch/model/LR,
unroll and block recomputation are unchanged; only warmup/update counts are
reduced. Command:
`uv run muzero-train evals/muzero/hex9-full-replay-smoke.toml --no-save --output artifacts/muzero/hex9-full-replay-smoke-20260906-a --hex-eval-engine-path /home/tedpsw/benzene-vanilla-cmake/build/src/mohex/mohex`.
Service `muzero-hex9-full-replay-smoke-20260906-a.service`; log
`artifacts/muzero/night-watch-20260905/hex9-full-replay-smoke.log`.
The failed supervisor's complete status/log remains preserved.

The replacement plan is `opening-progression-20260906-full-replay.toml`,
supervised by `muzero-opening-progression-20260906-b.service` with status/logs
under `artifacts/muzero/opening-progression-20260906-b/`. It waits for the
full-workload smoke to finish, pass the data/four-device gates and release its
TPU lock. Its memory limit is the actual 33,014,413,312-byte device limit;
admission now depends on the demonstrated full workload, not the earlier
reduced-smoke headroom estimate. The training sequence is otherwise unchanged.

**05:29 UTC:** the full initial Hex9 evaluation completed all six pairings
over 6,480 two-ply openings. An independent connectivity replay verified all
38,880 games, legal pre-terminal actions, rewards, and summary counts.
Raw audit: `artifacts/muzero/night-watch-20260905/hex9-initial-connectivity-audit.json`.
These are untrained-model baseline games, not learning-strength evidence.
The full-capacity replay smoke has completed its first warmup insertion;
optimizer and final-evaluation gates remain pending.

**05:38 UTC budget-semantics correction:** inspecting the production host loop
found that `num_iters` counts self-play steps, whereas the research adapter had
divided it by updates/cycle. Future default cycle counts now divide by
`cycle_n_selfplay`, matching production's fresh-run calculation. This corrects
Hex7/8/9 default optimizer ceilings to 150k/500k/2M. Historical resolved configs
remain intact; explicit pilot budgets (including Hex8's 2,500 and the future
Hex9 trial's 1,250 updates) are unchanged. Tests verify all three defaults and
explicit-budget preservation. My earlier Hex8 answer's 600k AZ-update comparison
was incorrect; the actual Hex8 run count and batch/LR figures were correct.

Collection cadence is also not identical: AZ calls its continuous one-step
self-play function 40/60/40 times per Hex7/8/9 cycle; MuZero collects complete
episodes with horizons 49/64/81 before 30/50/25 updates. Real transitions exclude
absorbing padding and are reported separately. Thus original batch size, LR,
model dimensions and optimizer-cycle count do not imply matched experience or
compute. No collection-algorithm change is mixed into the current trial.

The waiting host-only supervisor b was stopped for this correction; the full
TPU smoke continued uninterrupted. Its status is preserved with an operator
note. Replacement supervisor c uses the corrected frozen source/config set.

**06:02 UTC full-workload gate passed; learning launched:** the smoke completed
at 06:01:53, including four optimizer updates, both 81-game MoHex evaluations,
initial/final full two-ply matches and held-out diagnostics. All four replay
conservation checks passed; no anomalies, nonfinite metrics, truncated games
or checkpoints. Full replay capacity was allocated throughout (warmup content
was deliberately only two batches). Peak reservation was 28,849,815,552 bytes
per device; peak reported live allocation was 9,947,884,032 bytes/device.
These allocator categories are distinct and should not be added together.
Steady collection was 189.98 seconds, two optimizer updates 4.86 seconds.

Supervisor c admitted `hex9-initial` and started
`artifacts/muzero/hex9-kl-night-seed1-20260905-a/`. Live control/logs:
`artifacts/muzero/opening-progression-20260906-c/` (including
`hex9-initial.log`). This fresh 50-cycle/1,250-update trial tests learning
before the repeated-coverage phases. No checkpoint or smoke model was loaded.
Based on the measured collection/update cost plus initialization/evaluation,
this first learning trial is expected to take roughly four to five hours;
this is an estimate, not a completion result.

Hex9 learning W&B: https://wandb.ai/tdoubleu/nanoAlphaZero-muzero/runs/uj2gebjo.
