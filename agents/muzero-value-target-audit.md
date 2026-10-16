# Value targets, masks, and gradient-flow audit

Later follow-up: the [replay-coverage audit](muzero-replay-coverage-audit.md)
identifies a sampler index-correlation defect outside the target/gradient paths
tested here. The original conclusions below describe this earlier audit only.

2026-10-17 Asia/Tokyo. Follow-up to
[the controlled search comparison](muzero-recurrent-value-investigation.md).
No consistency loss, optimizer update, training setting change, interruption,
commit, or push was performed. All new execution was CPU-only.

## Conclusion

No incorrect target sign, sequence alignment, terminal-tail target, replay mask,
or disconnected dynamics gradient was found in the exercised paths. There is
not yet a justified implementation fix from this audit.

Two concrete observations need follow-up:

1. The configured exploration mask leaves most initial-position windows without
   value supervision. This is deliberate `policy-transition` behavior.
2. Even on recorded, legal, nonterminal trajectories after ply 30, the frozen
   model's recurrent values disagree substantially with fresh-board values and
   fit recorded game outcomes worse. Dynamics receives substantial gradients;
   the failure is not an entirely disconnected value-training path.

Neither observation alone proves why ACPL reverses near cycle 1500. Do not add
consistency or change exploration/value masks on the strength of this audit alone.

## Targets and masks checked

Production paths inspected:

- `muzero.py`: `make_muzero_selfplay`, `unroll_muzero_predictions`,
  `muzero_loss`, `make_muzero_train`.
- `buffers.py`: chronological return backfill, sequence gather, masks, terminal
  tails, and replay layout conversion.
- `muzero_selfplay.py`: delayed reset and explicit terminal-root collection.
- `model.py`: spatial normalization, shared prediction head, scalar value,
  gradient scaling, and recurrent application.
- `core.py`: action choice, exploration switch distribution, and episode counters.

`batch.reward` in the value loss contains completed returns, despite its name.
Immediate transition rewards are a separate field. Return backfill uses signed
discounts and terminal discount zero. The final real record is the position
before termination; absorbing targets alternate the outcome thereafter. An
explicit terminal root owns the terminal player's outcome and has no real
outgoing transition. No sign/index discrepancy was observed.

New independent chess test drives the production collector and buffer with
Fool's Mate and Scholar's Mate in separate lanes. Python-chess supplies the
winner independently. Across 24 collection calls, repeated episodes, and an
eight-slot circular buffer, it checks every eligible retained start's six value
targets, recorded action alignment, value/policy/reward masks, and absorbing
boundary. Both winner colors and delayed-reset terminal roots pass.

Existing tests additionally exercise mixed exploration patterns, missing/masked
supervision, lane/game boundaries, packed chess observations, sparse policies,
layout round trips, and rematerialization. These are regression fixtures, not a
dump of the running job's replay buffer. The preserved checkpoint is partial;
its replay contents are unavailable for inspection without changing the run.

### Opening supervision coverage

The saved configuration has `num_exploratory_moves=30`, unroll length 5, and
`muzero_exploration_losses=policy-transition`. Switch S is uniform on 0..29.
A real position at zero-based ply t has a value label only when t >= S.
Before accounting for survival and replay sampling:

| Root ply | Switch choices allowing root value | Choices allowing any value in six-position live window |
| ---: | ---: | ---: |
| 0 | 1/30 (3.3%) | 6/30 (20%) |
| 10 | 11/30 (36.7%) | 16/30 (53.3%) |
| 20 | 21/30 (70%) | 26/30 (86.7%) |
| 29 | 30/30 | 30/30 |

A new test enumerates all 30 choices through the production gather function,
using sparse chess storage. Policy supervision remains enabled throughout.
Terminal tails are an exception: their known outcomes are supervised even when
the preceding real positions were exploratory.

This is a real coverage limitation, not proof that enabling value loss on all
exploratory positions is better. Their game results reflect exploratory actions,
which is precisely why excluding them can be intentional. Measure this tradeoff
before changing the setting. Errors measured after ply 30 also show this cannot
be the sole explanation.

## Gradient flow

The unroll scales the incoming latent's derivative by 0.5 at every recurrent
step. An analytic linear model run through the actual production unroll confirms:

- Root representation derivative from depth k: `0.5**k` in the linear identity
  example; at depth 5 it is 1/32.
- Shared additive dynamics parameter derivative: `2*(1-0.5**k)` in that example;
  at depth 5 it is 1.9375, not 1/32. Each recurrent application receives its own
  local parameter gradient.
- Prediction-head derivatives remain present.
- A masked root does not detach the latent: an unmasked future value loss still
  updates representation and dynamics.
- Rematerialization preserves these derivatives. Altering masked targets does
  not alter the objective.

Thus removing the scaling is not justified by a claim that dynamics gets no
gradient. It does suppress deep value feedback to the representation, which is
a plausible optimization tradeoff to measure separately.

## Frozen-checkpoint probe

Cycle-3200 checkpoint from the prior experiment; 16 deterministically selected
PGN windows, all roots at ply >=30, all six positions nonterminal. Actual recorded
actions and full board histories are retained. Labels are each recorded game's
outcome expressed for the current player at each offset. These are held-out
diagnostic trajectories from cycle-2400 games, not current replay samples or
optimal-play value labels. No fitting or parameter update was done.

| Depth | Recurrent MSE vs outcome | Fresh-board MSE vs outcome | Recurrent/fresh mean absolute difference | Representation gradient norm | Dynamics gradient norm | Prediction gradient norm |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | .716 | .716 | ~0 | 17.805 | 0 | 6.072 |
| 1 | .883 | .494 | .404 | 5.951 | 29.004 | 5.497 |
| 2 | .846 | .647 | .339 | 1.939 | 30.140 | 4.443 |
| 3 | .886 | .501 | .409 | .679 | 29.405 | 5.398 |
| 4 | .823 | .635 | .357 | .270 | 27.462 | 5.631 |
| 5 | .681 | .496 | .422 | .097 | 24.269 | 4.243 |

Each gradient is for that depth's separately averaged value MSE on 16 positions,
before clipping. These norms are not comparable directly with the production
8192-sample, combined policy/value gradient norm. They establish finite, nonzero
paths, not the effectiveness of Adam updates or the absence of gradient conflict.
The sample is small; do not treat its MSEs as population estimates.

The mismatch is already large after one transition. Merely increasing unroll
length is therefore not an evidenced fix. The probe also shows this is not
exclusively an illegal-action or off-trajectory search phenomenon.

## Remaining investigation before proposing a repair

1. Expand per-depth measurements to a fresh, larger held-out sample, stratifying
   opening/middlegame/endgame, real outcomes, and played vs searched actions.
2. Measure interaction of policy and value gradients into h/g/f and actual
   optimizer updates using a bounded offline dataset. Distinguish insufficient
   fitting from interference and out-of-distribution behavior. No claim about
   those causes follows just from nonzero gradient norms.
3. Compare fitting on a fixed training subset with held-out error. If the
   recurrent path cannot fit even its supervised data, investigate objective,
   normalization, and optimization first. If it fits but generalizes poorly,
   investigate coverage and representation before adding another loss.
4. Preserve future model snapshots and repeat the same paired search comparison
   to establish when the recurrent-value gap emerges. This audit has no retained
   pre-regression model and does not establish the cycle-1500 timing.

## Reproduction and validation

From `/nfs_share/nz/.worktrees/muzero-multihost`:

```sh
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \
  taskset -c 0-15 uv run --no-sync pytest -q \
  tests/test_muzero_exploration_losses.py tests/test_muzero_terminal_training.py \
  tests/test_muzero_remat.py tests/test_muzero_replay_layout.py

JAX_PLATFORMS=cpu taskset -c 16-23 uv run --no-sync pytest -q \
  tests/test_muzero_value_gradient_audit.py tests/test_muzero_value_target_audit.py

JAX_PLATFORMS=cpu taskset -c 24-39 uv run --no-sync python -u \
  tests/helpers/audit_chess_value_gradients.py
```

Results: **331 existing tests + 4 new tests passed**. Logs and raw probe values:
`artifacts/value-target-audit-20261017/`, particularly
`checkpoint-gradient-audit.json`, `tests.log`, `gradient-tests.log`, and
`real-games-tests.log`. The helper uses the frozen checkpoint and prior PGN
artifact paths; these are ignored, machine-local assets. Rerunning it overwrites
its audit JSON, so copy the output or change the output directory when comparing
new experiments.
