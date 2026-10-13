# MuZero terminal-state investigation handoff

## Subsequent implementation: training diagnostics

Added `src/nanoalphazero/diagnostics/chess/cases.py` (13 fixed terminal cases,
including embedded Qh1 move history; no local PGN dependency) and
`chess/terminals.py` (`make_chess_terminal_probe`). The diagnostics dispatcher
merges metrics/tables for `algorithm == "muzero"`; no changes to the host loop.
Existing diagnostic cadence and tracker enablement control invocation.
The new code is not injected into an already-running training process.

Per-case scalar series under `chess/terminal/<case>/` cover root, observed
terminal, recurrent terminal, two greedy continuations, rewards, manual Q
backups/errors, fixed-uniform-32 versus top-policy-8 branch errors, entropy,
and deterministic search visits/selection/Q. Unvisited Q is omitted and null
in the table. See `docs/muzero-components.md` for definitions.

Validation: 19 focused CPU tests passed (terminal diagnostics plus existing
chess diagnostics). Actual cycle-6000 model smoke matched the offline terminal
values, one/two-step Q, and action-selection results for all 13 cases; repeat
calls produced identical predictions/tables. CPU first probe 11.15 seconds,
repeat 3.36 seconds, excluding suite construction/model loading. These are NOT
TPU performance measurements. Output is
`artifacts/chess-value-only-terminal-cycle6000-20261013-comparison/training-diagnostics-smoke.json`.

Cycle-6000 comparison found improved Fool's/Scholar's Mate but worse Qh1
continuation: terminal V -0.915, greedy next V -0.647, search Q +0.134; the mate
was no longer chosen (both Gumbel settings). Cycle 4400 had Q +0.376 and chose
it. Uniform versus high-policy action errors still differ substantially.

Updated 2026-10-13. This records observations and unresolved questions, not a
request to change training or restart the current run.

## Workspace and safety

- Branch at handoff: `muzero-chess-action-encoding`, HEAD `1f51ff7`
  (`Add MuZero value-only board-game training mode`). Substantial subsequent
  implementation, tests, and probe scripts are dirty/untracked. Do not assume
  HEAD alone reproduces this investigation. Preserve unrelated demo files.
- Root `muzero.py` is the MuZero training assembly/host loop on this branch;
  shared components live in `src/nanoalphazero/`. Read
  `docs/muzero-components.md` as well as `AGENTS.md`.
- The user wants original AlphaZero behavior preserved. Keep main MuZero code
  readable; put component details in components. Do not commit/push unless asked.
- Chess training was still running when last checked, PID 3504549 (uv parent
  3504545). Recheck ownership; PIDs and progress are not durable facts. Do not
  stop it or initialize another TPU JAX process. All recent probes ran on CPU.

## Current run and stable model used for comparisons

Trackio run: `valiant-petal-41`.
Run directory: `artifacts/chess-value-only-20261012-103435/`.
Live full checkpoint: `latest.ckpt` within that directory; replaced every 400
cycles. ACPL runs every 400 cycles; ladder disabled.

Important settings: chess, TPU, spatial width 256/depth 8, action encoding
`from_to8`, batch sizes and self-play consume size 4096, self-play capacity 512,
replay capacity 1024, `buffer_layout=features`, value-only enabled,
`exploration_losses=policy-transition`, self-consistency disabled. Warmup and
LR warmup 1532 steps; num_iters 9180000. Load checkpoint configuration instead
of assuming current defaults when probing.

Stable snapshot used for every result below:

`artifacts/chess-value-only-terminal-cycle4400-20261013/model.ckpt`

This is a small model-only copy from the cycle-4400 full checkpoint, not a full
resumable training state. `source.json` records provenance. Do not substitute a
newer live checkpoint when comparing these numbers. Local artifact paths may
not exist on another machine; they are not committed assets.

## Actual training and search semantics

- `muzero.py:unroll_muzero_predictions` encodes only the sampled real root:
  `s0 = h(o_t)`. Each subsequent latent is `s[k+1] = g(s[k], action[k])` using
  the current parameters. Replay does not supply cached latents.
- K=5 means six policy/value predictions and five transitions. Future real
  observations do not replace recurrent latents. Optional consistency uses
  them as separate targets; it is off in this chess run.
- Gradients flow through the recurrence. `scale_gradient(latent, 0.5)` changes
  backward gradients only, not the forward latent. Latents are not zeroed or
  reset when a training window crosses termination.
- `buffers.py:gather_muzero_training_samples` uses recorded actions for real
  transitions and uniform random action IDs for padding. Padding is generated
  when assembling the replay sample, not freshly for each optimizer update.
  The random key currently derives from the maximum start global_step_id.
- Actual terminal boards are not sampled as training roots. The final real
  record is the source position BEFORE the terminating action. Following
  positions in its unroll receive synthetic padding targets, not new-game data.
- Policy and consistency supervision are masked at terminal/padding positions.
  Value supervision continues there for eligible samples.
- `--muzero.predict-value-only` is now the default. There is no instantiated
  reward head in this mode; recurrent reward output is zero for API compatibility.
  Search has player-alternating discount -1. No muzero_discount option remains.
- Crucially, model reward prediction being disabled does NOT erase environment
  rewards. Real terminal rewards still backfill completed returns.
- In reward mode a mate-in-one window has values `[1,0,0,0,0,0]` and immediate
  rewards `[1,0,0,0,0]`. In value-only mode values are `[1,-1,1,-1,1,-1]`.
  Immediate reward data still exists but reward loss is disabled.

The two backup formulations are consistent when predictions are correct:

```text
Reward mode:     mating edge r=+1, terminal V=0, later r=V=0.
                Every backup across the mating edge adds +1 again.
Value-only:     r=0, terminal V=-1, next V=+1, next V=-1.
                Perspective flips preserve the same winning outcome.
```

An absorbing outcome does not require identical latent vectors. Our losses
constrain predictions, not `g(s_terminal,a) == s_terminal`. The failure under
investigation is incorrect outcome predictions under some continuation actions.

### Exploration modes

`--muzero.exploration-losses none|policy-transition|all` replaces the two old
experimental booleans. Default is `policy-transition`.

- `none`: real exploration positions have all direct losses masked; an eligible
  window needs a real exploit position within its K+1 states.
- `policy-transition`: exploration policy, reward (if enabled), and consistency
  (if enabled) can train; exploration value is masked. All-explore windows can
  be eligible. In value-only mode this means policy and optional consistency.
- `all`: all enabled losses can train on real exploration positions.
- Padding is separate: no policy/consistency, but value targets remain trained
  in eligible windows. Padding alone does not qualify a window in `none` mode.

## What was actually tested

### Real self-play buffer targets: correct in tested cases

`scripts/audit_muzero_mating_episode_targets.py` runs complete Fool's Mate and
Scholar's Mate episodes through production self-play wrapping, buffer insertion,
return backfill, and sample gathering. Scripted one-hot actions replace MCTS.
Both all-exploit and all-explore episodes were tested, including continuation
into new games and circular storage.

Results: `complete-episode-targets.json` under the snapshot directory.

- Fool's Mate real rewards `[0,0,0,1]`, discounts `[-1,-1,-1,0]`, completed
  returns `[-1,+1,-1,+1]`.
- Scholar's Mate completed returns `[+1,-1,+1,-1,+1,-1,+1]`.
- The winning source gets `1 + 0 * carry = 1`, not zero.
- Last-root padding values alternate correctly; no new-game target leakage was
  found. All script assertions passed.

### Terminal prediction/search probes

`scripts/probe_muzero_chess_terminals.py`: report at `report.json` under the
snapshot directory. Includes mate, stalemate, repetition, fifty-move,
insufficient-material and step-cap cases. CPU assertions and manual search
backup checks passed. Sparse FEN cases lack full game history; Fool's Mate,
Scholar's Mate, and the historical Qh1 case were replayed with history.

`scripts/probe_muzero_terminal_action_distribution.py`: outputs
`action-distribution/report.json` and per-case `.npz` arrays. Each tested action
branches independently from the SAME terminal latent. Seed 0, 512 uniformly
sampled action IDs without replacement plus top 64 policy IDs and pseudo-legal
IDs, eight cases. All reward predictions were zero and outputs finite.

Absolute error from the correct next value (+1 for these winning mates):

| Case | Uniform-512 mean error | Top-8 mean error | Greedy next value |
|---|---:|---:|---:|
| Fool's Mate | 0.0355 | 0.4022 | +0.5332 |
| Scholar's Mate | 0.0642 | 0.6192 | +0.2176 |
| Historical Qh1# | 0.0136 | 0.2125 | -0.1727 |

Many uniform IDs refer to empty/opponent source squares. In these cases they
were easier than own-piece actions. This supports a padding/search action
distribution mismatch hypothesis, but does not prove its cause. Not universal:
sparse queen-mate was bad for both sets; repetition was better for top actions.

### Policy/value chains

`scripts/probe_muzero_terminal_chains.py` outputs `state-chains/report.json`,
`state-chains/logits.npz`, with a sibling `state-chains.log`.

Starting from the real pre-move encoding, apply the known mate, then greedy
unmasked policy actions. Values are in alternating player perspectives:

| Case | Pre-move | Terminal latent | One past terminal | Two past terminal |
|---|---:|---:|---:|---:|
| Fool's Mate | +0.4577 | -0.7955 | +0.5332 | -0.5923 |
| Scholar's Mate | +0.3799 | -0.6649 | +0.2176 | **+0.5448** |
| Qh1# | +0.5597 | -0.9245 | **-0.1727** | **+0.7145** |

Expected signs are +,-,+,-. Greedy paths after mate:

- Fool's: e1d1, h7h5.
- Scholar's: d7d5, e4d5.
- Qh1: c6h1, c7h2. The bishop's apparent queen capture c6h1 is blocked.

These are action-coordinate decodings, NOT legal continuations or simulator
rollouts. Some arbitrary IDs decode to impossible moves. Policy probabilities
are unmasked softmax; policy is not directly supervised after termination.
Chess-like logits do not prove that the latent represents a particular board.

The script ALSO directly encodes the real terminal board with h, using ordinary
`env.step` (not autostep). Terminal values were -0.4551, -0.5262, -0.3389 for
Fool's, Scholar's, Qh1 respectively. These observations are not sampled training
roots, so this is an additional out-of-training-input diagnostic, not the same
path as the recurrent terminal latent. Do not confuse a real terminal board,
a reset starting board, and zero-filled padding observations.

Reproduce with a NEW output directory (scripts refuse to overwrite):

```bash
JAX_PLATFORMS=cpu uv run python -u scripts/probe_muzero_terminal_chains.py \
  --checkpoint artifacts/chess-value-only-terminal-cycle4400-20261013/model.ckpt \
  --output artifacts/terminal-chains-NEW
```

## External implementation findings (checked online 2026-10-13)

### DeepMind MuZero

- [Author-linked pseudocode](https://gist.github.com/Mononofu/6c2d27ea1b3a9b3c1a293ebabed062ed)
  has a placeholder `sample_position` returning -1. It does not establish
  whether terminal observations were ever sampled as roots in production.
- `make_target` supplies absorbing-state labels at/beyond the end. Its action
  history slice does not show how padding actions are generated. Do NOT claim
  that this pseudocode proves DeepMind used uniform padding actions.
- [Author clarification](https://www.julian.ac/blog/2020/12/22/muzero-intuition/#4271a1239cc6ba61ec2af4974dc781ebd95332f22cd54c41c340bc6505105a39):
  post-terminal policy loss is masked; reward-predicting mode has zero future
  value/reward labels; value-only board-game mode retains outcome labels.
  The comment does not explicitly specify our alternating-sign convention.
  It can be retrieved from the site's `comments.jsonl.gz` and located by that
  comment ID if the browser does not display comments.
- [Paper](https://arxiv.org/html/1911.08265v2), Figure 1 and Appendix G: encode
  root once, unroll recorded actions, joint backpropagation through time. No
  requirement for latent identity with real environment state.

### MuZero-General (master at inspection)

- [Replay buffer](https://github.com/werner-duvaud/muzero-general/blob/master/replay_buffer.py):
  sampler chooses within `len(root_values)`, excluding final observation as
  a root. Terminal/padding values are zero. Final transition keeps its actual
  reward; later rewards zero. Later actions sampled uniformly.
- It gives terminal/padding positions UNIFORM policy labels, not masked labels.
  [Trainer](https://github.com/werner-duvaud/muzero-general/blob/master/trainer.py)
  applies policy loss to them. This differs from DeepMind's clarification.
- [Self-play](https://github.com/werner-duvaud/muzero-general/blob/master/self_play.py)
  stores the final observation but no extra terminal-root search statistics.

### Original EfficientZero (YeWR/EfficientZero main, not EfficientZero V2)

- [Replay buffer](https://github.com/YeWR/EfficientZero/blob/main/core/replay_buffer.py)
  indexes `range(len(game))`; [game history](https://github.com/YeWR/EfficientZero/blob/main/core/game.py)
  defines length as action count. Terminal observation is not sampled as root.
- [Batch builder](https://github.com/YeWR/EfficientZero/blob/main/core/reanalyze_worker.py)
  marks actual actions valid and random padding actions invalid. Terminal value
  target is zero and terminal policy label is all zeros.
- [Training](https://github.com/YeWR/EfficientZero/blob/main/core/train.py)
  applies the action mask to recurrent value/policy/value-prefix/consistency
  losses. The LAST REAL action remains valid: its terminal successor value is
  trained. Later padding transitions have ALL these losses masked.
- With consistency enabled, the actual final observation is encoded as a
  consistency target for the last real transition. Thus terminal observations
  can enter training even though they are not sampled roots.
- This is an Atari reward/value-prefix implementation, not evidence for
  value-only chess. Do not port its masking as though the formulations matched.

## Limits, hypotheses, and next work

- Do not say "we do not train terminal states" without specifying root
  observations vs recurrent latents. We DO train terminal/padding values.
- Tests support correct target generation for audited episodes, not correctness
  of every conceivable environment/storage condition.
- Greedy continuation failures are real in this snapshot. Uniform padding
  training may under-cover actions favored by the policy. Mixing uniform and
  policy-guided padding is a proposed experiment, NOT implemented or validated.
- No basis yet for claiming terminal-root training is required by DeepMind or
  that adding it fixes this issue. Also no evidence that resetting/zeroing
  latents caused the problem: our training does not do that.
- Existing CPU exploration-mask tests previously passed (303 tests); related
  regression groups passed as recorded in the session. These are historical
  results, not a fresh full-suite validation of all current dirty changes.
- Keep probes separate from algorithm changes; use fixed checkpoints for
  comparisons and preserve original PGNs/results. No training changes were
  made during the latest policy/value chain and online-source investigation.
