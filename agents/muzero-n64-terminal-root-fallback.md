# Chess fallback after latest-checkpoint sweep

User authorized an overnight restart after the cycle-3600 sweep: use a validated
improvement if found, otherwise disable terminal-position training and use
64 simulations / 32 root candidates. No commit or push authorized.

The held-out scale sweep did not validate a positive-scale improvement over
both current search and prior-only selection. See
`muzero-latest-scale-and-readout-probes.md` for paired uncertainty and readout
tests. We did not install fitted readouts or change the value scale.

## Configuration and continuation

New artifacts: `artifacts/chess-pod-m32-n64-no-terminal-root-20261016T170250Z`.
Exact launch command is in its `launch.sh`; resolved preflight configuration is
in `resolved-config-preflight.json`.
Launcher logs: `logs/pod/20261016T170409-8cca015b23c8`.
Detached tmux session: `chess-pod-n64`.

Changes from the stopped run:

- Exact Gumbel simulations 32 → 64; considered root actions 16 → 32.
- `muzero_train_on_terminal_states=True` → `False`: no explicit terminal-board
  roots. Absorbing-tail value targets and `muzero_terminal_policy=uniform`
  remain unchanged. This is not disabling every loss on padded positions.
- New checkpoint output, ACPL directory and Trackio identity.

Retained: 384×10 model, value-only/from_to8, policy-transition exploration,
global self-play/train/consume batch 8192, self-play capacity 512, replay time
capacity 1024 (8,388,608 windows), BNK128 targets, learning rate 1e-4,
no consistency loss, partial checkpoint and 496-game ACPL every 400 cycles.
Max search depth remains untruncated and full-tree output remains disabled.

Source: `artifacts/chess-pod-exact-m16-n32-local-gather-20261015/latest.ckpt`,
generation `gen-d6806e5f0ce64ad0bf2710a9f266c2df`, cycle 3600.
The new explicit `--ckpt.fork` retains parameters, optimizer, RNG and cycle,
rebuilds games/replay, and starts a new tracker. It allows only the three changed
fields above beyond existing resume exceptions. Full-state forks and writing
directly over the source are rejected. Ordinary resume stays strict.
Future snapshots record source path/config/cycle in metadata.

Also fixed preflight compatibility treating unresolved `game_obs_shape=None`
and `game_num_actions=None` as already-resolved dimensions. Actual dimensions
are still checked after assembly and before restore. No checkpoint content or
original tracking history was rewritten.

The partial restore warms replay for 76 cycles (1532 // 20), without optimizer
updates. First training cycle is 3601, with a checkpoint saved there before
evaluation. The pre-existing resume-budget issue (optimizer steps subtracted
from a self-play-step budget) is unchanged; this is not an exact budget fix.

## Checks and launch evidence

`uv run --no-sync pod status` verified all four workers free before launch:
10.130.0.2, .4, .5, .3. Pod launcher repeats ownership/NFS/code/dependency checks.
Topology is four hosts / 16 TPU v4 devices, 512 positions per device.
Disk free before launch: 7.4 GiB; original partial checkpoint: 358 MiB.
Full buffers do not fit disk, so partial saving is retained.

CPU checks (not TPU performance tests):

```bash
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \
  uv run --no-sync pytest -q tests/test_muzero_checkpoint_fork.py \
  tests/test_muzero_checkpoint_cli.py tests/test_muzero_review_regressions.py \
  tests/test_muzero_cli.py
```

38 tests passed before adding the unresolved-dimensions regression. The changed
fork/CLI modules were rerun afterward: 13 passed. A separate actual source-config
compatibility check passed. All workers restored and entered compilation.
Further runtime verification is recorded below after completion.

To watch without controlling the job:

```bash
tail -f artifacts/chess-pod-m32-n64-no-terminal-root-20261016T170250Z/launcher.log
```

Ctrl+C exits tail only. To control/stop the actual launch, attach with
`tmux attach -t chess-pod-n64`, then Ctrl+C. Detach without stopping with
Ctrl+B then D. No automatic restart is configured.
