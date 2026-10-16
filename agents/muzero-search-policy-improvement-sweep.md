# Frozen cycle-3600 search policy-improvement sweep

User requested a CPU-only sweep while the TPU fallback run continues: find
whether search can beat policy-only move selection by changing root candidates,
simulations and Q scale; if not, try the square-root visit scaling in `2026mu`.
No production settings are changed by this diagnostic. No commit/push.

## Preregistered design

Artifacts: `artifacts/search-improvement-20261017`.
Runner: `tests/helpers/sweep_search_improvement.py`.
Frozen source: `artifacts/latest-value-scale-20261017/checkpoint`, cycle 3600,
generation `gen-d6806e5f0ce64ad0bf2710a9f266c2df`.
Never loads the concurrently advancing TPU model.

Grid (130 combinations):

- Root candidates: 2, 4, 8, 16, 32, 64.
- Simulations: 4, 8, 16, 32, 64, 128, requiring simulations >= candidates.
- Q scale: 0.01, 0.03, 0.1, 0.3, 1.0.
- First pass uses linear visit scaling; conditional second pass uses exponent 0.5.
- Both use mixed-value completion, max-visit aggregation, offset 50, no value
  min/max normalization, no Gumbel noise, no depth truncation, no full-tree output.
  Saved model's root temperature is retained. Target BNK is outside search and
  does not affect selected moves.

Use 32 screening positions from distinct games; take the best four by mean
paired CP gain versus prior argmax to 64 disjoint validation games. Old search
(m16/n32/scale1) and current TPU search (m32/n64/scale1) are controls. If the best
validation result's paired 95% bootstrap interval includes zero or is negative,
repeat screening/validation using square-root scaling. Select ONE configuration
by validation mean gain, save it before testing, then compare that fixed choice
and both controls on 96 previously unused games. All 96 exclude every game in
the previous 288-position scale/readout dataset. One position per game; complete
move histories retained. Seed 2026101719. Exact positions/settings are persisted
before search starts.

This is a bounded coarse-to-fine search, not proof of a global optimum. Screening
and validation results are exploratory and subject to selection bias. The fresh
test estimates the chosen setting's improvement; it does not choose settings.

Stockfish scores **all legal root moves** at depth 18, Threads=1, Hash=64 MiB,
without the previous time cap. Every reported reference must reach depth 18.
Scores use the root player's perspective and are clipped to [-1000,1000] cp.
Primary paired gain is `SF(search move) - SF(prior argmax move)`: positive means
search improves the greedy policy's choice. Also report mean move loss against
Stockfish's best legal move, changed moves, improvements and regressions.
These are fixed-position choices, not full-game ACPL or proof that stochastic
training targets improve. Gumbel's distributional policy-improvement claim is
not identical to beating greedy prior choice on every finite position set.

## Branch and implementation verification

`git ls-remote https://github.com/wtedw/mctx.git` resolves `2026mu` to
`6cf1a3918fca41569d43b773fefdc5fba70af506`, also pinned in this checkout's lockfile.
Downloaded that commit's `mctx/_src/qtransforms.py` and verified byte equality
with the installed source using `cmp`. It implements
`(maxvisit_init + max(visits)) ** visit_exponent * value_scale * completed_q`.
Exponent 0.5 is sqrt scaling, not multiplication by sqrt(2).

The diagnostic passes a separate transform into the exact-search function;
production search files are unchanged. A four-position CPU smoke checked:

- baseline action equality and target agreement with production search;
- exact transform output equality with `2026mu` at exponents 1 and 0.5;
- backend is CPU (`TFRT_CPU_0`).

Each worker repeats these checks before its sweep and writes a check artifact.
The pipeline uses `uv run --no-sync pod run --cpu` with CPU affinity 24–47 per
host. It never initializes distributed JAX/libtpu and does not stop TPU jobs.
It runs detached in tmux `search-sweep-cpu`; the TPU launch remains independently
supervised in `chess-pod-n64`.

## Commands

```bash
JAX_PLATFORMS=cpu uv run --no-sync python tests/helpers/sweep_search_improvement.py \
  artifacts/search-improvement-20261017 prepare
taskset -c 0-15 uv run --no-sync python -u tests/helpers/sweep_search_improvement.py \
  artifacts/search-improvement-20261017 orchestrate
```

The orchestrator invokes the CPU pod workers, scores references, selects
finalists and runs confirmation automatically. Progress: `orchestrator.log`
and `<stage>-launcher.log`. Results are written per configuration, then
`<stage>-summary.json`. `chosen-before-test.json` freezes the winner;
`COMPLETE.json` is written only after the final test finishes.

Status at initial note: CPU smoke passed; linear screening running. Results
will be appended after scoring and confirmation. Do not interpret a started
pipeline or an incomplete result directory as a finished experiment.
