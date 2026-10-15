# Exact sequential Gumbel MuZero

Goal: run full sequential Gumbel MuZero (`mctx.gumbel_muzero_policy`) fast on TPU,
numerically equivalent to upstream mctx, instead of the fixed two-rung 1sh search.

**Use `exact_gumbel.gumbel_muzero_policy_node2`.** It is a drop-in replacement
for upstream's `gumbel_muzero_policy(params, rng_key, root, recurrent_fn,
num_simulations, invalid_actions, max_depth, *, qtransform,
max_num_considered_actions, gumbel_scale)`; pass `return_tree=False` in
production. It calls upstream's own selection, qtransform and
sequential-halving code; only the tree storage and update mechanics change.
`gumbel_muzero_policy_node` is the previous version, kept for comparison.

## Results (TPU, one chip, 2048 roots, chess b10c256, A=4672, m=8), ms

| sims | network only | 1sh (8+4) | upstream mctx | node | **node2** | fork `_opt` K=32 |
|-----:|-------------:|----------:|--------------:|-----:|----------:|-----------------:|
| 12   | 277  | 328 | 877   | 339  | **303**  | 391  |
| 24   | 531  |     | 3009  | 707  | **592**  | 670  |
| 48   | 1040 |     | 10703 | 1575 | **1191** | 1348 |
| 96   | 2057 |     | not run | 3890 | **2460** | 2846 |

node2 temp memory: 2.0 GiB at 12 sims, 5.7 GiB at 48, 10.6 GiB at 96
(upstream: 10.8 GiB at 12). The fork's `_opt` restricts nodes to top-K
children and uses a modified qtransform, so it is not equivalent to upstream.

At 512 roots per chip: network 67 ms, 1sh 81, node2 77 at 12 sims; network
168 ms, node2 202 at 32 sims. The network is saturated from about 512 roots per chip (about 11 us
per row) and its outputs are bitwise batch-size invariant on TPU.

## Using it in maz: `maz_integration.make_exact_muzero_mcts`

A drop-in for `nanoalphazero.mcts.make_muzero_mcts(config, env, model,
data_sharding)`. It returns maz's `PolicyOutput`, with root `visit_counts`
for the exploration sampler and the BNK fields when
`exp_bnk_action_weights` is set. Root temperature is applied as maz applies
it. It uses `config["mcts_num_simulations"]` and
`config["mcts_num_root_considered"]` (`mcts_num_survivors` is unused), and
uses the shard_map wrapper whenever `data_sharding` spans several devices.

Measured as maz's self-play search call (`bench_maz.py`: chess, 8192 real
positions, 4 chips, including observation, legal-move unpacking, root
temperature and BNK compression):

| search | ms per call |
|---|---:|
| maz 1sh (8+4) | 414 |
| exact node2, 12 sims | **392** |
| exact node2, 32 sims | 878 |

`test_maz_integration.py` (CPU, hex5 and connect4, 1 or 4 devices) checks
that actions and visit counts equal upstream on the same roots. Without
`--xla_force_host_platform_device_count` it runs on one device. Its
`action_weights` differ by about 2e-7: computing the root network inside the
same program changes XLA's fusion of the final softmax, the same
compile-context effect described below.

## Multi-chip: use the shard_map wrapper

Under plain `jit` with a batch-sharded input (maz's data parallelism), GSPMD
puts all-gathers and all-reduces inside the search loop. On 4 chips with
8192 roots and 12 sims, node2 took 430 ms there, against 303 ms on one chip
and 328 ms for 1sh. `gumbel_muzero_policy_node2_sharded(mesh, axis, ...)` runs
each device's rows with `jax.shard_map`. Searches are independent per row, so
this needs no collectives: 304 ms at 12 sims and 792 ms at 32 (GSPMD: 1132),
with half the memory. It also returns the root visit counts that maz's
exploration sampler needs, so the tree is not required
(`gumbel_muzero_policy_node2_with_visits` does the same unsharded).

Masking and the Gumbel draw happen on the global batch, as upstream does. On
TPU, though, upstream's own output depends on the batch size it runs with:
rows 0-255 searched alone differ from the same rows inside a batch of 1024
(`batch_dependence.py`). The root network (`model.initial`) also changes
slightly when its batch is sharded, while the recurrent network does not.
"Equal to upstream" under sharding therefore means equal to upstream run on
each device's local batch. `tpu_equiv_sharded.py` checks exactly that: bitwise
identical action, `action_weights` and root visits on 4 chips (3
configurations, including `rescale_values=True`). On CPU with 4 virtual
devices, it also equals upstream on the global batch (`test_sharded.py`).

## Equivalence

- CPU: 144/144 configurations are bitwise identical to upstream, covering every
  `PolicyOutput` field and every search-tree array. The configurations vary
  n, m, `rescale_values`, `value_scale`, `max_depth`, seeds, and rows with 0, 1
  and 2 legal moves (`test_equivalence.py`).
- TPU, production chess network (`tpu_equiv.py chess gumbel_muzero_policy_node2`):
  node2 is bitwise identical to upstream in the tree, the action and
  `action_weights` (max difference 0.0) across 5 configurations, including
  `rescale_values=True` and rows with few legal moves.
- The older `node` variant matched the tree and action but differed in
  `action_weights` by up to 1.2e-7, because its final qtransform fused with
  the root-prior masking. node2 routes the root prior through an
  optimization barrier, which removes that. Upstream's own `action_weights`
  shift by 1.5e-8 if only its final step is compiled separately
  (`tpu_ctx.py`), so such differences come from XLA fusion, not the search.

## Why upstream is slow on TPU

Network calls are not the problem: 12 sequential calls at batch 2048 cost
278 ms, against 329 ms for 1sh's two wide calls. Upstream's extra time is tree
bookkeeping that grows with n²:

1. Per-row scatters into `[B, N, A]` edge arrays. XLA's TPU backend lowers each
   one by flattening to 1-D, which needs full relayout copies of the array.
   The compiled loop body had about 12 full `[B, N, A]` copies per simulation.
   Storing the arrays as `[B, N*A]` does not help (variant 4).
2. The vmapped backward `while_loop` carries the whole tree, so every iteration
   selects over all of it.

## What the node variant changes (all exact)

- Every edge statistic equals a statistic of the child node: visits, value,
  and the reward/discount into it. The child of (p, a) is the node with
  `parent == p` and `action_from_parent == a`. So only `[B, N]` node arrays,
  `[B, N, A]` priors and embeddings are stored.
- With `max_depth >= num_simulations` (including the default), simulation `i`
  always creates node `i + 1` in every row. All writes therefore become one
  `dynamic_update_slice` at a scalar node index, with no scatters.
- Edge rows are rebuilt with a dense `[A, N]` equality mask that has at most
  one match. These sums are exact, and they avoid TPU gathers, which were slow
  (variant 1).
- The backward pass carries only `[N]` arrays.
- `max_depth < num_simulations` falls back to variant 2 (upstream layout with
  the small-carry backward), which is also bitwise exact.

## Dead ends (kept in `exact_gumbel.py` for reference)

- Variant 1, compact tree with gathers: exact, but 6.6x slower than upstream
  on TPU.
- Round batching (`round_batched=True`), which evaluates each
  sequential-halving round in one call: exact decisions with
  `rescale_values=False`. Rows with 1–2 legal moves still need about n or n/2
  sequential rounds, and padding doubles the evaluations, so it is 25x slower
  than upstream at batch 2048. The wider batch also changes matmul rounding.
  It is only worth considering when the per-chip batch is too small to fill
  the TPU (eval/play).
- Variants 2 and 3 (small-carry backward, slice expand): exact, but only 12–55%
  faster than upstream.

## What node2 adds (all exact)

Found with `profile_ops.py`, which prints a per-op TPU time breakdown:

- Child-stat rows are built for only two per-node arrays: visits, and Q
  computed per node (the same elementwise op on the same inputs). They use a
  one-hot matmul at HIGHEST precision, which reproduces each value exactly.
- Root selection runs once per simulation outside the descent loop. Under
  `vmap`, upstream's in-loop `cond` evaluates root selection at every depth.
  The root prior row is passed in directly; reading it from the tree there
  made XLA relayout the whole prior array every simulation.
- Prior logits are stored as `[B, N, 8, A/8]`. In `[B, N, A]` the TPU tiles
  (node, action) by (8, 128), so every node write rewrote 8 nodes' tiles
  (1.1 ms per simulation at n=48).
- The root's edge rows are maintained incrementally, since each simulation
  changes exactly one root child.

Tree-only overhead with a trivial network (`tree_bench.py`): 27.7 ms at 12
sims and 218 ms at 48. The figures were 62.6 / 724 ms for node and
587 / 11327 ms for upstream.

## Remaining headroom

node2 is within 10-20% of the network-only time. Most of what remains is the
in-loop child-stat rows (`[A, N]` work per tree level) and the embedding
reads and writes. Restricting children to the top-K prior (BNK) would
remove most of it. That is not bitwise: the interior softmax would normalise
over K actions instead of A. With K >= num_simulations plus a stored full
log-sum-exp per node, the decisions are identical in exact arithmetic, but
results can differ at the ulp level.

## BNK targets

`compress_action_weights(action_weights, K)` returns `(k_indices, k_weights,
retained_mass)` with the same arithmetic as maz's `compress_muzero_policy`
(bitwise identical, `test_bnk.py`). It runs after the search, so the search
stays exact; only the stored training target is truncated to the top K.

## Running

```bash
cd experiments/exact_gumbel && ./make_ref.sh            # vendors upstream mctx as mctx_ref
PYTHONPATH=$PWD/ref:$PWD JAX_PLATFORMS=cpu ../../.venv/bin/python test_equivalence.py
PYTHONPATH=$PWD/ref:$PWD:../../src JAX_PLATFORMS=cpu ../../.venv/bin/python test_bnk.py
PYTHONPATH=$PWD/ref:$PWD JAX_PLATFORMS=cpu ../../.venv/bin/python quick_equiv.py '{"fn":"gumbel_muzero_policy_node2"}' full
PYTHONPATH=$PWD/ref:$PWD JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 ../../.venv/bin/python test_sharded.py
PYTHONPATH=$PWD/ref:$PWD:../../src JAX_PLATFORMS=cpu ../../.venv/bin/python test_maz_integration.py hex5
PYTHONPATH=$PWD/ref:$PWD:../../src ../../.venv/bin/python bench_maz.py 12,32                                   # TPU, all chips
PYTHONPATH=$PWD/ref:$PWD ../../.venv/bin/python bench_sharded.py 8192 12,32                                    # TPU, all chips
PYTHONPATH=$PWD/ref:$PWD ../../.venv/bin/python tpu_equiv_sharded.py                                           # TPU, all chips
PYTHONPATH=$PWD/ref:$PWD ../../.venv/bin/python bench_search.py --sims 12,24,48 --variants net,1sh,node2   # TPU
PYTHONPATH=$PWD/ref:$PWD ../../.venv/bin/python tpu_equiv.py chess gumbel_muzero_policy_node2           # TPU
```
