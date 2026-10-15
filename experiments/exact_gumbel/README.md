# Exact sequential Gumbel MuZero

Goal: run full sequential Gumbel MuZero (`mctx.gumbel_muzero_policy`) fast on TPU,
numerically equivalent to upstream mctx, instead of the fixed two-rung 1sh search.

`exact_gumbel.gumbel_muzero_policy_node` is a drop-in replacement for upstream's
`gumbel_muzero_policy(params, rng_key, root, recurrent_fn, num_simulations,
invalid_actions, max_depth, *, qtransform, max_num_considered_actions,
gumbel_scale)`. It calls upstream's own selection, qtransform and
sequential-halving code. Only the tree storage and update mechanics change.

## Results (TPU, one chip, 2048 roots, chess b10c256, A=4672, m=8)

| sims | network only | 1sh (8+4) | upstream mctx | node variant |
|-----:|-------------:|----------:|--------------:|-------------:|
| 12   | 278 ms | 329 ms | 877 ms (10.8 GiB temp) | 340 ms (2.0 GiB) |
| 24   | 533 ms |        | 3009 ms (21 GiB)       | 709 ms (3.4 GiB) |
| 48   | 1041 ms |       | 10703 ms (37 GiB)      | 1577 ms (5.7 GiB) |
| 96   | 2058 ms |       | not run                | 3892 ms (10.5 GiB) |

The installed fork's `gumbel_muzero_policy` times the same as upstream.

## Equivalence

- CPU: 144/144 configurations are bitwise identical to upstream, covering every
  `PolicyOutput` field and every search-tree array. The configurations vary
  n, m, `rescale_values`, `value_scale`, `max_depth`, seeds, and rows with 0, 1
  and 2 legal moves (`test_equivalence.py`).
- TPU, production chess network: the whole search tree (visits, values,
  children, priors, embeddings) and the chosen action are bitwise identical.
  `action_weights` differ by up to 1.2e-7 (`tpu_equiv.py`). Upstream does
  the same to itself: recompiling only its final qtransform/softmax step on
  its own tree changes `action_weights` by 1.5e-8 (`tpu_ctx.py`). That last
  ulp-level gap depends on XLA's compilation context and cannot be closed
  across different programs.

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

## Remaining headroom

What remains at large n is the `[A, N]` mask work, up to about 1.9x the
network time at n=96. Restricting children to the top-K prior (BNK) would
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
PYTHONPATH=$PWD/ref:$PWD ../../.venv/bin/python bench_search.py --sims 12,24,48 --variants net,1sh,node   # TPU
PYTHONPATH=$PWD/ref:$PWD ../../.venv/bin/python tpu_equiv.py chess                                      # TPU
```
