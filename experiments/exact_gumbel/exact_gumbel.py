"""Gumbel MuZero intended to be numerically identical to mctx.gumbel_muzero_policy.

Two changes, neither of which alters any decision or value:

1. Compact tree. Upstream keeps six dense [B, N, A] edge arrays. Visits, values,
   rewards and discounts of an edge are always equal to the child node's own
   statistics, so they are stored once per node and gathered on demand. Only
   `prior_logits` and `children_index` stay [B, N, A]. The backward pass then
   carries only [B, N] arrays.

2. Optional round batching. Sequential Halving visits each surviving root
   action exactly once per round, and different root actions own disjoint
   subtrees. All leaves of a round are therefore evaluated in one batched
   recurrent call; the tree is then updated in upstream's exact sequential
   order, re-deriving every root choice with upstream's own selection function.
   A row whose replayed choice is not among the evaluated leaves is reported
   in `mismatch` (only possible on exact score ties or with rescale_values).

Upstream selection and qtransform functions are called unchanged through a
view object, so the arithmetic is shared rather than re-implemented.
"""
import functools
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from mctx_ref._src import action_selection as AS
from mctx_ref._src import base
from mctx_ref._src import qtransforms as QT
from mctx_ref._src import seq_halving as SH
from mctx_ref._src.policies import _mask_invalid_actions

UNVISITED = -1
NO_PARENT = -1
ROOT = 0


class CTree(NamedTuple):
    visits: Any              # [B, N] int32
    raw_values: Any          # [B, N]
    values: Any              # [B, N]
    parents: Any             # [B, N] int32
    action_from_parent: Any  # [B, N] int32
    reward_in: Any           # [B, N] reward of the edge into the node
    discount_in: Any         # [B, N]
    prior_logits: Any        # [B, N, A]
    children_index: Any      # [B, N, A] int32
    embeddings: Any          # [B, N, ...]
    root_invalid_actions: Any
    root_gumbel: Any


class _Indexer:
    def __init__(self, fn):
        self.fn = fn

    def __getitem__(self, index):
        return self.fn(index)


class View:
    """Unbatched stand-in for upstream `Tree`, as read by its selection code."""

    def __init__(self, t, root_visits=None):
        self.t = t
        self.children_prior_logits = t.prior_logits
        self.raw_values = t.raw_values
        self.root_invalid_actions = t.root_invalid_actions
        self.extra_data = AS.GumbelMuZeroExtraData(root_gumbel=t.root_gumbel)
        self.children_visits = _Indexer(self._visits)
        self._root_visits = root_visits

    def _edge(self, node):
        index = self.t.children_index[node]
        seen = index != UNVISITED
        safe = jnp.where(seen, index, 0)
        pick = lambda x: jnp.where(seen, x[safe], jnp.zeros((), x.dtype))
        return (pick(self.t.visits), pick(self.t.values), pick(self.t.reward_in),
                pick(self.t.discount_in))

    def _visits(self, node):
        visits = self._edge(node)[0]
        if self._root_visits is not None:
            visits = jnp.where(node == ROOT, self._root_visits, visits)
        return visits

    def qvalues(self, node):
        _, values, rewards, discounts = self._edge(node)
        return rewards + discounts * values


def _selectors(num_simulations, max_num_considered_actions, qtransform):
    root = functools.partial(
        AS.gumbel_muzero_root_action_selection, num_simulations=num_simulations,
        max_num_considered_actions=max_num_considered_actions, qtransform=qtransform)
    interior = functools.partial(AS.gumbel_muzero_interior_action_selection,
                                 qtransform=qtransform)
    return root, interior


def _simulate(t, forced, root_select, interior_select, max_depth):
    """Upstream `simulate` on one tree; `forced >= 0` fixes the root action."""
    view = View(t)

    def cond(state):
        return state[4]

    def body(state):
        _, _, node, depth, _ = state
        chosen = jax.lax.cond(
            depth == 0,
            lambda: jnp.where(forced >= 0, forced, root_select(None, view, node)),
            lambda: interior_select(None, view, node, depth))
        next_node = t.children_index[node, chosen]
        depth = depth + 1
        cont = jnp.logical_and(next_node != UNVISITED, depth < max_depth)
        return node, chosen, next_node, depth, cont

    state = (jnp.int32(NO_PARENT), jnp.int32(NO_PARENT), jnp.int32(ROOT),
             jnp.zeros((), t.prior_logits.dtype), jnp.array(True))
    parent, action, _, _, _ = jax.lax.while_loop(cond, body, state)
    return parent, action


def _write(t, parent, action, node, step, embedding, active=None):
    """Upstream `expand` bookkeeping; inactive rows are dropped via OOB index."""
    batch = parent.shape[0]
    rows = jnp.arange(batch)
    if active is not None:
        node = jnp.where(active, node, t.visits.shape[1])
        parent = jnp.where(active, parent, t.visits.shape[1])
    set_ = lambda x, v: x.at[rows, node].set(v, mode="drop")
    new_visit = t.visits.at[rows, node].get(mode="clip") + 1
    return t._replace(
        prior_logits=set_(t.prior_logits, step.prior_logits),
        raw_values=set_(t.raw_values, step.value),
        values=set_(t.values, step.value),
        visits=set_(t.visits, new_visit),
        embeddings=jax.tree.map(set_, t.embeddings, embedding),
        children_index=t.children_index.at[rows, parent, action].set(node, mode="drop"),
        reward_in=set_(t.reward_in, step.reward),
        discount_in=set_(t.discount_in, step.discount),
        parents=set_(t.parents, parent),
        action_from_parent=set_(t.action_from_parent, action),
    )


@jax.vmap
def _backward_small(values, visits, parents, reward_in, discount_in, leaf):
    """Upstream `backward`, over per-node arrays only. leaf == ROOT is a no-op."""
    def cond(state):
        return state[3] != ROOT

    def body(state):
        values, visits, leaf_value, index = state
        parent = parents[index]
        count = visits[parent]
        leaf_value = reward_in[index] + discount_in[index] * leaf_value
        parent_value = (values[parent] * count + leaf_value) / (count + 1.0)
        return (values.at[parent].set(parent_value), visits.at[parent].set(count + 1),
                leaf_value, parent)

    leaf = jnp.asarray(leaf, jnp.int32)
    values, visits, _, _ = jax.lax.while_loop(cond, body, (values, visits, values[leaf], leaf))
    return values, visits


def _backward(t, leaf):
    values, visits = _backward_small(t.values, t.visits, t.parents, t.reward_in,
                                     t.discount_in, leaf)
    return t._replace(values=values, visits=visits)


def _init_tree(root, num_nodes, invalid_actions, gumbel):
    batch, num_actions = root.prior_logits.shape
    dtype = root.value.dtype
    bn, bna = (batch, num_nodes), (batch, num_nodes, num_actions)
    zeros = lambda x: jnp.zeros(bn + x.shape[1:], x.dtype)
    t = CTree(
        visits=jnp.zeros(bn, jnp.int32), raw_values=jnp.zeros(bn, dtype),
        values=jnp.zeros(bn, dtype), parents=jnp.full(bn, NO_PARENT, jnp.int32),
        action_from_parent=jnp.full(bn, NO_PARENT, jnp.int32),
        reward_in=jnp.zeros(bn, dtype), discount_in=jnp.zeros(bn, dtype),
        prior_logits=jnp.zeros(bna, root.prior_logits.dtype),
        children_index=jnp.full(bna, UNVISITED, jnp.int32),
        embeddings=jax.tree.map(zeros, root.embedding),
        root_invalid_actions=invalid_actions, root_gumbel=gumbel)
    rows, node = jnp.arange(batch), jnp.zeros(batch, jnp.int32)
    set_ = lambda x, v: x.at[rows, node].set(v)
    return t._replace(
        prior_logits=set_(t.prior_logits, root.prior_logits),
        raw_values=set_(t.raw_values, root.value), values=set_(t.values, root.value),
        visits=set_(t.visits, t.visits[rows, node] + 1),
        embeddings=jax.tree.map(set_, t.embeddings, root.embedding))


def round_schedule(max_num_considered_actions, num_simulations):
    """Per num_considered k: sizes of consecutive equal-visit blocks (rounds)."""
    table = SH.get_table_of_considered_visits(max_num_considered_actions, num_simulations)
    rounds = []
    for seq in table:
        blocks = []
        for v in seq:
            if blocks and blocks[-1][0] == v:
                blocks[-1][1] += 1
            else:
                blocks.append([v, 1])
        rounds.append([c for _, c in blocks])
    steps = max(len(r) for r in rounds)
    sizes = np.zeros((len(rounds), steps), np.int32)
    for k, r in enumerate(rounds):
        sizes[k, :len(r)] = r
    return sizes


def gumbel_muzero_policy_exact(
    params, rng_key, root, recurrent_fn, num_simulations, invalid_actions=None,
    max_depth=None, *, qtransform=QT.qtransform_completed_by_mix_value,
    max_num_considered_actions=16, gumbel_scale=1.0, round_batched=False,
    min_round_considered=0):
    """Drop-in for mctx.gumbel_muzero_policy; returns (PolicyOutput, info).

    `min_round_considered`: rows with fewer considered actions than this are
    still searched exactly, but their rounds are not padded into wide steps.
    """
    root = root.replace(prior_logits=_mask_invalid_actions(root.prior_logits, invalid_actions))
    rng_key, gumbel_rng = jax.random.split(rng_key)
    gumbel = gumbel_scale * jax.random.gumbel(
        gumbel_rng, shape=root.prior_logits.shape, dtype=root.prior_logits.dtype)
    if max_depth is None:
        max_depth = num_simulations
    if invalid_actions is None:
        invalid_actions = jnp.zeros_like(root.prior_logits)
    batch = root.value.shape[0]
    rows = jnp.arange(batch)
    root_select, interior_select = _selectors(
        num_simulations, max_num_considered_actions, qtransform)
    simulate = jax.vmap(_simulate, in_axes=(0, 0, None, None, None))
    t = _init_tree(root, num_simulations + 1, invalid_actions, gumbel)
    info = {}

    if not round_batched:
        def body(sim, carry):
            key, t = carry
            key, _, expand_key = jax.random.split(key, 3)
            parent, action = simulate(t, jnp.full(batch, -1, jnp.int32), root_select,
                                      interior_select, max_depth)
            nxt = t.children_index[rows, parent, action]
            nxt = jnp.where(nxt == UNVISITED, sim + 1, nxt)
            embedding = jax.tree.map(lambda x: x[rows, parent], t.embeddings)
            step, embedding = recurrent_fn(params, expand_key, action, embedding)
            t = _write(t, parent, action, nxt, step, embedding)
            return key, _backward(t, nxt)

        _, t = jax.lax.fori_loop(0, num_simulations, body, (rng_key, t))
    else:
        t, info = _round_batched(params, rng_key, t, recurrent_fn, num_simulations,
                                 max_num_considered_actions, root_select,
                                 interior_select, max_depth, simulate)

    view_root = jax.vmap(lambda t: View(t)._visits(ROOT))(t)
    considered_visit = jnp.max(view_root, axis=-1, keepdims=True)
    completed_qvalues = jax.vmap(lambda t: qtransform(View(t), ROOT))(t)
    to_argmax = SH.score_considered(considered_visit, gumbel, root.prior_logits,
                                    completed_qvalues, view_root)
    action = AS.masked_argmax(to_argmax, invalid_actions)
    completed_search_logits = _mask_invalid_actions(
        root.prior_logits + completed_qvalues, invalid_actions)
    action_weights = jax.nn.softmax(completed_search_logits)
    out = base.PolicyOutput(action=action, action_weights=action_weights, search_tree=None)
    info.update(tree=t, root_visits=view_root, completed_qvalues=completed_qvalues)
    return out, info


def _round_batched(params, rng_key, t, recurrent_fn, num_simulations, m, root_select,
                   interior_select, max_depth, simulate):
    batch = t.visits.shape[0]
    rows = jnp.arange(batch)
    sizes_np = round_schedule(m, num_simulations)
    sizes = jnp.asarray(sizes_np)
    num_valid = jnp.sum(1 - t.root_invalid_actions, axis=-1).astype(jnp.int32)
    k = jnp.minimum(m, num_valid)
    sim = jnp.zeros(batch, jnp.int32)
    mismatch = jnp.zeros(batch, bool)
    width = sizes_np.max(axis=0)
    calls = 0
    for step_i, w in enumerate(width):
        w = int(w)
        size = sizes[k, step_i]
        # 1. Speculative root order: upstream root selection with each pick
        #    marked visited. Eligible scores do not depend on unseen Q-values.
        root_visits = jax.vmap(lambda t: View(t)._visits(ROOT))(t)
        picks = []
        for j in range(w):
            active = j < size
            a = jax.vmap(lambda t, v: root_select(None, View(t, v), ROOT))(t, root_visits)
            a = jnp.where(active, a, 0)
            picks.append(a)
            root_visits = root_visits + jax.nn.one_hot(a, root_visits.shape[-1], dtype=root_visits.dtype) * active[:, None]
        picks = jnp.stack(picks, 1)                               # [B, w]
        active = jnp.arange(w)[None, :] < size[:, None]           # [B, w]
        # 2. Leaves of all w root actions, against the round-start tree.
        flat = lambda x: x.reshape(batch * w, *x.shape[2:])
        rep = lambda x: jnp.repeat(x[:, None], w, axis=1)
        tf = jax.tree.map(lambda x: flat(rep(x)), t)
        parent, action = simulate(tf, flat(picks), root_select, interior_select, max_depth)
        parent, action = parent.reshape(batch, w), action.reshape(batch, w)
        nxt = jnp.take_along_axis(
            t.children_index.reshape(batch, -1),
            parent * t.children_index.shape[-1] + action, axis=1)
        # 3. One recurrent call for the whole round. Upstream consumes a key per
        #    simulation; recurrent functions used here must ignore it.
        embedding = jax.tree.map(
            lambda x: flat(jnp.take_along_axis(
                x, parent.reshape(batch, w, *([1] * (x.ndim - 2))), axis=1)), t.embeddings)
        rng_key, key = jax.random.split(rng_key)
        step, embedding = recurrent_fn(params, key, flat(action), embedding)
        calls += 1
        unflat = lambda x: x.reshape(batch, w, *x.shape[1:])
        step = jax.tree.map(unflat, step)
        embedding = jax.tree.map(unflat, embedding)
        # 4. Replay in upstream order.
        for j in range(w):
            live = j < size
            true_a = jax.vmap(lambda t: root_select(None, View(t), ROOT))(t)
            match = (picks == true_a[:, None]) & active
            slot = jnp.argmax(match, axis=1)
            mismatch = mismatch | (live & ~jnp.any(match, axis=1))
            g = lambda x: x[rows, slot]
            p, a, n = g(parent), g(action), g(nxt)
            node = jnp.where(n == UNVISITED, sim + 1, n)
            t = _write(t, p, a, node, jax.tree.map(g, step), jax.tree.map(g, embedding),
                       active=live)
            t = _backward(t, jnp.where(live, node, ROOT))
            sim = sim + live.astype(jnp.int32)
    return t, dict(mismatch=mismatch, recurrent_calls=calls, round_widths=width.tolist())


# ---------------------------------------------------------------------------
# Variant 2: upstream Tree layout, upstream simulate/expand, small-carry backward.
# Upstream's vmapped backward while_loop carries the whole tree (embeddings and
# all [B, N, A] arrays); with a batched predicate every iteration selects over
# the full carry. This backward carries only [N] node arrays plus the path, and
# writes the edge statistics with one scatter afterwards.
# ---------------------------------------------------------------------------
from mctx_ref._src import search as S
from mctx_ref._src import tree as tree_lib


def _make_backward_fast(max_path):
    @jax.vmap
    def path_update(node_values, node_visits, parents, action_from_parent,
                    children_rewards, children_discounts, children_visits, leaf):
        num_nodes = node_values.shape[0]

        def cond(state):
            return state[3] != tree_lib.Tree.ROOT_INDEX

        def body(state):
            values, visits, leaf_value, index, d, pp, pa, pv, pc = state
            parent = parents[index]
            count = visits[parent]
            action = action_from_parent[index]
            reward = children_rewards[parent, action]
            leaf_value = reward + children_discounts[parent, action] * leaf_value
            parent_value = (values[parent] * count + leaf_value) / (count + 1.0)
            child_value = values[index]
            child_count = children_visits[parent, action] + 1
            return (values.at[parent].set(parent_value), visits.at[parent].set(count + 1),
                    leaf_value, parent, d + 1, pp.at[d].set(parent), pa.at[d].set(action),
                    pv.at[d].set(child_value), pc.at[d].set(child_count))

        leaf = jnp.asarray(leaf, jnp.int32)
        init = (node_values, node_visits, node_values[leaf], leaf, jnp.int32(0),
                jnp.full(max_path, num_nodes, jnp.int32), jnp.zeros(max_path, jnp.int32),
                jnp.zeros(max_path, node_values.dtype), jnp.zeros(max_path, jnp.int32))
        values, visits, _, _, _, pp, pa, pv, pc = jax.lax.while_loop(cond, body, init)
        return values, visits, pp, pa, pv, pc

    def backward(tree, leaf):
        values, visits, pp, pa, pv, pc = path_update(
            tree.node_values, tree.node_visits, tree.parents, tree.action_from_parent,
            tree.children_rewards, tree.children_discounts, tree.children_visits, leaf)
        rows = jnp.arange(values.shape[0])[:, None]
        # A path visits each edge at most once; unused slots point out of bounds.
        return tree.replace(
            node_values=values, node_visits=visits,
            children_values=tree.children_values.at[rows, pp, pa].set(pv, mode="drop"),
            children_visits=tree.children_visits.at[rows, pp, pa].set(pc, mode="drop"))
    return backward


def gumbel_muzero_policy_fastbw(
    params, rng_key, root, recurrent_fn, num_simulations, invalid_actions=None,
    max_depth=None, *, qtransform=QT.qtransform_completed_by_mix_value,
    max_num_considered_actions=16, gumbel_scale=1.0):
    """mctx.gumbel_muzero_policy with only the backward pass replaced."""
    root = root.replace(prior_logits=_mask_invalid_actions(root.prior_logits, invalid_actions))
    rng_key, gumbel_rng = jax.random.split(rng_key)
    gumbel = gumbel_scale * jax.random.gumbel(
        gumbel_rng, shape=root.prior_logits.shape, dtype=root.prior_logits.dtype)
    extra_data = AS.GumbelMuZeroExtraData(root_gumbel=gumbel)
    root_select, interior_select = _selectors(
        num_simulations, max_num_considered_actions, qtransform)
    action_selection_fn = AS.switching_action_selection_wrapper(
        root_action_selection_fn=root_select, interior_action_selection_fn=interior_select)
    batch = root.value.shape[0]
    rows = jnp.arange(batch)
    depth_limit = num_simulations if max_depth is None else max_depth
    if invalid_actions is None:
        invalid_actions = jnp.zeros_like(root.prior_logits)
    backward = _make_backward_fast(min(depth_limit, num_simulations) + 1)

    def body(sim, carry):
        key, tree = carry
        key, simulate_key, expand_key = jax.random.split(key, 3)
        simulate_keys = jax.random.split(simulate_key, batch)
        parent, action = S.simulate(simulate_keys, tree, action_selection_fn, depth_limit)
        nxt = tree.children_index[rows, parent, action]
        nxt = jnp.where(nxt == tree_lib.Tree.UNVISITED, sim + 1, nxt)
        tree = S.expand(params, expand_key, tree, recurrent_fn, parent, action, nxt)
        return key, backward(tree, nxt)

    tree = S.instantiate_tree_from_root(root, num_simulations,
                                        root_invalid_actions=invalid_actions,
                                        extra_data=extra_data)
    _, tree = jax.lax.fori_loop(0, num_simulations, body, (rng_key, tree))
    summary = tree.summary()
    considered_visit = jnp.max(summary.visit_counts, axis=-1, keepdims=True)
    completed_qvalues = jax.vmap(qtransform, in_axes=[0, None])(tree, tree.ROOT_INDEX)
    to_argmax = SH.score_considered(considered_visit, gumbel, root.prior_logits,
                                    completed_qvalues, summary.visit_counts)
    action = AS.masked_argmax(to_argmax, invalid_actions)
    completed_search_logits = _mask_invalid_actions(
        root.prior_logits + completed_qvalues, invalid_actions)
    return base.PolicyOutput(action=action, action_weights=jax.nn.softmax(completed_search_logits),
                             search_tree=tree)


# ---------------------------------------------------------------------------
# Variant 3: variant 2 plus a slice-based expand.
# With max_depth >= num_simulations a path can never stop on an existing node,
# so simulation `sim` always creates node `sim + 1` in every row. Node writes
# become one dynamic_update_slice along the node axis instead of a batched
# scatter (which XLA lowers with full-tree layout copies). Edge writes stay
# per-row element scatters.
# ---------------------------------------------------------------------------
def _expand_slice(params, rng_key, tree, recurrent_fn, parent, action, node):
    rows = jnp.arange(parent.shape[0])
    embedding = jax.tree.map(lambda x: x[rows, parent], tree.embeddings)
    step, embedding = recurrent_fn(params, rng_key, action, embedding)
    put = lambda x, v: jax.lax.dynamic_update_index_in_dim(x, v.astype(x.dtype), node, axis=1)
    new_visit = jax.lax.dynamic_index_in_dim(tree.node_visits, node, axis=1, keepdims=False) + 1
    full_node = jnp.full(parent.shape, node, jnp.int32)
    return tree.replace(
        children_prior_logits=put(tree.children_prior_logits, step.prior_logits),
        raw_values=put(tree.raw_values, step.value),
        node_values=put(tree.node_values, step.value),
        node_visits=put(tree.node_visits, new_visit),
        embeddings=jax.tree.map(put, tree.embeddings, embedding),
        children_index=tree.children_index.at[rows, parent, action].set(full_node),
        children_rewards=tree.children_rewards.at[rows, parent, action].set(step.reward),
        children_discounts=tree.children_discounts.at[rows, parent, action].set(step.discount),
        parents=put(tree.parents, parent),
        action_from_parent=put(tree.action_from_parent, action))


def gumbel_muzero_policy_fast(
    params, rng_key, root, recurrent_fn, num_simulations, invalid_actions=None,
    max_depth=None, *, qtransform=QT.qtransform_completed_by_mix_value,
    max_num_considered_actions=16, gumbel_scale=1.0):
    """mctx.gumbel_muzero_policy with slice-based expand and small-carry backward."""
    if max_depth is not None and max_depth < num_simulations:
        return gumbel_muzero_policy_fastbw(
            params, rng_key, root, recurrent_fn, num_simulations, invalid_actions, max_depth,
            qtransform=qtransform, max_num_considered_actions=max_num_considered_actions,
            gumbel_scale=gumbel_scale)
    root = root.replace(prior_logits=_mask_invalid_actions(root.prior_logits, invalid_actions))
    rng_key, gumbel_rng = jax.random.split(rng_key)
    gumbel = gumbel_scale * jax.random.gumbel(
        gumbel_rng, shape=root.prior_logits.shape, dtype=root.prior_logits.dtype)
    extra_data = AS.GumbelMuZeroExtraData(root_gumbel=gumbel)
    root_select, interior_select = _selectors(
        num_simulations, max_num_considered_actions, qtransform)
    action_selection_fn = AS.switching_action_selection_wrapper(
        root_action_selection_fn=root_select, interior_action_selection_fn=interior_select)
    batch = root.value.shape[0]
    depth_limit = num_simulations if max_depth is None else max_depth
    if invalid_actions is None:
        invalid_actions = jnp.zeros_like(root.prior_logits)
    backward = _make_backward_fast(num_simulations + 1)

    def body(sim, carry):
        key, tree = carry
        key, simulate_key, expand_key = jax.random.split(key, 3)
        simulate_keys = jax.random.split(simulate_key, batch)
        parent, action = S.simulate(simulate_keys, tree, action_selection_fn, depth_limit)
        node = sim + 1
        tree = _expand_slice(params, expand_key, tree, recurrent_fn, parent, action, node)
        return key, backward(tree, jnp.full((batch,), node, jnp.int32))

    tree = S.instantiate_tree_from_root(root, num_simulations,
                                        root_invalid_actions=invalid_actions,
                                        extra_data=extra_data)
    _, tree = jax.lax.fori_loop(0, num_simulations, body, (rng_key, tree))
    summary = tree.summary()
    considered_visit = jnp.max(summary.visit_counts, axis=-1, keepdims=True)
    completed_qvalues = jax.vmap(qtransform, in_axes=[0, None])(tree, tree.ROOT_INDEX)
    to_argmax = SH.score_considered(considered_visit, gumbel, root.prior_logits,
                                    completed_qvalues, summary.visit_counts)
    action = AS.masked_argmax(to_argmax, invalid_actions)
    completed_search_logits = _mask_invalid_actions(
        root.prior_logits + completed_qvalues, invalid_actions)
    return base.PolicyOutput(action=action, action_weights=jax.nn.softmax(completed_search_logits),
                             search_tree=tree)


# ---------------------------------------------------------------------------
# Variant 4: flat edge storage.
# Edge arrays are stored as [B, N * A]. XLA lowers per-row element scatters
# into a 3-D [B, N, A] array by flattening it, which needs full relayout
# copies on TPU; on a 2-D array the same scatter is in place. Rows are read
# with dynamic_slice, and upstream's selection/qtransform code runs unchanged
# against a view.
# ---------------------------------------------------------------------------
class FlatTree(NamedTuple):
    node_visits: Any
    raw_values: Any
    node_values: Any
    parents: Any
    action_from_parent: Any
    children_index: Any        # [B, N * A]
    children_prior_logits: Any
    children_values: Any
    children_visits: Any
    children_rewards: Any
    children_discounts: Any
    embeddings: Any
    root_invalid_actions: Any
    root_gumbel: Any


class FlatView:
    def __init__(self, t, num_actions):
        self.t, self.A = t, num_actions
        row = lambda x: _Indexer(lambda n: jax.lax.dynamic_slice_in_dim(x, n * num_actions, num_actions))
        self.children_visits = row(t.children_visits)
        self.children_prior_logits = row(t.children_prior_logits)
        self.children_index = row(t.children_index)
        self._values, self._rewards, self._discounts = (
            row(t.children_values), row(t.children_rewards), row(t.children_discounts))
        self.raw_values = t.raw_values
        self.root_invalid_actions = t.root_invalid_actions
        self.extra_data = AS.GumbelMuZeroExtraData(root_gumbel=t.root_gumbel)

    def qvalues(self, node):
        return self._rewards[node] + self._discounts[node] * self._values[node]


def _flat_simulate(t, A, root_select, interior_select, max_depth):
    view = FlatView(t, A)
    # Same as upstream's switching wrapper; closures because the view is not a pytree.
    selector = lambda node, depth: jax.lax.cond(
        depth == 0, lambda: root_select(None, view, node),
        lambda: interior_select(None, view, node, depth))

    def cond(state):
        return state[4]

    def body(state):
        _, _, node, depth, _ = state
        action = selector(node, depth)
        next_node = t.children_index[node * A + action]
        depth = depth + 1
        return node, action, next_node, depth, jnp.logical_and(next_node != UNVISITED, depth < max_depth)

    state = (jnp.int32(NO_PARENT), jnp.int32(NO_PARENT), jnp.int32(ROOT),
             jnp.zeros((), t.children_prior_logits.dtype), jnp.array(True))
    parent, action, _, _, _ = jax.lax.while_loop(cond, body, state)
    return parent, action


def _make_flat_backward(max_path, A):
    @jax.vmap
    def path_update(node_values, node_visits, parents, action_from_parent,
                    children_rewards, children_discounts, children_visits, leaf):
        num_nodes = node_values.shape[0]

        def cond(state):
            return state[3] != ROOT

        def body(state):
            values, visits, leaf_value, index, d, pe, pv, pc = state
            parent = parents[index]
            count = visits[parent]
            edge = parent * A + action_from_parent[index]
            leaf_value = children_rewards[edge] + children_discounts[edge] * leaf_value
            parent_value = (values[parent] * count + leaf_value) / (count + 1.0)
            return (values.at[parent].set(parent_value), visits.at[parent].set(count + 1),
                    leaf_value, parent, d + 1, pe.at[d].set(edge),
                    pv.at[d].set(values[index]), pc.at[d].set(children_visits[edge] + 1))

        leaf = jnp.asarray(leaf, jnp.int32)
        init = (node_values, node_visits, node_values[leaf], leaf, jnp.int32(0),
                jnp.full(max_path, num_nodes * A, jnp.int32),
                jnp.zeros(max_path, node_values.dtype), jnp.zeros(max_path, jnp.int32))
        values, visits, _, _, _, pe, pv, pc = jax.lax.while_loop(cond, body, init)
        return values, visits, pe, pv, pc

    def backward(t, leaf):
        values, visits, pe, pv, pc = path_update(
            t.node_values, t.node_visits, t.parents, t.action_from_parent,
            t.children_rewards, t.children_discounts, t.children_visits, leaf)
        rows = jnp.arange(values.shape[0])[:, None]
        return t._replace(
            node_values=values, node_visits=visits,
            children_values=t.children_values.at[rows, pe].set(pv, mode="drop"),
            children_visits=t.children_visits.at[rows, pe].set(pc, mode="drop"))
    return backward


def flat_to_tree(t, A):
    """Upstream Tree view of a FlatTree (for comparisons, outside the loop)."""
    B, N = t.node_visits.shape
    r = lambda x: x.reshape(B, N, A)
    return tree_lib.Tree(
        node_visits=t.node_visits, raw_values=t.raw_values, node_values=t.node_values,
        parents=t.parents, action_from_parent=t.action_from_parent,
        children_index=r(t.children_index), children_prior_logits=r(t.children_prior_logits),
        children_values=r(t.children_values), children_visits=r(t.children_visits),
        children_rewards=r(t.children_rewards), children_discounts=r(t.children_discounts),
        embeddings=t.embeddings, root_invalid_actions=t.root_invalid_actions,
        extra_data=AS.GumbelMuZeroExtraData(root_gumbel=t.root_gumbel))


def gumbel_muzero_policy_flat(
    params, rng_key, root, recurrent_fn, num_simulations, invalid_actions=None,
    max_depth=None, *, qtransform=QT.qtransform_completed_by_mix_value,
    max_num_considered_actions=16, gumbel_scale=1.0, return_tree=True):
    """Bitwise-equivalent mctx.gumbel_muzero_policy with flat edge storage.

    Requires max_depth is None or >= num_simulations (then every simulation
    creates node sim + 1); otherwise falls back to the fast-backward variant.
    """
    if max_depth is not None and max_depth < num_simulations:
        return gumbel_muzero_policy_fastbw(
            params, rng_key, root, recurrent_fn, num_simulations, invalid_actions, max_depth,
            qtransform=qtransform, max_num_considered_actions=max_num_considered_actions,
            gumbel_scale=gumbel_scale)
    root = root.replace(prior_logits=_mask_invalid_actions(root.prior_logits, invalid_actions))
    rng_key, gumbel_rng = jax.random.split(rng_key)
    gumbel = gumbel_scale * jax.random.gumbel(
        gumbel_rng, shape=root.prior_logits.shape, dtype=root.prior_logits.dtype)
    root_select, interior_select = _selectors(
        num_simulations, max_num_considered_actions, qtransform)
    batch, A = root.prior_logits.shape
    N = num_simulations + 1
    depth_limit = num_simulations if max_depth is None else max_depth
    if invalid_actions is None:
        invalid_actions = jnp.zeros_like(root.prior_logits)
    dtype = root.value.dtype
    bn, bna = (batch, N), (batch, N * A)
    zeros = lambda x: jnp.zeros(bn + x.shape[1:], x.dtype)
    t = FlatTree(
        node_visits=jnp.zeros(bn, jnp.int32), raw_values=jnp.zeros(bn, dtype),
        node_values=jnp.zeros(bn, dtype), parents=jnp.full(bn, NO_PARENT, jnp.int32),
        action_from_parent=jnp.full(bn, NO_PARENT, jnp.int32),
        children_index=jnp.full(bna, UNVISITED, jnp.int32),
        children_prior_logits=jnp.zeros(bna, root.prior_logits.dtype),
        children_values=jnp.zeros(bna, dtype), children_visits=jnp.zeros(bna, jnp.int32),
        children_rewards=jnp.zeros(bna, dtype), children_discounts=jnp.zeros(bna, dtype),
        embeddings=jax.tree.map(zeros, root.embedding),
        root_invalid_actions=invalid_actions, root_gumbel=gumbel)
    put_node = lambda x, v, node: jax.lax.dynamic_update_index_in_dim(x, v.astype(x.dtype), node, axis=1)
    put_row = lambda x, v, node: jax.lax.dynamic_update_slice_in_dim(x, v.astype(x.dtype), node * A, axis=1)
    t = t._replace(
        children_prior_logits=put_row(t.children_prior_logits, root.prior_logits, 0),
        raw_values=put_node(t.raw_values, root.value, 0),
        node_values=put_node(t.node_values, root.value, 0),
        node_visits=put_node(t.node_visits, t.node_visits[:, 0] + 1, 0),
        embeddings=jax.tree.map(lambda x, v: put_node(x, v, 0), t.embeddings, root.embedding))
    simulate = jax.vmap(_flat_simulate, in_axes=(0, None, None, None, None))
    backward = _make_flat_backward(num_simulations + 1, A)
    rows = jnp.arange(batch)

    def body(sim, carry):
        key, t = carry
        key, _, expand_key = jax.random.split(key, 3)
        parent, action = simulate(t, A, root_select, interior_select, depth_limit)
        node = sim + 1
        embedding = jax.tree.map(lambda x: x[rows, parent], t.embeddings)
        step, embedding = recurrent_fn(params, expand_key, action, embedding)
        new_visit = jax.lax.dynamic_index_in_dim(t.node_visits, node, axis=1, keepdims=False) + 1
        edge = parent * A + action
        t = t._replace(
            children_prior_logits=put_row(t.children_prior_logits, step.prior_logits, node),
            raw_values=put_node(t.raw_values, step.value, node),
            node_values=put_node(t.node_values, step.value, node),
            node_visits=put_node(t.node_visits, new_visit, node),
            embeddings=jax.tree.map(lambda x, v: put_node(x, v, node), t.embeddings, embedding),
            children_index=t.children_index.at[rows, edge].set(node),
            children_rewards=t.children_rewards.at[rows, edge].set(step.reward),
            children_discounts=t.children_discounts.at[rows, edge].set(step.discount),
            parents=put_node(t.parents, parent, node),
            action_from_parent=put_node(t.action_from_parent, action, node))
        return key, backward(t, jnp.full((batch,), node, jnp.int32))

    _, t = jax.lax.fori_loop(0, num_simulations, body, (rng_key, t))
    root_visits = t.children_visits[:, :A]
    considered_visit = jnp.max(root_visits, axis=-1, keepdims=True)
    completed_qvalues = jax.vmap(lambda t: qtransform(FlatView(t, A), ROOT))(t)
    to_argmax = SH.score_considered(considered_visit, gumbel, root.prior_logits,
                                    completed_qvalues, root_visits)
    action = AS.masked_argmax(to_argmax, invalid_actions)
    completed_search_logits = _mask_invalid_actions(
        root.prior_logits + completed_qvalues, invalid_actions)
    return base.PolicyOutput(action=action, action_weights=jax.nn.softmax(completed_search_logits),
                             search_tree=flat_to_tree(t, A) if return_tree else None)


# ---------------------------------------------------------------------------
# Variant 5: node-only storage, no scatters into large arrays.
# Every per-edge quantity equals a per-node quantity of the child (visits,
# value, the reward/discount of the edge into it), and the child of (p, a) is
# the node whose (parent, action_from_parent) is (p, a). So the tree keeps only
# [B, N] node arrays, [B, N, A] prior logits and the embeddings, all written by
# dynamic_update_slice at node sim + 1. Edge rows are rebuilt with a dense
# [A, N] equality mask (exactly one match or none), avoiding TPU gathers.
# ---------------------------------------------------------------------------
class NodeTree(NamedTuple):
    node_visits: Any
    raw_values: Any
    node_values: Any
    parents: Any
    action_from_parent: Any
    reward_in: Any
    discount_in: Any
    children_prior_logits: Any   # [B, N, A]
    embeddings: Any
    root_invalid_actions: Any
    root_gumbel: Any


class NodeView:
    def __init__(self, t, num_actions, edge_rows="mask"):
        self.t, self.A, self.edge_rows = t, num_actions, edge_rows
        self.children_prior_logits = t.children_prior_logits
        self.raw_values = t.raw_values
        self.root_invalid_actions = t.root_invalid_actions
        self.extra_data = AS.GumbelMuZeroExtraData(root_gumbel=t.root_gumbel)
        self.children_visits = _Indexer(lambda n: self._pick(n, self.t.node_visits))

    def _match(self, node):
        actions = jnp.arange(self.A, dtype=self.t.action_from_parent.dtype)
        return (self.t.parents == node)[None, :] & (self.t.action_from_parent[None, :] == actions[:, None])

    def _key(self, node):
        # Action of each node if it is a child of `node`, else -1.
        return jnp.where(self.t.parents == node, self.t.action_from_parent, -1)

    def _pick_many(self, node, xs):
        """Edge rows [A] for several per-node arrays, sharing one comparison."""
        actions = jnp.arange(self.A, dtype=self.t.action_from_parent.dtype)
        match = self._key(node)[None, :] == actions[:, None]            # [A, N]
        if self.edge_rows == "matmul":
            # One-hot matmul at HIGHEST precision reproduces each value exactly.
            stacked = jnp.stack([x.astype(jnp.float32) for x in xs], -1)  # [N, S]
            rows = jnp.matmul(match.astype(jnp.float32), stacked,
                              precision=jax.lax.Precision.HIGHEST)
            return [rows[:, i].astype(x.dtype) for i, x in enumerate(xs)]
        return [jnp.sum(jnp.where(match, x[None, :], jnp.zeros((), x.dtype)), axis=-1) for x in xs]

    def _pick(self, node, x):
        if self.edge_rows in ("mask2", "matmul"):
            return self._pick_many(node, [x])[0]
        if self.edge_rows == "scatter":
            # Scatter this node's children (at most N) into a fresh [A] row.
            target = jnp.where(self.t.parents == node, self.t.action_from_parent, self.A)
            return jnp.zeros((self.A,), x.dtype).at[target].set(x, mode="drop")
        # Exactly one match per visited action; sums of a single term are exact.
        return jnp.sum(jnp.where(self._match(node), x[None, :], jnp.zeros((), x.dtype)), axis=-1)

    def qvalues(self, node):
        if self.edge_rows in ("mask2", "matmul"):
            # Same elementwise op on the same inputs, computed before the pick.
            # Unvisited actions pick 0, as upstream's 0 + 0 * 0.
            return self._pick(node, self.t.reward_in + self.t.discount_in * self.t.node_values)
        return (self._pick(node, self.t.reward_in)
                + self._pick(node, self.t.discount_in) * self._pick(node, self.t.node_values))


def _node_child(t, node, action):
    hit = (t.parents == node) & (t.action_from_parent == action)
    index = jnp.arange(t.parents.shape[0], dtype=jnp.int32)
    return jnp.where(jnp.any(hit), jnp.sum(jnp.where(hit, index, 0)), UNVISITED)


def _node_simulate(t, A, root_select, interior_select, max_depth, hoist_root=False,
                   edge_rows="mask"):
    view = NodeView(t, A, edge_rows)

    def cond(state):
        return state[4]

    def body(state):
        _, _, node, depth, _ = state
        if hoist_root:
            action = interior_select(None, view, node, depth)
        else:
            action = jax.lax.cond(depth == 0, lambda: root_select(None, view, node),
                                  lambda: interior_select(None, view, node, depth))
        next_node = _node_child(t, node, action)
        depth = depth + 1
        return node, action, next_node, depth, jnp.logical_and(next_node != UNVISITED, depth < max_depth)

    depth0 = jnp.zeros((), t.children_prior_logits.dtype)
    if hoist_root:
        # Upstream's first loop iteration, done once outside the loop: under vmap
        # the in-loop cond would otherwise evaluate root selection at every depth.
        root_action = root_select(None, view, jnp.int32(ROOT))
        child = _node_child(t, jnp.int32(ROOT), root_action)
        depth1 = depth0 + 1
        state = (jnp.int32(ROOT), root_action, child, depth1,
                 jnp.logical_and(child != UNVISITED, depth1 < max_depth))
    else:
        state = (jnp.int32(NO_PARENT), jnp.int32(NO_PARENT), jnp.int32(ROOT),
                 depth0, jnp.array(True))
    parent, action, _, _, _ = jax.lax.while_loop(cond, body, state)
    return parent, action


def node_to_tree(t, A):
    """Upstream Tree built from a NodeTree (for comparisons, outside the loop)."""
    N = t.node_visits.shape[1]

    def per_row(t):
        view = NodeView(t, A)
        nodes = jnp.arange(N)
        rows = lambda x: jax.vmap(lambda n: view._pick(n, x))(nodes)
        child = jax.vmap(lambda n: jax.vmap(lambda a: _node_child(t, n, a))(jnp.arange(A)))(nodes)
        return tree_lib.Tree(
            node_visits=t.node_visits, raw_values=t.raw_values, node_values=t.node_values,
            parents=t.parents, action_from_parent=t.action_from_parent,
            children_index=child, children_prior_logits=t.children_prior_logits,
            children_values=rows(t.node_values), children_visits=rows(t.node_visits),
            children_rewards=rows(t.reward_in), children_discounts=rows(t.discount_in),
            embeddings=t.embeddings, root_invalid_actions=t.root_invalid_actions,
            extra_data=AS.GumbelMuZeroExtraData(root_gumbel=t.root_gumbel))
    return jax.vmap(per_row)(t)


def gumbel_muzero_policy_node(
    params, rng_key, root, recurrent_fn, num_simulations, invalid_actions=None,
    max_depth=None, *, qtransform=QT.qtransform_completed_by_mix_value,
    max_num_considered_actions=16, gumbel_scale=1.0, return_tree=True,
    hoist_root=False, edge_rows="mask", node_major_priors=False):
    """Bitwise-equivalent mctx.gumbel_muzero_policy without large scatters.

    Requires max_depth is None or >= num_simulations; otherwise falls back.
    """
    if max_depth is not None and max_depth < num_simulations:
        return gumbel_muzero_policy_fastbw(
            params, rng_key, root, recurrent_fn, num_simulations, invalid_actions, max_depth,
            qtransform=qtransform, max_num_considered_actions=max_num_considered_actions,
            gumbel_scale=gumbel_scale)
    root = root.replace(prior_logits=_mask_invalid_actions(root.prior_logits, invalid_actions))
    rng_key, gumbel_rng = jax.random.split(rng_key)
    gumbel = gumbel_scale * jax.random.gumbel(
        gumbel_rng, shape=root.prior_logits.shape, dtype=root.prior_logits.dtype)
    root_select, interior_select = _selectors(
        num_simulations, max_num_considered_actions, qtransform)
    batch, A = root.prior_logits.shape
    N = num_simulations + 1
    depth_limit = num_simulations if max_depth is None else max_depth
    if invalid_actions is None:
        invalid_actions = jnp.zeros_like(root.prior_logits)
    dtype = root.value.dtype
    bn = (batch, N)
    zeros = lambda x: jnp.zeros(bn + x.shape[1:], x.dtype)
    t = NodeTree(
        node_visits=jnp.zeros(bn, jnp.int32), raw_values=jnp.zeros(bn, dtype),
        node_values=jnp.zeros(bn, dtype), parents=jnp.full(bn, NO_PARENT, jnp.int32),
        action_from_parent=jnp.full(bn, NO_PARENT, jnp.int32),
        reward_in=jnp.zeros(bn, dtype), discount_in=jnp.zeros(bn, dtype),
        children_prior_logits=jnp.zeros((N, batch, A) if node_major_priors else (batch, N, A),
                                        root.prior_logits.dtype),
        embeddings=jax.tree.map(zeros, root.embedding),
        root_invalid_actions=invalid_actions, root_gumbel=gumbel)
    put = lambda x, v, node: jax.lax.dynamic_update_index_in_dim(x, v.astype(x.dtype), node, axis=1)
    # Node-major priors make each node write cover whole (8, 128) TPU tiles.
    put_prior = (lambda x, v, node: jax.lax.dynamic_update_index_in_dim(x, v.astype(x.dtype), node, axis=0)) \
        if node_major_priors else put
    axes = NodeTree(*([0] * len(NodeTree._fields)))._replace(
        children_prior_logits=1 if node_major_priors else 0)
    t = t._replace(
        children_prior_logits=put_prior(t.children_prior_logits, root.prior_logits, 0),
        raw_values=put(t.raw_values, root.value, 0), node_values=put(t.node_values, root.value, 0),
        node_visits=put(t.node_visits, t.node_visits[:, 0] + 1, 0),
        embeddings=jax.tree.map(lambda x, v: put(x, v, 0), t.embeddings, root.embedding))
    simulate = jax.vmap(functools.partial(_node_simulate, hoist_root=hoist_root, edge_rows=edge_rows),
                        in_axes=(axes, None, None, None, None))
    rows = jnp.arange(batch)

    def body(sim, carry):
        key, t = carry
        key, _, expand_key = jax.random.split(key, 3)
        parent, action = simulate(t, A, root_select, interior_select, depth_limit)
        node = sim + 1
        embedding = jax.tree.map(lambda x: x[rows, parent], t.embeddings)
        step, embedding = recurrent_fn(params, expand_key, action, embedding)
        new_visit = jax.lax.dynamic_index_in_dim(t.node_visits, node, axis=1, keepdims=False) + 1
        t = t._replace(
            children_prior_logits=put_prior(t.children_prior_logits, step.prior_logits, node),
            raw_values=put(t.raw_values, step.value, node),
            node_values=put(t.node_values, step.value, node),
            node_visits=put(t.node_visits, new_visit, node),
            embeddings=jax.tree.map(lambda x, v: put(x, v, node), t.embeddings, embedding),
            reward_in=put(t.reward_in, step.reward, node),
            discount_in=put(t.discount_in, step.discount, node),
            parents=put(t.parents, parent, node),
            action_from_parent=put(t.action_from_parent, action, node))
        values, visits = _backward_small(t.node_values, t.node_visits, t.parents, t.reward_in,
                                         t.discount_in, jnp.full((batch,), node, jnp.int32))
        return key, t._replace(node_values=values, node_visits=visits)

    _, t = jax.lax.fori_loop(0, num_simulations, body, (rng_key, t))
    # Materialize the root's edge rows behind a barrier, then run upstream's own
    # final computation on a one-node upstream Tree, so its fusion matches.
    view_rows = jax.vmap(lambda t: (
        NodeView(t, A)._pick(ROOT, t.node_visits), NodeView(t, A)._pick(ROOT, t.node_values),
        NodeView(t, A)._pick(ROOT, t.reward_in), NodeView(t, A)._pick(ROOT, t.discount_in)),
        in_axes=(axes,))(t)
    rv, cv, cr, cd = jax.lax.optimization_barrier(view_rows)
    one = lambda x: x[:, None]
    root_tree = tree_lib.Tree(
        node_visits=t.node_visits[:, :1], raw_values=t.raw_values[:, :1],
        node_values=t.node_values[:, :1], parents=t.parents[:, :1],
        action_from_parent=t.action_from_parent[:, :1],
        children_index=jnp.where(one(rv) > 0, 0, UNVISITED).astype(jnp.int32),
        children_prior_logits=(jnp.swapaxes(t.children_prior_logits[:1], 0, 1) if node_major_priors
                               else t.children_prior_logits[:, :1]), children_values=one(cv),
        children_visits=one(rv), children_rewards=one(cr), children_discounts=one(cd),
        embeddings=None, root_invalid_actions=invalid_actions,
        extra_data=AS.GumbelMuZeroExtraData(root_gumbel=gumbel))
    root_visits = root_tree.summary().visit_counts
    considered_visit = jnp.max(root_visits, axis=-1, keepdims=True)
    completed_qvalues = jax.vmap(qtransform, in_axes=[0, None])(root_tree, root_tree.ROOT_INDEX)
    to_argmax = SH.score_considered(considered_visit, gumbel, root.prior_logits,
                                    completed_qvalues, root_visits)
    action = AS.masked_argmax(to_argmax, invalid_actions)
    completed_search_logits = _mask_invalid_actions(
        root.prior_logits + completed_qvalues, invalid_actions)
    return base.PolicyOutput(action=action, action_weights=jax.nn.softmax(completed_search_logits),
                             search_tree=node_to_tree(t._replace(children_prior_logits=jnp.swapaxes(
                                 t.children_prior_logits, 0, 1)) if node_major_priors else t, A)
                             if return_tree else None)


# ---------------------------------------------------------------------------
# BNK target compression (same arithmetic as nanoalphazero.mcts.compress_muzero_policy).
# Applied after search, so the search itself stays exact.
# ---------------------------------------------------------------------------
def compress_action_weights(action_weights, num_k_actions):
    """Returns (k_indices, k_weights, retained_mass) for the top-K of the target.

    Renormalising the top-K probabilities equals the softmax of the selected
    search logits; `retained_mass` is the probability mass kept.
    """
    weights, indices = jax.lax.top_k(action_weights, num_k_actions)
    mass = jnp.sum(weights, axis=-1, keepdims=True)
    index_dtype = jnp.uint16 if action_weights.shape[-1] <= 65536 else jnp.int32
    return (indices.astype(index_dtype),
            weights / jnp.maximum(mass, jnp.finfo(weights.dtype).tiny), mass[..., 0])


# ---------------------------------------------------------------------------
# Variant 6 (node2): variant 5 plus
#   * edge rows built for only two per-node arrays (visits, and Q computed per
#     node), by one-hot matmul at HIGHEST precision (exact for 0/1 weights);
#   * root selection done once per simulation outside the descent loop, with
#     the root prior row passed in directly (reading it from the tree there
#     made XLA relayout the whole prior array every simulation);
#   * prior logits stored flat as [B, N * A_pad], A_pad a multiple of 128, so a
#     node write covers whole TPU tiles instead of 1 row of every 8-row tile.
# ---------------------------------------------------------------------------
class NodeTree2(NamedTuple):
    node_visits: Any
    raw_values: Any
    node_values: Any
    parents: Any
    action_from_parent: Any
    reward_in: Any
    discount_in: Any
    prior_flat: Any            # [B, N * A_pad]
    embeddings: Any
    root_invalid_actions: Any
    root_gumbel: Any
    root_prior_logits: Any     # [B, A]
    root_visits: Any = None    # [B, A] maintained root edge visits (root_rows=True)
    root_q: Any = None         # [B, A] maintained root edge Q-values


class NodeView2:
    """Upstream-Tree view; `root_only` serves the root prior from its own row."""

    def __init__(self, t, num_actions, a_pad, root_only=False, edge_rows="matmul"):
        self.t, self.A, self.a_pad, self.edge_rows = t, num_actions, a_pad, edge_rows
        self.root_rows = root_only and t.root_visits is not None
        if root_only:
            self.children_prior_logits = _Indexer(lambda n: t.root_prior_logits)
        elif t.prior_flat.ndim == 3:  # tiled [N, 8, A_pad / 8] per row
            self.children_prior_logits = _Indexer(
                lambda n: t.prior_flat[n].reshape(-1)[:num_actions])
        else:
            if t.prior_flat.ndim == 1:  # flat [N * A_pad] per row
                self.children_prior_logits = _Indexer(
                    lambda n: jax.lax.dynamic_slice_in_dim(t.prior_flat, n * a_pad, num_actions))
            else:  # [N, A] per row
                self.children_prior_logits = t.prior_flat
        self.raw_values = t.raw_values
        self.root_invalid_actions = t.root_invalid_actions
        self.extra_data = AS.GumbelMuZeroExtraData(root_gumbel=t.root_gumbel)
        if self.root_rows:
            self.children_visits = _Indexer(lambda n: t.root_visits)
        else:
            self.children_visits = _Indexer(lambda n: self._pick(n, t.node_visits))

    def _pick(self, node, x):
        actions = jnp.arange(self.A, dtype=self.t.action_from_parent.dtype)
        key = jnp.where(self.t.parents == node, self.t.action_from_parent, -1)
        match = key[None, :] == actions[:, None]                         # [A, N]
        if self.edge_rows == "matmul":
            return jnp.matmul(match.astype(jnp.float32), x.astype(jnp.float32),
                              precision=jax.lax.Precision.HIGHEST).astype(x.dtype)
        return jnp.sum(jnp.where(match, x[None, :], jnp.zeros((), x.dtype)), axis=-1)

    def qvalues(self, node):
        if self.root_rows:
            return self.t.root_q
        # Same elementwise op on the same inputs as upstream, applied per node.
        return self._pick(node, self.t.reward_in + self.t.discount_in * self.t.node_values)


def _root_child_on_path(parents, leaf):
    """Ancestor of `leaf` whose parent is the root (leaf itself if depth 1)."""
    def cond(i):
        return parents[i] != ROOT
    return jax.lax.while_loop(cond, lambda i: parents[i], leaf)


def _node2_simulate(t, A, a_pad, root_select, interior_select, max_depth, edge_rows):
    view = NodeView2(t, A, a_pad, edge_rows=edge_rows)
    root_view = NodeView2(t, A, a_pad, root_only=True, edge_rows=edge_rows)
    depth = jnp.zeros((), t.root_prior_logits.dtype) + 1
    root_action = root_select(None, root_view, jnp.int32(ROOT))
    child = _node_child(t, jnp.int32(ROOT), root_action)

    def cond(state):
        return state[4]

    def body(state):
        _, _, node, depth, _ = state
        action = interior_select(None, view, node, depth)
        next_node = _node_child(t, node, action)
        depth = depth + 1
        return node, action, next_node, depth, jnp.logical_and(next_node != UNVISITED, depth < max_depth)

    state = (jnp.int32(ROOT), root_action, child, depth,
             jnp.logical_and(child != UNVISITED, depth < max_depth))
    parent, action, _, _, _ = jax.lax.while_loop(cond, body, state)
    return parent, action


def gumbel_muzero_policy_node2(
    params, rng_key, root, recurrent_fn, num_simulations, invalid_actions=None,
    max_depth=None, *, qtransform=QT.qtransform_completed_by_mix_value,
    max_num_considered_actions=16, gumbel_scale=1.0, return_tree=True,
    edge_rows="matmul", flat_priors=False, tiled_priors=True, root_rows=True):
    """Bitwise-equivalent mctx.gumbel_muzero_policy; fastest variant so far.

    Requires max_depth is None or >= num_simulations; otherwise falls back.
    """
    if max_depth is not None and max_depth < num_simulations:
        return gumbel_muzero_policy_fastbw(
            params, rng_key, root, recurrent_fn, num_simulations, invalid_actions, max_depth,
            qtransform=qtransform, max_num_considered_actions=max_num_considered_actions,
            gumbel_scale=gumbel_scale)
    root = root.replace(prior_logits=_mask_invalid_actions(root.prior_logits, invalid_actions))
    rng_key, gumbel_rng = jax.random.split(rng_key)
    gumbel = gumbel_scale * jax.random.gumbel(
        gumbel_rng, shape=root.prior_logits.shape, dtype=root.prior_logits.dtype)
    root_select, interior_select = _selectors(
        num_simulations, max_num_considered_actions, qtransform)
    batch, A = root.prior_logits.shape
    a_pad = -(-A // 128) * 128 if flat_priors else (-(-A // 8) * 8 if tiled_priors else A)
    N = num_simulations + 1
    depth_limit = num_simulations if max_depth is None else max_depth
    if invalid_actions is None:
        invalid_actions = jnp.zeros_like(root.prior_logits)
    dtype = root.value.dtype
    bn = (batch, N)
    zeros = lambda x: jnp.zeros(bn + x.shape[1:], x.dtype)
    put = lambda x, v, node: jax.lax.dynamic_update_index_in_dim(x, v.astype(x.dtype), node, axis=1)
    put_prior = (lambda x, v, node: jax.lax.dynamic_update_slice_in_dim(x, v.astype(x.dtype), node * a_pad, axis=1)) \
        if flat_priors else ((lambda x, v, node: put(x, jnp.pad(v, ((0, 0), (0, a_pad - A))).reshape(
            v.shape[0], 8, a_pad // 8), node)) if tiled_priors else put)
    t = NodeTree2(
        node_visits=jnp.zeros(bn, jnp.int32), raw_values=jnp.zeros(bn, dtype),
        node_values=jnp.zeros(bn, dtype), parents=jnp.full(bn, NO_PARENT, jnp.int32),
        action_from_parent=jnp.full(bn, NO_PARENT, jnp.int32),
        reward_in=jnp.zeros(bn, dtype), discount_in=jnp.zeros(bn, dtype),
        prior_flat=jnp.zeros((batch, N * a_pad) if flat_priors else
                             ((batch, N, 8, a_pad // 8) if tiled_priors else (batch, N, A)),
                             root.prior_logits.dtype),
        embeddings=jax.tree.map(zeros, root.embedding),
        root_invalid_actions=invalid_actions, root_gumbel=gumbel,
        root_prior_logits=root.prior_logits,
        root_visits=jnp.zeros((batch, A), jnp.int32) if root_rows else None,
        root_q=jnp.zeros((batch, A), dtype) if root_rows else None)
    t = t._replace(
        prior_flat=put_prior(t.prior_flat, root.prior_logits, 0),
        raw_values=put(t.raw_values, root.value, 0), node_values=put(t.node_values, root.value, 0),
        node_visits=put(t.node_visits, t.node_visits[:, 0] + 1, 0),
        embeddings=jax.tree.map(lambda x, v: put(x, v, 0), t.embeddings, root.embedding))
    simulate = jax.vmap(_node2_simulate, in_axes=(0, None, None, None, None, None, None))
    rows = jnp.arange(batch)

    def body(sim, carry):
        key, t = carry
        key, _, expand_key = jax.random.split(key, 3)
        parent, action = simulate(t, A, a_pad, root_select, interior_select, depth_limit, edge_rows)
        node = sim + 1
        embedding = jax.tree.map(lambda x: x[rows, parent], t.embeddings)
        step, embedding = recurrent_fn(params, expand_key, action, embedding)
        new_visit = jax.lax.dynamic_index_in_dim(t.node_visits, node, axis=1, keepdims=False) + 1
        t = t._replace(
            prior_flat=put_prior(t.prior_flat, step.prior_logits, node),
            raw_values=put(t.raw_values, step.value, node),
            node_values=put(t.node_values, step.value, node),
            node_visits=put(t.node_visits, new_visit, node),
            embeddings=jax.tree.map(lambda x, v: put(x, v, node), t.embeddings, embedding),
            reward_in=put(t.reward_in, step.reward, node),
            discount_in=put(t.discount_in, step.discount, node),
            parents=put(t.parents, parent, node),
            action_from_parent=put(t.action_from_parent, action, node))
        values, visits = _backward_small(t.node_values, t.node_visits, t.parents, t.reward_in,
                                         t.discount_in, jnp.full((batch,), node, jnp.int32))
        t = t._replace(node_values=values, node_visits=visits)
        if root_rows:
            # Only the root child on this path changed; refresh its entry with
            # the same expressions the row rebuild would use.
            child = jax.vmap(_root_child_on_path)(t.parents, jnp.full((batch,), node, jnp.int32))
            hit = jnp.arange(A)[None, :] == t.action_from_parent[rows, child][:, None]
            q = t.reward_in + t.discount_in * t.node_values
            t = t._replace(
                root_visits=jnp.where(hit, t.node_visits[rows, child][:, None], t.root_visits),
                root_q=jnp.where(hit, q[rows, child][:, None], t.root_q))
        return key, t

    _, t = jax.lax.fori_loop(0, num_simulations, body, (rng_key, t))
    # Same final step as variant 5: upstream's own code on a one-node Tree.
    v = jax.vmap(lambda t: (lambda w: (w._pick(ROOT, t.node_visits), w._pick(ROOT, t.node_values),
                                       w._pick(ROOT, t.reward_in), w._pick(ROOT, t.discount_in)))(
        NodeView2(t, A, a_pad, root_only=True, edge_rows=edge_rows)))(t)
    # The root prior goes through the barrier too, so the final qtransform does
    # not fuse with the masking that produced it (that changes rounding).
    rv, cv, cr, cd, root_prior = jax.lax.optimization_barrier(v + (t.root_prior_logits,))
    one = lambda x: x[:, None]
    root_tree = tree_lib.Tree(
        node_visits=t.node_visits[:, :1], raw_values=t.raw_values[:, :1],
        node_values=t.node_values[:, :1], parents=t.parents[:, :1],
        action_from_parent=t.action_from_parent[:, :1],
        children_index=jnp.where(one(rv) > 0, 0, UNVISITED).astype(jnp.int32),
        children_prior_logits=one(root_prior), children_values=one(cv),
        children_visits=one(rv), children_rewards=one(cr), children_discounts=one(cd),
        embeddings=None, root_invalid_actions=invalid_actions,
        extra_data=AS.GumbelMuZeroExtraData(root_gumbel=gumbel))
    root_visits = root_tree.summary().visit_counts
    considered_visit = jnp.max(root_visits, axis=-1, keepdims=True)
    completed_qvalues = jax.vmap(qtransform, in_axes=[0, None])(root_tree, root_tree.ROOT_INDEX)
    to_argmax = SH.score_considered(considered_visit, gumbel, root.prior_logits,
                                    completed_qvalues, root_visits)
    action = AS.masked_argmax(to_argmax, invalid_actions)
    completed_search_logits = _mask_invalid_actions(
        root.prior_logits + completed_qvalues, invalid_actions)
    tree = None
    if return_tree:
        priors = (t.prior_flat.reshape(batch, N, a_pad)[:, :, :A] if (flat_priors or tiled_priors)
                  else t.prior_flat)
        tree = node_to_tree(NodeTree(
            node_visits=t.node_visits, raw_values=t.raw_values, node_values=t.node_values,
            parents=t.parents, action_from_parent=t.action_from_parent, reward_in=t.reward_in,
            discount_in=t.discount_in, children_prior_logits=priors, embeddings=t.embeddings,
            root_invalid_actions=t.root_invalid_actions, root_gumbel=t.root_gumbel), A)
    return base.PolicyOutput(action=action, action_weights=jax.nn.softmax(completed_search_logits),
                             search_tree=tree)
