"""Exact sequential Gumbel MuZero with node-only storage.

Ported from wtedw/nanoAlphaZero 01c4e68 (Claude's node variant).
Selection and Q transforms use pinned upstream mctx, not the installed fork.
BNK target compression is applied by the caller after the dense search.
Upstream-derived masking/selection arithmetic is Apache-2.0 licensed;
see _vendor/mctx/LICENSE and PROVENANCE (Copyright 2021 DeepMind Technologies).
"""
import functools
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import chex

from nanoalphazero._vendor.mctx import action_selection as AS
from nanoalphazero._vendor.mctx import base
from nanoalphazero._vendor.mctx import qtransforms as QT
from nanoalphazero._vendor.mctx import seq_halving as SH
from nanoalphazero._vendor.mctx import tree as tree_lib

UNVISITED = -1
NO_PARENT = -1
ROOT = 0

def _mask_invalid_actions(logits, invalid_actions):
    """Upstream mctx masking, including finite all-invalid terminal policies."""
    if invalid_actions is None:
        return logits
    chex.assert_equal_shape([logits, invalid_actions])
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    min_logit = jnp.finfo(logits.dtype).min
    return jnp.where(invalid_actions, min_logit, logits)


class _Indexer:
    def __init__(self, fn):
        self.fn = fn

    def __getitem__(self, index):
        return self.fn(index)


def _selectors(num_simulations, max_num_considered_actions, qtransform):
    root = functools.partial(
        AS.gumbel_muzero_root_action_selection, num_simulations=num_simulations,
        max_num_considered_actions=max_num_considered_actions, qtransform=qtransform)
    interior = functools.partial(AS.gumbel_muzero_interior_action_selection,
                                 qtransform=qtransform)
    return root, interior


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
    def __init__(self, t, num_actions):
        self.t, self.A = t, num_actions
        self.children_prior_logits = t.children_prior_logits
        self.raw_values = t.raw_values
        self.root_invalid_actions = t.root_invalid_actions
        self.extra_data = AS.GumbelMuZeroExtraData(root_gumbel=t.root_gumbel)
        self.children_visits = _Indexer(lambda n: self._pick(n, self.t.node_visits))

    def _match(self, node):
        actions = jnp.arange(self.A, dtype=self.t.action_from_parent.dtype)
        return (self.t.parents == node)[None, :] & (self.t.action_from_parent[None, :] == actions[:, None])

    def _pick(self, node, x):
        # Exactly one match per visited action; sums of a single term are exact.
        return jnp.sum(jnp.where(self._match(node), x[None, :], jnp.zeros((), x.dtype)), axis=-1)

    def qvalues(self, node):
        return (self._pick(node, self.t.reward_in)
                + self._pick(node, self.t.discount_in) * self._pick(node, self.t.node_values))


def _node_child(t, node, action):
    hit = (t.parents == node) & (t.action_from_parent == action)
    index = jnp.arange(t.parents.shape[0], dtype=jnp.int32)
    return jnp.where(jnp.any(hit), jnp.sum(jnp.where(hit, index, 0)), UNVISITED)


def _node_simulate(t, A, root_select, interior_select, max_depth):
    view = NodeView(t, A)

    def cond(state):
        return state[4]

    def body(state):
        _, _, node, depth, _ = state
        action = jax.lax.cond(depth == 0, lambda: root_select(None, view, node),
                              lambda: interior_select(None, view, node, depth))
        next_node = _node_child(t, node, action)
        depth = depth + 1
        return node, action, next_node, depth, jnp.logical_and(next_node != UNVISITED, depth < max_depth)

    state = (jnp.int32(NO_PARENT), jnp.int32(NO_PARENT), jnp.int32(ROOT),
             jnp.zeros((), t.children_prior_logits.dtype), jnp.array(True))
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
    max_num_considered_actions=16, gumbel_scale=1.0, return_tree=True):
    """Upstream-equivalent Gumbel MuZero without large scatters.

    With return_tree=False retain only the root summary, for visits/diagnostics.
    Truncated-depth search is deliberately unsupported by this node-only port.
    """
    if max_depth is not None and max_depth < num_simulations:
        raise ValueError("Node-only search requires max_depth >= num_simulations")
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
        children_prior_logits=jnp.zeros((batch, N, A), root.prior_logits.dtype),
        embeddings=jax.tree.map(zeros, root.embedding),
        root_invalid_actions=invalid_actions, root_gumbel=gumbel)
    put = lambda x, v, node: jax.lax.dynamic_update_index_in_dim(x, v.astype(x.dtype), node, axis=1)
    t = t._replace(
        children_prior_logits=put(t.children_prior_logits, root.prior_logits, 0),
        raw_values=put(t.raw_values, root.value, 0), node_values=put(t.node_values, root.value, 0),
        node_visits=put(t.node_visits, t.node_visits[:, 0] + 1, 0),
        embeddings=jax.tree.map(lambda x, v: put(x, v, 0), t.embeddings, root.embedding))
    simulate = jax.vmap(_node_simulate, in_axes=(0, None, None, None, None))

    def body(sim, carry):
        key, t = carry
        key, _, expand_key = jax.random.split(key, 3)
        parent, action = simulate(t, A, root_select, interior_select, depth_limit)
        node = sim + 1
        # Batch the node lookup explicitly. Global advanced indexing with
        # x[arange(batch), parent] makes SPMD gather/reduce entire latent batches
        # across hosts; each game's parent is actually local to its batch shard.
        embedding = jax.tree.map(
            lambda x: jax.vmap(lambda nodes, index: jax.lax.dynamic_index_in_dim(
                nodes, index, axis=0, keepdims=False))(x, parent), t.embeddings)
        step, embedding = recurrent_fn(params, expand_key, action, embedding)
        new_visit = jax.lax.dynamic_index_in_dim(t.node_visits, node, axis=1, keepdims=False) + 1
        t = t._replace(
            children_prior_logits=put(t.children_prior_logits, step.prior_logits, node),
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
        NodeView(t, A)._pick(ROOT, t.reward_in), NodeView(t, A)._pick(ROOT, t.discount_in)))(t)
    rv, cv, cr, cd = jax.lax.optimization_barrier(view_rows)
    one = lambda x: x[:, None]
    root_tree = tree_lib.Tree(
        node_visits=t.node_visits[:, :1], raw_values=t.raw_values[:, :1],
        node_values=t.node_values[:, :1], parents=t.parents[:, :1],
        action_from_parent=t.action_from_parent[:, :1],
        children_index=jnp.where(one(rv) > 0, 0, UNVISITED).astype(jnp.int32),
        children_prior_logits=t.children_prior_logits[:, :1], children_values=one(cv),
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
                             search_tree=node_to_tree(t, A) if return_tree else root_tree)


def search(params, key, root, recurrent_fn, *, num_simulations,
           num_root_considered, invalid_actions, gumbel_scale,
           value_scale=1.0, maxvisit_init=50.0, rescale_values=False,
           return_diagnostics=False):
    """Adapt exact search to the training output; leave target compression outside."""
    from nanoalphazero.mcts import PolicyOutput

    qtransform = functools.partial(QT.qtransform_completed_by_mix_value,
        value_scale=value_scale, maxvisit_init=maxvisit_init, rescale_values=rescale_values)
    result = gumbel_muzero_policy_node(
        params, key, root, recurrent_fn, num_simulations,
        invalid_actions=invalid_actions, qtransform=qtransform,
        max_num_considered_actions=num_root_considered, gumbel_scale=gumbel_scale,
        return_tree=False)
    tree = result.search_tree
    visits = tree.children_visits[:, 0]
    diagnostics = None
    if return_diagnostics:
        priors = tree.children_prior_logits[:, 0]
        q = jax.vmap(lambda t: t.qvalues(ROOT))(tree)
        mixed = jax.vmap(QT._compute_mixed_value)(
            root.value, q, visits, jax.nn.softmax(priors))
        completed = jnp.where(visits > 0, q, mixed[:, None])
        transformed = jax.vmap(qtransform, in_axes=(0, None))(tree, ROOT)
        gumbel = tree.extra_data.root_gumbel
        diagnostics = dict(
            prior_logit=priors, gumbel=gumbel, backed_up_q=q,
            completed_q=completed, transformed_q=transformed,
            advantage=completed - mixed[:, None],
            final_score=SH.score_considered(jnp.max(visits, -1, keepdims=True),
                gumbel, priors, transformed, visits),
            search_logit=_mask_invalid_actions(priors + transformed, invalid_actions),
            child_value=tree.children_values[:, 0], raw_value=root.value,
            mixed_value=mixed)
    return PolicyOutput(action=result.action, action_weights=result.action_weights,
                        visit_counts=visits, diagnostics=diagnostics)

