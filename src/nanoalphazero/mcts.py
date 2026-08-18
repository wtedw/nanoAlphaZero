"""One-round sequential-halving MCTS used by nanoAlphaZero."""

import functools
from typing import Any, Callable, Optional, Tuple

import chex
import jax
import jax.numpy as jnp

from nanoalphazero.buffers import unpack_bitmask_vmap


# =============================================================================
# MCTS
#
# Adapted from https://github.com/google-deepmind/mctx.
#
# "1sh": a custom search that runs a single round of Sequential Halving,
# batched across all root actions. Full MCTS expands nodes one at a time; 1sh
# does the whole round in two parallel network calls:
#   1. Evaluate all `num_root_considered` root actions at once.
#   2. Keep the better half (`num_survivors`), expand one child of each, and
#      pick the best action.
# The search budget is fixed by the two rungs.
#
# Since 1sh is a fixed-shape single round, several config knobs are UNUSED --
# placeholders for swapping in MCTX's full node-by-node MCTS later:
#   - mcts_num_simulations : node-expansion budget for the full search
#   - mcts_epsilon         : qtransform epsilon
#   - mcts_max_m           : max sampled actions at the root
#   - mcts_use_gumbel      : Gumbel-MuZero vs. regular MuZero
#   - mcts_variant         : which MCTX policy to dispatch to
# =============================================================================


# Parameters are an arbitrary nested structure of chex.Array.
Params = chex.ArrayTree
Action = chex.Array
RecurrentState = Any


@chex.dataclass(frozen=True)
class RecurrentFnOutput:
    """The output of a `RecurrentFn`.

    reward: `[B]` an approximate reward from the state-action transition.
    discount: `[B]` the discount between the `reward` and the `value`.
    prior_logits: `[B, num_actions]` the logits produced by a policy network.
    value: `[B]` an approximate value of the state after the state-action
      transition.
    """

    reward: chex.Array
    discount: chex.Array
    prior_logits: chex.Array
    value: chex.Array


@chex.dataclass(frozen=True)
class RootFnOutput:
    """The output of a representation network.

    prior_logits: `[B, num_actions]` the logits produced by a policy network.
    value: `[B]` an approximate value of the current state.
    embedding: `[B, ...]` the inputs to the next `recurrent_fn` call.
    """

    prior_logits: chex.Array
    value: chex.Array
    embedding: RecurrentState
    k_indices: Optional[Any] = None


RecurrentFn = Callable[
    [Params, chex.PRNGKey, Action, RecurrentState],
    Tuple[RecurrentFnOutput, RecurrentState],
]


@chex.dataclass(frozen=True)
class PolicyOutput:
    """The output of a policy.

    action: `[B]` the proposed action.
    action_weights: `[B, num_actions]` the targets used to train a policy network.
    """

    action: chex.Array
    action_weights: chex.Array

    # visit counts over actions, used by the selfplay exploration sampler.
    visit_counts: Optional[Any] = None

    # BNK compressed fields (populated when use_bnk=True)
    bnk_k_indices: Optional[Any] = None
    bnk_action_weights: Optional[Any] = None


# ─────────────────────────────────────────────────────────────────────────────
# Inlined helpers from mctx._src.action_selection
# ─────────────────────────────────────────────────────────────────────────────
def _mask_invalid_actions(logits, invalid_actions):
    """Returns logits with zero mass to invalid actions."""
    if invalid_actions is None:
        return logits
    chex.assert_equal_shape([logits, invalid_actions])
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    # At the end of an episode, all actions can be invalid. A softmax would then
    # produce NaNs, if using -inf for the logits. We avoid the NaNs by using
    # a finite `min_logit` for the invalid actions.
    min_logit = jnp.finfo(logits.dtype).min
    return jnp.where(invalid_actions, min_logit, logits)


def masked_argmax(to_argmax, invalid_actions):
    """Returns a valid action with the highest `to_argmax`."""
    if invalid_actions is not None:
        chex.assert_equal_shape([to_argmax, invalid_actions])
        to_argmax = jnp.where(invalid_actions, -jnp.inf, to_argmax)
    return jnp.argmax(to_argmax, axis=-1).astype(jnp.int32)


# ─────────────────────────────────────────────────────────────────────────────
# Inlined qtransform helpers from mctx._src.qtransforms (doc #1)
# ─────────────────────────────────────────────────────────────────────────────
def _rescale_qvalues(qvalues, epsilon):
    """Rescales the given completed Q-values to be from the [0, 1] interval."""
    min_value = jnp.min(qvalues, axis=-1, keepdims=True)
    max_value = jnp.max(qvalues, axis=-1, keepdims=True)
    return (qvalues - min_value) / jnp.maximum(max_value - min_value, epsilon)


def _complete_qvalues(qvalues, *, visit_counts, value):
    """Returns completed Q-values, with the `value` for unvisited actions."""
    chex.assert_equal_shape([qvalues, visit_counts])
    chex.assert_shape(value, [])

    # The missing qvalues are replaced by the value.
    completed_qvalues = jnp.where(visit_counts > 0, qvalues, value)
    chex.assert_equal_shape([completed_qvalues, qvalues])
    return completed_qvalues


def _compute_mixed_value(raw_value, qvalues, visit_counts, prior_probs):
    """Interpolates the raw_value and weighted qvalues."""
    sum_visit_counts = jnp.sum(visit_counts, axis=-1)
    # Ensuring non-nan weighted_q, even if the visited actions have zero
    # prior probability.
    prior_probs = jnp.maximum(jnp.finfo(prior_probs.dtype).tiny, prior_probs)
    # Summing the probabilities of the visited actions.
    sum_probs = jnp.sum(jnp.where(visit_counts > 0, prior_probs, 0.0), axis=-1)
    weighted_q = jnp.sum(
        jnp.where(
            visit_counts > 0,
            prior_probs * qvalues / jnp.where(visit_counts > 0, sum_probs, 1.0),
            0.0,
        ),
        axis=-1,
    )
    return (raw_value + sum_visit_counts * weighted_q) / (sum_visit_counts + 1)


def final_qtransform_completed_by_mix_value(
    root_qvalues,
    root_raw_value,
    root_prior_logits,
    layer1_visit_counts,
    *,
    value_scale: chex.Numeric = 1.0,
    maxvisit_init: chex.Numeric = 50.0,
    rescale_values: bool = False,
    use_mixed_value: bool = True,
    epsilon: chex.Numeric = 1e-8,
) -> chex.Array:
    """Returns the completed, transformed Q-values used to pick actions.

    The missing Q-values of the unvisited actions are replaced by the mixed
    value, defined in Appendix D of "Policy improvement by planning with
    Gumbel": https://openreview.net/forum?id=bERaNdoegnO

    The Q-values are transformed by a linear transformation:
      `(maxvisit_init + max(visit_counts)) * value_scale * qvalues`.
    """
    qvalues = root_qvalues
    visit_counts = layer1_visit_counts
    raw_value = root_raw_value
    prior_probs = jax.nn.softmax(root_prior_logits)

    # Computing the mixed value and producing completed_qvalues.
    mixed_value = _compute_mixed_value(
        raw_value, qvalues=qvalues, visit_counts=visit_counts, prior_probs=prior_probs
    )
    if use_mixed_value:
        value = mixed_value
    else:
        value = raw_value
    completed_qvalues = _complete_qvalues(
        qvalues, visit_counts=visit_counts, value=value
    )

    # Scaling the Q-values.
    rescaled_qvalues = _rescale_qvalues(completed_qvalues, epsilon)
    if rescale_values:
        completed_qvalues = rescaled_qvalues
    maxvisit = jnp.max(visit_counts, axis=-1)
    visit_scale = maxvisit_init + maxvisit

    return visit_scale * value_scale * completed_qvalues


# ─────────────────────────────────────────────────────────────────────────────
# Inlined fast-gather helpers from gumbel_muzero_policy
# ─────────────────────────────────────────────────────────────────────────────
def _fast_gather2d(x: jnp.ndarray, idx: jnp.ndarray) -> jnp.ndarray:
    """TPU-friendly replacement for `jnp.take_along_axis(x, idx, axis=1)` on
    rank-2 tensors.

    Parameters
    ----------
    x   : [B, N]  – values to gather from
    idx : [B, K]  – int32 / int64 row indices to take (axis 1)

    Returns
    -------
    out : [B, K]  – same as the Gather version
    """
    # one-hot mask: [B, K, N]   (B=batch, K=number of indices, N=source length)
    mask = jax.nn.one_hot(idx, x.shape[1], dtype=x.dtype)  # idx : [B, K]
    out = jnp.einsum("bkn,bn->bk", mask, x)  # result [B, K]
    return out


def _fast_gather_rows(x: jnp.ndarray, idx: jnp.ndarray) -> jnp.ndarray:
    """TPU-friendly replacement for

        jnp.take_along_axis(x, idx[..., None], axis=1)

    Works for **both** shapes

        x   : [B, N]                 (rank-2)
        x   : [B, N, F1, F2, …]      (rank ≥ 3)

    Returns out of shape [B, K, ...] – rows selected from `x`.  Trailing
    feature axes (`...`) are preserved if present.  For rank-2 input the
    result is [B, K].
    """
    if idx.ndim != 2 or idx.shape[0] != x.shape[0]:
        raise ValueError("`idx` must be [B, K] with the same batch size as `x`")
    if x.ndim < 2:
        raise ValueError("`x` must be rank ≥ 2 with the gather axis at pos 1")

    N = x.shape[1]
    # One-hot mask: [B, K, N]  (stored in x.dtype ⇒ keeps bf16/f32 throughput)
    mask = jax.nn.one_hot(idx, N, dtype=x.dtype)
    # Batched matmul:  mask[b, k, n] ⋅ x[b, n, …]  → out[b, k, …]
    out = jax.lax.dot_general(
        mask,
        x,
        (
            ((2,), (1,)),  # contract N-axis of mask with row-axis of x
            ((0,), (0,)),
        ),
    )  # keep batch axis
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main policy
# ─────────────────────────────────────────────────────────────────────────────
def gumbel_muzero_policy_1sh(
    params: Params,
    rng_key: chex.PRNGKey,
    root: RootFnOutput,
    recurrent_fn: RecurrentFn,
    *,
    num_root_considered: int = 16,  # first SH rung
    num_survivors: int = 8,  # second SH rung (num_root_considered // 2)
    gumbel_scale: chex.Numeric = 1.0,
    invalid_actions: Optional[chex.Array] = None,
    value_scale: chex.Numeric = 1.0,
    maxvisit_init: chex.Numeric = 50.0,
    rescale_values: bool = False,
    use_mixed_value: bool = True,
    epsilon: chex.Numeric = 1e-8,
    use_bnk: bool = False,
    num_k_actions: Optional[int] = None,
) -> PolicyOutput:
    """
    Sequential-Halving BFS (2 rungs).
    1.  Visit R = `num_root_considered` root actions once in parallel.
    2.  Keep the best S = `num_survivors`, visit *one* child of each of those once.
    3.  Back-up the two visits and pick the root move that maximises
        gumbel + prior + completed-Q.
    Every expansion is batched → friendly to TPU.

    Note: this policy has NO `num_simulations` knob — its search budget is fixed
    by the two rungs and controlled entirely by `num_root_considered` / `num_survivors`.
    """

    # ------------------------------------------------------------------------
    # 0) Root pre-processing
    # ------------------------------------------------------------------------
    root = root.replace(
        prior_logits=_mask_invalid_actions(root.prior_logits, invalid_actions)
    )
    B, A = root.prior_logits.shape
    R, S = num_root_considered, num_survivors  # SH rungs: R root actions, S survivors
    rng_key, g_root_key = jax.random.split(rng_key)
    root_gumbel = gumbel_scale * jax.random.gumbel(
        g_root_key, shape=root.prior_logits.shape, dtype=root.prior_logits.dtype
    )

    # ------------------------------------------------------------------------
    # 1) FIRST RUNG  –– expand R distinct root actions once
    # ------------------------------------------------------------------------
    #    score = g + logit  (initial completed-Q is 0 so it drops out)
    first_score = root_gumbel + root.prior_logits
    _, first_idx = jax.lax.top_k(first_score, R)  # [B, R]

    # Expand root in parallel -------------------------------------------------
    BxR = B * R
    root_flat_actions = first_idx.reshape(-1)

    rng_key, _rng = jax.random.split(rng_key)
    root_flat_keys = jax.random.split(_rng, BxR).reshape(BxR, -1)
    root_flat_embed = jax.tree.map(lambda x: jnp.repeat(x, R, axis=0), root.embedding)

    layer1_flat_out, layer1_flat_emb = recurrent_fn(
        params, root_flat_keys, root_flat_actions, root_flat_embed
    )

    def unflat(x):
        return x.reshape(B, R, *x.shape[1:])

    layer1_out = jax.tree.map(unflat, layer1_flat_out)  # [BxR,] -> [B, R]
    layer1_embeds = jax.tree.map(unflat, layer1_flat_emb)
    layer1_qvalues = layer1_out.reward + layer1_out.discount * layer1_out.value
    layer1_visits = jnp.ones_like(layer1_qvalues, dtype=jnp.int32)  # visits = 1

    # ------------------------------------------------------------------
    # 1-bis)  Mask out actions that are invalid at the root
    # ------------------------------------------------------------------
    if invalid_actions is not None:
        # valid_mask : 1 for legal actions, 0 for invalid
        layer1_valid_mask = 1 - _fast_gather2d(invalid_actions, first_idx)  # [B,R]

        layer1_qvalues = layer1_qvalues * layer1_valid_mask
        layer1_visits = layer1_visits * layer1_valid_mask.astype(
            layer1_qvalues.dtype
        )  # visits = 0 for invalid

        # actions that are invalid should never survive to rung-2
        # set their score to −inf so top_k ignores them
        layer1_score_mask = layer1_valid_mask == 0
    else:
        layer1_score_mask = jnp.zeros_like(layer1_qvalues, dtype=bool)

    # 1-fin) Calc the completed_qvalues
    def layer1_qtransform(q1):
        alpha = value_scale * (maxvisit_init + 1.0)  # same scale as paper
        if rescale_values:
            q_min = jnp.min(q1, axis=1, keepdims=True)
            q_max = jnp.max(q1, axis=1, keepdims=True)
            q_norm = (q1 - q_min) / jnp.maximum(q_max - q_min, epsilon)
        else:
            q_norm = q1  # no rescaling
        cq = alpha * q_norm  # completed-Q for the R parents
        return cq, q_norm

    # If we visit and the value is negative, we should pick that over invalid action
    # This will happen when we do top_k with masked_score
    layer1_cqvalues, _ = layer1_qtransform(layer1_qvalues)

    # ------------------------------------------------------------------------
    # 2) SECOND RUNG  –– keep best S roots, add one extra rollout inside each
    # ------------------------------------------------------------------------
    # score_after_1 = g + logit + q1
    score1 = (
        jnp.take_along_axis(root_gumbel, first_idx, -1)
        + jnp.take_along_axis(root.prior_logits, first_idx, -1)
        + layer1_cqvalues
    )

    masked_score1 = jnp.where(layer1_score_mask, -jnp.inf, score1)
    _, second_loc = jax.lax.top_k(masked_score1, S)  # [B, S] the idx within the R
    second_idx = _fast_gather2d(first_idx, second_loc)  # [B, S]

    # 2-b) which of those S parents were illegal to begin with? --------------
    illegal_parent = _fast_gather2d(layer1_score_mask, second_loc)  # [B,S] Bool

    # Expand *one child* of each of those S parents in layer 1 ---------------
    # Gather the chosen parents' logits so we can pick a child
    layer1_halved_logits = jnp.take_along_axis(
        layer1_out.prior_logits,  # [B,R,A]
        second_loc[..., None],
        1,
    )

    # completed-Q values for each of the S parents (all children unvisited → 0)
    layer1_halved_completed_q = jnp.zeros_like(layer1_halved_logits)

    # Apply the interior selection heuristic once, batched over [B,S]
    probs = jax.nn.softmax(layer1_halved_logits + layer1_halved_completed_q, axis=-1)
    to_argmax = probs  # since visits=0
    best_child = jnp.argmax(to_argmax, axis=-1).astype(jnp.int32)  # [B,S]

    # Flatten and expand those leaf actions
    BxS = B * S
    leaf_actions = best_child.reshape(-1)

    rng_key, key_leaf = jax.random.split(rng_key)
    leaf_keys = jax.random.split(key_leaf, BxS).reshape(BxS, -1)

    def gather_parents_leaf(x: jnp.ndarray) -> jnp.ndarray:
        """
        Pick the S survivors (rows indexed by `second_loc`) from the R parents
        and flatten to [B*S, …].  Works for rank-2 and rank-≥3 tensors.
        """
        picked = _fast_gather_rows(x, second_loc)  # [B, S, …] or [B, S]
        return picked.reshape(BxS, *x.shape[2:])  # flatten first two axes

    layer2_parent_emb_flat = jax.tree.map(gather_parents_leaf, layer1_embeds)

    flat2_out, _ = recurrent_fn(
        params, leaf_keys, leaf_actions, layer2_parent_emb_flat
    )  # [BxS]

    def unflat2(x):
        return x.reshape(B, S, *x.shape[1:])

    layer2 = jax.tree.map(unflat2, flat2_out)  # [B, S]

    # 2-e) compute q₂ only for *legal* parents --------------------------------
    q2_leaf = layer2.reward + layer2.discount * layer2.value  # [B,S]

    # ------------------------------------------------------------------
    # (a) rewards and discounts of the S survivors  (rank-2)
    # ------------------------------------------------------------------
    r1 = _fast_gather2d(layer1_out.reward, second_loc)  # [B,S]
    γ1 = _fast_gather2d(layer1_out.discount, second_loc)  # [B,S]
    q2_full = r1 + γ1 * q2_leaf  # [B,S]

    # ------------------------------------------------------------------
    # (b) q / visit counts from the first rung that correspond to the
    #     same S survivors                                      (rank-2)
    # ------------------------------------------------------------------
    q1_sel = _fast_gather2d(layer1_qvalues, second_loc)  # [B,S]
    v1_sel = _fast_gather2d(layer1_visits, second_loc)  # [B,S]

    # mask-out the illegal parents: keep their original q₁, no extra visit
    q2 = jnp.where(illegal_parent, 0.0, q2_full)  # [B,S]
    v2 = jnp.where(illegal_parent, 0, 1).astype(jnp.int32)  # [B,S]

    q_comb = (q1_sel * v1_sel + q2) / (v1_sel + v2 + 1e-6)  # [B,S]
    vcnt2 = v1_sel + v2  # [B,S]

    # ------------------------------------------------------------------------
    # 3) [optimized] Assemble per-action arrays for the root (q & visit-count)
    # ------------------------------------------------------------------------
    #  – layer-1 contribution ………………   first_idx,     layer1_qvalues / layer1_visits
    #  – layer-2 overwrite   ………………   second_idx,    q_comb        / vcnt2
    #    (second_idx ⊂ first_idx, so we "mask-away & add" to overwrite)

    # One-hot masks -----------------------------------------------------------
    mask1 = jax.nn.one_hot(first_idx, A)  # [B, K₁, A]
    mask2 = jax.nn.one_hot(second_idx, A)  # [B, K₂, A]

    # Σ mask * value  →  [B, A] -----------------------------
    q_l1 = jnp.sum(mask1 * layer1_qvalues[:, :, None], 1)  # first-rung q
    v_l1 = jnp.sum(mask1 * layer1_visits[:, :, None], 1)

    q_l2 = jnp.sum(mask2 * q_comb[:, :, None], 1)  # second-rung q
    v_l2 = jnp.sum(mask2 * vcnt2[:, :, None], 1)

    mask2_sum = jnp.sum(mask2, axis=1)  # [B, A]  1 on survivors

    # Overwrite: zero-out the survivors in layer-1 arrays,
    # then add layer-2 values -----------------------------------------------
    q_root = q_l1 * (1 - mask2_sum) + q_l2  # [B, A]
    visit_root = v_l1 * (1 - mask2_sum) + v_l2.astype(v_l1.dtype)

    # ------------------------------------------------------------------------
    # 4) Completed-Q transform & final root decision
    # ------------------------------------------------------------------------
    qtransform_fn = functools.partial(
        final_qtransform_completed_by_mix_value,
        value_scale=value_scale,
        maxvisit_init=maxvisit_init,
        rescale_values=rescale_values,
        use_mixed_value=use_mixed_value,
        epsilon=epsilon,
    )

    completed_q = jax.vmap(qtransform_fn, in_axes=[0, 0, 0, 0])(
        q_root, root.value, root.prior_logits, visit_root
    )

    final_score = root_gumbel + root.prior_logits + completed_q
    best_a = masked_argmax(final_score, invalid_actions)

    search_logits = root.prior_logits + completed_q

    # Final mask to ensure invalid actions are -inf
    search_logits = _mask_invalid_actions(search_logits, invalid_actions)
    action_weights = jax.nn.softmax(search_logits)

    # BNK compressed fields: top-k over the full search_logits so we capture
    # completed-Q info for all A actions (not just the num_root_considered explored ones).
    if use_bnk:
        _, bnk_k_indices = jax.lax.top_k(search_logits, k=num_k_actions)  # [B, K]
        k_search_logits = _fast_gather2d(search_logits, bnk_k_indices)  # [B, K]
        bnk_action_weights = jax.nn.softmax(k_search_logits)  # [B, K]
    else:
        bnk_k_indices = None
        bnk_action_weights = None

    return PolicyOutput(
        # --- decision & training targets ---
        action=best_a,  # int32  [B]
        action_weights=action_weights,  # float [B, A]
        visit_counts=visit_root,  # [B, A]
        # BNK compressed fields (populated when use_bnk=True)
        bnk_k_indices=bnk_k_indices,  # [B, K1] or None
        bnk_action_weights=bnk_action_weights,  # [B, K1] or None
    )


def make_mcts(config, wenv, model, data_sharding=None):
    is_chess = config["env_id"] == "chess"
    if config.get("enable_sharding", False) and data_sharding is None:
        mesh = jax.sharding.Mesh(jax.devices(), "x")
        data_sharding = jax.sharding.NamedSharding(
            mesh, jax.sharding.PartitionSpec("x")
        )

    # custom pgx chess exposes legal as packed uint32 bitmask
    #     (legal_action_bitmask), not legal_action_mask. Unpack on read.
    def _legal_from_state(env_state):
        if is_chess:
            return unpack_bitmask_vmap(env_state.legal_action_bitmask)
        return env_state.legal_action_mask

    def get_root_fn(params):
        def root_fn(env_state, _rng_key: chex.PRNGKey) -> RootFnOutput:
            obs = wenv.observe(env_state, env_state.current_player)
            legal = _legal_from_state(env_state)
            # pin obs/legal to data-parallel sharding
            if config.get("enable_sharding", False):
                obs = jax.lax.with_sharding_constraint(obs, data_sharding)
                legal = jax.lax.with_sharding_constraint(legal, data_sharding)
            model_state = {"params": params}
            prior_logits, value = model.apply(model_state, obs, legal)
            # KataGo root-temperature softening (good for chess)
            if config.get("exp_use_root_temperature", False):
                tau = config.get("exp_root_temperature", 1.3)
                prior_logits = jnp.where(
                    legal, prior_logits / tau, jnp.finfo(prior_logits.dtype).min
                )
            return RootFnOutput(
                prior_logits=prior_logits,
                value=value,
                embedding=env_state,
            )

        return root_fn

    def get_recurrent_fn():
        def recurrent_fn(params, rng_key, action, env_state):
            action = jnp.asarray(action, dtype=jnp.int32)
            prev_player = env_state.current_player

            if config.get("enable_sharding", False):
                action = jax.lax.with_sharding_constraint(
                    action, data_sharding
                )
                prev_player = jax.lax.with_sharding_constraint(
                    prev_player, data_sharding
                )

            # 1sh uses a single rng_key (not batched)
            env_state = wenv.autostep(env_state, action, rng_key)

            obs = wenv.observe(env_state, env_state.current_player)
            legal = _legal_from_state(env_state)
            model_state = {"params": params}
            prior_logits, value = model.apply(model_state, obs, legal)

            B = env_state.rewards.shape[0]
            reward = env_state.rewards[jnp.arange(B), prev_player]
            discount = jnp.where(env_state.terminated, 0, -1).astype(jnp.float32)
            final_value = jnp.where(env_state.terminated, 0, value).astype(jnp.float32)

            recurrent_fn_output = RecurrentFnOutput(
                reward=reward,
                discount=discount,
                prior_logits=prior_logits,
                value=final_value,
            )
            return recurrent_fn_output, env_state

        return recurrent_fn

    @functools.partial(jax.jit, static_argnums=(4, 5))
    def run_mcts(
        rng_key: chex.PRNGKey,
        env_state,
        params,
        gumbel_scale,
        batch_size,
        num_simulations=config["mcts_num_simulations"],  # unused by 1sh; see note above
    ):
        key1, key2 = jax.random.split(rng_key)
        root_fn = get_root_fn(params)
        root = root_fn(env_state, jax.random.split(key2, batch_size))

        recurrent_fn = get_recurrent_fn()
        # chess uses bitmask + unpack; others use bool mask
        if is_chess:
            invalid_actions = ~unpack_bitmask_vmap(env_state.legal_action_bitmask)
        else:
            invalid_actions = ~env_state.legal_action_mask

        policy_output = gumbel_muzero_policy_1sh(
            params=params,
            invalid_actions=invalid_actions,
            rng_key=key2,
            root=root,
            recurrent_fn=recurrent_fn,
            gumbel_scale=gumbel_scale,
            value_scale=config["mcts_value_scale"],
            rescale_values=config["mcts_rescale_values"],
            maxvisit_init=config["mcts_maxvisit_init"],
            num_root_considered=config["mcts_num_root_considered"],
            num_survivors=config["mcts_num_survivors"],  # just num_root_considered // 2
            use_bnk=config.get("exp_bnk_action_weights", False),
            num_k_actions=config.get("mcts_num_k_actions", None),
        )

        return policy_output

    return run_mcts


# =============================================================================

