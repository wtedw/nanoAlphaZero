# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pgx @ git+https://github.com/wtedw/pgx.git@dcb18cb",
#     "flashbax @ git+https://github.com/instadeepai/flashbax.git@e0199d7bb232c622a19d3c28f9d6b34eb8215eab",
#     "flax==0.10.1",
#     "optax==0.2.7",
#     "chex==0.1.91",
#     "jax[tpu]==0.8.1",
#     "numpy",
# ]
# ///

# =============================================================================
# Imports
# =============================================================================
import functools
import os
import pickle
import re
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Generic, NamedTuple, Optional, Tuple

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pgx
from flax.training import train_state
from jax import Array
from jax.scipy.special import erf
from jax.typing import ArrayLike
import flashbax as fbx
from flashbax.buffers.trajectory_buffer import (
    TrajectoryBuffer,
    Experience,
    TrajectoryBufferState,
)
from pgx.experimental import auto_reset

P = jax.sharding.PartitionSpec
mesh = jax.sharding.Mesh(jax.devices(), "x")
DATA_PARALLEL_SHARDING = jax.sharding.NamedSharding(mesh, P("x"))
REPLICATED_SHARDING = jax.sharding.NamedSharding(mesh, P())


# =============================================================================
# Data types
# =============================================================================
@chex.dataclass(frozen=True)
class SelfplayOutput:
    col_id: ArrayLike
    row_id: ArrayLike
    global_step_id: Optional[ArrayLike]
    game_id: Optional[ArrayLike]
    action: ArrayLike
    action_weights: ArrayLike
    reward: ArrayLike
    is_valid_sample: ArrayLike
    is_from_selfplay: ArrayLike
    player: Optional[ArrayLike]
    just_terminated: Optional[ArrayLike]
    ep_step: Optional[ArrayLike]
    ep_termination_step: Optional[ArrayLike]
    is_exploration: Optional[ArrayLike]
    is_pending_reward_i8: Optional[ArrayLike]
    is_fresh_i8: Optional[ArrayLike]
    k_indices: Optional[ArrayLike] = None
    observation: Optional[ArrayLike] = None
    legal_action_mask: Optional[ArrayLike] = None
    # compressed chess fields
    board_bool: Optional[ArrayLike] = None
    board_float: Optional[ArrayLike] = None
    legal_action_bitmask: Optional[ArrayLike] = None


class SelfplayState(NamedTuple):
    env_state: Any
    ep_step: chex.Array
    switch_step: chex.Array
    step_count: int
    next_game_id: int


class CustomTrainState(train_state.TrainState):
    key: jax.Array = field(default_factory=lambda: jax.random.PRNGKey(0))
    n_updates: int = 0


class RunnerState(NamedTuple):
    model_ts: CustomTrainState
    selfplay_state: SelfplayState
    selfplay_buffer_state: TrajectoryBufferState
    replay_buffer_state: TrajectoryBufferState
    rng: chex.PRNGKey


# =============================================================================
# Neural network model
# =============================================================================
class DynamicTanh(nn.Module):
    features: int
    init_alpha: float = 0.5

    @nn.compact
    def __call__(self, x):
        alpha = self.param(
            "alpha", lambda rng, shape: self.init_alpha * jnp.ones(shape), (1,)
        )
        gamma = self.param(
            "gamma", lambda rng, shape: jnp.ones(shape), (self.features,)
        )
        beta = self.param("beta", lambda rng, shape: jnp.zeros(shape), (self.features,))
        y = jnp.tanh(alpha * x)
        return gamma * y + beta


class Derf(nn.Module):
    features: int
    init_alpha: float = 0.5
    init_shift: float = 0.0

    @nn.compact
    def __call__(self, x):
        alpha = self.param(
            "alpha", lambda rng, shape: self.init_alpha * jnp.ones(shape), (1,)
        )
        shift = self.param(
            "shift", lambda rng, shape: self.init_shift * jnp.ones(shape), (1,)
        )
        gamma = self.param(
            "gamma", lambda rng, shape: jnp.ones(shape), (self.features,)
        )
        beta = self.param("beta", lambda rng, shape: jnp.zeros(shape), (self.features,))
        y = erf(alpha * x + shift)
        return gamma * y + beta

class KataValueHeadGPool(nn.Module):
    """Global pooling over spatial positions: mean + max.

    Input  x: [B, H, W, C]  ->  out: [B, 2C]
    """

    @nn.compact
    def __call__(self, x):
        layer_mean = jnp.mean(x, axis=(1, 2))  # [B, C]
        layer_max = jnp.max(x, axis=(1, 2))  # [B, C]
        return jnp.concatenate([layer_mean, layer_max], axis=-1)  # [B, 2C]

class ReZeroBlock(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        identity = x
        y = nn.Conv(self.features, (3, 3), padding="SAME", use_bias=False)(x)
        y = nn.relu(y)
        y = nn.Conv(self.features, (3, 3), padding="SAME", use_bias=False)(y)
        alpha = self.param("alpha", nn.initializers.zeros, ())
        y = alpha * y
        return identity + y


class ConvModelReZero(nn.Module):
    action_space: int
    conv_width: int = 256
    conv_depth: int = 32
    use_derf: bool = False
    use_names: bool = False
    use_kata_gpool: bool = False

    @nn.compact
    def _trunk(self, obs):
        x = nn.Conv(
            self.conv_width, (3, 3), padding="SAME", use_bias=True, name="input_conv"
        )(obs)
        x = nn.relu(x)
        for _ in range(self.conv_depth):
            x = ReZeroBlock(self.conv_width)(x)
        return x

    @nn.compact
    def __call__(self, obs, valid, deterministic: bool = False):
        NormLayer = Derf if self.use_derf else DynamicTanh

        x = self._trunk(obs)

        p = nn.Conv(2, (1, 1), use_bias=False)(x)
        p = NormLayer(2)(p)
        p = nn.relu(p)
        p = p.reshape(p.shape[0], -1)
        logits = nn.Dense(
            self.action_space,
            kernel_init=nn.initializers.normal(stddev=1e-2),
            bias_init=nn.initializers.zeros,
            name="policy_out" if self.use_names else None,
        )(p)
        masked_logits = jnp.where(valid, logits, jnp.finfo(logits.dtype).min)

        # --- Value head ---
        if self.use_kata_gpool:
            v = nn.Conv(16, (1, 1), use_bias=False)(x)
            v = NormLayer(16)(v)
            v = nn.relu(v)
            v = KataValueHeadGPool()(v)  # [B, 48]
        else:
            v = nn.Conv(1, (1, 1), use_bias=False)(x)
            v = NormLayer(1)(v)
            v = nn.relu(v)
            v = v.reshape(v.shape[0], -1)

        v = nn.Dense(self.conv_width, name="value_dense" if self.use_names else None)(v)
        v = nn.relu(v)
        value = nn.Dense(
            1,
            kernel_init=nn.initializers.normal(stddev=1e-2),
            bias_init=nn.initializers.zeros,
            name="value_out" if self.use_names else None,
        )(v).squeeze(-1)
        value = jnp.tanh(value)

        return masked_logits, value

    def sample(self, logits, key, test: bool = False):
        return (
            jnp.argmax(logits, axis=-1) if test else jax.random.categorical(key, logits)
        )


def make_model(config, rng, sharding=None):
    conv_width = config["conv_width"]
    conv_depth = config["conv_depth"]
    observation_space = config["game_obs_shape"]
    action_space = config["game_num_actions"]
    use_derf = config.get("conv_use_derf", False)

    model = ConvModelReZero(
        action_space=action_space,
        conv_width=conv_width,
        conv_depth=conv_depth,
        use_derf=use_derf,
        use_names=config["conv_use_names"],
        use_kata_gpool=config.get("conv_use_kata_gpool", False),
    )

    observation = jnp.zeros((1,) + observation_space)
    valid_action_mask = jnp.ones((1, action_space), dtype=bool)
    # sharded init path (params replicated across mesh)
    model_state = init_and_shard_model(
        config, model, rng, observation, valid_action_mask, sharding
    )

    return model, model_state


# shards model params as REPLICATED across the mesh when enabled
def init_and_shard_model(config, model, rng, obs, valid_mask, sharding):
    use_bf16 = config.get("use_bf16", False)

    def _init_fn(rng, obs, mask):
        variables = model.init(rng, obs, mask)
        if use_bf16:
            variables = jax.tree_util.tree_map(
                lambda x: x.astype(jnp.bfloat16), variables
            )
        return variables

    if not config.get("enable_sharding", False) or sharding is None:
        return jax.jit(_init_fn)(rng, obs, valid_mask)

    abstract_variables = jax.eval_shape(_init_fn, rng, obs, valid_mask)
    sharding_tree = jax.tree_util.tree_map(lambda _: sharding, abstract_variables)
    sharded_init = jax.jit(_init_fn, out_shardings=sharding_tree)
    with sharding.mesh:
        model_state = sharded_init(rng, obs, valid_mask)
    return model_state


# =============================================================================
# Environment
# =============================================================================
class WrappedEnv:
    def __init__(
        self,
        obs_shape,
        num_actions,
        init_fn,
        step_fn,
        autostep_fn,
        *,
        init_dummy_estate_fn,
        single_estate,
        observe_fn=None,
    ):
        self.obs_shape = obs_shape
        self.num_actions = num_actions
        self.init = init_fn
        self.step = step_fn
        self.autostep = autostep_fn
        self.init_dummy_estate = init_dummy_estate_fn
        self.single_estate = single_estate
        self.observe = observe_fn

    def __repr__(self) -> str:
        return f"WrappedEnv(obs_shape={self.obs_shape}, num_actions={self.num_actions})"


def make_env(config):
    env_id = config["env_id"]
    env = pgx.make(env_id)
    e_step = env.step
    a_step = auto_reset(e_step, env.init)
    vmap_env_init = jax.jit(jax.vmap(env.init))
    vmap_env_step = jax.jit(jax.vmap(e_step))
    vmap_auto_step = jax.jit(jax.vmap(a_step))

    single_estate = env.init(jax.random.PRNGKey(0))

    def init_dummy_estate(batch_size: int):
        rng_key = jax.random.PRNGKey(0)
        rng_keys = jax.random.split(rng_key, batch_size)
        return vmap_env_init(rng_keys)

    batch_size = 1
    keys = jax.random.split(jax.random.PRNGKey(42), batch_size)
    env_state = vmap_env_init(keys)

    vmap_observe_fn = jax.jit(jax.vmap(env.observe))
    es_obs = vmap_observe_fn(env_state, env_state.current_player)

    pgx_num_actions = env.num_actions
    pgx_obs_shape = jnp.squeeze(es_obs, axis=0).shape

    return WrappedEnv(
        pgx_obs_shape,
        pgx_num_actions,
        vmap_env_init,
        vmap_env_step,
        vmap_auto_step,
        init_dummy_estate_fn=init_dummy_estate,
        observe_fn=vmap_observe_fn,
        single_estate=single_estate,
    )


# =============================================================================
# 1sh MCTS policy (standalone Sequential-Halving BFS, inlined)
# =============================================================================

# ── Standalone "1sh" Sequential-Halving BFS policy (no MCTX dependency) ──

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

    ### everything down below is for diagnostics / compression
    search_logits: Optional[Any] = None

    # root-level diagnostics
    children_values: Optional[Any] = None
    visit_counts: Optional[Any] = None
    root_gumbel: Optional[Any] = None
    root_prior_logits: Optional[Any] = None

    # layer 1 diagnostics
    layer1_value: Optional[Any] = None

    # after q-transform
    final_qvalues: Optional[Any] = None
    final_score: Optional[Any] = None
    advantages: Optional[Any] = None

    # internals of q-transform
    raw_value: Optional[Any] = None
    mixed_value: Optional[Any] = None
    maxvisit: Optional[Any] = None
    rescaled_qvalues: Optional[Any] = None
    rescaled_qvalues2: Optional[Any] = None

    # BNK compressed fields
    bnk_k_indices: Optional[Any] = None
    k_root_prior_logits: Optional[Any] = None
    k_search_logits: Optional[Any] = None
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
) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
    """Returns completed qvalues.

    The missing Q-values of the unvisited actions are replaced by the mixed
    value, defined in Appendix D of "Policy improvement by planning with
    Gumbel": https://openreview.net/forum?id=bERaNdoegnO

    The Q-values are transformed by a linear transformation:
      `(maxvisit_init + max(visit_counts)) * value_scale * qvalues`.

    Returns a tuple `(raw_value, mixed_value, maxvisit, rescaled_qvalues,
    original_res)` to expose intermediates for diagnostics.
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

    original_res = visit_scale * value_scale * completed_qvalues
    return (raw_value, mixed_value, maxvisit, rescaled_qvalues, original_res)


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
    The `mcts_num_simulations` config field is unused here (it's reserved for the
    future MCTX full-MCTS policy).
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
    layer1_cqvalues, rescaled_qvalues1 = layer1_qtransform(layer1_qvalues)

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

    (raw_value, mixed_value, maxvisit, rescaled_qvalues2, completed_q) = jax.vmap(
        qtransform_fn, in_axes=[0, 0, 0, 0]
    )(q_root, root.value, root.prior_logits, visit_root)

    final_score = root_gumbel + root.prior_logits + completed_q
    best_a = masked_argmax(final_score, invalid_actions)

    # these are not really "advantages" for the 1sh policy.
    # we populate the advantages field for wandb graphs
    final_advantages = completed_q
    search_logits = root.prior_logits + completed_q

    # Final mask to ensure invalid actions are -inf
    search_logits = _mask_invalid_actions(search_logits, invalid_actions)
    action_weights = jax.nn.softmax(search_logits)

    # BNK compressed fields: top-k over the full search_logits so we capture
    # completed-Q info for all A actions (not just the num_root_considered explored ones).
    if use_bnk:
        _, bnk_k_indices = jax.lax.top_k(search_logits, k=num_k_actions)  # [B, K]
        k_root_prior_logits = _fast_gather2d(root.prior_logits, bnk_k_indices)  # [B, K]
        k_search_logits = _fast_gather2d(search_logits, bnk_k_indices)  # [B, K]
        bnk_action_weights = jax.nn.softmax(k_search_logits)  # [B, K]
    else:
        bnk_k_indices = None
        k_root_prior_logits = None
        k_search_logits = None
        bnk_action_weights = None

    # Reshape layer1_value for debugging
    layer1_value_full = jnp.sum(mask1 * layer1_out.value[:, :, None], 1)
    rescaled_qvalues1 = jnp.sum(mask1 * rescaled_qvalues1[:, :, None], 1)

    return PolicyOutput(
        # --- decision & training targets ---
        action=best_a,  # int32  [B]
        action_weights=action_weights,  # float [B, A]
        # --- convenience for analysis ---
        search_logits=search_logits,  # [B, A]
        # --- root-level diagnostics ---
        children_values=q_root,  # [B, A]
        visit_counts=visit_root,  # [B, A]
        root_gumbel=root_gumbel,  # [B, A]
        root_prior_logits=root.prior_logits,  # [B, A]
        # layer 1 diagnostics
        layer1_value=layer1_value_full,  # [B, A]
        # after q-transform
        final_qvalues=completed_q,  # [B, A]
        final_score=final_score,  # [B, A]
        advantages=final_advantages,
        # internals of q-transform
        raw_value=raw_value,  # [B]
        mixed_value=mixed_value,  # [B]
        maxvisit=maxvisit,  # [B]
        rescaled_qvalues=rescaled_qvalues1,  # [B, A]  (optional)
        rescaled_qvalues2=rescaled_qvalues2,  # [B, A]  (optional)
        # BNK compressed fields (populated when use_bnk=True)
        bnk_k_indices=bnk_k_indices,  # [B, K1] or None
        k_root_prior_logits=k_root_prior_logits,  # [B, K1] or None
        k_search_logits=k_search_logits,  # [B, K1] or None
        bnk_action_weights=bnk_action_weights,  # [B, K1] or None
    )


# =============================================================================
# MCTS
# =============================================================================
def make_mcts(config, wenv, model):
    is_chess = config["env_id"] == "chess"

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
                obs = jax.lax.with_sharding_constraint(obs, DATA_PARALLEL_SHARDING)
                legal = jax.lax.with_sharding_constraint(legal, DATA_PARALLEL_SHARDING)
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
                    action, DATA_PARALLEL_SHARDING
                )
                prev_player = jax.lax.with_sharding_constraint(
                    prev_player, DATA_PARALLEL_SHARDING
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

    # We run a custom "1sh" search (gumbel_muzero_policy_1sh, inlined below)
    # instead of the full MCTX MCTS. The full MCTS expands nodes one at a time,
    # sequentially. What we implement inline is a fast, fully-parallel version of
    # what that MCTS would do if it ran a single round of Sequential Halving:
    #   - SH always explores `num_root_considered` unique actions, so we
    #     evaluate all of them in parallel in one batched network call.
    #   - We then discard the worse half, keep the survivors (`num_survivors`),
    #     expand exactly one inner node per survivor (again in parallel), and
    #     pick the best action.
    # This approximates one round of the full MCTS and empirically works very
    # well -- well enough to train chess models with it. The search budget is
    # fixed by the two rungs (num_root_considered / num_survivors).
    #
    # Because 1sh is this fixed-shape single round, several MCTS config knobs are
    # currently UNUSED -- they are placeholders for when we swap in MCTX's full
    # node-by-node MCTS, which will use them:
    #   - mcts_num_simulations : node-expansion budget for the full search
    #   - mcts_epsilon         : for qtransform
    #   - mcts_max_m           : max number of sampled actions at the root
    #   - mcts_use_gumbel      : toggle Gumbel-MuZero / regular MuZero
    #   - mcts_variant         : which MCTX policy to dispatch to
    # They are kept in the configs so the values are ready when that lands.
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
# Replay & self-play buffers
# =============================================================================
@chex.dataclass(frozen=True)
class CustomTrajectoryBufferState(TrajectoryBufferState[Experience]):
    num_valid_consumable: jax.Array = 0


@dataclass(frozen=True)
class Buffer(TrajectoryBuffer, Generic[Experience]):
    add_backfill: Optional[
        Callable[
            [CustomTrajectoryBufferState[Experience], Experience, Array, Array],
            tuple[CustomTrajectoryBufferState[Experience], dict],
        ]
    ] = None
    consume: Optional[
        Callable[
            [CustomTrajectoryBufferState[Experience]],
            tuple[CustomTrajectoryBufferState[Experience], Experience, dict],
        ]
    ] = None


def get_dummy_selfplay_output(config) -> SelfplayOutput:
    num_actions = config["game_num_actions"]
    obs_shape = config["game_obs_shape"]  # patched by make_alphazero from the live env
    is_chess = config["env_id"] == "chess"
    # build a kwargs dict so chess/bnk variants can override fields
    common = dict(
        col_id=jnp.zeros([], dtype=jnp.uint32),
        row_id=jnp.zeros([], dtype=jnp.uint32),
        global_step_id=jnp.zeros([], dtype=jnp.uint32),
        game_id=jnp.zeros([], dtype=jnp.uint32),
        action=jnp.zeros([], dtype=jnp.int32),
        action_weights=jnp.zeros((num_actions,), dtype=jnp.float32),
        reward=jnp.zeros([], dtype=jnp.float32),
        is_from_selfplay=jnp.zeros([], dtype=jnp.bool_),
        player=jnp.full([], -1, dtype=jnp.int32),
        just_terminated=jnp.zeros([], dtype=jnp.bool_),
        ep_step=jnp.full([], -127, dtype=jnp.int16),
        ep_termination_step=jnp.full([], 0, dtype=jnp.int16),
        is_exploration=jnp.zeros([], dtype=jnp.bool_),
        is_pending_reward_i8=jnp.ones([], dtype=jnp.int8),
        is_fresh_i8=jnp.zeros([], dtype=jnp.int8),
        is_valid_sample=jnp.zeros([], dtype=jnp.bool_),
    )
    # bnk stores (K,) policy targets + k_indices instead of full (A,)
    if config.get("exp_bnk_action_weights", False):
        k = config["mcts_num_k_actions"]
        common["action_weights"] = jnp.zeros((k,), dtype=jnp.float32)
        common["k_indices"] = jnp.zeros((k,), dtype=jnp.int32)
    # chess stores compressed obs + bitmask; others store raw obs/mask
    if is_chess:
        common["board_bool"] = jnp.zeros((936,), dtype=jnp.uint8)
        common["board_float"] = jnp.zeros((8, 8, 2), dtype=jnp.bfloat16)
        common["legal_action_bitmask"] = jnp.zeros((NUM_WORDS,), dtype=jnp.uint32)
    else:
        common["observation"] = jnp.zeros(obs_shape, dtype=jnp.bool_)
        common["legal_action_mask"] = jnp.zeros((num_actions,), dtype=jnp.bool_)
    return SelfplayOutput(**common)


def make_replay_buffer(config, dummy_selfplay_output, data_sharding=None):
    # The replay buffer holds finished training samples: positions whose final
    # reward has already been filled in (backfilled from the game's outcome).
    # Phase 3 samples gradient batches from here. Nothing in this buffer is
    # "in progress" -- by the time a sample lands here it is complete and
    # trainable. It is fed by draining the selfplay buffer (see below).
    replay_buffer = fbx.make_trajectory_buffer(
        add_batch_size=config["replay_buffer_add_batch_size"],
        sample_batch_size=config["replay_buffer_sample_batch_size"],
        sample_sequence_length=1,
        period=1,
        min_length_time_axis=config["replay_buffer_min_len"],
        max_length_time_axis=config["replay_buffer_max_len"],
    )
    replay_buffer = replay_buffer.replace(
        add=jax.jit(replay_buffer.add, donate_argnums=0),
        can_sample=jax.jit(replay_buffer.can_sample),
    )

    # data-parallel sharded init + sample, replicated for scalars
    if config.get("enable_sharding", False) and data_sharding is not None:
        sample_fn = jax.jit(replay_buffer.sample, out_shardings=data_sharding)
        state_shape_tree = jax.eval_shape(replay_buffer.init, dummy_selfplay_output)

        def _spec(shape_struct):
            return data_sharding if shape_struct.ndim > 0 else REPLICATED_SHARDING

        out_sharding_tree = jax.tree_util.tree_map(_spec, state_shape_tree)
        init_fn = jax.jit(replay_buffer.init, out_shardings=out_sharding_tree)
        with data_sharding.mesh:
            replay_buffer_state = init_fn(dummy_selfplay_output)
    else:
        sample_fn = jax.jit(replay_buffer.sample)
        init_fn = jax.jit(replay_buffer.init)
        replay_buffer_state = init_fn(dummy_selfplay_output)

    buffer = Buffer(
        init=init_fn,
        add=replay_buffer.add,
        sample=sample_fn,
        can_sample=replay_buffer.can_sample,
    )
    return buffer, replay_buffer_state


def make_selfplay_buffer(config, dummy_selfplay_output, data_sharding=None):
    # The selfplay buffer is a staging area for games that are still in progress.
    # Positions are written here as games are played, but they don't yet have a
    # reward -- `add_backfill` fills the reward in once the game terminates. Once
    # a position has its reward it is "consumed": handed off to the replay buffer
    # and marked is_fresh=False.
    #
    # The is_fresh flag is the key to correctness. is_fresh=True means "this
    # position just received its reward and has not been consumed yet". `consume`
    # only ever returns fresh positions and immediately flips them to
    # is_fresh=False. Without this, the same positions could be returned over and
    # over due to how top_k works.
    selfplay_buffer = fbx.make_trajectory_buffer(
        add_batch_size=config["selfplay_buffer_add_batch_size"],
        sample_batch_size=config["selfplay_buffer_sample_batch_size"],
        sample_sequence_length=1,
        period=1,
        min_length_time_axis=config["selfplay_buffer_min_len"],
        max_length_time_axis=config["selfplay_buffer_max_len"],
    )
    selfplay_buffer = selfplay_buffer.replace(
        add=jax.jit(selfplay_buffer.add, donate_argnums=0),
        sample=jax.jit(selfplay_buffer.sample),
        can_sample=jax.jit(selfplay_buffer.can_sample),
    )

    # sharded init for selfplay buffer state, REPLICATED scalar counter
    if config.get("enable_sharding", False) and data_sharding is not None:
        state_shape_tree = jax.eval_shape(selfplay_buffer.init, dummy_selfplay_output)

        def _spec(shape_struct):
            return data_sharding if shape_struct.ndim > 0 else REPLICATED_SHARDING

        out_sharding_tree = jax.tree_util.tree_map(_spec, state_shape_tree)
        init_fn = jax.jit(selfplay_buffer.init, out_shardings=out_sharding_tree)
        with data_sharding.mesh:
            selfplay_buffer_state = init_fn(dummy_selfplay_output)
            selfplay_buffer_state = CustomTrajectoryBufferState(
                experience=selfplay_buffer_state.experience,
                current_index=selfplay_buffer_state.current_index,
                is_full=selfplay_buffer_state.is_full,
                num_valid_consumable=jax.lax.with_sharding_constraint(
                    jnp.array(0, dtype=jnp.int32), REPLICATED_SHARDING
                ),
            )
    else:
        init_fn = jax.jit(selfplay_buffer.init)
        selfplay_buffer_state = init_fn(dummy_selfplay_output)
        selfplay_buffer_state = CustomTrajectoryBufferState(
            experience=selfplay_buffer_state.experience,
            current_index=selfplay_buffer_state.current_index,
            is_full=selfplay_buffer_state.is_full,
            num_valid_consumable=jnp.array(0, dtype=jnp.int32),
        )

    @functools.partial(jax.jit, donate_argnums=(0,))
    def add_backfill(
        selfplay_buffer_state,
        selfplay_output,
        env_state_terminated,
        env_state_rewards,
    ):
        """Append a fresh slice of selfplay data and backfill rewards onto past positions.

        This is one of the trickiest parts of the RL cycle, and the cause of countless
        headaches. Read it carefully before touching anything.

        The core problem: in self-play we generate the positions of a game *before* we
        know who won. A position is written into the buffer at the step it was played,
        but its reward (+1/-1/0) only becomes known later, at the moment the game
        terminates. So `add()` writes positions with a placeholder reward of 0, and on
        every subsequent call we have to find the positions belonging to games that have
        *just* terminated and patch their rewards in-place. That patching is what
        "backfill" means here.

        Several bookkeeping flags coordinate this:

          - `is_from_selfplay`  : the slot holds real self-play data (vs. uninitialized
                                  default data that's just sitting in the buffer).
          - `is_pending_reward_i8`: this position is still waiting for its terminal
                                  reward to be filled in.
          - `is_fresh_i8`       : this position just received its reward and hasn't been
                                  handed downstream yet.
          - `is_valid_sample`   : this position is fully formed and eligible to be
                                  consumed into the replay buffer and, subsequently,
                                  valid for training.

        The flow below is, roughly: (1) identify which buffer slots belong to a game
        that terminated this step, (2) add the now-known per-player reward onto those
        slots, (3) flip their pending/fresh/valid flags accordingly, and (4) propagate
        a bit of per-game metadata (termination step, game id) used only for logging.
        """
        selfplay_buffer_state = selfplay_buffer.add(
            selfplay_buffer_state, selfplay_output
        )

        # `is_from_selfplay` is necessary to make sure we're not operating on default
        # data sitting in the selfplay buffer. We only want to touch data that actually
        # came out of selfplay_fn. `real_samples` narrows that further to slots that are
        # still awaiting their terminal reward (is_pending_reward_i8).
        is_from_selfplay = selfplay_buffer_state.experience.is_from_selfplay
        real_samples = (
            is_from_selfplay & selfplay_buffer_state.experience.is_pending_reward_i8
        )

        # Of those pending slots, the ones whose game terminated *this* step are the ones
        # we now have a reward for. Split them by which player the position belongs to so
        # we can assign each side its own +1/-1.
        entries_to_update_mask = real_samples * env_state_terminated[:, None]
        player1_entries = entries_to_update_mask * (
            selfplay_buffer_state.experience.player == 0
        )
        player2_entries = entries_to_update_mask * (
            selfplay_buffer_state.experience.player == 1
        )

        player1_rewards = env_state_rewards[:, 0].reshape(-1, 1)
        player2_rewards = env_state_rewards[:, 1].reshape(-1, 1)

        # The reward-backfill trick. SelfplayOutput is always emitted with reward == 0
        # (see selfplay_fn), so adding the terminal reward onto the existing value is
        # equivalent to a masked write: untouched slots keep their 0 (or prior reward),
        # and only the just-terminated slots for each player pick up their +1/-1.
        old_experience = selfplay_buffer_state.experience
        new_rewards = old_experience.reward + (player1_entries * player1_rewards)
        new_rewards = new_rewards + (player2_entries * player2_rewards)
        # A slot that just got its reward is no longer pending, and is now "fresh":
        # carrying a brand-new reward that downstream hasn't seen yet. `consume` clears
        # is_fresh_i8 the first time it hands a slot out, so we never systematically
        # return the same data over and over — that repetition would inject a subtle
        # sampling bias into training.
        new_is_pending_reward_i8 = (
            old_experience.is_pending_reward_i8 - entries_to_update_mask
        )
        new_is_fresh_i8 = old_experience.is_fresh_i8 + entries_to_update_mask

        # A slot becomes a valid, consumable sample once it is real self-play data, not
        # an exploration move, freshly rewarded, and no longer pending. For games that
        # forbid draws we additionally require a non-zero reward as a guard: a 0 reward
        # there can only mean the backfill hasn't actually landed yet.
        assert config["env_allows_draws"] != config["env_forbids_draws"]
        if config["env_allows_draws"]:
            new_is_valid_sample = (
                is_from_selfplay
                & (~old_experience.is_exploration)
                & (new_is_fresh_i8 == 1)
                & (new_is_pending_reward_i8 == 0)
            )
        else:
            new_is_valid_sample = (
                is_from_selfplay
                & (~old_experience.is_exploration)
                & (new_rewards != 0)
                & (new_is_fresh_i8 == 1)
                & (new_is_pending_reward_i8 == 0)
            )

        # Propagate per-game metadata onto the just-terminated slots. This is purely for
        # logging/metrics (e.g. game length and per-game grouping) — it does not affect
        # training. Same masked-write pattern: where(entries_to_update_mask, new, old).
        newly_added_term_step = selfplay_output.ep_termination_step
        old_term_steps = old_experience.ep_termination_step
        update_values = jnp.broadcast_to(newly_added_term_step, old_term_steps.shape)
        new_ep_termination_step = jnp.where(
            entries_to_update_mask, update_values, old_term_steps
        )

        old_game_ids = old_experience.game_id
        update_game_ids = jnp.broadcast_to(selfplay_output.game_id, old_game_ids.shape)
        new_game_id = jnp.where(entries_to_update_mask, update_game_ids, old_game_ids)

        new_experience = selfplay_buffer_state.experience.replace(
            reward=new_rewards,
            is_pending_reward_i8=new_is_pending_reward_i8,
            is_fresh_i8=new_is_fresh_i8,
            is_valid_sample=new_is_valid_sample,
            ep_termination_step=new_ep_termination_step,
            game_id=new_game_id,
        )
        selfplay_buffer_state = selfplay_buffer_state.replace(
            experience=new_experience,
            num_valid_consumable=jnp.sum(new_is_valid_sample),
        )

        return selfplay_buffer_state, ({}, {})

    @functools.partial(jax.jit, donate_argnums=(0,))
    def consume(selfplay_buffer_state):
        new_is_fresh_i8 = selfplay_buffer_state.experience.is_fresh_i8
        new_is_valid_sample = selfplay_buffer_state.experience.is_valid_sample

        k = config["selfplay_buffer_consume_size"]
        B, T = new_is_fresh_i8.shape

        returnable_mask = new_is_valid_sample
        returnable_mask_flat = returnable_mask.flatten()
        is_fresh_i8_flat = new_is_fresh_i8.flatten()

        consume_seed = jnp.max(selfplay_buffer_state.experience.global_step_id).astype(
            jnp.uint32
        )
        consume_rng = jax.random.key(consume_seed)
        noise = jax.random.uniform(consume_rng, shape=returnable_mask_flat.shape)
        scores = jnp.where(returnable_mask_flat, noise, -jnp.inf)
        _, top_indices = jax.lax.top_k(scores, k=k)

        experience_flat = jax.tree.map(
            lambda x: x.reshape(-1, *x.shape[2:]), selfplay_buffer_state.experience
        )
        completed_states = jax.tree.map(lambda x: x[top_indices], experience_flat)

        valid_selection = returnable_mask_flat[top_indices]
        completed_states = jax.tree.map(
            lambda x: jnp.where(valid_selection, x, jnp.zeros_like(x))
            if x.ndim <= 1
            else jnp.where(
                jnp.expand_dims(valid_selection, axis=tuple(range(1, x.ndim))),
                x,
                jnp.zeros_like(x),
            ),
            completed_states,
        )

        completed_games_with_time_axis = jax.tree.map(
            lambda x: jnp.expand_dims(x, axis=1), completed_states
        )
        is_fresh_i8_flat_after_update = is_fresh_i8_flat.at[top_indices].set(
            jnp.int8(0)
        )
        is_fresh_i8_after_return = is_fresh_i8_flat_after_update.reshape(B, T)

        new_is_valid_after_consume = (
            selfplay_buffer_state.experience.is_valid_sample
            & (is_fresh_i8_after_return == 1)
        )
        new_experience = selfplay_buffer_state.experience.replace(
            is_fresh_i8=is_fresh_i8_after_return,
            is_valid_sample=new_is_valid_after_consume,
        )
        selfplay_buffer_state = selfplay_buffer_state.replace(
            experience=new_experience,
            num_valid_consumable=jnp.sum(new_is_valid_after_consume),
        )

        return selfplay_buffer_state, completed_games_with_time_axis, {}

    buffer = Buffer(
        init=init_fn,
        add=selfplay_buffer.add,
        add_backfill=add_backfill,
        consume=consume,
        sample=selfplay_buffer.sample,
        can_sample=selfplay_buffer.can_sample,
    )

    return buffer, selfplay_buffer_state


# =============================================================================
# Self-play
# =============================================================================
def make_selfplay(
    config, wenv, run_mcts_fn, data_sharding=None, allow_exploration=True
):
    config_gumbel_scale = config["mcts_gumbel_scale"]
    if not allow_exploration:
        config_gumbel_scale = 0.0

    def _init_selfplay_state(rng):
        selfplay_batch_size = config["selfplay_batch_size"]
        num_exploratory_moves = config["num_exploratory_moves"]

        rng, env_rng = jax.random.split(rng)
        env_rngs = jax.random.split(env_rng, selfplay_batch_size)
        env_state = wenv.init(env_rngs)

        ep_step = jnp.zeros((selfplay_batch_size), dtype=jnp.int16)

        rng, _rng = jax.random.split(rng)
        if allow_exploration:
            random_switch_step = jax.random.randint(
                _rng,
                shape=(selfplay_batch_size,),
                minval=0,
                maxval=num_exploratory_moves,
            )
        else:
            random_switch_step = jnp.zeros((selfplay_batch_size,))

        # Init with JAX int32 scalars (not Python 0) so their avals match what selfplay
        # returns each step -- otherwise run_fn will recompile when is_warmup=False
        step_count = jnp.zeros((), dtype=jnp.int32)
        next_game_id = jnp.zeros((), dtype=jnp.int32)

        # pin batched selfplay state to data-parallel sharding
        if config.get("enable_sharding", False) and data_sharding is not None:
            env_state = jax.lax.with_sharding_constraint(env_state, data_sharding)
            ep_step = jax.lax.with_sharding_constraint(ep_step, data_sharding)
            random_switch_step = jax.lax.with_sharding_constraint(
                random_switch_step, data_sharding
            )
            step_count = jax.lax.with_sharding_constraint(
                step_count, REPLICATED_SHARDING
            )
            next_game_id = jax.lax.with_sharding_constraint(
                next_game_id, REPLICATED_SHARDING
            )

        return SelfplayState(
            env_state=env_state,
            ep_step=ep_step,
            switch_step=random_switch_step,
            step_count=step_count,
            next_game_id=next_game_id,
        )

    rng = jax.random.key(1)
    selfplay_state = _init_selfplay_state(rng)

    def _collect_selfplay_metrics(
        config,
        selfplay_state,
        ep_termination_step,
        just_terminated,
        rewards,
        next_env_state,
        ep_step,
        action,
        prev_ep_step,
        is_exploration,
    ):
        dummy_max = 1e6
        min_masked = jnp.where(
            ep_termination_step == -1, dummy_max, ep_termination_step
        )
        ep_term_step_min = jnp.min(min_masked)

        dummy_min = -1e6
        max_masked = jnp.where(
            ep_termination_step == -1, dummy_min, ep_termination_step
        )
        ep_term_step_max = jnp.max(max_masked)
        ep_term_step_max = jnp.where(
            ep_term_step_max == dummy_min, -1.0, ep_term_step_max
        )

        avg_valid_mask = ep_termination_step != -1
        avg_sum_valid = jnp.sum(jnp.where(avg_valid_mask, ep_termination_step, 0))
        avg_count_valid = jnp.sum(avg_valid_mask)
        ep_term_step_avg = jnp.where(
            avg_count_valid > 0, avg_sum_valid / avg_count_valid, -1.0
        )

        p1_just_won = just_terminated & (rewards[:, 0] == 1)
        p2_just_won = just_terminated & (rewards[:, 1] == 1)
        just_tied = just_terminated & jnp.all(rewards == 0, axis=-1)

        p1_wins = jnp.sum(p1_just_won)
        p2_wins = jnp.sum(p2_just_won)
        n_ties = jnp.sum(just_tied)

        valid_1s_aft_term = jnp.sum(
            just_terminated
            & (
                jnp.all(rewards == jnp.array([1, -1]), axis=-1)
                | jnp.all(rewards == jnp.array([-1, 1]), axis=-1)
            )
        )
        valid_0s_aft_term = jnp.sum(just_terminated & jnp.all(rewards == 0, axis=-1))
        valid_0s_no_term = jnp.sum(~just_terminated & ~rewards.any(axis=-1))

        if config["env_id"] == "chess":
            num_legal_moves = jnp.sum(
                jax.lax.population_count(next_env_state.legal_action_bitmask), axis=-1
            )
        else:
            num_legal_moves = jnp.sum(next_env_state.legal_action_mask, axis=-1)
        avg_num_legal_moves = jnp.mean(num_legal_moves)

        def masked_average(data, mask):
            masked_data = jnp.where(mask, data, 0)
            count = jnp.sum(mask)
            return jnp.sum(masked_data) / jnp.maximum(count, 1)

        avg_legal_moves_mid = masked_average(
            num_legal_moves, (ep_step > 10) & (ep_step <= 30)
        )

        ep_step_min = jnp.min(ep_step)
        ep_step_max = jnp.max(ep_step)

        scalar_metrics = {
            "selfplay/global_step": selfplay_state.step_count,
            "selfplay/ep_term_step_max": ep_term_step_max,
            "selfplay/ep_term_step_min": ep_term_step_min,
            "selfplay/ep_term_step_avg": ep_term_step_avg,
            "selfplay/p1_wins": p1_wins,
            "selfplay/p2_wins": p2_wins,
            "selfplay/p_just_tied": n_ties,
            "selfplay/n_legal_moves_avg": avg_num_legal_moves,
            "selfplay/n_legal_moves_avg_mid": avg_legal_moves_mid,
            "selfplay/ep_step_min": ep_step_min,
            "selfplay/ep_step_max": ep_step_max,
            "selfplay-reward/valid_1s_aft_term": valid_1s_aft_term,
            "selfplay-reward/valid_0s_aft_term": valid_0s_aft_term,
            "selfplay-reward/valid_0s_no_term": valid_0s_no_term,
        }

        return scalar_metrics, {}

    def selfplay(
        rng: chex.PRNGKey,
        selfplay_state: SelfplayState,
        params,
        gumbel_scale=config_gumbel_scale,
    ):
        selfplay_batch_size = config["selfplay_batch_size"]
        num_exploratory_moves = config["num_exploratory_moves"]

        env_state = selfplay_state.env_state
        ep_step = prev_ep_step = selfplay_state.ep_step
        switch_step = selfplay_state.switch_step
        global_step_count = selfplay_state.step_count

        rng, _rng = jax.random.split(rng)
        policy_output = run_mcts_fn(
            _rng,
            selfplay_state.env_state,
            params,
            gumbel_scale,
            selfplay_batch_size,
        )

        rng, _rng = jax.random.split(rng)
        is_exploration = allow_exploration & (ep_step < switch_step)

        # 1sh variant: sample from visit counts during exploration
        # NOTE: 1sh always returns full-A visit_counts (bnk only compresses the
        # stored training target, not the search tree), so this sampler is
        # unchanged for chess+bnk.
        game_b, num_actions = policy_output.visit_counts.shape
        total_counts = jnp.sum(policy_output.visit_counts, axis=-1, keepdims=True)
        visit_probs = policy_output.visit_counts / jnp.maximum(total_counts, 1)
        visit_probs = jnp.where(total_counts > 0, visit_probs, 1 / num_actions)
        sample_keys = jax.random.split(_rng, game_b)
        sampled_action = jax.vmap(lambda k, p: jax.random.choice(k, num_actions, p=p))(
            sample_keys, visit_probs
        )
        action = jnp.where(is_exploration, sampled_action, policy_output.action)

        cur_observation = wenv.observe(env_state, env_state.current_player)
        # bnk reads (K,) targets from the bnk_* policy fields
        if config.get("exp_bnk_action_weights", False):
            cur_action_weights = policy_output.bnk_action_weights
        else:
            cur_action_weights = policy_output.action_weights
        cur_player = env_state.current_player

        # chess reads compressed obs + the already-packed bitmask
        #     directly from the env. Non-chess envs read the raw bool mask.
        is_chess = config["env_id"] == "chess"
        if is_chess:
            cur_board_bool, cur_board_float = split_observation_vmap(cur_observation)
            cur_legal_action_bitmask = env_state.legal_action_bitmask
        else:
            cur_legal_action_mask = env_state.legal_action_mask

        already_done = env_state.terminated

        rng, step_rng = jax.random.split(rng)
        step_rngs = jax.random.split(step_rng, selfplay_batch_size)

        ### Step 2: Env Step
        next_env_state = wenv.autostep(selfplay_state.env_state, action, step_rngs)
        just_terminated = ~already_done & next_env_state.terminated
        next_ep_step = jnp.where(just_terminated, 0, selfplay_state.ep_step + 1)

        rewards = next_env_state.rewards
        ep_termination_step = jnp.where(just_terminated, selfplay_state.ep_step, -1)

        global_step_ids = jnp.full(
            (selfplay_batch_size,), global_step_count, dtype=jnp.uint32
        )
        col_ids = global_step_ids % (config["game_max_steps"] * 4)
        row_ids = jnp.arange(start=1, stop=(selfplay_batch_size + 1), dtype=jnp.uint32)

        GAME_ID_MODULUS = (1 << 20) - 1
        n_terminated = jnp.sum(just_terminated)
        offsets = jnp.cumsum(just_terminated.astype(jnp.int32)) - 1
        assigned_game_ids = (
            selfplay_state.next_game_id + offsets
        ) % GAME_ID_MODULUS + 1
        game_id = jnp.where(
            just_terminated, assigned_game_ids.astype(jnp.uint32), jnp.uint32(0)
        )

        reward = jnp.zeros((selfplay_batch_size,))
        is_pending_reward_i8 = jnp.ones((selfplay_batch_size,), dtype=jnp.int8)
        is_fresh_i8 = jnp.zeros((selfplay_batch_size,), dtype=jnp.int8)
        is_valid_sample = jnp.zeros((selfplay_batch_size,), dtype=jnp.bool_)
        is_from_selfplay = jnp.ones((selfplay_batch_size,), dtype=jnp.bool_)

        # build SelfplayOutput without obs/mask, then attach per-env below
        selfplay_output = SelfplayOutput(
            col_id=col_ids,
            row_id=row_ids,
            global_step_id=global_step_ids,
            game_id=game_id,
            action=action,
            action_weights=cur_action_weights,
            reward=reward,
            is_valid_sample=is_valid_sample,
            is_from_selfplay=is_from_selfplay,
            player=cur_player,
            just_terminated=just_terminated,
            ep_step=ep_step,
            ep_termination_step=ep_termination_step,
            is_exploration=is_exploration,
            is_pending_reward_i8=is_pending_reward_i8,
            is_fresh_i8=is_fresh_i8,
        )
        # chess stores compressed obs + bitmask; others store raw obs/mask
        if is_chess:
            selfplay_output = selfplay_output.replace(
                board_bool=cur_board_bool,
                board_float=cur_board_float,
                legal_action_bitmask=cur_legal_action_bitmask,
            )
        else:
            selfplay_output = selfplay_output.replace(
                observation=cur_observation,
                legal_action_mask=cur_legal_action_mask,
            )
        # bnk stores the k_indices mapping K-slots back to real actions
        if config.get("exp_bnk_action_weights", False):
            selfplay_output = selfplay_output.replace(
                k_indices=policy_output.bnk_k_indices
            )

        selfplay_output = jax.tree.map(
            lambda x: jnp.expand_dims(x, axis=1), selfplay_output
        )

        ep_step = next_ep_step

        if allow_exploration:
            rng, _rng = jax.random.split(rng)
            random_switch_step = jax.random.randint(
                _rng,
                shape=(selfplay_batch_size,),
                minval=0,
                maxval=num_exploratory_moves,
            )
            switch_step = jnp.where(just_terminated, random_switch_step, switch_step)
        else:
            random_switch_step = jnp.zeros((selfplay_batch_size,))
            switch_step = random_switch_step

        selfplay_state = selfplay_state._replace(
            env_state=next_env_state,
            ep_step=ep_step,
            switch_step=switch_step,
            step_count=global_step_count + 1,
            next_game_id=(selfplay_state.next_game_id + n_terminated) % GAME_ID_MODULUS,
        )

        selfplay_scalar_metrics, selfplay_array_metrics = _collect_selfplay_metrics(
            config=config,
            selfplay_state=selfplay_state,
            ep_termination_step=ep_termination_step,
            just_terminated=just_terminated,
            rewards=rewards,
            next_env_state=next_env_state,
            ep_step=ep_step,
            action=action,
            prev_ep_step=prev_ep_step,
            is_exploration=is_exploration,
        )

        return (
            selfplay_state,
            selfplay_output,
            (selfplay_scalar_metrics, selfplay_array_metrics),
        )

    return selfplay, selfplay_state


# =============================================================================
# Training step (one gradient update)
# =============================================================================
def make_train(config, model, model_state, data_sharding=None):
    base_learning_rate = config["learning_rate"]
    weight_decay = config.get("weight_decay", 0.0001)
    warmup_steps = config.get("lr_warmup_steps", 0)

    if warmup_steps > 0:
        lr_schedule = optax.join_schedules(
            schedules=[
                optax.linear_schedule(
                    init_value=1e-6,
                    end_value=base_learning_rate,
                    transition_steps=warmup_steps,
                ),
                optax.constant_schedule(value=base_learning_rate),
            ],
            boundaries=[warmup_steps],
        )
    else:
        lr_schedule = base_learning_rate

    # [todo] figure out better default, or remove clipping
    max_grad_norm = config.get("max_grad_norm", 1.0)
    tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adamw(lr_schedule, weight_decay=weight_decay),
    )

    # factor out creation so we can JIT it with REPLICATED out_shardings
    def _create_train_state():
        return CustomTrainState.create(
            apply_fn=model.apply,
            params=model_state["params"],
            key=jax.random.PRNGKey(0),
            tx=tx,
            n_updates=0,
        )

    # replicate the full train state (params + opt state) across the mesh
    if config.get("enable_sharding", False) and data_sharding is not None:
        state_shape_tree = jax.eval_shape(_create_train_state)
        out_sharding_tree = jax.tree_util.tree_map(
            lambda _: REPLICATED_SHARDING, state_shape_tree
        )
        sharded_create = jax.jit(_create_train_state, out_shardings=out_sharding_tree)
        with data_sharding.mesh:
            initial_train_state = sharded_create()
    else:
        initial_train_state = _create_train_state()

    def train_step(state: CustomTrainState, batch: SelfplayOutput, is_warmup):
        # is_warmup is a *traced* flag (not static), so warmup and the real
        # training cycle compile to the same executable -- only one XLA scratch
        # region is ever reserved, which is what keeps chess from OOMing when it
        # switches from warmup to training. When warming up we still run the full
        # backward pass + optimizer update (so the graph is identical), then throw
        # the result away
        batch_size = config["train_batch_size"]

        rng, dropout_key = jax.random.split(state.key)

        # pin batch to data-parallel sharding so the train step is DP
        if config.get("enable_sharding", False) and data_sharding is not None:
            batch = jax.lax.with_sharding_constraint(batch, data_sharding)

        # chess decompresses observations + bitmask; TTT uses stored fields
        if config["env_id"] == "chess":
            observations = combine_observation_vmap(batch.board_bool, batch.board_float)
            legal_masks = unpack_bitmask_vmap(batch.legal_action_bitmask)
        else:
            observations = batch.observation
            legal_masks = batch.legal_action_mask

        action_weights, rewards, is_valid_samples = (
            batch.action_weights,
            batch.reward,
            batch.is_valid_sample,
        )

        # scatter (B,K) bnk weights back into full (B,A) before the KL loss
        if not config["mcts_bnk_rehydrate_fields"] and config.get(
            "exp_bnk_action_weights", False
        ):
            k_indices = batch.k_indices
            full_weights = jnp.zeros(
                (batch_size, config["game_num_actions"]), dtype=action_weights.dtype
            )
            batch_idx = jnp.arange(batch_size)[:, None]  # [B, 1] -> [B, K]
            action_weights = full_weights.at[batch_idx, k_indices].set(action_weights)

        def loss_fn(params):
            logits, values = state.apply_fn(
                {"params": params},
                observations,
                legal_masks,
                deterministic=True,
            )

            predicted_pi = jax.nn.softmax(logits)
            batch_loss_pi = jnp.sum(
                jax.scipy.special.rel_entr(action_weights, predicted_pi), axis=-1
            )
            batch_loss_v = optax.l2_loss(values, rewards)

            masked_loss_pi = batch_loss_pi * is_valid_samples
            masked_loss_v = batch_loss_v * is_valid_samples

            loss_pi = jnp.sum(masked_loss_pi) / batch_size
            loss_v = jnp.sum(masked_loss_v) / batch_size
            total_loss = loss_pi + loss_v

            aux = {
                "loss_v": loss_v,
                "loss_pi": loss_pi,
                "values": values,
                "rewards": rewards,
            }
            return total_loss, aux

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, aux), grads = grad_fn(state.params)

        current_lr = lr_schedule(state.step)
        updates, new_opt_state = state.tx.update(grads, state.opt_state, state.params)
        grad_norm = optax.global_norm(grads)
        update_norm = optax.global_norm(updates)
        param_norm = optax.global_norm(state.params)

        aux["norm_metrics"] = {
            "norms/current_lr": current_lr,
            "norms/grad_norm": grad_norm,
            "norms/param_norm": param_norm,
            "norms/update_norm": update_norm,
        }

        new_params = optax.apply_updates(state.params, updates)
        new_state = state.replace(
            params=new_params,
            opt_state=new_opt_state,
            key=rng,
            step=state.step + 1,
            n_updates=state.n_updates + 1,
        )

        new_state = jax.lax.cond(
            is_warmup,
            lambda upd, orig: orig,  # warmup: keep the original (frozen) state
            lambda upd, orig: upd,  # train:  keep the gradient update
            new_state,
            state,
        )

        return new_state, (loss, aux)

    train_fn = jax.jit(train_step, donate_argnums=(0,))

    return train_fn, initial_train_state


# =============================================================================
# AlphaZero system (self-play + train + buffers + runner_state + run_fn)
# =============================================================================
def make_alphazero(config, rng, data_sharding=None):
    # default to the module-global data-parallel sharding when enabled
    if config.get("enable_sharding", False) and data_sharding is None:
        data_sharding = DATA_PARALLEL_SHARDING

    wenv = make_env(config)
    # Derive actual obs/action dimensions from the live environment
    config = config.copy()
    config["game_obs_shape"] = wenv.obs_shape
    config["game_num_actions"] = wenv.num_actions

    # thread sharding through model/train/buffers/selfplay
    model, model_state = make_model(
        config,
        rng,
        sharding=REPLICATED_SHARDING if config.get("enable_sharding", False) else None,
    )
    train_fn, model_ts = make_train(
        config, model, model_state, data_sharding=data_sharding
    )
    run_mcts_fn = make_mcts(config, wenv, model)

    selfplay_fn, selfplay_state = make_selfplay(
        config, wenv, run_mcts_fn, data_sharding=data_sharding
    )
    dummy_selfplay_output = get_dummy_selfplay_output(config)
    replay_buffer, replay_buffer_state = make_replay_buffer(
        config, dummy_selfplay_output, data_sharding=data_sharding
    )
    selfplay_buffer, selfplay_buffer_state = make_selfplay_buffer(
        config, dummy_selfplay_output, data_sharding=data_sharding
    )

    # run_fn is the top-level AlphaZero step: one call = one training cycle.
    # We repeatedly call this function, each call advancing runner_state
    # through three phases:
    #
    #   1. self-play : play cycle_n_selfplay batches of games
    #   2. drain     : move finished positions from selfplay buffer -> replay buffer
    #   3. train     : run cycle_n_train gradient updates
    @functools.partial(jax.jit, donate_argnums=(0,))
    def run_fn(runner_state, is_warmup):
        # is_warmup is a traced bool flag
        # jnp.array(True) for warmup / jnp.array(False) for normal training.
        def _compute_batch_metrics(batch):
            rewards = batch.reward
            is_valid_samples = batch.is_valid_sample
            n_reward_pos = jnp.sum(rewards == 1)
            n_reward_neg = jnp.sum(rewards == -1)
            n_reward_zero = jnp.sum(rewards == 0)
            n_is_valid = jnp.sum(is_valid_samples)
            n_is_invalid = jnp.sum(~is_valid_samples)

            scalar_metrics = {
                "train_batch/n_reward_pos": n_reward_pos,
                "train_batch/n_reward_neg": n_reward_neg,
                "train_batch/n_reward_zero": n_reward_zero,
                "train_batch/n_is_valid": n_is_valid,
                "train_batch/n_is_invalid": n_is_invalid,
            }
            return scalar_metrics, {}

        # === Phase 1: Call selfplay_fn `cycle_n_selfplay` times ===
        def _selfplay_step(carry, _):
            model_ts = carry.model_ts
            selfplay_state = carry.selfplay_state
            selfplay_buffer_state = carry.selfplay_buffer_state
            rng = carry.rng

            rng, selfplay_rng = jax.random.split(rng)
            (
                selfplay_state,
                selfplay_output,
                (selfplay_scalar_metrics, selfplay_array_metrics),
            ) = selfplay_fn(selfplay_rng, selfplay_state, model_ts.params)

            selfplay_buffer_state, (spbuf_scalar, spbuf_array) = (
                selfplay_buffer.add_backfill(
                    selfplay_buffer_state,
                    selfplay_output,
                    selfplay_state.env_state.terminated,
                    selfplay_state.env_state.rewards,
                )
            )

            sp_scalar_metrics = {**selfplay_scalar_metrics, **spbuf_scalar}
            sp_array_metrics = {**selfplay_array_metrics, **spbuf_array}

            new_carry = carry._replace(
                model_ts=model_ts,
                selfplay_state=selfplay_state,
                selfplay_buffer_state=selfplay_buffer_state,
                rng=rng,
            )
            return new_carry, (sp_scalar_metrics, sp_array_metrics)

        carry, (selfplay_scalar_stack, selfplay_array_stack) = jax.lax.scan(
            _selfplay_step, runner_state, None, length=config["cycle_n_selfplay"]
        )

        agg_sp_scalars = jax.tree.map(lambda x: x[-1], selfplay_scalar_stack)
        agg_sp_arrays = jax.tree.map(lambda x: x[-1], selfplay_array_stack)

        # === Phase 2: Drain selfplay buffer into replay buffer ===
        K = config["selfplay_buffer_consume_size"]
        # num_valid_consumable here = positions staged by Phase 1 selfplay (before
        # this drain). n_slices = how many full K-sized slices we drain this cycle.
        num_valid_consumable = carry.selfplay_buffer_state.num_valid_consumable
        n_slices = num_valid_consumable // K
        drain_scalar_metrics = {
            "drain/num_valid_consumable": num_valid_consumable,
            "drain/n_slices": n_slices,
        }

        def _drain_step(i, drain_carry):
            buf_state, replay_state = drain_carry
            buf_state, completed, _consume_metrics = selfplay_buffer.consume(buf_state)
            replay_state = replay_buffer.add(replay_state, completed)
            return (buf_state, replay_state)

        buf_state, replay_state = jax.lax.fori_loop(
            0,
            n_slices,
            _drain_step,
            (carry.selfplay_buffer_state, carry.replay_buffer_state),
        )
        carry = carry._replace(
            selfplay_buffer_state=buf_state,
            replay_buffer_state=replay_state,
        )

        # === Phase 3: Call train_fn `cycle_n_train` times ===
        # Always runs, even during warmup -- but train_fn freezes the model when
        # is_warmup is set (see make_train), so warmup primes the buffers without
        # actually training, while compiling to the same executable as a real cycle.
        def _train_step(carry, _):
            rng, sample_rng = jax.random.split(carry.rng)
            batch = replay_buffer.sample(
                carry.replay_buffer_state, sample_rng
            ).experience
            batch = jax.tree.map(lambda x: x.squeeze(axis=1), batch)

            new_model_ts, (loss, aux) = train_fn(carry.model_ts, batch, is_warmup)

            batch_scalar_metrics, batch_array_metrics = _compute_batch_metrics(batch)
            norm_metrics = aux.get("norm_metrics", {})

            train_scalar_metrics = {
                "total_loss": loss,
                "loss_v": aux["loss_v"],
                "loss_pi": aux["loss_pi"],
                "runner_state/n_updates": new_model_ts.n_updates,
                **drain_scalar_metrics,
                **agg_sp_scalars,
                **norm_metrics,
                **batch_scalar_metrics,
            }
            train_array_metrics = {**agg_sp_arrays, **batch_array_metrics}

            new_carry = carry._replace(model_ts=new_model_ts, rng=rng)
            return new_carry, (train_scalar_metrics, train_array_metrics)

        final_carry, all_metrics = jax.lax.scan(
            _train_step, carry, None, length=config["cycle_n_train"]
        )
        return final_carry, all_metrics

    rng, init_rng = jax.random.split(rng)
    # Commit the top-level rng to the replicated mesh sharding so it matches what
    # run_fn returns (a freshly split key is uncommitted on a single device; if it
    # differs from run_fn's output the first call compiles for SingleDeviceSharding
    # and the next recompiles for the mesh -> a second executable -> OOM).
    if config.get("enable_sharding", False) and data_sharding is not None:
        init_rng = jax.device_put(init_rng, REPLICATED_SHARDING)
    runner_state = RunnerState(
        model_ts=model_ts,
        selfplay_state=selfplay_state,
        selfplay_buffer_state=selfplay_buffer_state,
        replay_buffer_state=replay_buffer_state,
        rng=init_rng,
    )

    # make_alphazero is the top-level factory: it constructs every component of
    # the system once and returns them together, so callers never have to wire
    # the pieces up themselves. The bundle has three kinds of members:
    #   - the step functions that advance the system: run_fn (one full cycle)
    #     and warmup_fn (a cycle that skips training, to prime the buffers);
    #   - the initial runner_state that those functions consume and return;
    #   - the underlying building blocks (env, replay/selfplay buffers, and the
    #     mcts / selfplay / gradient-step functions), exposed so callers can use
    #     a single piece in isolation -- e.g. profiling an individual function,
    #     or reaching for az.run_mcts_fn to run a self-play strength eval
    return SimpleNamespace(
        run_fn=run_fn,
        runner_state=runner_state,
        selfplay_fn=selfplay_fn,
        run_mcts_fn=run_mcts_fn,
        selfplay_buffer=selfplay_buffer,
        replay_buffer=replay_buffer,
        env=wenv,
    )


# =============================================================================
# Training loop
# =============================================================================
def _ascii_loss_chart(series_map, height=12, max_width=70):
    """Render one or more value series as a compact, colored terminal line chart.

    `series_map` is a dict {name: list-of-values}; all series share one y-axis.
    A single list is also accepted (rendered as one unnamed series).
    """
    RESET = "\033[0m"
    COLORS = {"total": "", "value": "\033[36m", "policy": "\033[33m"}  # cyan / yellow
    if not isinstance(series_map, dict):
        series_map = {"loss": series_map}

    def downsample(s):
        s = [float(v) for v in s if v == v]  # drop NaNs
        if len(s) <= max_width:
            return s
        bucket = len(s) / max_width  # bucket-average so long runs stay readable
        out = []
        for i in range(max_width):
            chunk = s[int(i * bucket) : int((i + 1) * bucket)] or [s[int(i * bucket)]]
            out.append(sum(chunk) / len(chunk))
        return out

    cleaned = {n: downsample(s) for n, s in series_map.items()}
    cleaned = {n: s for n, s in cleaned.items() if len(s) >= 2}
    if not cleaned:
        return ""
    allvals = [v for s in cleaned.values() for v in s]
    minimum, maximum = min(allvals), max(allvals)
    interval = (maximum - minimum) or 1.0
    ratio = height / interval
    min2 = int(round(minimum * ratio))
    max2 = int(round(maximum * ratio))
    rows = max(1, max2 - min2)
    offset = 10
    width = max(len(s) for s in cleaned.values()) + offset
    grid = [[" "] * width for _ in range(rows + 1)]
    cgrid = [[""] * width for _ in range(rows + 1)]
    # y-axis labels + axis ticks
    for y in range(min2, max2 + 1):
        label = f"{maximum - (y - min2) * interval / rows:8.4f}"
        for i, ch in enumerate(label):
            grid[y - min2][i] = ch
        grid[y - min2][offset - 1] = "┤"  # ┤
    # plot each series with box-drawing connectors
    for name, s in cleaned.items():
        col = COLORS.get(name, "")

        def put(r, c, ch, col=col):
            grid[r][c] = ch
            cgrid[r][c] = col

        for x in range(len(s) - 1):
            y0 = int(round(s[x] * ratio) - min2)
            y1 = int(round(s[x + 1] * ratio) - min2)
            if y0 == y1:
                put(rows - y0, x + offset, "─")  # ─
            else:
                put(rows - y1, x + offset, "╰" if y0 > y1 else "╭")  # ╰ ╭
                put(rows - y0, x + offset, "╮" if y0 > y1 else "╯")  # ╮ ╯
                for y in range(min(y0, y1) + 1, max(y0, y1)):
                    put(rows - y, x + offset, "│")  # │
    lines = []
    for r in range(rows + 1):
        cells = []
        for c in range(width):
            ch, col = grid[r][c], cgrid[r][c]
            cells.append(col + ch + RESET if (col and ch != " ") else ch)
        lines.append("".join(cells).rstrip())
    legend = "  ".join(f"{COLORS.get(n, '')}● {n}{RESET}" for n in cleaned)
    return "\n".join(lines) + "\n  " + legend


def _probe_executables(run_fn, runner_state):
    """Pre-flight: force-compile run_fn for BOTH is_warmup branches up front, so a
    second-executable recompile (-> doubled HBM scratch -> OOM) surfaces now rather
    than after a long warmup. is_warmup is traced, so both branches must share one
    executable; cache size jumping to 2 means run_fn is not a fixed point on
    runner_state and the train path would OOM once warmup ends.

    Donation-safe: run_fn donates arg 0, so we probe on a throwaway *copy* of
    runner_state and discard the result, leaving the real state intact for warmup.
    Two sequential calls (output -> input) are what actually exercise the fixed
    point. The is_warmup=False call trains on the still-empty replay buffer, but
    that garbage dies with the copy. Note: the copy transiently doubles the
    runner_state's HBM (buffers included).
    """
    # .copy() gives a genuinely distinct buffer (the documented way to survive a
    # donated jit call) -- unlike device_put-to-same-sharding, which aliases the
    # original and would get killed when run_fn donates the probe.
    probe = jax.tree.map(lambda x: x.copy(), runner_state)
    out, _ = run_fn(probe, jnp.array(False))  # train path (heavier: optimizer step)
    out, _ = run_fn(out, jnp.array(True))  # warmup path -- must reuse the same exe
    jax.block_until_ready(out)
    n = run_fn._cache_size()
    if n == 1:
        print(
            "[probe] OK: warmup and train share 1 executable; no post-warmup "
            "recompile/OOM.",
            flush=True,
        )
    else:
        print(
            f"[probe] WARN: run_fn cache size {n} (expected 1) -- not a fixed "
            f"point on runner_state; train path would recompile -> likely OOM "
            f"after warmup.",
            flush=True,
        )
    del out  # drop the throwaway state; real runner_state is untouched


def run_alphazero(config, ckpt_path=None):
    rng = jax.random.PRNGKey(42)

    rng, az_rng = jax.random.split(rng)
    az = make_alphazero(config, az_rng)

    run_fn = az.run_fn
    runner_state = az.runner_state
    wenv = az.env

    print("Successfully initialized all components.")

    # Optional pre-flight: compile both is_warmup branches now (on a throwaway copy)
    # so a recompile-induced OOM surfaces before the long warmup, not after it.
    if config.get("debug_probe_executables", True):
        _probe_executables(run_fn, runner_state)

    # --- Warmup Phase: run cycles with is_warmup=True (model frozen) to prime the
    # buffers before real training starts. ---
    num_warmup_cycles = (
        config.get("replay_buffer_warmup_steps", 100) // config["cycle_n_selfplay"]
    )
    print(f"Starting buffer warmup for {num_warmup_cycles} cycles...")
    start_time = time.time()

    for cycle_i in range(num_warmup_cycles):
        call_start = time.time()
        runner_state, _ = run_fn(runner_state, jnp.array(True))
        runner_state.model_ts.step.block_until_ready()
        print(
            f"  Warmup {cycle_i}/{num_warmup_cycles} | {time.time() - call_start:.2f}s"
        )

    warmup_duration = time.time() - start_time
    print(f"Warmup finished in {warmup_duration:.1f}s.")

    num_params = sum(x.size for x in jax.tree.leaves(runner_state.model_ts.params))
    print(f"Model has {num_params:,} parameters.")

    # --- Main Training Cycle ---
    total_steps = runner_state.model_ts.step
    n_cycles = (config["num_iters"] - total_steps) // config["cycle_n_selfplay"]
    print(f"Starting training for {n_cycles} cycles...")

    cycle_total_duration = 0
    start_time = time.time()

    loss_history = {
        "total": [],
        "value": [],
        "policy": [],
    }  # per-cycle, for ASCII chart
    chart_period = config.get("loss_chart_period", 50)

    # --- Strength eval setup: a GATED ELO LADDER.
    # The anchor is a frozen opponent we measure ΔElo against; the current model's
    # absolute Elo = anchor_elo + ΔElo(current vs anchor). The ladder starts pinned
    # to the random opponent (anchor_params=None, anchor_elo=0). We only PROMOTE the
    # current checkpoint to be the new anchor once it beats the current anchor in the
    # *informative* band of the logistic — score ≥ eval_promote_score, sustained for
    # eval_promote_patience consecutive evals (hysteresis). Promoting in-band (rather
    # than at ~100%, where score saturates and the ΔElo gap is unmeasurable) keeps
    # each rung's height a real measurement instead of a clamp-capped guess, so the
    # ladder Elo stays calibrated to absolute strength rather than inflating.
    eval_period = config.get("eval_period", 100)
    # Promote when current scores this well vs the anchor (top of the informative
    # band: ~0.85 ≈ +300 ΔElo). Higher → fewer, taller rungs but riskier saturation.
    eval_promote_score = config.get("eval_promote_score", 0.85)
    # ...and only after it holds for this many consecutive evals (noise guard).
    eval_promote_patience = config.get("eval_promote_patience", 2)
    # Periodic checkpointing: every ckpt_period cycles, overwrite ckpt_path with
    # the latest params (crash recovery / resumable inference). Disabled if no
    # ckpt_period set or no path given (e.g. --no-save).
    ckpt_period = config.get("ckpt_period")
    eval_openings = all_opening_actions(wenv, config)
    eval_fn = az.run_mcts_fn
    # Ladder state. anchor_params=None ⇒ rung 0 = random opponent, pinned at Elo 0.
    anchor_params = None
    anchor_elo = 0.0
    anchor_cycle = 0
    rung = 0
    qualify_deltas = []  # in-band ΔElo samples accumulating toward a promotion
    best_score_rand = 0.0  # high-water mark vs random, for a forgetting warning
    elo_curve = [0.0]  # current model's absolute Elo over time (anchor_elo + live Δ)
    elo_cycles = [0]

    # Graceful Ctrl+C: the first press finishes the in-flight cycle, then breaks
    # out of training so we still save params + play against the model. A second
    # press restores the default handler and aborts hard.
    import signal

    interrupt = {"flag": False}

    def _handle_sigint(signum, frame):
        if interrupt["flag"]:
            signal.signal(signal.SIGINT, orig_sigint)
            raise KeyboardInterrupt
        interrupt["flag"] = True
        print(
            "\n⚠️  Ctrl+C — will stop after this cycle, then save + play. "
            "Press again to abort.",
            flush=True,
        )

    orig_sigint = signal.signal(signal.SIGINT, _handle_sigint)

    for cycle_n in range(1, int(n_cycles) + 1):
        cycle_start_time = time.time()

        runner_state, (all_scalar_metrics, _) = run_fn(runner_state, jnp.array(False))
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), runner_state)

        cycle_duration = time.time() - cycle_start_time
        cycle_total_duration += cycle_duration

        # Log last train step's metrics, grouped by phase (selfplay / drain / train)
        last = jax.tree_util.tree_map(lambda x: float(x[-1]), all_scalar_metrics)
        print(
            f"[{config.get('game_name', config['env_id'])}] "
            f"Cycle {cycle_n}/{int(n_cycles)} | {cycle_duration:.2f}s\n"
            f"  phase1 selfplay | "
            f"p1_wins={last.get('selfplay/p1_wins', 0):.0f} "
            f"p2_wins={last.get('selfplay/p2_wins', 0):.0f} "
            f"ties={last.get('selfplay/p_just_tied', 0):.0f} "
            f"n_legal_avg_mid={last.get('selfplay/n_legal_moves_avg_mid', 0):.2f}\n"
            f"  phase2 drain    | "
            f"consumable={last.get('drain/num_valid_consumable', 0):.0f} "
            f"slices={last.get('drain/n_slices', 0):.0f}\n"
            f"  phase3 train    | "
            f"loss={last.get('total_loss', 0):.4f} "
            f"loss_v={last.get('loss_v', 0):.4f} "
            f"loss_pi={last.get('loss_pi', 0):.4f} "
            f"batch[r+={last.get('train_batch/n_reward_pos', 0):.0f} "
            f"r-={last.get('train_batch/n_reward_neg', 0):.0f} "
            f"r0={last.get('train_batch/n_reward_zero', 0):.0f} "
            f"valid={last.get('train_batch/n_is_valid', 0):.0f} "
            f"invalid={last.get('train_batch/n_is_invalid', 0):.0f}]",
            flush=True,
        )

        loss_history["total"].append(last.get("total_loss", float("nan")))
        loss_history["value"].append(last.get("loss_v", float("nan")))
        loss_history["policy"].append(last.get("loss_pi", float("nan")))
        if (
            len(loss_history["total"]) > 4000
        ):  # cap memory; chart bucket-averages anyway
            loss_history = {k: v[-4000:] for k, v in loss_history.items()}
        if (
            chart_period
            and cycle_n % chart_period == 0
            and len(loss_history["total"]) >= 2
        ):
            n_pts = len(loss_history["total"])
            print(f"\n── loss over last {n_pts} cycles ──", flush=True)
            print(
                _ascii_loss_chart(
                    loss_history,
                    height=config.get("chart_height", 24),
                    max_width=config.get("chart_width", 160),
                ),
                flush=True,
            )
            print(flush=True)

        is_diagnostic_time = (cycle_n == 1) or (
            cycle_n % config.get("diagnostic_period", 100) == 0
        )
        if is_diagnostic_time and config["env_id"] == "tic_tac_toe":
            _run_ttt_diagnostics(runner_state.model_ts, wenv, config)
        # hex value-head-vs-perfect-play ASCII diagnostic
        elif is_diagnostic_time and config["env_id"].startswith("hex"):
            _run_hex_diagnostics(runner_state.model_ts, wenv, config)
        # connect4 opening-column value-vs-perfect-play diagnostic
        elif is_diagnostic_time and config["env_id"] == "connect_four":
            _run_connect4_diagnostics(runner_state.model_ts, wenv, config)

        # --- Strength eval: gated Elo ladder (see setup block above).
        # Always measure vs random (forgetting detector + rung-0 yardstick); on
        # higher rungs also measure vs the frozen anchor. Promote in-band.
        if eval_period and cycle_n % eval_period == 0:
            print(
                f"\n--- Strength eval at cycle {cycle_n} "
                f"({len(eval_openings)} openings x2 colors per opponent) ---"
            )
            cur_params = runner_state.model_ts.params
            ek_rand, ek_anchor = jax.random.split(jax.random.PRNGKey(1234 + cycle_n))

            # vs random: permanent baseline. On rung 0 this IS the anchor match.
            score_rand, delta_rand = evaluate_vs(
                eval_fn,
                wenv,
                config,
                cur_params,
                None,
                openings=eval_openings,
                key=ek_rand,
                label="random (Elo 0)",
            )
            if score_rand > best_score_rand:
                best_score_rand = score_rand
            elif rung > 0 and score_rand < best_score_rand - 0.15:
                print(
                    f"  [eval] ⚠️  forgetting? vs-random score {score_rand:.3f} is "
                    f"well below high-water {best_score_rand:.3f}",
                    flush=True,
                )

            # Score/ΔElo against the *current* anchor drives the ladder.
            if anchor_params is None:  # rung 0: the anchor is random
                score_vs_anchor, delta_vs_anchor = score_rand, delta_rand
            else:
                score_vs_anchor, delta_vs_anchor = evaluate_vs(
                    eval_fn,
                    wenv,
                    config,
                    cur_params,
                    anchor_params,
                    openings=eval_openings,
                    key=ek_anchor,
                    label=f"anchor R{rung}@{anchor_cycle}",
                )

            # Absolute Elo = height of the current rung + live gap above it.
            live_elo = anchor_elo + float(delta_vs_anchor)
            elo_curve.append(live_elo)
            elo_cycles.append(cycle_n)
            print(
                f"  [eval] ladder Elo: {live_elo:+.0f}  (rung {rung} @ "
                f"{anchor_elo:+.0f}, +{float(delta_vs_anchor):.0f} over anchor)",
                flush=True,
            )

            # Promotion: in-band score, sustained eval_promote_patience evals. We
            # freeze the rung height as the *mean* in-band gap over the streak,
            # which halves its variance vs a single 1-batch measurement.
            if score_vs_anchor >= eval_promote_score:
                qualify_deltas.append(float(delta_vs_anchor))
                if len(qualify_deltas) >= eval_promote_patience:
                    rung_height = float(np.mean(qualify_deltas))
                    anchor_elo += rung_height
                    anchor_params = jax.tree_util.tree_map(jnp.copy, cur_params)
                    anchor_cycle = cycle_n
                    rung += 1
                    qualify_deltas = []
                    print(
                        f"  [eval] ⬆ PROMOTE to rung {rung}: anchor now @ "
                        f"{anchor_elo:+.0f} Elo (rung height +{rung_height:.0f}, "
                        f"avg of {eval_promote_patience} evals)",
                        flush=True,
                    )
            else:
                qualify_deltas = []  # streak broken — reset hysteresis

            if len(elo_curve) >= 3:
                print(
                    f"\n── ladder Elo over cycles {elo_cycles[1]}..{elo_cycles[-1]} ──",
                    flush=True,
                )
                print(_ascii_loss_chart({"total": elo_curve}), flush=True)
                print(flush=True)

        if ckpt_period and ckpt_path and cycle_n % ckpt_period == 0:
            print(f"\n--- Checkpoint at cycle {cycle_n} ---", flush=True)
            save_params(runner_state.model_ts.params, ckpt_path)

        if interrupt["flag"]:
            print(f"Stopping early at cycle {cycle_n} (Ctrl+C).", flush=True)
            break

    signal.signal(signal.SIGINT, orig_sigint)  # restore default Ctrl+C handling
    total_duration = time.time() - start_time
    print(f"Training finished in {total_duration:.1f}s.")
    return runner_state


# =============================================================================
# Head-to-head evaluation (any game)
# =============================================================================
# Self-play loss curves don't tell you if the model is getting stronger (both
# players co-improve). These helpers give a fixed yardstick that works for every
# pgx game (incl. chess, no external engine): play one game per legal opening
# move (ttt->9, connect4->7, hex4x4->16, chess->20) head-to-head between two
# param sets and report the win/draw/loss split. Two useful opponents:
#   * None         -> uniform-random legal play  (sanity: is it learning at all?)
#   * anchor_params -> a frozen earlier snapshot  (is it STILL improving? -> ΔElo)
# Enumerating every opening covers the position space deterministically (greedy
# MCTS), so the same model gives the same scoreline every time -> low variance.
# Everything is batched on-TPU and reuses run_mcts + the env, so it's cheap.


def _legal_mask_from_state(env_state, config):
    if config["env_id"] == "chess":
        return unpack_bitmask_vmap(env_state.legal_action_bitmask)
    return env_state.legal_action_mask


def _random_legal_actions(env_state, config, key):
    """Uniform random over legal actions (batched)."""
    legal = _legal_mask_from_state(env_state, config)
    logits = jnp.where(legal, 0.0, -jnp.inf)
    g = jax.random.gumbel(key, logits.shape, dtype=logits.dtype)
    return jnp.argmax(logits + g, axis=-1).astype(jnp.int32)


def all_opening_actions(wenv, config):
    """Every legal first move from the initial position (player-1 openings).
    ttt->9, connect4->7, hex4x4->16, chess->20, etc."""
    env_state = wenv.init_dummy_estate(batch_size=1)
    legal = np.asarray(_legal_mask_from_state(env_state, config)[0])
    return np.nonzero(legal)[0].astype(np.int32)


def run_eval_match(
    run_mcts_fn,
    wenv,
    config,
    params_p0,
    params_p1,
    *,
    opening_actions,
    key,
    max_plies=None,
    num_simulations=None,
    gumbel_scale=0.0,
):
    """Play one game per forced opening, fully batched.

    `opening_actions` is a 1-D array of player-1 first moves; one game is played
    per entry. Seat 0 plays with `params_p0`, seat 1 with `params_p1`. A param
    set of `None` means uniform-random legal play. After the forced opening move,
    both sides play greedy MCTS. Returns (p0_wins, draws, p1_wins) as ints; games
    that don't finish within `max_plies` count as draws.
    """
    if max_plies is None:
        max_plies = config.get("eval_max_plies") or config.get("game_max_steps") or 512
    if num_simulations is None:
        num_simulations = config["mcts_num_simulations"]

    opening_actions = np.asarray(opening_actions, dtype=np.int32)
    num_real = len(opening_actions)

    # Pad the batch up to a multiple of the device count so data-parallel
    # sharding is happy; padded games duplicate real openings and are dropped.
    mult = jax.device_count() if config.get("enable_sharding", False) else 1
    pad = (-num_real) % mult
    batch_actions = (
        np.concatenate([opening_actions, opening_actions[:pad]])
        if pad
        else opening_actions
    )
    num_games = len(batch_actions)

    env_state = wenv.init_dummy_estate(batch_size=num_games)
    env_state = wenv.step(env_state, jnp.asarray(batch_actions))  # forced opening

    final_rewards = jnp.zeros((num_games, 2), dtype=jnp.float32)
    finished = env_state.terminated

    for _ply in range(int(max_plies)):
        newly = env_state.terminated & ~finished
        final_rewards = jnp.where(newly[:, None], env_state.rewards, final_rewards)
        finished = finished | env_state.terminated
        if bool(jnp.all(finished)):
            break

        # Live games are in lockstep (player alternates each ply), so one seat
        # moves for the whole batch this ply.
        cp = jnp.where(~finished, env_state.current_player, -1)
        seat = int(jnp.max(cp))
        params = params_p0 if seat == 0 else params_p1

        key, mk = jax.random.split(key)
        if params is None:
            actions = _random_legal_actions(env_state, config, mk)
        else:
            actions = run_mcts_fn(
                mk, env_state, params, gumbel_scale, num_games, num_simulations
            ).action
        env_state = wenv.step(env_state, actions)

    # Credit any game that terminated on the very last step.
    newly = env_state.terminated & ~finished
    final_rewards = jnp.where(newly[:, None], env_state.rewards, final_rewards)

    r0 = np.asarray(final_rewards[:num_real, 0])  # drop padded games
    p0_wins = int(np.sum(r0 > 0))
    p1_wins = int(np.sum(r0 < 0))
    draws = num_real - p0_wins - p1_wins
    return p0_wins, draws, p1_wins


def evaluate_vs(
    run_mcts_fn,
    wenv,
    config,
    cur_params,
    opp_params,
    *,
    openings,
    key,
    label,
    num_simulations=None,
):
    """Play `cur_params` vs `opp_params` over every opening, from BOTH colors.
    Prints a W/D/L line + score + ΔElo estimate, and returns the score in [0,1]."""
    k1, k2 = jax.random.split(key)
    # current as seat 0
    w0, d0, l0 = run_eval_match(
        run_mcts_fn,
        wenv,
        config,
        cur_params,
        opp_params,
        opening_actions=openings,
        key=k1,
        num_simulations=num_simulations,
    )
    # current as seat 1 (swap): current's wins are the seat-1 wins
    l1, d1, w1 = run_eval_match(
        run_mcts_fn,
        wenv,
        config,
        opp_params,
        cur_params,
        opening_actions=openings,
        key=k2,
        num_simulations=num_simulations,
    )
    wins, draws, losses = w0 + w1, d0 + d1, l0 + l1
    n = max(wins + draws + losses, 1)
    score = (wins + 0.5 * draws) / n
    s = min(max(score, 1e-4), 1 - 1e-4)
    elo = -400.0 * np.log10(1.0 / s - 1.0)
    print(
        f"  [eval] vs {label:<14} {wins:>3}W {draws:>3}D {losses:>3}L "
        f"| score={score:.3f} | ΔElo≈{elo:+.0f}",
        flush=True,
    )
    return score, elo


# =============================================================================
# Checkpointing & interactive play
# =============================================================================
def default_ckpt_path(env_name: str) -> str:
    """Default on-disk location for a saved alphazero checkpoint."""
    return os.path.join("artifacts", f"alphazero_{env_name}.pkl")


def save_params(params, path: str) -> None:
    """Pickle the model params (converted to numpy) to `path`.

    We store only the learned params pytree (not the optimizer / train state),
    which is all that's needed to run inference / play against the model.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    np_params = jax.tree_util.tree_map(lambda x: np.asarray(x), params)
    with open(path, "wb") as f:
        pickle.dump(np_params, f)
    print(f"✅ Saved model params to {path}")


def load_params(path: str):
    """Load a params pytree previously written by `save_params`."""
    if not os.path.exists(path):
        raise SystemExit(
            f"No checkpoint found at {path}. Train a model first, or point "
            f"--load at an existing checkpoint."
        )
    with open(path, "rb") as f:
        params = pickle.load(f)
    print(f"✅ Loaded model params from {path}")
    return params


def make_play(config):
    """Build a single-game (batch=1) inference setup: env, model, MCTS fn.

    Sharding is disabled so a batch of one plays cleanly on a single device.
    """
    config = config.copy()
    config["enable_sharding"] = False

    wenv = make_env(config)
    config["game_obs_shape"] = wenv.obs_shape
    config["game_num_actions"] = wenv.num_actions

    rng = jax.random.PRNGKey(0)
    model, model_state = make_model(config, rng, sharding=None)

    play_cfg = config.copy()
    play_cfg["selfplay_batch_size"] = 1
    play_cfg["mcts_bnk_rehydrate_fields"] = True
    run_mcts_fn = make_mcts(play_cfg, wenv, model)

    return wenv, model, model_state, run_mcts_fn, config


def _p0p1_board(env_state, wenv):
    """Fixed-perspective board: +1 = player 0's stones, -1 = player 1's."""
    obs = np.asarray(wenv.observe(env_state, env_state.current_player))[0].astype(
        np.int8
    )
    board = obs[:, :, 0] - obs[:, :, 1]
    if int(env_state.current_player[0]) == 1:
        board = -board
    return board


def _sym(v, empty="."):
    return "X" if v > 0 else ("O" if v < 0 else empty)


def _print_board(env_state, wenv, env_id):
    board = _p0p1_board(env_state, wenv)
    H, W = board.shape
    print()
    if env_id == "connect_four":
        for r in range(H):
            print("   " + "  ".join(_sym(board[r, c]) for c in range(W)))
        print("   " + "  ".join(str(c + 1) for c in range(W)))
    elif env_id.startswith("hex"):
        # Columns are letters (a, b, ...), rows are 1-indexed numbers.
        # Each row is shifted half a cell to the right to form the rhombus.
        col_letters = " ".join(chr(ord("a") + c) for c in range(W))
        print("     " + col_letters)
        for r in range(H):
            cells = " ".join(_sym(board[r, c]) for c in range(W))
            print(f"  {r + 1:>2} " + " " * r + cells)
    else:  # tic_tac_toe and other square grids
        for r in range(H):
            print(
                "   "
                + " ".join(
                    _sym(board[r, c], empty=str(r * W + c + 1)) for c in range(W)
                )
            )
    print()


def _parse_move(raw: str, env_id: str, H: int, W: int, legal):
    """Parse a human move string into a legal action index, or None."""
    raw = raw.strip().lower()
    try:
        if env_id == "connect_four":
            # input is a 1-indexed column number
            action = int(raw) - 1
        elif env_id.startswith("hex"):
            m = re.fullmatch(r"([a-z])\s*(\d+)", raw)
            if m:  # algebraic, e.g. "a1" / "c3" (column letter + row number)
                col = ord(m.group(1)) - ord("a")
                row = int(m.group(2)) - 1
                action = row * W + col
            else:
                parts = raw.split()
                if len(parts) == 2:  # "row col", 1-indexed
                    action = (int(parts[0]) - 1) * W + (int(parts[1]) - 1)
                else:  # single 1-indexed cell number
                    action = int(raw) - 1
        else:
            parts = raw.split()
            if len(parts) == 2:  # "row col", 1-indexed
                action = (int(parts[0]) - 1) * W + (int(parts[1]) - 1)
            else:  # single 1-indexed cell number
                action = int(raw) - 1
    except (ValueError, TypeError):
        return None
    n_actions = legal.shape[0]
    if 0 <= action < n_actions and bool(legal[action]):
        return action
    return None


def _action_to_str(action: int, env_id: str, W: int) -> str:
    """Human-readable description of a model's chosen action."""
    if env_id == "connect_four":
        return f"column {action + 1}"
    row, col = action // W, action % W
    if env_id.startswith("hex"):
        return f"{chr(ord('a') + col)}{row + 1}"
    return f"cell {action + 1} (row {row + 1}, col {col + 1})"


def _print_model_eval(model, params, wenv, state, env_id, W):
    """Print the raw network output (policy logits/probs + value head) for the
    current position, from the side-to-move's (i.e. the model's) point of view.
    """
    obs = wenv.observe(state, state.current_player)
    legal = state.legal_action_mask
    logits, value = model.apply({"params": params}, obs, legal)
    logits = np.asarray(logits[0], dtype=np.float32)
    probs = np.asarray(jax.nn.softmax(jnp.asarray(logits)), dtype=np.float32)
    value = float(np.asarray(value[0]))
    legal_np = np.asarray(legal[0])

    print(f"  value head: {value:+.3f}  (current player's perspective)")
    print("  policy (legal actions, by probability):")
    legal_idxs = sorted(np.where(legal_np)[0], key=lambda i: -probs[i])
    for i in legal_idxs:
        bar = "#" * int(round(probs[i] * 20))
        print(
            f"    {_action_to_str(int(i), env_id, W):>22}  "
            f"logit={logits[i]:+7.3f}  p={probs[i]:5.3f}  {bar}"
        )


def play_against_model(config, params=None, *, human_player=0, num_simulations=None):
    """Play an interactive terminal game against the trained model.

    `human_player` is 0 (you move first) or 1 (model moves first).
    Commands during your turn: `undo`, `restart`, `quit`.
    """
    env_id = config["env_id"]
    if env_id == "chess":
        print("Playing against chess model currently not supported.")
        return

    wenv, model, model_state, run_mcts_fn, config = make_play(config)
    H, W = wenv.obs_shape[0], wenv.obs_shape[1]

    if params is None:
        params = model_state["params"]
        print("⚠️  No params provided — playing against an UNTRAINED model.")
    params = jax.device_put(params)

    if num_simulations is None:
        num_simulations = config["mcts_num_simulations"]

    human_sym = "X" if human_player == 0 else "O"
    print("\n" + "=" * 50)
    print(f"Playing {env_id}.  You are '{human_sym}' (player {human_player + 1}).")
    if env_id == "connect_four":
        print("Enter a column number (1-7) to drop your piece.")
    elif env_id.startswith("hex"):
        print("Enter a cell as a column letter + row number, e.g. 'a1' or 'c3'.")
    else:
        print("Enter the number shown in an empty cell (or 'row col').")
    print("Commands: undo, restart, quit")
    print("=" * 50)

    rng = jax.random.PRNGKey(int(time.time()))
    rng, env_rng = jax.random.split(rng)
    initial_state = wenv.init(jax.random.split(env_rng, 1))
    state = initial_state
    history = [initial_state]

    while not bool(state.terminated[0]):
        _print_board(state, wenv, env_id)
        current_player = int(state.current_player[0])

        if current_player == human_player:
            _print_model_eval(model, params, wenv, state, env_id, W)
            legal = np.asarray(state.legal_action_mask[0])
            while True:
                raw = input(f"Your move ({human_sym}): ").strip().lower()
                if raw in ("quit", "q", "exit"):
                    print("Bye.")
                    return
                if raw == "undo":
                    # step back to the human's previous turn (drop human + model plies)
                    if len(history) >= 3:
                        history = history[:-2]
                    elif len(history) >= 2:
                        history = history[:-1]
                    else:
                        print("Nothing to undo.")
                    state = history[-1]
                    break
                if raw == "restart":
                    state = initial_state
                    history = [initial_state]
                    print("Game restarted.")
                    break
                action = _parse_move(raw, env_id, H, W, legal)
                if action is None:
                    print("  Illegal or unparseable move, try again.")
                    continue
                state = wenv.step(state, jnp.array([action], dtype=jnp.int32))
                history.append(state)
                break
        else:
            print("Model is thinking...")
            _print_model_eval(model, params, wenv, state, env_id, W)
            rng, mcts_rng = jax.random.split(rng)
            policy_output = run_mcts_fn(
                mcts_rng,
                state,
                params,
                gumbel_scale=0.0,
                batch_size=1,
                num_simulations=num_simulations,
            )
            action = int(policy_output.action[0])
            print(f"Model plays {_action_to_str(action, env_id, W)}.")
            state = wenv.step(state, jnp.array([action], dtype=jnp.int32))
            history.append(state)

    # --- Game over ---
    _print_board(state, wenv, env_id)
    rewards = np.asarray(state.rewards[0])
    print("=" * 50)
    print("--- GAME OVER ---")
    human_reward = rewards[human_player]
    if human_reward > 0:
        print("You win! 🎉")
    elif human_reward < 0:
        print("Model wins. 🤖")
    else:
        print("Draw.")
    print(f"Rewards [P1, P2] = {rewards.tolist()}")
    print("=" * 50)


# =============================================================================
# Configuration
# =============================================================================
def get_ttt_config():
    board_size = 3
    game_max_steps = board_size * board_size
    batch_size = 4096
    REPLAY_BUFFER_TOTAL_SIZE = 1_024_000

    selfplay_buffer_len = game_max_steps + 10
    replay_buffer_len = REPLAY_BUFFER_TOTAL_SIZE // batch_size
    buffer_warmup_steps = (selfplay_buffer_len + replay_buffer_len) * 1

    return {
        # --- Game ---
        "env_id": "tic_tac_toe",
        "game_max_steps": game_max_steps,
        "num_exploratory_moves": 4,
        "env_forbids_draws": False,
        "env_allows_draws": True,
        "boardsize": board_size,
        # game_obs_shape and game_num_actions are derived from the live env in make_alphazero
        "game_obs_shape": None,
        "game_num_actions": board_size * board_size,
        # --- Model ---
        "use_conv_model": True,
        "conv_width": 32,
        "conv_depth": 4,
        "conv_use_names": True,
        "conv_use_derf": True,
        "conv_use_kata_gpool": False,
        # --- MCTS ---
        "mcts_num_simulations": 13,
        "mcts_variant": "1sh",
        "mcts_max_m": 9,
        "mcts_num_root_considered": 9,
        "mcts_num_survivors": 4,
        "mcts_num_k_actions": 9,
        "mcts_use_gumbel": True,
        "mcts_gumbel_scale": 1.0,
        "mcts_epsilon": 1e-8,
        "mcts_rescale_values": False,
        "mcts_value_scale": 1.0,
        "mcts_use_mixed_value": True,
        "mcts_maxvisit_init": 50,
        "mcts_bnk_rehydrate_fields": False,  # [todo] not implemented
        # --- Training ---
        "num_iters": 10_000,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "use_bf16": False,
        "lr_warmup_steps": buffer_warmup_steps,
        "train_batch_size": batch_size,
        "cycle_n_selfplay": 10,
        "cycle_n_train": 10,
        # --- Self-play & Buffers ---
        "selfplay_batch_size": batch_size,
        "selfplay_buffer_add_batch_size": batch_size,
        "selfplay_buffer_sample_batch_size": batch_size,
        "selfplay_buffer_min_len": game_max_steps,
        "selfplay_buffer_max_len": game_max_steps,
        "selfplay_buffer_consume_size": batch_size,
        "replay_buffer_total_size": REPLAY_BUFFER_TOTAL_SIZE,
        "replay_buffer_add_batch_size": batch_size,
        "replay_buffer_sample_batch_size": batch_size,
        "replay_buffer_min_len": 1,
        "replay_buffer_max_len": replay_buffer_len,
        "replay_buffer_warmup_steps": buffer_warmup_steps,
        # --- Diagnostics & strength eval ---
        "diagnostic_period": 100,
        "eval_period": 50,  # run every N cycles
        "eval_max_plies": None,
        "ckpt_period": None,  # save a checkpoint every N cycles (None = only at end)
        # --- System ---
        "enable_sharding": True,
    }


def get_hex_config(board_size=4):
    board_cfgs = {
        4: dict(
            conv_width=64,
            conv_depth=4,
            num_iters=(500 * 10),
            learning_rate=1e-3,
            cycle_n_selfplay=10,
            cycle_n_train=10,
        ),
        5: dict(
            conv_width=128,
            conv_depth=4,
            num_iters=(800 * 20),
            learning_rate=1e-4,
            cycle_n_selfplay=20,
            cycle_n_train=20,
        ),
        6: dict(
            conv_width=256,
            conv_depth=8,
            num_iters=(3500 * 30),
            learning_rate=1e-4,
            cycle_n_selfplay=30,
            cycle_n_train=30,
        ),
        7: dict(
            conv_width=256,
            conv_depth=16,
            num_iters=(5000 * 40),
            learning_rate=1e-4,
            cycle_n_selfplay=40,
            cycle_n_train=30,
        ),
        8: dict(
            conv_width=256,
            conv_depth=16,
            num_iters=(10_000 * 60),
            learning_rate=1e-4,
            cycle_n_selfplay=60,
            cycle_n_train=50,
        ),
        9: dict(
            conv_width=256,
            conv_depth=32,
            num_iters=(40_000 * 80),
            learning_rate=1e-4,
            cycle_n_selfplay=80,
            cycle_n_train=50,
        ),
    }
    if board_size not in board_cfgs:
        raise ValueError(f"Unsupported hex board size: {board_size}")
    board_cfg = board_cfgs[board_size]

    game_max_steps = board_size * board_size

    # ---- shrunk for easy single-file runs (hex prod = 8192) ----
    batch_size = 4096
    REPLAY_BUFFER_TOTAL_SIZE = 2_048_000
    # ------------------------------------------------------------

    selfplay_buffer_len = game_max_steps + 20
    replay_buffer_len = REPLAY_BUFFER_TOTAL_SIZE // batch_size
    buffer_warmup_steps = (selfplay_buffer_len + replay_buffer_len) * 1

    return {
        # --- Game ---
        "env_id": f"hexnoswap_{board_size}x{board_size}",
        "game_max_steps": game_max_steps,
        "num_exploratory_moves": game_max_steps // 2,
        # hex has no draws: the last player to move always wins.
        "env_forbids_draws": True,
        "env_allows_draws": False,
        "boardsize": board_size,
        "game_obs_shape": None,
        "game_num_actions": None,  # patched from live env in make_alphazero
        # --- Model ---
        "use_conv_model": True,
        "conv_width": board_cfg["conv_width"],
        "conv_depth": board_cfg["conv_depth"],
        "conv_use_names": True,
        "conv_use_derf": True,
        "conv_use_kata_gpool": True,
        # --- MCTS (1sh) ---
        "mcts_num_simulations": 24,
        "mcts_variant": "1sh",
        "mcts_max_m": 16,
        "mcts_num_root_considered": 16,
        "mcts_num_survivors": 8,
        "mcts_num_k_actions": game_max_steps,  # = board_size**2
        "mcts_use_gumbel": True,
        "mcts_gumbel_scale": 1.0,
        "mcts_epsilon": 1e-8,
        "mcts_rescale_values": False,
        "mcts_value_scale": 1.0,
        "mcts_use_mixed_value": True,
        "mcts_maxvisit_init": 50,
        "mcts_bnk_rehydrate_fields": False,
        # bnk off for hex (full (A,) policy targets); root temperature from prod hex
        "exp_bnk_action_weights": False,
        "exp_use_root_temperature": True,
        "exp_root_temperature": 1.3,
        # --- Training ---
        "num_iters": board_cfg["num_iters"],
        "learning_rate": board_cfg["learning_rate"],
        "weight_decay": 1e-4,
        "use_bf16": False,
        "lr_warmup_steps": buffer_warmup_steps,
        "train_batch_size": batch_size,
        "cycle_n_selfplay": board_cfg["cycle_n_selfplay"],
        "cycle_n_train": board_cfg["cycle_n_train"],
        # --- Self-play & Buffers ---
        "selfplay_batch_size": batch_size,
        "selfplay_buffer_add_batch_size": batch_size,
        "selfplay_buffer_sample_batch_size": batch_size,
        "selfplay_buffer_min_len": selfplay_buffer_len,
        "selfplay_buffer_max_len": selfplay_buffer_len,
        "selfplay_buffer_consume_size": batch_size,
        "replay_buffer_total_size": REPLAY_BUFFER_TOTAL_SIZE,
        "replay_buffer_add_batch_size": batch_size,
        "replay_buffer_sample_batch_size": batch_size,
        "replay_buffer_min_len": 1,
        "replay_buffer_max_len": replay_buffer_len,
        "replay_buffer_warmup_steps": buffer_warmup_steps,
        # --- Diagnostics & strength eval ---
        "diagnostic_period": 50,
        "eval_period": 50,  # run every N cycles
        "eval_max_plies": None,
        "ckpt_period": None,  # save a checkpoint every N cycles (None = only at end)
        # --- System ---
        "enable_sharding": True,
    }


def get_connect4_config():
    game_max_steps = 42  # 6 rows x 7 cols
    batch_size = 8192
    REPLAY_BUFFER_TOTAL_SIZE = 2_048_000

    selfplay_buffer_len = game_max_steps + 10
    replay_buffer_len = REPLAY_BUFFER_TOTAL_SIZE // batch_size
    buffer_warmup_steps = (selfplay_buffer_len + replay_buffer_len) * 1

    return {
        # --- Game ---
        "env_id": "connect_four",
        "game_max_steps": game_max_steps,
        "num_exploratory_moves": 21,
        # connect4 can end in a draw (full board), so draws are allowed.
        "env_forbids_draws": False,
        "env_allows_draws": True,
        "boardsize": 7,  # number of columns (= action space); board is 6x7
        "game_obs_shape": None,
        "game_num_actions": 7,
        # --- Model ---
        "use_conv_model": True,
        "conv_width": 256,
        "conv_depth": 8,
        "conv_use_names": True,
        "conv_use_derf": True,
        "conv_use_kata_gpool": True,
        # --- MCTS (1sh) ---
        "mcts_num_simulations": 64,
        "mcts_variant": "1sh",
        "mcts_max_m": 7,
        "mcts_num_root_considered": 7,
        "mcts_num_survivors": 3,
        "mcts_num_k_actions": 7,
        "mcts_use_gumbel": True,
        "mcts_gumbel_scale": 1.0,
        "mcts_epsilon": 1e-8,
        "mcts_rescale_values": False,
        "mcts_value_scale": 1.0,
        "mcts_use_mixed_value": True,
        "mcts_maxvisit_init": 50,
        "mcts_bnk_rehydrate_fields": False,
        "exp_bnk_action_weights": False,
        # --- Training ---
        "num_iters": 2000 * 20,
        "learning_rate": 5e-4,
        "weight_decay": 1e-4,
        "use_bf16": False,
        "lr_warmup_steps": buffer_warmup_steps,
        "train_batch_size": batch_size,
        "cycle_n_selfplay": 20,
        "cycle_n_train": 12,
        # --- Self-play & Buffers ---
        "selfplay_batch_size": batch_size,
        "selfplay_buffer_add_batch_size": batch_size,
        "selfplay_buffer_sample_batch_size": batch_size,
        "selfplay_buffer_min_len": selfplay_buffer_len,
        "selfplay_buffer_max_len": selfplay_buffer_len,
        "selfplay_buffer_consume_size": batch_size,
        "replay_buffer_total_size": REPLAY_BUFFER_TOTAL_SIZE,
        "replay_buffer_add_batch_size": batch_size,
        "replay_buffer_sample_batch_size": batch_size,
        "replay_buffer_min_len": 1,
        "replay_buffer_max_len": replay_buffer_len,
        "replay_buffer_warmup_steps": buffer_warmup_steps,
        # --- Diagnostics & strength eval ---
        "diagnostic_period": 50,
        "eval_period": 50,  # run every N cycles
        "eval_max_plies": None,
        "ckpt_period": None,  # save a checkpoint every N cycles (None = only at end)
        # --- System ---
        "enable_sharding": True,
    }


def get_chess_config():
    board_size = 8
    GAME_MAX_STEPS = 512

    selfplay_bs = 4096
    train_bs = 4096
    REPLAY_BUFFER_TOTAL_SIZE = 4096000 * 2
    # -----------------------------------------------------------------------

    selfplay_buffer_len = GAME_MAX_STEPS + 20
    replay_buffer_len = REPLAY_BUFFER_TOTAL_SIZE // train_bs
    buffer_warmup_steps = selfplay_buffer_len + replay_buffer_len

    return {
        # --- Game ---
        "env_id": "chess",
        "game_max_steps": GAME_MAX_STEPS,
        "num_exploratory_moves": 30,
        "env_forbids_draws": False,
        "env_allows_draws": True,
        "boardsize": board_size,
        "game_obs_shape": None,
        "game_num_actions": None,  # patched from live env in make_alphazero
        # --- Model ---
        "use_conv_model": True,
        "conv_width": 384,
        "conv_depth": 32,
        "conv_use_names": True,
        "conv_use_derf": True,
        "conv_use_kata_gpool": True,
        # --- MCTS (1sh) ---
        "mcts_num_simulations": 12,
        "mcts_variant": "1sh",
        "mcts_max_m": 8,
        "mcts_num_root_considered": 8,
        "mcts_num_survivors": 4,
        "mcts_num_k_actions": 128,
        "mcts_use_gumbel": True,
        "mcts_gumbel_scale": 1.0,
        "mcts_epsilon": 1e-8,
        "mcts_rescale_values": False,
        "mcts_value_scale": 1.0,
        "mcts_use_mixed_value": True,
        "mcts_maxvisit_init": 50,
        "mcts_bnk_rehydrate_fields": False,
        # bnk: store compressed (K,) policy targets instead of full (4672,)
        "exp_bnk_action_weights": True,
        "exp_use_root_temperature": True,  # from KataGo
        "exp_root_temperature": 1.5,
        # --- Training ---
        "num_iters": 128_000 * 20,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "use_bf16": False,
        "lr_warmup_steps": buffer_warmup_steps,
        "train_batch_size": train_bs,
        "cycle_n_selfplay": 20,
        "cycle_n_train": 20,
        # --- Self-play & Buffers ---
        "selfplay_batch_size": selfplay_bs,
        "selfplay_buffer_add_batch_size": selfplay_bs,
        "selfplay_buffer_sample_batch_size": train_bs,
        "selfplay_buffer_min_len": selfplay_buffer_len,
        "selfplay_buffer_max_len": selfplay_buffer_len,
        "selfplay_buffer_consume_size": train_bs,
        "replay_buffer_total_size": REPLAY_BUFFER_TOTAL_SIZE,
        "replay_buffer_add_batch_size": train_bs,
        "replay_buffer_sample_batch_size": train_bs,
        "replay_buffer_min_len": 1,
        "replay_buffer_max_len": replay_buffer_len,
        "replay_buffer_warmup_steps": buffer_warmup_steps,
        "diagnostic_period": 50,
        # --- Strength eval (vs random + frozen anchor; no external engine) ---
        # Plays one game per legal opening (chess = 20) from both colors.
        "eval_period": 50,  # run every N cycles
        "eval_max_plies": 200,  # cap match length; unfinished games = draw
        "ckpt_period": 800,  # save a checkpoint every 800 cycles
        # --- System ---
        "enable_sharding": True,
    }


# =============================================================================
# Compression utils (chess)
# =============================================================================
# Chess observation is (8,8,119); we compress the bool channels into packed uint8
# and store the legal-action mask as a uint32 bitset. This drastically reduces
# replay-buffer memory vs. storing the raw (8,8,119) float obs + (4672,) bool mask.
def split_observation(obs_array):
    bool_indices = jnp.concatenate([jnp.arange(113), jnp.arange(114, 118)])
    float_indices = jnp.array([113, 118])
    bool_part = obs_array[:, :, bool_indices].astype(jnp.bool_)
    float_part = obs_array[:, :, float_indices].astype(jnp.bfloat16)
    packed_bool_part = jnp.packbits(bool_part.flatten())
    return packed_bool_part, float_part


split_observation_vmap = jax.vmap(split_observation)


def combine_observation(packed_bool_part, float_part):
    bool_flat = jnp.unpackbits(packed_bool_part)
    bool_part = bool_flat.reshape((8, 8, 117))
    obs_reconstructed = jnp.zeros((8, 8, 119), dtype=jnp.float32)
    bool_indices = jnp.concatenate([jnp.arange(113), jnp.arange(114, 118)])
    obs_reconstructed = obs_reconstructed.at[:, :, bool_indices].set(
        bool_part.astype(jnp.float32)
    )
    float_indices = jnp.array([113, 118])
    obs_reconstructed = obs_reconstructed.at[:, :, float_indices].set(
        float_part.astype(jnp.float32)
    )
    return obs_reconstructed


combine_observation_vmap = jax.vmap(combine_observation)

NUM_ACTIONS = 4672
NUM_WORDS = (NUM_ACTIONS + 31) // 32  # 146


def pack_mask(mask):
    reshaped_mask = mask.reshape(NUM_WORDS, 32)
    powers_of_2 = jnp.left_shift(jnp.uint32(1), jnp.arange(32, dtype=jnp.uint32))
    return jnp.sum(reshaped_mask * powers_of_2, axis=1, dtype=jnp.uint32)


def unpack_bitmask(bitset):
    powers_of_2 = jnp.left_shift(jnp.uint32(1), jnp.arange(32, dtype=jnp.uint32))
    return ((bitset[:, None] & powers_of_2[None, :]) > 0).flatten()


pack_mask_vmap = jax.vmap(pack_mask)
unpack_bitmask_vmap = jax.vmap(unpack_bitmask)

# =============================================================================
# Game-specific diagnostics & perfect-play tables
# =============================================================================
def _run_ttt_diagnostics(model_ts, wenv, config):
    boardsize = config.get("boardsize", 3)
    env = pgx.make("tic_tac_toe")
    dummy_state = env.init(jax.random.PRNGKey(0))
    obs = env.observe(dummy_state, dummy_state.current_player)
    obs_b = obs[jnp.newaxis, ...]
    legal_b = dummy_state.legal_action_mask[jnp.newaxis, ...]

    logits, values = model_ts.apply_fn(
        {"params": model_ts.params}, obs_b, legal_b, deterministic=True
    )
    logits = logits.flatten()
    value = float(values.flatten()[0])

    print(f"  Diagnostics: P1 argmax={int(jnp.argmax(logits))}, value={value:.3f}")
    logits_2d = np.array(logits).reshape((boardsize, boardsize))

    print(f"  Logits (empty board), value={value:.3f}:")
    for r in range(boardsize):
        print("    " + "  ".join(f"{logits_2d[r, c]:+.2f}" for c in range(boardsize)))


# hex perfect-play ground truth
# Value matrix from the perspective of the player to move (White) *after* Black's
# opening. A Black-winning opening leaves White in a lost position -> value -1.0;
# a Black-losing opening leaves White winning -> value +1.0.
def _get_hex_perfect_play_values(boardsize: int):
    # fmt: off
    winning = {
        4: [3, 6, 9, 12],
        5: [4, 6, 7, 8, 9, 11, 12, 13, 15, 16, 17, 18, 20],
        6: [5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 30],
        7: [6, 9, 11, 12, 13, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 29,
            30, 31, 32, 33, 35, 36, 37, 39, 42],
        8: [7, 14, 15, 17, 18, 19, 20, 21, 22, 25, 26, 27, 28, 29, 30, 31, 32, 33,
            34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 48, 49, 56],
        9: [8,9,10,11,16,17,19,20,21,22,23,24,25,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,55,56,57,58,59,60,61,63,64,69,70,71,72]
    }
    # fmt: on
    if boardsize not in winning:
        return None
    n = boardsize * boardsize
    vals = np.ones(n, dtype=np.float32)
    vals[np.array(winning[boardsize])] = -1.0
    return vals.reshape((boardsize, boardsize))


def _run_hex_diagnostics(model_ts, wenv, config):
    boardsize = config["boardsize"]
    batch_size = boardsize * boardsize

    # One blank board per opening; each plays a distinct first move.
    env_state = wenv.init_dummy_estate(batch_size=batch_size)
    all_moves = jnp.arange(batch_size)
    env_state = wenv.step(env_state, all_moves)
    obs = wenv.observe(env_state, env_state.current_player)

    _logits, values = model_ts.apply_fn(
        {"params": model_ts.params},
        obs,
        env_state.legal_action_mask,
        deterministic=True,
    )
    values_2d = np.array(values.flatten()).reshape((boardsize, boardsize))

    print(
        "\n--- Hex value-head after each Black opening "
        "(value = White-to-move perspective; negative => Black-winning) ---"
    )
    for r in range(boardsize):
        print("  " + " ".join(f"{values_2d[r, c]:+.2f}" for c in range(boardsize)))

    gt = _get_hex_perfect_play_values(boardsize)
    if gt is None:
        print(f"  (no ground-truth perfect-play table for {boardsize}x{boardsize})")
        return

    # B = Black-winning opening (gt -1), . = Black-losing opening (gt +1)
    print("  Perfect play (B=Black wins / .=Black loses), [x]=model sign mismatch:")
    pred_sign = np.sign(values_2d)
    for r in range(boardsize):
        cells = []
        for c in range(boardsize):
            truth = "B" if gt[r, c] < 0 else "."
            mismatch = pred_sign[r, c] != np.sign(gt[r, c])
            cells.append(f"[{truth}]" if mismatch else f" {truth} ")
        print("  " + "".join(cells))

    mse = float(np.mean((values_2d - gt) ** 2))
    sign_acc = float(np.mean(pred_sign == np.sign(gt)))
    print(f"  MSE vs perfect = {mse:.4f} | sign accuracy = {sign_acc:.3f}")


# connect4 perfect-play opening values + ASCII diagnostic
# Connect 4 is solved: with perfect play, P1 wins iff they open in the center
# column (col 4 / 0-indexed 3); the two adjacent columns (3 & 5 / idx 2 & 4) draw;
# the four edge columns (1,2,6,7 / idx 0,1,5,6) are losses for P1.
# Values are from the perspective of the player to move *after* the opening (P2):
#   P1-win opening   -> P2 is lost   -> -1.0
#   draw opening     ->  0.0
#   P1-loss opening  -> P2 wins      -> +1.0
def _get_connect4_perfect_play_values():
    # idx:            0    1    2    3    4    5    6
    vals = np.array([+1.0, +1.0, 0.0, -1.0, 0.0, +1.0, +1.0], dtype=np.float32)
    labels = ["L", "L", "D", "W", "D", "L", "L"]  # outcome for P1
    return vals, labels


def _run_connect4_diagnostics(model_ts, wenv, config):
    num_cols = 7
    # One blank board per opening column; each plays a distinct first move.
    env_state = wenv.init_dummy_estate(batch_size=num_cols)
    all_moves = jnp.arange(num_cols)
    env_state = wenv.step(env_state, all_moves)
    obs = wenv.observe(env_state, env_state.current_player)

    _logits, values = model_ts.apply_fn(
        {"params": model_ts.params},
        obs,
        env_state.legal_action_mask,
        deterministic=True,
    )
    values = np.array(values.flatten())  # (7,) P2-to-move perspective

    gt, labels = _get_connect4_perfect_play_values()

    print(
        "\n--- Connect4 value-head after each opening column "
        "(value = P2-to-move perspective; negative => P1-winning) ---"
    )
    print("  col:    " + "  ".join(f"{c+1:>5d}" for c in range(num_cols)))
    print("  value:  " + "  ".join(f"{values[c]:+.2f}" for c in range(num_cols)))
    print(
        "  perfect:"
        + "  ".join(f"{labels[c]:>5s}" for c in range(num_cols))
        + "    (W=P1 wins, D=draw, L=P1 loses)"
    )

    # Directional correctness: W -> value<0, L -> value>0, D -> |value| small.
    draw_tol = 0.5
    correct = []
    for c in range(num_cols):
        if labels[c] == "W":
            correct.append(values[c] < 0)
        elif labels[c] == "L":
            correct.append(values[c] > 0)
        else:  # draw
            correct.append(abs(values[c]) < draw_tol)
    print("  match:  " + "  ".join("  ok " if ok else "  x  " for ok in correct))

    mse = float(np.mean((values - gt) ** 2))
    acc = float(np.mean(correct))
    print(f"  MSE vs perfect = {mse:.4f} | directional accuracy = {acc:.3f}")


# =============================================================================
# Entry point
# =============================================================================
CONFIG_FACTORIES = {
    "chess": get_chess_config,
    "ttt": get_ttt_config,
    "hex4": lambda: get_hex_config(board_size=4),
    "hex5": lambda: get_hex_config(board_size=5),
    "hex6": lambda: get_hex_config(board_size=6),
    "hex7": lambda: get_hex_config(board_size=7),
    "hex8": lambda: get_hex_config(board_size=8),
    "hex9": lambda: get_hex_config(board_size=9),
    "connect4": get_connect4_config,
}


def parse_args():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="ttt", choices=list(CONFIG_FACTORIES))
    # --- checkpointing ---
    ap.add_argument(
        "--save",
        default=None,
        help="path to save params after training "
        "(default: artifacts/alphazero_<env>.pkl)",
    )
    ap.add_argument(
        "--no-save", action="store_true", help="do not save a checkpoint after training"
    )
    ap.add_argument(
        "--load",
        default=None,
        help="path to load params from (defaults to the save path "
        "when --play-only is set)",
    )
    # --- play ---
    ap.add_argument(
        "--play",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="play against the model interactively after training "
        "(on by default; pass --no-play to train only)",
    )
    ap.add_argument(
        "--play-only",
        action="store_true",
        help="skip training; load a checkpoint and play",
    )
    ap.add_argument(
        "--play-as",
        type=int,
        default=1,
        choices=[1, 2],
        help="which player you are (1 = you move first)",
    )
    ap.add_argument(
        "--play-sims",
        type=int,
        default=None,
        help="MCTS simulations the model uses while playing",
    )
    return ap.parse_args()


def run_play(config, args):
    """Load a checkpoint and play interactively (no training)."""
    save_path = args.save or default_ckpt_path(args.env)
    params = load_params(args.load or save_path)
    play_against_model(
        config,
        params,
        human_player=args.play_as - 1,
        num_simulations=args.play_sims,
    )


def main():
    args = parse_args()
    config = CONFIG_FACTORIES[args.env]()
    config["game_name"] = args.env

    if args.play_only:
        run_play(config, args)
        return

    # Run AlphaZero end to end, then save and (optionally) play the result.
    save_path = args.save or default_ckpt_path(args.env)
    runner_state = run_alphazero(config, ckpt_path=None if args.no_save else save_path)
    params = runner_state.model_ts.params
    if not args.no_save:
        save_params(params, save_path)
    if args.play:
        play_against_model(
            config,
            params,
            human_player=args.play_as - 1,
            num_simulations=args.play_sims,
        )


if __name__ == "__main__":
    main()
