# =============================================================================
# Imports
# =============================================================================
import functools
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, NamedTuple, Optional, Tuple

import chex
import flax.traverse_util
import jax
import jax.numpy as jnp
import optax
import pgx1
from flax.training import train_state
from flashbax.buffers.trajectory_buffer import TrajectoryBufferState
from pgx1.experimental import auto_reset

from nanoalphazero.buffers import (
    Buffer,
    CustomTrajectoryBufferState,
    NUM_ACTIONS,
    NUM_WORDS,
    SelfplayOutput,
    combine_observation,
    combine_observation_vmap,
    get_dummy_selfplay_output,
    make_replay_buffer,
    make_selfplay_buffer,
    pack_mask,
    pack_mask_vmap,
    split_observation,
    split_observation_vmap,
    unpack_bitmask,
    unpack_bitmask_vmap,
)
from nanoalphazero.checkpoint import (
    apply_checkpoint_model_config,
    checkpoint_model_config,
    default_ckpt_path,
    load_checkpoint,
    save_checkpoint,
)
from nanoalphazero.config import (
    CONFIG_FACTORIES,
    get_chess_config,
    get_connect4_config,
    get_go_config,
    get_hex_config,
    get_ttt_config,
)
from nanoalphazero.mcts import (
    Action,
    Params,
    PolicyOutput,
    RecurrentFn,
    RecurrentFnOutput,
    RecurrentState,
    RootFnOutput,
    final_qtransform_completed_by_mix_value,
    gumbel_muzero_policy_1sh,
    make_mcts,
    masked_argmax,
)
from nanoalphazero.model import (
    ChessPolicyHead,
    GenericPolicyHead,
    GoPolicyHead,
    KataConvAndGPool,
    KataGoTrunk,
    KataModel,
    NestedBottleneckResBlock,
    NormActConv,
    NormMask,
    PRESETS,
    RepVGGLinearConv,
    ResBlock,
    ValueHead,
    init_and_shard_model,
    kata_gpool,
    kata_init,
    kata_value_head_gpool,
    make_model,
    resolve_preset,
    value_from_logits,
)

P = jax.sharding.PartitionSpec
mesh = jax.sharding.Mesh(jax.devices(), "x")
DATA_PARALLEL_SHARDING = jax.sharding.NamedSharding(mesh, P("x"))
REPLICATED_SHARDING = jax.sharding.NamedSharding(mesh, P())


# =============================================================================
# Data types
# =============================================================================


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
# Environment
# =============================================================================
@dataclass
class WrappedEnv:
    obs_shape: Tuple
    num_actions: int
    init: Callable
    step: Callable
    autostep: Callable
    init_dummy_estate: Callable
    single_estate: Any
    observe: Optional[Callable] = None
    replay_batch: Optional[Callable] = None

    def __repr__(self) -> str:
        return f"WrappedEnv(obs_shape={self.obs_shape}, num_actions={self.num_actions})"


def make_env(config):
    env_id = config["env_id"]
    # chess carries legality as a packed uint32 bitmask; see make_mcts
    env_kwargs = {"use_bitmask": True} if env_id == "chess" else {}
    env = pgx1.make(env_id, **env_kwargs)
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
        obs_shape=pgx_obs_shape,
        num_actions=pgx_num_actions,
        init=vmap_env_init,
        step=vmap_env_step,
        autostep=vmap_auto_step,
        init_dummy_estate=init_dummy_estate,
        single_estate=single_estate,
        observe=vmap_observe_fn,
        replay_batch=getattr(env, "replay_batch", None),
    )



# =============================================================================
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
        ep_step_std = jnp.std(ep_step)  # jnp.std upcasts int16 ep_step to float

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
            "selfplay/ep_step_std": ep_step_std,
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
    wdl_loss_weight = config.get("wdl_loss_weight", 1.0)
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
        lr_schedule = optax.constant_schedule(base_learning_rate)

    max_grad_norm = config.get("max_grad_norm", 1.0)

    # Decay convolution/dense kernels only. RVGL stores its parallel branches
    # as kernel_3x3/kernel_1x1; biases and Fixup norm parameters are excluded.
    def _decay_mask_fn(params):
        flat = flax.traverse_util.flatten_dict(params)
        kernel_names = ("kernel", "kernel_3x3", "kernel_1x1")
        mask = {path: (path[-1] in kernel_names) for path in flat}
        return flax.traverse_util.unflatten_dict(mask)

    adamw_kwargs = dict(weight_decay=weight_decay)
    if config.get("weight_decay_kernels_only", False):
        adamw_kwargs["mask"] = _decay_mask_fn

    tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adamw(lr_schedule, **adamw_kwargs),
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

        rng, _ = jax.random.split(state.key)

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
            logits, values, value_logits = state.apply_fn(
                {"params": params},
                observations,
                legal_masks,
                deterministic=True,
                return_wdl_logits=True,
            )

            predicted_pi = jax.nn.softmax(logits)
            batch_loss_pi = jnp.sum(
                jax.scipy.special.rel_entr(action_weights, predicted_pi), axis=-1
            )
            # KataGo orders the value classes as {win, loss, no-result/draw}.
            wdl_labels = jnp.where(
                rewards == 1, 0, jnp.where(rewards == -1, 1, 2)
            ).astype(jnp.int32)
            batch_loss_v = optax.softmax_cross_entropy_with_integer_labels(
                value_logits, wdl_labels
            )

            masked_loss_pi = batch_loss_pi * is_valid_samples
            masked_loss_v = batch_loss_v * is_valid_samples

            loss_pi = jnp.sum(masked_loss_pi) / batch_size
            loss_v = jnp.sum(masked_loss_v) / batch_size
            total_loss = loss_pi + wdl_loss_weight * loss_v

            value_probs = jax.nn.softmax(value_logits, axis=-1)
            n_valid = jnp.maximum(jnp.sum(is_valid_samples), 1)
            aux = {
                "loss_v": loss_v,
                "loss_pi": loss_pi,
                "values": values,
                "rewards": rewards,
                "wdl_metrics": {
                    "wdl/p_win_mean": jnp.sum(
                        value_probs[..., 0] * is_valid_samples
                    ) / n_valid,
                    "wdl/p_loss_mean": jnp.sum(
                        value_probs[..., 1] * is_valid_samples
                    ) / n_valid,
                    "wdl/p_draw_mean": jnp.sum(
                        value_probs[..., 2] * is_valid_samples
                    ) / n_valid,
                },
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
def _print_config(config):
    """Print the full run config as a sorted key/value table before training."""
    name = config.get("game_name", config.get("env_id", "?"))
    print("\n" + "=" * 60)
    print(f"Run config [{name}]")
    print("=" * 60)
    width = max(len(k) for k in config)
    for k in sorted(config):
        print(f"  {k:<{width}} : {config[k]}")
    print("=" * 60 + "\n", flush=True)


def make_alphazero(config, rng, data_sharding=None):
    # default to the module-global data-parallel sharding when enabled
    if config.get("enable_sharding", False) and data_sharding is None:
        data_sharding = DATA_PARALLEL_SHARDING

    wenv = make_env(config)
    # Derive actual obs/action dimensions from the live environment
    config = config.copy()
    config["game_obs_shape"] = wenv.obs_shape
    config["game_num_actions"] = wenv.num_actions

    _print_config(config)

    # thread sharding through model/train/buffers/selfplay
    model, model_state = make_model(
        config,
        rng,
        sharding=REPLICATED_SHARDING if config.get("enable_sharding", False) else None,
    )
    train_fn, model_ts = make_train(
        config, model, model_state, data_sharding=data_sharding
    )
    run_mcts_fn = make_mcts(config, wenv, model, data_sharding=data_sharding)

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
            wdl_metrics = aux.get("wdl_metrics", {})

            train_scalar_metrics = {
                "total_loss": loss,
                "loss_v": aux["loss_v"],
                "loss_pi": aux["loss_pi"],
                "runner_state/n_updates": new_model_ts.n_updates,
                **drain_scalar_metrics,
                **agg_sp_scalars,
                **norm_metrics,
                **wdl_metrics,
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
        config=config,
    )


# Compatibility wrappers for callers that imported host-side helpers from core.
# New code should import these from training, play, checkpoint, config, or cli.
def run_alphazero(config, ckpt_path=None):
    from nanoalphazero.training import run_alphazero as _run_alphazero

    return _run_alphazero(config, ckpt_path=ckpt_path)


def all_opening_actions(wenv, config, plies=1):
    from nanoalphazero.training import all_opening_actions as _all_opening_actions

    return _all_opening_actions(wenv, config, plies=plies)


def run_eval_match(*args, **kwargs):
    from nanoalphazero.training import run_eval_match as _run_eval_match

    return _run_eval_match(*args, **kwargs)


def evaluate_vs(*args, **kwargs):
    from nanoalphazero.training import evaluate_vs as _evaluate_vs

    return _evaluate_vs(*args, **kwargs)


def make_play(config):
    from nanoalphazero.play import make_play as _make_play

    return _make_play(config)


def play_against_model(config, params=None, *, human_player=0, num_simulations=None):
    from nanoalphazero.play import play_against_model as _play_against_model

    return _play_against_model(
        config,
        params,
        human_player=human_player,
        num_simulations=num_simulations,
    )


def play_both(config, params=None):
    from nanoalphazero.play import play_both as _play_both

    return _play_both(config, params)


def parse_args():
    from nanoalphazero.cli import parse_train_args

    return parse_train_args()


def run_play(config, args):
    from nanoalphazero.cli import run_play as _run_play

    return _run_play(config, args)


def run_play_both(config, args):
    from nanoalphazero.cli import run_play_both as _run_play_both

    return _run_play_both(config, args)


def main():
    from nanoalphazero.cli import train_main

    return train_main()
