"""Replay/self-play buffers and their stored sample representation."""

import functools
from dataclasses import dataclass
from typing import Callable, Generic, Optional

import chex
import flashbax as fbx
import jax
import jax.numpy as jnp
from flashbax.buffers.trajectory_buffer import (
    Experience,
    TrajectoryBuffer,
    TrajectoryBufferState,
)
from jax import Array
from jax.typing import ArrayLike


# =============================================================================
# Self-play records
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
        replicated_sharding = jax.sharding.NamedSharding(
            data_sharding.mesh, jax.sharding.PartitionSpec()
        )
        sample_fn = jax.jit(replay_buffer.sample, out_shardings=data_sharding)
        state_shape_tree = jax.eval_shape(replay_buffer.init, dummy_selfplay_output)

        def _spec(shape_struct):
            return data_sharding if shape_struct.ndim > 0 else replicated_sharding

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
        replicated_sharding = jax.sharding.NamedSharding(
            data_sharding.mesh, jax.sharding.PartitionSpec()
        )
        state_shape_tree = jax.eval_shape(selfplay_buffer.init, dummy_selfplay_output)

        def _spec(shape_struct):
            return data_sharding if shape_struct.ndim > 0 else replicated_sharding

        out_sharding_tree = jax.tree_util.tree_map(_spec, state_shape_tree)
        init_fn = jax.jit(selfplay_buffer.init, out_shardings=out_sharding_tree)
        with data_sharding.mesh:
            selfplay_buffer_state = init_fn(dummy_selfplay_output)
            selfplay_buffer_state = CustomTrajectoryBufferState(
                experience=selfplay_buffer_state.experience,
                current_index=selfplay_buffer_state.current_index,
                is_full=selfplay_buffer_state.is_full,
                num_valid_consumable=jax.lax.with_sharding_constraint(
                    jnp.array(0, dtype=jnp.int32), replicated_sharding
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

