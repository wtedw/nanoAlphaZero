"""Completed-episode staging using AlphaZero's actual fresh-position consumer.

Collection/backfill remains synchronous: no unresolved outcome enters staging.
Only consumption selects positions; Flashbax replay stores materialized unrolls.
"""

import chex
import flashbax
import jax
import jax.numpy as jnp
from flax import struct
from flashbax.buffers.trajectory_buffer import TrajectoryBufferState

from nanoalphazero.buffers import CustomTrajectoryBufferState, make_selfplay_buffer
from nanoalphazero.research.muzero.replay import sequences


@chex.dataclass(frozen=True)
class ConsumptionView:
    global_step_id: jax.Array
    is_fresh_i8: jax.Array
    is_valid_sample: jax.Array
    lane: jax.Array
    slot: jax.Array
    start: jax.Array


@struct.dataclass
class StagingState:
    episodes: object
    fresh: jax.Array
    generation: jax.Array


def make_staging(config):
    batch = config["selfplay_batch_size"]
    slots = config["staging_batches"]
    length = config["max_steps"]
    consume_size = config["consume_size"]
    storage = flashbax.make_trajectory_buffer(
        add_batch_size=batch, sample_batch_size=1, sample_sequence_length=1,
        period=1, min_length_time_axis=1, max_length_time_axis=slots)
    # The production consumer needs only these six scalar fields. Its unused
    # initial buffer is tiny; real payload stays in the episode storage above.
    dummy = ConsumptionView(global_step_id=jnp.uint32(0), is_fresh_i8=jnp.int8(0),
                            is_valid_sample=jnp.bool_(False), lane=jnp.int32(0),
                            slot=jnp.int32(0), start=jnp.int32(0))
    production, _ = make_selfplay_buffer(dict(
        enable_sharding=False, selfplay_buffer_add_batch_size=1,
        selfplay_buffer_sample_batch_size=1, selfplay_buffer_min_len=1,
        selfplay_buffer_max_len=1, selfplay_buffer_consume_size=consume_size), dummy)

    def init(example):
        return StagingState(vars(storage.init(example)), jnp.zeros((batch, slots, length), bool),
                            jnp.uint32(0))

    def add_backfill(state, episodes):
        # make_collect completed the signed return scan before handing us the
        # batch. Real prefix validity must agree with its explicit length.
        real = jnp.arange(length)[None] < episodes["length"][:, None]
        malformed = jnp.any(real != episodes["valid"]) | jnp.any(episodes["length"] <= 0)
        index = state.episodes["current_index"]
        overwritten = jnp.sum(state.fresh[:, index])
        eligible = real & ~episodes["exploration"]
        if "game_id" in episodes:
            episodes = {**episodes, "game_id": state.generation * batch + jnp.arange(batch, dtype=jnp.uint32) + 1}
        updated = storage.add(TrajectoryBufferState(**state.episodes), jax.tree.map(lambda x: x[:, None], episodes))
        return state.replace(episodes=vars(updated), fresh=state.fresh.at[:, index].set(eligible),
                             generation=state.generation + 1), {
            "staging/overwritten_fresh": overwritten,
            "anomalies/selfplay_buffer/eviction_n_is_from_selfplay": jnp.sum(state.episodes["experience"]["length"][:, index]),
            "staging/malformed_episodes": malformed.astype(jnp.int32),
            "staging/new_eligible": jnp.sum(eligible),
            "staging/exploration_excluded": jnp.sum(real & episodes["exploration"]),
        }

    def consume(state, key):
        # Use the exact production randomized top-k/fresh-clearing logic.
        # A slot contains a complete episode; slot+start identify its unroll.
        shape = (batch, slots, length)
        flatten = lambda x: jnp.broadcast_to(x, shape).reshape(batch, slots * length)
        valid = state.fresh.reshape(batch, slots * length)
        view = ConsumptionView(
            global_step_id=jnp.full(valid.shape, state.generation, jnp.uint32),
            is_fresh_i8=valid.astype(jnp.int8), is_valid_sample=valid,
            lane=flatten(jnp.arange(batch)[:, None, None]),
            slot=flatten(jnp.arange(slots)[None, :, None]),
            start=flatten(jnp.arange(length)[None, None, :]))
        production_state = CustomTrajectoryBufferState(
            experience=view, current_index=jnp.int32(0), is_full=jnp.bool_(True),
            num_valid_consumable=jnp.sum(valid))
        after, selected, _ = production.consume(production_state)
        selected = jax.tree.map(lambda x: x[:, 0], selected)
        episodes = jax.tree.map(lambda x: x[selected.lane, selected.slot], state.episodes["experience"])
        unroll = sequences(episodes, key, config["unroll"], starts=selected.start)
        unroll["sample_info"].update(row_id=selected.lane + 1, col_id=selected.slot * length + selected.start,
                                     is_fresh_i8=selected.is_fresh_i8,
                                     is_from_selfplay=selected.is_valid_sample)
        # Defensive masking if called with fewer than consume_size positions.
        # The drain loop below never inserts these partial batches into replay.
        unroll = {**unroll, **{name: unroll[name] & selected.is_valid_sample[:, None]
                              for name in ("policy_mask", "value_mask", "reward_mask")}}
        return state.replace(fresh=after.experience.is_valid_sample.reshape(shape)), unroll

    replay = flashbax.make_trajectory_buffer(
        add_batch_size=consume_size, sample_batch_size=config["train_batch_size"],
        sample_sequence_length=1, period=1, min_length_time_axis=1,
        max_length_time_axis=max(1, config["replay_positions"] // consume_size))

    def drain(state, replay_state, key):
        count = jnp.sum(state.fresh) // consume_size

        def consumption_metrics(unroll, before, after):
            valid = unroll["policy_mask"][:, 0]
            info = unroll["sample_info"]
            prefix = "anomalies/selfplay_buffer-consume/"
            result = {
                prefix + "n_is_valid_but_stale": jnp.sum(valid & (info["is_fresh_i8"] == 0)),
                prefix + "n_is_valid_but_not_real": jnp.sum(valid & ~info["is_from_selfplay"]),
                prefix + "n_is_valid_but_is_pending_reward_i8": jnp.sum(valid & (info["is_pending_reward_i8"] == 1)),
                prefix + "n_is_valid_but_is_exploration": jnp.sum(valid & info["is_exploration"]),
                prefix + "n_not_is_valid_": jnp.sum(~valid),
                prefix + "n_is_valid_but_game_id_zero": jnp.sum(valid & (info["game_id"] == 0)),
            }
            position_ids = jnp.sort(jnp.where(valid, info["row_id"] * slots * length + info["col_id"], -1))
            result[prefix + "n_dupes"] = jnp.sum((position_ids[1:] == position_ids[:-1]) & (position_ids[1:] >= 0))
            ids = jnp.sort(jnp.where(valid, info["game_id"], 0))
            result["selfplay_buffer-consume/n_unique_game_ids"] = jnp.sum((ids != jnp.roll(ids, 1)) & (ids > 0)) + ((ids[0] == ids[-1]) & (ids[0] > 0))
            for name, field in (("term_step", "ep_termination_step"), ("ep_step", "ep_step")):
                result[f"selfplay_buffer-comp/max_{name}"] = jnp.max(jnp.where(valid, info[field], -1))
                result[f"selfplay_buffer-comp/min_{name}"] = jnp.min(jnp.where(valid, info[field], length))
                result[f"selfplay_buffer-comp/avg_{name}"] = jnp.sum(jnp.where(valid, info[field], 0)) / jnp.maximum(jnp.sum(valid), 1)
            for name, values in (("n_is_from_selfplay", info["is_from_selfplay"]), ("n_is_valid", valid),
                                  ("n_is_pending_reward_i8", info["is_pending_reward_i8"]), ("n_is_exploration", info["is_exploration"])):
                result[f"selfplay_buffer-comp/{name}"] = jnp.sum(values)
            after_real = jnp.arange(length)[None, None] < after.episodes["experience"]["length"][..., None]
            result.update({
                "selfplay_buffer-consume/n_consumables": jnp.sum(before.fresh),
                "selfplay_buffer-consume/returnable_n_all_valid_completed": jnp.sum(before.fresh),
                "selfplay_buffer-consume/returnable_n_fresh_returnable_mask": jnp.sum(after.fresh),
                "selfplay_buffer-consume/after_return_n_overall_fresh": jnp.sum(after.fresh),
                "selfplay_buffer-consume/after_return_n_overall_fresh_valid": jnp.sum(after.fresh & after_real),
                "selfplay_buffer-consume/after_return_n_overall_fresh_not_from_selfplay": jnp.sum(after.fresh & ~after_real),
                "selfplay_buffer-consume/after_return_n_overall_fresh_from_selfplay": jnp.sum(after.fresh & after_real),
                "selfplay_buffer-consume/after_return_n_overall_fresh_exploration": jnp.sum(after.fresh & after.episodes["experience"]["exploration"]),
            })
            return result

        # Infer scalar shapes without executing a consume or clearing freshness.
        dummy = jax.eval_shape(consume, state, key)[1]
        zero_batch = jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), dummy)
        initial_metrics = jax.tree.map(jnp.zeros_like, consumption_metrics(zero_batch, state, state))

        def one(_, carry):
            staged, replayed, rng, _ = carry
            rng, subkey = jax.random.split(rng)
            before = staged
            staged, unroll = consume(staged, subkey)
            replayed = replay.add(replayed, jax.tree.map(lambda x: x[:, None], unroll))
            return staged, replayed, rng, consumption_metrics(unroll, before, staged)

        state, replay_state, key, consume_metrics = jax.lax.fori_loop(0, count, one, (state, replay_state, key, initial_metrics))
        return state, replay_state, key, {
            **consume_metrics,
            "drain/num_valid_consumable": count * consume_size + jnp.sum(state.fresh),
            "drain/n_slices": count, "drain/inserted_positions": count * consume_size,
            "drain/remaining_fresh": jnp.sum(state.fresh)}

    return init, add_backfill, consume, drain, replay
