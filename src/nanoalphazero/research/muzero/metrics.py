"""AlphaZero metric names with explicit root/sequence and collection semantics."""

import numpy as np


RENAMES = {
    "loss": "total_loss", "policy_loss": "loss_pi", "value_loss": "loss_v",
    "reward_loss": "muzero/loss_r", "grad_norm": "norms/grad_norm",
    "updates": "runner_state/n_updates", "seconds": "timing/cycle_seconds",
    "elapsed_seconds": "timing/training_seconds", "cycle_wall_seconds": "loop/loop_duration",
    "diagnostics/opening_value_mse": "hex_perf/mse_vs_perfect",
    "diagnostics/opening_value_sign_accuracy": "hex_perf/sign_accuracy",
    "staging/overwritten_fresh": "anomalies/selfplay_buffer/eviction_n_isvalid_fresh",
    "staging/exploration_excluded": "selfplay_buffer/n_excluded_exploration",
    "drain/inserted_positions": "muzero/drain/inserted_positions",
    "drain/remaining_fresh": "muzero/drain/remaining_fresh",
}
AZ_PREFIXES = (
    "norms/", "timing/", "loop/", "warmup/", "runner_state/", "selfplay/", "selfplay-reward/",
    "selfplay_buffer", "selfplay-intr/", "train_batch/", "training/", "drain/", "entropy/", "wdl/",
    "loss_per_step_", "errors/", "anomalies/", "openings_", "stats/", "debug/",
    "model/", "params-logits/", "hex_perf/", "hex_eval/",
)


def standardize(metrics):
    """One boundary for local JSON and W&B; extras cannot scatter namespaces."""
    result = {}
    for name, value in metrics.items():
        name = RENAMES.get(name, name)
        if name not in ("cycle", "total_loss", "loss_pi", "loss_v") and not name.startswith((*AZ_PREFIXES, "muzero/")):
            name = "muzero/" + name
        if name in result:
            raise ValueError(f"Metric collision: {name}")
        result[name] = value
    return result


def _mean(values, mask):
    return float(np.sum(np.where(mask, values, 0)) / max(np.sum(mask), 1))


def batch_metrics(batch, allows_draws):
    """AZ counters count B root positions, never B*(K+1) unroll targets."""
    values = np.asarray(batch["value"])[:, 0]
    valid = np.asarray(batch["policy_mask"])[:, 0]
    info = {k: np.asarray(v) for k, v in batch["sample_info"].items()}
    fresh = info["is_fresh_i8"]
    produced = (fresh == 1) & info["is_from_selfplay"]
    invalid = ~produced | info["is_exploration"] | (info["is_pending_reward_i8"] == 1)
    # A zero boundary bootstrap is legitimate, including in no-draw games.
    invalid_no_draw = invalid | ((values == 0) & info["terminal"])
    output = {
        "train_batch/n_reward1": int(np.sum(values == 1)),
        "train_batch/n_rewardneg1": int(np.sum(values == -1)),
        "train_batch/n_reward0": int(np.sum(values == 0)),
        "train_batch/n_valid": int(np.sum(valid)), "train_batch/n_invalid": int(np.sum(~valid)),
        "train_batch/n_from_selfplay_samples": int(np.sum(info["is_from_selfplay"])),
        "train_batch/n_is_fresh_i8_eq1": int(np.sum(fresh == 1)),
        "train_batch/n_is_fresh_i8_eq0": int(np.sum(fresh == 0)),
        "train_batch/n_unique_game_ids": int(len(np.unique(info["game_id"][valid]))),
        "anomalies/train_batch/n_invalid_term_at_step0": int(np.sum(valid & info["terminal"] & (info["ep_termination_step"] == 0))),
        "anomalies/train_batch/n_invalid_term_at_step1": int(np.sum(valid & info["terminal"] & (info["ep_termination_step"] == 1))),
        "anomalies/train_batch/n_is_fresh_i8_gte_2": int(np.sum(fresh >= 2)),
        "anomalies/train_batch/n_not_is_fresh_i8_but_valid": int(np.sum(valid & (fresh != 1))),
        "anomalies/train_batch/n_invalid_mismatches_chess": int(np.sum(~valid) - np.sum(invalid)),
        "anomalies/train_batch/n_invalid_mismatches_hex": int(np.sum(~valid) - np.sum(invalid_no_draw)),
        "debug/batch_checksum": float(np.sum(info["row_id"])),
        "muzero/train_batch/n_bootstrapped_roots": int(np.sum(valid & ~info["terminal"])),
        "muzero/train_batch/n_non_outcome_values": int(np.sum(valid & ~np.isin(values, [-1, 0, 1]))),
        "muzero/train_batch/n_real_targets": int(np.sum(batch["policy_mask"])),
        "muzero/train_batch/n_absorbing_targets": int(np.sum(np.asarray(batch["value_mask"]) & ~np.asarray(batch["policy_mask"]) & info["terminal"][:, None])),
        "muzero/train_batch/n_boundary_bootstraps": int(np.sum(np.asarray(batch["value_mask"]) & ~np.asarray(batch["policy_mask"]) & ~info["terminal"][:, None])),
    }
    suffix = "envallowsdraws" if allows_draws else "envforbidsdraws"
    output[f"train_batch/n_is_invalid_{suffix}_doublecheck"] = int(np.sum(invalid if allows_draws else invalid_no_draw))
    return output


def collection_metrics(episodes, global_step, allows_draws):
    """Aggregate actual transitions in the collection, excluding padded work.

    AZ logs its last asynchronous step; synchronous collection aggregates the
    whole collection. Names/units match, but the aggregation window is explicit.
    """
    e = {k: np.asarray(v) for k, v in episodes.items()}
    real, terminal, lengths = e["valid"], e["terminal"], e["length"]
    steps = np.broadcast_to(np.arange(real.shape[1]), real.shape)
    term_steps = lengths[terminal] - 1
    last = steps == lengths[:, None] - 1
    just_terminal = real & last & terminal[:, None]
    legal = e["next_legal_count"]
    rewards = e["reward"]
    output = {
        "selfplay/global_step": global_step,
        "selfplay/ep_term_step_min": int(np.min(term_steps)) if len(term_steps) else -1,
        "selfplay/ep_term_step_max": int(np.max(term_steps)) if len(term_steps) else -1,
        "selfplay/ep_term_step_avg": float(np.mean(term_steps)) if len(term_steps) else -1.,
        "selfplay/p1_wins": int(np.sum(terminal & (e["final_rewards"][:, 0] == 1))),
        "selfplay/p2_wins": int(np.sum(terminal & (e["final_rewards"][:, 1] == 1))),
        "selfplay/p_just_tied": int(np.sum(terminal & np.all(e["final_rewards"] == 0, -1))),
        "selfplay/n_legal_moves_min": int(np.min(legal[real])),
        "selfplay/n_legal_moves_max": int(np.max(legal[real])),
        "selfplay/n_legal_moves_avg": _mean(legal, real),
        "selfplay/n_legal_moves_avg_early": _mean(legal, real & (steps <= 10)),
        "selfplay/n_legal_moves_avg_mid": _mean(legal, real & (steps > 10) & (steps <= 30)),
        "selfplay/n_legal_moves_avg_late": _mean(legal, real & (steps > 30)),
        "selfplay/ep_step_min": int(np.min(lengths - 1)),
        "selfplay/ep_step_max": int(np.max(lengths - 1)),
        "selfplay/ep_step_std": float(np.std(lengths - 1)),
        "selfplay-reward/valid_1s_aft_term": int(np.sum(just_terminal & (np.abs(rewards) == 1))),
        "selfplay-reward/valid_0s_no_term": int(np.sum(real & ~just_terminal & (rewards == 0))),
        "selfplay_buffer/n_all_just_terminated": int(np.sum(lengths[terminal])),
        "selfplay_buffer/n_rows_just_terminated": int(np.sum(terminal)),
        "selfplay_buffer/n_hit_max_ep_step_ROWS": int(np.sum(lengths == real.shape[1])),
        "selfplay_buffer/n_hit_max_ep_step_ENTRIES": int(np.sum(lengths[lengths == real.shape[1]])),
        "selfplay_buffer-comp/n_es_term": int(np.sum(terminal)),
        "selfplay_buffer-comp/n_entries_to_update": int(np.sum(lengths[terminal])),
    }
    if allows_draws:
        output["selfplay-reward/valid_0s_aft_term"] = int(np.sum(just_terminal & (rewards == 0)))
    else:
        output["errors/selfplay-reward/invalid_0s_aft_term"] = int(np.sum(just_terminal & (rewards == 0)))
        output["errors/selfplay-reward/invalid_1s_no_term"] = int(np.sum(real & ~just_terminal & (rewards != 0)))
    logits = e["raw_policy_logits"]
    logits = logits - np.max(logits, -1, keepdims=True)
    log_probs = logits - np.log(np.sum(np.exp(logits), -1, keepdims=True))
    policy = e["policy"]
    kl = np.sum(policy * (np.log(np.maximum(policy, 1e-30)) - log_probs), -1)
    selected = kl[real]
    output.update({"selfplay-intr/n_kl_close_to_zero": int(np.sum(selected < 1e-5)),
                   "selfplay-intr/n_kl_le_one": int(np.sum(selected <= 1)),
                   "selfplay-intr/kl_mean": float(np.mean(selected)),
                   "selfplay-intr/kl_std": float(np.std(selected)),
                   "selfplay-intr/kl_max": float(np.max(selected))})
    for name, q in (("p25", 25), ("median", 50), ("p75", 75), ("p90", 90), ("p95", 95)):
        output[f"selfplay-intr/kl_{name}"] = float(np.percentile(selected, q))
    peak = np.zeros_like(real)
    peak[np.arange(len(real)), np.argmax(np.where(real, kl, -np.inf), -1)] = True
    peak &= terminal[:, None]
    others = real & terminal[:, None] & ~peak
    diff = np.abs(e["predicted_value"] - e["value"])
    for name, values in (("value_diff", diff), ("kl_divergence", kl)):
        output[f"selfplay_buffer/{name}_peak_interest"] = _mean(values, peak)
        output[f"selfplay_buffer/{name}_others"] = _mean(values, others)
    output["selfplay_buffer/num_peak_samples"] = int(np.sum(peak))
    if policy.shape[-1] <= 100:
        for i in range(policy.shape[-1]):
            output[f"openings_explore/{i}"] = int(np.sum((e["action"][:, 0] == i) & e["exploration"][:, 0]))
            output[f"openings_exploit/{i}"] = int(np.sum((e["action"][:, 0] == i) & ~e["exploration"][:, 0]))
    return output


def histogram_data(episodes, batch, consumed=None):
    e, info = episodes, batch["sample_info"]
    valid = np.asarray(batch["policy_mask"])[:, 0]
    arrays = {
        "selfplay/opening_actions_distribution": np.asarray(e["action"])[:, 0],
        "selfplay/opening_actions_distribution_not_explore": np.asarray(e["action"])[:, 0][~np.asarray(e["exploration"])[:, 0]],
        "train_batch/ep_step_distribution": np.asarray(info["ep_step"])[valid],
        "train_batch/opening_actions_distribution": np.asarray(info["action"])[valid & (np.asarray(info["ep_step"]) == 0)],
        "train_batch/sampled_row_ids_distribution": np.asarray(info["row_id"])[valid],
        "train_batch/sampled_game_ids_distribution": np.asarray(info["game_id"])[valid],
    }
    if "col_id" in info:
        arrays["train_batch/sampled_col_ids_distribution"] = np.asarray(info["col_id"])[valid]
    logits = np.asarray(e["raw_policy_logits"])
    logits = logits - np.max(logits, -1, keepdims=True)
    log_probs = logits - np.log(np.sum(np.exp(logits), -1, keepdims=True))
    policies = np.asarray(e["policy"])
    kl = np.sum(policies * (np.log(np.maximum(policies, 1e-30)) - log_probs), -1)
    arrays["selfplay-intr/current_kl_divergence_distribution"] = kl[np.asarray(e["valid"])]
    if consumed is not None:
        for label, field in (("col_ids", "col_id"), ("valid_col_ids", "col_id"),
                             ("row_ids", "row_id"), ("valid_row_ids", "row_id"),
                             ("ep_step_all", "ep_step"), ("valid_game_ids", "game_id")):
            arrays[f"selfplay_buffer-comp-histogram/{label}_distribution"] = np.asarray(consumed[field])
    return {name: values for name, values in arrays.items() if len(values)}
