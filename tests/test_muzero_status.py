import json

from nanoalphazero.research.muzero.status import inspect_run


def test_status_retains_prior_anomalies_and_tolerates_partial_append(tmp_path):
    run = tmp_path / "hex7-example"
    run.mkdir()
    (run / "config.json").write_text(json.dumps(dict(env="hex7", cycles=100, save_checkpoints=False)))
    (run / "manifest.json").write_text(json.dumps(dict(wandb_url="https://example.invalid/run")))
    records = [dict(cycle=1, **{"anomalies/selfplay_buffer-consume/n_dupes": 2,
                               "errors/selfplay-reward/invalid_0s_aft_term": 1}),
               dict(cycle=2, **{"anomalies/selfplay_buffer-consume/n_dupes": 0,
                                "anomalies/selfplay_buffer/eviction_n_is_from_selfplay": 1000,
                                "runner_state/n_updates": 60})]
    (run / "metrics.jsonl").write_text("\n".join(map(json.dumps, records)) + '\n{"cycle":')
    result = inspect_run(run, "uv run muzero-train --output hex7-example")
    assert result["state"] == "running"
    assert result["cycle"] == 2 and result["updates"] == 60
    assert result["anomalies"] == {"anomalies/selfplay_buffer-consume/n_dupes": 2,
                                   "errors/selfplay-reward/invalid_0s_aft_term": 1}
    assert result["incomplete_metric_records"] == 1
    assert result["checkpoint_files"] == 0


def test_status_distinguishes_pending_and_operator_stopped(tmp_path):
    assert inspect_run(tmp_path / "future")["state"] == "not_started"
    (tmp_path / "operator-status.json").write_text(json.dumps(dict(status="stopped_for_validation")))
    assert inspect_run(tmp_path)["state"] == "stopped_for_validation"


def test_status_keeps_scores_when_mohex_and_self_matches_have_different_cadences(tmp_path):
    rows = [{"cycle": 50, "muzero/eval/search-vs-policy/seat0_score": .7,
             "muzero/eval/policy-vs-search/seat0_score": .5},
            {"cycle": 75, "hex_eval/perfect_opening_wins": 10}]
    (tmp_path / "metrics.jsonl").write_text("\n".join(map(json.dumps, rows)))
    result = inspect_run(tmp_path)
    assert result["last_evaluation_cycle"] == 75
    assert result["last_self_match_cycle"] == 50
    assert result["search_vs_policy_balanced"] == .6


def test_status_checks_flow_across_warmup_and_catches_a_lost_position(tmp_path):
    def row(eligible, inserted, remaining, **extra):
        return {"muzero/staging/new_eligible": eligible,
                "muzero/drain/inserted_positions": inserted,
                "muzero/drain/remaining_fresh": remaining, **extra}
    warmup = row(3, 0, 3, **{"warmup/loop_n": 1,
                             "anomalies/selfplay_buffer-consume/n_dupes": 2})
    (tmp_path / "warmup.jsonl").write_text(json.dumps(warmup))
    rows = [row(3, 4, 2, cycle=1), row(3, 4, 0, cycle=2)]
    (tmp_path / "metrics.jsonl").write_text("\n".join(map(json.dumps, rows)))
    result = inspect_run(tmp_path)
    assert result["warmup_batches"] == 1
    assert result["replay_flow_checks"] == 3
    assert result["replay_flow_failures"] == [{"cycle": 2, "warmup": None,
                                               "incoming": 5, "accounted": 4}]
    assert result["anomalies"] == {"anomalies/selfplay_buffer-consume/n_dupes": 2}
