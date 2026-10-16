"""Paired summaries for frozen probes; bootstrap unit is the root/window."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


def correlation(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 3 or np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    return correlation(rankdata(x), rankdata(y))


def estimate(values):
    x = np.array([v for v in values if v is not None], dtype=float)
    if not len(x):
        return dict(n=0)
    means = np.random.default_rng(20261017).choice(x, (5000, len(x)), replace=True).mean(1)
    return dict(n=len(x), mean=float(x.mean()), median=float(np.median(x)),
                ci95=np.quantile(means, [.025, .975]).tolist())


def regression(train_x, train_y, test_x, test_y):
    coef = np.linalg.lstsq(np.column_stack([np.ones(len(train_x)), train_x]), train_y, rcond=None)[0]
    prediction = np.column_stack([np.ones(len(test_x)), test_x]) @ coef
    mse = np.mean((prediction-test_y)**2)
    return dict(coefficients=coef.tolist(), heldout_mse=float(mse),
                heldout_r2=float(1-mse / np.var(test_y)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = json.loads((args.output / "search.json").read_text())
    previous = json.loads(Path("artifacts/dynamics-comparison-20261016/expanded/results.json").read_text())
    for row in rows:
        assert sum(row["depth_counts"]) == 32
        assert sum(c["visits"] for c in row["candidates"]) == 32
        if row["noise_scale"] == 0:
            chosen = next(c["move"] for c in row["candidates"] if c["selected"])
            assert chosen == previous[row["index"]]["results"]["learned"]["move"]
    sf_path = args.output / "stockfish.json"
    sf = json.loads(sf_path.read_text()) if sf_path.exists() else None
    scores = {r["index"]: r["scores"] for r in sf["records"]} if sf else {}
    report = dict(search={})
    for noise in (0., 1.):
        group = [r for r in rows if r["noise_scale"] == noise]
        summary = {key: estimate([r[key] for r in group]) for key in (
            "target_entropy", "target_max", "prior_entropy", "target_kl_prior",
            "prior_span", "candidate_prior_span", "transformed_span", "candidate_q_span",
            "retained_top128", "candidate_count")}
        depths = np.array([d for r in group for d in r["depths"]])
        summary.update(depth_counts=np.bincount(depths).tolist(), mean_expansion_depth=float(depths.mean()),
            max_depth=int(depths.max()), fraction_depth_above3=float(np.mean(depths > 3)),
            roots_reaching_depth_above3=int(sum(max(r["depths"]) > 3 for r in group)),
            target_max_above95=int(sum(r["target_max"] > .95 for r in group)),
            target_max_above99=int(sum(r["target_max"] > .99 for r in group)))
        measures = {k: [] for k in ("rec_fresh_spearman", "candidate_value_mae", "selected_optimism",
            "backed_up_selected_optimism", "rec_fresh_std_ratio", "error_vs_prior_rank_spearman",
            "rec_sf_spearman", "fresh_sf_spearman", "backed_up_sf_spearman", "fresh_rec_sf_correlation_delta",
            "selected_loss_cp", "rec_argmax_loss_cp", "fresh_argmax_loss_cp", "prior_best_loss_cp")}
        counts = dict(terminal_candidates=0, total_candidates=0, references=0)
        for row in group:
            candidates = row["candidates"]
            counts["terminal_candidates"] += sum(c["terminal"] for c in candidates)
            counts["total_candidates"] += len(candidates)
            # Value discrepancy/optimism excludes known terminal successors;
            # search ranking against SF uses exact terminal reward for fresh Q.
            live = [c for c in candidates if not c["terminal"]]
            rv = np.array([c["recurrent_q"] for c in live])
            fv = np.array([c["fresh_q"] for c in live])
            bq = np.array([c["backed_up_q"] for c in live])
            error = rv-fv
            selected = np.array([c["selected"] for c in live])
            measures["rec_fresh_spearman"].append(spearman(rv, fv))
            if len(live):
                measures["candidate_value_mae"].append(float(np.mean(abs(error))))
                measures["error_vs_prior_rank_spearman"].append(spearman(
                    [c["prior_rank"] for c in live], abs(error)))
                if np.std(fv) > 1e-6:
                    measures["rec_fresh_std_ratio"].append(float(np.std(rv)/np.std(fv)))
                if selected.any():
                    measures["selected_optimism"].append(float(error[selected][0]-error.mean()))
                    measures["backed_up_selected_optimism"].append(float((bq-fv)[selected][0]-(bq-fv).mean()))
            if row["index"] not in scores:
                continue
            counts["references"] += 1
            cp = np.array([np.clip(scores[row["index"]][c["move"]]["cp"], -1000, 1000) for c in candidates])
            rv = np.array([c["recurrent_q"] for c in candidates])
            fv = np.array([c["terminal_reward"] if c["terminal"] else c["fresh_q"] for c in candidates])
            bq = np.array([c["backed_up_q"] for c in candidates])
            rec_rho, fresh_rho = spearman(rv, cp), spearman(fv, cp)
            measures["rec_sf_spearman"].append(rec_rho)
            measures["fresh_sf_spearman"].append(fresh_rho)
            measures["backed_up_sf_spearman"].append(spearman(bq, cp))
            if rec_rho is not None and fresh_rho is not None:
                measures["fresh_rec_sf_correlation_delta"].append(fresh_rho-rec_rho)
            chosen = next(i for i,c in enumerate(candidates) if c["selected"])
            for name, index in [("selected", chosen), ("rec_argmax", rv.argmax()),
                                ("fresh_argmax", fv.argmax()),
                                ("prior_best", np.argmin([c["prior_rank"] for c in candidates]))]:
                measures[name+"_loss_cp"].append(float(cp.max()-cp[index]))
        summary["candidate_metrics"] = {key: estimate(value) for key,value in measures.items()}
        if measures["selected_loss_cp"]:
            summary["candidate_metrics"]["fresh_argmax_improvement_over_selected_cp"] = estimate(
                np.array(measures["selected_loss_cp"])-np.array(measures["fresh_argmax_loss_cp"]))
        summary["counts"] = counts
        report["search"][str(noise)] = summary
    data = np.load(args.output / "persistence.npz")
    rec, fresh, single, target = [data[k] for k in ("recurrent", "fresh", "one_step", "targets")]
    np.testing.assert_allclose(rec[:, 0], fresh[:, 0], atol=1e-6)
    np.testing.assert_allclose(rec[:, 1], single[:, 1], atol=1e-6)
    previous_fit = json.loads(Path("artifacts/value-fit-tpu-20261017/fit-results.json").read_text())
    # BF16 CPU and TPU kernels are not numerically identical. Report the
    # difference; comparisons below all use the same CPU inference outputs.
    report["cpu_vs_previous_tpu_mse_max_abs_difference"] = float(np.max(np.abs(
        np.mean((rec[128:]-target[128:])**2, axis=0) - np.array(
        previous_fit["variants"]["value_only"][0]["heldout"]["value_mse"]))))
    report["persistence"] = []
    for depth in range(1, 6):
        carry = (-1)**depth * rec[:, 0]
        r, f, s, z = rec[:, depth], fresh[:, depth], single[:, depth], target[:, depth]
        train, test = slice(0,128), slice(128,None)
        row = dict(depth=depth, heldout_count=len(r[test]),
            recurrent_outcome_mse=float(np.mean((r[test]-z[test])**2)),
            fresh_outcome_mse=float(np.mean((f[test]-z[test])**2)),
            one_step_outcome_mse=float(np.mean((s[test]-z[test])**2)),
            carry_outcome_mse=float(np.mean((carry[test]-z[test])**2)),
            recurrent_fresh_mse=float(np.mean((r[test]-f[test])**2)),
            carry_fresh_mse=float(np.mean((carry[test]-f[test])**2)),
            one_step_fresh_mse=float(np.mean((s[test]-f[test])**2)),
            paired_refresh_improvement=estimate(((r-f)**2-(s-f)**2)[test]),
            rec_carry_correlation=correlation(r[test], carry[test]),
            fresh_carry_correlation=correlation(f[test], carry[test]),
            successor_change_correlation=correlation((r-carry)[test], (f-carry)[test]),
            recurrent_change_rms=float(np.sqrt(np.mean((r[test]-carry[test])**2))),
            fresh_change_rms=float(np.sqrt(np.mean((f[test]-carry[test])**2))))
        for name, features in [("root_predicts_recurrent", carry[:,None]),
                                ("root_and_fresh_predict_recurrent", np.column_stack([carry,f]))]:
            row[name] = regression(features[train], r[train], features[test], r[test])
        report["persistence"].append(row)
    if sf:
        depths = [v["depth"] for r in sf["records"] for v in r["scores"].values()]
        report["stockfish"] = dict(roots=len(sf["records"]), candidates=len(depths),
            minimum_depth=min(depths), below18=sum(d < 18 for d in depths),
            depth_counts={str(d): depths.count(d) for d in sorted(set(depths))})
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
