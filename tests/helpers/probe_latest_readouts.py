"""Frozen-state readout probes, with game-disjoint validation and test roots.

Fits only disposable kernel readouts. The MuZero model is never loaded/updated.
Both linear and RBF kernels are centered within roots, so the task is candidate
ranking rather than predicting material advantage shared by all candidates.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

from summarize_frozen_dynamics import estimate, spearman


def centered(x, groups):
    x = np.array(x, dtype=np.float64, copy=True)
    for group in np.unique(groups):
        mask = groups == group
        x[mask] -= x[mask].mean(axis=0)
    return x


def center_kernel(kernel, row_groups, train_groups):
    kernel = centered(kernel, row_groups)
    return centered(kernel.T, train_groups).T


def metrics(prediction, cp, fresh, groups):
    losses, correlations, fresh_correlations = [], [], []
    for group in np.unique(groups):
        mask = groups == group
        p, reference = prediction[mask], cp[mask]
        losses.append(float(reference.max()-reference[np.argmax(p)]))
        correlations.append(spearman(p, reference))
        fresh_correlations.append(spearman(p, fresh[mask]))
    return dict(loss_cp=estimate(losses), spearman_sf=estimate(correlations),
        spearman_fresh=estimate(fresh_correlations), per_root_loss_cp=losses,
        per_root_spearman=correlations, indices=np.unique(groups).tolist())


def write_paired_comparisons(output, report):
    comparisons = {}
    for name, probe in report["probes"].items():
        for label, item in probe["selected"].items():
            heldout = item["heldout"]
            paired = {}
            for reference in ("recurrent", "fresh"):
                baseline = report["baseline"][reference]
                pairs = [(a,b) for a,b in zip(heldout["per_root_spearman"],baseline["per_root_spearman"])
                         if a is not None and b is not None]
                paired["rank_improvement_vs_"+reference] = estimate([a-b for a,b in pairs])
                paired["cp_improvement_vs_"+reference] = estimate(
                    np.array(baseline["per_root_loss_cp"])-heldout["per_root_loss_cp"])
            comparisons[name+"_"+label] = paired
    (output/"paired-readout-comparisons.json").write_text(json.dumps(comparisons,indent=2,allow_nan=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.summarize_only:
        write_paired_comparisons(args.output,json.loads((args.output/"readout-results.json").read_text()))
        return
    raw = np.load(args.output / "features.npz")
    rows = json.loads((args.output / "feature-rows.json").read_text())
    reference = {r["index"]:r["scores"] for r in json.loads((args.output/"reference.json").read_text())}
    live = ~raw["terminal"].astype(bool)
    rows = [r for r, valid in zip(rows, live) if valid]
    groups = np.array([r["index"] for r in rows])
    cp = np.array([np.clip(reference[r["index"]][r["move"]]["cp"],-1000,1000) for r in rows])
    fresh, recurrent = raw["fresh_q"][live], raw["rec_q"][live]
    splits = [groups<128, (groups>=128)&(groups<192), groups>=192]
    gtrain, gval, gtest = [groups[m] for m in splits]
    labels = np.column_stack([centered(cp/1000,groups), centered(fresh,groups)])
    ytrain = labels[splits[0]]
    control_actions=np.zeros((len(rows),134))
    for i,row in enumerate(rows):
        control_actions[i,row["from_square"]]=1
        control_actions[i,64+row["to_square"]]=1
        control_actions[i,128+row["promotion"]]=1
    features = {name:raw[name][live].astype(np.float32) for name in
        ["rec_pen","real_pen","rec_pool","real_pool","rec_spatial","real_spatial"]}
    features["no_dynamics_control"] = np.column_stack([raw["root_pool"][live],control_actions])
    report=dict(train_roots=len(np.unique(gtrain)),validation_roots=len(np.unique(gval)),
        test_roots=len(np.unique(gtest)),nonterminal_candidates=len(rows),
        excluded_terminal_candidates=int((~live).sum()),
        baseline={name:metrics(values[splits[2]],cp[splits[2]],fresh[splits[2]],gtest)
                  for name,values in [("recurrent",recurrent),("fresh",fresh)]}, probes={})
    outputs={}
    for name, source in features.items():
        # Normalization statistics are training-only, including for control inputs.
        mean = source[splits[0]].mean(0)
        std = np.maximum(source[splits[0]].std(0), 1e-3)
        normalized = (source-mean)/std
        train, val, test = [np.asarray(normalized[m],np.float64) for m in splits]
        dimension = train.shape[1]
        dots = [x @ train.T / dimension for x in (train,val,test)]
        distances = [np.maximum(0,np.mean(x*x,axis=1)[:,None]+
            np.mean(train*train,axis=1)[None,:]-2*dot) for x,dot in zip((train,val,test),dots)]
        best = {label:None for label in ("stockfish","fresh_teacher")}
        trials=[]
        for kernel in ("linear", "rbf_0.3", "rbf_1.0", "rbf_3.0"):
            matrices = dots if kernel=="linear" else [np.exp(-float(kernel.split('_')[1])*d) for d in distances]
            kt,kv,ks = [center_kernel(k,g,gtrain) for k,g in zip(matrices,(gtrain,gval,gtest))]
            eigenvalues, eigenvectors = eigh((kt+kt.T)/2,check_finite=False)
            eigenvalues=np.maximum(eigenvalues,0)
            projected=eigenvectors.T @ ytrain
            for alpha in (.001,.01,.1,1.,10.):
                coefficients=eigenvectors @ (projected/(eigenvalues[:,None]+alpha))
                pv,pt=kv @ coefficients,ks @ coefficients
                for column,label in enumerate(("stockfish","fresh_teacher")):
                    # Validation choice uses mean CP loss, never test labels.
                    losses=[]
                    for root in np.unique(gval):
                        mask=gval==root
                        values=cp[splits[1]][mask]
                        losses.append(float(values.max()-values[np.argmax(pv[mask,column])]))
                    loss=float(np.mean(losses))
                    trial=dict(kernel=kernel,alpha=alpha,label=label,validation_loss_cp=loss)
                    trials.append(trial)
                    if best[label] is None or loss < best[label]["validation_loss_cp"]:
                        best[label]=dict(**trial,prediction=pt[:,column].copy())
        report["probes"][name]=dict(dimension=dimension,trials=trials,selected={})
        for label, item in best.items():
            prediction=item.pop("prediction")
            outputs[name+"_"+label]=prediction
            result=metrics(prediction,cp[splits[2]],fresh[splits[2]],gtest)
            result["paired_improvement_vs_recurrent_cp"]=estimate(
                np.array(report["baseline"]["recurrent"]["per_root_loss_cp"])-result["per_root_loss_cp"])
            report["probes"][name]["selected"][label]=dict(**item,heldout=result)
        (args.output/"readout-results.json").write_text(json.dumps(report,indent=2,allow_nan=False))
        print(name,{k:(v["kernel"],v["alpha"],v["validation_loss_cp"],
            v["heldout"]["loss_cp"]["mean"]) for k,v in report["probes"][name]["selected"].items()},flush=True)
    np.savez_compressed(args.output/"readout-test-predictions.npz",**outputs,groups=gtest,
                         cp=cp[splits[2]],fresh=fresh[splits[2]],recurrent=recurrent[splits[2]])
    write_paired_comparisons(args.output,report)
    print("Readout probes finished; checkpoint weights never loaded by this process",flush=True)


if __name__=="__main__":
    main()
