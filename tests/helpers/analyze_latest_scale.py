"""Reference scoring, scale selection, and final held-out comparison. No JAX."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from score_frozen_candidates import score
from summarize_frozen_dynamics import estimate


def save(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False))
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output",type=Path)
    parser.add_argument("--phase",choices=["score","choose","test"],required=True)
    args=parser.parse_args()
    if args.phase=="score":
        rows=json.loads((args.output/"feature-rows.json").read_text())
        grouped={}
        for row in rows:
            entry=grouped.setdefault(row["index"],dict(index=row["index"],moves=row["moves"],candidates=[]))
            entry["candidates"].append(row["move"])
        output=[]
        with ProcessPoolExecutor(max_workers=16) as pool:
            for result in pool.map(score,grouped.values()):
                output.append(result)
                save(args.output/"reference.json",output)
                print(f"scored {len(output)}/{len(grouped)}",flush=True)
        return
    reference={r["index"]:r["scores"] for r in json.loads((args.output/"reference.json").read_text())}
    phase="selection" if args.phase=="choose" else "test"
    rows=json.loads((args.output/f"{phase}-search.json").read_text())
    assert len(rows)==96
    if args.phase=="test" and (args.output/"positive-test-search.json").exists():
        other=json.loads((args.output/"positive-test-search.json").read_text())
        assert len(other)==len(rows)
        for row,extra in zip(rows,other):
            assert row["index"]==extra["index"] and row["moves"]==extra["moves"]
            row["results"].update(extra["results"])
    losses={key:[] for key in rows[0]["results"]}
    losses["prior"]=[]
    for row in rows:
        scores={k:np.clip(v["cp"],-1000,1000) for k,v in reference[row["index"]].items()}
        best=max(scores.values())
        losses["prior"].append(float(best-scores[row["policy_move"]]))
        for key,value in row["results"].items():
            losses[key].append(float(best-scores[value["move"]]))
    report=dict(phase=phase,count=len(rows),losses={k:estimate(v) for k,v in losses.items()})
    if args.phase=="choose":
        key=min((k for k in losses if k!="prior"),key=lambda k:(np.mean(losses[k]),-float(k)))
        report["value_scale"]=float(key)
        report["best_positive_scale"]=float(min((k for k in losses if k!="prior" and float(k)>0),
            key=lambda k:(np.mean(losses[k]),-float(k))))
        save(args.output/"chosen-scale.json",report)
    else:
        key=str(json.loads((args.output/"chosen-scale.json").read_text())["value_scale"])
        report["value_scale"]=float(key)
        report["paired_improvement_vs_current"]=estimate(np.array(losses["1.0"])-losses[key])
        report["paired_improvement_vs_prior"]=estimate(np.array(losses["prior"])-losses[key])
        positive=str(json.loads((args.output/"chosen-scale.json").read_text())["best_positive_scale"])
        if positive in losses:
            report["positive_scale"]=float(positive)
            report["positive_improvement_vs_current"]=estimate(np.array(losses["1.0"])-losses[positive])
            report["positive_improvement_vs_prior"]=estimate(np.array(losses["prior"])-losses[positive])
        report["positions"]=dict(indices=[r["index"] for r in rows],losses=losses)
        save(args.output/"heldout-scale-results.json",report)
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
