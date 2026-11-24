#!/usr/bin/env python3
"""E9-lite paired contrasts: TaskLoRA-cls − Shared on the holdout classification slice, per data fraction.

Same statistics as E1/E2 (bootstrap.py `seeds`): two-level bootstrap (resample the 3 seeds, then holdout
questions stratified by dataset), statistic = mean over seeds of the classification dataset macro (BA),
paired on identical questions. Holm across the 6 fractions (one comparison family). 100% = the E1 runs
(their full-holdout predictions filtered to classification).

  python code/analysis/e9_bootstrap.py --out <dir> [--B 10000] [--runs /data/ugrip2026/journal/runs]
CPU only. Writes contrasts.json and a printed table.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bootstrap as bs  # noqa: E402

SEEDS = (1337, 2026, 42)
FRACTIONS = ("cls1", "cls5", "cls10", "cls25", "cls50", "full")
TASK = "classification"


def run_dir(runs, arm, frac, seed):
    return os.path.join(runs, f"j26_q35_{arm}_{frac}_768_s{seed}")


def load_cls(pred_file):
    """predictions_holdout.json -> bootstrap.load_run-style {classification: {...}} (cls rows only)."""
    preds = [p for p in json.load(open(pred_file)) if p["TaskType"] == TASK]
    if not preds:
        raise SystemExit(f"no classification rows in {pred_file}")
    t = {"ids": [], "dataset": [], "pred": [], "gt": [], "score": []}
    for p in sorted(preds, key=lambda p: p["id"]):
        t["ids"].append(p["id"]); t["dataset"].append(p["dataset"])
        t["pred"].append(p["Answer"]); t["gt"].append(p["gt_answer"]); t["score"].append(None)
    t["dataset"] = np.array(t["dataset"])
    return {TASK: t}


def contrast(args):
    frac, runs, B, seed = args
    a = [load_cls(os.path.join(run_dir(runs, "shared", frac, s), "predictions_holdout.json")) for s in SEEDS]
    b = [load_cls(os.path.join(run_dir(runs, "taskcls", frac, s), "predictions_holdout.json")) for s in SEEDS]
    for r in a + b:
        if len(r[TASK]["ids"]) != 1689:
            raise SystemExit(f"{frac}: expected 1689 cls questions, got {len(r[TASK]['ids'])}")
    st = bs.strata(a[0][TASK])
    res = bs.seeds(a, b, B, seed)[TASK]
    res.pop("p_holm", None)  # one task per call; Holm is applied across fractions below
    res["per_seed_a"] = [bs.macro(r[TASK], TASK, st) for r in a]
    res["per_seed_b"] = [bs.macro(r[TASK], TASK, st) for r in b]
    return frac, res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=os.environ.get("JOURNAL_RUNS", "/data/ugrip2026/journal/runs"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261013)
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists; use a fresh directory")
    with ProcessPoolExecutor(max_workers=len(FRACTIONS)) as ex:
        res = dict(ex.map(contrast, [(f, a.runs, a.B, a.seed) for f in FRACTIONS]))
    for frac, p in bs.holm({f: r["p"] for f, r in res.items()}).items():
        res[frac]["p_holm"] = p
    out = {"surface": "holdout (classification slice, 1689 q)", "a": "shared", "b": "taskcls",
           "statistic": "mean over seeds of cls dataset-macro BA; diff = TaskLoRA - Shared",
           "seeds": list(SEEDS), "B": a.B, "random_seed": a.seed, "holm_family": list(FRACTIONS),
           "per_fraction": {f: res[f] for f in FRACTIONS}}
    os.makedirs(a.out)
    json.dump(out, open(os.path.join(a.out, "contrasts.json"), "w"), indent=1, allow_nan=False)
    print(f"{'frac':>6} {'Shared':>7} {'TaskLoRA':>8} {'diff':>7}  {'95% CI':>17}  {'p':>6} {'p_holm':>6}")
    for f in FRACTIONS:
        r = res[f]
        print(f"{f:>6} {r['mean_a']:7.3f} {r['mean_b']:8.3f} {r['diff_b_minus_a']:+7.3f}  "
              f"[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]  {r['p']:6.4f} {r['p_holm']:6.4f}")


if __name__ == "__main__":
    main()
