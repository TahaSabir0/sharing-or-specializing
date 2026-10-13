#!/usr/bin/env python3
"""Bootstrap statistics for the journal (PLAN.md §1.11). Metrics come from the repo scorer
(code/eval/evaluation.py); this file only resamples.

- Per-task CI: 10k resamples of holdout questions, stratified by dataset (each dataset resampled
  with replacement at its own size), dataset macro recomputed on each resample.
- Paired difference (B − A) on the same questions: identical resample indices for both runs.
- Two-level seed bootstrap: resample seeds, then questions; statistic = mean over sampled seeds.
- Holm–Bonferroni across the tasks of one pairwise comparison (adjusted and unadjusted p).

Report generation uses CRIMSON per-sample scores (crimson_per_sample.jsonl), so its macro is the
mean score (one dataset). Orientation: MAE is lower-is-better; differences are reported raw
(B − A) with `higher_is_better` alongside, so a negative MAE difference means B is better.

  python code/analysis/bootstrap.py ci     --run <eval_dir>                       [--B 10000]
  python code/analysis/bootstrap.py paired --a <eval_dir> --b <eval_dir>
  python code/analysis/bootstrap.py seeds  --a d1,d2,d3 --b e1,e2,e3
Writes JSON to stdout or --out.
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval"))
import evaluation as E  # noqa: E402

REPORT = "report_generation"
HIGHER_IS_BETTER = {"classification": True, "multi-label classification": True, "detection": True,
                    "counting": False, "regression": False, REPORT: True}


# ------------------------------------------------------------------------------- loading
def load_run(d):
    """-> {task: {"ids": [...], "dataset": np.array, "pred": [...], "gt": [...], "score": np.array|None}}"""
    preds = json.load(open(os.path.join(d, "all_predictions.json")))
    crim = {}
    cp = os.path.join(d, "crimson_per_sample.jsonl")
    if os.path.exists(cp):
        for l in open(cp):
            r = json.loads(l)
            crim[r["sample_key"]] = r["crimson_score"]
    out = collections.defaultdict(lambda: {"ids": [], "dataset": [], "pred": [], "gt": [], "score": []})
    for p in sorted(preds, key=lambda p: p["id"]):
        t = out[p["TaskType"]]
        t["ids"].append(p["id"])
        t["dataset"].append(p["dataset"])
        t["pred"].append(p["Answer"])
        t["gt"].append(p["gt_answer"])
        t["score"].append(crim.get(p["id"]) if p["TaskType"] == REPORT else None)
    for task, t in out.items():
        t["dataset"] = np.array(t["dataset"])
        if task == REPORT:
            if any(s is None for s in t["score"]):
                t["score"] = None  # CRIMSON not run (or incomplete): report excluded
            else:
                t["score"] = np.array(t["score"], dtype=float)
    return dict(out)


def headline(metric_dict):
    for k in ("balanced_accuracy", "f1_score", "detection_f1", "mean_absolute_error"):
        if k in metric_dict:
            return metric_dict[k]
    raise KeyError(metric_dict)


# ------------------------------------------------------------------------------- statistic
def macro(t, task, idx_by_ds):
    """Unweighted mean over datasets of the per-dataset metric, on the given indices."""
    vals = []
    for ds, idx in idx_by_ds.items():
        if task == REPORT:
            vals.append(float(t["score"][idx].mean()))
        else:
            vals.append(headline(E.calculate_task_metrics([t["pred"][i] for i in idx],
                                                          [t["gt"][i] for i in idx], task)))
    return float(np.mean(vals))


def strata(t):
    return {ds: np.where(t["dataset"] == ds)[0] for ds in sorted(set(t["dataset"]))}


def resample(st, rng):
    return {ds: rng.choice(idx, size=len(idx), replace=True) for ds, idx in st.items()}


def usable(t, task):
    return task != REPORT or t["score"] is not None


# ------------------------------------------------------------------------------- analyses
def ci(run, B, seed, alpha=0.05):
    rng = np.random.default_rng(seed)
    res = {}
    for task, t in run.items():
        if not usable(t, task):
            continue
        st = strata(t)
        point = macro(t, task, st)
        boots = np.array([macro(t, task, resample(st, rng)) for _ in range(B)])
        res[task] = {"macro": point, "ci95": [float(np.quantile(boots, alpha / 2)),
                                               float(np.quantile(boots, 1 - alpha / 2))],
                     "n": len(t["ids"]), "B": B, "higher_is_better": HIGHER_IS_BETTER[task]}
    return res


def align(ta, tb):
    if ta["ids"] != tb["ids"]:
        common = sorted(set(ta["ids"]) & set(tb["ids"]))
        raise SystemExit(f"runs differ in question ids ({len(ta['ids'])} vs {len(tb['ids'])}, "
                         f"{len(common)} common): paired tests need identical question sets")


def p_two_sided(diffs, point):
    """Bootstrap p for H0: difference = 0 (shift method: centre the distribution at 0)."""
    centred = diffs - diffs.mean()
    return float(min(1.0, (np.abs(centred) >= abs(point)).mean() + 1.0 / (len(diffs) + 1)))


def holm(pvals):
    """{task: p} -> {task: Holm-adjusted p} (step-down, monotone)."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, adj, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    return adj


def paired(run_a, run_b, B, seed, alpha=0.05):
    rng = np.random.default_rng(seed)
    res, pv = {}, {}
    for task in sorted(set(run_a) & set(run_b)):
        ta, tb = run_a[task], run_b[task]
        if not (usable(ta, task) and usable(tb, task)):
            continue
        align(ta, tb)
        st = strata(ta)
        point = macro(tb, task, st) - macro(ta, task, st)
        diffs = []
        for _ in range(B):
            s = resample(st, rng)
            diffs.append(macro(tb, task, s) - macro(ta, task, s))
        diffs = np.array(diffs)
        pv[task] = p_two_sided(diffs, point)
        res[task] = {"diff_b_minus_a": point,
                     "ci95": [float(np.quantile(diffs, alpha / 2)), float(np.quantile(diffs, 1 - alpha / 2))],
                     "p": pv[task], "higher_is_better": HIGHER_IS_BETTER[task], "B": B}
    for task, p in holm(pv).items():
        res[task]["p_holm"] = p
    return res


def seeds(runs_a, runs_b, B, seed, alpha=0.05):
    """Two-level bootstrap of mean(B seeds) − mean(A seeds): resample seeds, then questions."""
    rng = np.random.default_rng(seed)
    res, pv = {}, {}
    tasks = set.intersection(*[set(r) for r in runs_a + runs_b])
    for task in sorted(tasks):
        ta, tb = [r[task] for r in runs_a], [r[task] for r in runs_b]
        if not all(usable(t, task) for t in ta + tb):
            continue
        for t in ta[1:] + tb:
            align(ta[0], t)
        st = strata(ta[0])
        ma = [macro(t, task, st) for t in ta]
        mb_ = [macro(t, task, st) for t in tb]
        point = float(np.mean(mb_) - np.mean(ma))
        diffs = []
        for _ in range(B):
            sa = rng.integers(0, len(ta), len(ta))
            sb = rng.integers(0, len(tb), len(tb))
            s = resample(st, rng)
            diffs.append(np.mean([macro(tb[i], task, s) for i in sb]) - np.mean([macro(ta[i], task, s) for i in sa]))
        diffs = np.array(diffs)
        pv[task] = p_two_sided(diffs, point)
        res[task] = {"mean_a": float(np.mean(ma)), "sd_a": float(np.std(ma, ddof=1)) if len(ma) > 1 else None,
                     "mean_b": float(np.mean(mb_)), "sd_b": float(np.std(mb_, ddof=1)) if len(mb_) > 1 else None,
                     "diff_b_minus_a": point,
                     "ci95": [float(np.quantile(diffs, alpha / 2)), float(np.quantile(diffs, 1 - alpha / 2))],
                     "p": pv[task], "higher_is_better": HIGHER_IS_BETTER[task], "B": B}
    for task, p in holm(pv).items():
        res[task]["p_holm"] = p
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["ci", "paired", "seeds"])
    ap.add_argument("--run")
    ap.add_argument("--a")
    ap.add_argument("--b")
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261013)
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.cmd == "ci":
        res = {"run": a.run, "per_task": ci(load_run(a.run), a.B, a.seed)}
    elif a.cmd == "paired":
        res = {"a": a.a, "b": a.b, "per_task": paired(load_run(a.a), load_run(a.b), a.B, a.seed)}
    else:
        ra = [load_run(d) for d in a.a.split(",")]
        rb = [load_run(d) for d in a.b.split(",")]
        res = {"a": a.a.split(","), "b": a.b.split(","), "per_task": seeds(ra, rb, a.B, a.seed)}
    s = json.dumps(res, indent=1)
    if a.out:
        open(a.out, "w").write(s)
    print(s)


if __name__ == "__main__":
    main()
