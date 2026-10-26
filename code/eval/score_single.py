"""Score single general-adapter predictions.json (from infer_single_adapter.py) using the EXACT metric
functions from the repo evaluation.py. Groups by task type (pooled) AND by (task, dataset) so the
per-dataset macro can be read off ([[project-per-dataset-not-pooled-metric]]). Flags malformed/empty
predictions per task. report_generation is skipped here (scored separately by CRIMSON, score_crimson.py).

  python score_single.py --pred_file preds.json --out_file metrics.json
"""
import os, sys, json, argparse, collections

# locate the repo evaluation.py: prefer the one in this checkout (../../evaluation), else ~/ME-VLIP
_HERE = os.path.dirname(os.path.abspath(__file__))
_CANDIDATES = [_HERE,  # journal repo: evaluation.py sits next to this file (code/eval/)
               os.path.normpath(os.path.join(_HERE, "..", "..", "evaluation")),
               os.path.expanduser("~/ME-VLIP/evaluation")]
for _p in _CANDIDATES:
    if os.path.exists(os.path.join(_p, "evaluation.py")):
        sys.path.insert(0, _p); break
import evaluation as E


def is_malformed(task, ans):
    a = (ans or "").strip()
    if a == "" or a.startswith("Error:"):
        return True
    t = task.lower().strip()
    if t in ("detection", "instance_detection"):
        return not (a.startswith("{") or a.startswith("["))
    if t in ("counting", "regression"):
        import re
        return re.search(r"[-+]?[0-9]*\.?[0-9]+", a) is None
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_file", required=True)
    ap.add_argument("--out_file", required=True)
    args = ap.parse_args()
    data = json.load(open(args.pred_file))

    by_task_pred = collections.defaultdict(list)
    by_task_gt = collections.defaultdict(list)
    by_td_pred = collections.defaultdict(list)   # (task, dataset) -> preds
    by_td_gt = collections.defaultdict(list)
    malformed = collections.Counter()

    for s in data:
        t = s["TaskType"].strip().lower()
        if t == "report_generation":   # scored separately via CRIMSON
            continue
        pred = s.get("Answer", ""); gt = s.get("gt_answer", "")
        by_task_pred[t].append(pred); by_task_gt[t].append(gt)
        by_td_pred[(t, s["dataset"])].append(pred); by_td_gt[(t, s["dataset"])].append(gt)
        if is_malformed(t, pred):
            malformed[t] += 1

    results = {"by_task": {}, "by_task_dataset": {}, "task_macro": {},
               "malformed_counts": dict(malformed),
               "task_counts": {t: len(v) for t, v in by_task_gt.items()}}

    for t in sorted(by_task_gt):
        m = E.calculate_task_metrics(by_task_pred[t], by_task_gt[t], t)
        m["num_examples"] = len(by_task_gt[t])
        results["by_task"][t] = m
        print(f"[{t}] POOLED n={len(by_task_gt[t])} {m}")

    # per-(task,dataset) + per-task macro over its datasets (the ChallengeR-style readout)
    macro_acc = collections.defaultdict(list)
    for (t, ds) in sorted(by_td_gt):
        m = E.calculate_task_metrics(by_td_pred[(t, ds)], by_td_gt[(t, ds)], t)
        m["num_examples"] = len(by_td_gt[(t, ds)])
        results["by_task_dataset"][f"{t}/{ds}"] = m
        # Select by metric name: a valid score can equal num_examples (e.g. F1=1
        # on a singleton dataset). Filtering by numeric value silently dropped it.
        scalar = next((m[k] for k in ("balanced_accuracy", "f1_score", "detection_f1",
                                      "mean_absolute_error") if k in m), None)
        if scalar is not None:
            macro_acc[t].append(scalar)
        print(f"  [{t}/{ds}] n={len(by_td_gt[(t, ds)])} {m}")

    for t, vals in macro_acc.items():
        results["task_macro"][t] = round(sum(vals) / len(vals), 4) if vals else None
        print(f"MACRO[{t}] over {len(vals)} datasets = {results['task_macro'][t]}")

    print("MALFORMED:", dict(malformed))
    os.makedirs(os.path.dirname(os.path.abspath(args.out_file)), exist_ok=True)
    json.dump(results, open(args.out_file, "w"), indent=2)
    print("Saved", args.out_file)


if __name__ == "__main__":
    main()
