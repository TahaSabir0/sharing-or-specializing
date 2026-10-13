#!/usr/bin/env python3
"""Score one eval_harness output folder (PLAN.md §1.8). Uses the repo scorer; nothing re-implemented.

  python code/eval/score_run.py --out_dir <eval dir>

Writes into <out_dir>:
  metrics_score_single.json   score_single.py output (per dataset + task macro; reports skipped)
  reportgen_pairs.jsonl       {gt, pred, sample_key} for CRIMSON and the omission rule
  omission.json               reportgen_omission.profile (normal-on-abnormal rate etc.)
  metrics.json                the record: per task n, per-dataset metric, task macro, malformed
                              rate, truncation rate, error count; report: CRIMSON (if
                              crimson_metrics.json exists; run crimson_ws.sh first) + omission.

Report predictions are free text: the CRIMSON/omission `pred` is the raw output with any think
block stripped, NOT parse_answer's output (which keeps only the text after the first newline).
"""
import argparse
import collections
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from reportgen_omission import profile  # noqa: E402
from vlm_prompt import strip_think_block  # noqa: E402

REPORT = "report_generation"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    d = a.out_dir
    preds = json.load(open(os.path.join(d, "all_predictions.json")))
    meta = json.load(open(os.path.join(d, "run_meta.json")))

    # --- score_single on the non-report tasks (it skips reports itself) ---
    ss_out = os.path.join(d, "metrics_score_single.json")
    subprocess.run([sys.executable, os.path.join(HERE, "score_single.py"),
                    "--pred_file", os.path.join(d, "all_predictions.json"), "--out_file", ss_out],
                   check=True, stdout=subprocess.DEVNULL)
    ss = json.load(open(ss_out))

    by_task = collections.defaultdict(list)
    for p in preds:
        by_task[p["TaskType"]].append(p)

    metrics = {"surface": meta.get("surface"), "mode": meta.get("mode"), "protocol": meta.get("protocol"),
               "data": meta.get("data"), "code_commit": meta.get("code_commit"), "tasks": {}}
    for task, ps in sorted(by_task.items()):
        n = len(ps)
        rec = {"n": n,
               "truncation_rate": round(sum(bool(p.get("truncated")) for p in ps) / n, 4),
               "errors": sum(str(p.get("Answer", "")).startswith("Error:") for p in ps),
               "mean_new_tokens": round(sum(p.get("n_new_tokens", 0) for p in ps) / n, 1)}
        if task != REPORT:
            t = task.lower().strip()
            rec["malformed_rate"] = round(ss["malformed_counts"].get(t, 0) / n, 4)
            rec["per_dataset"] = {k.split("/", 1)[1]: v for k, v in ss["by_task_dataset"].items()
                                  if k.split("/", 1)[0] == t}
            rec["macro"] = ss["task_macro"].get(t)
        metrics["tasks"][task] = rec

    # --- report generation: pairs, omission, CRIMSON ---
    if REPORT in by_task:
        pairs = os.path.join(d, "reportgen_pairs.jsonl")
        with open(pairs, "w") as f:
            for p in by_task[REPORT]:
                pred = strip_think_block(p.get("raw_output", "") or "").strip()
                f.write(json.dumps({"gt": p["gt_answer"], "pred": pred, "sample_key": p["id"]},
                                   ensure_ascii=False) + "\n")
        om, _, _ = profile(os.path.basename(os.path.normpath(d)), pairs)
        json.dump(om, open(os.path.join(d, "omission.json"), "w"), indent=1)
        rec = metrics["tasks"][REPORT]
        rec["omission_normal_on_abnormal_rate"] = om["normal_on_abnormal_rate"]
        rec["malformed_rate"] = round(sum(not (p.get("raw_output") or "").strip() for p in by_task[REPORT])
                                      / len(by_task[REPORT]), 4)  # empty output
        cm = os.path.join(d, "crimson_metrics.json")
        if os.path.exists(cm):
            c = json.load(open(cm))
            rec["per_dataset"] = {"IU_XRay": {"crimson_score": c["crimson_score_mean"], "n": c["n"],
                                              "n_scored": c["n_scored"]}}
            rec["macro"] = c["crimson_score_mean"]
        else:
            rec["macro"] = None
            rec["note"] = "CRIMSON pending: run code/eval/crimson_ws.sh on this folder"

    json.dump(metrics, open(os.path.join(d, "metrics.json"), "w"), indent=1)
    print(f"[{metrics['surface']}] {meta.get('mode')} {meta.get('protocol') or ''}")
    for task, r in metrics["tasks"].items():
        print(f"  {task:28s} n={r['n']:5d} macro={r.get('macro')} malformed={r.get('malformed_rate')} "
              f"trunc={r['truncation_rate']} err={r['errors']}")
    print("->", os.path.join(d, "metrics.json"))


if __name__ == "__main__":
    main()
