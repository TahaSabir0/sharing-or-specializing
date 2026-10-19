#!/usr/bin/env python3
"""Write CRIMSON input pairs from an eval_arm.py predictions file (the first E0/E1 evals used it).

eval_harness.py folders get their pairs from score_run.py; this does the same for eval_arm output:
{gt, pred, sample_key} per report question, pred = raw output with any think block stripped
(not parse_answer's output), exactly as score_run.py builds them.

  python code/eval/eval_arm_pairs.py <run_dir>/predictions_holdout.json <run_dir>/crimson/reportgen_pairs.jsonl
Then: bash code/eval/crimson_box.sh <run_dir>/crimson/reportgen_pairs.jsonl
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vlm_prompt import strip_think_block  # noqa: E402

src, dst = sys.argv[1], sys.argv[2]
rows = [p for p in json.load(open(src)) if p["TaskType"] == "report_generation"]
os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
with open(dst, "w") as f:
    for p in rows:
        pred = strip_think_block(p.get("raw_output", "") or "").strip()
        f.write(json.dumps({"gt": p["gt_answer"], "pred": pred, "sample_key": p["id"]}, ensure_ascii=False) + "\n")
print(f"{len(rows)} report pairs -> {dst}")
