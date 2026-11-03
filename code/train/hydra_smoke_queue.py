#!/usr/bin/env python3
"""Run the four Hydra integration smokes sequentially through smoke.sh only.

Requires no Python packages. Stops on any unsuccessful/ambiguous summary. It
never starts full training or holdout inference and never overrides the smoke lock.
"""
import argparse
import ast
import json
import math
from pathlib import Path
import re
import subprocess
import sys

CODE = Path(__file__).resolve().parents[1]
REPO = CODE.parent


def verify(summary, root, backbone, mode):
    text = Path(summary).read_text()
    if not re.search(r"training exit code:\s*0\s+eval:\s*0\b", text):
        raise ValueError("training/eval did not both finish with exit 0")
    match = re.search(r"\[smoke\] veronika: (j26_\S+)", text)
    if match is None:
        raise ValueError("cannot identify the smoke output directory")
    rid = match[1]
    if not rid.startswith(f"j26_{backbone}_{mode}_") or Path(rid).name != rid:
        raise ValueError("unexpected smoke identity")
    run = Path(root) / rid
    spec_name = "target_modules_qwen35.json" if backbone == "q35" else "target_modules_medgemma15.json"
    spec = json.loads((CODE / "train" / spec_name).read_text())
    a = b = 0
    for n, shape in spec["shapes_in_x_out"].values():
        inputs, outputs = map(int, shape.split("x"))
        a += n * 8 * inputs
        b += n * 8 * outputs
    stored = a + 6 * b if mode == "hydraT" else 6 * a + b
    runtime = json.loads((run / "runtime.json").read_text())
    for key, expected in {"optimizer_steps": 50, "trainable_params": stored,
                          "stored_params": stored, "active_params_per_example": spec["P8"]}.items():
        if runtime.get(key) != expected:
            raise ValueError(f"{key}: expected {expected}, got {runtime.get(key)}")
    for key in ("s_per_it", "peak_mem_reserved_gb"):
        value = runtime.get(key, float("nan"))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"invalid runtime field: {key}")
    method = json.loads((run / "method_config.json").read_text())
    if method["method"] != "hydra" or method["cfg"]["mode"] != mode or method["n_params"] != stored:
        raise ValueError("saved method configuration/count does not match the queue")
    losses = [json.loads(line)["loss"] for line in (run / "trainer_log.jsonl").read_text().splitlines()
              if '"loss"' in line]
    if len(losses) < 2 or not all(math.isfinite(x) for x in losses) or losses[-1] >= losses[0]:
        raise ValueError("loss is non-finite or not decreasing; manual review required")
    tasks = json.loads((run / "eval_smoke/metrics.json").read_text())["tasks"]
    labels = set(json.loads((CODE / "data/label_ids.json").read_text())["task"])
    if set(tasks) != labels:
        raise ValueError("evaluation does not cover all six tasks")
    for task, values in tasks.items():
        if values.get("n") != 10 or any(values.get(k) != 0 for k in
                ("malformed_rate", "truncation_rate", "errors")):
            raise ValueError(f"{task}: format/truncation/errors need manual review")
    route = re.search(r"5\. routing_stats\.npz: (\{[^\n]+\})", text)
    expected = {"sequences": 60, "layers": spec["n_modules"], "experts": 6}
    if route is None or ast.literal_eval(route[1]) != expected:
        raise ValueError("missing or invalid routing summary")
    return rid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_root", required=True, type=Path, help="new directory in your personal folder")
    parser.add_argument("--stage", choices=("all", "qwen", "medgemma"), default="all")
    args = parser.parse_args()
    dest = args.out_root.resolve()
    if dest.is_relative_to(REPO) or dest.exists():
        raise SystemExit("use a new output directory outside the source checkout")
    if subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True).strip():
        raise SystemExit("source checkout is dirty; preserve/commit changes before validating this revision")
    dest.mkdir(parents=True)
    root = Path("/data/ugrip2026/journal/runs/smoke")
    backbones = ("q35", "mg15") if args.stage == "all" else (("q35",) if args.stage == "qwen" else ("mg15",))
    completed = []
    for backbone in backbones:
        folder = "v2" if backbone == "q35" else "e2"
        for mode in ("hydraT", "hydraR"):
            cfg = f"code/configs/{folder}/{mode}_s1337.yaml"
            summary = dest / f"{backbone}_{mode}.log"
            print(f"START {backbone} {mode}: {summary}", flush=True)
            with summary.open("w") as log:
                result = subprocess.run(["bash", "code/train/smoke.sh", "veronika", cfg], cwd=REPO,
                                        stdout=log, stderr=subprocess.STDOUT)
            print(summary.read_text(), flush=True)
            if result.returncode:
                raise SystemExit(f"STOP: smoke launcher exited {result.returncode}; see {summary}")
            try:
                rid = verify(summary, root, backbone, mode)
            except (ValueError, OSError, KeyError) as exc:
                raise SystemExit(f"STOP: {exc}; see {summary}") from exc
            completed.append(rid)
            (dest / "completed.json").write_text(json.dumps(completed, indent=2) + "\n")
            print(f"PASS {rid}", flush=True)
    from collect_hydra_smoke import collect
    collect(root, completed, dest / "smoke-evidence.tar.gz")
    print(f"ALL REQUESTED SMOKES PASSED; evidence: {dest / 'smoke-evidence.tar.gz'}", flush=True)


if __name__ == "__main__":
    main()
