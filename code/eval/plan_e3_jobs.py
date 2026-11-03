#!/usr/bin/env python3
"""Write the E3 (transfer matrix) job files for run_eval_chain.sh / run_crimson_chain.sh.

Each TaskLoRA adapter (6 tasks x 3 seeds x 2 backbones = 36) is evaluated on the FULL holdout,
with the script its baselines used (research rule 3; as E4/E5):
  Qwen      eval_arm.py                       (E0 / E1 baselines)
  MedGemma  eval_harness.py --batch_size 1    (E0 / E2 baselines)
The on-task slice repeats the E1/E2 diagonal: the official diagonal stays E1/E2, the repeat is a
reproducibility check. Outputs go to <runs>/e3/<name>/, never into the training run directory.

One slot per seed: Qwen chain, then MedGemma chain, then that seed's CRIMSON (report slice of the
5 non-report adapters per backbone; mostly junk by design, kept and flagged, PLAN.md E3).

  python code/eval/plan_e3_jobs.py [--runs /data/ugrip2026/journal/runs] [--out code/eval/jobs/e3]
"""
import argparse
from pathlib import Path

SEEDS = (1337, 2026, 42)
TASKS = ("cls", "ml", "det", "count", "reg", "report")
RES = {"q35": 768, "mg15": 896}
DATA = "/data/ugrip2026/journal/data/j26_holdout.jsonl"


def name(backbone, task, seed):
    return f"e3_{backbone}_task{task}_s{seed}"


def adapter(runs, backbone, task, seed):
    return f"{runs}/j26_{backbone}_task{task}_full_{RES[backbone]}_s{seed}"


def eval_line(runs, backbone, task, seed):
    n, out = name(backbone, task, seed), f"{runs}/e3/{name(backbone, task, seed)}"
    if backbone == "q35":
        return (f"{n} --adapter_path {adapter(runs, backbone, task, seed)} --data_path {DATA} "
                f"--output_file {out}/predictions_holdout.json")
    return (f"{n} --mode adapter --adapter {adapter(runs, backbone, task, seed)} --backbone mg15 "
            f"--batch_size 1 --data {DATA} --surface holdout --out_dir {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="/data/ugrip2026/journal/runs")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "jobs" / "e3"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        for backbone in RES:
            script = "eval_arm.py" if backbone == "q35" else "eval_harness.py --batch_size 1 (EVAL_PY)"
            lines = [f"# E3 {backbone} s{seed}: each TaskLoRA on the full holdout, {script}; plan_e3_jobs.py"]
            lines += [eval_line(a.runs, backbone, t, seed) for t in TASKS]
            (out / f"eval_{backbone}_s{seed}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        lines = [f"# E3 CRIMSON s{seed}: report slice of the non-report adapters (junk by design, flagged)",
                 f"WAIT_FOR {a.runs}/evalchain_e3_mg15_s{seed}.log eval chain done"]
        lines += [f"{name(b, t, seed)} {a.runs}/e3/{name(b, t, seed)}" for b in RES for t in TASKS if t != "report"]
        (out / f"crimson_s{seed}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {len(SEEDS) * 3} job files to {out}")


if __name__ == "__main__":
    main()
