#!/usr/bin/env python3
"""Create balanced real-training and holdout-evaluation queues for E4/E5.

This is planning only: it never starts a process or touches a run directory. It consumes the
``GROUP_MANIFEST.json`` produced by ``build_group_decks.py``, validates every checked-in config,
then uses longest-processing-time scheduling over three chains. Cost is proportional to training
rows; MedGemma receives PLAN.md's estimated 0.6 relative cost factor (30/50).

The job files are for the existing runners, so E4/E5 launch exactly like E1/E2:
  train_chain_{1,2,3}.txt  config paths for code/train/run_chain.sh (export JOURNAL_DATA first)
  eval_q35.txt             code/eval/run_eval_chain.sh with eval_arm.py, as the Qwen E1 baselines
  eval_mg15.txt            the same with EVAL_PY=code/eval/eval_harness.py --batch_size 1, as E2
  crimson.txt              code/eval/run_crimson_chain.sh (X-ray modality adapters: reports)
Each backbone is evaluated with the script its baselines used (research rule 3).
Only Taha launches the resulting queues (PLAN.md section 7.3).

  python code/train/plan_e4_e5.py --manifest /data/ugrip2026/journal/data_e4e5/GROUP_MANIFEST.json \
    --data_dir /data/ugrip2026/journal/data_e4e5 --out_dir code/eval/jobs/e4e5
"""

import argparse
import hashlib
import json
import sys
import shlex
from pathlib import Path

import yaml

CODE = Path(__file__).resolve().parents[1]
REPO = CODE.parent
CONFIG_ROOT = CODE / "configs"
sys.path.insert(0, str(CODE / "data"))
from grouping import CLASSIFICATION_DATASET_SLUGS, E5_REUSES_E4  # noqa: E402
BACKBONE_COST = {"q35": 1.0, "mg15": 0.6}
RUNS = "/data/ugrip2026/journal/runs"
DATA = "/data/ugrip2026/journal/data"


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_configs(config_root=CONFIG_ROOT):
    sys.path.insert(0, str(CONFIG_ROOT))
    from generate_e4_e5 import expected_configs
    expected = {str(p): yaml.safe_load(t) for p, t in expected_configs()}
    configs = []
    for study in ("e4", "e5"):
        for path in sorted((Path(config_root) / study).glob("*.yaml")):
            cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
            if cfg != expected[str(path.relative_to(config_root))]:
                raise ValueError(f"{path}: config differs from the fixed E4/E5 matrix")
            cfg["_path"] = path
            configs.append(cfg)
    if len(configs) != len(expected):
        raise ValueError(f"expected {len(expected)} E4/E5 configs, found {len(configs)}")
    ids = [cfg["run_id"] for cfg in configs]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate E4/E5 run_id")
    return configs


def schedule(configs, manifest, n_chains=3):
    if not 1 <= n_chains <= 3:
        raise ValueError("use 1 to 3 chains (the planned concurrency budget)")
    decks = manifest["decks"]
    jobs = []
    for cfg in configs:
        deck = cfg["deck"]
        if deck not in decks or decks[deck]["kind"] != "train":
            raise ValueError(f"{cfg['_path']}: missing training deck {deck!r} in manifest")
        holdout = cfg["eval_deck"].replace("j26_devselect_", "j26_holdout_", 1)
        if holdout not in decks or decks[holdout]["kind"] != "holdout":
            raise ValueError(f"{cfg['_path']}: missing holdout deck {holdout!r} in manifest")
        rows = int(decks[deck]["rows"])
        if rows <= 0 or int(decks[holdout]['rows']) <= 0:
            raise ValueError(f"empty deck for {cfg['run_id']}")
        if cfg['eval_deck'] not in decks:
            raise ValueError(f"missing dev deck for {cfg['run_id']}")
        jobs.append({"cfg": cfg, "train_rows": rows,
                     "weight": rows * BACKBONE_COST[cfg["backbone"]], "holdout_deck": holdout})
    chains = [{"weight": 0.0, "jobs": []} for _ in range(n_chains)]
    for job in sorted(jobs, key=lambda j: (-j["weight"], j["cfg"]["run_id"])):
        chain = min(chains, key=lambda c: (c["weight"], len(c["jobs"])))
        chain["jobs"].append(job)
        chain["weight"] += job["weight"]
    return chains


def config_repo_path(path):
    return Path(path).resolve().relative_to(REPO.resolve()).as_posix()


def pred_dir(backbone, run_dir):
    """Where holdout predictions land: eval_arm.py (q35) writes into the run directory,
    eval_harness.py (mg15) into eval_holdout/, as for the E1/E2 baselines."""
    return run_dir if backbone == "q35" else f"{run_dir}/eval_holdout"


def eval_line(job, data_dir=DATA, runs_dir=RUNS):
    cfg, holdout = job["cfg"], job["holdout_deck"]
    rid = cfg["run_id"]
    run_dir = f"{runs_dir}/{rid}"
    deck = shlex.quote(f"{data_dir}/{holdout}.jsonl")
    if cfg["backbone"] == "q35":
        return (f"{rid} --adapter_path {shlex.quote(run_dir)} --data_path {deck} "
                f"--output_file {shlex.quote(run_dir + '/predictions_holdout.json')}")
    return (f"{rid} --mode adapter --adapter {shlex.quote(run_dir)} --backbone {cfg['backbone']} --batch_size 1 "
            f"--data {deck} --surface holdout --out_dir {shlex.quote(pred_dir(cfg['backbone'], run_dir))}")


def write_plan(chains, manifest_path, out_dir, data_dir=DATA, runs_dir=RUNS):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    plan = {"schema_version": 1, "experiment": "E4/E5", "n_chains": len(chains),
            "manifest": str(Path(manifest_path).resolve()), "manifest_md5": md5(manifest_path),
            "data_dir": str(data_dir), "runs_dir": str(runs_dir), "chains": [], "jobs": []}
    crimson = []
    evals = {"q35": ["# E4/E5 Qwen holdout evals (eval_arm.py, as the E1 baselines); generated by plan_e4_e5.py."],
             "mg15": ["# E4/E5 MedGemma holdout evals: run with EVAL_PY=code/eval/eval_harness.py (as E2); "
                      "generated by plan_e4_e5.py."]}
    for i, chain in enumerate(chains, 1):
        train_path = out_dir / f"train_chain_{i}.txt"
        # config paths only, so the file expands straight into run_chain.sh's arguments
        train_lines = []
        run_ids = []
        for job in chain["jobs"]:
            cfg = job["cfg"]
            train_lines.append(config_repo_path(cfg["_path"]))
            evals[cfg["backbone"]].append(eval_line(job, data_dir, runs_dir))
            run_ids.append(cfg["run_id"])
            plan['jobs'].append({"chain": i, "run_id": cfg['run_id'],
                                 "study": cfg['study'], "backbone": cfg['backbone'],
                                 "seed": cfg['seed'], "config": config_repo_path(cfg['_path']),
                                 "config_md5": md5(cfg['_path']), "train_deck": cfg['deck'],
                                 "holdout_deck": job['holdout_deck'],
                                 "pred_dir": pred_dir(cfg['backbone'], f"{runs_dir}/{cfg['run_id']}"),
                                 "reports": 'report_generation' in manifest['decks'][job['holdout_deck']]['task_counts']})
            if cfg["study"] == "E4" and cfg["deck"] == "j26_train_modxray":
                crimson.append((cfg["run_id"], pred_dir(cfg["backbone"], f"{runs_dir}/{cfg['run_id']}")))
        train_path.write_text("\n".join(train_lines) + "\n", encoding="utf-8", newline="\n")
        plan["chains"].append({"chain": i, "estimated_weight": chain["weight"],
                               "n_runs": len(run_ids), "run_ids": run_ids,
                               "train_job_file": str(train_path)})
    for backbone, lines in evals.items():
        (out_dir / f"eval_{backbone}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    # E5 slots filled by an E4 adapter trained on the identical deck: no extra training or eval.
    reverse = {modality: dataset for dataset, modality in E5_REUSES_E4.items()}
    for job in [j for j in plan["jobs"] if j["study"] == "E4"]:
        label = manifest["decks"][job["train_deck"]]["group_label"]
        if label not in reverse:
            continue
        holdout = f"j26_holdout_ds{CLASSIFICATION_DATASET_SLUGS[reverse[label]]}"
        if manifest["decks"][holdout]["md5"] != manifest["decks"][job["holdout_deck"]]["md5"]:
            raise ValueError(f"{holdout} differs from {job['holdout_deck']}: cannot reuse {job['run_id']}")
        plan["jobs"].append(dict(job, study="E5", holdout_deck=holdout, reports=False, reuses_e4=True))
    crimson_path = out_dir / "crimson.txt"
    crimson_lines = ["# E4 X-ray modality adapters contain report generation; score after eval chains."]
    crimson_lines += [f"{name} {path}" for name, path in sorted(crimson)]
    crimson_path.write_text("\n".join(crimson_lines) + "\n", encoding="utf-8", newline="\n")
    plan["crimson_jobs"] = len(crimson)
    (out_dir / "plan.json").write_text(json.dumps(plan, indent=1) + "\n", encoding="utf-8", newline="\n")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--config_root", type=Path, default=CONFIG_ROOT)
    parser.add_argument("--chains", type=int, default=3)
    parser.add_argument("--data_dir", default=DATA)
    parser.add_argument("--runs_dir", default=RUNS)
    args = parser.parse_args()
    if args.chains < 1:
        raise SystemExit("--chains must be positive")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    chains = schedule(load_configs(args.config_root), manifest, args.chains)
    plan = write_plan(chains, args.manifest, args.out_dir, args.data_dir, args.runs_dir)
    for chain in plan["chains"]:
        print(f"chain {chain['chain']}: {chain['n_runs']} runs, weight {chain['estimated_weight']:.1f}")
    print(f"CRIMSON jobs: {plan['crimson_jobs']}; wrote plan to {args.out_dir}")


if __name__ == "__main__":
    main()
