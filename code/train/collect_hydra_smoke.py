#!/usr/bin/env python3
"""Collect small smoke evidence on CPU; excludes weights, images and predictions.

Run from your personal Blackwell clone. This reads completed smoke directories
and writes one archive in your own folder; it never starts a model or modifies runs.
"""
import argparse
import datetime
import hashlib
import io
import json
from pathlib import Path
import tarfile

DEFAULT_RUNS = [
    "j26_q35_hydraT_full_768_s1337_smoke_veronika_11032130",
    "j26_q35_hydraR_full_768_s1337_smoke_veronika_11032152",
]
REQUIRED = ("config.yaml", "git_commit.txt", "decks.json", "runtime.json",
            "method_config.json", "eval_smoke/metrics.json", "eval_smoke/routing_stats.npz")
OPTIONAL = ("registry_row.csv", "trainer_log.jsonl", "task_loss.jsonl")


def collect(runs_root, run_ids, out):
    root, out = Path(runs_root).resolve(), Path(out).resolve()
    if out.exists():
        raise ValueError(f"archive exists: {out}; choose a new name")
    if not run_ids or len(set(run_ids)) != len(run_ids):
        raise ValueError("supply unique smoke run IDs")
    files = []
    for rid in run_ids:
        if Path(rid).name != rid or not rid.startswith("j26_") or "_smoke_" not in rid:
            raise ValueError(f"expected a smoke run ID, got {rid!r}")
        run = root / rid
        for rel in REQUIRED + OPTIONAL:
            path = run / rel
            if not path.is_file():
                if rel in REQUIRED:
                    raise ValueError(f"missing smoke evidence: {path}")
                continue
            if not path.resolve().is_relative_to(run.resolve()):
                raise ValueError(f"evidence path escapes run directory: {path}")
            files.append((path, f"{rid}/{rel}"))
    manifest = {"schema": 1, "surface": "devselect-smoke",
                "collected_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "run_ids": run_ids,
                "files": [{"path": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                          for path, name in files]}
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "x:gz") as archive:
        for path, name in files:
            archive.add(path, arcname=name, recursive=False)
        payload = (json.dumps(manifest, indent=2) + "\n").encode()
        info = tarfile.TarInfo("collection.json")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs_root", default="/data/ugrip2026/journal/runs/smoke")
    parser.add_argument("--run_ids", nargs="+", default=DEFAULT_RUNS)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    manifest = collect(args.runs_root, args.run_ids, args.out)
    print(f"Collected {len(manifest['files'])} evidence files -> {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
