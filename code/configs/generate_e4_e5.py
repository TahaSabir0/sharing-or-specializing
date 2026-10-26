#!/usr/bin/env python3
"""Generate/check the complete E4/E5 run-config matrix.

The checked-in YAMLs are generated artifacts, but the study requires one config per
real run.  Run this script after changing the matrix; CI/tests use ``--check`` to
detect missing or stale files.
"""

import argparse
import difflib
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "data"))
from grouping import CLASSIFICATION_DATASET_SLUGS, MODALITY_SLUGS  # noqa: E402

SEEDS = (1337, 2026, 42)
BACKBONES = {
    "q35": {"resolution": 768},
    "mg15": {"resolution": 896},
}


def config_text(study, group_type, label, slug, backbone, seed):
    is_modality = group_type == "modality"
    prefix = "mod" if is_modality else "ds"
    description = f"{label} modality" if is_modality else f"{label} classification dataset"
    deck = f"j26_train_{prefix}{slug}"
    eval_deck = f"j26_devselect_{prefix}{slug}"
    resolution = BACKBONES[backbone]["resolution"]
    return (
        f"# {study} {description} adapter on {backbone}, seed {seed} (PLAN.md v2 {study}).\n"
        f"# One independent rank-8 LoRA; only the grouping level differs from the fixed protocol.\n"
        f"run_id: j26_{backbone}_{prefix}{slug}_full_{resolution}_s{seed}\n"
        "owner: alex\n"
        f"study: {study}\n"
        f"backbone: {backbone}\n"
        "method: lora\n"
        "method_cfg: {}\n"
        f"deck: {deck}\n"
        "data_tag: full\n"
        f"seed: {seed}\n"
        f"eval_deck: {eval_deck}\n"
        "overrides: {}\n"
    )


def expected_configs():
    for study, group_type, mapping in (
        ("E4", "modality", MODALITY_SLUGS),
        ("E5", "classification_dataset", CLASSIFICATION_DATASET_SLUGS),
    ):
        folder = study.lower()
        prefix = "mod" if group_type == "modality" else "ds"
        for label, slug in mapping.items():
            for backbone in BACKBONES:
                for seed in SEEDS:
                    name = f"{prefix}{slug}_{backbone}_s{seed}.yaml"
                    yield Path(folder) / name, config_text(
                        study, group_type, label, slug, backbone, seed
                    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if checked-in configs are stale")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent,
                        help="config root (default: directory containing this script)")
    args = parser.parse_args()
    failures = []
    expected_paths = set()
    for relative, content in expected_configs():
        path = args.root / relative
        expected_paths.add(path)
        if args.check:
            actual = path.read_text(encoding="utf-8") if path.exists() else ""
            if actual != content:
                diff = "".join(difflib.unified_diff(
                    actual.splitlines(True), content.splitlines(True),
                    fromfile=str(path), tofile=f"{path} (expected)"
                ))
                failures.append(diff or f"missing {path}\n")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8", newline="\n")
            print(path)
    if args.check:
        extras = sorted(
            p for folder in (args.root / "e4", args.root / "e5") if folder.exists()
            for p in folder.glob("*.yaml") if p not in expected_paths
        )
        if extras:
            failures.append("unexpected generated configs:\n" + "\n".join(map(str, extras)) + "\n")
        if failures:
            raise SystemExit("".join(failures))
        print(f"OK: {len(expected_paths)} E4/E5 configs are current")


if __name__ == "__main__":
    main()
