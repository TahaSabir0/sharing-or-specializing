#!/usr/bin/env python3
"""Build E4 modality and E5 classification-dataset decks from frozen W1 decks.

This intentionally consumes ``j26_train_mix.jsonl``, ``j26_devselect.jsonl`` and
``j26_holdout.jsonl`` instead of the upstream split files.  Therefore every emitted
row has the same rendered fields already used by E1/E2; only the grouping changes.
Building a holdout slice does not run inference or make a selection decision.

Teammate smoke-test example (writes only in the teammate's area)::

  python code/data/build_group_decks.py \
    --source_dir /data/ugrip2026/journal/data \
    --out_dir /data/ugrip2026/journal/people/alex/group_decks

For real runs, Taha runs the same command with the shared journal data directory as
``--out_dir`` after reviewing ``GROUP_MANIFEST.json``.
"""

import argparse
import collections
import hashlib
import json
import os
import tempfile
from pathlib import Path

from grouping import CLASSIFICATION_DATASET_SLUGS, MODALITY_SLUGS

TRAIN_SOURCE = "j26_train_mix"
DEV_SOURCE = "j26_devselect"
HOLDOUT_SOURCE = "j26_holdout"
CLASSIFICATION = "classification"


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _md5(path):
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl_atomic(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return _md5(path)


def _validate_source(rows, source_name):
    required = {"id", "task", "modality", "dataset", "modality_id", "dataset_id"}
    missing = [(i, sorted(required - set(row))) for i, row in enumerate(rows)
               if not required <= set(row)]
    if missing:
        raise ValueError(f"{source_name}: rows missing routing fields, e.g. {missing[:3]}")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{source_name}: duplicate row ids")


def _dataset_info_entry(name):
    return {
        "file_name": f"{name}.jsonl",
        "formatting": "sharegpt",
        "columns": {"messages": "conversations", "images": "images"},
        "tags": {
            "role_tag": "from",
            "content_tag": "value",
            "user_tag": "human",
            "assistant_tag": "gpt",
        },
    }


def build_group_decks(source_dir, out_dir):
    source_dir, out_dir = Path(source_dir), Path(out_dir)
    train = _read_jsonl(source_dir / f"{TRAIN_SOURCE}.jsonl")
    dev = _read_jsonl(source_dir / f"{DEV_SOURCE}.jsonl")
    holdout = _read_jsonl(source_dir / f"{HOLDOUT_SOURCE}.jsonl")
    _validate_source(train, TRAIN_SOURCE)
    _validate_source(dev, DEV_SOURCE)
    _validate_source(holdout, HOLDOUT_SOURCE)

    seen_modalities = {row["modality"] for row in train}
    if seen_modalities != set(MODALITY_SLUGS):
        raise ValueError(
            f"frozen modality set mismatch: expected {sorted(MODALITY_SLUGS)}, "
            f"found {sorted(seen_modalities)}"
        )
    train_cls = [row for row in train if row["task"] == CLASSIFICATION]
    seen_cls_datasets = {row["dataset"] for row in train_cls}
    if seen_cls_datasets != set(CLASSIFICATION_DATASET_SLUGS):
        raise ValueError(
            f"frozen classification dataset set mismatch: expected "
            f"{sorted(CLASSIFICATION_DATASET_SLUGS)}, found {sorted(seen_cls_datasets)}"
        )

    outputs = []
    for group_type, mapping, field, task_filter, prefix in (
        ("modality", MODALITY_SLUGS, "modality", None, "mod"),
        ("classification_dataset", CLASSIFICATION_DATASET_SLUGS, "dataset", CLASSIFICATION, "ds"),
    ):
        for label, slug in mapping.items():
            for split, source_rows, source_name in (
                ("train", train, TRAIN_SOURCE),
                ("devselect", dev, DEV_SOURCE),
                ("holdout", holdout, HOLDOUT_SOURCE),
            ):
                rows = [row for row in source_rows
                        if row[field] == label and (task_filter is None or row["task"] == task_filter)]
                if not rows:
                    raise ValueError(f"empty {split} group: {group_type}={label}")
                name = f"j26_{split}_{prefix}{slug}"
                path = out_dir / f"{name}.jsonl"
                outputs.append({
                    "name": name,
                    "kind": split,
                    "group_type": group_type,
                    "group_label": label,
                    "source_deck": source_name,
                    "rows": len(rows),
                    "md5": _write_jsonl_atomic(path, rows),
                    "task_counts": dict(sorted(collections.Counter(r["task"] for r in rows).items())),
                    "dataset_counts": dict(sorted(collections.Counter(r["dataset"] for r in rows).items())),
                })

    # Every source row belongs to exactly one E4 modality group.  E5 is intentionally
    # only classification, where every classification row belongs to one dataset group.
    assert sum(x["rows"] for x in outputs
               if x["kind"] == "train" and x["group_type"] == "modality") == len(train)
    assert sum(x["rows"] for x in outputs
               if x["kind"] == "train" and x["group_type"] == "classification_dataset") == len(train_cls)

    info_path = out_dir / "dataset_info.json"
    info = {}
    if info_path.exists():
        with open(info_path, encoding="utf-8") as f:
            info = json.load(f)
    for item in outputs:
        info[item["name"]] = _dataset_info_entry(item["name"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(info_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(info, f, indent=1, sort_keys=True)
        f.write("\n")

    manifest = {
        "schema_version": 1,
        "source_dir": str(source_dir),
        "sources": {
            TRAIN_SOURCE: {"rows": len(train), "md5": _md5(source_dir / f"{TRAIN_SOURCE}.jsonl")},
            DEV_SOURCE: {"rows": len(dev), "md5": _md5(source_dir / f"{DEV_SOURCE}.jsonl")},
            HOLDOUT_SOURCE: {
                "rows": len(holdout), "md5": _md5(source_dir / f"{HOLDOUT_SOURCE}.jsonl")
            },
        },
        "decks": {item["name"]: {k: v for k, v in item.items() if k != "name"}
                  for item in outputs},
    }
    with open(out_dir / "GROUP_MANIFEST.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
        f.write("\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source_dir", required=True, help="directory containing frozen W1 source decks")
    parser.add_argument("--out_dir", required=True, help="destination (personal area for teammate smoke tests)")
    args = parser.parse_args()
    manifest = build_group_decks(args.source_dir, args.out_dir)
    for name, item in manifest["decks"].items():
        print(f"{name:32s} {item['rows']:6d} rows  md5 {item['md5']}")
    print(f"wrote {len(manifest['decks'])} decks and GROUP_MANIFEST.json to {args.out_dir}")


if __name__ == "__main__":
    main()
