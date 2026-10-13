#!/usr/bin/env python3
"""W1: build the journal decks, dev-select, holdout file and scaling subsets (PLAN.md §1.5, §1.7, §1.9).

Adapted from build_decks.py (the challenge exporter), which is kept for reference. Differences:
one row per question (no chunking; first 4 images kept, each with an Image-i marker), bare prompt
only, routing labels on every row, instance detection and endoscopy dropped.

Training pool: A.jsonl ∪ B.jsonl (these include A_dev/B_dev; Taha's decision 2026-10-13, see the
PLAN.md changelog). Dev-select is built from A_dev ∪ B_dev and is therefore train-seen.

Subcommands
  decks      6 per-task decks + the all-task mixture, dev-select, holdout, dataset_info.json,
             md5 manifest
  subsets    nested scaling subsets (seed 20261009 at 1/5/10/25/50%) and sensitivity draws
             (seeds 20261010, 20261011 at 5/10/25%) -> data/subsets/<task>_seed<seed>.json
  subset-deck  mixture deck in which only the target task is shrunk to a subset (PLAN.md §1.9, D5)

  python build_journal_decks.py decks --splits_dir .../internal_splits_v2 --out_dir /share/sda/ugrip/journal/data
  python build_journal_decks.py subsets --splits_dir ... --out_dir <repo>/data/subsets
  python build_journal_decks.py subset-deck --splits_dir ... --subsets_dir <repo>/data/subsets \
      --task classification --seed 20261009 --pct 10 --out_dir /share/sda/ugrip/journal/data
"""
import argparse
import collections
import hashlib
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from journal_format import keep_row, image_group, to_row, load_label_ids, MAX_IMAGES  # noqa: E402

TASK_SHORT = {"classification": "cls", "multi-label classification": "ml", "detection": "det",
              "counting": "count", "regression": "reg", "report_generation": "report"}
TRAIN_FILES = ["A.jsonl", "B.jsonl"]
DEV_FILES = ["A_dev.jsonl", "B_dev.jsonl"]
HOLDOUT_FILE = "true_test.jsonl"
DEV_CAP = 200
DEV_SEED = 20261013
MAIN_SEED, SENS_SEEDS = 20261009, (20261010, 20261011)
MAIN_PCTS, SENS_PCTS = (1, 5, 10, 25, 50), (5, 10, 25)


def load(splits_dir, names):
    rows = []
    for n in names:
        with open(os.path.join(splits_dir, n)) as f:
            rows += [json.loads(l) for l in f if l.strip()]
    return rows


def kept(rows):
    out = [r for r in rows if keep_row(r)]
    keys = [r["sample_key"] for r in out]
    assert len(keys) == len(set(keys)), "duplicate sample_key"
    return out


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return md5(path)


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def census(rows):
    c = collections.Counter((r["task"], r["dataset"]) for r in rows)
    return {f"{t}/{d}": n for (t, d), n in sorted(c.items())}


def check_images(rows):
    missing = [p for r in rows for p in r["image"][:MAX_IMAGES] if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"{len(missing)} missing images, e.g. {missing[:3]}")


def dataset_info(names):
    """LF dataset_info.json entries for the journal decks (ShareGPT, image column)."""
    cols = {"messages": "conversations", "images": "images"}
    tags = {"role_tag": "from", "content_tag": "value", "user_tag": "human", "assistant_tag": "gpt"}
    return {n: {"file_name": f"{n}.jsonl", "formatting": "sharegpt", "columns": cols, "tags": tags}
            for n in names}


# ---------------------------------------------------------------------------------------- decks
def cmd_decks(a):
    ids = load_label_ids()
    os.makedirs(a.out_dir, exist_ok=True)
    train = kept(load(a.splits_dir, TRAIN_FILES))
    dev = kept(load(a.splits_dir, DEV_FILES))
    hold = kept(load(a.splits_dir, [HOLDOUT_FILE]))
    for rows in (train, dev, hold):
        check_images(rows)

    manifest = {"splits_dir": a.splits_dir, "train_files": TRAIN_FILES, "max_images": MAX_IMAGES,
                "decks": {}}
    names = []

    def emit(name, recs, kind):
        rows = [to_row(r, ids) for r in recs]
        m = write_jsonl(os.path.join(a.out_dir, f"{name}.jsonl"), rows)
        manifest["decks"][name] = {"kind": kind, "rows": len(rows), "md5": m,
                                   "truncated_gt4_images": sum(r["n_images_orig"] > MAX_IMAGES for r in rows),
                                   "census": census(recs)}
        names.append(name)
        print(f"{name:28s} {len(rows):6d} rows  md5 {m}")

    # per-task decks keep the split-file order; LF's sampler shuffles each epoch (seeded).
    for task, short in TASK_SHORT.items():
        emit(f"j26_train_{short}", [r for r in train if r["task"] == task], "train")
    # mixture = natural A∪B distribution (no balancing), pre-shuffled with a fixed seed
    mix = list(train)
    random.Random(MAIN_SEED).shuffle(mix)
    emit("j26_train_mix", mix, "train")

    # dev-select: <= DEV_CAP questions per (task, dataset) cell, seeded sample
    cells = collections.defaultdict(list)
    for r in dev:
        cells[(r["task"], r["dataset"])].append(r)
    sel = []
    for k in sorted(cells):
        rs = sorted(cells[k], key=lambda r: r["sample_key"])
        sel += rs if len(rs) <= DEV_CAP else random.Random(f"{DEV_SEED}|{k}").sample(rs, DEV_CAP)
    emit("j26_devselect", sel, "eval-dev (train-seen)")
    emit("j26_dev_full", dev, "eval-dev (train-seen)")
    emit("j26_holdout", hold, "eval-holdout")

    with open(os.path.join(a.out_dir, "dataset_info.json"), "w") as f:
        json.dump(dataset_info(names), f, indent=1)
    with open(os.path.join(a.out_dir, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print("wrote dataset_info.json and MANIFEST.json to", a.out_dir)


# -------------------------------------------------------------------------------------- subsets
def task_groups(train, task):
    """image_group -> stratum for the target task. Stratum = dataset; for classification
    (dataset, label), label = answer of the group's first cls question by sample_key."""
    first = {}
    for r in sorted((r for r in train if r["task"] == task), key=lambda r: r["sample_key"]):
        g = image_group(r)
        if g not in first:
            first[g] = (r["dataset"], r["answer"]) if task == "classification" else (r["dataset"],)
    return first


def draw(groups, seed, pcts):
    """Nested draw: per stratum one fixed permutation; p% subset = its first max(1, round(p·n))."""
    strata = collections.defaultdict(list)
    for g, s in groups.items():
        strata[s].append(g)
    perms = {}
    for s in sorted(strata):
        gs = sorted(strata[s])
        random.Random(f"{seed}|{'|'.join(map(str, s))}").shuffle(gs)
        perms[s] = gs
    out = {}
    for p in pcts:
        sel = []
        for s, gs in perms.items():
            sel += gs[:max(1, round(p / 100 * len(gs)))]
        out[str(p)] = sorted(sel)
    # nestedness check
    for lo, hi in zip(pcts, pcts[1:]):
        assert set(out[str(lo)]) <= set(out[str(hi)])
    return out, {"|".join(map(str, s)): len(g) for s, g in perms.items()}


def cmd_subsets(a):
    train = kept(load(a.splits_dir, TRAIN_FILES))
    os.makedirs(a.out_dir, exist_ok=True)
    for task, short in TASK_SHORT.items():
        groups = task_groups(train, task)
        q_per_group = collections.Counter(image_group(r) for r in train if r["task"] == task)
        for seed, pcts in [(MAIN_SEED, MAIN_PCTS)] + [(s, SENS_PCTS) for s in SENS_SEEDS]:
            sub, strata = draw(groups, seed, pcts)
            summary = {p: {"groups": len(g), "questions": sum(q_per_group[x] for x in g)}
                       for p, g in sub.items()}
            doc = {"_doc": "PLAN.md §1.9 scaling subset. Lists the selected image groups of the target "
                           "task (sorted image paths joined by '|'). Drop only target-task questions on "
                           "unselected groups (D5). 100% = all groups.",
                   "task": task, "seed": seed, "pool": TRAIN_FILES,
                   "stratum": "(dataset, first-cls-answer)" if task == "classification" else "(dataset)",
                   "rule": "per stratum: fixed permutation, prefix of max(1, round(p*n)) groups",
                   "n_groups_total": len(groups),
                   "n_questions_total": sum(q_per_group.values()),
                   "strata_sizes": strata, "summary": summary, "groups": sub}
            path = os.path.join(a.out_dir, f"{short}_seed{seed}.json")
            with open(path, "w") as f:
                json.dump(doc, f, indent=0, ensure_ascii=False)
            print(f"{short:6s} seed {seed}: " + ", ".join(f"{p}%={v['questions']}q/{v['groups']}g"
                                                          for p, v in summary.items()))


def cmd_subset_deck(a):
    ids = load_label_ids()
    short = TASK_SHORT[a.task]
    with open(os.path.join(a.subsets_dir, f"{short}_seed{a.seed}.json")) as f:
        sel = set(json.load(f)["groups"][str(a.pct)])
    train = kept(load(a.splits_dir, TRAIN_FILES))
    keep = [r for r in train if r["task"] != a.task or image_group(r) in sel]
    random.Random(MAIN_SEED).shuffle(keep)
    rows = [to_row(r, ids) for r in keep]
    name = f"j26_train_mix_{short}{a.pct}_s{a.seed}" if not a.task_only else f"j26_train_{short}{a.pct}_s{a.seed}"
    if a.task_only:
        rows = [r for r in rows if r["task"] == a.task]
    path = os.path.join(a.out_dir, f"{name}.jsonl")
    m = write_jsonl(path, rows)
    info_path = os.path.join(a.out_dir, "dataset_info.json")
    info = json.load(open(info_path)) if os.path.exists(info_path) else {}
    info.update(dataset_info([name]))
    json.dump(info, open(info_path, "w"), indent=1)
    print(f"{name}: {len(rows)} rows, md5 {m}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("decks", "subsets", "subset-deck"):
        p = sp.add_parser(name)
        p.add_argument("--splits_dir", required=True)
        p.add_argument("--out_dir", required=True)
        if name == "subset-deck":
            p.add_argument("--subsets_dir", required=True)
            p.add_argument("--task", required=True, choices=list(TASK_SHORT))
            p.add_argument("--seed", type=int, default=MAIN_SEED)
            p.add_argument("--pct", type=int, required=True)
            p.add_argument("--task_only", action="store_true",
                           help="emit only the target task's rows (TaskLoRA scaling points)")
    a = ap.parse_args()
    {"decks": cmd_decks, "subsets": cmd_subsets, "subset-deck": cmd_subset_deck}[a.cmd](a)


if __name__ == "__main__":
    main()
