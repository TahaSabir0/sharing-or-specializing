#!/bin/bash
# E9-lite (Qwen, cls target, seed-1 subsets 20261009 at 1/5/10/25/50%): build the scaling decks into their own
# data dir, never into the frozen W1 dir. Mixture decks shrink only the cls questions (PLAN.md §1.9, D5);
# --task_only decks are the TaskLoRA-cls points. Then verify every row is byte-identical to a W1 row.
#   bash code/data/build_e9_decks.sh            (on the box, from the shared repo)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
SRC=/data/ugrip2026/journal/data
OUT="${E9_DATA:-/data/ugrip2026/journal/data_e9}"
SPLITS=/data/ugrip2026/splits/internal_splits_v2
source "${CONDA_SH:-/data/ugrip2026/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-flare-blackwell-v5}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
mkdir -p "$OUT"
[ -z "$(ls -A "$OUT")" ] || { echo "$OUT is not empty; refusing to overwrite" >&2; exit 2; }
cd "$REPO"
# dev-loss decks used by the configs (same files as E1: symlinks, so the md5 is the W1 one)
ln -s "$SRC/j26_devselect.jsonl" "$OUT/j26_devselect.jsonl"
ln -s "$SRC/j26_devselect_cls.jsonl" "$OUT/j26_devselect_cls.jsonl"
python - "$SRC" "$OUT" <<'EOF'
import json, sys
src, out = sys.argv[1:]
info = json.load(open(f"{src}/dataset_info.json"))
json.dump({k: info[k] for k in ("j26_devselect", "j26_devselect_cls")}, open(f"{out}/dataset_info.json", "w"), indent=1)
EOF
for P in 1 5 10 25 50; do
  python code/data/build_journal_decks.py subset-deck --splits_dir "$SPLITS" --subsets_dir data/subsets \
    --task classification --seed 20261009 --pct $P --out_dir "$OUT"
  python code/data/build_journal_decks.py subset-deck --splits_dir "$SPLITS" --subsets_dir data/subsets \
    --task classification --seed 20261009 --pct $P --task_only --out_dir "$OUT"
done
python - "$SRC" "$OUT" "$REPO/data/subsets/cls_seed20261009.json" <<'EOF'
import collections, hashlib, json, sys
src, out, subs = sys.argv[1:]
summ = json.load(open(subs))["summary"]
def rows(p): return [l.rstrip("\n") for l in open(p, encoding="utf-8") if l.strip()]
mix = collections.Counter(rows(f"{src}/j26_train_mix.jsonl"))
n_other = sum(1 for r in mix.elements() if json.loads(r)["task"] != "classification")
man = {"source": src, "subsets": subs, "decks": {}}
for p in (1, 5, 10, 25, 50):
    for name, want in ((f"j26_train_mix_cls{p}_s20261009", n_other + summ[str(p)]["questions"]),
                       (f"j26_train_cls{p}_s20261009", summ[str(p)]["questions"])):
        path = f"{out}/{name}.jsonl"; rs = rows(path)
        extra = collections.Counter(rs) - mix
        assert not extra, f"{name}: {sum(extra.values())} rows not in the W1 mixture"
        assert len(rs) == want, f"{name}: {len(rs)} rows, expected {want}"
        man["decks"][name] = {"md5": hashlib.md5(open(path, "rb").read()).hexdigest(), "rows": len(rs)}
        print(f"{name}: {len(rs)} rows OK (all in W1 mixture)")
json.dump(man, open(f"{out}/MANIFEST.json", "w"), indent=1)
EOF
echo "E9 decks ready in $OUT"
