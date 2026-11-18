#!/usr/bin/env bash
# E1/E2 paired arm contrasts on the holdout (PLAN.md §1 stats: paired bootstrap on the same questions,
# two-level over seeds then questions stratified by dataset, Holm across the 6 tasks per comparison).
# Reuses compare_groups.py (the E4/E5 method) on the full-holdout runs assembled by analyze_e4_e5.py.
# CPU only. Usage: bash code/analysis/e1_e2_bootstrap.sh <out_dir>
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
source "${CONDA_SH:-/data/ugrip2026/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-flare-blackwell-v5}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 PYTHONUNBUFFERED=1 \
  PYTHONPATH="$REPO/code/analysis:$REPO/code/eval:$REPO/code/data"
OUT="${1:?usage: e1_e2_bootstrap.sh <out_dir>}"
HYDRA="${E4E5_HYDRA:-/data/ugrip2026/journal/e4e5_results_hydra_20261116}"     # shared, task, hydraT, hydraR
PLAIN="${E4E5_PLAIN:-/data/ugrip2026/journal/e4e5_results_plain_lora_20261106}" # shared48
TRAIN="${JOURNAL_SOURCE:-/data/ugrip2026/journal/data}/j26_train_mix.jsonl"
[ -e "$OUT" ] && { echo "$OUT exists; use a fresh directory" >&2; exit 2; }
mkdir -p "$OUT"
dir() { if [ "$1" = shared48 ]; then echo "$PLAIN/$1_$2_s$3"; else echo "$HYDRA/$1_$2_s$3"; fi; }
# reference (a) and compared arm (b): difference is b - a
PAIRS="shared:task shared48:task shared:shared48 hydraR:hydraT shared:hydraT task:hydraT shared:hydraR task:hydraR"
pids=()
for bb in q35 mg15; do
  for pair in $PAIRS; do
    a="${pair%%:*}"; b="${pair##*:}"
    python code/analysis/compare_groups.py \
      --a "$(dir "$a" $bb 1337)" "$(dir "$a" $bb 2026)" "$(dir "$a" $bb 42)" \
      --b "$(dir "$b" $bb 1337)" "$(dir "$b" $bb 2026)" "$(dir "$b" $bb 42)" \
      --train "$TRAIN" --out "$OUT/${b}_minus_${a}_${bb}" > "$OUT/${b}_minus_${a}_${bb}.log" 2>&1 &
    pids+=($!)
  done
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "E1/E2 BOOTSTRAP DONE rc=$rc: $OUT"
exit $rc
