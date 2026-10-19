#!/bin/bash
# CRIMSON scoring of report generation on mbz-yaqub GPU 1 (the FLARE 2026 report metric, PLAN.md §1.8).
# Input: an eval folder from eval_harness.py (needs reportgen_pairs.jsonl from score_run.py), or any
# pairs file with {gt, pred, sample_key} per line. Writes crimson_metrics.json next to it; for an
# eval folder, re-runs score_run.py so metrics.json picks the score up.
#   bash code/eval/crimson_box.sh <eval_out_dir | reportgen_pairs.jsonl>
# Env `crimson-score` (Python 3.11, crimson-score 0.2.0, as on ws) is kept apart from the training env.
# Batch size stays 1 (the challenge setting) unless CRIMSON_BATCH_SIZE is set; batching can change
# the judge's greedy outputs, so keep it fixed across everything that is compared.
set -euo pipefail
ARG="${1:?usage: crimson_box.sh <eval_out_dir | reportgen_pairs.jsonl>}"
GPU="${GPU:-1}"
if [ "$GPU" != "1" ] && [ -z "${ALLOW_OTHER_GPU:-}" ]; then
  echo "refusing GPU $GPU: our Blackwell allocation is GPU 1" >&2; exit 2
fi
if [ -d "$ARG" ]; then DIR="$ARG"; PAIRS="$ARG/reportgen_pairs.jsonl"; else DIR="$(dirname "$ARG")"; PAIRS="$ARG"; fi
[ -f "$PAIRS" ] || { echo "missing $PAIRS: run score_run.py on the eval folder first"; exit 1; }
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${CONDA_SH:-/data/ugrip2026/miniconda3/etc/profile.d/conda.sh}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}" HF_HUB_OFFLINE=1
conda activate "${CRIMSON_ENV:-crimson-score}"
CUDA_VISIBLE_DEVICES=$GPU python "$HERE/score_crimson.py" --pred "$PAIRS" --crimson_api hf \
    --crimson_model "${CRIMSON_MODEL:-/data/ugrip2026/models/CRIMSON-medgemma-4b}" \
    --batch_size "${CRIMSON_BATCH_SIZE:-1}" --out "$DIR/crimson_metrics.json"
if [ -f "$DIR/run_meta.json" ]; then
  conda activate "${CONDA_ENV:-flare-blackwell-v5}"
  python "$HERE/score_run.py" --out_dir "$DIR"
fi
