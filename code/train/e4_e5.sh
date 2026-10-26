#!/bin/bash
# E4/E5 workflow. prepare/check/status/test are CPU-only; full GPU stages are
# launched by the designated operator after Phase A and the method gates pass.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
STAGE="${1:?usage: e4_e5.sh prepare|check|status|test|train|eval|crimson|analyze [chain]}"
source "${CONDA_SH:-/data/ugrip2026/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-flare-blackwell-v5}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export PYTHONUNBUFFERED=1
SOURCE="${JOURNAL_SOURCE:-/data/ugrip2026/journal/data}"
export JOURNAL_DATA="${JOURNAL_DATA:-$(dirname "$REPO")/group_decks}"
export JOURNAL_RUNS="${JOURNAL_RUNS:-/data/ugrip2026/journal/runs}"
JOBS="${E4E5_JOBS:-$(dirname "$REPO")/e4e5_jobs}"
PLAN="$JOBS/plan.json"
case "$STAGE" in
  prepare)
    python code/configs/generate_e4_e5.py --check
    python code/data/build_group_decks.py --source_dir "$SOURCE" --out_dir "$JOURNAL_DATA"
    python code/train/plan_e4_e5.py --manifest "$JOURNAL_DATA/GROUP_MANIFEST.json" \
      --data_dir "$JOURNAL_DATA" --runs_dir "$JOURNAL_RUNS" --out_dir "$JOBS"
    python code/train/group_pipeline.py check --plan "$PLAN"
    ;;
  check|status)
    python code/train/group_pipeline.py "$STAGE" --plan "$PLAN"
    ;;
  test)
    export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
    python -m pytest -q code/tests/test_group_decks.py code/tests/test_group_workflow.py
    ;;
  train|eval|crimson)
    CHAIN="${2:?choose chain 1, 2, or 3}"
    case "$CHAIN" in 1|2|3) ;; *) echo 'chain must be 1, 2, or 3' >&2; exit 2;; esac
    export CUDA_VISIBLE_DEVICES=1 GPU=1
    python code/train/group_pipeline.py "$STAGE" --plan "$PLAN" --chain "$CHAIN"
    ;;
  analyze)
    OUT="${E4E5_RESULTS:-$(dirname "$REPO")/e4e5_results_$(date +%Y%m%d_%H%M%S)}"
    python code/analysis/analyze_e4_e5.py --plan "$PLAN" \
      --holdout "$SOURCE/j26_holdout.jsonl" --train "$SOURCE/j26_train_mix.jsonl" \
      --baseline-runs "${BASELINE_RUNS:-/data/ugrip2026/journal/runs}" --out "$OUT"
    ;;
  *) echo "unknown stage: $STAGE" >&2; exit 2;;
esac
