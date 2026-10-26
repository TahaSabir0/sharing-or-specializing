#!/bin/bash
# E4/E5 workflow, CPU only: prepare | check | status | verify | test | analyze.
# Nothing here launches GPU work. Real runs use the job files that `prepare` writes, through
# run_chain.sh / run_eval_chain.sh / run_crimson_chain.sh (Taha only; code/E4_E5_RUNBOOK.md).
# In the shared repo the decks go to /data/ugrip2026/journal/data_e4e5 and the job files to
# code/eval/jobs/e4e5 (committed); in a personal clone both stay next to the clone.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
STAGE="${1:?usage: e4_e5.sh prepare|check|status|verify|test|analyze}"
source "${CONDA_SH:-/data/ugrip2026/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-flare-blackwell-v5}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export PYTHONUNBUFFERED=1
SOURCE="${JOURNAL_SOURCE:-/data/ugrip2026/journal/data}"
if [ "$REPO" = "/data/ugrip2026/journal/repo" ]; then
  export JOURNAL_DATA="${JOURNAL_DATA:-/data/ugrip2026/journal/data_e4e5}"
  JOBS="${E4E5_JOBS:-$REPO/code/eval/jobs/e4e5}"
else
  export JOURNAL_DATA="${JOURNAL_DATA:-$(dirname "$REPO")/group_decks}"
  JOBS="${E4E5_JOBS:-$(dirname "$REPO")/e4e5_jobs}"
fi
[ "$JOURNAL_DATA" = "$SOURCE" ] && { echo "keep the group decks out of the frozen W1 data dir" >&2; exit 2; }
export JOURNAL_RUNS="${JOURNAL_RUNS:-/data/ugrip2026/journal/runs}"
PLAN="$JOBS/plan.json"
case "$STAGE" in
  prepare)
    python code/configs/generate_e4_e5.py --check
    python code/data/build_group_decks.py --source_dir "$SOURCE" --out_dir "$JOURNAL_DATA"
    python code/train/plan_e4_e5.py --manifest "$JOURNAL_DATA/GROUP_MANIFEST.json" \
      --data_dir "$JOURNAL_DATA" --runs_dir "$JOURNAL_RUNS" --out_dir "$JOBS"
    python code/train/group_pipeline.py check --plan "$PLAN"
    ;;
  check|status|verify)
    python code/train/group_pipeline.py "$STAGE" --plan "$PLAN"
    ;;
  test)
    export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
    python -m pytest -q code/tests/test_group_decks.py code/tests/test_group_workflow.py
    ;;
  analyze)
    OUT="${E4E5_RESULTS:-$(dirname "$REPO")/e4e5_results_$(date +%Y%m%d_%H%M%S)}"
    python code/analysis/analyze_e4_e5.py --plan "$PLAN" \
      --holdout "$SOURCE/j26_holdout.jsonl" --train "$SOURCE/j26_train_mix.jsonl" \
      --baseline-runs "${BASELINE_RUNS:-/data/ugrip2026/journal/runs}" --out "$OUT"
    ;;
  *) echo "unknown stage: $STAGE" >&2; exit 2;;
esac
