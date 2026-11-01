#!/usr/bin/env bash
# Alex's E3/E6/E7/E8 entry point. E4/E5 retain e4_e5.sh.
# CPU preparation/analysis: Alex. Real GPU stages: Taha after Phase A and reviewed gates.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
source "${CONDA_SH:-/data/ugrip2026/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-flare-blackwell-v5}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
DATA="${JOURNAL_SOURCE:-/data/ugrip2026/journal/data}"
GROUPS_DIR="${JOURNAL_DATA:-$(dirname "$REPO")/group_decks}"
RUNS_DIR="${JOURNAL_RUNS:-/data/ugrip2026/journal/runs}"
PB="${PHASE_B_DIR:-$(dirname "$REPO")/phase_b}"
GPU_OUT="${PHASE_B_GPU_OUT:-$RUNS_DIR/phase_b}"
B="${BOOTSTRAPS:-10000}"
STAGE="${1:?usage: phase_b.sh <stage> [chain|seed]}"
case "$STAGE" in
  preflight)
    EXTRA=()
    if [ -n "${GLICLASS_INITIAL:-}" ]; then EXTRA+=(--gliclass-initial "$GLICLASS_INITIAL"); fi
    if [ -n "${RESNET_INITIAL:-}" ]; then EXTRA+=(--resnet-initial "$RESNET_INITIAL"); fi
    python code/train/phase_b_preflight.py --data "$DATA" --groups "$GROUPS_DIR" "${EXTRA[@]}" ;;
  e3-prepare)
    python code/eval/plan_e3.py prepare --data "$DATA" --runs "$RUNS_DIR" --outputs "$GPU_OUT/e3" --out "$PB/e3_jobs" ;;
  e3-check|e3-status|e3-eval|e3-crimson)
    export CUDA_VISIBLE_DEVICES=1 GPU=1
    EXTRA=(); if [ -n "${2:-}" ]; then EXTRA=(--chain "$2"); fi
    python code/eval/phase_b_jobs.py "${STAGE#e3-}" --plan "$PB/e3_jobs/plan.json" "${EXTRA[@]}" ;;
  e3-collect)
    python code/eval/plan_e3.py collect --plan "$PB/e3_jobs/plan.json" --out "$PB/e3_collected" ;;
  e3-analyze)
    python code/analysis/transfer_matrix.py --index "$PB/e3_collected/index.json" --train "$DATA/j26_train_mix.jsonl" \
      --holdout "$DATA/j26_holdout.jsonl" --out "$PB/e3_results" --B "$B" ;;
  e6-train-task|e6-train-modality)
    export CUDA_VISIBLE_DEVICES=1 GPU=1
    AXIS="${STAGE#e6-train-}"; SEED="${2:?seed: 1337, 2026, or 42}"
    if [ "$AXIS" = task ]; then INITIAL="${GLICLASS_INITIAL:?set local clean pretrained GLiClass directory}";
    else INITIAL="${RESNET_INITIAL:?set local ImageNet ResNet18 state dictionary path}"; fi
    python code/train/routers.py train --axis "$AXIS" --train "$DATA/j26_train_mix.jsonl" --holdout "$DATA/j26_holdout.jsonl" \
      --initial "$INITIAL" --out "$PB/routers/${AXIS}_s${SEED}" --seed "$SEED" ;;
  e6-predict)
    for SEED in 1337 2026 42; do
      for AXIS in task modality; do
        python code/train/routers.py predict --router "$PB/routers/${AXIS}_s${SEED}" --data "$DATA/j26_holdout.jsonl" \
          --out "$PB/routes/${AXIS}_s${SEED}.json" --device cpu
      done
    done ;;
  e6-prepare)
    python code/eval/plan_e6.py prepare --data "$DATA" --groups "$GROUPS_DIR" --runs "$RUNS_DIR" \
      --routes "$PB/routes" --outputs "$GPU_OUT/e6" --out "$PB/e6_jobs" ;;
  e6-check|e6-status|e6-eval|e6-crimson)
    export CUDA_VISIBLE_DEVICES=1 GPU=1
    EXTRA=(); if [ -n "${2:-}" ]; then EXTRA=(--chain "$2"); fi
    python code/eval/phase_b_jobs.py "${STAGE#e6-}" --plan "$PB/e6_jobs/plan.json" "${EXTRA[@]}" ;;
  e6-collect)
    python code/eval/plan_e6.py collect --plan "$PB/e6_jobs/plan.json" --out "$PB/e6_collected" ;;
  e6-analyze)
    python code/analysis/routing_results.py --index "$PB/e6_collected/index.json" --out "$PB/e6_results" --B "$B" ;;
  e7-probe)
    python code/analysis/representations.py probe --holdout "$DATA/j26_holdout.jsonl" --out "$PB/probe_ids.json" ;;
  e7-prepare)
    python code/analysis/plan_e7.py prepare --data "$DATA" --groups "$GROUPS_DIR" --runs "$RUNS_DIR" \
      --probe "$PB/probe_ids.json" --outputs "$GPU_OUT/e7" --out "$PB/e7_jobs" ;;
  e7-check|e7-run)
    export CUDA_VISIBLE_DEVICES=1 GPU=1
    EXTRA=(); if [ -n "${2:-}" ]; then EXTRA=(--chain "$2"); fi
    python code/analysis/plan_e7.py "${STAGE#e7-}" --plan "$PB/e7_jobs/plan.json" "${EXTRA[@]}" ;;
  e7-analyze)
    python code/analysis/representations.py analyze --index "$PB/e7_jobs/index.json" --probe "$PB/probe_ids.json" \
      --holdout "$DATA/j26_holdout.jsonl" --transfer "$PB/e3_results/transfer.json" --out "$PB/e7_results" --B "$B" ;;
  e8-prepare)
    python code/analysis/rq4_synthesis.py prepare --runs "$RUNS_DIR" --holdout "$DATA/j26_holdout.jsonl" --out "$PB/e8_collected" ;;
  e8-analyze)
    python code/analysis/rq4_synthesis.py analyze --index "$PB/e8_collected/index.json" --train "$DATA/j26_train_mix.jsonl" \
      --transfer "$PB/e3_results/transfer.json" --representations "$PB/e7_results/representations.json" --out "$PB/e8_results" --B "$B" ;;
  test)
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q code/tests/test_transfer_matrix.py code/tests/test_phase_b.py code/tests/test_phase_b_analysis.py ;;
  *) echo "unknown stage: $STAGE" >&2; exit 2 ;;
esac
