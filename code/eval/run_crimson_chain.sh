#!/bin/bash
# Run CRIMSON scoring jobs one after another on GPU 1 (durable queue, like run_eval_chain.sh).
#   nohup bash code/eval/run_crimson_chain.sh JOBFILE > /data/ugrip2026/journal/runs/crimsonchain_X.log 2>&1 &
# JOBFILE: one job per line, "<name> <run_dir>"; blank lines and # comments skipped.
# For each job: if <run_dir>/crimson/reportgen_pairs.jsonl is missing, build it from
# <run_dir>/reportgen_pairs.jsonl (an eval_harness.py folder) or from
# <run_dir>/predictions_holdout.json (eval_arm_pairs.py), then score it with crimson_box.sh
# (batch size 1, fixed for everything compared). Output: <run_dir>/crimson/crimson_metrics.json.
# A line "WAIT_FOR <file> <pattern>" blocks until <pattern> appears in <file>.
set -uo pipefail
GPU="${GPU:-1}"
if [ "$GPU" != "1" ] && [ -z "${ALLOW_OTHER_GPU:-}" ]; then
  echo "refusing GPU $GPU: our Blackwell allocation is GPU 1" >&2; exit 2
fi
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
RUNS="${JOURNAL_RUNS:-/data/ugrip2026/journal/runs}"
CONDA_SH="${CONDA_SH:-/data/ugrip2026/miniconda3/etc/profile.d/conda.sh}"
cd "$REPO"
JOBS=$(grep -v '^\s*#' "$1" | grep -v '^\s*$')
while read -r NAME ARGS; do
  if [ "$NAME" = "WAIT_FOR" ]; then
    read -r F P <<< "$ARGS"
    echo "[$(date -Is)] waiting for '$P' in $F"
    until grep -q "$P" "$F" 2>/dev/null; do sleep 300; done
    continue
  fi
  DIR="$ARGS"
  PAIRS="$DIR/crimson/reportgen_pairs.jsonl"
  LOG="$RUNS/crimson_$NAME.log"
  echo "[$(date -Is)] GPU $GPU start crimson $NAME"
  if [ ! -f "$PAIRS" ] && [ -f "$DIR/reportgen_pairs.jsonl" ]; then
    # eval_harness.py folder (MedGemma): score_run.py already wrote the pairs
    mkdir -p "$DIR/crimson" && cp "$DIR/reportgen_pairs.jsonl" "$PAIRS"
  fi
  if [ ! -f "$PAIRS" ]; then
    ( source "$CONDA_SH" && conda activate "${CONDA_ENV:-flare-blackwell-v5}" && \
      python code/eval/eval_arm_pairs.py "$DIR/predictions_holdout.json" "$PAIRS" ) > "$LOG" 2>&1
  fi
  if [ -f "$PAIRS" ]; then
    GPU=$GPU bash code/eval/crimson_box.sh "$PAIRS" < /dev/null >> "$LOG" 2>&1
    RC=$?
  else
    echo "no pairs for $NAME" >> "$LOG"; RC=1
  fi
  echo "[$(date -Is)] GPU $GPU end crimson $NAME rc=$RC"
done <<< "$JOBS"
echo "[$(date -Is)] crimson chain done"
