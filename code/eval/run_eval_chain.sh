#!/bin/bash
# Run eval_arm.py jobs one after another on GPU 1 (durable eval queue; same env as run_chain.sh).
#   nohup bash code/eval/run_eval_chain.sh JOBFILE > /data/ugrip2026/journal/runs/evalchain_X.log 2>&1 &
# JOBFILE: one job per line, "<name> <eval_arm.py args...>"; blank lines and # comments skipped.
# EVAL_PY=code/eval/eval_harness.py runs the harness instead (MedGemma: eval_arm.py is Qwen-only);
# each successful harness job is then scored with score_run.py on its --out_dir.
# A line "WAIT_FOR <file> <pattern>" blocks until <pattern> appears in <file> (e.g. a training chain's
# "chain done"), so an eval can be queued behind a run that is still training.
set -uo pipefail
GPU="${GPU:-1}"
if [ "$GPU" != "1" ] && [ -z "${ALLOW_OTHER_GPU:-}" ]; then
  echo "refusing GPU $GPU: our Blackwell allocation is GPU 1" >&2; exit 2
fi
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
RUNS="${JOURNAL_RUNS:-/data/ugrip2026/journal/runs}"
source "${CONDA_SH:-/data/ugrip2026/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-flare-blackwell-v5}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}" \
       OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}" NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
EVAL_PY="${EVAL_PY:-code/eval/eval_arm.py}"
cd "$REPO"
JOBS=$(grep -v '^\s*#' "$1" | grep -v '^\s*$')
while read -r NAME ARGS; do
  if [ "$NAME" = "WAIT_FOR" ]; then
    read -r F P <<< "$ARGS"
    echo "[$(date -Is)] waiting for '$P' in $F"
    until grep -q "$P" "$F" 2>/dev/null; do sleep 300; done
    continue
  fi
  echo "[$(date -Is)] GPU $GPU start eval $NAME"
  # eval keeps quoted args intact (task names contain spaces)
  eval "CUDA_VISIBLE_DEVICES=$GPU python $EVAL_PY $ARGS" < /dev/null > "$RUNS/eval_$NAME.log" 2>&1
  RC=$?
  if [ $RC -eq 0 ] && [[ "$EVAL_PY" == *eval_harness.py ]]; then
    OUT=$(sed -n 's/.*--out_dir \([^ ]*\).*/\1/p' <<< "$ARGS")
    python code/eval/score_run.py --out_dir "$OUT" >> "$RUNS/eval_$NAME.log" 2>&1
    RC=$?
  fi
  echo "[$(date -Is)] GPU $GPU end eval $NAME rc=$RC"
done <<< "$JOBS"
echo "[$(date -Is)] eval chain done"
