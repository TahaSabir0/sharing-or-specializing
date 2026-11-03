#!/bin/bash
# Launch E3 on GPU 1 (Taha only): 3 slots, one per seed. Each slot runs its Qwen evals
# (eval_arm.py), then its MedGemma evals (eval_harness.py), and a CRIMSON chain waits for it.
#   bash code/eval/jobs/e3/launch_e3.sh      (from /data/ugrip2026/journal/repo)
set -u
cd "$(dirname "$0")/../../../.."
RUNS=/data/ugrip2026/journal/runs
J=code/eval/jobs/e3
for S in 1337 2026 42; do
  nohup bash -c "bash code/eval/run_eval_chain.sh $J/eval_q35_s$S.txt > $RUNS/evalchain_e3_q35_s$S.log 2>&1; \
EVAL_PY=code/eval/eval_harness.py bash code/eval/run_eval_chain.sh $J/eval_mg15_s$S.txt > $RUNS/evalchain_e3_mg15_s$S.log 2>&1" \
    > /dev/null 2>&1 &
  echo "eval slot s$S PID $!"
  nohup bash code/eval/run_crimson_chain.sh $J/crimson_s$S.txt > $RUNS/crimsonchain_e3_s$S.log 2>&1 &
  echo "crimson s$S PID $!"
done
