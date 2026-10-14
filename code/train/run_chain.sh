#!/bin/bash
# Run journal_train configs one after another on ONE GPU (durable queue for a card).
# Default box: mbz-yaqub (Blackwell); our allocation is GPU 1 only (ops/cluster-config.md §B).
#   nohup bash code/train/run_chain.sh code/configs/s1/shared_s1337.yaml [more.yaml ...] \
#       > /data/ugrip2026/journal/runs/chain_gpu1.log 2>&1 &
# Each run logs to <runs>/<run_id>.log. A failed run is recorded and the chain moves on.
set -uo pipefail
GPU="${GPU:-1}"
if [ "$GPU" != "1" ] && [ -z "${ALLOW_OTHER_GPU:-}" ]; then
  echo "refusing GPU $GPU: our Blackwell allocation is GPU 1 (set ALLOW_OTHER_GPU=1 on another box)" >&2
  exit 2
fi
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
RUNS="${JOURNAL_RUNS:-/data/ugrip2026/journal/runs}"
mkdir -p "$RUNS"
source "${CONDA_SH:-/data/ugrip2026/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-flare-blackwell-v5}"
cd "$REPO"
for CFG in "$@"; do
  RID=$(python -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['run_id'])" "$CFG")
  echo "[$(date -Is)] GPU $GPU start $RID ($CFG)"
  CUDA_VISIBLE_DEVICES=$GPU python code/train/journal_train.py --config "$CFG" > "$RUNS/$RID.log" 2>&1
  RC=$?
  echo "[$(date -Is)] GPU $GPU end $RID rc=$RC"
done
echo "[$(date -Is)] chain done"
