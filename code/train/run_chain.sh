#!/bin/bash
# Run journal_train configs one after another on ONE GPU (durable queue for a card).
#   GPU=1 nohup bash code/train/run_chain.sh code/configs/s1/shared_s1337.yaml [more.yaml ...] \
#       > /share/sda/ugrip/journal/runs/chain_gpu1.log 2>&1 &
# Each run logs to <runs>/<run_id>.log. A failed run is recorded and the chain moves on.
set -uo pipefail
GPU="${GPU:?set GPU=<free card index>}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
RUNS="${JOURNAL_RUNS:-/share/sda/ugrip/journal/runs}"
source "${CONDA_SH:-/share/sda/ugrip/taha/miniconda3/etc/profile.d/conda.sh}"
conda activate flare-ws-v5
cd "$REPO"
for CFG in "$@"; do
  RID=$(python -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['run_id'])" "$CFG")
  echo "[$(date -Is)] GPU $GPU start $RID ($CFG)"
  CUDA_VISIBLE_DEVICES=$GPU python code/train/journal_train.py --config "$CFG" > "$RUNS/$RID.log" 2>&1
  echo "[$(date -Is)] GPU $GPU end $RID rc=$?"
done
echo "[$(date -Is)] chain done"
