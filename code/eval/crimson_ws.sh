#!/bin/bash
# CRIMSON scoring of one eval folder on ws (replaces the SLURM score_reportgen_crimson.sh).
# Needs reportgen_pairs.jsonl from score_run.py; writes crimson_metrics.json + crimson_per_sample.jsonl,
# then re-runs score_run.py so metrics.json picks the score up.
#   GPU=1 bash code/eval/crimson_ws.sh <eval_out_dir>
set -euo pipefail
DIR="${1:?usage: GPU=<n> crimson_ws.sh <eval_out_dir>}"
GPU="${GPU:?set GPU=<free card index>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
CONDA_SH="${CONDA_SH:-/share/sda/ugrip/taha/miniconda3/etc/profile.d/conda.sh}"
CRIMSON_ENV="${CRIMSON_ENV:-crimson-score}"
CRIMSON_MODEL="${CRIMSON_MODEL:-/share/sda/ugrip/models/CRIMSON-medgemma-4b}"
[ -f "$DIR/reportgen_pairs.jsonl" ] || { echo "missing $DIR/reportgen_pairs.jsonl: run score_run.py first"; exit 1; }
source "$CONDA_SH"
conda activate "$CRIMSON_ENV"
export HF_HUB_OFFLINE=1
CUDA_VISIBLE_DEVICES=$GPU python "$HERE/score_crimson.py" --pred "$DIR/reportgen_pairs.jsonl" \
    --crimson_api hf --crimson_model "$CRIMSON_MODEL" --out "$DIR/crimson_metrics.json"
conda activate flare-ws-v5
python "$HERE/score_run.py" --out_dir "$DIR"
