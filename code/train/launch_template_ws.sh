#!/bin/bash
# ============================================================================================
# JOURNAL REPO NOTE (2026-10-11): launcher pattern copied from experiments/qwen35_sprint @ bec2d1f.
# W2 replaces the plain `llamafactory-cli train` with the journal_train wrapper (per-example loss,
# plug-in hook, run registry). The conda path below is Taha's; teammates use their own.
# ============================================================================================
# Generic Qwen3.5-9B ws arm launcher — BioMedIA-A5000, env flare-ws-v5,
# LF v0.9.5 at /share/sda/ugrip/LLaMA-Factory-v5 (tf5 compat patch applied).
# Usage: GPU=<n> bash train_qwen35_arm_ws.sh <config.yaml>
# Shared box: check nvidia-smi first; pin ONE card (0/3 = teammates as of 08-30).
set -euo pipefail
CFG="${1:?usage: GPU=<n> train_qwen35_arm_ws.sh <config.yaml>}"
GPU="${GPU:?set GPU=<free card index>}"
source /share/sda/ugrip/taha/miniconda3/etc/profile.d/conda.sh
conda activate flare-ws-v5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export DISABLE_VERSION_CHECK=1
cd /share/sda/ugrip/LLaMA-Factory-v5
CUDA_VISIBLE_DEVICES=$GPU llamafactory-cli train "$CFG"
