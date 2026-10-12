#!/bin/bash
#SBATCH --job-name=crimson_score
#SBATCH --partition=ugrip
#SBATCH --qos=prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/crimson_score_%x_%j.out
# NOTE: --mail-type/--mail-user do NOT work on this cluster — never add them.
#
# OFFLINE CRIMSON scoring for the per-task held-out eval's report_generation slice (the official
# FLARE-2026 report-gen metric, which score_single.py deliberately SKIPS). No GPU re-inference: it
# reads the report-gen predictions already saved by eval_full_system.sh, strips the Qwen3 </think>
# artifact, and scores them with the medgemma CRIMSON judge in the separate `crimson-score` env.
#
#   sbatch -J crimson_valpub   scripts/score_reportgen_crimson.sh validation-public
#   sbatch -J crimson_truetest scripts/score_reportgen_crimson.sh true_test
#
# Both held-out splits are 100% IU_XRay for report-gen -> the mean CRIMSON IS the per-dataset macro.
set -euo pipefail

SPLIT="${1:?usage: score_reportgen_crimson.sh <validation-public|true_test>}"
LF_DIR="${LF_DIR:-$HOME/ME-VLIP-qwen3/llama_factory_internvl}"
OUT="$LF_DIR/results/full_system/$SPLIT"
PREDS="$OUT/preds_report_generation.json"
PAIRS="$OUT/reportgen_pairs.jsonl"
METRICS="$OUT/crimson_metrics.json"
CRIMSON_ENV="${CRIMSON_ENV:-crimson-score}"
CRIMSON_MODEL="${CRIMSON_MODEL:-$HOME/models/CRIMSON-medgemma-4b}"

cd "$LF_DIR"
[ -f "$PREDS" ] || { echo "missing $PREDS — run eval_full_system.sh $SPLIT first"; exit 1; }

module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CRIMSON_ENV"

# --- convert eval preds -> CRIMSON pairs {gt, pred, sample_key}, stripping the </think> artifact ---
# (infer_single_adapter's report-gen parser does NOT strip </think>; CRIMSON judges raw text, so strip here.)
python - "$PREDS" "$PAIRS" <<'PY'
import json, sys, re
src, dst = sys.argv[1], sys.argv[2]
TB = re.compile(r"^\s*<think>.*?</think>\s*", re.S)
OC = re.compile(r"^\s*</think>\s*")
def strip(o):
    o = TB.sub("", o); o = OC.sub("", o); return o.strip()
data = json.load(open(src)); n = 0
with open(dst, "w") as f:
    for i, s in enumerate(data):
        key = s.get("ImageName")
        key = key if isinstance(key, str) else json.dumps(key)
        f.write(json.dumps({"gt": s.get("gt_answer", ""),
                            "pred": strip(s.get("Answer", "")),
                            "sample_key": f"{key}#{i}"}) + "\n")
        n += 1
print(f"wrote {n} report-gen pairs -> {dst}")
PY

# --- score with the CRIMSON judge (mean over the report-gen pairs) ---
echo "=== [$SPLIT] CRIMSON scoring (env=$CRIMSON_ENV, model=$CRIMSON_MODEL) ==="
python "$(dirname "$0")/score_crimson.py" \
    --pred "$PAIRS" \
    --crimson_api hf \
    --crimson_model "$CRIMSON_MODEL" \
    --out "$METRICS"

echo "=== [$SPLIT] CRIMSON DONE -> $METRICS ==="
cat "$METRICS"
