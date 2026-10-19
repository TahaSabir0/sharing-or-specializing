#!/bin/bash
# Smoke test on mbz-yaqub GPU 1: the only GPU job teammates launch themselves (PLAN.md §7.3).
# 50 training steps of a run config + a 10-questions-per-task eval on dev-select, then a summary.
#
#   bash code/train/smoke.sh <your_name> <config.yaml> [--no-eval]
#
# Run it from YOUR clone (/data/ugrip2026/journal/people/<your_name>/repo), on your branch.
# What it enforces, so nobody has to remember:
#   - one smoke test at a time on the box (a lock; if it is taken you are told who holds it);
#   - GPU 1 only; refuses if GPU 1 has < 25 GB free (the 3 real runs need theirs);
#   - always short: --max_steps 50, no dev-loss eval, a 1-hour hard limit;
#   - output only under /data/ugrip2026/journal/runs/smoke/ (never next to real runs);
#   - CPU threads capped (the box has 512 cores; uncapped workers swamp it).
# Exit code: 0 = all steps ran (read the summary: passing the gate is a judgement, PLAN.md §3.3).
set -uo pipefail
WHO="${1:?usage: smoke.sh <your_name> <config.yaml> [--no-eval]}"
CFG="${2:?usage: smoke.sh <your_name> <config.yaml> [--no-eval]}"
DO_EVAL=1; [ "${3:-}" = "--no-eval" ] && DO_EVAL=0
case "$WHO" in *[!a-z0-9_]*|"") echo "name must be lowercase letters/digits/_ (e.g. veronika)"; exit 2;; esac
[ -f "$CFG" ] || { echo "no such config: $CFG"; exit 2; }

GPU=1
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
SMOKE_ROOT=/data/ugrip2026/journal/runs/smoke
mkdir -p "$SMOKE_ROOT"
if [ "$REPO" = "/data/ugrip2026/journal/repo" ] && [ "$WHO" != "taha" ]; then
  echo "run smoke tests from your own clone (/data/ugrip2026/journal/people/$WHO/repo), not the shared repo"; exit 2
fi

# ---- one smoke test at a time
LOCK="$SMOKE_ROOT/.lock"
exec 9>>"$LOCK"
if ! flock -n 9; then
  echo "another smoke test is running: $(cat "$SMOKE_ROOT/.holder" 2>/dev/null || echo unknown). Try again later."
  exit 3
fi
echo "$WHO since $(date '+%F %H:%M') ($(basename "$CFG"), pid $$)" > "$SMOKE_ROOT/.holder"
trap 'rm -f "$SMOKE_ROOT/.holder"' EXIT

# ---- room on GPU 1?
FREE=$(nvidia-smi -i $GPU --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
if [ "${FREE:-0}" -lt 25000 ]; then
  echo "GPU 1 has only ${FREE} MiB free (< 25 GB): the real runs need it. Try again later or tell Taha."
  exit 4
fi

source "${CONDA_SH:-/data/ugrip2026/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-flare-blackwell-v5}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export JOURNAL_RUNS="$SMOKE_ROOT" CUDA_VISIBLE_DEVICES=$GPU
cd "$REPO"

STAMP=$(date +%m%d%H%M)
SUFFIX="smoke_${WHO}_${STAMP}"
RID=$(python -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['run_id'])" "$CFG")_$SUFFIX
METHOD=$(python -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1])).get('method','lora'))" "$CFG")
RUN_DIR="$SMOKE_ROOT/$RID"
LOG="$SMOKE_ROOT/$RID.log"
echo "[smoke] $WHO: $RID  (branch $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD))"
echo "[smoke] training 50 steps -> $LOG"

timeout 3600 python code/train/journal_train.py --config "$CFG" --max_steps 50 --no_eval \
    --run_suffix "$SUFFIX" > "$LOG" 2>&1
TRC=$?
[ $TRC -eq 124 ] && echo "[smoke] training hit the 1-hour limit"

ERC=skipped
if [ $TRC -eq 0 ] && [ $DO_EVAL -eq 1 ]; then
  echo "[smoke] eval: 10 questions per task on dev-select -> $RUN_DIR/eval_smoke"
  if [ "$METHOD" = "lora" ]; then MODE=(--mode adapter --adapter "$RUN_DIR"); else MODE=(--mode plugin --run_dir "$RUN_DIR"); fi
  timeout 1800 python code/eval/eval_harness.py "${MODE[@]}" \
      --data /data/ugrip2026/journal/data/j26_devselect.jsonl --surface devselect --per_task 10 \
      --out_dir "$RUN_DIR/eval_smoke" >> "$LOG" 2>&1 \
    && python code/eval/score_run.py --out_dir "$RUN_DIR/eval_smoke" >> "$LOG" 2>&1
  ERC=$?
fi

# ---- summary (gate items 1, 2, 4, 5, 6 of PLAN.md §3.3; item 3, the template gate, is separate)
python - "$RUN_DIR" "$LOG" "$TRC" "$ERC" <<'EOF'
import json, os, re, sys
run_dir, log, trc, erc = sys.argv[1:]
print(f"\n===== smoke summary: {os.path.basename(run_dir)} =====")
print(f"training exit code: {trc}   eval: {erc}")
tl = os.path.join(run_dir, "trainer_log.jsonl")
if os.path.exists(tl):
    losses = [json.loads(l)["loss"] for l in open(tl) if '"loss"' in l]
    if losses:
        print(f"1. loss: first {losses[0]:.3f} -> last {losses[-1]:.3f}  ({'decreasing' if losses[-1] < losses[0] else 'NOT decreasing'})")
txt = open(log, errors="replace").read()
m = re.findall(r"trainable params: ([\d,]+)", txt)
print(f"2. trainable params: {m[-1] if m else 'not printed (plug-ins: see method_config.json n_params)'}")
mc = os.path.join(run_dir, "method_config.json")
if os.path.exists(mc):
    print(f"   method_config.json n_params: {json.load(open(mc)).get('n_params')}")
rt = os.path.join(run_dir, "runtime.json")
if os.path.exists(rt):
    r = json.load(open(rt))
    print(f"6. runtime.json: " + ", ".join(f"{k}={r[k]}" for k in r if k in
          ("s_per_it", "s_per_example", "peak_mem_reserved_gb", "trainable_params", "trainable_matches_P8")))
    print("   (s/it is inflated while the 3 real runs share the card; compare like with like)")
met = os.path.join(run_dir, "eval_smoke", "metrics.json")
if os.path.exists(met):
    mt = json.load(open(met))
    print("4. eval (dev-select, 10/task; a format check, not a result):")
    for t, v in mt.get("tasks", {}).items():
        keep = {k: v.get(k) for k in ("n", "malformed_rate", "truncation_rate", "errors", "macro")}
        print(f"   {t}: {keep}")
rs = os.path.join(run_dir, "eval_smoke", "routing_stats.npz")
if os.path.exists(rs):
    sys.path.insert(0, "code")
    from methods.base import validate_routing_stats
    try:
        print(f"5. routing_stats.npz: {validate_routing_stats(rs)}")
    except Exception as e:
        print(f"5. routing_stats.npz INVALID: {e}")
if "Traceback" in txt:
    print("!! a Traceback is in the log:")
    print(txt[txt.rfind("Traceback"):][:2000])
print(f"log: {log}")
EOF
exit $(( TRC != 0 ))
