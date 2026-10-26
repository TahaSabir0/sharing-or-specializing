# E4/E5: preparation, real training, evaluation and analysis

The code implements 42 E4 modality runs and 60 E5 classification-dataset runs:
both Qwen/MedGemma, seeds 1337/2026/42, the frozen 3-epoch r8 protocol. A smoke
test is only a gate; it does not produce the experimental results.

Alex's actual clone is `/data/ugrip2026/journal/people/thinh/repo`.
GPU 1 is shared. Under PLAN section 7.3 Taha launches the real stages after
Phase A training finishes. The commands below are the operator handoff as well
as the CPU preparation commands Alex can run now. No new scheduler permissions
or hardware allocation are implied by these scripts.

## 1. Update and prepare (Alex; CPU)

Do not update a checkout while a queue is running from it. In a fresh SSH shell:

```bash
cd /data/ugrip2026/journal/people/thinh/repo &&
git pull --ff-only origin alex/modality-decks &&
bash code/train/e4_e5.sh prepare
```

The wrapper activates `flare-blackwell-v5` before using Python and caps threads.
It writes 51 group decks, `dataset_info.json`, and `GROUP_MANIFEST.json` to
`../group_decks`, then makes `../e4e5_jobs/plan.json` plus training, evaluation,
and CRIMSON job lists. Expected final message: `CHECK PASS: 102 configs ...`.
Every question/prompt/image field is preserved; dev-select is train-seen and
holdout IDs/image groups must be disjoint. Source/derived hashes are checked.

```bash
bash code/train/e4_e5.sh test     # CPU regression/integration tests; requires pytest
bash code/train/e4_e5.sh check    # checks real deck and config hashes, no GPU
bash code/train/e4_e5.sh status   # reports artifacts present; not a claim of completion
```

Overrides (set before prepare and keep identical for all stages): `JOURNAL_SOURCE`
(original W1 decks), `JOURNAL_DATA` (derived decks), `JOURNAL_RUNS`, `E4E5_JOBS`.
On Taha's merged checkout, explicitly set writable data/job directories before
preparation. No shared data directory is modified unless selected explicitly.

## 2. Gates (Alex, before handing off)

Run CPU template gates after activating the frozen environment:

```bash
source /data/ugrip2026/miniconda3/etc/profile.d/conda.sh
conda activate flare-blackwell-v5
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export JOURNAL_DATA=/data/ugrip2026/journal/people/thinh/group_decks
CUDA_VISIBLE_DEVICES= python code/eval/template_gate.py --backbone q35 \
  "$JOURNAL_DATA/j26_train_modxray.jsonl" "$JOURNAL_DATA/j26_devselect_modxray.jsonl"
CUDA_VISIBLE_DEVICES= python code/eval/template_gate.py --backbone mg15 \
  "$JOURNAL_DATA/j26_train_dsbonemarrow.jsonl" "$JOURNAL_DATA/j26_devselect_dsbonemarrow.jsonl"
```

The existing smoke gate remains `bash code/train/smoke.sh thinh <config>`.
Representative configs are modxray and dsbonemarrow on both backbones, seed 1337.
Training now streams to the terminal and its saved log; evaluation failures return
nonzero. Save the summaries for review. No smoke result is a journal result.

## 3. Real execution (designated operator)

Use the reviewed/merged checkout and the prepared plan. A single chain performs
its full training queue, followed by exact grouped holdout inference and CRIMSON.
The scripts pin GPU 1; inference is batch 1 on both backbones. Training settings
come unchanged from `journal_train.py` and the 102 configs (no step cap).

```bash
nvidia-smi -i 1
tmux new -s e4e5-chain1
```

Inside tmux, from the prepared checkout:

```bash
set -o pipefail
{ bash code/train/e4_e5.sh train 1 &&
  bash code/train/e4_e5.sh eval 1 &&
  bash code/train/e4_e5.sh crimson 1; } 2>&1 | tee ../e4e5_chain1.log
```

Detach with **Ctrl+B, release, D**. SSH can disconnect; tmux keeps running.
Reconnect with `tmux attach -t e4e5-chain1`. Logs stream actual training progress.
The operator can run chains 2 and 3 in their own tmux sessions by replacing every
`1` above; there are at most three concurrent jobs, on the same allocated GPU 1.
Do not launch extra chains alongside an already full Phase A allocation.

Use the Python runner via the wrapper instead of passing raw job lists to legacy
chain scripts: it stops on failures, checks completion, checks exact coverage,
and locks each run against duplicate workers. It skips completed stages only
after artifact checks. An interrupted training/eval directory is preserved and
causes an actionable error; it is never automatically deleted or restarted over.
The operator should inspect it and deliberately recover or archive it before
retrying. Do not delete another person's files or clear locks held by live jobs.

## 4. CPU results and comparisons (Alex)

After **all 102 runs**, grouped holdout evaluations and six X-ray CRIMSON jobs finish:

```bash
bash code/train/e4_e5.sh analyze
```

This reads the trained runs and existing E0/E1/E2 predictions. It writes a fresh
`../e4e5_results_<timestamp>` containing:

- `grouped/E4_<backbone>_s<seed>`: all seven modality slices joined by question ID.
- `grouped/E5_<backbone>_s<seed>`: all ten classification dataset slices joined.
- Normalized existing Shared, TaskLoRA, r48, Hydra-T, Hydra-R and ZS-prompt outputs.
- `E4_vs_<arm>_<backbone>/contrasts.json` and `cells.csv` for the five reference arms.
- `E5_vs_{shared,task}_<backbone>`: classification macro and per-dataset gains;
  correlations with training size and ZS-prompt difficulty.

Metrics use the existing scorer. Task metrics are unweighted dataset macros,
not averages of modality macros. Reports use the original per-question CRIMSON
scores. JSON includes mean/SD across seeds, 10k two-level bootstrap contrast CIs,
raw and Holm-adjusted p-values across tasks; cell comparisons have a separate
Holm family. CSV benefit is oriented positive=better (MAE differences negated).
Error predictions, missing/duplicate IDs, wrong ground truth and incomplete
report scores fail validation. A non-finite numeric metric is preserved in the
raw scores but statistical comparison stops rather than reporting a fake CI.

If a Phase-A arm is not available yet, explicitly label a partial analysis:

```bash
python code/analysis/analyze_e4_e5.py \
  --plan ../e4e5_jobs/plan.json \
  --holdout /data/ugrip2026/journal/data/j26_holdout.jsonl \
  --train /data/ugrip2026/journal/data/j26_train_mix.jsonl \
  --out ../e4e5_results_plain_lora --arms shared task shared48
```

Historical artifacts may be `predictions_holdout.json` plus `crimson/`, or
`all_predictions.json` plus `run_meta.json`. Both are accepted. If a historical
run uses a different directory name, use `group_results.py baseline --sources
<file-or-folder> ...` and then `compare_groups.py --a <three normalized dirs>
--b <three grouped dirs> --train <train deck> --out <new analysis dir>`.
Baseline metadata supplied by the caller for legacy files is recorded as such;
validate it against RUNS.csv before making paper claims. No holdout is re-inferred.

The new analysis directories include source paths/hashes and metadata, and the
training directories retain the existing config/commit/deck/runtime/registry
artifacts. Submit results and log paths to Taha for RUNS.csv/LEDGER integration.
E3 transfer, E6 routing and E7/E8 representations/synthesis are separate tasks;
this package implements E4/E5.
