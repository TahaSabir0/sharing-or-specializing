# E4/E5: preparation, real training, evaluation and analysis

102 runs: 42 E4 modality adapters (7 modalities) and 60 E5 classification-dataset adapters
(10 datasets), each on Qwen and MedGemma, seeds 1337/2026/42, the fixed 3-epoch r8 protocol.
Only the grouping level differs from PLAN.md §1 (the subject of E4/E5).

Real runs launch the same way as E1/E2: `run_chain.sh`, `run_eval_chain.sh` and
`run_crimson_chain.sh` with committed job files. Taha only (PLAN.md §7.3). The E4/E5 scripts
themselves are CPU-only: they build decks, plan, check and audit; they never launch GPU work.

## 1. Decks and job files (CPU; done once, in the shared repo)

```bash
cd /data/ugrip2026/journal/repo
bash code/train/e4_e5.sh prepare
```

- `build_group_decks.py` cuts the frozen W1 decks (`j26_train_mix`, `j26_devselect`,
  `j26_holdout`, md5-checked against `MANIFEST.json`) into 51 group decks in
  `/data/ugrip2026/journal/data_e4e5/` with `dataset_info.json` and `GROUP_MANIFEST.json`.
  Rows are copied unchanged; the frozen data dir is never written.
- `plan_e4_e5.py` writes `code/eval/jobs/e4e5/`: `train_chain_{1,2,3}.txt` (balanced by
  training rows; MedGemma weighted 0.6), `eval_q35.txt`, `eval_mg15.txt`, `crimson.txt`,
  `plan.json` (config and deck hashes). These are committed.
- Each backbone is evaluated with the script its baselines used (research rule 3): Qwen with
  `eval_arm.py` (as E1), MedGemma with `eval_harness.py --batch_size 1` (as E2).

A teammate can run `prepare` from a personal clone: decks and job files then stay next to the
clone (`../group_decks`, `../e4e5_jobs`), for smoke tests:

```bash
JOURNAL_DATA=/data/ugrip2026/journal/people/<you>/group_decks \
  bash code/train/smoke.sh <you> code/configs/e4/modxray_q35_s1337.yaml
```

## 2. Launch (Taha; after Phase A frees GPU 1)

Check first: `nvidia-smi -i 1` and `bash code/train/e4_e5.sh check` (configs and decks still
match `plan.json`). Training reads the group decks, so export `JOURNAL_DATA`:

```bash
cd /data/ugrip2026/journal/repo
export JOURNAL_DATA=/data/ugrip2026/journal/data_e4e5
for i in 1 2 3; do
  nohup bash code/train/run_chain.sh $(cat code/eval/jobs/e4e5/train_chain_$i.txt) \
    > /data/ugrip2026/journal/runs/chain_e4e5_$i.log 2>&1 &
done
```

When the training chains end (`chain done` in each log):

```bash
nohup bash code/eval/run_eval_chain.sh code/eval/jobs/e4e5/eval_q35.txt \
  > /data/ugrip2026/journal/runs/evalchain_e4e5_q35.log 2>&1 &
EVAL_PY=code/eval/eval_harness.py nohup bash code/eval/run_eval_chain.sh code/eval/jobs/e4e5/eval_mg15.txt \
  > /data/ugrip2026/journal/runs/evalchain_e4e5_mg15.log 2>&1 &
```

Then CRIMSON on the six X-ray modality runs:

```bash
nohup bash code/eval/run_crimson_chain.sh code/eval/jobs/e4e5/crimson.txt \
  > /data/ugrip2026/journal/runs/crimsonchain_e4e5.log 2>&1 &
```

A failed run is logged (`rc=` in the chain log) and the chain moves on, as in E1/E2.

## 3. Audit

```bash
bash code/train/e4_e5.sh status   # which runs have runtime.json / predictions
bash code/train/e4_e5.sh verify   # all 102: protocol run, exact holdout coverage, CRIMSON; exit 1 otherwise
```

`verify` rejects smoke or deviating configs, a wrong training deck or deck hash, a step cap,
missing adapter files, predictions that miss, duplicate or alter holdout questions, `Error:`
answers, and missing or incomplete CRIMSON scores.

## 4. CPU results and comparisons (Alex)

After `verify` passes:

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
report scores fail validation (bootstrap.py would otherwise drop reports silently).
A non-finite numeric metric is preserved in the raw scores but statistical
comparison stops rather than reporting a fake CI.

If a Phase-A arm is not available yet (Hydra), explicitly label a partial analysis:

```bash
python code/analysis/analyze_e4_e5.py \
  --plan code/eval/jobs/e4e5/plan.json \
  --holdout /data/ugrip2026/journal/data/j26_holdout.jsonl \
  --train /data/ugrip2026/journal/data/j26_train_mix.jsonl \
  --out ../e4e5_results_plain_lora --arms shared task shared48
```

Historical artifacts may be `predictions_holdout.json` plus `crimson/`, or
`all_predictions.json` plus `run_meta.json`. Both are accepted. If a historical
run uses a different directory name, use `group_results.py baseline --sources
<file-or-folder> ...` and then `compare_groups.py --a <three normalized dirs>
--b <three grouped dirs> --train <train deck> --out <new analysis dir>`.
No holdout is re-inferred.

The analysis directories record source paths, hashes and metadata; the run directories keep
config, commit, deck, runtime and registry. Send results and log paths to Taha for
RUNS.csv/LEDGER. E3, E6 and E7/E8 are separate tasks.
