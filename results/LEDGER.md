# Journal results ledger

The numbers of record for the journal study. One section per experiment (PLAN.md v2: E0–E9), appended as
runs finish.

For every result, record:
- run ID(s) from `results/RUNS.csv`;
- backbone, method, data (subset / seed);
- code branch and commit;
- evaluation surface (holdout / dev / validation-public), with sample counts;
- per-dataset metrics plus the task macro, and the malformed, truncation and omission
  rates;
- deviations from PLAN.md §1, if any.

Raw predictions go to `predictions/<study>/<run_id>/`.

Challenge-era numbers never go here; they live in `../references/challenge_results/`.

---

## Setup findings (2026-10-14; moved here from STATUS on 2026-10-19)

**ws and the box are not bit-reproducible.** Same code, data, decks, packages and seed; the
50-step smoke adapters (`j26_q35_shared_full_768_s1337_smoke` on ws, `..._smoke_bw` on the box)
give different greedy outputs on 50 holdout questions per task. Identical raw outputs per task:
cls 39/50, ML 36/50, report 41/50, count 23/50, reg 15/50, det 0/50. Kernels differ (sm_120 vs
sm_86). Counting MAE 38.2 (ws) vs 76.2 (box), almost all from three degenerate answers of a barely
trained adapter. Consequence: every compared run is trained and evaluated on the box (PLAN.md §1.12).

**Throughput on GPU 1 (Qwen, 768², 50-step runs).** One run alone: 1.196 s/example (ws 1.665),
card ~53% busy. Three at once (`..._conc3_{1,2,3}`): 1.75–1.76 s/example each, ≈ 1.70 examples/s
in total = 2.0× one run; ~17 GB reserved each. Same config and seed, final losses 0.6376 / 0.6424
/ 0.6401 (alone 0.6417): run-to-run noise at the ~1% loss level even on one box.

**Full runs (3 at once, measured).** Shared s1337 ≈ 48.5 h; TaskLoRAs s1337: cls 26.9 h, report
10.0 h, ML 9.0 h, reg 6.0 h, det 4.5 h, count 1.2 h (57.6 h total); Shared r48 s1337 ≈ 52 h
(12.3 s/it averaged, partly with 4 jobs on the card).

---

**Evaluator parity, Qwen (2026-11-07).** `eval_harness.py` (needed for Hydra: `--mode plugin`) vs `eval_arm.py`
(all Qwen E0/E1/E3/E4/E5 numbers) on the same adapter, Shared s1337, 50 holdout questions per task (300), box GPU 1,
greedy, EOS pinned, thinking off, 512 tokens, 768² cap, batch 1 (`runs/parity/harness_q35_shared_s1337`, commit
`d7a16cd`, run 4 Nov). Raw outputs identical on 300/300 (same token counts); task macros identical (cls 0.8149,
ML 0.5000, det 0.8257, count 8.68, reg 3.38). Qwen Hydra holdout evals with the harness are comparable to the
`eval_arm.py` arms.

## E0 / E1 — Qwen3.5-9B, seed 1337, holdout (first numbers; 2026-10-17)

**Superseded by the 3-seed section below (2026-10-23); kept as the first record.** Provisional: one seed; report generation not yet scored; no bootstrap CIs.

- Runs: `j26_q35_zsbare_holdout_768`, `j26_q35_zsprompt_holdout_768` (E0);
  `j26_q35_shared_full_768_s1337`, `j26_q35_task{cls,det,count,reg,ml}_full_768_s1337` (E1).
- Training commit `5199a58` (Shared and TaskLoRAs); eval commit `a0ff450`, `code/eval/eval_arm.py`
  (one question at a time, greedy, EOS pinned, 512 tokens), box `mbz-yaqub` GPU 1.
- Surface: holdout, 4,447 questions (TaskLoRAs: their own task slice, oracle route).
- Scorer: `code/eval/score_single.py`. Metrics files on the box:
  `<run_dir>/metrics_holdout_noreport.json` (per dataset + task macro); predictions
  `<run_dir>/predictions_holdout.json`.

| Task (macro over datasets) | ZS-bare | ZS-prompt | Shared | Shared r48 | TaskLoRA |
|---|---:|---:|---:|---:|---:|
| Classification, BA ↑ | 0.004 | 0.118 | 0.729 | 0.739 | 0.734 |
| Multi-label, micro-F1 ↑ | 0.273 | 0.501 | 0.618 | 0.588 | 0.617 |
| Detection, F1@0.5 ↑ | 0.000 | 0.000 | 0.897 | 0.897 | 0.871 |
| Counting, MAE ↓ | 113.9 | 103.7 | 13.01 | 20.03 | 14.21 |
| Regression, MAE ↓ | 56.45 | 59.50 | 11.51 | 13.17 | 10.12 |
| — boneresorption MAE ↓ | 34.38 | 37.92 | 20.70 | 24.27 | 17.78 |
| — iugc MAE ↓ | 78.53 | 81.07 | 2.32 | 2.08 | 2.46 |
| Report generation, CRIMSON ↑ | pending | pending | pending | pending | pending |

Shared r48 (`j26_q35_shared48_full_768_s1337`, r48 α96, training commit `a0ff450`) added 2026-10-19;
0 truncated / 0 malformed.

- Fine-tuned arms: 0 truncated, 0 malformed on every task.
- ZS-bare truncated at 512 tokens on cls 244, ML 42, reg 48, count 14, report 3; malformed
  detection 99 (ZS-prompt: 41). The zero-shot cls/det numbers are mostly a format failure (the
  answer is not in a parseable form), not only a capability gap. PLAN.md E0 asks for exactly this
  ZS-prompt − ZS-bare reading; do not read 0.004 as "the model knows nothing".


---

## E0 / E1 — Qwen3.5-9B, 3 seeds, holdout (complete; 2026-10-23)

**All E0 and plain-LoRA E1 arms on Qwen: Shared, TaskLoRA, Shared r48 × seeds 1337 / 2026 / 42, plus
CRIMSON for every report output. Mean ± SD over seeds (sample SD, n = 3). No bootstrap CIs yet (deferred).**

- Runs (`results/RUNS.csv`): `j26_q35_{shared,shared48}_full_768_s{1337,2026,42}`,
  `j26_q35_task{cls,ml,report,det,reg,count}_full_768_s{1337,2026,42}`, E0 `j26_q35_zs{bare,prompt}_holdout_768`.
- Training commits: s1337 `5199a58` (Shared, TaskLoRA), `a0ff450` (r48); s2026/s42 chains launched at `fe34bff`, repo
  moved to `b2e8963` / `dfbce98` mid-chain (per-run commit in RUNS.csv). `journal_train.py` and the configs are
  identical across `fe34bff..dfbce98` (only job files, docs, `smoke.sh`, `lora_ref` changed).
- Eval: `code/eval/eval_arm.py` (one question at a time, greedy, EOS pinned, 512 tokens), eval jobs
  `code/eval/jobs/v2_e1_*.txt` (s2026/s42 at `9d3d05e`), box `mbz-yaqub` GPU 1. TaskLoRAs: own task slice (oracle route).
- Surface: holdout, 4,447 questions (report generation: 781, IU-XRay). Scorer `code/eval/score_single.py` →
  `<run_dir>/metrics_holdout_noreport.json`; predictions `<run_dir>/predictions_holdout.json`.
- CRIMSON: `code/eval/run_crimson_chain.sh` → `crimson_box.sh` (CRIMSON-medgemma-4b, env `crimson-score`, batch 1,
  commit `2da9ea8`) → `<run_dir>/crimson/crimson_metrics.json`. Pairs: raw output, think block stripped
  (`eval_arm_pairs.py`). Unscored by the judge: ZS-bare 1/781, every other run 0.
- Omission: `code/eval/reportgen_omission.py` → `<run_dir>/crimson/omission.json`.
- Truncated / malformed: 0 / 0 on every task for all 9 fine-tuned runs (E0 rates in the section above).
- Deviations from PLAN §1: none.

### Task macro (mean ± SD over 3 seeds)

| Task | ZS-bare | ZS-prompt | Shared | TaskLoRA | Shared r48 |
|---|---:|---:|---:|---:|---:|
| Classification, BA ↑ | 0.004 | 0.118 | 0.737 ± 0.009 | 0.724 ± 0.010 | 0.737 ± 0.010 |
| Multi-label, micro-F1 ↑ | 0.273 | 0.501 | 0.610 ± 0.008 | 0.614 ± 0.003 | 0.596 ± 0.008 |
| Detection, F1@0.5 ↑ | 0.000 | 0.000 | 0.903 ± 0.009 | 0.892 ± 0.021 | 0.882 ± 0.015 |
| Counting, MAE ↓ | 113.9 | 103.7 | 12.96 ± 1.39 | 15.03 ± 1.77 | 19.68 ± 1.02 |
| Regression, MAE ↓ | 56.45 | 59.50 | 11.35 ± 0.73 | 10.63 ± 0.51 | 12.31 ± 0.84 |
| Report generation, CRIMSON ↑ | 0.722 | 0.797 | 0.822 ± 0.008 | 0.829 ± 0.004 | 0.817 ± 0.008 |

### Per seed (s1337 / s2026 / s42)

| Task | Shared | TaskLoRA | Shared r48 |
|---|---|---|---|
| Classification | 0.729 / 0.735 / 0.746 | 0.734 / 0.726 / 0.714 | 0.739 / 0.746 / 0.725 |
| Multi-label | 0.618 / 0.602 / 0.609 | 0.617 / 0.612 / 0.612 | 0.588 / 0.603 / 0.597 |
| Detection | 0.897 / 0.914 / 0.899 | 0.871 / 0.913 / 0.892 | 0.897 / 0.883 / 0.867 |
| Counting | 13.01 / 11.55 / 14.33 | 14.21 / 17.07 / 13.82 | 20.03 / 18.53 / 20.48 |
| Regression | 11.51 / 10.55 / 11.99 | 10.12 / 10.63 / 11.15 | 13.17 / 11.49 / 12.27 |
| Report (CRIMSON) | 0.815 / 0.830 / 0.821 | 0.826 / 0.833 / 0.826 | 0.814 / 0.826 / 0.812 |

### Per dataset (mean ± SD over 3 seeds; per-seed values in each run's metrics file)

| Task / dataset | n | ZS-bare | ZS-prompt | Shared | TaskLoRA | Shared r48 |
|---|---:|---:|---:|---:|---:|---:|
| classification/BUS-UCLM | 52 | 0.000 | 0.056 | 0.875 ± 0.022 | 0.920 ± 0.024 | 0.920 ± 0.061 |
| classification/BUSI | 62 | 0.000 | 0.030 | 0.879 ± 0.038 | 0.869 ± 0.030 | 0.882 ± 0.022 |
| classification/CMMD | 357 | 0.040 | 0.349 | 0.478 ± 0.005 | 0.485 ± 0.012 | 0.474 ± 0.012 |
| classification/bcn20000 | 286 | 0.000 | 0.014 | 0.552 ± 0.027 | 0.561 ± 0.041 | 0.575 ± 0.014 |
| classification/bone_marrow | 72 | 0.000 | 0.087 | 0.728 ± 0.010 | 0.688 ± 0.083 | 0.701 ± 0.039 |
| classification/dental | 420 | 0.000 | 0.324 | 0.656 ± 0.009 | 0.642 ± 0.006 | 0.654 ± 0.016 |
| classification/fundus | 138 | 0.000 | 0.317 | 0.658 ± 0.010 | 0.661 ± 0.010 | 0.665 ± 0.020 |
| classification/iugc | 131 | 0.000 | 0.000 | 0.958 ± 0.022 | 0.944 ± 0.020 | 0.962 ± 0.013 |
| classification/neojaundice | 60 | 0.000 | 0.000 | 0.796 ± 0.015 | 0.736 ± 0.016 | 0.792 ± 0.017 |
| classification/retino | 111 | 0.000 | 0.000 | 0.786 ± 0.059 | 0.737 ± 0.008 | 0.741 ± 0.025 |
| counting/neurips22cell (MAE) | 100 | 113.93 | 103.70 | 12.96 ± 1.39 | 15.03 ± 1.77 | 19.68 ± 1.02 |
| detection/BUS-UCLM-det | 22 | 0.000 | 0.000 | 0.859 ± 0.026 | 0.844 ± 0.044 | 0.815 ± 0.026 |
| detection/BUSI-det | 50 | 0.000 | 0.000 | 0.851 ± 0.000 | 0.832 ± 0.020 | 0.836 ± 0.027 |
| detection/iugc | 516 | 0.000 | 0.000 | 0.999 ± 0.002 | 0.999 ± 0.001 | 0.995 ± 0.001 |
| multi-label/chestdr | 388 | 0.001 | 0.360 | 0.556 ± 0.005 | 0.560 ± 0.008 | 0.540 ± 0.001 |
| multi-label/periapical | 315 | 0.545 | 0.641 | 0.663 ± 0.016 | 0.668 ± 0.003 | 0.652 ± 0.016 |
| regression/boneresorption (MAE) | 70 | 34.38 | 37.92 | 20.43 ± 1.59 | 18.83 ± 1.04 | 22.40 ± 1.81 |
| regression/iugc (MAE) | 516 | 78.53 | 81.07 | 2.26 ± 0.15 | 2.44 ± 0.03 | 2.22 ± 0.13 |
| report_generation/iu_xray (CRIMSON) | 781 | 0.722 | 0.797 | 0.822 ± 0.008 | 0.829 ± 0.004 | 0.817 ± 0.008 |

### Report omission (normal-on-abnormal; 223 of 781 gold reports abnormal)

`normal_on_abnormal` = P(prediction is a normal statement | gold is abnormal); definition in `reportgen_omission.py`.

| Run | normal-on-abnormal ↓ | abnormal-on-normal | pred normal | unique-pred ratio | modal-pred fraction |
|---|---:|---:|---:|---:|---:|
| ZS-bare | 0.457 | 0.393 | 0.565 | 0.624 | 0.033 |
| ZS-prompt | 0.812 | 0.036 | 0.921 | 0.167 | 0.070 |
| Shared s1337 / s2026 / s42 | 0.632 / 0.596 / 0.605 | 0.041 / 0.027 / 0.041 | 0.866 / 0.866 / 0.858 | 0.138 / 0.131 / 0.133 | 0.195 / 0.234 / 0.214 |
| TaskLoRA s1337 / s2026 / s42 | 0.583 / 0.556 / 0.565 | 0.034 / 0.036 / 0.045 | 0.857 / 0.848 / 0.844 | 0.147 / 0.138 / 0.140 | 0.191 / 0.202 / 0.213 |
| Shared r48 s1337 / s2026 / s42 | 0.722 / 0.709 / 0.749 | 0.025 / 0.025 / 0.022 | 0.903 / 0.899 / 0.913 | 0.092 / 0.086 / 0.092 | 0.256 / 0.268 / 0.241 |

Means: normal-on-abnormal Shared 0.611, TaskLoRA 0.568, r48 0.727. Every fine-tuned arm misses most abnormal
findings (reports collapse toward normal boilerplate); TaskLoRA misses the fewest, r48 the most and is the most
collapsed (fewest unique reports). ZS-bare omits less only because it calls 39% of normal reports abnormal.

### Reading (descriptive; no CIs yet)
- Fine-tuning lifts every task far above zero-shot; reports the least (ZS-prompt 0.797 → ~0.82–0.83 CRIMSON).
- **TaskLoRA − Shared:** Shared ahead on cls (−0.012), detection (−0.011) and counting (+2.07 MAE); TaskLoRA ahead on
  regression (−0.72 MAE, all from boneresorption) and reports (+0.006 CRIMSON, lower omission); multi-label tied.
  Every gap is about 1–1.5 seed SDs.
- **Shared r48 vs Shared:** 6× parameters give no gain on any task; worse on counting, detection, multi-label and
  report omission.
- bone_marrow TaskLoRA s42 (0.597 vs 0.71–0.76 elsewhere) checked: all 72 answers are valid class letters; 10 classes
  with 4–12 questions each, so ~10 extra errors move BA by 0.1. Seed noise on a small dataset, not a bug.

----

## E0 / E2 — MedGemma-1.5-4B, 3 seeds, holdout (complete; 2026-10-26)

**All E0 and plain-LoRA E2 arms on MedGemma: Shared, TaskLoRA, Shared r48 × seeds 1337 / 2026 / 42, plus CRIMSON
for every report output. Mean ± SD over seeds (sample SD, n = 3). No bootstrap CIs yet (deferred).**

- Runs (`results/RUNS.csv`): `j26_mg15_{shared,shared48}_full_896_s{1337,2026,42}`,
  `j26_mg15_task{cls,ml,report,det,reg,count}_full_896_s{1337,2026,42}`, E0 `j26_mg15_zs{bare,prompt}_holdout_896`.
- Backbone `mg15` (MedGemma-1.5-4B-it, native 896², LF `gemma3` template with the `token_type_ids` patch; port
  merged `5aa8128`). P8 = 14,901,248; r48 = 89,407,488 (6× P8). Configs `code/configs/e2/`.
- Training commits: chains launched at `fd0a803`; the repo moved to `52368b1` mid-chain (only eval job files,
  `run_crimson_chain.sh` and STATUS changed), so 7 runs record `52368b1` (per-run commit in RUNS.csv).
- Eval: `code/eval/eval_harness.py --batch_size 1` (one question at a time, like Qwen's `eval_arm.py`; greedy, EOS
  pinned, 512 tokens), jobs `code/eval/jobs/v2_e0_mg15_zeroshot.txt` (commit `c0123ce`) and
  `v2_e2_mg15_holdout.txt` (`52368b1`), box `mbz-yaqub` GPU 1. TaskLoRAs: own task slice (oracle route).
- Surface: holdout, 4,447 questions. Scoring `score_run.py` (wraps `score_single.py`) →
  `<run_dir>/eval_holdout/metrics.json`; raw outputs `<run_dir>/eval_holdout/{cls,ml,det,count,reg,report}.jsonl`.
- CRIMSON: `run_crimson_chain.sh` → `crimson_box.sh` (same judge and settings as Qwen) →
  `<run_dir>/eval_holdout/crimson/crimson_metrics.json`. Unscored by the judge: ZS-prompt 13/781, every other run 0.
- Omission: `reportgen_omission.profile` via `score_run.py` → `<run_dir>/eval_holdout/omission.json` (same
  definition as the Qwen table).
- Truncated / malformed (fine-tuned): 0 / 0 on every task, except one Shared seed with ≤ 0.13% truncated on one task.
  E0 is messy: ZS-bare detection 63% malformed; ZS-prompt multi-label 21% and reports 10% truncated.
- Deviations from PLAN §1: none (896² is MedGemma's native input, per PLAN §1.1).

### Task macro (mean ± SD over 3 seeds)

| Task | ZS-bare | ZS-prompt | Shared | TaskLoRA | Shared r48 |
|---|---:|---:|---:|---:|---:|
| Classification, BA ↑ | 0.000 | 0.024 | 0.725 ± 0.009 | 0.720 ± 0.013 | 0.699 ± 0.012 |
| Multi-label, micro-F1 ↑ | 0.052 | 0.219 | 0.569 ± 0.007 | 0.583 ± 0.007 | 0.568 ± 0.009 |
| Detection, F1@0.5 ↑ | 0.019 | 0.000 | 0.468 ± 0.034 | 0.528 ± 0.028 | 0.442 ± 0.034 |
| Counting, MAE ↓ | 117.69 | 75269.41 | 25.28 ± 2.54 | 29.20 ± 0.80 | 28.35 ± 2.57 |
| Regression, MAE ↓ | 73.97 | 51.77 | 12.16 ± 0.39 | 11.91 ± 0.79 | 13.30 ± 0.46 |
| Report generation, CRIMSON ↑ | 0.584 | 0.676 | 0.825 ± 0.003 | 0.832 ± 0.002 | 0.822 ± 0.000 |

ZS-prompt counting MAE comes from a few huge numeric answers, not from generally bad counts.

### Per seed (s1337 / s2026 / s42)

| Task | Shared | TaskLoRA | Shared r48 |
|---|---|---|---|
| Classification | 0.733 / 0.727 / 0.714 | 0.727 / 0.705 / 0.728 | 0.707 / 0.706 / 0.685 |
| Multi-label | 0.576 / 0.562 / 0.568 | 0.575 / 0.582 / 0.590 | 0.575 / 0.558 / 0.571 |
| Detection | 0.459 / 0.506 / 0.440 | 0.502 / 0.558 / 0.525 | 0.466 / 0.456 / 0.403 |
| Counting | 22.66 / 27.73 / 25.46 | 28.36 / 29.28 / 29.96 | 31.31 / 27.00 / 26.73 |
| Regression | 12.58 / 12.09 / 11.81 | 12.27 / 11.00 / 12.45 | 12.90 / 13.20 / 13.80 |
| Report (CRIMSON) | 0.823 / 0.829 / 0.824 | 0.833 / 0.829 / 0.833 | 0.822 / 0.822 / 0.823 |

### Per dataset (mean ± SD over 3 seeds)

| Task / dataset | n | ZS-bare | ZS-prompt | Shared | TaskLoRA | Shared r48 |
|---|---:|---:|---:|---:|---:|---:|
| classification/BUS-UCLM | 52 | 0.000 | 0.022 | 0.862 ± 0.040 | 0.877 ± 0.050 | 0.870 ± 0.055 |
| classification/BUSI | 62 | 0.000 | 0.010 | 0.864 ± 0.014 | 0.849 ± 0.043 | 0.867 ± 0.009 |
| classification/CMMD | 357 | 0.000 | 0.205 | 0.457 ± 0.016 | 0.456 ± 0.012 | 0.449 ± 0.011 |
| classification/bcn20000 | 286 | 0.000 | 0.000 | 0.521 ± 0.040 | 0.526 ± 0.020 | 0.511 ± 0.012 |
| classification/bone_marrow | 72 | 0.000 | 0.000 | 0.707 ± 0.014 | 0.708 ± 0.016 | 0.689 ± 0.014 |
| classification/dental | 420 | 0.000 | 0.000 | 0.585 ± 0.009 | 0.578 ± 0.010 | 0.431 ± 0.008 |
| classification/fundus | 138 | 0.000 | 0.000 | 0.724 ± 0.044 | 0.709 ± 0.018 | 0.718 ± 0.018 |
| classification/iugc | 131 | 0.000 | 0.000 | 0.962 ± 0.022 | 0.965 ± 0.023 | 0.929 ± 0.007 |
| classification/neojaundice | 60 | 0.000 | 0.000 | 0.731 ± 0.026 | 0.713 ± 0.048 | 0.715 ± 0.051 |
| classification/retino | 111 | 0.000 | 0.000 | 0.834 ± 0.055 | 0.817 ± 0.035 | 0.812 ± 0.063 |
| counting/neurips22cell (MAE) | 100 | 117.69 | 75269.41 | 25.28 ± 2.54 | 29.20 ± 0.80 | 28.35 ± 2.57 |
| detection/BUS-UCLM-det | 22 | 0.000 | 0.000 | 0.178 ± 0.044 | 0.281 ± 0.051 | 0.163 ± 0.068 |
| detection/BUSI-det | 50 | 0.057 | 0.000 | 0.251 ± 0.057 | 0.316 ± 0.053 | 0.178 ± 0.052 |
| detection/iugc | 516 | 0.000 | 0.000 | 0.977 ± 0.007 | 0.988 ± 0.001 | 0.984 ± 0.002 |
| multi-label/chestdr | 388 | 0.035 | 0.108 | 0.521 ± 0.015 | 0.537 ± 0.010 | 0.510 ± 0.006 |
| multi-label/periapical | 315 | 0.069 | 0.330 | 0.617 ± 0.028 | 0.628 ± 0.020 | 0.626 ± 0.017 |
| regression/boneresorption (MAE) | 70 | 61.62 | 26.27 | 21.75 ± 0.77 | 21.02 ± 1.47 | 24.19 ± 0.82 |
| regression/iugc (MAE) | 516 | 86.32 | 77.28 | 2.57 ± 0.08 | 2.79 ± 0.20 | 2.42 ± 0.10 |
| report_generation/iu_xray (CRIMSON) | 781 | 0.584 | 0.676 | 0.825 ± 0.003 | 0.832 ± 0.002 | 0.822 ± 0.000 |

### Report omission (normal-on-abnormal; 223 of 781 gold reports abnormal)

| Run | normal-on-abnormal ↓ | abnormal-on-normal | pred normal | unique-pred ratio | modal-pred fraction |
|---|---:|---:|---:|---:|---:|
| ZS-bare | 0.206 | 0.690 | 0.280 | 0.837 | 0.035 |
| ZS-prompt | 0.471 | 0.283 | 0.647 | 0.607 | 0.035 |
| Shared s1337 / s2026 / s42 | 0.731 / 0.691 / 0.686 | 0.027 / 0.034 / 0.032 | 0.904 / 0.887 / 0.887 | 0.108 / 0.127 / 0.118 | 0.256 / 0.248 / 0.239 |
| TaskLoRA s1337 / s2026 / s42 | 0.735 / 0.700 / 0.709 | 0.030 / 0.032 / 0.029 | 0.903 / 0.891 / 0.896 | 0.099 / 0.109 / 0.105 | 0.227 / 0.245 / 0.257 |
| Shared r48 s1337 / s2026 / s42 | 0.704 / 0.749 / 0.749 | 0.027 / 0.025 / 0.030 | 0.896 / 0.910 / 0.906 | 0.092 / 0.086 / 0.078 | 0.255 / 0.259 / 0.277 |

Means: normal-on-abnormal Shared 0.703, TaskLoRA 0.715, r48 0.734 (Qwen: 0.611 / 0.568 / 0.727). MedGemma reports
collapse to normal boilerplate more than Qwen's, yet score slightly higher on CRIMSON.

### Reading (descriptive; no CIs yet)
- **TaskLoRA − Shared:** TaskLoRA ahead on detection (+0.060, every seed; BUS-UCLM and BUSI, not iugc), multi-label
  (+0.014), regression (−0.25 MAE) and reports (+0.007 CRIMSON); Shared ahead on counting (−3.9 MAE) and cls (+0.005).
  On Qwen Shared led on detection and counting, so the arm ranking depends on the backbone.
- **Shared r48 vs Shared:** no gain on any task, as on Qwen; worst on dental classification (0.431 vs 0.585).
- **MedGemma vs Qwen (same arms):** close on cls, regression and reports (MedGemma +0.003 CRIMSON); behind on
  multi-label (−0.04), counting (~2× the MAE) and detection (0.47 vs 0.90). Detection: iugc is fine (0.98); the
  ultrasound sets (BUS-UCLM, BUSI) are 0.16–0.32 vs Qwen's ~0.85.
- **Detection check (W9 item, closed).** Coordinate convention verified: on the Shared s1337 holdout, mean predicted
  box centres match gold (cx/W 0.44 vs 0.47 BUSI, 0.52 vs 0.53 BUS-UCLM) and widths are right (ratio ≈ 1). But
  predicted-vs-gold x-correlation is 0.22 (BUSI) / 0.14 (BUS-UCLM) against y 0.72 / 0.65 (Qwen x 0.67 / 0.92).
  MedGemma localises poorly along x on these wide images (aspect 1.24–1.41, squashed to 896²). Hypothesis, untested.
- Zero-shot MedGemma is weaker than zero-shot Qwen on every task except regression (with prompt).

---

## E4 / E5 — modality and classification-dataset adapters, both backbones, 3 seeds, holdout (2026-11-07)

**E4: one r8 adapter per imaging modality (7), trained on all of that modality's questions. E5: one r8 adapter per
classification dataset (10). Each is evaluated on its own holdout slice (oracle route); the slices are joined so each
seed covers the whole holdout (E4) or all classification questions (E5). Compared on the same questions with Shared,
TaskLoRA and Shared r48 (E1/E2 runs above). Hydra comparisons pending (partial analysis, labelled as such).**

- Runs: 84 (`j26_{q35,mg15}_mod<modality>_full_{768,896}_s{1337,2026,42}`, `..._ds<dataset>_...`); 42 per backbone.
  Clinical, Dermatology and Mammography hold one dataset each (neojaundice, bcn20000, CMMD): E5 reuses those E4
  adapters (byte-identical decks; PLAN changelog 26 Oct). Not yet in `results/RUNS.csv` (to add).
- Decks: 51 group decks in `/data/ugrip2026/journal/data_e4e5` cut from the frozen W1 decks (`GROUP_MANIFEST.json`).
  Job files `code/eval/jobs/e4e5/` (`f1fc40d`). Training launched 26 Oct 22:46 at `f1fc40d`, done 31 Oct 00:47;
  holdout evals done 2 Nov 03:31 (Qwen `eval_arm.py`, MedGemma `eval_harness.py --batch_size 1`, as E1/E2);
  CRIMSON on the 6 X-ray modality runs done 2 Nov 06:17. All rc=0.
- Audit: `e4_e5.sh verify` (box `d7a16cd`, 6 Nov): 102/102 slots complete (protocol config, deck hash, 3 epochs,
  exact holdout coverage, CRIMSON complete). Malformed / truncated: 0 / 0 on every task, every grouped run.
- Analysis: `code/analysis/analyze_e4_e5.py --arms shared task shared48` (box `d7a16cd`, finished 6 Nov 17:08) →
  `/data/ugrip2026/journal/e4e5_results_plain_lora_20261106/` (`E4_vs_<arm>_<bb>/`, `E5_vs_{shared,task}_<bb>/`:
  `contrasts.json`, `cells.csv`). Two-level bootstrap (seeds, then questions stratified by dataset), B = 10,000;
  per-task p Holm-adjusted across the 6 tasks of each comparison; per-cell p Holm-adjusted across cells.
- Deviations from PLAN §1: none in training or evaluation (only the grouping differs, which is the experiment).
  No MedGemma smoke on the group decks (logged deviation from §3.3, PLAN changelog 26 Oct).

### E4 task macro and contrasts (mean ± SD over 3 seeds; difference = E4 − reference, [95% CI], Holm p)

For MAE a positive difference means E4 is worse.

**Qwen3.5-9B**

| Task | E4 modality | vs Shared | vs TaskLoRA | vs Shared r48 |
|---|---:|---:|---:|---:|
| Cls, BA ↑ | 0.721 ± 0.010 | −0.016 [−0.037, +0.005] p=0.80 | −0.004 [−0.025, +0.017] p=1.00 | −0.016 [−0.040, +0.007] p=0.68 |
| ML, F1 ↑ | 0.619 ± 0.008 | +0.010 [−0.010, +0.028] p=1.00 | +0.005 [−0.013, +0.024] p=1.00 | +0.023 [+0.002, +0.045] p=0.21 |
| Det, F1 ↑ | 0.908 ± 0.011 | +0.004 [−0.035, +0.045] p=1.00 | +0.016 [−0.017, +0.057] p=1.00 | +0.026 [−0.008, +0.068] p=0.68 |
| Count, MAE ↓ | 15.12 ± 1.38 | +2.16 [−2.10, +6.44] p=1.00 | +0.09 [−3.22, +3.30] p=1.00 | −4.56 [−10.85, +1.20] p=0.64 |
| Reg, MAE ↓ | 11.76 ± 0.40 | +0.41 [−0.80, +1.66] p=1.00 | +1.12 [−0.49, +2.58] p=0.93 | −0.56 [−1.96, +0.90] p=0.86 |
| Report, CRIMSON ↑ | 0.821 ± 0.004 | −0.001 [−0.014, +0.013] p=1.00 | −0.007 [−0.018, +0.004] p=0.95 | +0.004 [−0.011, +0.019] p=0.86 |

**MedGemma-1.5-4B**

| Task | E4 modality | vs Shared | vs TaskLoRA | vs Shared r48 |
|---|---:|---:|---:|---:|
| Cls, BA ↑ | 0.704 ± 0.024 | −0.020 [−0.049, +0.010] p=1.00 | −0.016 [−0.046, +0.017] p=1.00 | +0.005 [−0.025, +0.038] p=1.00 |
| ML, F1 ↑ | 0.557 ± 0.005 | −0.012 [−0.029, +0.006] p=1.00 | −0.025 [−0.044, −0.007] p=0.06 | −0.010 [−0.030, +0.010] p=1.00 |
| Det, F1 ↑ | 0.507 ± 0.028 | +0.038 [−0.039, +0.116] p=1.00 | −0.022 [−0.088, +0.040] p=1.00 | +0.065 [−0.008, +0.141] p=0.52 |
| Count, MAE ↓ | 27.14 ± 3.79 | +1.85 [−4.85, +9.53] p=1.00 | −2.06 [−8.35, +4.61] p=1.00 | −1.21 [−8.55, +5.96] p=1.00 |
| Reg, MAE ↓ | 11.92 ± 0.84 | −0.24 [−1.43, +0.95] p=1.00 | +0.02 [−1.72, +1.78] p=1.00 | −1.38 [−3.06, +0.22] p=0.52 |
| Report, CRIMSON ↑ | 0.828 ± 0.004 | +0.003 [−0.005, +0.012] p=1.00 | −0.003 [−0.013, +0.005] p=1.00 | +0.006 [−0.005, +0.017] p=1.00 |

Counting is one modality (microscopy, which also holds bone-marrow cls) and report one (X-ray, with ChestDR,
periapical, dental, bone resorption): those E4 adapters train on more than their task's data.

### E5 classification macro (BA ↑; mean ± SD over 3 seeds; E5 − reference, [95% CI], p)

| Backbone | E5 dataset | vs Shared | vs TaskLoRA-cls |
|---|---:|---:|---:|
| Qwen3.5-9B | 0.702 ± 0.010 | −0.035 [−0.058, −0.008] p=0.010 | −0.022 [−0.048, +0.005] p=0.103 |
| MedGemma-1.5-4B | 0.666 ± 0.026 | −0.059 [−0.092, −0.026] p=0.001 | −0.054 [−0.088, −0.020] p=0.002 |

Per dataset, E5 − TaskLoRA-cls (Qwen / MedGemma; per-cell Holm p in `E5_vs_task_<bb>/contrasts.json`; none < 0.05):

| Dataset | Qwen | MedGemma |
|---|---:|---:|
| BUS-UCLM | −0.074 | −0.104 |
| BUSI | −0.000 | −0.015 |
| CMMD | −0.008 | −0.001 |
| bcn20000 | +0.050 | −0.034 |
| bone_marrow | −0.155 (raw CI excludes 0; Holm p=0.08) | −0.216 (Holm p=0.12) |
| dental | −0.002 | +0.008 |
| fundus | +0.007 | +0.008 |
| iugc | +0.021 | +0.014 |
| neojaundice | +0.021 | −0.093 |
| retino | −0.083 | −0.108 |

Exploratory Spearman (10 datasets, permutation p): gain over TaskLoRA-cls vs training size ρ = 0.16 (p=0.65) Qwen,
0.60 (p=0.08) MedGemma; vs zero-shot difficulty ρ = 0.35 (p=0.33) Qwen, 0.02 (p=0.96) MedGemma.

### Reading
- **E4 (by modality) ≈ Shared ≈ TaskLoRA:** no task differs after Holm on either backbone. Closest: MedGemma
  multi-label, E4 below TaskLoRA (−0.025, p=0.06). Cutting by modality is neither better nor worse than by task.
- **E5 (by dataset) is worse:** below Shared on both backbones, and below TaskLoRA-cls on MedGemma (−0.054, p=0.002;
  Qwen −0.022, p=0.10). The losses sit in the small datasets (bone_marrow, BUS-UCLM, retino, MedGemma neojaundice);
  large ones (CMMD, dental) are unchanged and iugc gains slightly. Classification datasets benefit from training
  together; splitting below the task level costs the small ones.
- Hydra columns: rerun `analyze_e4_e5.py` with all five arms once Hydra holdout evals exist (fresh `--out`).
