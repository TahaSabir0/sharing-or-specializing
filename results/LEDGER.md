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
- ~~Hydra columns: rerun `analyze_e4_e5.py` with all five arms once Hydra holdout evals exist (fresh `--out`).~~ Done 16 Nov: see "E4 vs Hydra" below.

### E4 vs Hydra-T and Hydra-R (2026-11-16)
- `analyze_e4_e5.py --arms hydraT hydraR` (box `930bbb5`, 16 Nov 09:29–15:36) → `/data/ugrip2026/journal/e4e5_results_hydra_20261116/`
  (`E4_vs_hydra{T,R}_<bb>/contrasts.json`). Same method as above (two-level bootstrap, B = 10,000, Holm across the 6 tasks).
  E5 vs Shared/TaskLoRA-cls recomputed in the same run: identical to the table above.
- **No task differs after Holm on either backbone.** E4 − Hydra (MAE: positive = E4 worse):

| Task | Qwen − Hydra-T | Qwen − Hydra-R | MedGemma − Hydra-T | MedGemma − Hydra-R |
|---|---:|---:|---:|---:|
| Cls | −0.015 [−0.035, +0.006] p=0.83 | −0.014 [−0.038, +0.008] p=1.00 | −0.027 [−0.061, +0.008] p=0.57 | −0.008 [−0.038, +0.025] p=1.00 |
| ML | −0.013 [−0.033, +0.008] p=1.00 | −0.000 [−0.019, +0.019] p=1.00 | −0.017 [−0.039, +0.004] p=0.57 | −0.010 [−0.031, +0.010] p=1.00 |
| Det | −0.016 [−0.061, +0.025] p=1.00 | +0.007 [−0.037, +0.053] p=1.00 | −0.014 [−0.077, +0.054] p=1.00 | −0.009 [−0.078, +0.059] p=1.00 |
| Count | +1.52 [−2.25, +5.39] p=1.00 | +0.59 [−3.74, +4.56] p=1.00 | +5.88 [−0.52, +13.19] p=0.52 | +3.85 [−3.16, +11.65] p=1.00 |
| Reg | +0.78 [−0.48, +2.02] p=1.00 | +0.65 [−0.97, +2.12] p=1.00 | +0.32 [−1.27, +1.89] p=1.00 | −0.21 [−2.10, +1.68] p=1.00 |
| Report | −0.002 [−0.015, +0.010] p=1.00 | −0.009 [−0.021, +0.003] p=0.99 | +0.006 [−0.005, +0.016] p=0.93 | +0.005 [−0.005, +0.015] p=1.00 |

- No per-cell difference survives Holm. E5 is not compared with Hydra by this script (its references are Shared and
  TaskLoRA-cls); descriptively both Hydra arms are above E5 on classification.

---

## E3 — transfer matrix, both backbones, 3 seeds, holdout (2026-11-08)

**Each TaskLoRA_i (E1/E2 adapters) evaluated on every other task j's holdout slice; diagonal = the E1/E2 numbers.
Outcome per pair: PGF/μ = (M − M_base)/(best − M_base) with M_base = ZS-prompt (MAE: (MAE_base − MAE)/MAE_base):
the fraction of the zero-shot gap the off-task adapter closes; 0 = no help, < 0 = worse than zero-shot.**

- Inference: 36 full-holdout evals (6 TaskLoRAs × 3 seeds × 2 backbones), job files `code/eval/jobs/e3/`, launched
  3 Nov at `fcbe2fa` (Qwen `eval_arm.py`, MedGemma `eval_harness.py --batch_size 1`, as E1/E2); outputs
  `/data/ugrip2026/journal/runs/e3/e3_<bb>_task<src>_s<seed>/`. CRIMSON on the report slice of the 30 non-report
  runs, done 5 Nov 22:27 (14 MedGemma jobs resumed beside Hydra; STATUS).
- Collection: `plan_e3.py collect` at `d119d5f` (7 Nov) → `/data/ugrip2026/journal/phase_b/e3_collected/`.
  Analysis: `transfer_matrix.py` at `cc8f561`, B = 10,000 (two-level: seeds, then questions stratified by
  dataset), finished 8 Nov 12:22 → `/data/ugrip2026/journal/phase_b/e3_results/transfer.json` + heatmaps.
  Covariate test recomputed at `08b5905` (`--recompute-covariates`; previous block kept as
  `covariate_tests_superseded`).
- Rules (PLAN changelog 7–8 Nov): unscored CRIMSON reports (judge gave no score) left out of report means for
  zero-shot baselines and non-report adapters' report cells — Qwen det→report 1/0/2, count→report 1/0/0;
  MedGemma cls→report 1/13/0, ml→report 0/1/2, det→report 7/2/0, count→report 3/3/0, reg→report 1/0/0 (per seed,
  of 781). Overflowing numbers scored like no number: MedGemma reg→count s2026 is undefined (MAE inf).
- Pair significance: paired bootstrap of C_ij vs ZS-prompt, Holm across the 5 targets of each source (★ = p < .05).
- Deviations from PLAN §1: none.

### μ matrix (mean over 3 seeds; row = adapter, column = target task)

**Qwen3.5-9B**

| src \ tgt | cls | ml | det | count | reg | report |
|---|---:|---:|---:|---:|---:|---:|
| cls | — | −0.397★ | 0.000 | +0.036 | +0.108★ | −0.100 |
| ml | −0.033★ | — | 0.000 | −0.006 | +0.018 | −0.203★ |
| det | −0.123★ | −0.401★ | — | −1.194★ | −1.279★ | −0.248★ |
| count | −0.012 | −0.001 | 0.000 | — | +0.156★ | +0.007 |
| reg | +0.074★ | −0.076★ | 0.000 | +0.123 | — | −0.127 |
| report | −0.021 | −0.128★ | 0.000 | +0.030 | +0.089★ | — |

**MedGemma-1.5-4B** (counting column: ZS-prompt MAE 75,269, so μ ≈ 1 for any sane output; not interpretable)

| src \ tgt | cls | ml | det | count | reg | report |
|---|---:|---:|---:|---:|---:|---:|
| cls | — | +0.035 | 0.000 | (0.948★) | −0.440 | −0.406★ |
| ml | +0.011★ | — | 0.000 | (0.854★) | +0.012 | −0.241★ |
| det | −0.017★ | −0.247★ | — | (0.998★) | −1.419★ | +0.372★ |
| count | +0.177★ | −0.081 | 0.000 | — | +0.200★ | +0.166 |
| reg | −0.022★ | −0.260★ | 0.000 | undefined (s2026) | — | +0.423★ |
| report | −0.024★ | −0.275★ | 0.000 | (0.998★) | −0.056★ | — |

Significant pairs: Qwen 14/30, MedGemma 19/30. Report column = non-report adapters answering report questions
(often not reports): kept and flagged, not interpreted.

### Diagnostics
- **Detection column = 0 everywhere:** ZS-prompt detection F1 is 0 and no non-detection adapter produces valid boxes
  (malformed: Qwen reg→det 83%, report→det 28%, count→det 26%; MedGemma 64–100% for every source).
- **Sign flips vs ZS-bare base** (any seed): Qwen 17 pairs (cls→count/ml/report, count→cls/ml/report, det→cls/ml/report,
  ml→cls/count/reg/report, reg→ml/report, report→cls/ml); MedGemma 13 (cls→count/reg/report, count→ml,
  det→cls/count/ml, ml→count/reg/report, reg→cls, report→count/reg). Not flipped: det→count, det→reg (both
  backbones), and the positive transfers into regression on Qwen (cls, count, report → reg) and reg→cls.
- **IUGC natural experiment** (same ultrasound images, three output formats; μ on the iugc cell): Qwen cls→reg +0.222,
  det→reg +0.231, reg→cls +0.038; MedGemma cls→reg −0.433, det→reg −0.223, reg→cls 0.000; every →det 0.000.
- Transferability / malleability (Sachdeva Eq 1/2, off-diagonal, μ): in `transfer.json` `aggregates`. MedGemma's
  include the counting column, so they are inflated; use with care.

### RQ3 covariate test (Spearman of μ vs pair covariate; exact label permutations; Holm across the 5 covariates)

| Covariate | Qwen ρ (p, Holm p), 30 pairs | MedGemma ρ (p, Holm p), 25 pairs, counting target excluded |
|---|---|---|
| Source training size | −0.03 (0.92, 1.00) | −0.35 (0.14, 0.47) |
| Target training size | −0.38 (0.17, 0.83) | +0.17 (0.43, 0.87) |
| Output-structure similarity | +0.19 (0.60, 1.00) | +0.12 (0.51, 0.87) |
| Modality overlap | −0.15 (0.57, 1.00) | −0.36 (0.12, 0.47) |
| Dataset overlap | +0.08 (0.77, 1.00) | −0.31 (0.03, 0.17) |

### Reading
- Off-task transfer is mostly absent or negative; the detection adapter hurts every other task on both backbones
  (worse than zero-shot on counting and regression, μ ≈ −1.2 to −1.4).
- Regression is the target others help (Qwen: cls, count, report → reg, +0.09 to +0.16).
- No pair covariate predicts transfer on either backbone after Holm (RQ3: a negative result).
- The IUGC same-image comparison disagrees between backbones (helps regression on Qwen, hurts it on MedGemma).
- Many near-zero cells change sign with the zero-shot base; only the large effects are robust.

----

## E1 — Qwen3.5-9B Hydra-T, 3 seeds, holdout (2026-11-08)

**Hydra-T (shared LoRA A, one B per task; Veronika's plug-in) on Qwen, seeds 1337 / 2026 / 42, with CRIMSON.
Mean ± SD over seeds (sample SD, n = 3). Descriptive: no bootstrap CIs or tests yet (they come with the E1 CI pass).**

- Runs: `j26_q35_hydraT_full_768_s{1337,2026,42}`; trained by `chain_hydra{1,2,3}` from 4 Nov (box `d7a16cd`),
  done 7 Nov ~03:40, all rc=0.
- Eval: `code/eval/eval_harness.py --mode plugin` (oracle task route, batch 1, greedy, 512 tokens), box `38d183e`, jobs
  `code/eval/jobs/hydra/eval_q35_hydraT_s<seed>.txt`. Comparable to the `eval_arm.py` arms: parity check 7 Nov,
  300/300 identical raw outputs (Setup findings).
- Outputs: `<run_dir>/eval_holdout/` (`metrics_score_single.json`, `all_predictions.json`, `crimson/`,
  `omission.json`). Task macro = mean of the per-dataset values in `metrics_score_single.json` (its `by_task` block is
  pooled over datasets and is not used).
- CRIMSON unscored: 0/781 in every seed. Truncated / malformed: 0 / 0 on every task, every seed.
- Deviations from PLAN §1: none.

### Task macro (mean ± SD over 3 seeds; Shared and TaskLoRA from the E0/E1 section above)

| Task | Hydra-T | per seed (s1337 / s2026 / s42) | Shared | TaskLoRA | Hydra-T − Shared | Hydra-T − TaskLoRA |
|---|---:|---|---:|---:|---:|---:|
| Classification, BA ↑ | 0.736 ± 0.009 | 0.726 / 0.740 / 0.741 | 0.737 | 0.724 | −0.001 | +0.012 |
| Multi-label, micro-F1 ↑ | 0.632 ± 0.012 | 0.638 / 0.639 / 0.618 | 0.610 | 0.614 | +0.022 | +0.018 |
| Detection, F1@0.5 ↑ | 0.923 ± 0.008 | 0.928 / 0.914 / 0.928 | 0.903 | 0.892 | +0.020 | +0.031 |
| Counting, MAE ↓ | 13.61 ± 2.53 | 13.55 / 11.11 / 16.16 | 12.96 | 15.03 | +0.65 | −1.42 |
| Regression, MAE ↓ | 10.97 ± 0.32 | 10.87 / 10.72 / 11.33 | 11.35 | 10.63 | −0.38 | +0.34 |
| Report generation, CRIMSON ↑ | 0.823 ± 0.007 | 0.821 / 0.818 / 0.831 | 0.822 | 0.829 | +0.001 | −0.006 |

### Per dataset (mean ± SD over 3 seeds)

| Task / dataset | n | Hydra-T | Shared | TaskLoRA |
|---|---:|---:|---:|---:|
| classification/BUS-UCLM | 52 | 0.889 ± 0.026 | 0.875 | 0.920 |
| classification/BUSI | 62 | 0.858 ± 0.032 | 0.879 | 0.869 |
| classification/CMMD | 357 | 0.461 ± 0.011 | 0.478 | 0.485 |
| classification/bcn20000 | 286 | 0.584 ± 0.049 | 0.552 | 0.561 |
| classification/bone_marrow | 72 | 0.719 ± 0.053 | 0.728 | 0.688 |
| classification/dental | 420 | 0.659 ± 0.005 | 0.656 | 0.642 |
| classification/fundus | 138 | 0.682 ± 0.011 | 0.658 | 0.661 |
| classification/iugc | 131 | 0.963 ± 0.024 | 0.958 | 0.944 |
| classification/neojaundice | 60 | 0.809 ± 0.019 | 0.796 | 0.736 |
| classification/retino | 111 | 0.731 ± 0.009 | 0.786 | 0.737 |
| counting/neurips22cell (MAE) | 100 | 13.61 ± 2.53 | 12.96 | 15.03 |
| detection/BUS-UCLM-det | 22 | 0.919 ± 0.026 | 0.859 | 0.844 |
| detection/BUSI-det | 50 | 0.852 ± 0.000 | 0.851 | 0.832 |
| detection/iugc | 516 | 0.999 ± 0.001 | 0.999 | 0.999 |
| multi-label/chestdr | 388 | 0.568 ± 0.006 | 0.556 | 0.560 |
| multi-label/periapical | 315 | 0.696 ± 0.019 | 0.663 | 0.668 |
| regression/boneresorption (MAE) | 70 | 19.69 ± 0.67 | 20.43 | 18.83 |
| regression/iugc (MAE) | 516 | 2.25 ± 0.03 | 2.26 | 2.44 |
| report_generation/iu_xray (CRIMSON) | 781 | 0.823 ± 0.007 | 0.822 | 0.829 |

### Report omission (normal-on-abnormal; 223 of 781 gold reports abnormal)

| Run | normal-on-abnormal ↓ | abnormal-on-normal | pred normal | unique-pred ratio | modal-pred fraction |
|---|---:|---:|---:|---:|---:|
| Hydra-T s1337 / s2026 / s42 | 0.583 / 0.561 / 0.579 | 0.039 / 0.045 / 0.031 | 0.853 / 0.843 / 0.858 | 0.146 / 0.167 / 0.149 | 0.193 / 0.199 / 0.186 |

Mean normal-on-abnormal 0.574 (Shared 0.611, TaskLoRA 0.568, r48 0.727).

### Reading (descriptive; no CIs yet)
- Hydra-T is level with Shared on classification, counting, regression and reports, and ahead on multi-label
  (+0.022 F1, both datasets) and detection (+0.020 F1, all from BUS-UCLM-det, n = 22). Both gaps are about 2 seed SDs;
  whether they hold needs the bootstrap.
- Against TaskLoRA it is ahead on classification, multi-label, detection and counting, behind on regression
  (boneresorption) and reports. On Qwen, sharing A and splitting B by task does not cost anything relative to either
  plain arm.
- Report omission sits with TaskLoRA (0.574 vs 0.568), below Shared.
- Counting has the largest seed spread of any arm (SD 2.53; s42 16.16).

----

## E1 / E2 — Hydra-R (Qwen) and Hydra-T, Hydra-R (MedGemma), 3 seeds, holdout (2026-11-13)

**The remaining Hydra arms (Veronika's plug-in): Reversed Hydra (one A per task, shared B) on Qwen; Hydra-T and
Reversed Hydra on MedGemma. Seeds 1337 / 2026 / 42, with CRIMSON. Mean ± SD over seeds (sample SD, n = 3).
Descriptive: no bootstrap CIs or tests yet.** With this section every Phase A arm has holdout numbers on both backbones.

- Runs: `j26_q35_hydraR_full_768_s{1337,2026,42}`, `j26_mg15_hydra{T,R}_full_896_s{1337,2026,42}`; trained by
  `chain_hydra{1,2,3}` from 4 Nov (box `d7a16cd`); Qwen Hydra-R done 9 Nov ~18:10, MedGemma Hydra-T 10 Nov ~16:10,
  MedGemma Hydra-R 11 Nov ~14:10; all rc=0.
- Parameters (`runtime.json`): active per example = P8 for every Hydra arm. Stored: Qwen Hydra-T 78,716,928
  (3.64 P8), Hydra-R 72,757,248 (3.36 P8); MedGemma Hydra-T 55,287,808 (3.71 P8), Hydra-R 49,020,928 (3.29 P8).
- Eval: `code/eval/eval_harness.py --mode plugin` (oracle task route, batch 1, greedy, EOS pinned, 512 tokens), box
  `997e0f5`, launched 12 Nov 14:03, done 18:02 (CRIMSON 18:48); jobs `code/eval/jobs/hydra/{eval,crimson}_hydra_rest_s<seed>.txt`.
  MedGemma baselines were evaluated with the same harness; Qwen harness/`eval_arm.py` parity: Setup findings.
- Outputs: `<run_dir>/eval_holdout/` as for Qwen Hydra-T. Task macro = mean of the per-dataset values in
  `metrics_score_single.json`.
- CRIMSON unscored: 0/781 in every run. Truncated / malformed: 0 / 0 on every task, every run.
- Deviations from PLAN §1: none.

### Qwen3.5-9B Hydra-R: task macro (Shared and TaskLoRA from the E0/E1 section)

| Task | Hydra-R | Hydra-R per seed | Shared | TaskLoRA | Hydra-R − Shared | Hydra-R − TaskLoRA |
|---|---:|---|---:|---:|---:|---:|
| Classification, BA ↑ | 0.734 ± 0.007 | 0.739 / 0.737 / 0.727 | 0.737 | 0.724 | −0.003 | +0.010 |
| Multi-label, micro-F1 ↑ | 0.619 ± 0.009 | 0.613 / 0.615 / 0.629 | 0.610 | 0.614 | +0.009 | +0.005 |
| Detection, F1@0.5 ↑ | 0.901 ± 0.030 | 0.891 / 0.935 / 0.877 | 0.903 | 0.892 | −0.002 | +0.009 |
| Counting, MAE ↓ | 14.53 ± 0.80 | 15.39 / 13.80 / 14.41 | 12.96 | 15.03 | +1.57 | −0.50 |
| Regression, MAE ↓ | 11.11 ± 1.08 | 12.13 / 11.21 / 9.98 | 11.35 | 10.63 | −0.24 | +0.48 |
| Report generation, CRIMSON ↑ | 0.830 ± 0.006 | 0.823 / 0.835 / 0.831 | 0.822 | 0.829 | +0.008 | +0.001 |

### Qwen3.5-9B Hydra-R: per dataset (mean ± SD over 3 seeds)

| Task / dataset | n | Hydra-R | Shared | TaskLoRA |
|---|---:|---:|---:|---:|
| classification/BUS-UCLM | 52 | 0.867 ± 0.013 | 0.875 | 0.920 |
| classification/BUSI | 62 | 0.853 ± 0.005 | 0.879 | 0.869 |
| classification/CMMD | 357 | 0.478 ± 0.007 | 0.478 | 0.485 |
| classification/bcn20000 | 286 | 0.557 ± 0.036 | 0.552 | 0.561 |
| classification/bone_marrow | 72 | 0.735 ± 0.007 | 0.728 | 0.688 |
| classification/dental | 420 | 0.652 ± 0.011 | 0.656 | 0.642 |
| classification/fundus | 138 | 0.675 ± 0.034 | 0.658 | 0.661 |
| classification/iugc | 131 | 0.965 ± 0.015 | 0.958 | 0.944 |
| classification/neojaundice | 60 | 0.805 ± 0.012 | 0.796 | 0.736 |
| classification/retino | 111 | 0.757 ± 0.025 | 0.786 | 0.737 |
| counting/neurips22cell (MAE) | 100 | 14.53 ± 0.80 | 12.96 | 15.03 |
| detection/BUS-UCLM-det | 22 | 0.859 ± 0.068 | 0.859 | 0.844 |
| detection/BUSI-det | 50 | 0.845 ± 0.023 | 0.851 | 0.832 |
| detection/iugc | 516 | 0.999 ± 0.001 | 0.999 | 0.999 |
| multi-label/chestdr | 388 | 0.562 ± 0.004 | 0.556 | 0.560 |
| multi-label/periapical | 315 | 0.676 ± 0.016 | 0.663 | 0.668 |
| regression/boneresorption (MAE) | 70 | 19.85 ± 2.22 | 20.43 | 18.83 |
| regression/iugc (MAE) | 516 | 2.36 ± 0.10 | 2.26 | 2.44 |
| report_generation/iu_xray (CRIMSON) | 781 | 0.830 ± 0.006 | 0.822 | 0.829 |

### Qwen3.5-9B Hydra-R: report omission

| Run | normal-on-abnormal ↓ | abnormal-on-normal | pred normal | unique-pred ratio | modal-pred fraction |
|---|---:|---:|---:|---:|---:|
| Hydra-R s1337 / s2026 / s42 | 0.596 / 0.592 / 0.579 | 0.030 / 0.032 / 0.027 | 0.863 / 0.860 / 0.860 | 0.134 / 0.115 / 0.127 | 0.211 / 0.251 / 0.237 |

Mean normal-on-abnormal 0.589 (Hydra-T 0.574, Shared 0.611, TaskLoRA 0.568, r48 0.727).

### MedGemma-1.5-4B Hydra-T and Hydra-R: task macro (Shared and TaskLoRA from the E0/E2 section)

| Task | Hydra-T | Hydra-T per seed | Hydra-R | Hydra-R per seed | Shared | TaskLoRA | Hydra-T − Shared | Hydra-T − TaskLoRA | Hydra-R − Shared | Hydra-R − TaskLoRA |
|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|
| Classification, BA ↑ | 0.731 ± 0.022 | 0.750 / 0.737 / 0.707 | 0.713 ± 0.012 | 0.714 / 0.700 / 0.723 | 0.725 | 0.720 | +0.006 | +0.011 | −0.012 | −0.007 |
| Multi-label, micro-F1 ↑ | 0.575 ± 0.012 | 0.586 / 0.576 / 0.562 | 0.568 ± 0.012 | 0.582 / 0.563 / 0.558 | 0.569 | 0.583 | +0.006 | −0.008 | −0.001 | −0.015 |
| Detection, F1@0.5 ↑ | 0.520 ± 0.017 | 0.500 / 0.530 / 0.531 | 0.515 ± 0.023 | 0.488 / 0.528 / 0.530 | 0.468 | 0.528 | +0.052 | −0.008 | +0.047 | −0.013 |
| Counting, MAE ↓ | 21.26 ± 1.68 | 23.15 / 19.94 / 20.68 | 23.29 ± 1.92 | 25.32 / 23.04 / 21.50 | 25.28 | 29.20 | −4.02 | −7.94 | −1.99 | −5.91 |
| Regression, MAE ↓ | 11.60 ± 0.75 | 11.99 / 12.08 / 10.74 | 12.13 ± 1.14 | 13.32 / 11.06 / 12.01 | 12.16 | 11.91 | −0.56 | −0.31 | −0.03 | +0.22 |
| Report generation, CRIMSON ↑ | 0.823 ± 0.005 | 0.823 / 0.827 / 0.818 | 0.823 ± 0.004 | 0.826 / 0.819 / 0.824 | 0.825 | 0.832 | −0.002 | −0.009 | −0.002 | −0.009 |

### MedGemma-1.5-4B Hydra-T and Hydra-R: per dataset (mean ± SD over 3 seeds)

| Task / dataset | n | Hydra-T | Hydra-R | Shared | TaskLoRA |
|---|---:|---:|---:|---:|---:|
| classification/BUS-UCLM | 52 | 0.886 ± 0.048 | 0.878 ± 0.049 | 0.862 | 0.877 |
| classification/BUSI | 62 | 0.869 ± 0.040 | 0.855 ± 0.009 | 0.864 | 0.849 |
| classification/CMMD | 357 | 0.453 ± 0.015 | 0.446 ± 0.017 | 0.457 | 0.456 |
| classification/bcn20000 | 286 | 0.514 ± 0.026 | 0.524 ± 0.023 | 0.521 | 0.526 |
| classification/bone_marrow | 72 | 0.726 ± 0.044 | 0.696 ± 0.009 | 0.707 | 0.708 |
| classification/dental | 420 | 0.594 ± 0.017 | 0.503 ± 0.079 | 0.585 | 0.578 |
| classification/fundus | 138 | 0.731 ± 0.011 | 0.721 ± 0.025 | 0.724 | 0.709 |
| classification/iugc | 131 | 0.962 ± 0.032 | 0.985 ± 0.009 | 0.962 | 0.965 |
| classification/neojaundice | 60 | 0.731 ± 0.000 | 0.742 ± 0.010 | 0.731 | 0.713 |
| classification/retino | 111 | 0.845 ± 0.057 | 0.776 ± 0.020 | 0.834 | 0.817 |
| counting/neurips22cell (MAE) | 100 | 21.26 ± 1.68 | 23.29 ± 1.92 | 25.28 | 29.20 |
| detection/BUS-UCLM-det | 22 | 0.237 ± 0.026 | 0.281 ± 0.026 | 0.178 | 0.281 |
| detection/BUSI-det | 50 | 0.336 ± 0.041 | 0.284 ± 0.060 | 0.251 | 0.316 |
| detection/iugc | 516 | 0.988 ± 0.004 | 0.981 ± 0.000 | 0.977 | 0.988 |
| multi-label/chestdr | 388 | 0.536 ± 0.008 | 0.528 ± 0.008 | 0.521 | 0.537 |
| multi-label/periapical | 315 | 0.614 ± 0.018 | 0.607 ± 0.032 | 0.617 | 0.628 |
| regression/boneresorption (MAE) | 70 | 20.61 ± 1.45 | 21.71 ± 2.23 | 21.75 | 21.02 |
| regression/iugc (MAE) | 516 | 2.60 ± 0.10 | 2.55 ± 0.07 | 2.57 | 2.79 |
| report_generation/iu_xray (CRIMSON) | 781 | 0.823 ± 0.005 | 0.823 ± 0.004 | 0.825 | 0.832 |

### MedGemma-1.5-4B Hydra-T and Hydra-R: report omission

| Run | normal-on-abnormal ↓ | abnormal-on-normal | pred normal | unique-pred ratio | modal-pred fraction |
|---|---:|---:|---:|---:|---:|
| Hydra-T s1337 / s2026 / s42 | 0.682 / 0.659 / 0.655 | 0.036 / 0.043 / 0.043 | 0.883 / 0.872 / 0.871 | 0.131 / 0.131 / 0.137 | 0.237 / 0.241 / 0.229 |
| Hydra-R s1337 / s2026 / s42 | 0.700 / 0.700 / 0.668 | 0.029 / 0.034 / 0.036 | 0.894 / 0.890 / 0.880 | 0.110 / 0.108 / 0.117 | 0.255 / 0.252 / 0.259 |

Means: normal-on-abnormal Hydra-T 0.665, Hydra-R 0.689 (Shared 0.703, TaskLoRA 0.715, r48 0.734).

### Reading (descriptive; no CIs yet)
- **Qwen Hydra-R** is level with Shared on cls, detection and regression, slightly ahead on multi-label (+0.009) and
  reports (+0.008 CRIMSON, the highest Qwen value with TaskLoRA's 0.829), behind on counting (+1.57 MAE, between
  Shared and TaskLoRA). Detection has the widest seed spread of any Qwen arm (0.877–0.935).
- **Qwen Hydra-T vs Hydra-R:** T ahead on multi-label (+0.013), detection (+0.022) and counting (−0.92 MAE), R ahead on
  reports (+0.007); cls and regression level. All within about one to two seed SDs.
- **MedGemma Hydra-T** is level with or ahead of Shared on every task except reports (−0.002): counting −4.02 MAE (the
  best MedGemma counting; TaskLoRA is +3.9 worse than Shared), detection +0.052 (close to TaskLoRA's +0.060, again from
  BUS-UCLM and BUSI), regression −0.56, cls +0.006. Reports stay below TaskLoRA (−0.009 CRIMSON).
- **MedGemma Hydra-R** gets most of Hydra-T's detection gain (+0.047 vs Shared) and part of its counting gain (−1.99),
  ties Shared on multi-label and regression, and is lowest on cls (−0.012), all from dental: 0.503 ± 0.079
  (s1337 / s2026 / s42 = 0.519 / 0.418 / 0.573 vs Shared 0.585). Shared r48 shows the same dental drop (0.431).
  0 malformed answers; not investigated further.
- **Pattern across backbones (to be tested with the bootstrap):** Hydra-T is never behind Shared by more than seed noise
  on either backbone, and on each backbone it keeps the gains of whichever plain arm leads a task (MedGemma detection
  from TaskLoRA, counting from Shared). Hydra-T is at least level with Hydra-R on every task on both backbones except
  Qwen reports, so sharing A and splitting B looks the better split.
- **Report omission:** on MedGemma both Hydra arms omit fewer abnormal findings than any plain arm (0.665 / 0.689 vs
  0.703–0.734); on Qwen both sit between TaskLoRA and Shared.

----

## E1 / E2 — paired bootstrap between arms, both backbones (2026-11-18)

**No difference between any two arms survives Holm correction, on any task, on either backbone** (smallest adjusted
p = 0.076, Qwen Shared r48 − Shared counting). No per-dataset cell survives Holm either. The "Reading" paragraphs in
the E1/E2 sections above are descriptive; this section is the test.

- `code/analysis/e1_e2_bootstrap.sh` (box `5de7899`, 18 Nov 08:48–09:57, CPU) → `/data/ugrip2026/journal/e1e2_bootstrap_20261118/`
  (`<b>_minus_<a>_<bb>/contrasts.json`, `cells.csv`). Same method as E4/E5 (`compare_groups.py`): two-level paired bootstrap
  (seeds, then questions stratified by dataset), B = 10,000, seed 20261013; Holm across the 6 tasks of each comparison;
  per-cell p Holm-adjusted across cells. Inputs: the full-holdout runs assembled by `analyze_e4_e5.py`
  (`e4e5_results_hydra_20261116/`, Shared r48 from `e4e5_results_plain_lora_20261106/`); means match the E1/E2 tables.
- Values: difference b − a (mean over seeds) [95% CI] Holm p.

**Qwen3.5-9B** (b − a; MAE: negative = b better)

| b − a | Cls | ML | Det | Count | Reg | Report |
|---|---:|---:|---:|---:|---:|---:|
| TaskLoRA − Shared | -0.012 [-0.034, +0.008] p=1.00 | +0.004 [-0.015, +0.023] p=1.00 | -0.011 [-0.057, +0.034] p=1.00 | +2.07 [-2.08, +6.49] p=1.00 | -0.72 [-2.19, +0.84] p=1.00 | +0.006 [-0.007, +0.020] p=1.00 |
| TaskLoRA − Shared r48 | -0.012 [-0.036, +0.010] p=0.56 | +0.018 [-0.002, +0.039] p=0.44 | +0.010 [-0.027, +0.052] p=0.60 | -4.65 [-11.04, +1.31] p=0.54 | -1.68 [-3.39, +0.11] p=0.35 | +0.011 [-0.004, +0.026] p=0.54 |
| Shared r48 − Shared | +0.000 [-0.023, +0.022] p=1.00 | -0.014 [-0.036, +0.007] p=0.97 | -0.021 [-0.065, +0.012] p=0.97 | +6.72 [+1.89, +12.08] p=0.08 | +0.96 [-0.60, +2.57] p=0.97 | -0.005 [-0.020, +0.011] p=1.00 |
| Hydra-T − Hydra-R | +0.001 [-0.020, +0.020] p=1.00 | +0.013 [-0.008, +0.032] p=1.00 | +0.022 [-0.022, +0.075] p=1.00 | -0.93 [-4.94, +3.19] p=1.00 | -0.13 [-1.63, +1.32] p=1.00 | -0.007 [-0.020, +0.007] p=1.00 |
| Hydra-T − Shared | -0.001 [-0.021, +0.017] p=1.00 | +0.022 [+0.001, +0.042] p=0.21 | +0.020 [-0.008, +0.056] p=1.00 | +0.64 [-3.26, +4.92] p=1.00 | -0.38 [-1.69, +0.96] p=1.00 | +0.001 [-0.013, +0.016] p=1.00 |
| Hydra-T − TaskLoRA | +0.011 [-0.010, +0.031] p=1.00 | +0.018 [-0.002, +0.037] p=0.44 | +0.031 [-0.016, +0.086] p=1.00 | -1.43 [-5.38, +2.41] p=1.00 | +0.34 [-0.86, +1.53] p=1.00 | -0.005 [-0.018, +0.008] p=1.00 |
| Hydra-R − Shared | -0.002 [-0.021, +0.018] p=1.00 | +0.010 [-0.010, +0.029] p=1.00 | -0.002 [-0.048, +0.043] p=1.00 | +1.57 [-1.32, +5.26] p=1.00 | -0.24 [-1.88, +1.41] p=1.00 | +0.008 [-0.006, +0.022] p=1.00 |
| Hydra-R − TaskLoRA | +0.010 [-0.012, +0.035] p=1.00 | +0.005 [-0.013, +0.023] p=1.00 | +0.009 [-0.035, +0.056] p=1.00 | -0.50 [-4.61, +3.54] p=1.00 | +0.47 [-1.11, +1.99] p=1.00 | +0.001 [-0.011, +0.013] p=1.00 |

**MedGemma-1.5-4B** (b − a; MAE: negative = b better)

| b − a | Cls | ML | Det | Count | Reg | Report |
|---|---:|---:|---:|---:|---:|---:|
| TaskLoRA − Shared | -0.005 [-0.029, +0.018] p=1.00 | +0.013 [-0.006, +0.033] p=0.83 | +0.060 [-0.013, +0.134] p=0.66 | +3.92 [-2.50, +10.70] p=0.83 | -0.25 [-1.79, +1.34] p=1.00 | +0.006 [-0.003, +0.016] p=0.83 |
| TaskLoRA − Shared r48 | +0.021 [-0.002, +0.047] p=0.45 | +0.015 [-0.006, +0.035] p=0.45 | +0.087 [+0.018, +0.161] p=0.11 | +0.85 [-5.75, +6.83] p=0.78 | -1.39 [-3.02, +0.18] p=0.45 | +0.009 [-0.002, +0.021] p=0.45 |
| Shared r48 − Shared | -0.025 [-0.049, -0.002] p=0.23 | -0.001 [-0.022, +0.019] p=1.00 | -0.027 [-0.098, +0.040] p=1.00 | +3.06 [-2.86, +8.68] p=1.00 | +1.14 [-0.30, +2.56] p=0.58 | -0.003 [-0.015, +0.009] p=1.00 |
| Hydra-T − Hydra-R | +0.019 [-0.009, +0.047] p=1.00 | +0.007 [-0.015, +0.028] p=1.00 | +0.005 [-0.059, +0.061] p=1.00 | -2.03 [-7.75, +3.33] p=1.00 | -0.53 [-2.27, +1.24] p=1.00 | -0.000 [-0.010, +0.010] p=1.00 |
| Hydra-T − Shared | +0.007 [-0.020, +0.033] p=1.00 | +0.006 [-0.016, +0.027] p=1.00 | +0.052 [-0.015, +0.120] p=0.66 | -4.03 [-9.37, +0.74] p=0.66 | -0.56 [-2.03, +0.89] p=1.00 | -0.002 [-0.013, +0.008] p=1.00 |
| Hydra-T − TaskLoRA | +0.011 [-0.017, +0.040] p=1.00 | -0.008 [-0.030, +0.014] p=1.00 | -0.008 [-0.073, +0.055] p=1.00 | -7.94 [-14.41, -2.10] p=0.09 | -0.30 [-1.75, +1.21] p=1.00 | -0.009 [-0.019, +0.001] p=0.38 |
| Hydra-R − Shared | -0.012 [-0.035, +0.011] p=1.00 | -0.001 [-0.023, +0.020] p=1.00 | +0.047 [-0.024, +0.116] p=1.00 | -2.00 [-7.67, +3.62] p=1.00 | -0.03 [-1.72, +1.64] p=1.00 | -0.002 [-0.012, +0.008] p=1.00 |
| Hydra-R − TaskLoRA | -0.007 [-0.032, +0.020] p=1.00 | -0.015 [-0.038, +0.007] p=0.75 | -0.013 [-0.079, +0.050] p=1.00 | -5.91 [-13.06, +0.24] p=0.40 | +0.22 [-1.52, +1.95] p=1.00 | -0.009 [-0.018, +0.001] p=0.40 |


- Closest calls (CI excludes 0, Holm p > 0.05): Qwen Shared r48 − Shared counting +6.72 MAE (p=0.08); MedGemma
  Hydra-T − TaskLoRA counting −7.94 MAE (p=0.10); MedGemma TaskLoRA − Shared r48 detection +0.087 (p=0.11); Qwen
  Hydra-T − Shared multi-label +0.022 (p=0.21); MedGemma Shared r48 − Shared classification −0.025 (p=0.23).
- MedGemma TaskLoRA − Shared detection (+0.060, every seed, from the breast-ultrasound datasets) is not significant
  (CI [−0.013, +0.134], p=0.66): BUS-UCLM-det has n=22 and BUSI-det n=50.

---

## E7 — representations, main test (input similarity vs transfer), both backbones (2026-11-14)

**Does an adapter help a target dataset more when the base model sees the two tasks' inputs as similar? No, on either
backbone.** Descriptive measures (shift from base, adapter-pair CKA, seed ceilings) are in the output file and not yet
interpreted.

- Features: 164 forward-only configurations (base + every adapter, 3 seeds, both backbones) on the 562 frozen probe IDs
  (`data/e7_probe_ids.json`), `phase_b.sh e7-run`, box `0cc6217`, 13 Nov 00:01–03:33; `runs/phase_b/e7/`.
- Analysis: `phase_b.sh e7-analyze`, box `e89ac0b`, 14 Nov 10:37–10:52, B = 10,000 →
  `/data/ugrip2026/journal/phase_b/e7_results/representations.json`. Transfer from `phase_b/e3_results/transfer.json`.
- Test (PLAN changelog 2026-11-14): Spearman ρ of base-model centroid cosine similarity (source task's probe inputs vs
  target cell's) against μ (PGF, mean over seeds), off-task pairs only (95 Qwen; 90 MedGemma, counting excluded as in
  E3); target cells permuted within target task; Holm across layers per pooling. Supplementary: RSA profile.
- First run (raw C, all 114 pairs incl. own-task cells) gave ρ +0.25 to +0.51 on Qwen, all significant, by construction;
  MedGemma undefined. Kept as `phase_b/e7_results_rawC_alltasks_20261114`; not a result.

| Backbone | Pooling | ρ range over layers (centroid) | smallest Holm p | RSA ρ range |
|---|---|---|---|---|
| Qwen3.5-9B (L1, 8, 16, 24, 32) | image tokens | −0.07 to +0.03 | 1.0 | −0.07 to +0.04 |
| | question tokens | −0.12 to +0.08 | 1.0 | 0.00 to +0.13 |
| | last prompt token | 0.00 to +0.27 | 1.0 | +0.01 to +0.27 |
| MedGemma-1.5-4B (L1, 4, …, 32, 34) | image tokens | −0.31 to −0.01 | 0.54 | −0.15 to −0.01 |
| | question tokens | −0.33 to −0.02 | 0.48 | −0.15 to +0.15 |
| | last prompt token | −0.23 to +0.17 | 0.83 | −0.11 to +0.20 |

### Reading
- No layer or pooling reaches significance on either backbone (smallest uncorrected p 0.048, MedGemma question L12).
  RQ4's representation test is negative, consistent with E3's negative covariate test (RQ3): neither task metadata
  nor base-model input similarity predicts which off-task transfer helps.
- Qwen's last-token ρ is weakly positive in middle layers (+0.21 to +0.27), MedGemma's image-token ρ weakly negative
  (−0.25 to −0.31 at L12–L20); neither survives correction and they disagree in sign.

### Check: features standardized per dimension (2026-11-15)
- In MedGemma's middle layers 5 of 2,560 dimensions carry 70–82% of the probe variance (Qwen: ≤ 21%, mostly ≤ 10%), so
  raw cosine/CKA there mostly measures those dimensions (raw Shared-vs-base CKA 0.04–0.17 at L8–L16, seed ceiling 0.46).
- Rerun with every dimension z-scored by the base model's probe statistics (`--standardize`, box `e56042a4`, 15 Nov
  03:12–03:28 → `phase_b/e7_results_std/representations.json`). Main test still negative on both backbones: centroid ρ
  Qwen +0.01 to +0.16, MedGemma −0.01 to +0.17, smallest Holm p 0.69. MedGemma's weak negative raw ρ (image L12–L20)
  was an outlier-dimension artifact.
- Standardized shift from base is sensible: last-token CKA(Shared, base) Qwen 0.95 → 0.58–0.76, MedGemma 0.98 → 0.71–0.82
  over depth, seed ceilings 0.88–0.98. **Use the standardized view for MedGemma's descriptive measures.**

### Hydra heads (2026-11-15)
- `code/analysis/hydra_heads.py` (Veronika's; reads the saved factors, CPU) on all 12 Hydra runs →
  `/data/ugrip2026/journal/phase_b/e7_heads/` (per-target 6×6 cosine; `--head_map_out` maps also written). Cosine of the
  task heads (B_t for Hydra-T, A_t for Hydra-R), mean over the 238 targets (MedGemma) or all targets (Qwen) and 3 seeds:

| Arm | mean off-diagonal cosine (per seed) | most similar pair | least similar | smallest head norm |
|---|---|---|---|---|
| Qwen Hydra-T (B) | 0.025 (0.026 / 0.026 / 0.023) | cls–ml 0.063 | count–report 0.013 | counting (0.71 vs 1.7–2.1) |
| Qwen Hydra-R (A) | 0.009 (0.009 / 0.009 / 0.010) | cls–ml 0.020 | count–report 0.004 | counting |
| MedGemma Hydra-T (B) | 0.032 (0.033 / 0.032 / 0.032) | cls–ml 0.057 | det–report 0.013 | counting (0.66 vs 1.5–1.9) |
| MedGemma Hydra-R (A) | 0.011 (0.011 / 0.011 / 0.011) | det–reg 0.020 | count–report 0.005 | counting |

- Task heads end up nearly orthogonal in every arm, seed and backbone; cls–ml (the two label-picking tasks) is the most
  aligned pair in 3 of 4 arms; the counting head (smallest task, 100 holdout / fewest training rows) moves least.
- Hydra-T's B heads start at zero, so their low cosine is learned; Hydra-R's A heads start random and near-orthogonal,
  so their low cosine is partly the initialization. Raw-factor cosine is basis-dependent: descriptive only.

## E8 — RQ4 synthesis: what predicts where specializing beats Shared, both backbones (2026-11-15)

Box `930bbb5`; `phase_b/e8_results/{synthesis.json,cells.csv}` (collected runs `phase_b/e8_collected/`, log `phase_b/e8.log`).
Outcome per cell (19 task/dataset cells): mean over 3 seeds of PGF(specialized) − PGF(Shared), ZS-prompt anchor
(ZS-bare as sensitivity). Specialized arms: TaskLoRA, Hydra-T, Hydra-R. Predictors: training counts, output type and
modality (one-hot), E3 transferability/malleability, E7 input/adapter similarity per layer (E7 main run, raw features),
ZS difficulty. Spearman ρ, cell permutation, B=10,000, Holm across predictors within each outcome. First attempt failed
on the one unscored CRIMSON report in Qwen ZS-bare; fixed as in E3 (`930bbb5`; failed attempt kept as
`e8_collected_failed_zsbare_20261115`).

| Backbone | Arm | Mean PGF benefit (ZS-prompt) | Cells > 0 | Tests (undefined) | Min Holm p (prompt / bare) |
|---|---|---|---|---|---|
| Qwen | TaskLoRA | −0.005 | 9/19 | 51 (0) | 1.00 / 1.00 |
| Qwen | Hydra-T | +0.010 | 13/19 | 51 (0) | 1.00 / 1.00 |
| Qwen | Hydra-R | +0.003 | 9/19 | 51 (0) | 1.00 / 1.00 |
| MedGemma | TaskLoRA | +0.011 | 11/19 | 77 (4) | 0.18 / 1.00 |
| MedGemma | Hydra-T | +0.014 | 13/19 | 77 (4) | 0.43 / 0.34 |
| MedGemma | Hydra-R | +0.000 | 11/19 | 77 (4) | 1.00 / 1.00 |

### Reading
- **Negative:** no predictor survives Holm correction for any arm, backbone or anchor. Nothing measured predicts
  which cells gain from specializing.
- Benefits are small (mean |PGF benefit| ≤ 0.015). The largest single cells are MedGemma detection on BUS-UCLM/BUSI
  (+0.06 to +0.10 for all three arms) and Qwen Hydra-T periapical multi-label (+0.09).
- Uncorrected p < 0.05 hits are scattered and don't repeat across backbones, except "box output" on MedGemma
  (ρ ≈ +0.5 in all six outcomes), which is the detection cells above. Qwen X-ray (ρ 0.48) appears in 3 of 6 outcomes.
- Caveat: the E7 similarity predictors come from raw features. For MedGemma, the standardized E7 run is the better
  measure (LEDGER E7 check); not rerun with it here.

### Check: E8 with standardized E7 features (2026-11-15)
- `rq4_synthesis.py analyze --representations phase_b/e7_results_std/representations.json` → `phase_b/e8_results_std/`.
  Same conclusion: no predictor survives Holm (smallest 0.18, MedGemma TaskLoRA/ZS-prompt; every Qwen outcome 1.00).
  On MedGemma the 4 E3 predictors (transferability/malleability ±) are undefined because counting is excluded as an E3
  target. Paper uses both runs ("raw or standardized: same result").

----

## E9-lite — supervision scaling, Qwen3.5-9B, cls target, seed 1337, holdout (2026-11-17)

**RQ2 pilot (PLAN changelog 14 Nov): Shared (mixture with cls shrunk to p%, other tasks 100%) vs TaskLoRA-cls
(cls p% only), p ∈ {1, 5, 10, 25, 50}%, subset draw 20261009 (nested, stratified by dataset and label), training
seed 1337. 100% = the E1 s1337 runs. One seed: descriptive only. Seeds 2026 and 42 launched 18 Nov.**

- Runs: `j26_q35_{shared,taskcls}_cls{1,5,10,25,50}_768_s1337`; configs `code/configs/e9/`; decks
  `/data/ugrip2026/journal/data_e9` (`code/data/build_e9_decks.sh`: every row byte-identical to a W1
  `j26_train_mix` row; md5s in its `MANIFEST.json`). Training 14 Nov 09:10 → 17 Nov 11:55, 4 jobs at once on GPU 1,
  all rc=0, 3 epochs. Commits: Shared 1/10/25/50% `158c806`; Shared 5% and all TaskLoRA `930bbb5` (the box repo moved
  mid-chain; no change under `code/train`, `code/methods`, `code/configs`, `code/data` or the eval/scorer files).
- Smoke (14 Nov): both arm types passed (`runs/smoke/j26_q35_{shared,taskcls}_cls1_768_s1337_smoke_taha_*`).
- Eval: `eval_arm.py --only_task classification` on the holdout cls slice (1,689 questions; greedy, EOS pinned,
  512 tokens, as E1), jobs `code/eval/jobs/e9/eval_chain_{1..4}.txt`; Shared runs evaluated on cls only.
  Scorer `score_single.py` → `<run_dir>/metrics_holdout_noreport.json`. Malformed 0/1689 in every run.
- Deviations from PLAN §1: none (only the deck changes). Scope cut from PLAN §E9: see changelog 14 Nov.

### Classification task macro (BA ↑; mean of 10 per-dataset BAs)

| cls data | cls train questions | Shared | TaskLoRA | TaskLoRA − Shared |
|---|---:|---:|---:|---:|
| 1% | 184 | 0.514 | 0.477 | −0.038 |
| 5% | 769 | 0.563 | 0.540 | −0.023 |
| 10% | 1,532 | 0.583 | 0.578 | −0.005 |
| 25% | 3,814 | 0.652 | 0.637 | −0.015 |
| 50% | 7,600 | 0.688 | 0.702 | +0.014 |
| 100% (E1) | 15,218 | 0.729 | 0.733 | +0.005 |

### Per dataset (BA, Shared / TaskLoRA)

| Dataset | n | 1% | 5% | 10% | 25% | 50% | 100% (E1) |
|---|---:|---|---|---|---|---|---|
| BUS-UCLM | 52 | 0.768 / 0.727 | 0.824 / 0.757 | 0.797 / 0.805 | 0.849 / 0.827 | 0.849 / 0.875 | 0.897 / 0.897 |
| BUSI | 62 | 0.797 / 0.702 | 0.759 / 0.790 | 0.829 / 0.714 | 0.878 / 0.845 | 0.894 / 0.848 | 0.838 / 0.876 |
| CMMD | 357 | 0.370 / 0.356 | 0.402 / 0.422 | 0.470 / 0.424 | 0.479 / 0.435 | 0.456 / 0.456 | 0.473 / 0.497 |
| bcn20000 | 286 | 0.346 / 0.393 | 0.395 / 0.387 | 0.407 / 0.452 | 0.375 / 0.419 | 0.543 / 0.515 | 0.529 / 0.594 |
| bone_marrow | 72 | 0.224 / 0.190 | 0.248 / 0.210 | 0.381 / 0.335 | 0.506 / 0.534 | 0.581 / 0.615 | 0.720 / 0.761 |
| dental | 420 | 0.352 / 0.325 | 0.499 / 0.450 | 0.494 / 0.507 | 0.574 / 0.564 | 0.625 / 0.616 | 0.667 / 0.648 |
| fundus | 138 | 0.471 / 0.421 | 0.512 / 0.452 | 0.510 / 0.489 | 0.614 / 0.623 | 0.694 / 0.655 | 0.670 / 0.666 |
| iugc | 131 | 0.845 / 0.660 | 0.798 / 0.833 | 0.786 / 0.827 | 0.896 / 0.850 | 0.890 / 0.879 | 0.948 / 0.942 |
| neojaundice | 60 | 0.493 / 0.493 | 0.630 / 0.478 | 0.598 / 0.641 | 0.774 / 0.709 | 0.756 / 0.749 | 0.810 / 0.719 |
| retino | 111 | 0.476 / 0.501 | 0.560 / 0.621 | 0.561 / 0.591 | 0.576 / 0.566 | 0.596 / 0.815 | 0.736 / 0.736 |

Datasets where Shared > TaskLoRA: 1% 7/10, 5% 6/10, 10% 4/10, 25% 7/10, 50% 7/10, 100% (E1) 5/10.

### Reading (one seed; no CIs)
- The ordering flips with data size: Shared is ahead at 1% (+0.038 BA) and 5% (+0.023), the arms are level at
  10% (+0.005) and close at 25% (+0.015), and TaskLoRA is ahead at 50% (+0.014) and 100% (+0.005, s1337). This is the
  direction RQ2 predicts (sharing helps most when the target task has little data); the crossover lies between 25%
  and 50% on this seed.
- The 1% and 5% gaps are 2–4× the E1 seed SD at 100% (~0.01); the 10–50% gaps are within it. Seed SD at low data is
  unknown until seeds 2026/42 finish.
- At 50% TaskLoRA's macro lead comes from one dataset: retino 0.818 vs 0.604 (n = 111); Shared is ahead on 7 of the
  10 datasets there. The 50% point is not a clean TaskLoRA win.
- Both arms gain steadily with data; neither saturates by 50% (Shared 0.688, TaskLoRA 0.702 vs 0.729 / 0.733 at 100%).
- No malformed answers at any fraction, so low-data scores reflect wrong labels, not format failures.

## E3 format analysis — format lock-in in off-task transfer, both backbones, 3 seeds, holdout (2026-11-23)

**Tests the Discussion's untested claim that negative off-task transfer comes from adapters answering in their own
task's format. Plan: `results/format_analysis_plan.md` (Tausifa's protocol, revised; rules fixed before any
lock-in rate, `656b644`).**

- Inputs: the E3 predictions (`phase_b/e3_collected`, `e3_results/transfer.json`), E8 joint arms, Shared r48 run
  dirs. CPU only. Code `62b6fe2` (`code/analysis/format_lockin.py`, `format_rules.py`, 34 unit tests), run on the box
  from a frozen copy `phase_b/format_analysis/code_62b6fe2/` → `phase_b/format_analysis/out_B10000/`
  (`format_tables.md` has every cell; `format_analysis.json`; panels per backbone). B = 10,000, two-level (seeds,
  then questions stratified by dataset), seed 20261013. μ recomputed here equals Table VI in every cell.
- F0 checks: every TaskLoRA answers its own task in its own format 100% (one MedGemma regression answer,
  `A(105.92`); no real rule conflicts among 80k answers per backbone.
- Deviations (plan, "Changes after the first lock-in run"): label fragments ("Endodontic Treatment") count as their
  option; F7c canonicalises the base as well. Holm across the μ targets of each source (`format_paper.py`,
  `format_analysis_holm.json`): every result whose CI excludes 0 stays significant except the Qwen report→reg
  format effect (−0.031; Holm p 0.069). Conclusions unchanged.

**Definitions.** Lock-in = answer in the source task's format and not valid for the target, or a label from the
source's training vocabulary. F7a = μ after replacing locked-in answers with the ZS-prompt base's answers (by
construction it can only pull μ toward the base); effect = μ_F7a − μ. F3 = μ on the questions answered in the target
format only (cells with < 30 questions or < 20% kept: descriptive).

### Lock-in rate (%, mean over seeds; row = adapter, column = target)

| Qwen | cls | ml | det | count | reg | report | | MedGemma | cls | ml | det | count | reg | report |
|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| cls | — | 0 | 0 | 0 | 0 | 33 | | cls | — | 0 | 21 | 0 | 4 | 3 |
| ml | 0 | — | 0 | 0 | 0 | 35 | | ml | 2 | — | 1 | 0 | 4 | 66 |
| det | 77 | 46 | — | 98 | 100 | 14 | | det | 94 | 68 | — | 96 | 97 | 90 |
| count | 0 | 0 | 0 | — | 0 | 0 | | count | 7 | 40 | 7 | — | 0 | 59 |
| reg | 0 | 0 | 23 | 56 | — | 2 | | reg | 83 | 93 | 51 | 72 | — | 98 |
| report | 8 | 11 | 28 | 33 | 37 | — | | report | 83 | 98 | 100 | 100 | 96 | — |

### Negative cells: μ, after the swap (F7a), and on well-formed answers (F3) [95% CI]

| Backbone | Cell | Lock-in | μ (E3) | μ F7a | F3 μ_valid (kept) | Reading |
|---|---|---:|---:|---:|---|---|
| Qwen | det→count | 98% | −1.194 | +0.000 | — (0%) | format |
| Qwen | det→reg | 100% | −1.279 | +0.000 | — (0%) | format |
| Qwen | det→ml | 46% | −0.401 | −0.091 [−0.144, −0.033] | −0.033 [−0.088, +0.016] (42%) | mostly format |
| Qwen | det→cls | 77% | −0.123 | −0.018 [−0.044, −0.003] | desc. (12%) | mostly format |
| Qwen | report→ml | 11% | −0.128 | −0.072 [−0.125, −0.022] | −0.077 [−0.143, −0.022] (89%) | part format, part content |
| Qwen | **cls→ml** | **0%** | **−0.397** | −0.397 | **−0.381 [−0.450, −0.316]** (97%) | **not format: one label where 2+ needed (38% of answers; 44% of questions have 2+)** |
| Qwen | reg→ml | 0% | −0.076 | −0.076 | −0.075 [−0.140, −0.021] (100%) | content (under-cardinality 24%) |
| Qwen | ml→cls | 0% | −0.033 | −0.033 | −0.032 [−0.055, −0.012] (99%) | content (μ +0.026 once both sides use letters) |
| MedGemma | det→reg | 97% | −1.419 | −0.009 [−0.022, +0.000] | desc. (3%) | format |
| MedGemma | report→ml | 98% | −0.275 | −0.002 [−0.005, −0.000] | desc. (2%) | format |
| MedGemma | reg→ml | 93% | −0.260 | −0.004 [−0.017, +0.006] | desc. (2%) | format |
| MedGemma | det→ml | 68% | −0.247 | −0.059 [−0.169, +0.003] | desc. (0%) | format |
| MedGemma | **cls→reg** | **4%** | **−0.440** | −0.436 [−0.894, −0.118] | −0.526 [−1.227, −0.036] (68%) | **not format: wrong numbers in the right format** |

- **Positive transfer survives on well-formed answers:** Qwen cls/count/reg/report→reg and reg→cls, MedGemma
  count→cls/reg keep the same μ (CIs exclude 0). Qwen reg→count: lock-in (decimal counts, 56%) *helps*; swapping them
  for the base's answers lowers μ (+0.123 → +0.027).
- **Joint training (F6):** Shared, Hydra-T, Hydra-R and Shared r48 answer 0.0% of questions outside their own task's
  format, every task, both backbones, 3 seeds.
- **Base model (side finding):** ZS-prompt answers 44.6% (Qwen) and 53.3% (MedGemma) of classification questions as
  "C. Moderate NPDR", which the scorer marks wrong; part of the zero-shot classification gap is format.
- F2 (descriptive): Spearman ρ(lock-in, μ) = −0.38 (Qwen, 20 pairs), −0.66 (MedGemma, 15 pairs).
- F4 gate (|effect| ≥ 0.05 in ≥ 3 cells): passed on both (Qwen 7/20, MedGemma 5/15). In the near-total lock-in
  cells this is automatic (swap ≈ base). F4 not run yet: Taha's decision.

### Reading
- On MedGemma, and for the detection adapter on both backbones, negative transfer is mainly format lock-in.
- Qwen's largest non-detection negative cell (cls→ml, −0.40) is not format: the classification adapter answers in a
  valid format but keeps its one-label habit (under-cardinality). MedGemma cls→reg (−0.44) is wrong content.
- Joint training removes format errors entirely, consistent with the question telling the model the format.
- Wording for the paper: "associated with"; F7 is a counterfactual approximation, not a forced re-evaluation.
- Paper tables generated by `code/analysis/format_paper.py` (Tables "lockin", "lockin_cells", appendix "formats").
  Note: Table VI prints MedGemma report→ml as −0.28; the value is −0.2749 (−0.27).
