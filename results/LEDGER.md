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
