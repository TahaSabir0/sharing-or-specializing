# Status

**Updated 2026-11-12.** Plan: `PLAN.md` v2 (team and rules: §7). Numbers: `results/LEDGER.md`.
Runs: `results/RUNS.csv`. New teammates: `KICKSTART.md`.

## Where we are
- **All 12 Hydra training runs done (rc=0):** Qwen Reversed Hydra 9 Nov ~18:10, MedGemma Hydra-T 10 Nov ~16:10,
  MedGemma Reversed Hydra 11 Nov 14:10. GPU idle 11 Nov 14:10 → 12 Nov 14:03.
  **Remaining 9 Hydra holdout evals + CRIMSON launched 12 Nov 14:03** (box `997e0f5`; harness plug-in mode, oracle
  task route, batch 1). One chain per seed, each Qwen hydraR → MedGemma hydraT → MedGemma hydraR:
  job files `code/eval/jobs/hydra/{eval,crimson}_hydra_rest_s<seed>.txt`; eval PIDs 2435276/78/80, CRIMSON PIDs
  2435277/79/81 (each CRIMSON job waits for its eval's rc=0); logs `runs/evalchain_hydra_rest_s<seed>.log`,
  `runs/crimsonchain_hydra_rest_s<seed>.log`. Rough ETA: all done ~13 Nov early morning (box time).
  GPU access ends 13 Nov: E4/E5 Hydra rerun, E6, E7, E9 have no GPU time scheduled.
- **Alex follow-up (6 Nov):** E3 collection now validates and consumes Taha's chain-runner
  outputs, including Qwen's legacy file, then combines only the off-diagonal slice with the
  official E1/E2 diagonal. Output structure is now a graded Jaccard feature similarity.
  The E6 package delta is `gliclass==0.1.20` in a separate environment; E7 adds no package.
  E7 probe generation now targets `data/e7_probe_ids.json` and enforces 562 IDs. Alex generated
  the frozen manifest from the canonical Blackwell holdout and committed it as `aa658d5`:
  562 unique IDs, seed 20261019, up to 30 per task/dataset cell, holdout MD5
  `fe43237b3412e3ef0f2f7c778395ba58`.
- **E0 and the E1 plain-LoRA arms on Qwen are complete:** Shared, TaskLoRA and Shared r48 × 3 seeds,
  all trained, evaluated on the holdout and CRIMSON-scored (LEDGER, 2026-10-23 section).
- **Result so far (mean over 3 seeds, no CIs yet):** Shared vs TaskLoRA is close to a tie. Shared is
  slightly ahead on cls, detection and counting; TaskLoRA on regression (boneresorption) and reports
  (CRIMSON, omission); multi-label tied. Every gap is ~1–1.5 seed SDs. Shared r48 (6× parameters) is no
  better than Shared on any task.
- **E0 and E2 on MedGemma are complete (26 Oct):** same 3 arms × 3 seeds, evaluated and CRIMSON-scored
  (LEDGER, 2026-10-26 section). Port (W9) merged 23 Oct. TaskLoRA leans ahead of Shared on MedGemma (detection
  +0.06 on every seed); Shared r48 again no better. MedGemma trails Qwen on detection (0.47 vs 0.90: ultrasound
  boxes poorly placed along x; coordinate convention verified), counting and multi-label; ties elsewhere.
- **Hydra-T / Reversed Hydra (Veronika): reviewed and merged 4 Nov.** All four smoke gates passed 3 Nov (Qwen and
  MedGemma, both modes; `results/smoke/hydra_integrated_20261103.md`, `code/HYDRA_RUNBOOK.md`).
  **12 full runs launched 4 Nov 10:01** (box `d7a16cd`): `chain_hydra{1,2,3}.log` (PIDs 3751748/49/51), chain i =
  seed i: Qwen T, Qwen R, MedGemma T, MedGemma R (`code/eval/jobs/hydra/`). **Qwen Hydra-T done 7 Nov ~03:40**
  (3/3 rc=0); Qwen Reversed Hydra running, ETA ~9 Nov 14:00–15:00; MedGemma T+R after, training done ~11 Nov
  afternoon. Holdout evals not queued yet: Qwen Hydra needs `eval_harness.py`, which passed parity 7 Nov (300/300 raw
  outputs identical to `eval_arm.py` on Shared s1337; LEDGER "Setup findings"). **Qwen Hydra-T holdout evals + CRIMSON done; in LEDGER ("E1 — Qwen3.5-9B Hydra-T") and paper Table IV, 8 Nov.** Evals were launched 7 Nov 12:11 (box `38d183e`;
  harness plug-in mode, oracle task route, batch 1; job files `code/eval/jobs/hydra/{eval,crimson}_q35_hydraT_s<seed>.txt`;
  eval PIDs 4171979/81/83, CRIMSON PIDs 4171980/82/84 wait for their eval's rc=0; logs
  `runs/evalchain_hydraT_q35_s<seed>.log`, `runs/crimsonchain_hydraT_q35_s<seed>.log`). ETA evals ~18:00–20:00,
  CRIMSON ~1–1.5 h after each; slows Reversed Hydra training while they run.
- **E3 inference + CRIMSON complete** (30/30 CRIMSON; last 14 MedGemma jobs ran beside Hydra 5 Nov 11:37–22:27,
  logs `runs/crimsonchain_e3resume_s<seed>.log`; Hydra slipped ~4 h). Alex's collect fix merged 7 Nov
  (`f32d64d`) plus two fixes at review: `adapter_identity` deck key (`ce56acd`) and unscored CRIMSON reports
  (`c447ac3`, `d119d5f`: allowed only in zero-shot baselines and non-report adapters' report cells, left out of
  the mean as in LEDGER, counted per cell). **E3 analysis done 8 Nov 12:22** (first run crashed 7 Nov on an overflowed
  count, fixed `cc8f561`; covariate test on μ with MedGemma counting excluded, `08b5905`): LEDGER 2026-11-08 E3
  section, paper Table VII. Off-task transfer mostly absent or negative; detection adapter hurts everything; no
  covariate predicts transfer (RQ3 negative).
- **E4/E5 ready to launch (26 Oct):** Alex's `alex/modality-decks` reviewed, reworked and merged
  (`code/E4_E5_RUNBOOK.md`). 51 group decks in `/data/ugrip2026/journal/data_e4e5`; job files in
  `code/eval/jobs/e4e5/` (3 training chains × 28 runs; 42 evals per backbone; 6 CRIMSON). 84 runs, not 102:
  Clinical/Derm/Mammo hold one dataset each, so E5 reuses those E4 adapters (PLAN changelog 26 Oct).
  **Training done 31 Oct 00:47:** all 84 runs rc=0 and pass the protocol check (3 epochs, deck hash,
  P8). No MedGemma smoke (deviation, PLAN changelog). **Holdout evals done 2 Nov 03:31** (84/84 rc=0) and
  **CRIMSON done 2 Nov 06:17** (6/6 rc=0). **Verified and analysed 6 Nov** (plain-LoRA reference arms;
  `/data/ugrip2026/journal/e4e5_results_plain_lora_20261106/`): LEDGER 2026-11-07 section, paper §E4/E5.
  Modality adapters ≈ Shared ≈ TaskLoRA; dataset adapters lower cls (sig. vs Shared on both backbones). Rerun
  with the Hydra arms once their evals exist. 84 runs not yet in RUNS.csv.
- **E3/E6/E7/E8 code (Alex) merged 3 Nov** (`de4b21f`). At review E3 inference moved to the usual
  chain runners with each backbone's baseline evaluator (Qwen `eval_arm.py`, not the harness);
  job files `code/eval/jobs/e3/`. **E3 launched 3 Nov 10:52** (box `fcbe2fa`; eval slot PIDs
  3630186/88/90, CRIMSON 3630187/89/91; logs `runs/evalchain_e3_{q35,mg15}_s<seed>.log`,
  `runs/crimsonchain_e3_s<seed>.log`). ETA: Qwen ~3 Nov 20:00, MedGemma ~4 Nov 07:00, CRIMSON ~4 Nov 14:00. Asked Alex
  (3 Nov) to adapt `plan_e3.py collect` to those outputs, grade the output-structure covariate
  (6 distinct types make "same structure" 0 for every pair) and freeze the E7 probe IDs / list
  the E6 router packages.
- Scorer fix merged (`score_single.py` dropped a score equal to the question count): rescoring all 52
  existing holdout prediction files changed no number in the LEDGER.

## Phase A tracker (5 arms × 3 seeds × 2 backbones = 30 training runs)

| Arm | Qwen s1337 | Qwen s2026 | Qwen s42 | MedGemma (×3) |
|---|---|---|---|---|
| Shared | done | done | done | done ×3 |
| TaskLoRA (6 adapters) | done | done | done | done ×3 |
| Shared r48 | done | done | done | done ×3 |
| Hydra-T | smoke passed; full run pending | configs ready | configs ready | smoke passed; full runs pending |
| Reversed Hydra | smoke passed; full run pending | configs ready | configs ready | smoke passed; full runs pending |

"done" = trained, holdout-evaluated, CRIMSON-scored. Zero-shot (E0): done on both backbones (bare + prompt,
incl. CRIMSON).

## Next steps (Taha)
1. Hydra (Veronika): the only Phase A work left (Qwen ~320 slot-h; MedGemma 6 runs ≈ 140 slot-h). Decide
   whether to start Hydra-independent Phase B (E3/E4/E5) while waiting.
2. Review `alex/modality-decks`. Update PLAN §5.2: MedGemma full-mixture run ≈ 21–25 h wall with 3 jobs
   (~3× the planned 8 slot-h).
3. Stats (`bootstrap.py`: CIs and paired contrasts for E1): deferred for now.
4. E1 extras (PLAN E1 "also report"): per-dataset deltas by modality, output-format error table, μ.
5. **`paper/main.tex` restructured to v2 (28 Oct, branch `taha/paper-v2`).** Tausifa answered
   (28 Oct): rewrite the body to v2 and keep the dropped studies (Soft-LoRA, latent experts/router,
   location, practical factors) in a supplementary "if time permits" section. Done: RQ1–RQ4 with the
   three wording clarifications, the five arms, setup, E0–E9 methods, E0/E1/E2 results drafted (3-seed
   tables, Δ_spec, descriptive text; Hydra columns and CIs marked `\todo`). Compiles (checked with the
   stock IEEEtran class; `ieeecolor.cls`/`generic.sty` are not in the repo). Next: copy it to Overleaf
   and check it against Tausifa's Overleaf version (her RQ wording may differ); owners draft their
   sections (target 7 Nov).

## Last runs (MedGemma E0/E2, 23–26 Oct, box time +04, repo `fd0a803`; logs in `/data/ugrip2026/journal/runs/`)

Nothing running. All chains below ended rc=0.

| Chain (PID) | Runs, in order | State / ETA |
|---|---|---|
| `evalchain_mgE0` (1528780) | ZS-prompt → ZS-bare, full holdout, batch 1 | **done** 24 Oct 04:54 (4.7 h + 11.4 h: untrained outputs are long) |
| `evalchain_mgE2` (1709582, from 25 Oct 10:11, `52368b1`) | `code/eval/jobs/v2_e2_mg15_holdout.txt`: holdout evals of the finished runs, then each later run as its chain ends | **done** 26 Oct 13:26 (24 evals) |
| `crimsonchain_mg15` (1709583) | `code/eval/jobs/v2_crimson_mg15.txt`: ZS-prompt, ZS-bare, then each report-bearing E2 eval as it ends | **done** 26 Oct 13:54 (11 jobs) |
| `chain_mgH` (1530566) | Shared s1337 → 6 TaskLoRAs s1337 → r48 s2026 | **done** 26 Oct 11:00 |
| `chain_mgI` (1530568) | 6 TaskLoRAs s2026 (cls, report, ml, det, reg, count) → r48 s1337 → Shared s42 | **done** 26 Oct 11:16 |
| `chain_mgJ` (1530570) | r48 s42 → Shared s2026 → 6 TaskLoRAs s42 | **done** 26 Oct 11:07 |

First launch at 12:45 aborted after 2 min: the Shared/r48 configs were cp1252, not UTF-8 (fixed in `fd0a803`);
logs `chain_mg{H,I,J}_aborted_cp1252.log`, partial run dirs removed. Finished Qwen chains: `chain_{E,F,G}.log`,
`evalchain_E1chain{1,2}.log`, `crimsonchain_{1,2}.log`.

Check: `ssh blackwell 'nvidia-smi -i 1; cat /data/ugrip2026/journal/runs/smoke/.holder'`.

## Things to remember
- Every compared run is trained and evaluated on the box (ws and the box are not
  bit-reproducible; PLAN.md §1.12; numbers in LEDGER "Setup findings").
- `/data/ugrip2026/journal/repo` updates from a git bundle, not GitHub (`ops/cluster-config.md`, top).
  Relay and check the commit before launching anything from it.
- Never update that repo while a chain will start a new run from it with changed code: chains read the
  code at each run's start (RUNS.csv notes the harmless cases).
- Thread caps are mandatory on the box (`run_chain.sh`, `smoke.sh` and the eval/CRIMSON chains set them).
- 3 jobs at once = 2.0× one job, card ~90% busy; a 4th job mostly slows the others.
- Measured: holdout eval ≈ 75 min per Shared/r48 run, ~1.5 h for all 6 TaskLoRAs; CRIMSON ≈ 45 min per
  run (781 reports, batch 1).

## For Tausifa
- Dev is inside the training pool, so v2 makes no selection decisions (PLAN.md §1.7).
- v2 decisions N1–N8 (PLAN.md §6) and the team split (§7.1).
- Qwen E0/E1 complete: LEDGER 2026-10-23 section.
