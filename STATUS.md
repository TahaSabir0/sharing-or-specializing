# Status

**Updated 2026-11-23.** Plan: `PLAN.md` v2 (team and rules: §7). Numbers: `results/LEDGER.md`.
Runs: `results/RUNS.csv`. New teammates: `KICKSTART.md`.

## Where we are
- **F4 v2 forced decoding running (Tausifa's 25 Nov spec; plan `results/format_analysis_plan.md` §F4 v2, PLAN
  changelog 25 Nov).** All 14 negative cells of Table VIII; source TaskLoRA + ZS-prompt + Shared, 3 seeds, holdout.
  Branch `forced-decoding` (**not merged to main yet**); runs launched 26 Nov 05:21 box time from
  `people/taha/repo` at `2742679` (3 chains, PIDs 2387239/40/41, logs `runs/evalchain_f4_s{1337,2026,42}.log`,
  outputs `runs/f4/`). Smoke passed (`results/smoke/f4_forced_20261126.md`); analysis script checked on a mock
  (reproduces Table VIII μ exactly). ETA ~26 Nov 11:00. Then: `git pull` the branch in `people/taha/repo` and
  run `python code/analysis/forced_mu.py --out /data/ugrip2026/journal/phase_b/format_analysis/f4_B10000 --B 10000`.
- **E3 format analysis done 23 Nov** (Tausifa's protocol after the 21 Nov meeting, revised; plan
  `results/format_analysis_plan.md`; branch `format-analysis`, code `62b6fe2`; CPU on the box,
  `phase_b/format_analysis/out_B10000/`; LEDGER 2026-11-23). Negative transfer is mainly format lock-in for the
  detection adapter (both backbones) and on MedGemma; Qwen cls→ml (−0.40) is not format (one label where several
  are needed); joint arms 0% wrong format. **Open: F4 (forced-format GPU re-run, gate passed) — Taha's call; paper
  Discussion sentence to update.**
- **E1/E2 paired bootstrap done 18 Nov 09:57** (box `5de7899`, `code/analysis/e1_e2_bootstrap.sh`): no arm differs
  from another on any task on either backbone after Holm (smallest 0.076); LEDGER 2026-11-18. **E4 vs Hydra** done
  16 Nov: no task differs (LEDGER E4/E5). Paper repo (`9e2fda6`): E1/E2 text now descriptive with CIs, contrast table
  in the appendix, per-dataset appendix table, E4 Hydra columns, E7/E8 sections, abstract/discussion/conclusion
  drafted. Left for the paper: E9 (E9-lite seed 1337 done 17 Nov; seeds 42/2026 launched 18 Nov), related work.
- **E6 dropped 18 Nov** (Taha; PLAN changelog): no specialized arm beats Shared even with the oracle route, so
  predicted routing cannot change the answer. Paper `c2b87c1`: E6 removed, oracle route a limitation. **E9-lite
  seeds 42/2026 are the last experiment.**
- **E7 analysis done 14 Nov 10:52** (box `e89ac0b`): main test negative on both backbones (input similarity does
  not predict off-task transfer; LEDGER 2026-11-14 E7). First run was flawed (raw C incl. own-task cells; MedGemma
  undefined) and fixed (PLAN changelog 14 Nov). 15 Nov: standardized-feature check (same negative result; MedGemma
  middle-layer outlier dimensions explained) and Hydra head similarity on all 12 runs (heads nearly orthogonal), both in
  LEDGER E7.
- **E8 done 15 Nov 04:57** (box `930bbb5`): negative; no predictor of where specializing beats Shared survives
  Holm on either backbone (LEDGER E8). Left: E7 and E8 into the paper repo.
- **E9-lite seed 1337 done 17 Nov 12:09** (10 runs + 10 cls evals rc=0; scored; LEDGER 2026-11-17 E9-lite; RUNS.csv).
  Qwen cls, Shared vs TaskLoRA-cls at 1/5/10/25/50%: TaskLoRA − Shared = −0.038 / −0.023 / −0.005 / −0.015 / +0.014
  (+0.005 at 100%). Shared leads at low data, crossover between 25% and 50% (one seed; descriptive).
- **E9-lite seeds 2026 and 42 launched 18 Nov 08:49** (box `aa02845`; GPU end date now open, PLAN changelog 17 Nov):
  20 runs, 4 training chains (`code/eval/jobs/e9/train_chain_{5..8}.txt`, PIDs 3865484/85/86/88, logs
  `runs/chain_e9_{5..8}.log`, JOURNAL_DATA=data_e9) + 4 eval chains (cls slice; PIDs 3865490/91/94/95, logs
  `runs/evalchain_e9_{5..8}.log`). Low fractions first. **24 Nov 04:00: 19/20 trained, evaluated and scored**
  (score_single → `metrics_holdout_noreport.json`, all rc=0, malformed none). `taskcls_cls50_s42` crashed 23 Nov
  16:19 at step 599/2850 (dataloader `unable to allocate shared memory(shm)`, transient; /dev/shm 11% used later)
  and hung on GPU 1 for ~11.5 h; killed it + chain 5 + eval chain 5, crashed dir kept as
  `runs/j26_q35_taskcls_cls50_768_s42.crashed_20261123`. **Rerun launched 24 Nov 03:58** (same config, `aa02845`,
  JOURNAL_DATA=data_e9): log `runs/chain_e9_5b.log`; eval queued behind it (`runs/eval_chain_e9_5b.txt`, log
  `runs/evalchain_e9_5b.log`). **Rerun done 24 Nov 14:48 rc=0, eval 15:15 rc=0, scored: cls macro 0.678.
  E9-lite all 30 runs scored.** 3-seed means Shared/TaskLoRA: 1% .502/.472, 5% .563/.539, 10% .581/.586,
  25% .646/.639, 50% .686/.697. Shared ahead at 1/5% on every seed; 10–50% within seed noise. Next: LEDGER 3-seed
  table + paired bootstrap. **Done 25 Nov (LEDGER 2026-11-25 E9-lite 3 seeds):** bootstrap: 1%/5% Shared ahead
  on every seed, p 0.031/0.048 but Holm (6 fractions) 0.19/0.24; 10–100% no difference. Epoch-checkpoint check
  (24 evals, PLAN changelog 24 Nov): every run improves every epoch, TaskLoRA as fast as Shared → no overtraining;
  the 50%→100% flip is seed noise. **E9 complete and in the paper (paper repo, 25 Nov): Tables XI–XII, methods, results, abstract, discussion, conclusion; the planned mixed-effects interaction model was not fit (one target, two arms: per-fraction bootstrap contrasts instead, stated in the paper).**
- **Paper, 25 Nov (Overleaf now synced to the paper repo; `git pull` before editing):** related work written
  (5 subsections, 11 new bib entries: HydraLoRA, LoRAMoE, MoLE, MedQwen, Hexapus, adapters, Caruana, Standley,
  PCGrad, Fifty, Shazeer MoE; check them); appendix Tables XIII–XIV (all training/decoding settings, adapted layers).
  **Waiting on the team's data discussion:** appendix transfer matrices, full E7, malformed/truncation/omission
  tables; related-work framing. **Taha's call:** detection-resize TODO (run it, or cut to a limitation).
- **E7 forward passes done 13 Nov 03:33** (164/164 features). Next (CPU): `phase_b.sh e7-analyze`, then E8.
- **All Hydra holdout evals + CRIMSON done 12 Nov 18:48** (9/9 eval rc=0, 9/9 CRIMSON rc=0; metrics and
  crimson_metrics.json present for every run). Phase A complete on both backbones. **In LEDGER (2026-11-13 Hydra
  section), paper Table IV and §E1/E2 text (13 Nov); 12 Hydra rows added to RUNS.csv.** Descriptive: Hydra-T is never
  behind Shared beyond seed noise on either backbone and is at least level with Hydra-R everywhere except Qwen reports;
  on MedGemma it has the best counting (21.3 MAE) and most of TaskLoRA's detection gain. Bootstrap CIs still pending.
- **E7 forward passes launched 13 Nov 00:01** (box `0cc6217`): `phase_b.sh e7-run {1,2,3}`, PIDs 1088706/07/08,
  logs `runs/e7_chain{1,2,3}.log`, features `runs/phase_b/e7/`, plan `/data/ugrip2026/journal/phase_b/e7_jobs`
  (env: JOURNAL_DATA=`data_e4e5`, PHASE_B_DIR=`journal/phase_b`). 164 configurations (base + 81 adapters per
  backbone), ~5 min each at 3 parallel → ETA ~13 Nov 05:00. Two fixes before launch: `plan_e7.py` points the E5
  datasets that reuse E4 adapters (cmmd, bcn20000, neojaundice) at the modality adapter (`46ef9c2`);
  `phase_b_jobs.adapter_identity` accepts the r48 arm's declared rank/alpha deviation (`60808b6`, `0cc6217`).
  Next (CPU): `phase_b.sh e7-analyze` (needs `e3_results/transfer.json`), then E8. E6 and E9: no GPU time left.
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
| Hydra-T | done | done | done | done ×3 |
| Reversed Hydra | done | done | done | done ×3 |

"done" = trained, holdout-evaluated, CRIMSON-scored. Zero-shot (E0): done on both backbones (bare + prompt,
incl. CRIMSON).

## Next steps (Taha)
1. Hydra (Veronika): the only Phase A work left (Qwen ~320 slot-h; MedGemma 6 runs ≈ 140 slot-h). Decide
   whether to start Hydra-independent Phase B (E3/E4/E5) while waiting.
2. Review `alex/modality-decks`. Update PLAN §5.2: MedGemma full-mixture run ≈ 21–25 h wall with 3 jobs
   (~3× the planned 8 slot-h).
3. Stats (`bootstrap.py`: CIs and paired contrasts for E1): deferred for now.
4. E1 extras (PLAN E1 "also report"): per-dataset deltas by modality, output-format error table, μ.
5. **Manuscript moved to its own repo (12 Nov): `TahaSabir0/sharing-or-specializing-paper`** (private;
   local clone `C:/Users/tahas/research/IEEE-paper`). It holds `main.tex` from `b2fbac7` (Hydra Table IV
   filled) plus the `ieeecolor.cls`/`generic.sty`/`logo.eps` template files from Tausifa's Overleaf. Her
   Overleaf copy matched `e33717f` exactly, so no Overleaf edits were lost. `paper/` here is frozen; edit the
   paper repo. Next: Overleaf → Import from GitHub needs a premium account. Asked Tausifa (13 Nov) whether she
   has one or MBZUAI has an institutional licence; if so she imports it and adds us as editors. Also add
   Veronika, Alberto and Alex as GitHub collaborators on the paper repo. Earlier: body rewritten to v2 on 28 Oct
   (RQ1–RQ4, five arms, E0–E9 methods; dropped v1 studies in a supplementary section).

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
