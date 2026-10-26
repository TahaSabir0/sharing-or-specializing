# Status

**Updated 2026-10-26.** Plan: `PLAN.md` v2 (team and rules: §7). Numbers: `results/LEDGER.md`.
Runs: `results/RUNS.csv`. New teammates: `KICKSTART.md`.

## Where we are
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
- Hydra-T / Reversed Hydra (Veronika): nothing pushed yet (target was 23 Oct). **GPU 1 idle since 26 Oct 13:54.**
- Alex pushed `alex/modality-decks` (E4/E5 code, runbook; touches `journal_train.py`, `score_single.py`,
  `run_crimson_chain.sh`, `smoke.sh`): needs review before any E4/E5 run.

## Phase A tracker (5 arms × 3 seeds × 2 backbones = 30 training runs)

| Arm | Qwen s1337 | Qwen s2026 | Qwen s42 | MedGemma (×3) |
|---|---|---|---|---|
| Shared | done | done | done | done ×3 |
| TaskLoRA (6 adapters) | done | done | done | done ×3 |
| Shared r48 | done | done | done | done ×3 |
| Hydra-T | needs plug-in | | | needs plug-in + port |
| Reversed Hydra | needs plug-in | | | needs plug-in + port |

"done" = trained, holdout-evaluated, CRIMSON-scored. Zero-shot (E0): done on both backbones (bare + prompt,
incl. CRIMSON).

## Next steps (Taha)
1. Hydra (Veronika): the only Phase A work left (Qwen ~320 slot-h; MedGemma 6 runs ≈ 140 slot-h). Decide
   whether to start Hydra-independent Phase B (E3/E4/E5) while waiting.
2. Review `alex/modality-decks`. Update PLAN §5.2: MedGemma full-mixture run ≈ 21–25 h wall with 3 jobs
   (~3× the planned 8 slot-h).
3. Stats (`bootstrap.py`: CIs and paired contrasts for E1): deferred for now.
4. E1 extras (PLAN E1 "also report"): per-dataset deltas by modality, output-format error table, μ.
5. Restructure `paper/main.tex` to v2 (week of 26 Oct); draft the Qwen E0/E1 results.
   **Waiting on Tausifa (asked 23 Oct):** her revised proposal replaces the 8 RQs with 4 that match
   PLAN.md RQ1–RQ4, but the rest of it is unchanged from `proposal_original.tex`, so it still has the
   studies v2 dropped (Soft-LoRA, latent experts/router, location, practical factors; N5/N6). Asked
   whether it is her latest copy and whether I can rewrite the body to v2. Her file isn't in the repo
   yet (only the RQ list differs). Small RQ wording points to raise when restructuring: RQ1 "capacity
   matched" = Shared r48 control; RQ3 "visual content" → E7 input similarity; RQ4 "measured before
   training" vs E8's trained-adapter predictors.

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
