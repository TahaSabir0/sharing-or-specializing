# Status

**Updated 2026-10-23.** Plan: `PLAN.md` v2 (team and rules: §7). Numbers: `results/LEDGER.md`.
Runs: `results/RUNS.csv`. New teammates: `KICKSTART.md`.

## Where we are
- **E0 and the E1 plain-LoRA arms on Qwen are complete:** Shared, TaskLoRA and Shared r48 × 3 seeds,
  all trained, evaluated on the holdout and CRIMSON-scored (LEDGER, 2026-10-23 section).
- **Result so far (mean over 3 seeds, no CIs yet):** Shared vs TaskLoRA is close to a tie. Shared is
  slightly ahead on cls, detection and counting; TaskLoRA on regression (boneresorption) and reports
  (CRIMSON, omission); multi-label tied. Every gap is ~1–1.5 seed SDs. Shared r48 (6× parameters) is no
  better than Shared on any task.
- **GPU 1 is idle since 23 Oct 08:17.** The remaining Phase A runs need code from teammates:
  - MedGemma port (Alberto): branch `alberto/medgemma-port` pushed 22 Oct; smoke test on the box 23 Oct
    ran end to end (0 malformed / 0 truncated on dev-select), detection 0.0 on its 10 questions: check
    in review before the gate.
  - Hydra-T / Reversed Hydra (Veronika): nothing pushed yet (target was 23 Oct).

## Phase A tracker (5 arms × 3 seeds × 2 backbones = 30 training runs)

| Arm | Qwen s1337 | Qwen s2026 | Qwen s42 | MedGemma (×3) |
|---|---|---|---|---|
| Shared | done | done | done | port in review |
| TaskLoRA (6 adapters) | done | done | done | port in review |
| Shared r48 | done | done | done | port in review |
| Hydra-T | needs plug-in | | | needs plug-in + port |
| Reversed Hydra | needs plug-in | | | needs plug-in + port |

"done" = trained, holdout-evaluated, CRIMSON-scored. Zero-shot (E0): Qwen done (bare + prompt, incl.
CRIMSON); MedGemma needs the port.

## Next steps (Taha)
1. Review Alberto's MedGemma PR (detection outputs first); if it passes, launch MedGemma E0 + E2
   Shared / TaskLoRA / r48 to fill the idle GPU.
2. Check in with Veronika on Hydra (6 Qwen runs, ~320 slot-h: the largest Phase A block left).
3. Stats (`bootstrap.py`: CIs and paired contrasts for E1): deferred for now.
4. E1 extras (PLAN E1 "also report"): per-dataset deltas by modality, output-format error table, μ.
5. Restructure `paper/main.tex` to v2 (week of 26 Oct); draft the Qwen E0/E1 results.

## Active runs
None (23 Oct). Logs of the finished chains in `/data/ugrip2026/journal/runs/`: training `chain_{E,F,G}.log`,
evals `evalchain_E1chain{1,2}.log`, CRIMSON `crimsonchain_{1,2}.log`.

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
