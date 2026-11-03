# Status

**Updated 2026-11-03.** Plan: `PLAN.md` v2 (team and rules: §7). Numbers: `results/LEDGER.md`.
Runs: `results/RUNS.csv`. New teammates: `KICKSTART.md`.

## Veronika handoff: integrated with main (3 Nov)

- Preserved the Qwen-tested implementation as `ad7497e`, then integrated main
  `f123fa8`, including Alberto's working MedGemma port and Alex's E7 tooling.
  Both backbone target sets dispatch through the existing model-type registry;
  Hydra parameter accounting is retained in the runtime callback.
- Twelve full-run configs are ready: six E1 Qwen and six E2 MedGemma, both modes
  and seeds 1337/2026/42, all using the common recipe without overrides.
  MedGemma Hydra stored budgets are 55,287,808 / 49,020,928 (T/R), active
  14,901,248. Actual NF4 MedGemma Hydra smoke is still pending.
- E7 code now exports exact checkpoint head maps compatible with Alex's
  `representations.py heads`, per target or concatenated per decoder block.
  Interoperability is CPU-tested. Actual E7 findings require full checkpoints.
- `paper/hydra_sections_draft.md` supplies methods/analysis text with explicit
  result placeholders; `code/HYDRA_RUNBOOK.md` documents the server handoff,
  remaining gates, collection and ownership. Small evidence collection is in
  `code/train/collect_hydra_smoke.py`; no weights or datasets are included.
- Validation: **97 focused tests passed** (Hydra, tiny real Qwen/Gemma, both full
  frozen target sets, config matrix, backbone/render/per-example loss and evidence
  collection). Full suite: **140 passed, 2 failed**; the identical two failures
  reproduce on pristine `f123fa8` (**69 passed, 2 failed**): CRIMSON layout fixture
  expectations, and global routing state left by the per-example-loss test.
  The CRIMSON shell test also emits macOS `date -Is` compatibility warnings.
  These existing failures are documented, not silently attributed to Hydra or
  fixed by changing another owner's production runners.
- Pending: collect raw Qwen artifacts, revalidate the integrated branch on Qwen,
  pass both MedGemma smokes, and Taha's review/full runs. Full-run E7 analysis,
  paper results and ledger additions depend on those completed checkpoints.

## Hydra smoke follow-up (2026-11-03)

- User-provided Blackwell logs: reference smoke
  `j26_q35_loraref_full_768_s1337_smoke_veronika_10262347` finished training/eval
  with exit codes 0/0, loss 1.562 -> 0.644, 21,639,168 trainable parameters,
  18.813 s/step and 16.97 GiB peak reserved memory; all 60 dev-select predictions
  had zero malformed/truncated/error counts, routing shape 60 x 248 x 1.
- Hydra-T `j26_q35_hydraT_full_768_s1337_smoke_veronika_10270023` timed out
  (exit 124) after 44/50 steps, 58:16 training time, last displayed 80.61 s/step.
  Loss 1.536 -> 0.808; stored/active counts 78,716,928 / 21,639,168 matched.
  Evaluation was skipped. This is NOT a passed Hydra smoke gate.
- Routing performance fix: prepare host task groups once per batch, share them
  across layers/checkpoint replay, and directly call the selected head for uniform
  batches (including protocol batch size 1). Mixed-batch row indices are cached
  per device; evaluation routing weights stay on CPU. No protocol/config changes.
- **63 CPU tests passed** on the isolated local torch 2.11.0 / transformers 5.16.1
  environment, including formula/gradient checks for uniform/mixed batches,
  dropout with both checkpoint modes, route lifecycle, inference-mode labels,
  and a profiler regression check for scalar sync/row ops on the batch-one path.
  Post-fix Blackwell smoke summaries were subsequently supplied by the user:

| Mode / run suffix | Train/eval exits | First -> last loss | Trainable params | s/step | Peak reserved GiB |
|---|---|---|---:|---:|---:|
| Hydra-T / `11032130` | 0 / 0 | 1.534 -> 0.641 | 78,716,928 | 14.122 | 17.89 |
| Hydra-R / `11032152` | 0 / 0 | 1.611 -> 0.630 | 72,757,248 | 14.147 | 17.79 |

- Full run IDs are `j26_q35_hydraT_full_768_s1337_smoke_veronika_11032130`
  and `j26_q35_hydraR_full_768_s1337_smoke_veronika_11032152`, under
  `/data/ugrip2026/journal/runs/smoke/`. Both completed the 50-step smoke and
  all 60 dev-select predictions, with zero malformed/truncation/error counts;
  both routing files validated as 60 sequences x 248 layers x 6 experts.
  Counts match the expected stored budgets. Prompts/data were unchanged.
- These summaries establish the reported Qwen smoke checks for both modes.
  Source was the uncommitted routing update on `codex/veronika-hydra` at base
  `dfbce98`, transferred in `hydra-routing-update-20261103.tar.gz` (SHA256
  `d997e1d4a8e7bf807010f2e2e84ee277f6b17f7dff2b3d16b6e022a22c323a48`).
  HEAD alone does not identify the tested source. Raw runtime/config/state files
  still need collection for the PR, including active-parameter runtime fields.
  Taha's review and the MedGemma gate remain pending. No full training or holdout
  evaluation has been launched by this work. Smoke task scores are format checks,
  not evidence for relative method quality; shared-GPU timings are not a controlled
  before/after performance benchmark.

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
- Hydra-T / Reversed Hydra (Veronika): Qwen 50-step smoke checks passed 3 Nov; MedGemma integration/configs and PR review in progress (details above).
- **E4/E5 ready to launch (26 Oct):** Alex's `alex/modality-decks` reviewed, reworked and merged
  (`code/E4_E5_RUNBOOK.md`). 51 group decks in `/data/ugrip2026/journal/data_e4e5`; job files in
  `code/eval/jobs/e4e5/` (3 training chains × 28 runs; 42 evals per backbone; 6 CRIMSON). 84 runs, not 102:
  Clinical/Derm/Mammo hold one dataset each, so E5 reuses those E4 adapters (PLAN changelog 26 Oct).
  **Training done 31 Oct 00:47:** all 84 runs rc=0 and pass the protocol check (3 epochs, deck hash,
  P8). No MedGemma smoke (deviation, PLAN changelog). **Holdout evals done 2 Nov 03:31** (84/84 rc=0) and
  **CRIMSON done 2 Nov 06:17** (6/6 rc=0). Next: `e4_e5.sh verify` and `analyze` (CPU), then LEDGER.
- **E3/E6/E7/E8 code (Alex) merged 3 Nov** (`de4b21f`). At review E3 inference moved to the usual
  chain runners with each backbone's baseline evaluator (Qwen `eval_arm.py`, not the harness);
  job files `code/eval/jobs/e3/`. **E3 launched 3 Nov 10:52** (box `fcbe2fa`; eval slot PIDs
  3630186/88/90, CRIMSON 3630187/89/91; logs `runs/evalchain_e3_{q35,mg15}_s<seed>.log`,
  `runs/crimsonchain_e3_s<seed>.log`). ETA: Qwen ~3 Nov 20:00, MedGemma ~4 Nov 07:00, CRIMSON ~4 Nov 14:00. Asked Alex
  (3 Nov) to adapt `plan_e3.py collect` to those outputs, grade the output-structure covariate
  (6 distinct types make "same structure" 0 for every pair) and freeze the E7 probe IDs / list
  the E6 router packages. Veronika's Hydra smoke source was uncommitted in her box clone (`codex/veronika-hydra`); the tested snapshot is now preserved locally in commit `ad7497e`.
- Scorer fix merged (`score_single.py` dropped a score equal to the question count): rescoring all 52
  existing holdout prediction files changed no number in the LEDGER.

## Phase A tracker (5 arms × 3 seeds × 2 backbones = 30 training runs)

| Arm | Qwen s1337 | Qwen s2026 | Qwen s42 | MedGemma (×3) |
|---|---|---|---|---|
| Shared | done | done | done | done ×3 |
| TaskLoRA (6 adapters) | done | done | done | done ×3 |
| Shared r48 | done | done | done | done ×3 |
| Hydra-T | smoke passed; full run pending | configs ready | configs ready | integration/configs ready; smoke pending |
| Reversed Hydra | smoke passed; full run pending | configs ready | configs ready | integration/configs ready; smoke pending |

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
