# Status

**Updated 2026-10-19.** Plan: `PLAN.md` v2 (team and rules: §7). Numbers: `results/LEDGER.md`.
Runs: `results/RUNS.csv`. New teammates: `KICKSTART.md`.

## Where we are
- **Phase A (main experiments) running on GPU 1.** Qwen Shared, TaskLoRA and Shared r48: seed 1337
  done and evaluated; seeds 2026 and 42 training (all done ~23 Oct).
- **First results (seed 1337, Qwen, no CIs, reports not yet scored):** fine-tuning lifts every
  task far above zero-shot; Shared vs TaskLoRA is close to a tie (TaskLoRA ahead on regression,
  driven by boneresorption, and slightly on cls; Shared ahead on detection and counting); Shared
  r48 (6× parameters) is not better than Shared overall. Details: LEDGER.
- **Blocked on code, now owned (PLAN.md §7.1):** Hydra-T / Reversed Hydra (Veronika) and the
  MedGemma port (Alberto) gate 21 of the 30 Phase A runs. Target: both pass the smoke gate by 23 Oct,
  when the GPU queue runs dry.
- **Team set-up done 2026-10-19:** eval harness merged and tested on the box; `smoke.sh`;
  `code/INTERFACE.md` + reference plug-in; `KICKSTART.md`; CRIMSON env and judge on the box;
  MedGemma weights on the box; plan §7.

## Phase A tracker (5 arms × 3 seeds × 2 backbones = 30 training runs)

| Arm | Qwen s1337 | Qwen s2026 | Qwen s42 | MedGemma (×3) |
|---|---|---|---|---|
| Shared | done, evaluated | training | training | needs port |
| TaskLoRA (6 adapters) | done, evaluated | cls/report/ml done (evals queued), det/reg/count training | queued (chain F) | needs port |
| Shared r48 | done, evaluated | queued (chain E) | queued (chain G) | needs port |
| Hydra-T | needs plug-in | | | needs plug-in + port |
| Reversed Hydra | needs plug-in | | | needs plug-in + port |

Zero-shot (E0): Qwen done (bare + prompt); MedGemma needs the port.

## Next steps (Taha)
1. CRIMSON for the 6 report prediction files already on the box (ZS ×2, Shared, TaskLoRA-report,
   Shared r48, TaskLoRA-report s2026 when evaluated), then LEDGER.
2. Queue holdout evals of seeds 2026/42 as their training finishes (TaskLoRA s2026 evals are queued).
3. Review the first PRs (Hydra, MedGemma); queue their runs as soon as they pass.
4. Restructure `paper/main.tex` to v2 (week of 26 Oct).

## Active runs (box time, +04; `mbz-yaqub` GPU 1; logs in `/data/ugrip2026/journal/runs/`)

| Chain | Runs, in order | State (19 Oct) |
|---|---|---|
| E | `shared_s2026` → `shared48_s2026` | shared s2026 at 82%; r48 next (~2.5 days) |
| F | TaskLoRA s2026 (cls ✓, report ✓, ml ✓, det, reg, count) → TaskLoRA s42 (same order) | det s2026 running; all done ~22–23 Oct |
| G | `shared_s42` → `shared48_s42` | shared s42 at 34% |
| eval | `code/eval/jobs/v2_e1_task_s2026_holdout.txt` (each TaskLoRA s2026 on its own slice; det/reg/count wait for chain F) | running |

Check: `ssh blackwell 'cat /data/ugrip2026/journal/runs/chain_{E,F,G}.log /data/ugrip2026/journal/runs/evalchain_*.log'`;
per run: `<run_id>.log`, `<run_id>/runtime.json`. Smoke lock: `cat /data/ugrip2026/journal/runs/smoke/.holder`.

## Things to remember
- Every compared run is trained and evaluated on the box (ws and the box are not
  bit-reproducible; PLAN.md §1.12; numbers in LEDGER "Setup findings").
- Never update `/data/ugrip2026/journal/repo` while a chain will start a new run from it with
  changed code: chains read the code at each run's start (RUNS.csv notes one harmless case).
- Thread caps are mandatory on the box (`run_chain.sh` and `smoke.sh` set them): uncapped
  tokenizer workers once loaded ~400 of 512 cores (14 Oct).
- 3 jobs at once = 2.0× one job, card ~90% busy; a 4th job mostly slows the others.

## For Tausifa
- Dev is inside the training pool, so v2 makes no selection decisions (PLAN.md §1.7).
- v2 decisions N1–N8 (PLAN.md §6) and the team split (§7.1).
