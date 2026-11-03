# Status

**Updated 2026-11-03 (Hydra only; team tracker below is historical).** Plan: `PLAN.md` v2 (team and rules: §7). Numbers: `results/LEDGER.md`.
Runs: `results/RUNS.csv`. New teammates: `KICKSTART.md`.

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

## Hydra implementation (historical 2026-10-20 record; smoke update above)

- `code/methods/hydra.py`: joint-training Hydra-T and Reversed Hydra, six oracle
  task heads, frozen base, routing statistics, stored/active parameter accounting.
  Six Qwen E1 configs cover seeds 1337/2026/42. CPU-only E7 head-cosine tooling
  is in `code/analysis/hydra_heads.py`; no experimental results are claimed.
- Local validation and server commands: `code/methods/HYDRA.md`.
  **45 CPU tests passed** (torch 2.11.0, transformers 5.16.1, accelerate 1.14.0),
  including tiny real Qwen3.5/Gemma3 training, checkpointing and cached generation,
  the pre-existing per-example loss test, state/routing roundtrips and E7 tooling.
- **Pending as of 20 Oct (resolved for Qwen above):** real NF4/Blackwell smoke
  tests for both modes, then Taha's review before any full run. MedGemma end-to-end
  integration and E2 configs depend on Alberto's frozen target list and backbone
  port. The historical run states below are from 19 Oct, not a live server check.

## Historical team status (19 Oct; Hydra updated 3 Nov)
- **Phase A (main experiments) running on GPU 1.** Qwen Shared, TaskLoRA and Shared r48: seed 1337
  done and evaluated; seeds 2026 and 42 training (all done ~23 Oct).
- **First results (seed 1337, Qwen, no CIs, reports not yet scored):** fine-tuning lifts every
  task far above zero-shot; Shared vs TaskLoRA is close to a tie (TaskLoRA ahead on regression,
  driven by boneresorption, and slightly on cls; Shared ahead on detection and counting); Shared
  r48 (6× parameters) is not better than Shared overall. Details: LEDGER.
- **Waiting on review/port (PLAN.md §7.1):** Hydra-T / Reversed Hydra (Veronika;
  Qwen smoke checks now passed, PR review pending) and the MedGemma port (Alberto)
  gate the remaining Phase A work.
  Target: both pass the smoke gate by 23 Oct,
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
| Hydra-T | 50-step smoke passed 3 Nov; full run not launched by us | configs ready; not run | configs ready; not run | needs port + smoke |
| Reversed Hydra | 50-step smoke passed 3 Nov; full run not launched by us | configs ready; not run | configs ready; not run | needs port + smoke |

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
