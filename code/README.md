# code/ — journal study code

All code for the journal study lives here. It was seeded on 2026-10-11 with the reusable
parts of the challenge code repo (`TahaSabir0/MICCAI-FLARE-2026-Task-3`, branch
`Taha/qwen35-sprint`, commit `bec2d1f`), plus two scripts that had never been committed. It
is a copy into fresh history; develop here, not in the old repo.

**The protocol every script must follow is PLAN.md §1.** Files copied from the challenge
still hold challenge settings, and the ones that matter carry a "JOURNAL REPO NOTE" banner
saying what to change.

## Layout and status

| Folder | Purpose | What's here now | Who builds the rest (PLAN.md) |
|---|---|---|---|
| `env/` | pinned stack | `requirements.txt` (Qwen3.5 stack: transformers 5.16.1, torch 2.11 cu128), `lf_v095_tf5_compat.patch` | Taha (W2) |
| `data/` | training decks and subsets | `build_decks.py` (the challenge deck exporter), `build_splits_REFERENCE.py` | Taha (W1) |
| `train/` | training harness | `launch_template_ws.sh` | Taha (W2) |
| `methods/` | one plug-in per method | — | Veronika (W4 HydraLoRA, W5 MoE family), Alex (W6 composition, W7 location) |
| `eval/` | evaluation and scoring | scorer, batched harness, single-adapter eval, CRIMSON, omission rule, gates | Taha (W3) |
| `analysis/` | stats and figures | — | Taha (W11a transfer), Alberto (W11b scaling), Alex (W8 representations) |
| `configs/` | one YAML per run family | `recipe_template_qwen35_768.yaml` | each study owner |

Also planned (PLAN.md §3.2): `code/INTERFACE.md` (the plug-in contract, written on day 1)
and `train/target_modules_qwen35.json` (the frozen LoRA target list).

## What each carried file is

**`eval/`: use these, don't re-implement them.**

| File | Source (@ `bec2d1f` unless noted) | Role |
|---|---|---|
| `evaluation.py` | `evaluation/evaluation.py` | the scorer: per-dataset metrics for all tasks, including the multi-label parser |
| `score_single.py` | `llama_factory_internvl/scripts/` | scores one predictions file per dataset; imports `evaluation.py` from this folder |
| `score_crimson.py`, `score_reportgen_crimson.sh` | `llama_factory_internvl/scripts/` | CRIMSON for report generation (`score_single.py` skips reports). The `.sh` was written for SLURM; adapt it to ws. |
| `vlm_prompt.py`, `format_templates.py`, `prompt_variants.py`, `infer_single_adapter.py` | `llama_factory_internvl/scripts/` | parsing helpers (`strip_think_block`, `parse_answer`, task-type normalisation) and the prompt-variant registry. The journal uses the bare variant (`none`); ZS-prompt still needs adding. |
| `eval_harness.py` | `experiments/qwen35_zeroshot/zeroshot_truetest.py` (**never committed**; rescued from the old local checkout, Sep 22 version) | batched, left-padded greedy evaluation with `--adapter_path` and `--image_max_pixels`. **This is the base for W3.** Its TASK_CONFIG and v1/v2 protocols are challenge-era (see the banner). |
| `eval_arm.py` | `experiments/qwen35_sprint/eval_qwen35_arm.py` | simpler single-adapter evaluation with the Qwen3.5 traps pinned |
| `template_gate.py` | `experiments/qwen35_sprint/` | train/inference byte-identity check (smoke gate item 3); needs LLaMA-Factory importable |
| `det_box_sanity.py` | `experiments/qwen35_sprint/` | detection output sanity (nested boxes, out of bounds, boxes per case) |
| `reportgen_omission.py` | old research repo, `stats/predictions/qwen35_sprint/reportgen_normal_on_abnormal.py` | the report omission rule (PLAN.md §1.8) |

**`data/`**
- `build_decks.py` (from `export_sharegpt.py`): builds ShareGPT decks from the frozen split
  jsonl files. It already has `--max_images`, `--markers`, `--task`, prompt variants and the
  k2a rebalance. W1 adapts it to one row per question (≤4 marked images) and adds
  task/modality/dataset labels on every row.
- `build_splits_REFERENCE.py`: how `internal_splits_v2` was built, kept for provenance.
  **Never rerun it**; the splits are frozen on ws.

**Not carried, fetch when needed:**
- The GLiClass router inference code (in the old repo's `docker_qwen35/inference.py`) and
  the ResNet-18 modality classifier (old branch `Taha/modality-adapters`). Both are needed
  for S4c only; their weights are on ws.
- `infer_single_adapter.py` has an env-gated, default-off border-crop hook that imports
  `experiments.qwen3_matched_border_crop` from the old repo. If S9.4 needs the crop, copy that
  module here.

## Environment (ws)
- Conda env `flare-ws-v5` (transformers 5.x, torch 2.11 cu128), with LLaMA-Factory v0.9.5 at
  `/share/sda/ugrip/LLaMA-Factory-v5` and `env/lf_v095_tf5_compat.patch` applied. Record the
  exact LLaMA-Factory commit in `env/` (W2).
- Data, splits and base models stay on ws under `/share/sda/ugrip/`. Run outputs go to
  `/share/sda/ugrip/journal/runs/<run_id>/`, and each run gets a row in `results/RUNS.csv`.
- On Windows, scripts that print `∪` need `PYTHONIOENCODING=utf-8`; Linux is fine.

## Rules
- Every new method is a plug-in behind the interface in `code/INTERFACE.md`, and must pass
  the smoke gate (PLAN.md §3.3) before a full run.
- Pin every generation and template setting explicitly (greedy, EOS, `enable_thinking=False`,
  `max_new_tokens=512`).
- Never hard-code personal paths in shared code; use arguments or environment variables.
