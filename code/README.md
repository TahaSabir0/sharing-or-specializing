# code/: journal study code

All code for the study lives here. The protocol every script follows is **PLAN.md §1**; how a
method plugs in, how to configure, smoke-test and request a run is **`INTERFACE.md`**; setting
yourself up is **`../KICKSTART.md`**.

Seeded on 2026-10-11 from the challenge code repo (`TahaSabir0/MICCAI-FLARE-2026-Task-3`,
branch `Taha/qwen35-sprint`, commit `bec2d1f`) into fresh history. Develop here, not there.

## Layout

| Folder | What's here | State (2026-10-19) | Owner of what's left |
|---|---|---|---|
| `train/` | `journal_train.py` (LF v0.9.5 wrapper: pinned protocol, per-example loss, routing ids, plug-in hook, run registry), `journal_loss.py`, `run_chain.sh` (real runs, Taha only), **`smoke.sh`**, `e4_e5.sh` + `plan_e4_e5.py` + `group_pipeline.py` (E4/E5 prepare/plan/audit, CPU) (teammates' smoke tests), `target_modules_qwen35.json` | done; trained every run so far | MedGemma port (target list, `backbone: mg15` switch, Gemma3 render and gate): Alberto, passes the smoke gate (2026-10-23) |
| `methods/` | `base.py` (plug-in plumbing), `lora_ref.py` (reference plug-in, copy it) | done | `hydra.py` (`hydraT` + `hydraR`): Veronika |
| `eval/` | `eval_harness.py` (batched; zs / adapter / oracle / plugin), `eval_arm.py` (simple one-at-a-time, used for the first runs), `score_single.py` + `evaluation.py` (the scorer), `score_run.py`, `crimson_box.sh` + `score_crimson.py` (reports), `template_gate.py`, `run_eval_chain.sh` + `jobs/` (Taha's eval queue) | done; harness tested on the box 2026-10-19 | E3 transfer jobs: Alex; E6 predicted-route eval: Alex |
| `data/` | `build_journal_decks.py` (W1), `build_group_decks.py` (E4/E5 partitions of the frozen W1 decks), `journal_format.py` (row schema), `label_ids.json` (frozen routing ids), `build_decks.py` (challenge exporter), `build_splits_REFERENCE.py` (provenance, never rerun) | W1 done; E4/E5 builder + manifest done | stage E4/E5 decks for real runs: Taha |
| `analysis/` | `bootstrap.py` (CIs, paired bootstrap, seed bootstrap, Holm) | bootstrap done | `transfer_matrix.py` (E3), `representations.py` (E7), RQ4 synthesis (E8): Alex; `scaling_model.py` (E9): Alberto |
| `configs/` | `v2/` (Qwen E1), `e2/` (MedGemma), `e4/` + `e5/` (generated Alex group-adapter matrices), `s1/` (seed-1337 provenance), smoke/reference recipes | | regenerate E4/E5 with `python code/configs/generate_e4_e5.py` |
| `env/` | `requirements.txt`, `LF_VERSION.md`, `lf_v095_tf5_compat.patch` | frozen | |
| `tests/` | protocol/backbone tests plus `test_group_decks.py` for the E4/E5 partition and config matrix | | add tests next to your code |

## Environments (box `mbz-yaqub`)

| Env | For | Notes |
|---|---|---|
| `flare-blackwell-v5` | training and evaluation (Qwen) | Python 3.12, transformers 5.16.1, torch 2.11 cu128, LF v0.9.5 at `/data/ugrip2026/LLaMA-Factory-v5` + tf5 patch. Frozen: never `pip install` into it. |
| `crimson-score` | CRIMSON report scoring only | Python 3.11, crimson-score 0.2.0, torch 2.13 cu130 (copy of the ws env) |

Paths: models `/data/ugrip2026/models/`, decks `/data/ugrip2026/journal/data/`, runs
`/data/ugrip2026/journal/runs/<run_id>/`, smoke runs `.../runs/smoke/`. Code defaults point there;
override with `JOURNAL_DATA`, `JOURNAL_RUNS`, `MODELS_DIR`, `QWEN35_MODEL`, `CONDA_SH`, `CONDA_ENV`.

## Rules

- A new method is a plug-in behind `INTERFACE.md` and passes the smoke gate before a full run.
- Pin every generation and template setting explicitly (greedy, EOS, `enable_thinking=False`,
  `max_new_tokens=512`). Our worst past bugs came from unset defaults.
- Use the existing scorer (`eval/score_single.py`); never re-implement parsers or metrics.
- No personal paths in shared code: arguments or environment variables.
- CPU-heavy scripts cap threads (`OMP_NUM_THREADS=4` etc.); the box has 512 cores.

## E4/E5 (Alex's group adapters)

`code/E4_E5_RUNBOOK.md`. `bash code/train/e4_e5.sh prepare` builds the 51 group decks from the
frozen W1 decks into `/data/ugrip2026/journal/data_e4e5` (a personal clone: `../group_decks`) and
writes the job files for `run_chain.sh` / `run_eval_chain.sh` / `run_crimson_chain.sh`. The E4/E5
scripts are CPU-only (prepare, check, status, verify, analyze); real runs launch through the usual
runners (Taha). Smoke tests: `JOURNAL_DATA=<your group_decks> bash code/train/smoke.sh <you> <e4/e5 config>`.

## E3/E6/E7/E8 (Alex's Phase B)

`code/PHASE_B_RUNBOOK.md`; CPU entry point `bash code/train/phase_b.sh` (collection and analysis).
E3 inference runs through `run_eval_chain.sh` / `run_crimson_chain.sh` from the job files in
`code/eval/jobs/e3/` (`plan_e3_jobs.py`): Qwen with `eval_arm.py`, MedGemma with
`eval_harness.py --batch_size 1`, as their E0/E1/E2 baselines.

## Carried from the challenge (provenance)

| File | Source (@ `bec2d1f` unless noted) |
|---|---|
| `eval/evaluation.py` | `evaluation/evaluation.py` (the official-style scorer) |
| `eval/score_single.py`, `score_crimson.py`, `vlm_prompt.py`, `format_templates.py`, `prompt_variants.py`, `infer_single_adapter.py` | `llama_factory_internvl/scripts/` |
| `eval/eval_harness.py` | rewritten from `experiments/qwen35_zeroshot/zeroshot_truetest.py` (never committed upstream) |
| `eval/eval_arm.py`, `template_gate.py`, `det_box_sanity.py` | `experiments/qwen35_sprint/` |
| `eval/reportgen_omission.py` | `stats/predictions/qwen35_sprint/reportgen_normal_on_abnormal.py` (old research repo) |
| `data/build_decks.py` | `export_sharegpt.py` |

Not carried (fetch when E6 needs them): the GLiClass task router (old repo
`docker_qwen35/inference.py`, weights on ws `/share/sda/ugrip/models/FLARE-gliclass-small-v1.0`) and
the ResNet-18 modality classifier (old branch `Taha/modality-adapters`).

Hydra-T/R implementation, validation and server handoff:
[`methods/HYDRA.md`](methods/HYDRA.md), [`HYDRA_RUNBOOK.md`](HYDRA_RUNBOOK.md).
