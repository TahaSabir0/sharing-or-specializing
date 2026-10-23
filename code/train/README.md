# train/: training harness

| File | What | Who runs it |
|---|---|---|
| `journal_train.py` | the training entry point: a wrapper around LLaMA-Factory v0.9.5 with the PLAN.md §1 protocol pinned (`PROTOCOL`), per-example loss, routing ids, the plug-in hook and the run registry | via the two scripts below |
| `smoke.sh` | **smoke test**: 50 steps + a 10-per-task dev-select eval, one at a time on the box, output in `runs/smoke/` | everyone (`KICKSTART.md` §5) |
| `run_chain.sh` | real runs: a list of configs, one after another, on GPU 1 | **Taha only** |
| `journal_loss.py` | per-example loss (PLAN.md §1.5); unit test in `code/tests/` | |
| `target_modules_qwen35.json` | the frozen LoRA target set (248 language-model linears) and P8 = 21,639,168 | |
| `target_modules_medgemma15.json` | the same for MedGemma-1.5-4B: 238 decoder linears, vision tower and projector excluded, P8 = 14,901,248 (checked against the checkpoint's tensor shapes; the first smoke run confirms it on the box) | |
| `launch_template_ws.sh` | the old ws launcher, reference only | nobody |

```bash
bash code/train/smoke.sh <your_name> code/configs/v2/<config>.yaml        # everyone
# check a config resolves, no training, CPU only (point JOURNAL_RUNS at your own folder!):
JOURNAL_RUNS=/data/ugrip2026/journal/people/<you>/dry python code/train/journal_train.py --config <config.yaml> --dry
```

What a run writes (`/data/ugrip2026/journal/runs/<run_id>/`): `config.yaml` (resolved, with
deviations), `git_commit.txt`, `decks.json` (md5s), `trainer_log.jsonl`, `task_loss.jsonl`,
`runtime.json` (s/it, peak memory, parameter counts, GPU snapshot), `registry_row.csv`, and the
adapter (LoRA) or `method_state.safetensors` + `method_config.json` (plug-ins).

## MedGemma (`backbone: mg15`)

A run config with `backbone: mg15` (registry: `code/methods/backbones.py`) gets, on top of `PROTOCOL`:
the MedGemma model, LF template `gemma3`, native 896² images (the Gemma3 processor does the one
resize; LF's own 768² default cap is moved out of reach), pan-and-scan pinned off, and the 238
language-model LoRA targets. `backbone: q35` (or no `backbone:`) adds nothing: every Qwen config
resolves to exactly the arguments it had before. Evaluation takes `--backbone mg15`
(`code/eval/eval_harness.py`); `smoke.sh` passes it from the YAML.

Gate for a MedGemma method (PLAN.md §3.3, item 3, needed because prompts changed), CPU only:

```bash
CUDA_VISIBLE_DEVICES= python code/eval/template_gate.py --backbone mg15 /data/ugrip2026/journal/data/j26_train_mix.jsonl /data/ugrip2026/journal/data/j26_devselect.jsonl
```

It must print `GATE_PASS` (it exits non-zero on a missing dependency instead of skipping).

### Review note: the Gemma-only `token_type_ids` patch (for Taha)

**Where.** `backbones.patch_lf_gemma3_token_type_ids()` (about 12 lines) and its single call site,
`install_patches(..., backbone)` in `journal_train.py`, guarded by `if backbone == "mg15"`.

**What it does.** Replaces one module-level function of LLaMA-Factory in the training process,
`llamafactory.data.mm_plugin._get_gemma3_token_type_ids`. No LF file is edited, nothing is installed,
the replacement lives only in that Python process, and Qwen runs never reach it.

**Why it is needed.** Gemma3 attends bidirectionally inside each image; the model builds those blocks
from `token_type_ids == 1`. At inference the processor marks all 256 `<image_soft_token>`s of an image.
In training LF marks `processor.image_token_id`; since transformers 5.x that attribute is the id of
`<start_of_image>` (255999), not of `<image_soft_token>` (262144), so LF marks **1 token per image**.
Training and evaluation would then use different attention over the image tokens, which can only hurt
and would be invisible in the loss. Reproduced with LF `7af9095` + transformers 5.16.1 and the real
Gemma3 processor: LF 1 token per image vs processor 256 (3 images: 3 vs 768). With the patch both are
identical (`template_gate.py`: `token_type_ids_equal`; `--unpatched` shows the failure).

**To review.**
1. Do you agree with deviating from stock LF here (the alternative is to train with the mismatch)?
2. The function it replaces and the one-line `if backbone == "mg15"` guard.
3. In the smoke log: `[journal_train] mg15: LF gemma3 token_type_ids patched ...` is printed once.
4. Running the gate on the real decks on the box (the local evidence is a synthetic deck, and the
   mirror's processor files).

**Revert.** Remove the guarded call in `install_patches`; nothing else depends on it.
