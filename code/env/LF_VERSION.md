# LLaMA-Factory version of record (ws)

Recorded 2026-10-13 (step 0).

- Path: `/share/sda/ugrip/LLaMA-Factory-v5`, editable install in conda env `flare-ws-v5`.
- **Not a git checkout** (no `.git`), so there is no commit to record. Version marker:
  `VERSION = "0.9.5"` in `src/llamafactory/extras/env.py`.
- Source-tree fingerprint (md5 of the sorted md5sums of all 211 `src/**/*.py` files):
  `3588a8f70c459d687ada6cf56276083c`. Recompute with
  `cd /share/sda/ugrip/LLaMA-Factory-v5 && find src -type f -name "*.py" -print0 | sort -z | xargs -0 md5sum | md5sum`.
- `lf_v095_tf5_compat.patch` is applied (`git apply --check -R` succeeds); it touches only
  `src/llamafactory/model/model_utils/checkpointing.py` (md5 `b86836f0e17555b9e060b72a61e7e0bd`).

## Env `flare-ws-v5` versions
python 3.12.13 · torch 2.11.0+cu128 · transformers 5.16.1 · peft 0.18.1 · bitsandbytes 0.50.2 ·
accelerate 1.14.0 · datasets 4.0.0 · llamafactory 0.9.5

## Step 0 checks (2026-10-13, ws GPU 1)
- Qwen3.5-9B loads as `Qwen3_5ForConditionalGeneration` in NF4 (double quant off, bf16 compute):
  37 s, 8.3 GB peak. Greedy generation with `enable_thinking=False` and EOS
  `[<|im_end|>=248046, <|endoftext|>=248044]` stops on EOS.
- 1-step LF dry run (r8/α16/dropout 0.05, bsz 1 × ga 8): rc 0, loss 1.129, peak ~11.2 GB,
  `trainable params: 21,639,168`.
