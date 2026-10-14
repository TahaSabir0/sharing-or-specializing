"""Simple one-at-a-time journal evaluator (smoke gate item 4; quick checks).

Adapted from the challenge eval_qwen35_arm.py: same model loading and Qwen3.5 traps pinned, but
journal rows (code/data/journal_format.py) rendered by code/eval/journal_render.py, bare prompts,
no answer seeds, no prompt variants. The batched full evaluator is eval_harness.py (W3).

  CUDA_VISIBLE_DEVICES=1 python code/eval/eval_arm.py --data_path /data/ugrip2026/journal/data/j26_holdout.jsonl \
      --adapter_path <run_dir> --per_task 50 --output_file preds.json
  --adapter_path none                  ZS-bare (base model)
  --adapter_path none --zs_prompt      ZS-prompt (generic system prompt, PLAN.md §1.6)

Output: a predictions list in score_single.py's format, plus raw_output, n_new_tokens, truncated.
"""
import argparse
import collections
import json
import os
import sys
import time

import torch
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from infer_single_adapter import parse_answer  # noqa: E402  (the repo parser; not re-implemented)
from journal_render import (IMAGE_MAX_PIXELS, ZS_PROMPT_SYSTEM, capped_images, chat_text,  # noqa: E402
                            eos_ids, load_images)
from vlm_prompt import strip_think_block  # noqa: E402

MAX_NEW_TOKENS = 512


def load_model(base, adapter_path):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=False)
    model = AutoModelForImageTextToText.from_pretrained(base, quantization_config=bnb, dtype=torch.bfloat16,
                                                        device_map="cuda:0", local_files_only=True)
    if adapter_path and adapter_path != "none":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    return model.eval()


def select_rows(path, per_task, only_task):
    rows, n = [], collections.Counter()
    for line in open(path):
        r = json.loads(line)
        if only_task and r["task"] != only_task:
            continue
        if per_task and n[r["task"]] >= per_task:
            continue
        n[r["task"]] += 1
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default=os.environ.get("QWEN35_MODEL", "/data/ugrip2026/models/Qwen3.5-9B"))
    ap.add_argument("--adapter_path", required=True, help="LoRA adapter dir, or 'none' for zero-shot")
    ap.add_argument("--zs_prompt", action="store_true", help="add the ZS-prompt system prompt")
    ap.add_argument("--data_path", required=True, help="journal deck jsonl (e.g. j26_holdout.jsonl)")
    ap.add_argument("--per_task", type=int, default=0, help="first N rows per task (0 = all)")
    ap.add_argument("--only_task", default=None)
    ap.add_argument("--output_file", required=True)
    a = ap.parse_args()
    if a.zs_prompt and a.adapter_path != "none":
        raise SystemExit("ZS-prompt is a zero-shot protocol; adapters are always evaluated bare")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    proc = AutoProcessor.from_pretrained(a.base_model, local_files_only=True)
    tok = proc.tokenizer
    model = load_model(a.base_model, a.adapter_path)
    eos = eos_ids(tok)
    system = ZS_PROMPT_SYSTEM if a.zs_prompt else None
    rows = select_rows(a.data_path, a.per_task, a.only_task)
    print(f"adapter={a.adapter_path} zs_prompt={a.zs_prompt} eos={eos} rows={len(rows)} "
          f"cap={IMAGE_MAX_PIXELS} max_new_tokens={MAX_NEW_TOKENS}")

    out = []
    for r in tqdm(rows):
        paths = capped_images({"image": r["images"]})
        try:
            text = chat_text(proc, r["question"], len(paths), system=system)
            inputs = proc(text=[text], images=load_images(paths), return_tensors="pt").to("cuda:0")
            t0 = time.time()
            with torch.inference_mode():
                gen = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                     num_beams=1, temperature=None, top_p=None, top_k=None,
                                     eos_token_id=eos, pad_token_id=tok.pad_token_id or eos[0])
            new = gen[0, inputs["input_ids"].shape[1]:]
            raw = tok.decode(new, skip_special_tokens=True)
            n_new = int(new.numel())
            truncated = n_new >= MAX_NEW_TOKENS and int(new[-1]) not in eos
            ans = parse_answer(strip_think_block(raw), r["task"])  # strip first, then parse
            dt = round(time.time() - t0, 3)
        except Exception as e:  # noqa: BLE001
            raw, ans, n_new, truncated, dt = "", f"Error: {e}", 0, False, None
        out.append(dict(id=r["id"], ImageName=r["images"], Question=r["question"], TaskType=r["task"],
                        dataset=r["dataset"], gt_answer=r["answer"], raw_output=raw, Answer=ans,
                        n_new_tokens=n_new, truncated=truncated, gen_seconds=dt))
    os.makedirs(os.path.dirname(os.path.abspath(a.output_file)), exist_ok=True)
    json.dump(out, open(a.output_file, "w"), indent=1)
    by = collections.defaultdict(list)
    for o in out:
        by[o["TaskType"]].append(o)
    for t, os_ in sorted(by.items()):
        print(f"{t}: n={len(os_)} truncated={sum(o['truncated'] for o in os_)} "
              f"errors={sum(str(o['Answer']).startswith('Error:') for o in os_)}")
    print(f"saved {len(out)} -> {a.output_file}")


if __name__ == "__main__":
    main()
