#!/usr/bin/env python3
"""W3: batched journal evaluation harness (PLAN.md §1.6, §3.1 W3).

Rewritten from the challenge zeroshot_truetest.py (batched, left-padded greedy generation,
resumable per-task checkpoints). Journal protocol: bare prompts rendered by journal_render.py
(identical to training), max_new_tokens 512 for every task, greedy, EOS pinned, no answer seeds,
the repo parser. Per backbone (--backbone, default q35):
  q35   Qwen3.5-9B: 768² cap, EOS [<|im_end|>, <|endoftext|>], enable_thinking=False
  mg15  MedGemma-1.5-4B: native 896² (the processor resizes; no cap), EOS [<end_of_turn>, <eos>],
        Gemma3 prompt built by hand, one image list per prompt (journal_render.py)

Modes
  --mode zs --protocol bare|prompt         zero-shot base (ZS-bare / ZS-prompt)
  --mode adapter --adapter <dir>           one LoRA adapter on every task (e.g. Shared)
  --mode oracle --adapter_set <json>       {task: adapter_dir}; the known task picks the adapter
                                           (TaskLoRA; also S2 rows via --tasks)
  --mode compose --compose <json>          {"adapters": {name: dir}, "weights": {name: w}}
                                           runtime weighted sum of LoRA deltas, nothing merged
                                           into the 4-bit base: out = base(x) + Σ w_i·ΔW_i x
  --mode plugin --run_dir <dir>            a method plug-in trained by journal_train
                                           (method_config.json + method_state.safetensors);
                                           routing ids set per batch; routing_stats.npz saved

  CUDA_VISIBLE_DEVICES=1 python code/eval/eval_harness.py --mode zs --protocol bare \
      --data /data/ugrip2026/journal/data/j26_holdout.jsonl --surface holdout --out_dir <dir>
Then: python code/eval/score_run.py --out_dir <dir>   (+ CRIMSON: code/eval/crimson_box.sh <dir>)

Outputs in --out_dir: <task>.jsonl (resumable), all_predictions.json, run_meta.json.
Every prediction carries raw_output, Answer (parsed), n_new_tokens and truncated (hit the cap
without EOS).
"""
import argparse
import collections
import datetime
import json
import os
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CODE))
from infer_single_adapter import parse_answer  # noqa: E402  (repo parser, not re-implemented)
from journal_render import (ZS_PROMPT_SYSTEM, capped_images, chat_text, encode_batch,  # noqa: E402
                            eos_ids as pinned_eos, image_rules, load_images)
from vlm_prompt import strip_think_block  # noqa: E402
from methods import backbones  # noqa: E402
from methods import base as mb  # noqa: E402

MAX_NEW_TOKENS = 512
MODELS_DIR = os.environ.get("MODELS_DIR", "/data/ugrip2026/models")
DEFAULT_BASE = {"q35": os.environ.get("QWEN35_MODEL", f"{MODELS_DIR}/Qwen3.5-9B"),
                "mg15": os.environ.get("MEDGEMMA_MODEL", f"{MODELS_DIR}/medgemma-1.5-4b-it")}
TASK_ORDER = ["counting", "detection", "regression", "multi-label classification",
              "classification", "report_generation"]
TASK_SHORT = {"classification": "cls", "multi-label classification": "ml", "detection": "det",
              "counting": "count", "regression": "reg", "report_generation": "report"}
SHORT_TASK = {v: k for k, v in TASK_SHORT.items()}


# ------------------------------------------------------------------------------------ model
def load_base(base):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=False)
    return AutoModelForImageTextToText.from_pretrained(base, quantization_config=bnb, dtype=torch.bfloat16,
                                                       device_map="cuda:0", local_files_only=True)


def load_peft_named(model, adapters):
    """Load several LoRA adapters into one PeftModel under their names (nothing merged)."""
    from peft import PeftModel
    names = list(adapters)
    model = PeftModel.from_pretrained(model, adapters[names[0]], adapter_name=names[0], is_trainable=False)
    for n in names[1:]:
        model.load_adapter(adapters[n], adapter_name=n, is_trainable=False)
    return model


def lora_layers(model):
    from peft.tuners.lora import LoraLayer
    return [m for m in model.modules() if isinstance(m, LoraLayer)]


class Composer:
    """Runtime weighted composition: activate several adapters and scale each one's delta by w_i."""

    def __init__(self, model, weights):
        self.model, self.weights = model, dict(weights)
        self.layers = lora_layers(model)
        self.base_scaling = [{n: l.scaling[n] for n in self.weights} for l in self.layers]

    def apply(self):
        self.model.base_model.set_adapter(list(self.weights))  # LoraModel accepts a list
        for l, base in zip(self.layers, self.base_scaling):
            for n, w in self.weights.items():
                l.scaling[n] = base[n] * float(w)
        active = self.model.base_model.active_adapters
        assert sorted(active) == sorted(self.weights), active


def build_model(a):
    """Returns (model, select_fn(task) -> None, meta dict)."""
    model = load_base(a.base_model)
    meta = {"mode": a.mode}
    select = lambda task: None  # noqa: E731
    if a.mode == "zs":
        meta["protocol"] = a.protocol
    elif a.mode == "adapter":
        model = load_peft_named(model, {"a": a.adapter})
        meta["adapter"] = a.adapter
    elif a.mode == "oracle":
        aset = json.load(open(a.adapter_set))
        aset = {SHORT_TASK.get(k, k): v for k, v in aset.items()}
        names = {t: TASK_SHORT[t] for t in aset}
        model = load_peft_named(model, {names[t]: p for t, p in aset.items()})
        meta["adapter_set"] = aset

        def select(task):  # noqa: F811
            if task not in names:
                raise KeyError(f"no adapter for task {task!r} in the oracle set")
            model.set_adapter(names[task])
    elif a.mode == "compose":
        spec = json.load(open(a.compose))
        model = load_peft_named(model, spec["adapters"])
        comp = Composer(model, spec["weights"])
        comp.apply()
        meta["compose"] = spec
    elif a.mode == "plugin":
        mcfg = json.load(open(os.path.join(a.run_dir, mb.METHOD_CONFIG)))
        import importlib
        plugin = importlib.import_module(f"methods.{mcfg['method']}")
        model.requires_grad_(False)
        model = plugin.apply_method(model, mcfg["cfg"])
        mb.load_method_state(model, a.run_dir)
        meta.update(method=mcfg["method"], method_cfg=mcfg["cfg"], run_dir=a.run_dir)
    return model.eval(), select, meta


# ---------------------------------------------------------------------------------- batching
def make_batches(rows, max_rows, max_imgs):
    rows = sorted(rows, key=lambda r: (len(capped_images({"image": r["images"]})), len(r["question"])))
    batches, cur, cur_imgs = [], [], 0
    for r in rows:
        n = len(capped_images({"image": r["images"]}))
        if cur and (len(cur) >= max_rows or cur_imgs + n > max_imgs):
            batches.append(cur)
            cur, cur_imgs = [], 0
        cur.append(r)
        cur_imgs += n
    if cur:
        batches.append(cur)
    return batches


def count_new(gen_row, eos):
    """Generated length up to and including the first EOS; truncated = no EOS within the cap."""
    ids = gen_row.tolist()
    for i, t in enumerate(ids):
        if t in eos:
            return i + 1, False
    return len(ids), len(ids) >= MAX_NEW_TOKENS


def git_commit():
    r = subprocess.run(f"git -C {CODE.parent} rev-parse HEAD", shell=True, capture_output=True, text=True)
    d = subprocess.run(f"git -C {CODE.parent} status --porcelain --untracked-files=no", shell=True,
                       capture_output=True, text=True)
    return r.stdout.strip() + ("-dirty" if d.stdout.strip() else "")


# --------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["zs", "adapter", "oracle", "compose", "plugin"])
    ap.add_argument("--protocol", choices=["bare", "prompt"], default="bare", help="zero-shot only")
    ap.add_argument("--adapter")
    ap.add_argument("--adapter_set")
    ap.add_argument("--compose")
    ap.add_argument("--run_dir")
    ap.add_argument("--backbone", choices=sorted(backbones.BACKBONES), default=backbones.DEFAULT_BACKBONE)
    ap.add_argument("--base_model", default=None, help="default: the backbone's model on the box")
    ap.add_argument("--data", required=True, help="journal eval deck (j26_holdout.jsonl, j26_devselect.jsonl)")
    ap.add_argument("--surface", required=True, choices=["holdout", "dev", "devselect", "valpub"],
                    help="label of the evaluation surface (PLAN rule: always labelled)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tasks", default="", help="comma list of tasks or short names; default all")
    ap.add_argument("--per_task", type=int, default=0, help="first N rows per task (smoke)")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_images_per_batch", type=int, default=12)
    a = ap.parse_args()
    if a.mode != "zs" and a.protocol == "prompt":
        raise SystemExit("ZS-prompt is zero-shot only; adapters are evaluated bare (PLAN.md §1.6)")
    for v in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        os.environ.setdefault(v, "1")
    a.base_model = a.base_model or DEFAULT_BASE[a.backbone]
    rules = image_rules(a.backbone)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = [SHORT_TASK.get(t.strip(), t.strip()) for t in a.tasks.split(",") if t.strip()] or list(TASK_ORDER)
    proc = AutoProcessor.from_pretrained(a.base_model, local_files_only=True)
    tok = proc.tokenizer
    tok.padding_side = "left"
    eos = pinned_eos(tok, a.backbone)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else eos[0]
    model, select, meta = build_model(a)
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    if a.backbone != "q35" and model_type is not None and model_type != backbones.get_backbone(a.backbone).model_type:
        raise SystemExit(f"--backbone {a.backbone} expects model_type "
                         f"{backbones.get_backbone(a.backbone).model_type!r}, loaded {model_type!r} from {a.base_model}")
    system = ZS_PROMPT_SYSTEM if (a.mode == "zs" and a.protocol == "prompt") else None
    stats = getattr(model, "journal_routing_stats", None)
    if stats is None and a.mode == "plugin":
        stats = getattr(getattr(model, "base_model", None), "journal_routing_stats", None)

    meta.update({"surface": a.surface, "data": a.data, "tasks": tasks, "per_task": a.per_task,
                 "backbone": a.backbone, "base_model": a.base_model, "model_type": model_type,
                 "system_prompt": system, "max_new_tokens": MAX_NEW_TOKENS,
                 "image_max_pixels": rules["max_pixels"], "image_min_pixels": rules["min_pixels"],
                 "eos_token_id": eos, "decoding": "greedy (do_sample=False, num_beams=1)",
                 "enable_thinking": False if a.backbone == "q35" else None, "batch_size": a.batch_size, "host": socket.gethostname(),
                 "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", ""), "code_commit": git_commit(),
                 "started": datetime.datetime.now().isoformat(timespec="seconds")})
    json.dump(meta, open(out_dir / "run_meta.json", "w"), indent=1)
    print(f"[eval] {json.dumps({k: meta[k] for k in ('mode', 'surface', 'system_prompt', 'eos_token_id')})}", flush=True)

    rows_all = [json.loads(l) for l in open(a.data) if l.strip()]
    t_grand = time.time()
    timings = {}
    for task in tasks:
        rows = [r for r in rows_all if r["task"] == task]
        if a.per_task:
            rows = rows[:a.per_task]
        if not rows:
            continue
        select(task)
        ckpt = out_dir / f"{TASK_SHORT[task]}.jsonl"
        done = set()
        if ckpt.exists():
            done = {json.loads(l)["id"] for l in open(ckpt) if l.strip()}
        todo = [r for r in rows if r["id"] not in done]
        print(f"\n=== {task}: {len(rows)} rows, {len(done)} done, {len(todo)} to run", flush=True)
        t0, n_done, fails = time.time(), 0, 0
        fout = open(ckpt, "a")

        def write(r, raw, n_new, trunc, err=None):
            ans = f"Error: {err}" if err else parse_answer(strip_think_block(raw), r["task"])
            fout.write(json.dumps(dict(
                id=r["id"], ImageName=r["images"], Question=r["question"], TaskType=r["task"],
                dataset=r["dataset"], modality=r["modality"], gt_answer=r["answer"], raw_output=raw,
                Answer=ans, n_new_tokens=n_new, truncated=trunc), ensure_ascii=False) + "\n")

        def process(batch):
            texts, image_lists = [], []
            for r in batch:
                paths = capped_images({"image": r["images"]})
                texts.append(chat_text(proc, r["question"], len(paths), system=system, backbone=a.backbone))
                image_lists.append(load_images(paths, **rules))
            inputs = encode_batch(proc, texts, image_lists, a.backbone).to("cuda:0")
            n_in = inputs["input_ids"].shape[1]
            ids = {k: torch.tensor([r[k] for r in batch], device="cuda:0") for k in mb.ROUTING_KEYS}
            with mb.routing.use(**ids), torch.inference_mode():
                gen = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, num_beams=1,
                                     temperature=None, top_p=None, top_k=None,
                                     eos_token_id=eos, pad_token_id=pad_id)
                if stats is not None:
                    stats.end_batch([r["id"] for r in batch])
            for r, g in zip(batch, gen[:, n_in:]):
                n_new, trunc = count_new(g, eos)
                write(r, tok.decode(g[:n_new], skip_special_tokens=True), n_new, trunc)

        for bi, batch in enumerate(make_batches(todo, a.batch_size, a.max_images_per_batch)):
            try:
                process(batch)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  OOM on batch of {len(batch)}; retrying one at a time", flush=True)
                for r in batch:
                    try:
                        process([r])
                    except Exception as e:  # noqa: BLE001
                        fails += 1
                        write(r, "", 0, False, err=e)
                        torch.cuda.empty_cache()
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                fails += len(batch)
                for r in batch:
                    write(r, "", 0, False, err=e)
            fout.flush()
            n_done += len(batch)
            if (bi + 1) % 10 == 0 or n_done == len(todo):
                el = time.time() - t0
                print(f"  [{task}] {n_done}/{len(todo)} {el / max(n_done, 1):.2f}s/q fails {fails} "
                      f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f}GiB", flush=True)
        fout.close()
        timings[task] = {"seconds": round(time.time() - t0, 1), "rows_run": len(todo)}

    combined = []
    for task in tasks:
        p = out_dir / f"{TASK_SHORT[task]}.jsonl"
        if p.exists():
            combined += [json.loads(l) for l in open(p) if l.strip()]
    json.dump(combined, open(out_dir / "all_predictions.json", "w"), indent=1, ensure_ascii=False)
    if stats is not None:
        stats.save(str(out_dir / "routing_stats.npz"))
    meta.update(finished=datetime.datetime.now().isoformat(timespec="seconds"),
                hours=round((time.time() - t_grand) / 3600, 3), timings=timings, n_predictions=len(combined),
                peak_mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2))
    json.dump(meta, open(out_dir / "run_meta.json", "w"), indent=1)
    print(f"\nDONE {len(combined)} predictions in {meta['hours']} h -> {out_dir}/all_predictions.json", flush=True)


if __name__ == "__main__":
    main()
