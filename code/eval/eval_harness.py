# ============================================================================================
# JOURNAL REPO NOTE (2026-10-11): copied from the challenge code (experiments/qwen35_zeroshot/zeroshot_truetest.py, uncommitted, Sep 22).
# It still holds CHALLENGE settings: TASK_CONFIG uses medical_cot/cot prompts, answer seeds and 2048² caps;
#   protocol v2 is the old format-assisted run (different budget/extractor), NOT ZS-prompt.
# Adapt it to PLAN.md §1 (bare prompts, 768² cap, max_new_tokens 512, both zero-shot protocols)
# before using it for any journal run. That is part of W3 (PLAN.md §3.1).
# ============================================================================================
"""Zero-shot (NO adapter) Qwen3.5-9B over the frozen true_test split, all scored tasks, batched.

Two protocols (see PROGRESS 2026-09-20 / stats/26):
  v1  "pipeline zero-shot"  = the shipped hexapus container with adapters disabled: same per-route
      prompt variant + image cap (TASK_CONFIG below mirrors docker_qwen35/inference.py), same answer
      seeds, same deployed parser (strip_think_block -> parse_answer). Comparable to ME-VLIP's "base".
  v2  "format-assisted zero-shot" = v1 prompt + an explicit 'Final answer: ...' suffix, NO answer
      seed (the model may reason), and a forgiving extractor that reads the final-answer line
      (fallback: deployed parser). Lower bound on base capability. Report generation is skipped
      (free text: v2 == v1 by construction).

Decode: greedy, enable_thinking=False, eos pinned to {<|im_end|>, eos}, max_new_tokens 512,
nf4 double_quant=False (matches the sprint evals). Batched generation with LEFT padding; identical
greedy tokens per sequence up to padding numerics. Resumable: per-task jsonl checkpoints.

  CUDA_VISIBLE_DEVICES=2 python zeroshot_truetest.py --protocol v1 --out_dir /data/ugrip2026/zeroshot/v1
"""
import os, sys, json, time, math, argparse, re, ast, collections, traceback
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

# journal repo: helper modules (vlm_prompt, infer_single_adapter, prompt_variants, format_templates) sit next to this file
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_prompt import strip_think_block                      # noqa: E402
from infer_single_adapter import parse_answer, subsample      # noqa: E402
from prompt_variants import get_variant                       # noqa: E402
from format_templates import normalize_task_type, question_has_letter_options  # noqa: E402

PIX_2048 = 4194304
PIX_768 = 589824
# task (split spelling) -> (prompt variant, image cap)  == shipped hexapus:latest routes (stats/26 §4)
TASK_CONFIG = {
    "classification":            ("none",        PIX_2048),
    "multi-label classification": ("medical_cot", PIX_2048),
    "counting":                  ("cot",         PIX_2048),
    "detection":                 ("none",        PIX_2048),
    "regression":                ("none",        PIX_2048),
    "report_generation":         ("none",        PIX_768),
}
TASK_ORDER = ["counting", "detection", "regression", "multi-label classification",
              "classification", "report_generation"]

FA = "Final answer:"


def v2_suffix(task, question):
    t = normalize_task_type(task)
    lead = "\n\nYou may reason briefly first. End your response with one final line of the form "
    if t == "classification":
        if question_has_letter_options(question):
            return lead + f"'{FA} X' where X is the option letter only."
        return lead + f"'{FA} X' where X is the exact class name copied from the question."
    if t == "multi-label classification":
        return (lead + f"'{FA} label1; label2' listing only the findings that are present, copied "
                "exactly from the options in the question and separated by semicolons. If no listed "
                "finding is present, use the option that denotes a normal study if one is listed, "
                f"otherwise write '{FA} None'.")
    if t in ("counting", "cell counting"):
        return lead + f"'{FA} N' where N is a single integer."
    if t == "regression":
        return lead + f"'{FA} N' where N is a single number without units."
    if t in ("detection", "instance detection"):
        return (lead + f"'{FA} [[x_min, y_min, x_max, y_max], ...]' with one box per detected object "
                f"in pixel coordinates of the original image. Write '{FA} []' if there is none.")
    return ""


def _clean(s):
    s = s.strip().strip("*_` ").strip()
    s = re.sub(r"^\*+|\*+$", "", s).strip()
    return s.rstrip(".").strip()


def _first_balanced_list(s):
    i = s.find("[")
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "[":
            depth += 1
        elif s[j] == "]":
            depth -= 1
            if depth == 0:
                return s[i:j + 1]
    return None


def v2_extract(raw, task, question):
    """Read the last 'Final answer:' line; normalise per task; fallback = deployed parser."""
    text = strip_think_block(raw)
    hits = list(re.finditer(re.escape(FA), text, flags=re.I))
    if not hits:
        return parse_answer(text, task), False
    tail = text[hits[-1].end():].strip()
    line = _clean(tail.splitlines()[0]) if tail else ""
    t = normalize_task_type(task)
    if t == "classification":
        if question_has_letter_options(question):
            m = re.match(r"\(?([A-G])(?:[).:\s]|$)", line)
            if m:
                return m.group(1), True
            m = re.search(r"\b([A-G])\b", line)
            return (m.group(1) if m else line), True
        return line, True
    if t == "multi-label classification":
        labels = [_clean(x) for x in re.split(r"[;,]", line) if _clean(x)]
        labels = [x for x in labels if x.lower() not in ("none", "no findings", "n/a")]
        return "; ".join(labels), True
    if t in ("counting", "cell counting", "regression"):
        m = re.search(r"[-+]?\d[\d,]*\.?\d*", line.replace(",", ""))
        return (m.group(0) if m else "0"), True
    if t in ("detection", "instance detection"):
        blob = _first_balanced_list(tail)
        if blob is None:
            return "[]", True
        try:
            v = ast.literal_eval(blob)
            if isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v) and len(v) == 4:
                v = [v]
            return json.dumps(v), True
        except Exception:
            return "[]", True
    return line, True


def cap_pixels(im, mx):
    w, h = im.size
    if mx and w * h > mx:
        f = math.sqrt(mx / (w * h))
        im = im.resize((max(1, int(w * f)), max(1, int(h * f))))
    return im


def build(proc, variant, task, question, n_imgs, protocol):
    parts = ["\n".join(f"Image-{i+1}: <image>" for i in range(n_imgs))]
    seed = ""
    if variant is not None:
        fi = variant.get_format_instruction(task, question)
        if fi:
            parts.append(fi)
        seed = variant.answer_seeds.get(normalize_task_type(task), "")
    q = question + (v2_suffix(task, question) if protocol == "v2" else "")
    parts.append(q)
    human = "\n".join(parts)
    content = []
    segs = human.split("<image>")
    for i, seg in enumerate(segs):
        if seg:
            content.append({"type": "text", "text": seg})
        if i < len(segs) - 1:
            content.append({"type": "image"})
    messages = []
    if variant is not None and variant.system_prompt:
        messages.append({"role": "system", "content": variant.system_prompt})
    messages.append({"role": "user", "content": content})
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                    enable_thinking=False)
    if protocol == "v1" and seed:
        text += seed
    return text, (seed if protocol == "v1" else "")


def make_batches(rows, max_rows, max_imgs):
    # group by image count so padding stays small; cap total images per batch
    rows = sorted(rows, key=lambda r: (len(subsample(r["image"])), len(r["question"])))
    batches, cur, cur_imgs = [], [], 0
    for r in rows:
        n = len(subsample(r["image"]))
        if cur and (len(cur) >= max_rows or cur_imgs + n > max_imgs):
            batches.append(cur); cur, cur_imgs = [], 0
        cur.append(r); cur_imgs += n
    if cur:
        batches.append(cur)
    return batches


def run_batch(model, proc, eos_ids, pad_id, texts, imgs_flat, max_new_tokens):
    inputs = proc(text=texts, images=imgs_flat, padding=True, return_tensors="pt").to("cuda:0")
    n_in = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             eos_token_id=eos_ids, pad_token_id=pad_id)
    return proc.batch_decode(gen[:, n_in:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True, choices=["v1", "v2"])
    ap.add_argument("--base_model", default="/data/ugrip2026/models/Qwen3.5-9B")
    ap.add_argument("--data_path", default="/data/ugrip2026/splits/internal_splits_v2/true_test.jsonl")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tasks", default="", help="comma list; default = all scored tasks")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_images_per_batch", type=int, default=12)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N rows per task")
    ap.add_argument("--image_max_pixels", type=int, default=0,
                    help="override the per-route image cap (e.g. 589824 for a 768-trained adapter); "
                         "0 = use the shipped TASK_CONFIG cap")
    ap.add_argument("--adapter_path", default="", help="optional LoRA adapter (PEFT dir) -> fine-tuned pipeline eval "
                    "with the same batched harness; use with --protocol v1 (shipped prompts/seeds/parser)")
    args = ap.parse_args()

    for v in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        os.environ.setdefault(v, "1")
    os.makedirs(args.out_dir, exist_ok=True)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()] or list(TASK_ORDER)
    if args.protocol == "v2" and "report_generation" in tasks and not args.tasks:
        tasks.remove("report_generation")   # free text: identical to v1, don't pay for it twice

    proc = AutoProcessor.from_pretrained(args.base_model, local_files_only=True)
    proc.tokenizer.padding_side = "left"
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=False)
    model = AutoModelForImageTextToText.from_pretrained(
        args.base_model, quantization_config=bnb, dtype=torch.bfloat16, device_map="cuda:0",
        local_files_only=True)
    if args.adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter_path, is_trainable=False)
    model.eval()
    tok = proc.tokenizer
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    eos_ids = sorted({im_end, tok.eos_token_id, tok.convert_tokens_to_ids("<|endoftext|>")} - {None})
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else im_end
    print(f"[{args.protocol}] adapter={args.adapter_path or 'NONE (zero-shot)'}; eos_token_id={eos_ids} pad={pad_id} "
          f"padding_side={tok.padding_side}", flush=True)

    all_rows = [json.loads(l) for l in open(args.data_path) if l.strip()]
    all_rows = [r for r in all_rows if not r.get("flagged_bug")]

    grand_t0 = time.time()
    for task in tasks:
        variant_name, max_pix = TASK_CONFIG[task]
        if args.image_max_pixels:
            max_pix = args.image_max_pixels
        variant = None if variant_name == "none" else get_variant(variant_name)
        rows = [r for r in all_rows if r["task"].strip().lower() == task.lower()]
        if args.limit:
            rows = rows[:args.limit]
        ckpt = Path(args.out_dir) / f"{task.replace(' ', '_')}.jsonl"
        done = {}
        if ckpt.exists():
            for l in open(ckpt):
                if l.strip():
                    d = json.loads(l); done[d["sample_key"]] = d
        todo = [r for r in rows if r["sample_key"] not in done]
        print(f"\n=== task {task!r}: {len(rows)} rows, {len(done)} done, {len(todo)} to run | "
              f"variant={variant_name} cap={max_pix}", flush=True)
        batches = make_batches(todo, args.batch_size, args.max_images_per_batch)
        t0, n_done, fails = time.time(), 0, 0
        fout = open(ckpt, "a")

        def process(batch, bs_note=""):
            nonlocal n_done, fails
            texts, imgs_flat, metas = [], [], []
            for r in batch:
                paths = subsample(r["image"])
                ims = [Image.open(p).convert("RGB") for p in paths]
                scales = []
                capped = []
                for im in ims:
                    w0, h0 = im.size
                    c = cap_pixels(im, max_pix)
                    scales.append(round(c.size[0] / w0, 6))
                    capped.append(c)
                text, seed = build(proc, variant, r["task"], r["question"], len(capped), args.protocol)
                texts.append(text); imgs_flat.extend(capped)
                metas.append((r, seed, scales))
            tb = time.time()
            raws = run_batch(model, proc, eos_ids, pad_id, texts, imgs_flat, args.max_new_tokens)
            dt = time.time() - tb
            for (r, seed, scales), raw in zip(metas, raws):
                full = seed + raw
                stripped = strip_think_block(full)
                if args.protocol == "v1":
                    ans, hit = parse_answer(stripped, r["task"]), None
                else:
                    ans, hit = v2_extract(full, r["task"], r["question"])
                rec = dict(sample_key=r["sample_key"], ImageName=r["image"], Question=r["question"],
                           TaskType=r["task"], dataset=r["dataset"], gt_answer=r.get("answer", ""),
                           raw_output=full, Answer=ans, final_answer_line_found=hit,
                           n_new_tokens=len(tok(raw)["input_ids"]), image_scale=scales,
                           batch_seconds=round(dt, 3), batch_size=len(batch))
                fout.write(json.dumps(rec) + "\n")
            fout.flush()
            n_done += len(batch)

        for bi, batch in enumerate(batches):
            try:
                process(batch)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  OOM on batch of {len(batch)} -> retry singly", flush=True)
                for r in batch:
                    try:
                        process([r])
                    except Exception as e:
                        fails += 1
                        fout.write(json.dumps(dict(sample_key=r["sample_key"], ImageName=r["image"],
                                   Question=r["question"], TaskType=r["task"], dataset=r["dataset"],
                                   gt_answer=r.get("answer", ""), raw_output="", Answer=f"Error: {e}",
                                   final_answer_line_found=None)) + "\n"); fout.flush()
                        torch.cuda.empty_cache()
            except Exception as e:
                fails += len(batch)
                traceback.print_exc()
                for r in batch:
                    fout.write(json.dumps(dict(sample_key=r["sample_key"], ImageName=r["image"],
                               Question=r["question"], TaskType=r["task"], dataset=r["dataset"],
                               gt_answer=r.get("answer", ""), raw_output="", Answer=f"Error: {e}",
                               final_answer_line_found=None)) + "\n"); fout.flush()
            if (bi + 1) % 5 == 0 or bi == len(batches) - 1:
                el = time.time() - t0
                rate = el / max(n_done, 1)
                eta = rate * (len(todo) - n_done)
                print(f"  [{task}] {n_done}/{len(todo)}  {rate:.2f}s/case  elapsed {el/60:.1f}m  "
                      f"ETA {eta/60:.1f}m  fails {fails}  peak {torch.cuda.max_memory_allocated()/2**30:.1f}GiB",
                      flush=True)
        fout.close()
        # materialise the per-task predictions json (score_single format)
        recs = [json.loads(l) for l in open(ckpt) if l.strip()]
        json.dump(recs, open(Path(args.out_dir) / f"{task.replace(' ', '_')}_predictions.json", "w"), indent=1)
        print(f"  [{task}] DONE {len(recs)} rows in {(time.time()-t0)/60:.1f} min", flush=True)

    # combined predictions for the non-reportgen scorer
    combined = []
    for task in tasks:
        p = Path(args.out_dir) / f"{task.replace(' ', '_')}.jsonl"
        if p.exists():
            combined += [json.loads(l) for l in open(p) if l.strip()]
    json.dump(combined, open(Path(args.out_dir) / "all_predictions.json", "w"), indent=1)
    print(f"\nALL DONE [{args.protocol}] {len(combined)} rows, {(time.time()-grand_t0)/3600:.2f} h -> "
          f"{args.out_dir}/all_predictions.json", flush=True)


if __name__ == "__main__":
    main()
