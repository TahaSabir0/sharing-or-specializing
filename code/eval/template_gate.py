"""Template identity gate (smoke gate item 3, PLAN.md §3.3), journal version.

For sample rows of each journal deck, renders the prompt two ways and byte-compares token ids:
  (a) training: LLaMA-Factory v0.9.5 `qwen3_5` template exactly as SFT sees it
      (mm_plugin.process_messages + encode_oneturn, enable_thinking=False, 768² cap);
  (b) inference: code/eval/journal_render.py (what every journal evaluator uses), built from the
      row's raw question and image paths, not from the deck's rendered text.
Rows checked per deck: the first row of every task present, plus the first multi-image row and
the first row whose question had more than 4 images.

  python code/eval/template_gate.py /data/ugrip2026/journal/data/j26_train_mix.jsonl [more decks]
Prints GATE_PASS only if every checked row has identical prompt ids, identical image-pad counts,
the empty think block in the prompt, and no think block in the training target.
(Adapted from the challenge template_gate.py: same two render paths, now journal-format rows.)
"""
import argparse
import json
import os
import re
import sys

from transformers import AutoProcessor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from journal_render import IMAGE_MAX_PIXELS, capped_images, chat_text, load_images  # noqa: E402

MODEL = os.environ.get("QWEN35_MODEL", "/data/ugrip2026/models/Qwen3.5-9B")


def pick_rows(path, max_scan=50000):
    chosen, tasks, multi, over4 = [], set(), False, False
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= max_scan:
                break
            r = json.loads(line)
            take = False
            if r["task"] not in tasks:
                tasks.add(r["task"]); take = True
            if not multi and len(r["images"]) > 1:
                multi = True; take = True
            if not over4 and r.get("n_images_orig", 0) > 4:
                over4 = True; take = True
            if take:
                chosen.append(r)
    return chosen


def collapse(s):
    return re.sub(r"(<\|image_pad\|>)+", "<|image_pad|>xN", s)


def check_row(r, processor, template):
    tok = processor.tokenizer
    human, gpt = r["conversations"][0]["value"], r["conversations"][1]["value"]
    imgs = r["images"]
    # (a) LF training render
    processor.image_max_pixels = IMAGE_MAX_PIXELS
    msgs = [{"role": "user", "content": human}, {"role": "assistant", "content": gpt}]
    mm = template.mm_plugin.process_messages(msgs, imgs, [], [], processor)
    p_ids, r_ids = template.encode_oneturn(tok, mm, None, None)
    lf_prompt, lf_resp = tok.decode(p_ids), tok.decode(r_ids)
    # (b) journal inference render from the raw question + paths
    paths = capped_images({"image": r["images"]})  # deck rows are already capped; idempotent
    text = chat_text(processor, r["question"], len(paths), system=None)
    enc = processor(text=[text], images=load_images(paths), return_tensors="pt")
    inf_ids = enc.input_ids[0].tolist()
    inf_prompt = tok.decode(inf_ids)
    ok = {
        "ids_identical": p_ids == inf_ids,
        "pads_equal": lf_prompt.count("<|image_pad|>") == inf_prompt.count("<|image_pad|>"),
        "think_in_prompt": "<think>" in lf_prompt,
        "no_think_in_target": "<think>" not in lf_resp,
        "target_ends_im_end": lf_resp.rstrip().endswith("<|im_end|>"),
    }
    return ok, collapse(lf_prompt), collapse(inf_prompt), lf_resp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("decks", nargs="+")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    from llamafactory.data.template import TEMPLATES
    processor = AutoProcessor.from_pretrained(MODEL)
    template = TEMPLATES["qwen3_5"]
    template.enable_thinking = False
    all_ok = True
    for deck in a.decks:
        for r in pick_rows(deck):
            ok, lf_p, inf_p, resp = check_row(r, processor, template)
            passed = all(ok.values())
            all_ok &= passed
            tag = f"{os.path.basename(deck)} | {r['task']}/{r['dataset']} | images={len(r['images'])}"
            print(("PASS " if passed else "FAIL ") + tag + " | " + json.dumps(ok))
            if a.verbose or not passed:
                print("  LF prompt :", repr(lf_p[-400:]))
                print("  INF prompt:", repr(inf_p[-400:]))
                print("  LF target :", repr(resp))
    print("GATE_PASS" if all_ok else "GATE_FAIL")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
