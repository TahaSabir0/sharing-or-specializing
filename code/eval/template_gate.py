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

  python code/eval/template_gate.py --backbone mg15 /data/ugrip2026/journal/data/j26_train_mix.jsonl
MedGemma (Gemma3), same two render paths, against LF's real `gemma3` template and Gemma3Plugin and
the real Gemma3Processor (needs LF importable, the processor files, and the deck's images). It checks,
per row: prompt token ids identical; exactly one <bos>; 256 image soft tokens (+ one <start_of_image>
and <end_of_image>) per image on both sides; the prompt ends at "<start_of_turn>model\n"; the training
target is the deck's answer verbatim + "<end_of_turn>\n" (detection coordinates untouched); the pinned
stop tokens equal the model's own generation_config.eos_token_id and the first one ends the target
(not the tokenizer's default eos_token, which is <eos> in Google's files); pixel_values bit-identical
(training path = LF's own image loading, inference path = journal_render.load_images + the
processor); token_type_ids identical, with the LF patch journal_train
installs (backbones.patch_lf_gemma3_token_type_ids; --unpatched shows the unpatched mismatch). Then, once
per deck, a left-padded multi-prompt batch must reproduce every single-prompt row, and the ZS-prompt
system fold must match LF's system handling. Text-template equality (ids) and full multimodal
preprocessing consistency (pixels, token types, batching) are reported as separate check groups.
It never skips: a missing dependency is an error, not a pass.
"""
import argparse
import json
import os
import re
import sys

from transformers import AutoProcessor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from journal_render import (IMAGE_MAX_PIXELS, ZS_PROMPT_SYSTEM, capped_images, chat_text,  # noqa: E402
                            declared_stop_ids, encode_batch, eos_contract, eos_ids, image_rules,
                            load_images)
from methods import backbones  # noqa: E402

MODELS_DIR = os.environ.get("MODELS_DIR", "/data/ugrip2026/models")
MODEL = os.environ.get("QWEN35_MODEL", f"{MODELS_DIR}/Qwen3.5-9B")
MEDGEMMA = os.environ.get("MEDGEMMA_MODEL", f"{MODELS_DIR}/medgemma-1.5-4b-it")
LF_MIN_PIXELS = 32 * 32   # LF's default image_min_pixels (hparams/model_args.py); journal_train does not set it


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


# ------------------------------------------------------------------------------------- MedGemma
def gemma_setup(model_dir, patched=True):
    """The processor as LF's training run configures it, LF's gemma3 template, and (like journal_train
    for mg15) the token_type_ids patch."""
    from llamafactory.data.template import TEMPLATES
    processor = AutoProcessor.from_pretrained(model_dir)
    ov = backbones.lf_overrides("mg15", MODELS_DIR)
    processor.image_max_pixels = ov["image_max_pixels"]          # what LF's patch_processor sets
    processor.image_min_pixels = LF_MIN_PIXELS
    processor.image_do_pan_and_scan = ov["image_do_pan_and_scan"]
    processor.crop_to_patches = False
    processor.tokenizer.padding_side = "left"                    # as eval_harness sets it
    if patched:
        backbones.patch_lf_gemma3_token_type_ids()
    return processor, TEMPLATES["gemma3"]


def _gemma_lf(r, processor, template, system=None):
    """Training render of a deck row: LF template ids + LF's multimodal inputs."""
    human, gpt = r["conversations"][0]["value"], r["conversations"][1]["value"]
    imgs = r["images"]
    msgs = [{"role": "user", "content": human}, {"role": "assistant", "content": gpt}]
    mm = template.mm_plugin.process_messages(msgs, imgs, [], [], processor)
    p_ids, r_ids = template.encode_oneturn(processor.tokenizer, mm, system, None)
    mmi = template.mm_plugin.get_mm_inputs(imgs, [], [], [len(imgs)], [0], [0], [p_ids + r_ids], processor)
    return p_ids, r_ids, mmi


def _gemma_inf(r, processor, system=None):
    """Inference render of the same row from its raw question and image paths."""
    paths = capped_images({"image": r["images"]})
    text = chat_text(processor, r["question"], len(paths), system=system, backbone="mg15")
    imgs = load_images(paths, **image_rules("mg15"))
    return encode_batch(processor, [text], [imgs], "mg15")


def collapse_gemma(s):
    return re.sub(r"(<image_soft_token>)+", "<image_soft_token>xN", s)


def check_row_gemma(r, processor, template, declared_stop):
    import torch
    tok = processor.tokenizer
    gpt = r["conversations"][1]["value"]
    n = len(r["images"])
    p_ids, r_ids, mmi = _gemma_lf(r, processor, template)
    enc = _gemma_inf(r, processor)
    inf_ids = enc.input_ids[0].tolist()
    bos, eot = tok.bos_token_id, tok.convert_tokens_to_ids("<end_of_turn>")
    soft, boi, eoi = (tok.convert_tokens_to_ids(t) for t in ("<image_soft_token>", "<start_of_image>", "<end_of_image>"))
    lf_prompt, inf_prompt = tok.decode(p_ids), tok.decode(inf_ids)
    lf_tt = [int(x) for x in mmi["token_type_ids"][0][:len(p_ids)]]
    inf_tt = enc.token_type_ids[0].tolist()
    text_ok = {
        "ids_identical": p_ids == inf_ids,
        "single_bos": all(x[:1] == [bos] and x.count(bos) == 1 for x in (p_ids, inf_ids)),
        "image_tokens": all(x.count(soft) == 256 * n and x.count(boi) == n and x.count(eoi) == n
                            for x in (p_ids, inf_ids)),
        "generation_boundary": all(x.endswith("<start_of_turn>model\n") and x.count("<end_of_turn>") == 1
                                   for x in (lf_prompt, inf_prompt)),
        "target_verbatim": tok.decode(r_ids) == gpt + "<end_of_turn>\n" and r_ids[-2] == eot,
        **eos_contract(eos_ids(tok, "mg15"), declared_stop, r_ids),
    }
    mm_ok = {
        "pixel_values_equal": torch.equal(torch.as_tensor(mmi["pixel_values"]), enc.pixel_values),
        "token_type_ids_equal": lf_tt == inf_tt and sum(inf_tt) == 256 * n,
    }
    return text_ok, mm_ok, collapse_gemma(lf_prompt), collapse_gemma(inf_prompt), tok.decode(r_ids)


def check_batch_gemma(rows, processor):
    """A left-padded batch of several prompts (mixed image counts) must reproduce each single row."""
    import torch
    tok = processor.tokenizer
    singles = [_gemma_inf(r, processor) for r in rows]
    texts, lists = [], []
    for r in rows:
        paths = capped_images({"image": r["images"]})
        texts.append(chat_text(processor, r["question"], len(paths), system=None, backbone="mg15"))
        lists.append(load_images(paths, **image_rules("mg15")))
    b = encode_batch(processor, texts, lists, "mg15")
    ok = {"ids": True, "token_type_ids": True, "left_padded": True}
    for i, s in enumerate(singles):
        m = b.attention_mask[i].bool()
        ok["ids"] &= b.input_ids[i][m].tolist() == s.input_ids[0].tolist()
        ok["token_type_ids"] &= b.token_type_ids[i][m].tolist() == s.token_type_ids[0].tolist()
        ok["left_padded"] &= bool((b.attention_mask[i].diff() >= 0).all())
    ok["pixel_values"] = torch.equal(b.pixel_values, torch.cat([s.pixel_values for s in singles]))
    ok["pad_is_pad_token"] = bool((b.input_ids[b.attention_mask == 0] == tok.pad_token_id).all())
    return ok


def check_zs_system_gemma(r, processor, template):
    """ZS-prompt: LF's system handling (folded into the first user turn) == the hand-built string."""
    p_ids, _, _ = _gemma_lf(r, processor, template, system=ZS_PROMPT_SYSTEM)
    return p_ids == _gemma_inf(r, processor, system=ZS_PROMPT_SYSTEM).input_ids[0].tolist()


def run_gemma_gate(decks, model_dir=None, verbose=False, patched=True):
    """Returns True only if every check ran and passed."""
    model_dir = model_dir or MEDGEMMA
    processor, template = gemma_setup(model_dir, patched=patched)
    declared = declared_stop_ids(model_dir)
    tok = processor.tokenizer
    print(f"stop tokens: pinned {eos_ids(tok, 'mg15')} | generation_config {declared} | tokenizer default "
          f"eos_token {tok.eos_token!r} (id {tok.eos_token_id}; informational: LF sets it to <end_of_turn> in "
          f"training, generation passes the pinned list explicitly)")
    all_ok = True
    for deck in decks:
        rows = pick_rows(deck)
        if not rows:
            print(f"FAIL {os.path.basename(deck)}: no rows picked")
            all_ok = False
            continue
        for r in rows:
            text_ok, mm_ok, lf_p, inf_p, resp = check_row_gemma(r, processor, template, declared)
            passed = all(text_ok.values()) and all(mm_ok.values())
            all_ok &= passed
            tag = f"{os.path.basename(deck)} | {r['task']}/{r['dataset']} | images={len(r['images'])}"
            print(("PASS " if passed else "FAIL ") + tag)
            print("     text-template :", json.dumps(text_ok))
            print("     multimodal    :", json.dumps(mm_ok))
            if verbose or not passed:
                print("  LF prompt :", repr(lf_p[-300:]))
                print("  INF prompt:", repr(inf_p[-300:]))
                print("  LF target :", repr(resp))
        by_n = {}
        for r in rows:
            by_n.setdefault(len(r["images"]), r)
        batch_rows = list(by_n.values()) + [r for r in rows if r not in by_n.values()][:2]
        bok = check_batch_gemma(batch_rows, processor)
        zs = check_zs_system_gemma(rows[0], processor, template)
        all_ok &= all(bok.values()) and zs
        print(("PASS " if all(bok.values()) and zs else "FAIL ")
              + f"{os.path.basename(deck)} | batch of {len(batch_rows)} (images per row "
              f"{[len(r['images']) for r in batch_rows]}) {json.dumps(bok)} | zs_system_fold={zs}")
    return all_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("decks", nargs="+")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--backbone", choices=sorted(backbones.BACKBONES), default=backbones.DEFAULT_BACKBONE)
    ap.add_argument("--unpatched", action="store_true", help="mg15: skip journal_train's token_type_ids patch")
    a = ap.parse_args()
    if a.backbone == "mg15":
        ok = run_gemma_gate(a.decks, verbose=a.verbose, patched=not a.unpatched)
        print("GATE_PASS" if ok else "GATE_FAIL")
        sys.exit(0 if ok else 1)
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
