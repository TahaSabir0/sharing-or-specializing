"""Unit test: Gemma3 (MedGemma) rendering for evaluation, and that Qwen's is untouched (PLAN.md §1.1, §1.6, W9).

CPU only. Needs Pillow (journal_render imports it). Two groups:
- pure checks with fake processors/tokenizers: prompt strings, call arguments, EOS ids, pixel rules,
  the token_type_ids rule. They also pin the Qwen path (same messages, same flat image list);
- real-processor checks, run only when MEDGEMMA_MODEL points at a folder with the Gemma3 processor
  files (config/tokenizer, no weights needed) AND llamafactory + torch + transformers import:
  the template gate on a synthetic deck (must pass), its two failure modes (the unpatched LF
  token_type_ids; a naive apply_chat_template prompt), batching and flat-vs-nested image lists.
  Without them these print SKIP; a SKIP is not a pass.

  MEDGEMMA_MODEL=<dir> PYTHONPATH=<LLaMA-Factory>/src python code/tests/test_gemma_render.py      (or pytest)
"""
import contextlib
import io
import os
import sys
import tempfile
from types import SimpleNamespace

CODE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(CODE, "eval"))
sys.path.insert(0, CODE)

try:
    from PIL import Image
except ImportError:
    print("SKIP test_gemma_render.py (no Pillow)")
    sys.exit(0)

import journal_render as jr  # noqa: E402
from methods import backbones as bb  # noqa: E402

BOI = "<start_of_image>"


class FakeTok:
    bos_token = "<bos>"
    ids = {"<end_of_turn>": 106, "<eos>": 1, "<|im_end|>": 5, "<|endoftext|>": 6}

    def convert_tokens_to_ids(self, t):
        return self.ids[t]


class FakeGemmaProc:
    boi_token = BOI
    tokenizer = FakeTok()


class FakeQwenProc:
    """Records the chat-template call; the Qwen path must keep calling it exactly this way."""

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kw):
        self.calls.append((messages, kw))
        return "QWEN-PROMPT"

    def __call__(self, **kw):
        self.calls.append(("call", kw))
        return "ENC"


# ------------------------------------------------------------------------------------ pure checks
def test_gemma_prompt_strings():
    p = FakeGemmaProc()
    assert jr.chat_text(p, "Q?", 1, backbone="mg15") == \
        f"<bos><start_of_turn>user\nImage-1: {BOI}\nQ?<end_of_turn>\n<start_of_turn>model\n"
    assert jr.chat_text(p, "Q?", 2, backbone="mg15") == \
        f"<bos><start_of_turn>user\nImage-1: {BOI}\nImage-2: {BOI}\nQ?<end_of_turn>\n<start_of_turn>model\n"
    s = jr.chat_text(p, "Q?\nmore", 1, system="SYS", backbone="mg15")      # system folds into the first user turn
    assert s == f"<bos><start_of_turn>user\nSYS\n\nImage-1: {BOI}\nQ?\nmore<end_of_turn>\n<start_of_turn>model\n"
    assert s.count("<bos>") == 1 and "<image>" not in s


def test_qwen_chat_text_unchanged():
    p = FakeQwenProc()
    assert jr.chat_text(p, "Q?", 2) == "QWEN-PROMPT"
    msgs, kw = p.calls[0]
    assert msgs == [{"role": "user", "content": [
        {"type": "text", "text": "Image-1: "}, {"type": "image"}, {"type": "text", "text": "\nImage-2: "},
        {"type": "image"}, {"type": "text", "text": "\nQ?"}]}]
    assert kw == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
    p2 = FakeQwenProc()
    jr.chat_text(p2, "Q?", 1, system="S")
    assert p2.calls[0][0][0] == {"role": "system", "content": "S"}


def test_encode_batch_arguments():
    imgs = [["a", "b"], ["c"]]
    q = FakeQwenProc()
    jr.encode_batch(q, ["t1", "t2"], imgs)                                   # default = q35: one flat list
    assert q.calls[0][1] == {"text": ["t1", "t2"], "images": ["a", "b", "c"], "padding": True, "return_tensors": "pt"}
    g = FakeQwenProc()
    jr.encode_batch(g, ["t1", "t2"], imgs, "mg15")                           # mg15: one list per prompt
    kw = g.calls[0][1]
    assert kw["images"] == imgs and kw["add_special_tokens"] is False and kw["do_pan_and_scan"] is False
    assert kw["padding"] is True and kw["return_tensors"] == "pt"


def test_eos_ids():
    t = FakeTok()
    assert jr.eos_ids(t) == [5, 6] and jr.eos_ids(t, "q35") == [5, 6]
    assert jr.eos_ids(t, "mg15") == [106, 1]


def test_pixel_rules():
    assert jr.image_rules("q35") == {"max_pixels": 589824, "min_pixels": None}
    assert jr.image_rules("mg15") == {"max_pixels": None, "min_pixels": 1024}
    assert bb.lf_overrides("mg15", "/m")["image_max_pixels"] > 8192 * 8192       # training side: no cap either
    big, tiny = Image.new("RGB", (3000, 2000)), Image.new("RGB", (20, 20))
    q = jr.cap_pixels(big)                                                      # Qwen: 768² area cap, as before
    assert q.size[0] * q.size[1] <= 589824 and q.size == (int(3000 * (589824 / 6e6) ** .5), int(2000 * (589824 / 6e6) ** .5))
    assert jr.cap_pixels(tiny).size == (20, 20)                                 # Qwen: no min rule
    assert jr.load_images([], **jr.image_rules("mg15")) == []
    g = jr.cap_pixels(big, **jr.image_rules("mg15"))
    assert g.size == (3000, 2000)                                               # Gemma: untouched, the processor resizes
    t = jr.cap_pixels(tiny, **jr.image_rules("mg15"))
    assert t.size == (32, 32)                                                   # LF's min-pixels rule (1024 px)


def test_token_type_rule():
    ids = [[2, 105, 255999, 262144, 262144, 256000, 7], [262144, 1]]
    assert bb.gemma3_token_type_ids(ids, 262144) == [[0, 0, 0, 1, 1, 0, 0], [1, 0]]


# ---------------------------------------------------------------------------- real-processor checks
def _real_env():
    d = os.environ.get("MEDGEMMA_MODEL")
    if not d or not os.path.exists(os.path.join(d, "processor_config.json")):
        return None, "MEDGEMMA_MODEL not set to a Gemma3 processor folder"
    try:
        import llamafactory.data.template  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception as e:  # noqa: BLE001
        return None, f"cannot import llamafactory/torch/transformers ({type(e).__name__}: {e})"
    return d, ""


def _deck(tmp):
    import json
    sys.path.insert(0, os.path.join(CODE, "data"))
    from journal_format import render_user_text

    def img(name, w, h, mode="RGB"):
        p = os.path.join(tmp, name + ".png")
        Image.new(mode, (w, h), (90, 40, 160) if mode == "RGB" else 128).save(p)
        return p
    rows = [("classification", "bcn", "Which?\nA. x\nB. y", "A", [img("a", 1024, 768)], 1),
            ("detection", "iugc", "Boxes?", "[[112, 87, 431, 390]]", [img("b", 3000, 2000)], 1),
            ("report_generation", "IU", "Report.", "Normal.  Clear.\nNone.", [img("c", 700, 700, "RGBA"), img("d", 512, 640)], 2),
            ("counting", "cell", "How many?", "37", [img("e", 20, 20)], 1),
            ("multi-label classification", "chestdr", "List.", "A; B", [img(f"m{i}", 800 + 40 * i, 600) for i in range(4)], 6)]
    path = os.path.join(tmp, "deck.jsonl")
    with open(path, "w") as f:
        for i, (task, ds, q, a, imgs, orig) in enumerate(rows):
            f.write(json.dumps(dict(
                id=f"s{i}", conversations=[{"from": "human", "value": render_user_text(q, len(imgs))},
                                           {"from": "gpt", "value": a}],
                images=imgs, question=q, answer=a, task=task, modality="x", dataset=ds,
                n_images_orig=orig)) + "\n")
    return path


def test_real_gate_passes_and_catches_the_known_failures():
    d, why = _real_env()
    if d is None:
        print(f"SKIP test_real_gate_passes_and_catches_the_known_failures ({why})")
        return
    import template_gate as tg
    with tempfile.TemporaryDirectory() as tmp:
        deck = _deck(tmp)
        # the LF patch is process-global and cannot be undone: the unpatched run must come first
        with contextlib.redirect_stdout(io.StringIO()):                              # expected FAIL lines
            unpatched = tg.run_gemma_gate([deck], model_dir=d, patched=False)       # LF's own token_type_ids: mismatch
        assert unpatched is False
        assert tg.run_gemma_gate([deck], model_dir=d, patched=True) is True
        # naive HF-template prompt (trimmed whitespace, double <bos>): the gate must reject it
        from journal_format import IMAGE_TOKEN, render_user_text
        good = jr._gemma3_chat_text

        def naive(processor, question, n_images, system=None):
            content, segs = [], render_user_text(question, n_images).split(IMAGE_TOKEN)
            for i, seg in enumerate(segs):
                if seg:
                    content.append({"type": "text", "text": seg})
                if i < len(segs) - 1:
                    content.append({"type": "image"})
            return processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                                 add_generation_prompt=True)
        jr._gemma3_chat_text = naive
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                naive_passes = tg.run_gemma_gate([deck], model_dir=d, patched=True)
            assert naive_passes is False
        finally:
            jr._gemma3_chat_text = good


def test_real_processor_flat_vs_nested_images():
    d, why = _real_env()
    if d is None:
        print(f"SKIP test_real_processor_flat_vs_nested_images ({why})")
        return
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(d)
    im = Image.new("RGB", (300, 200))
    t2 = jr.chat_text(proc, "Q", 2, backbone="mg15")
    t1 = jr.chat_text(proc, "Q", 1, backbone="mg15")
    b = jr.encode_batch(proc, [t2, t1], [[im, im], [im]], "mg15")
    assert tuple(b.pixel_values.shape) == (3, 3, 896, 896)
    try:
        proc(text=[t2, t1], images=[im, im, im], padding=True, add_special_tokens=False, return_tensors="pt")
    except ValueError:
        pass
    else:
        raise AssertionError("a flat image list across a 2-prompt batch should be rejected by Gemma3Processor")
    ids = b.input_ids[0].tolist()
    assert ids.count(proc.tokenizer.bos_token_id) == 1


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        fn()
        print("ok  ", name)
    print(f"{len(tests)} tests run (a SKIP line above means that test did not execute)")
