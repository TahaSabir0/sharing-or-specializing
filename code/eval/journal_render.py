"""Inference-side rendering for journal evaluation (PLAN.md §1.6).

Builds exactly what training saw (code/data/journal_format.py): the first 4 images, each with
an `Image-i:` marker, then the bare question; the backbone's chat template with the generation
prompt. ZS-prompt adds ZS_PROMPT_SYSTEM as a system turn; nothing else changes.
template_gate.py checks train/inference identity with these functions.

Backbones (`backbone=`, default "q35" = the original behaviour of every function here):
- q35  Qwen3.5: chat template with enable_thinking=False; images resized with the same int-rounded
       area cap LF applies in training (768² → 589824 px); EOS [<|im_end|>, <|endoftext|>].
- mg15 MedGemma-1.5 (Gemma3): the prompt string is built by hand, not with apply_chat_template. The
       HF template trims every text part (it would turn "Image-1: <img>\\nQuestion" into
       "Image-1:<img>Question") and emits <bos>, which the tokenizer then adds a second time; LF's
       `gemma3` template does neither. The string is tokenized with add_special_tokens=False so
       there is exactly one <bos>. No area cap: the Gemma3 processor resizes every image to 896²
       itself (PLAN.md §1.1); only LF's own min-pixels rule is mirrored. EOS [<end_of_turn>, <eos>].
       The processor wants one list of images per prompt (a flat list is one prompt's images).
"""
import json
import math
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from journal_format import (IMAGE_TOKEN, MAX_IMAGES, ZS_PROMPT_SYSTEM,  # noqa: E402,F401
                            capped_images, render_user_text)

IMAGE_MAX_PIXELS = 589824

# Pre-resize rules per backbone, mirroring what LF's mm_plugin does before the processor in training.
# mg15: LF's default image_min_pixels (32*32) is the only rule; the cap is out of reach (backbones.py).
IMAGE_RULES = {"q35": {"max_pixels": IMAGE_MAX_PIXELS, "min_pixels": None},
               "mg15": {"max_pixels": None, "min_pixels": 32 * 32}}


def cap_pixels(im, max_pixels=IMAGE_MAX_PIXELS, min_pixels=None):
    """Same resize rules as LF's mm_plugin (int rounding), so image-pad counts match training."""
    w, h = im.size
    if max_pixels and w * h > max_pixels:
        f = math.sqrt(max_pixels / (w * h))
        im = im.resize((max(1, int(w * f)), max(1, int(h * f))))
    w, h = im.size
    if min_pixels and w * h < min_pixels:
        f = math.sqrt(min_pixels / (w * h))
        im = im.resize((max(1, int(w * f)), max(1, int(h * f))))
    return im


def load_images(paths, max_pixels=IMAGE_MAX_PIXELS, min_pixels=None):
    return [cap_pixels(Image.open(p).convert("RGB"), max_pixels, min_pixels) for p in paths]


def image_rules(backbone="q35"):
    """kwargs for load_images for this backbone."""
    return dict(IMAGE_RULES[backbone])


def _gemma3_chat_text(processor, question, n_images, system=None):
    """LF `gemma3` template, by hand: <bos><start_of_turn>user\\n{system\\n\\n}{human}<end_of_turn>\\n<start_of_turn>model\\n.

    Each <image> becomes the processor's own placeholder (<start_of_image>), which Gemma3Processor
    expands to "\\n\\n<start_of_image>" + 256 soft tokens + "<end_of_image>\\n\\n", the same string LF's
    Gemma3Plugin substitutes in training."""
    human = render_user_text(question, n_images).replace(IMAGE_TOKEN, processor.boi_token)
    sys_part = f"{system}\n\n" if system else ""
    return f"{processor.tokenizer.bos_token}<start_of_turn>user\n{sys_part}{human}<end_of_turn>\n<start_of_turn>model\n"


def chat_text(processor, question, n_images, system=None, backbone="q35"):
    """Prompt string for generation. `system` is None for ZS-bare and every adapter."""
    if backbone == "mg15":
        return _gemma3_chat_text(processor, question, n_images, system)
    human = render_user_text(question, n_images)
    content, segs = [], human.split(IMAGE_TOKEN)
    for i, seg in enumerate(segs):
        if seg:
            content.append({"type": "text", "text": seg})
        if i < len(segs) - 1:
            content.append({"type": "image"})
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content})
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)


def encode_batch(processor, texts, image_lists, backbone="q35"):
    """Tokenize + preprocess a batch: `texts[i]` goes with the images `image_lists[i]`."""
    if backbone == "mg15":
        return processor(text=texts, images=image_lists, padding=True, add_special_tokens=False,
                         do_pan_and_scan=False, return_tensors="pt")
    flat = [im for images in image_lists for im in images]
    return processor(text=texts, images=flat, padding=True, return_tensors="pt")


def eos_ids(tokenizer, backbone="q35"):
    """PLAN.md §1.1: EOS pinned. q35: [<|im_end|>, <|endoftext|>]; mg15: [<end_of_turn>, <eos>]
    (the ids of generation_config.eos_token_id, which also ships do_sample=true / top_k / top_p:
    the harness overrides all three)."""
    names = ("<end_of_turn>", "<eos>") if backbone == "mg15" else ("<|im_end|>", "<|endoftext|>")
    ids = [tokenizer.convert_tokens_to_ids(n) for n in names]
    assert all(isinstance(i, int) and i >= 0 for i in ids), ids
    return ids


def declared_stop_ids(model_dir):
    """The stop tokens the model itself declares: generation_config.json `eos_token_id`, sorted.

    Not the tokenizer's default `eos_token`: Google's MedGemma files have `<eos>` (id 1) there, while
    LF replaces it with `<end_of_turn>` in training, and generation uses the explicit list from
    eos_ids() either way. A missing key raises (a gate must not skip)."""
    with open(os.path.join(model_dir, "generation_config.json")) as f:
        eos = json.load(f)["eos_token_id"]
    return sorted(set([eos] if isinstance(eos, int) else eos))


def eos_contract(pinned, declared, target_ids):
    """Gate checks on the stopping tokens (template_gate.py, MedGemma):
    - the pinned set equals the set the model declares in its generation config;
    - the first pinned token, <end_of_turn>, is the token LF's gemma3 template ends the training
      target with (`...<end_of_turn>\\n`: the second-to-last target token), so what the model is trained
      to emit is what generation stops on."""
    return {"stop_tokens_match_generation_config": sorted(set(pinned)) == sorted(set(declared)),
            "target_ends_with_pinned_stop": len(target_ids) >= 2 and target_ids[-2] == pinned[0]}
