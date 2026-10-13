"""Inference-side rendering for journal evaluation (PLAN.md §1.6).

Builds exactly what training saw (code/data/journal_format.py): the first 4 images, each with
an `Image-i:` marker, then the bare question; Qwen chat template with enable_thinking=False and
the generation prompt. ZS-prompt adds ZS_PROMPT_SYSTEM as a system turn; nothing else changes.
Images are resized with the same int-rounded area cap LF applies in training (768² → 589824 px).
template_gate.py checks train/inference byte identity with these functions.
"""
import math
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
from journal_format import (IMAGE_TOKEN, MAX_IMAGES, ZS_PROMPT_SYSTEM,  # noqa: E402,F401
                            capped_images, render_user_text)

IMAGE_MAX_PIXELS = 589824


def cap_pixels(im, max_pixels=IMAGE_MAX_PIXELS):
    """Same resize rule as LF's mm_plugin (int rounding), so image-pad counts match training."""
    w, h = im.size
    if max_pixels and w * h > max_pixels:
        f = math.sqrt(max_pixels / (w * h))
        im = im.resize((max(1, int(w * f)), max(1, int(h * f))))
    return im


def load_images(paths, max_pixels=IMAGE_MAX_PIXELS):
    return [cap_pixels(Image.open(p).convert("RGB"), max_pixels) for p in paths]


def chat_text(processor, question, n_images, system=None):
    """Prompt string for generation. `system` is None for ZS-bare and every adapter."""
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


def eos_ids(tokenizer):
    """PLAN.md §1.1: EOS pinned to [<|im_end|>, <|endoftext|>]."""
    ids = [tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.convert_tokens_to_ids("<|endoftext|>")]
    assert all(isinstance(i, int) and i >= 0 for i in ids), ids
    return ids
