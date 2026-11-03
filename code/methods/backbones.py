"""Backbone registry: what differs between Qwen3.5-9B (q35) and MedGemma-1.5-4B (mg15).

Torch-free on purpose, so it can be unit-tested anywhere (code/tests/test_backbones.py) and
imported by both the trainer (train/journal_train.py) and the plug-in plumbing (methods/base.py).

- `lf_overrides(tag)`: the LLaMA-Factory keys that differ from journal_train's PROTOCOL (which
  stays the Qwen recipe). Empty for q35, so every Qwen run resolves exactly as before.
- `targets_path_for_model_type(model_type)`: which frozen target-module file a loaded model uses.
  Unknown model types fall back to the Qwen file (the behaviour before this module existed).
- `matches_target(name, spec)`: the one predicate that picks a target Linear by name.
"""
import json
import os
from dataclasses import dataclass

_HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(_HERE, "..", "train")

# MedGemma's image budget is native and fixed: the Gemma3 processor resizes every image to 896x896
# (256 tokens), so the 768² cap of the Qwen recipe cannot apply (PLAN.md §1.1). LF's own default
# cap is 768*768 and Gemma3Plugin would apply it silently, so the cap is set out of reach and the
# processor does the single resize to 896².
NO_PIXEL_CAP = 2 ** 40


@dataclass(frozen=True)
class Backbone:
    tag: str                 # run-id tag: j26_{tag}_{arm}_{data}_{res}_s{seed}
    res_tag: str             # resolution tag in the run id
    model_type: str          # HF config.model_type of the loaded model
    model_dirname: str       # folder under MODELS_DIR
    targets_file: str        # in code/train/


BACKBONES = {
    "q35": Backbone("q35", "768", "qwen3_5", "Qwen3.5-9B", "target_modules_qwen35.json"),
    "mg15": Backbone("mg15", "896", "gemma3", "medgemma-1.5-4b-it", "target_modules_medgemma15.json"),
}
DEFAULT_BACKBONE = "q35"
QWEN_TARGETS_PATH = os.path.join(TRAIN_DIR, BACKBONES["q35"].targets_file)


def get_backbone(tag):
    if tag not in BACKBONES:
        raise ValueError(f"unknown backbone {tag!r}; known: {sorted(BACKBONES)}")
    return BACKBONES[tag]


def targets_path(tag):
    return os.path.join(TRAIN_DIR, get_backbone(tag).targets_file)


def load_spec(tag):
    with open(targets_path(tag)) as f:
        return json.load(f)


def targets_path_for_model_type(model_type):
    """Target file for a loaded model's config.model_type; Qwen's file when unknown."""
    for bb in BACKBONES.values():
        if bb.model_type == model_type:
            return os.path.join(TRAIN_DIR, bb.targets_file)
    return QWEN_TARGETS_PATH


def matches_target(name, spec):
    """True for a qualified module name that is a LoRA target: leaf name in the frozen list and
    inside the language model (`scope_prefix`). The caller also requires an nn.Linear."""
    return name.rsplit(".", 1)[-1] in set(spec["target_modules"]) and spec["scope_prefix"] in name


def lf_overrides(tag, models_dir):
    """LF/HF keys that differ from journal_train.PROTOCOL (the Qwen recipe) for this backbone."""
    get_backbone(tag)
    if tag == "q35":
        return {}
    spec = load_spec(tag)
    return {
        "model_name_or_path": f"{models_dir}/{BACKBONES[tag].model_dirname}",
        "template": "gemma3",
        "image_max_pixels": NO_PIXEL_CAP,
        "image_do_pan_and_scan": False,          # pinned: Gemma3's optional crops stay off
        "lora_target": ",".join(spec["target_modules"]),
    }


def expected_p8(tag):
    return load_spec(tag)["P8"]


# ------------------------------------------------------------------ Gemma3 token_type_ids (LF patch)
def gemma3_token_type_ids(batch_ids, image_soft_token_id):
    """1 on every `<image_soft_token>` (the 256 per image), 0 elsewhere: what Gemma3Processor returns.

    The model builds its bidirectional attention blocks over images from `token_type_ids == 1`."""
    return [[1 if t == image_soft_token_id else 0 for t in ids] for ids in batch_ids]


def patch_lf_gemma3_token_type_ids():
    """Make LLaMA-Factory's training-time token_type_ids equal the processor's (inference) ones.

    LF v0.9.5 marks `processor.image_token_id`. Since transformers 5.x that attribute is the id of
    `<start_of_image>` (255999), not of `<image_soft_token>` (262144), so training would mark 1 token
    per image while Gemma3Processor marks all 256 soft tokens at inference: a different attention
    pattern over the image in training and evaluation. Verified with LF 7af9095 + transformers 5.16.1
    (code/eval/template_gate.py checks it on real rows). Gemma-only: Qwen never calls this."""
    from llamafactory.data import mm_plugin as MP

    def fixed(batch_ids, processor):
        tok = processor.tokenizer
        soft = tok.convert_tokens_to_ids(tok.image_token)
        if not isinstance(soft, int) or soft < 0 or soft == tok.unk_token_id:
            raise RuntimeError(f"cannot resolve the image soft token {tok.image_token!r}: {soft}")
        return gemma3_token_type_ids(batch_ids, soft)

    MP._get_gemma3_token_type_ids = fixed   # Gemma3Plugin.get_mm_inputs looks it up at call time
