"""Single source of truth for how a journal row is rendered (PLAN.md §1.6).

Training decks (build_journal_decks.py) and every evaluation script import from here, so the
train and inference prompts are byte-identical by construction (smoke gate item 3).

Bare prompt: up to MAX_IMAGES images, each preceded by an `Image-i:` marker, then the
dataset's question text as given. No system prompt, no instruction. The target is the
dataset's answer string.
"""
import json
import os

MAX_IMAGES = 4
IMAGE_TOKEN = "<image>"

# ZS-prompt protocol only (PLAN.md §1.6). Never used in training.
ZS_PROMPT_SYSTEM = ("Answer the question about the medical image(s). "
                    "Return only the requested answer, without an explanation.")

# Excluded everywhere (PLAN.md §1.6): instance detection (unscored) and endoscopy.
EXCLUDED_TASKS = {"instance_detection"}
EXCLUDED_DATASETS = {"endo"}

LABEL_IDS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "label_ids.json")


def keep_row(rec):
    """The journal row filter: drop flagged, instance detection and endoscopy."""
    return (not rec.get("flagged_bug")
            and rec["task"] not in EXCLUDED_TASKS
            and rec["dataset"] not in EXCLUDED_DATASETS)


def image_group(rec):
    """The split grouping key: sorted image paths relative to the dataset root (questions on the
    same image(s) share it)."""
    return "|".join(sorted(p.split("FLARE-MLLM-2D/", 1)[-1] for p in rec["image"]))


def capped_images(rec):
    """First MAX_IMAGES images in their original order (truncates only the long tail)."""
    return list(rec["image"])[:MAX_IMAGES]


def render_user_text(question, n_images):
    """The user turn text with image placeholders. Image-i markers on every image."""
    parts = [f"Image-{i + 1}: {IMAGE_TOKEN}" for i in range(n_images)]
    parts.append(question)
    return "\n".join(parts)


def load_label_ids(path=LABEL_IDS_PATH):
    with open(path) as f:
        return json.load(f)


def to_row(rec, label_ids):
    """Split record -> journal deck row (LF ShareGPT columns + routing labels)."""
    images = capped_images(rec)
    return {
        "id": rec["sample_key"],
        "image_group": image_group(rec),
        "conversations": [
            {"from": "human", "value": render_user_text(rec["question"], len(images))},
            {"from": "gpt", "value": str(rec["answer"])},
        ],
        "images": images,
        "question": rec["question"],
        "answer": str(rec["answer"]),
        "task": rec["task"],
        "modality": rec["modality"],
        "dataset": rec["dataset"],
        "task_id": label_ids["task"][rec["task"]],
        "modality_id": label_ids["modality"][rec["modality"]],
        "dataset_id": label_ids["dataset"][rec["dataset"]],
        "n_images_orig": len(rec["image"]),
    }
