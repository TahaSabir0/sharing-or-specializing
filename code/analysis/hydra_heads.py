#!/usr/bin/env python3
"""E7: per-target cosine similarity of saved Hydra heads (CPU, no model loading).

Reads an existing completed checkpoint; never evaluates images or touches holdout.
Reports B_t for Hydra-T and A_t for Reversed Hydra, in frozen task-id order.
Zero-norm heads have undefined cosine (JSON null), not a fabricated zero similarity.
No averaging of unrelated target matrices or claim of statistical significance.
"""
import argparse
import json
import re
from pathlib import Path

import torch
from safetensors import safe_open

CODE = Path(__file__).resolve().parents[1]


def _metadata(run_dir):
    meta = json.loads((run_dir / "method_config.json").read_text())
    if meta["method"] != "hydra":
        raise ValueError("expected a Hydra plug-in checkpoint")
    mode = meta["cfg"].get("mode", "hydraT")
    if mode not in ("hydraT", "hydraR"):
        raise ValueError(f"unsupported Hydra mode: {mode}")
    factor = "B" if mode == "hydraT" else "A"
    task_map = json.loads((CODE / "data" / "label_ids.json").read_text())["task"]
    tasks = [name for name, _ in sorted(task_map.items(), key=lambda item: item[1])]
    return mode, factor, tasks


def _head_index(state, factor, n_tasks):
    pattern = re.compile(rf"^(.*)\.lora_{factor}\.(\d+)\.weight$")
    modules = {}
    for key in state.keys():
        match = pattern.match(key)
        if match:
            modules.setdefault(match[1], {})[int(match[2])] = key
    if not modules:
        raise ValueError("checkpoint has no routed Hydra heads")
    for name, heads in modules.items():
        if set(heads) != set(range(n_tasks)):
            raise ValueError(f"{name}: incomplete or unexpected task heads")
    return modules


def head_mapping(run_dir, group_by="target"):
    """Explicit parameter map accepted by Alex's representations.py heads command.

    target: one comparison per target matrix; decoder_layer: concatenate the
    routed matrices of a decoder block in a fixed order (energy-weighted cosine).
    """
    if group_by not in ("target", "decoder_layer"):
        raise ValueError("group_by must be target or decoder_layer")
    run_dir = Path(run_dir)
    mode, factor, tasks = _metadata(run_dir)
    with safe_open(run_dir / "method_state.safetensors", framework="pt", device="cpu") as state:
        modules = _head_index(state, factor, len(tasks))
    layers = {}
    for target, heads in sorted(modules.items()):
        group = target
        if group_by == "decoder_layer":
            match = re.match(r"^(.*\.layers\.\d+)\.", target)
            if match is None:
                raise ValueError(f"cannot identify decoder layer in {target}")
            group = match[1]
        entries = layers.setdefault(group, {task: [] for task in tasks})
        for i, task in enumerate(tasks):
            entries[task].append(heads[i])
    return {"mode": mode, "factor": factor, "group_by": group_by, "layers": layers,
            "note": "Raw factor cosine is basis/initialization dependent; descriptive only."}


def summarize_heads(run_dir):
    run_dir = Path(run_dir)
    mode, factor, task_names = _metadata(run_dir)
    with safe_open(run_dir / "method_state.safetensors", framework="pt", device="cpu") as state:
        modules = _head_index(state, factor, len(task_names))
        matrices = {}
        for name, heads in sorted(modules.items()):
            weights = torch.stack([state.get_tensor(heads[i]).float().flatten()
                                   for i in range(len(task_names))])
            if not torch.isfinite(weights).all():
                raise ValueError(f"{name}: non-finite head weights")
            norms = weights.norm(dim=1)
            normalized = weights / norms.clamp_min(torch.finfo(weights.dtype).tiny)[:, None]
            cosine = (normalized @ normalized.T).clamp(-1, 1)
            defined = (norms > 0)[:, None] & (norms > 0)[None, :]
            matrices[name] = {
                "norms": norms.tolist(),
                "cosine": [[float(cosine[i, j]) if defined[i, j] else None
                            for j in range(len(task_names))] for i in range(len(task_names))],
            }
    commit_file = run_dir / "git_commit.txt"
    return {"run_dir": str(run_dir.resolve()), "mode": mode, "factor": factor,
            "task_names": task_names, "n_targets": len(matrices),
            "training_provenance": commit_file.read_text() if commit_file.exists() else None,
            "targets": matrices,
            "note": "Raw factor cosine, per target; basis/initialization dependent, descriptive only."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--out", required=True, help="write analysis JSON in your own folder")
    parser.add_argument("--head_map_out", help="also export the explicit map for representations.py heads")
    parser.add_argument("--group_by", choices=("target", "decoder_layer"), default="target")
    args = parser.parse_args()
    result = summarize_heads(args.run_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if args.head_map_out:
        target = Path(args.head_map_out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(head_mapping(args.run_dir, args.group_by), indent=2) + "\n")


if __name__ == "__main__":
    main()
