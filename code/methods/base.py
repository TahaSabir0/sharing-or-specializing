"""Shared plumbing for method plug-ins (see code/INTERFACE.md).

A plug-in is a module `code/methods/<name>.py` exposing

    apply_method(model, cfg: dict) -> model

The harness calls it on the loaded, frozen 4-bit model (instead of LF's own LoRA setup) in both
training and evaluation. This file gives plug-ins what they need:

- `iter_target_modules(model)`: the frozen target set of the loaded backbone
  (train/target_modules_qwen35.json or target_modules_medgemma15.json, see methods/backbones.py).
- `routing`: the current batch's oracle labels, set by the harness before every forward.
- `aux_loss`: any module may set `self.aux_loss` (scalar tensor) in forward; the harness sums
  all of them into the loss and clears them (`pop_aux_loss`).
- `RoutingStats`: accumulate per-layer routing weights and write routing_stats.npz.
- `save_method_state` / `load_method_state`: only the trainable parameters are saved.
"""
import json
import os
from contextlib import contextmanager

import numpy as np
import torch

from methods import backbones

_HERE = os.path.dirname(os.path.abspath(__file__))
TARGETS_PATH = os.path.join(_HERE, "..", "train", "target_modules_qwen35.json")
ROUTING_KEYS = ("task_id", "modality_id", "dataset_id")


# ------------------------------------------------------------------------------ target modules
def load_targets(path=TARGETS_PATH):
    with open(path) as f:
        return json.load(f)


def iter_target_modules(model, path=None):
    """Yield (qualified_name, module) for every frozen target linear (248 on Qwen3.5-9B, 238 on
    MedGemma-1.5-4B).

    Matches LF `lora_target: all`: leaf name in the frozen list, inside the language model. The
    file is chosen from the model's config.model_type (methods/backbones.py); Qwen's when unknown."""
    if path is None:
        path = backbones.targets_path_for_model_type(getattr(getattr(model, "config", None), "model_type", None))
    spec = load_targets(path)
    found = []
    for name, mod in model.named_modules():
        if backbones.matches_target(name, spec) and isinstance(mod, torch.nn.Linear):
            found.append((name, mod))
    if len(found) != spec["n_modules"]:
        raise RuntimeError(f"expected {spec['n_modules']} target modules, found {len(found)}")
    return found


def replace_module(model, name, new_module):
    parent_name, leaf = name.rsplit(".", 1)
    setattr(model.get_submodule(parent_name), leaf, new_module)


# ------------------------------------------------------------------------------------ routing
class _Routing:
    """Holds the oracle labels of the batch being processed: LongTensors of shape [batch].

    Constant over a sequence, so per-sequence routes stay valid during generation (KV cache)."""

    def __init__(self):
        self._ids = {}

    def set(self, **ids):
        self._ids = {k: v for k, v in ids.items() if v is not None}

    def clear(self):
        self._ids = {}

    def get(self, key):
        """`routing.get("task_id")` -> LongTensor [batch], or None outside a labelled batch."""
        return self._ids.get(key)

    @contextmanager
    def use(self, **ids):
        prev = self._ids
        self.set(**ids)
        try:
            yield
        finally:
            self._ids = prev


routing = _Routing()


# ----------------------------------------------------------------------------------- aux loss
def pop_aux_loss(model):
    """Sum and clear every `aux_loss` tensor set by modules during the last forward."""
    total = None
    for mod in model.modules():
        val = getattr(mod, "aux_loss", None)
        if isinstance(val, torch.Tensor):
            total = val if total is None else total + val
            mod.aux_loss = None
    return total


# ------------------------------------------------------------------------------ routing stats
class RoutingStats:
    """Fixed routing_stats.npz schema (one file per evaluation or training run).

    Arrays (L = routed layers, E = experts/heads, N = sequences seen):
      layer_names  [L]      str   qualified module names
      weights      [N,L,E]  f32   mean routing weight per sequence (over its answer+prompt tokens)
      task_id      [N]      i64   oracle labels of each sequence (-1 if unknown)
      modality_id  [N]      i64
      dataset_id   [N]      i64
      sample_id    [N]      str   row `id` when the evaluator provides it, else ""
    """

    def __init__(self, layer_names, n_experts):
        self.layer_names = list(layer_names)
        self.n_experts = int(n_experts)
        self._rows, self._labels, self._ids = [], [], []
        self._pending = {}

    def add_layer(self, layer_name, weights):
        """Call from a routed module's forward: weights [batch, tokens, E] (or [batch, E])."""
        w = weights.detach().float()
        if w.dim() == 3:
            w = w.mean(dim=1)
        self._pending.setdefault(layer_name, []).append(w.cpu())

    def end_batch(self, sample_ids=None):
        """Call once per batch after the forward (the evaluator does this)."""
        if not self._pending:
            return
        w = torch.stack([self._stack_layer(n) for n in self.layer_names], dim=1)  # [B, L, E]
        self._rows.append(w.numpy())
        b = w.shape[0]
        lab = [routing.get(k) for k in ROUTING_KEYS]
        self._labels.append(np.stack([l.cpu().numpy() if l is not None else -np.ones(b, dtype=np.int64)
                                      for l in lab], axis=1))
        self._ids += list(sample_ids) if sample_ids is not None else [""] * b
        self._pending = {}

    def _stack_layer(self, name):
        chunks = self._pending.get(name)
        if not chunks:
            raise RuntimeError(f"no routing weights recorded for {name}")
        # generation calls forward many times; average the per-step means for each sequence
        return torch.stack(chunks, dim=0).mean(0)

    def save(self, path):
        lab = np.concatenate(self._labels, 0) if self._labels else np.zeros((0, 3), np.int64)
        np.savez_compressed(
            path,
            layer_names=np.array(self.layer_names),
            weights=np.concatenate(self._rows, 0).astype(np.float32) if self._rows
            else np.zeros((0, len(self.layer_names), self.n_experts), np.float32),
            task_id=lab[:, 0], modality_id=lab[:, 1], dataset_id=lab[:, 2],
            sample_id=np.array(self._ids))


def validate_routing_stats(path):
    """Smoke-gate item 5: file exists, non-empty, schema valid. Returns a short summary dict."""
    z = np.load(path, allow_pickle=False)
    need = {"layer_names", "weights", "task_id", "modality_id", "dataset_id", "sample_id"}
    missing = need - set(z.files)
    assert not missing, f"routing_stats missing {missing}"
    n, l, e = z["weights"].shape
    assert n > 0, "routing_stats is empty"
    assert l == len(z["layer_names"]), "layer_names / weights mismatch"
    for k in ("task_id", "modality_id", "dataset_id", "sample_id"):
        assert len(z[k]) == n, f"{k} length != N"
    assert np.isfinite(z["weights"]).all(), "non-finite routing weights"
    return {"sequences": n, "layers": l, "experts": e}


# ------------------------------------------------------------------------- save / load state
METHOD_STATE = "method_state.safetensors"
METHOD_CONFIG = "method_config.json"


def save_method_state(model, out_dir, method, cfg):
    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok=True)
    state = {n: p.detach().cpu().contiguous() for n, p in model.named_parameters() if p.requires_grad}
    save_file(state, os.path.join(out_dir, METHOD_STATE))
    with open(os.path.join(out_dir, METHOD_CONFIG), "w") as f:
        json.dump({"method": method, "cfg": cfg, "n_params": sum(v.numel() for v in state.values())}, f, indent=1)


def load_method_state(model, out_dir):
    from safetensors.torch import load_file
    state = load_file(os.path.join(out_dir, METHOD_STATE))
    params = dict(model.named_parameters())
    missing = [k for k in state if k not in params]
    if missing:
        raise RuntimeError(f"{len(missing)} saved params not in model, e.g. {missing[:3]}")
    with torch.no_grad():
        for k, v in state.items():
            params[k].data.copy_(v.to(params[k].device, params[k].dtype))
    return model


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
