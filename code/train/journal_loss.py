"""Per-example loss, routing ids and aux loss for the journal trainer (PLAN.md §1.5, §3.2).

Kept free of LLaMA-Factory imports so the unit test (tests/test_per_example_loss.py) can run on
any transformers install.

Per-example loss: with batch 1, each micro-batch holds one example and the model returns that
example's mean token cross-entropy, provided `num_items_in_batch` is NOT passed to the model.
We force `model_accepts_loss_kwargs = False`; HF's training_step then divides each micro-batch
loss by the accumulation steps, so one optimizer step averages 8 per-example means. (With
`num_items_in_batch`, HF would instead normalise by the total answer tokens across the 8
micro-batches: token weighting, which the protocol removes.)
"""
import collections
import json
import os
import sys

import torch

_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)
from methods.base import ROUTING_KEYS, pop_aux_loss, routing  # noqa: E402


class JournalLossMixin:
    """Mix in before a transformers Trainer subclass: `class T(JournalLossMixin, Trainer)`."""

    journal_task_names = None      # {task_id: name}, for per-task loss logging
    journal_task_loss_path = None  # jsonl written at every logging step (training only)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False  # per-example loss; see module docstring
        self._task_loss = collections.defaultdict(list)
        self._aux = []

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        ids = {k: inputs.pop(k) for k in ROUTING_KEYS if k in inputs}
        # Stays set until the next batch: with gradient checkpointing the forward re-runs during
        # backward, after this function has returned, and routed modules must see the same ids.
        routing.set(**ids)
        pop_aux_loss(model)  # discard aux losses left over from a checkpointing re-run
        out = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=None)
        loss, outputs = out
        aux = pop_aux_loss(model)
        if model.training:
            tid = ids.get("task_id")
            if tid is not None and tid.numel() == 1:
                self._task_loss[int(tid.item())].append(float(loss.detach().float().item()))
            if aux is not None:
                self._aux.append(float(aux.detach().float().item()))
        if aux is not None:
            loss = loss + aux
        return (loss, outputs) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        if "loss" in logs and (self._task_loss or self._aux):
            rec = {"step": self.state.global_step, "epoch": self.state.epoch}
            for tid, vals in sorted(self._task_loss.items()):
                name = (self.journal_task_names or {}).get(tid, str(tid))
                rec[f"loss/{name}"] = sum(vals) / len(vals)
                rec[f"n/{name}"] = len(vals)
            if self._aux:
                logs["aux_loss"] = sum(self._aux) / len(self._aux)
                rec["aux_loss"] = logs["aux_loss"]
            if self.journal_task_loss_path and self.is_world_process_zero():
                with open(self.journal_task_loss_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
            self._task_loss.clear()
            self._aux.clear()
        return super().log(logs, *args, **kwargs)


def routing_tensors(routing_rows):
    """[[task, modality, dataset], ...] -> dict of LongTensors [batch]."""
    t = torch.tensor(routing_rows, dtype=torch.long)
    return {k: t[:, i] for i, k in enumerate(ROUTING_KEYS)}
