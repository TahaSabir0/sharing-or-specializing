"""Reference plug-in: plain LoRA written as a plug-in. Copy this file to start a new method.

It exercises every hook of the contract (code/INTERFACE.md):
- wraps exactly the frozen target set (methods.base.iter_target_modules);
- reads the oracle routing ids (methods.base.routing) — here only to log them;
- sets `self.aux_loss` (an L2 penalty on B, coefficient `aux_coef`, default 0 = plain LoRA);
- records routing weights for routing_stats.npz (a single "expert" with weight 1 when
  `log_routing` is on, which is enough to validate the schema).

With aux_coef 0 it is numerically plain LoRA: rank r, alpha, dropout, B initialised to zero,
A Kaiming-uniform (as PEFT), so the trainable count equals P8 and step-0 loss equals the base.

Run config:
  method: lora_ref
  method_cfg: {r: 8, alpha: 16, dropout: 0.05, aux_coef: 0.0, log_routing: true}
"""
import math

import torch
from torch import nn

from methods.base import RoutingStats, iter_target_modules, replace_module, routing

DEFAULTS = {"r": 8, "alpha": 16, "dropout": 0.05, "aux_coef": 0.0, "log_routing": True}


class RefLoRALinear(nn.Module):
    def __init__(self, base, name, r, alpha, dropout, aux_coef, stats):
        super().__init__()
        self.base, self.name = base, name                      # frozen 4-bit Linear
        self.r, self.scaling = r, alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Linear(base.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.aux_coef, self.stats = aux_coef, stats
        self.aux_loss = None

    def forward(self, x):
        out = self.base(x)
        delta = self.lora_B(self.lora_A(self.dropout(x.to(self.lora_A.weight.dtype)))) * self.scaling
        if self.aux_coef:
            self.aux_loss = self.aux_coef * self.lora_B.weight.pow(2).sum()
        if self.stats is not None and not self.training:
            tid = routing.get("task_id")                         # LongTensor [batch] or None
            b = x.shape[0] if x.dim() == 3 else 1
            if tid is not None:
                assert tid.shape[0] == b, "routing ids and batch disagree"
            self.stats.add_layer(self.name, torch.ones(b, 1, device=x.device))
        return out + delta.to(out.dtype)


def apply_method(model, cfg):
    cfg = {**DEFAULTS, **(cfg or {})}
    targets = iter_target_modules(model)
    stats = RoutingStats([n for n, _ in targets], n_experts=1) if cfg["log_routing"] else None
    for name, mod in targets:
        new = RefLoRALinear(mod, name, cfg["r"], cfg["alpha"], cfg["dropout"], cfg["aux_coef"], stats)
        dev = mod.weight.device
        new.lora_A.to(dev)
        new.lora_B.to(dev)                    # only the new layers move; the 4-bit base stays put
        replace_module(model, name, new)
        for p in (*new.lora_A.parameters(), *new.lora_B.parameters()):
            p.requires_grad_(True)
    if stats is not None:
        model.journal_routing_stats = stats   # the eval harness saves it as routing_stats.npz
    return model
