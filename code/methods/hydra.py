"""PLAN v2 Hydra-T / Reversed Hydra with six oracle task heads.

Both factors train jointly on the mixture; there is no shared-adapter pretraining,
frozen A, learned router or auxiliary loss. Only the selected head is evaluated.
The target iterator is shared with the reference plug-in. ``target_spec`` can carry
the MedGemma port's frozen JSON spec verbatim, without embedding layer names here.
"""
import math

import torch
from torch import nn

from methods.base import RoutingStats, count_trainable, iter_target_modules, replace_module, routing

DEFAULTS = {"mode": "hydraT", "r": 8, "alpha": 16, "dropout": 0.05,
            "n_tasks": 6, "log_routing": True, "target_spec": None}


def resolve_config(cfg):
    cfg = dict(cfg or {})
    unknown = set(cfg) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown Hydra settings: {sorted(unknown)}")
    cfg = {**DEFAULTS, **cfg}
    if cfg["mode"] not in ("hydraT", "hydraR"):
        raise ValueError("mode must be hydraT or hydraR")
    if type(cfg["r"]) is not int or cfg["r"] <= 0:
        raise ValueError("r must be a positive integer")
    if type(cfg["n_tasks"]) is not int or cfg["n_tasks"] != 6:
        raise ValueError("the frozen journal task mapping requires n_tasks=6")
    for key in ("alpha", "dropout"):
        if type(cfg[key]) not in (int, float) or not math.isfinite(cfg[key]):
            raise ValueError(f"{key} must be finite")
    if cfg["alpha"] <= 0 or not 0 <= cfg["dropout"] < 1:
        raise ValueError("alpha must be positive and dropout must be in [0, 1)")
    if type(cfg["log_routing"]) is not bool:
        raise ValueError("log_routing must be a boolean")
    if cfg["target_spec"] is not None and not isinstance(cfg["target_spec"], dict):
        raise ValueError("target_spec must be a frozen target-spec mapping or null")
    return cfg


class HydraLinear(nn.Module):
    """Frozen base plus B_task A (hydraT), or B A_task (hydraR).

    Inputs are [batch, ..., in_features]; the oracle label is constant across all
    tokens of a sequence, including incremental generation and checkpoint replay.
    """

    def __init__(self, base, name, cfg, stats=None):
        super().__init__()
        cfg = resolve_config(cfg)
        self.base, self.name = base, name
        self.mode, self.r, self.n_tasks = cfg["mode"], cfg["r"], cfg["n_tasks"]
        self.scaling = cfg["alpha"] / self.r
        self.dropout = nn.Dropout(cfg["dropout"])
        self.stats = stats
        self.base.requires_grad_(False)

        def linear(n_in, n_out, zero=False):
            # Allocate only new fp32 parameters on the base device. Never .to() the base.
            layer = nn.Linear(n_in, n_out, bias=False, device=base.weight.device,
                              dtype=torch.float32)
            if zero:
                nn.init.zeros_(layer.weight)
            else:
                nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
            return layer

        if self.mode == "hydraT":
            self.lora_A = linear(base.in_features, self.r)
            self.lora_B = nn.ModuleList([linear(self.r, base.out_features, zero=True)
                                         for _ in range(self.n_tasks)])
        else:
            self.lora_A = nn.ModuleList([linear(base.in_features, self.r)
                                         for _ in range(self.n_tasks)])
            self.lora_B = linear(self.r, base.out_features, zero=True)

    @property
    def active_params_per_example(self):
        return self.r * (self.base.in_features + self.base.out_features)

    def _task_route(self, x):
        ids = routing.get("task_id")
        if not isinstance(ids, torch.Tensor):
            raise ValueError(f"{self.name}: Hydra requires oracle task_id labels")
        if x.ndim < 2 or ids.ndim != 1 or ids.shape[0] != x.shape[0]:
            raise ValueError(f"{self.name}: expected one task_id per batch row")
        if ids.dtype != torch.long:
            raise ValueError(f"{self.name}: task_id must be a LongTensor")
        return routing.task_groups(self.n_tasks)

    def forward(self, x):
        route = self._task_route(x)
        out = self.base(x)
        first_a = self.lora_A if self.mode == "hydraT" else self.lora_A[0]
        x_lora = self.dropout(x.to(first_a.weight.dtype))
        projected = self.lora_A(x_lora) if self.mode == "hydraT" else x_lora
        if route.single_task is not None:
            # The protocol trains with batch size 1. A uniform batch needs no
            # row gathering/scattering or full-width zero buffer, in either mode.
            task = route.single_task
            head = self.lora_B[task] if self.mode == "hydraT" else self.lora_A[task]
            delta = head(projected)
        else:
            width = self.base.out_features if self.mode == "hydraT" else self.r
            delta = projected.new_zeros((*x.shape[:-1], width))
            for task, rows in route.rows_on(x.device):
                head = self.lora_B[task] if self.mode == "hydraT" else self.lora_A[task]
                values = head(projected.index_select(0, rows))
                delta = delta.index_copy(0, rows, values)
        if self.mode == "hydraR":
            delta = self.lora_B(delta)
        if self.stats is not None and not self.training:
            self.stats.add_layer(self.name, route.weights)
        return out + (delta * self.scaling).to(out.dtype)


def apply_method(model, cfg):
    cfg = resolve_config(cfg)
    targets = iter_target_modules(model, spec=cfg["target_spec"])
    model.requires_grad_(False)
    stats = RoutingStats([name for name, _ in targets], cfg["n_tasks"]) if cfg["log_routing"] else None
    stored = active = 0
    for name, base in targets:
        new = HydraLinear(base, name, cfg, stats)
        replace_module(model, name, new)
        active += new.active_params_per_example
        a = cfg["r"] * base.in_features
        b = cfg["r"] * base.out_features
        stored += a + cfg["n_tasks"] * b if cfg["mode"] == "hydraT" else cfg["n_tasks"] * a + b
    actual = count_trainable(model)
    if actual != stored:
        raise RuntimeError(f"Hydra parameter mismatch: expected {stored}, got {actual}")
    model.journal_parameter_counts = {"stored_params": stored, "active_params_per_example": active}
    if stats is not None:
        model.journal_routing_stats = stats
    print(f"[hydra] mode={cfg['mode']} layers={len(targets)} "
          f"stored_params={stored:,} active_params_per_example={active:,}")
    return model
