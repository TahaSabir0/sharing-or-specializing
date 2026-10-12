# methods/ — one plug-in per adaptation method

Each method is a module exposing `apply_method(model, cfg) -> model`, following
`code/INTERFACE.md`. It wraps the frozen target-module list
(`code/train/target_modules_qwen35.json`), takes routing inputs as `task_id` / `modality_id`
/ `dataset_id`, returns auxiliary losses through `aux_loss`, and writes `routing_stats.npz`.

| Module (planned) | Methods | Owner | Spec |
|---|---|---|---|
| `hydra.py` | Hydra-T (oracle heads), Hydra-L (learned router) | Veronika | PLAN.md S4a, W4 — build oracle mode first |
| `moe.py` | LoRAMoE, top-k MoE, spectral init, modality-guided MI loss | Veronika | PLAN.md S6, W5 |
| `composition.py` | fixed λ composition, MoLE-style gates (hidden-state input) | Alex | PLAN.md S4b, W6 |
| `location.py` | vision / projector / late-LLM target sets | Alex | PLAN.md S8, W7 |

Shared LoRA and TaskLoRA need no plug-in: they are plain LoRA through the training harness.
Method details from the papers: `references/papers/METHOD_DIGEST.md`.
