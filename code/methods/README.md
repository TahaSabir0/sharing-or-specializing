# methods/: one plug-in per adaptation method

A plug-in is `code/methods/<name>.py` exposing `apply_method(model, cfg) -> model`. The contract,
the helpers in `base.py`, and the smoke-test workflow are in `code/INTERFACE.md`.

| Module | Methods | Owner | Spec | State |
|---|---|---|---|---|
| `base.py` | plumbing: target set, routing ids, aux loss, routing stats, save/load | Taha | INTERFACE.md §2 | done |
| `lora_ref.py` | plain LoRA as a plug-in: **the template to copy** | Taha | INTERFACE.md §1 | done, passes the smoke gate |
| `hydra.py` | `mode: hydraT` (shared A r8, 6 task heads B_t r8) and `mode: hydraR` (6 task heads A_t r8, shared B r8); the oracle `task_id` picks the head in training and inference | Veronika | PLAN.md §1.3, E1 | to build; must also run on MedGemma (E2) |

Shared LoRA, Shared r48 and TaskLoRA need no plug-in: they are plain LoRA (`method: lora`).
Method notes from the papers: `references/papers/METHOD_DIGEST.md` (HydraLoRA section).
