# train/ — training harness (W2, Taha)

To build (PLAN.md §3.1, W2): a `journal_train` wrapper around LLaMA-Factory v0.9.5 with
dropout 0.05, per-example loss (PLAN.md §1.5, with a unit test), the frozen target-module list
(`target_modules_qwen35.json`), task/modality/dataset ids passed to the model, the plug-in
hook, and run registration (`results/RUNS.csv`, `runtime.json`).

`launch_template_ws.sh` is the challenge launcher pattern, kept as a starting point.
