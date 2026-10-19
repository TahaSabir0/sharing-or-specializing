# train/: training harness

| File | What | Who runs it |
|---|---|---|
| `journal_train.py` | the training entry point: a wrapper around LLaMA-Factory v0.9.5 with the PLAN.md §1 protocol pinned (`PROTOCOL`), per-example loss, routing ids, the plug-in hook and the run registry | via the two scripts below |
| `smoke.sh` | **smoke test**: 50 steps + a 10-per-task dev-select eval, one at a time on the box, output in `runs/smoke/` | everyone (`KICKSTART.md` §5) |
| `run_chain.sh` | real runs: a list of configs, one after another, on GPU 1 | **Taha only** |
| `journal_loss.py` | per-example loss (PLAN.md §1.5); unit test in `code/tests/` | |
| `target_modules_qwen35.json` | the frozen LoRA target set (248 language-model linears) and P8 = 21,639,168 | |
| `launch_template_ws.sh` | the old ws launcher, reference only | nobody |

```bash
bash code/train/smoke.sh <your_name> code/configs/v2/<config>.yaml        # everyone
# check a config resolves, no training, CPU only (point JOURNAL_RUNS at your own folder!):
JOURNAL_RUNS=/data/ugrip2026/journal/people/<you>/dry python code/train/journal_train.py --config <config.yaml> --dry
```

What a run writes (`/data/ugrip2026/journal/runs/<run_id>/`): `config.yaml` (resolved, with
deviations), `git_commit.txt`, `decks.json` (md5s), `trainer_log.jsonl`, `task_loss.jsonl`,
`runtime.json` (s/it, peak memory, parameter counts, GPU snapshot), `registry_row.csv`, and the
adapter (LoRA) or `method_state.safetensors` + `method_config.json` (plug-ins).
