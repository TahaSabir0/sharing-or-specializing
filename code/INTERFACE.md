# Plug-in interface and run workflow

How to add a method, configure a run, smoke-test it and get it run for real. The protocol every
run follows is PLAN.md §1; the team rules are PLAN.md §7. If you are new, start with
`KICKSTART.md` at the repo root.

## 1. What a plug-in is

Shared LoRA, TaskLoRA and Shared r48 are plain LoRA and need no plug-in (`method: lora`). Any
other method (HydraLoRA, ...) is a **plug-in**: one file `code/methods/<name>.py` exposing

```python
def apply_method(model, cfg: dict) -> model
```

- **Training:** `code/train/journal_train.py` loads the frozen 4-bit model, calls your
  `apply_method` *instead of* LLaMA-Factory's own LoRA setup, trains whatever parameters you
  left with `requires_grad=True`, and saves only those.
- **Evaluation:** `code/eval/eval_harness.py --mode plugin --run_dir <run>` loads the base model,
  calls the same `apply_method` with the saved config, then loads your saved weights.

So one function serves both paths. **Start by copying `code/methods/lora_ref.py`**: it is
plain LoRA written as a plug-in and uses every hook below. Its smoke config is
`code/configs/smoke_lora_ref.yaml`.

## 2. What the harness gives you (`code/methods/base.py`)

| Helper | Use |
|---|---|
| `iter_target_modules(model)` | the `(name, module)` pairs to wrap: the frozen target set (248 language-model linears on Qwen3.5-9B, from `code/train/target_modules_qwen35.json`). It raises if the count is wrong. **Never hard-code layer names**: MedGemma has different ones. |
| `replace_module(model, name, new)` | swap a module in place |
| `routing.get("task_id")` | the current batch's oracle labels, a `LongTensor[batch]`, or `None`. Also `"modality_id"`, `"dataset_id"`. Ids are frozen in `code/data/label_ids.json` (task: cls 0, ML 1, det 2, count 3, reg 4, report 5). Set by the harness before every forward, in training and generation; constant over a sequence, so it stays valid with the KV cache. |
| `self.aux_loss` | any module may set a scalar tensor in `forward`; the trainer adds the sum to the loss |
| `RoutingStats` | for routed methods: record per-layer weights; the evaluator writes `routing_stats.npz` |
| `count_trainable(model)` | the number the smoke gate checks |

Saving and loading (`method_state.safetensors` + `method_config.json`) is automatic.

## 3. Rules for a plug-in

1. Wrap **exactly** `iter_target_modules(model)`: same layers for every method (PLAN.md §1.2).
2. Leave the base frozen. Only your new parameters get `requires_grad=True`.
3. Initialise so that **step 0 equals the base model** (LoRA: A random, B zeros). Then the
   first logged loss matches plain LoRA's: in a smoke test on the mix deck, 1.563 at step 10 and
   ~0.65 at step 50 (the reference plug-in gives exactly that).
4. Move only your new layers to the base layer's device; never move or cast the 4-bit base.
   The trainer upcasts your parameters to fp32; compute in that dtype and cast the delta back
   to the base output's dtype (as `lora_ref.py` does).
5. Read routes only through `routing.get(...)`. In training they come from the deck rows; in
   evaluation from the question's known task (oracle route).
6. Print or check the parameter count you expect (PLAN.md §1.3 lists stored vs active per arm).
7. No hidden defaults: every hyperparameter goes in `cfg` with an explicit default in the file.

## 4. Run configs

One YAML per run, in `code/configs/v2/`. Example (Hydra-T, seed 1337):

```yaml
# E1 Hydra-T, seed 1337 (PLAN.md v2 E1). One line on what this run is.
run_id: j26_q35_hydraT_full_768_s1337   # j26_{backbone}_{arm}_{data}_{res}_s{seed}
owner: veronika
study: E1
backbone: q35                            # q35 | mg15
method: hydra                            # file name in code/methods/, or lora
method_cfg: {mode: hydraT, r: 8, alpha: 16, dropout: 0.05}
deck: j26_train_mix                      # name in /data/ugrip2026/journal/data/dataset_info.json
data_tag: full
seed: 1337                               # 1337, 2026, 42
overrides: {}                            # protocol keys you change; each is logged as a deviation
```

- **Decks:** `j26_train_mix` (all tasks), `j26_train_{cls,ml,det,count,reg,report}` (one task).
  Eval decks: `j26_holdout.jsonl` (reporting), `j26_devselect.jsonl` (train-seen; smoke checks).
- **Protocol keys** (rank, lr, epochs, ...) are pinned in `journal_train.py` `PROTOCOL`. Change
  one only through `overrides:`, and only when the plan says so.

## 5. Smoke test (you run it yourself)

From your clone on the box (`KICKSTART.md` §4):

```bash
bash code/train/smoke.sh <your_name> code/configs/v2/<your_config>.yaml
```

It trains 50 steps, evaluates 10 questions per task on dev-select, and prints a summary.
One smoke test runs at a time on the whole box: if someone else's is running, it tells you who,
and you try again later. Output goes to `/data/ugrip2026/journal/runs/smoke/<run_id>_smoke_<you>_<time>/`.

**You pass the gate (PLAN.md §3.3) when the summary shows:**
1. loss decreasing over the 50 steps (and the first value close to plain LoRA's);
2. the trainable / `n_params` count you expect for your method;
3. *(only if you touched prompts or data)* `python code/eval/template_gate.py <deck>` prints `GATE_PASS`;
4. the eval ran on every task with a sensible malformed and truncation rate (50 steps is barely
   trained: the numbers are a format check, not a result);
5. routed methods: `routing_stats.npz` valid;
6. `runtime.json` written (s/it is slower than normal while the real runs share the card).

## 6. Getting a full run

Full training runs and holdout evaluations are launched **only by Taha** (they take the card
for days, and the holdout is touched once per finished arm). To ask for one:

1. push your branch and open a pull request into `main`;
2. in the PR, paste the smoke summary and list the configs to run;
3. Taha reviews, merges, and queues the runs. Results come back in `results/LEDGER.md`.

## 7. Evaluation and scoring (reference)

```bash
python code/eval/eval_harness.py --mode {zs|adapter|oracle|plugin} ... --data <deck> --surface holdout --out_dir <dir>
python code/eval/score_run.py --out_dir <dir>        # per-dataset metrics + task macro (score_single.py)
bash code/eval/crimson_box.sh <dir>                  # report generation (CRIMSON judge, GPU)
```

Modes: `zs --protocol bare|prompt` (zero-shot), `adapter --adapter <dir>` (one LoRA on every
task), `oracle --adapter_set <json>` (the task picks the LoRA: TaskLoRA, and the E3 transfer
cells via `--tasks`), `plugin --run_dir <dir>`. Never re-implement a parser or a metric: the
scorer is `code/eval/score_single.py` (rule 3 in CLAUDE.md).
