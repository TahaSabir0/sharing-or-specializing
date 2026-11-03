# Hydra-T and Reversed Hydra (PLAN v2)

`hydra.py` implements jointly trained, oracle-routed adapters. Hydra-T computes
`base(x) + (alpha/r) B_task A dropout(x)`; Reversed Hydra computes
`base(x) + (alpha/r) B A_task dropout(x)`. Six heads follow `data/label_ids.json`:
cls 0, ML 1, detection 2, counting 3, regression 4, report 5.

Both factors train from scratch: A uses Kaiming-uniform initialization, B is zero.
The initial model equals the frozen base. This replaces the historical challenge
experiment that pretrained A, froze it, then trained task heads separately.
There is no learned router, auxiliary loss, adapter pretraining or method tuning.

## Parameters and configuration

Every language-model target from `iter_target_modules` is wrapped, with no vision
or projector targets. Only new fp32 adapter parameters are allocated on the base
device; the 4-bit base is neither moved nor cast. Inputs to the delta are cast to
the adapter dtype; the delta returns to the base output dtype.

| Qwen3.5-9B, r8 | Stored/trainable | Active per example |
|---|---:|---:|
| Hydra-T | 78,716,928 | 21,639,168 |
| Reversed Hydra | 72,757,248 | 21,639,168 |

| MedGemma-1.5-4B, r8 | Stored/trainable | Active per example |
|---|---:|---:|
| Hydra-T | 55,287,808 | 14,901,248 |
| Reversed Hydra | 49,020,928 | 14,901,248 |

Counts are derived from actual target shapes, checked at load, printed, and added
to `runtime.json` as `stored_params` / `active_params_per_example`. Both Hydra
modes expose all factors as trainable, but each example activates only its head.
Unselected heads are not evaluated and receive no gradient from that example.

All method settings have explicit defaults: `mode: hydraT`, `r: 8`, `alpha: 16`,
`dropout: 0.05`, `n_tasks: 6`, `log_routing: true`, `target_spec: null`.
With null, the merged backbone registry selects the frozen target file from
`model.config.model_type` (Qwen 248, Gemma3 238); an explicit `target_spec`
remains supported for validated ports and tiny-model tests.
Unknown settings and missing/invalid task routes fail. Generation expects one
label per batch row, constant over prefill and decoding; beam expansion is not
supported by this oracle contract (the protocol uses greedy generation).

Six E1 configs are in `configs/v2/hydra{T,R}_s{1337,2026,42}.yaml`;
six E2 configs are in `configs/e2/hydra{T,R}_s{1337,2026,42}.yaml`.
Each uses the full `j26_train_mix`, the unchanged training protocol, and no
overrides. Data rendering, per-example loss, prompts, decoding and scoring use the
existing harness; no holdout selection or parser changes are introduced.

## Validation and server gate

Local CPU tests:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m pytest code/tests -q
```

These check formulas/gradients, frozen base, head isolation, checkpoint replay,
mixed-precision output, save/load, prefill/decode routing statistics and the exact
248-module Qwen parameter budget. Tiny real Transformers Qwen3.5/Gemma3 models
also test decreasing loss, reentrant gradient checkpointing, and cached versus
uncached generation. On 2026-10-20, all 45 tests passed locally with torch 2.11.0,
transformers 5.16.1 and accelerate 1.14.0 (CPU; no model downloads).
The suite needs pytest, PyYAML and the existing torch/transformers/accelerate/
safetensors/numpy dependencies. Use a separate local test environment; do not
install into the shared server environment.
CPU tests do not establish 4-bit CUDA loading,
real-model loss, throughput, peak memory or the Blackwell smoke gate.

### Routing performance update (2026-11-03)

The first Blackwell Hydra-T smoke timed out at 44/50 steps (exit 124); evaluation
was skipped. Loss decreased 1.536 -> 0.808 and parameter counts matched, but this
does not pass the gate. The user-supplied log showed about 80 s/step, versus about
19 s/step for the preceding reference smoke; shared-GPU conditions may also differ.

The original forward validated GPU labels and called `unique().tolist()` and
`nonzero()` in each target layer, including checkpoint replay. The updated routing
object resolves labels once per batch and shares metadata across all layers.
Uniform batches directly use the chosen head, without gathering/scattering rows
or allocating a full-width zero delta. Mixed batches reuse device row indices;
evaluation statistics reuse CPU one-hot weights. Formula, dropout placement,
head isolation, parameter names/budgets and the smoke time limit are unchanged.
Routes remain set through backward for checkpoint replay; set/clear/nested use
reset or restore the cache. In-place mutations of normal label tensors invalidate
the cache; inference tensors must remain immutable until the next routing context.

All 63 CPU tests passed after the change, including uniform/mixed outputs and
gradients, dropout/checkpoint replay, inference-mode prefill/decode, and a CPU
profiler regression check excluding row operations and scalar extraction from
the prepared batch-one path.

Subsequent user-provided Blackwell summaries on 3 Nov report successful 50-step
training and evaluation (both exit codes 0) for both Qwen modes:

| Mode | Run suffix | First -> last loss | Trainable params | s/step | Peak reserved GiB |
|---|---|---|---:|---:|---:|
| Hydra-T | `11032130` | 1.534 -> 0.641 | 78,716,928 | 14.122 | 17.89 |
| Hydra-R | `11032152` | 1.611 -> 0.630 | 72,757,248 | 14.147 | 17.79 |

Full run IDs are `j26_q35_hydraT_full_768_s1337_smoke_veronika_11032130`
and `j26_q35_hydraR_full_768_s1337_smoke_veronika_11032152`. Both evaluated 10
questions for each of six tasks with zero malformed/truncated/error counts;
routing statistics validated as 60 sequences x 248 layers x 6 experts. These are
successful smoke checks, not comparative task-quality results or a controlled
speed benchmark. No new prompt/template gate was required because data/prompts
were unchanged. See STATUS for tested-source provenance and pending artifact
collection/review. The interrupted October run remains a failed smoke.

On Blackwell, from Veronika's personal clone, first check `nvidia-smi -i 1` and
run the reference smoke if onboarding is unfinished. Then run the two method
smokes sequentially, following the lock enforced by `smoke.sh`:

```bash
bash code/train/smoke.sh veronika code/configs/v2/hydraT_s1337.yaml
bash code/train/smoke.sh veronika code/configs/v2/hydraR_s1337.yaml
```

Review every gate item in the summary: loss decreases, counts match the table,
all six tasks evaluate on dev-select with sensible malformed/truncation rates,
`routing_stats.npz` validates, and `runtime.json` records counts/time/memory.
The script's exit status alone is insufficient to establish the evaluation gate.
Prompts/data are unchanged; rerun `template_gate.py` if those change during the port.
Submit the summaries and six configs in the PR. Only Taha queues real runs and
holdout evaluations; never use the shared training checkout for personal work.

## MedGemma integration

Alberto's port is merged from current main. The existing registry provides the
Gemma3 model loader, frozen 238-target list, native 896-square image handling,
EOS and template, and the training token_type_ids fix. Hydra calls the common
target iterator and uses the same apply function at training and evaluation.
The runtime callback records stored/active counts alongside the backbone's P8.
No MedGemma-specific method hyperparameters or recipe overrides are introduced.

Both modes pass CPU shape/count tests against the full frozen MedGemma spec and
training/checkpointing/generation tests on tiny real Gemma3 models. **Real NF4
MedGemma Hydra smoke remains pending**; the original Qwen smoke does not certify
the subsequently merged backbone integration. See `../HYDRA_RUNBOOK.md` for exact
commands, source provenance, evidence collection and the remaining GPU gates.

## E7 head analysis (after training)

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python code/analysis/hydra_heads.py --run_dir <completed-run> --out <your-analysis-folder>/heads.json
```

Reads adapter weights on CPU only, without model loading or holdout inference.
Writes a 6-by-6 cosine matrix per target module: B heads for Hydra-T, A heads for
Reversed Hydra, with task labels and training provenance when available. Zero-norm
heads produce JSON null. Raw-factor cosine depends on the factor basis and random
initialization; treat it as descriptive, compare seeds, and do not infer task
transfer or statistical significance from it alone. Alex owns the broader
representation analysis and its layer/probe choices.

E7 integration with Alex: `--head_map_out <path>` exports the exact checkpoint
parameter map consumed by `representations.py heads`. `--group_by target` keeps
targets separate; `--group_by decoder_layer` concatenates heads within each
decoder block in a fixed order, giving a norm-weighted block cosine. Both map
formats are integration-tested against Alex's actual analysis function.
The layer grouping comes from parameter paths rather than assumed layer counts.
