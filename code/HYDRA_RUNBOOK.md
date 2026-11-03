# Veronika: Hydra handoff and server checks

**Completed:** all four integrated Qwen/MedGemma smokes on `87e58a0`; the archive
has been downloaded and checked locally (40 file hashes, source, recipe, counts,
losses and oracle routing). See `../results/smoke/hydra_integrated_20261103.md`
and the sanitized archive audit alongside it. Further smoke launches are not
needed for this unchanged implementation; commands below document reproduction.

The Qwen smoke source was preserved in commit `ad7497e`; it was originally tested
as an uncommitted update on base `dfbce98`. Subsequent integration merges the
current MedGemma port; do not describe the earlier run HEAD as this integrated
code's GPU validation. CPU tests cover the merged code. Taha reviews/merges and
launches all full training/holdout jobs. Veronika uses only `smoke.sh` on GPU 1.

## Completed Qwen checks (user-provided summaries)

| Mode | Smoke run | Loss first -> last | Trainable | s/step | Peak reserved GiB |
|---|---|---|---:|---:|---:|
| Hydra-T | `j26_q35_hydraT_full_768_s1337_smoke_veronika_11032130` | 1.534 -> 0.641 | 78,716,928 | 14.122 | 17.89 |
| Hydra-R | `j26_q35_hydraR_full_768_s1337_smoke_veronika_11032152` | 1.611 -> 0.630 | 72,757,248 | 14.147 | 17.79 |

Train/eval exits 0/0 for both. All six tasks had n=10, zero malformed, truncation
and error counts. Routing shape: 60 sequences, 248 targets, six experts. These
are development checks, not medical task-performance results. Raw evidence must
be collected and active counts verified in `runtime.json`.

## Personal server clone

Update only `/data/ugrip2026/journal/people/veronika/repo`, with no job reading its
source. Preserve the uncommitted source from the passed runs before changing it.
Do not update `/data/ugrip2026/journal/repo`, change shared environments, or use
other GPUs. The handoff bundle contains the tested snapshot and integrated code;
the transfer installer backs up matching uncommitted files before fast-forwarding.

Collect the existing Qwen evidence from the personal clone (stdlib only):

```bash
python3 code/train/collect_hydra_smoke.py \
  --out /data/ugrip2026/journal/people/veronika/qwen-hydra-smoke-evidence.tar.gz
```

This copies only configs, runtime, deck hashes, registry/task-loss summaries,
routing stats and metrics. It excludes training examples, full logs, raw
predictions, model weights and images. The archive manifest records file hashes.

## Integrated-branch smoke checks

For unattended sequential execution from a clean source checkout, use:

```bash
nohup python3 code/train/hydra_smoke_queue.py \
  --out_root /data/ugrip2026/journal/people/veronika/hydra-integration-smokes \
  > /data/ugrip2026/journal/people/veronika/hydra-integration-queue.log 2>&1 < /dev/null &
```

The queue calls only the existing `smoke.sh`, once for each config below. It
advances only after train/eval exits 0, decreasing finite loss, correct
stored/active counts and runtime fields, n=10 for all tasks, zero format/error/
truncation counts, and the expected validated routing summary. Any other result
stops for manual review; it does not adjust settings or classify a nonzero format
rate automatically. At completion it collects small evidence into the queue's
output directory. Use `--stage medgemma` to request only the two MedGemma smokes.
Output directories must be new. Each smoke keeps its original lock and time cap.

Run sequentially, reviewing each summary before starting the next. Qwen repeats
validate the merged branch; earlier Qwen evidence remains preserved separately.

```bash
nvidia-smi -i 1
bash code/train/smoke.sh veronika code/configs/v2/hydraT_s1337.yaml
bash code/train/smoke.sh veronika code/configs/v2/hydraR_s1337.yaml
bash code/train/smoke.sh veronika code/configs/e2/hydraT_s1337.yaml
bash code/train/smoke.sh veronika code/configs/e2/hydraR_s1337.yaml
```

Launch long jobs using nohup and a unique log in Veronika's folder. Never extend
the script's one-hour training cap or bypass its smoke lock/memory check.
For MedGemma expect 238 targets and six heads, stored counts 55,287,808 (T) /
49,020,928 (R), active 14,901,248. The registered Gemma3 template, image-token
type fix, native 896-square processor resize and EOS settings come from the
merged port; no method-specific recipe override is used. Prompts/data are
unchanged. A template change requires the existing `template_gate.py` again.

Full training configs are `code/configs/v2/hydra{T,R}_s{1337,2026,42}.yaml` and
`code/configs/e2/hydra{T,R}_s{1337,2026,42}.yaml`. Only Taha queues them after
the applicable gate/review. Completing a smoke does not finish these experiments.

## E7: analysis without model loading

After full checkpoints exist, run on CPU in your own output folder. No holdout
inference is needed for head cosines. Cap threads on the server:

```bash
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
python code/analysis/hydra_heads.py --run_dir <completed-run> \
  --out <own-analysis>/heads.json --head_map_out <own-analysis>/head_map.json
python code/analysis/representations.py heads \
  --state <completed-run>/method_state.safetensors \
  --mapping <own-analysis>/head_map.json --out <own-analysis>/alex_heads.json
```

Use `--group_by decoder_layer` when exporting a map to concatenate factors within
each decoder block. The default keeps each target separate. Parameter names are
read directly from the checkpoint; no assumed Qwen layer count or Gemma names.
Null denotes zero-norm cosine. Analyze seeds separately; the raw factor basis
and random initialization limit interpretation. Do not present smoke-head patterns
as E7 findings from trained models. Alex's separate representation/probe work
remains under his ownership.

## Manuscript and review

`paper/hydra_sections_draft.md` supplies methods, head-analysis definitions and
explicit full-result placeholders for Taha's v2 rewrite. After the full runs,
add reproducible E7 findings to the ledger through a PR and replace result
placeholders. No main manuscript rewrite, merge or full training is done here.
