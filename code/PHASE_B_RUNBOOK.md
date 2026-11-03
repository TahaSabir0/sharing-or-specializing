# Alex: complete E3–E8 code workflow

E4/E5 keep [their existing runbook](E4_E5_RUNBOOK.md) and `e4_e5.sh`.
The remaining implementation is entered through `bash code/train/phase_b.sh`.
These commands do real full-surface inference/analysis. They do not substitute a
50-step smoke run for an experiment.

**Alex:** CPU preparation, collection and analysis. **Taha:** router training,
real holdout inference, CRIMSON and feature extraction on GPU 1, after Phase A
training, review/merge and the applicable method gates (PLAN §7.3).
Local synthetic tests are not evidence of a real-model Blackwell pass.

## 1. Alex: pull, audit and prepare

From a fresh Blackwell shell, with no queue running from this checkout:

```bash
cd /data/ugrip2026/journal/people/thinh/repo &&
git pull --ff-only origin alex/modality-decks

export JOURNAL_SOURCE=/data/ugrip2026/journal/data
export JOURNAL_DATA=/data/ugrip2026/journal/people/thinh/group_decks
export JOURNAL_RUNS=/data/ugrip2026/journal/runs
export PHASE_B_DIR=/data/ugrip2026/journal/people/thinh/phase_b
export PHASE_B_GPU_OUT=/data/ugrip2026/journal/runs/phase_b

bash code/train/phase_b.sh preflight e3 &&
bash code/train/phase_b.sh e3-prepare &&
bash code/train/phase_b.sh e3-check
bash code/train/phase_b.sh e7-probe &&
bash code/train/phase_b.sh e7-prepare &&
bash code/train/phase_b.sh e7-check
```

Wrappers activate `flare-blackwell-v5` before Python and cap CPU threads at four.
Use the same environment paths for every stage and every operator. If Taha uses a
different checkout/path, regenerate the plans there: they record absolute paths
and input hashes. `preflight` reports missing packages/assets without modifying
the environment. Use `preflight e3` for E3 alone: it does not require the E6
router packages (`gliclass`, `torchvision`) or grouped decks. Plain `preflight`
still checks all Phase B dependencies and must pass before the later stages.
A missing `gliclass` blocks E6 task-router work, not E3. The frozen shared environment must not be changed to install
pytest or router dependencies. Any additional environment is provisioned by Taha.

Expected plans:

| Experiment | New jobs | What is reused |
|---|---:|---|
| E3 | 36 row jobs = 180 off-diagonal task slices; 30 CRIMSON jobs | E1/E2 diagonals and both E0 anchors |
| E4/E5 | Existing 102 training jobs plus eval/scoring | Existing `e4e5_jobs/plan.json` |
| E6 | 24 full-holdout evaluations + CRIMSON | 4 oracle arms × 2 backbones × 3 seeds |
| E7 | 164 forward-only configurations | All trained adapters; two base-model configurations |
| E8 | CPU only | Existing E0/E1/E2 predictions plus E3/E7 analysis |

Three chains are queues divided by seed, not three experiments. E3 has 12 row
jobs per chain. E7 includes every individual TaskLoRA, modality and dataset
adapter on **all** probe questions; Hydra uses oracle task heads.

`probe_ids.json` contains only the frozen IDs, counts and source hash. Commit
that JSON under `data/e7_probe_ids.json` before feature extraction. There are
30 questions per cell except smaller cells, which use all available questions
(BUS-UCLM detection has 22 in the planned census). Expected total is 562, not a
fabricated 570. Selection uses a fixed salted SHA256 order and does not inspect
answers or model predictions. The local workspace contains no real holdout
deck, so actual probe IDs must be generated from the frozen Blackwell deck.

```bash
cp "$PHASE_B_DIR/probe_ids.json" data/e7_probe_ids.json
git add data/e7_probe_ids.json
git commit -m "Freeze E7 probe IDs from the journal holdout"
git push origin alex/modality-decks
```

## 2. Taha: E3, durably

**Changed at review (Taha, 3 Nov):** E3 inference runs through the usual chain runners, not
`phase_b_jobs.py`, and each backbone uses its baselines' evaluator (research rule 3, as E4/E5):
Qwen `eval_arm.py` (E0/E1 were made with it), MedGemma `eval_harness.py --batch_size 1`.
Job files: `code/eval/jobs/e3/` (`python code/eval/plan_e3_jobs.py`). Each TaskLoRA is evaluated
on the **full** holdout into `/data/ugrip2026/journal/runs/e3/<name>/` (Qwen
`predictions_holdout.json`, MedGemma an eval_harness folder). The on-task slice repeats the
E1/E2 diagonal: the official diagonal stays E1/E2, the repeat is a reproducibility check.

```bash
bash code/eval/jobs/e3/launch_e3.sh   # from /data/ugrip2026/journal/repo; Taha only
```

Three slots, one per seed: Qwen evals, then MedGemma evals, then that seed's CRIMSON
(`crimsonchain_e3_s<seed>.log`; 5 non-report adapters x 2 backbones). Logs:
`runs/evalchain_e3_{q35,mg15}_s<seed>.log`.

**For Alex:** `plan_e3.py collect` still expects `phase_b_jobs.py` outputs (receipts,
`run_meta.json`, off-diagonal-only folders). Point it at the folders above: slice each to the
off-diagonal tasks (the diagonal comes from E1/E2), and read the Qwen legacy layout
(`group_results.assemble` already does).

Alex, once all three chains and CRIMSON are complete:

```bash
bash code/train/phase_b.sh e3-collect &&
bash code/train/phase_b.sh e3-analyze
```

Results: `$PHASE_B_DIR/e3_results/transfer.json` and two PDF heatmaps. Includes
raw oriented C, task/cell PGF, two-level CIs, Holm across target tasks, three-seed
Wilcoxon diagnostics, ZS-bare sensitivity/sign flips, transferability and
malleability, format/truncation rates, covariates and the iugc comparison.
Reports from non-report adapters remain scored and flagged.

## 3. Taha: E6 routers and predicted routing

Retrain from a clean public GLiClass initial checkpoint and an ImageNet ResNet18
state dictionary. The old challenge FLARE router cannot be assumed holdout-clean.
Provide **existing local** assets; these scripts never download models:

```bash
export GLICLASS_INITIAL=/absolute/path/to/clean/gliclass-small-v1.0
export RESNET_INITIAL=/absolute/path/to/imagenet/resnet18-state-dict.pth
```

These are deliberately explicit asset paths, not claims that either asset is
already present on Blackwell. `preflight` checks paths if the variables are set.
Use a durable session, with a slot scheduled by Taha:

```bash
tmux new -s e6-routers
set -o pipefail
{
  for seed in 1337 2026 42; do
    bash code/train/phase_b.sh e6-train-task "$seed" || exit
    bash code/train/phase_b.sh e6-train-modality "$seed" || exit
  done
} 2>&1 | tee "$PHASE_B_DIR/e6_routers.log"
```

Each router records training/input hashes, versions, recipe, seed, loss and
checkpoint hashes. GLiClass receives **question text only**; ResNet receives
pixels only, with training images deduplicated. The split audit checks every
image, including partial overlap between multi-image questions. Neither router
uses dev/holdout for selection. Train each seed once and reuse it across both
VLM backbones.

Alex can generate routes on CPU, then prepare the jobs (CPU inference on all
images can take time; tmux is appropriate here too):

```bash
bash code/train/phase_b.sh e6-predict &&
bash code/train/phase_b.sh e6-prepare &&
bash code/train/phase_b.sh e6-check
```

Taha runs each scheduled E6 chain exactly as E3, replacing the stages with
`e6-eval 1` and `e6-crimson 1` and using a new tmux session/log. These jobs need
Veronika's reviewed `methods.hydra` plugin/checkpoints and the completed E4
modality adapters. A wrong predicted route changes the selected adapter/head;
the original question, answer, dataset and scoring task are preserved.

Alex then runs:

```bash
bash code/train/phase_b.sh e6-collect &&
bash code/train/phase_b.sh e6-analyze
```

`e6_results/routing.json` reports per-label confusion/accuracy and the metric
drop against oracle routing with three-seed uncertainty. Positive drop means
predicted routing is worse. There is no unplanned Shared fallback threshold.

## 4. Taha: E7 forward passes; Alex: analysis

Once all required adapters exist and probe IDs are committed, start the three
scheduled queues in durable sessions. Example for chain 1:

```bash
tmux new -s e7-chain1
set -o pipefail
bash code/train/phase_b.sh e7-run 1 2>&1 | tee "$PHASE_B_DIR/e7_chain1.log"
```

E7 makes no optimizer steps and generates no answers. Hooks pool image tokens,
question tokens and the last prompt token. The question mask comes from the
same processor's character offsets, after image-token expansion. Decoder
layers are numbered from one; Qwen uses the proposed 1/8/16/... list up to its
actual depth plus the final layer. MedGemma uses 1/4/8/.../32 plus its final
layer. Actual decoder depth, hook paths, layers and token counts are recorded;
nonexistent proposed layers are not invented.

```bash
bash code/train/phase_b.sh e7-analyze
```

`e7_results/representations.json` includes input/transfer correlations with
target-cell permutations and Holm across layers, descriptive adapter-pair CKA,
shift from base, RSA and seed-pair noise ceilings. All three pooling views are
reported. It also exports the cell predictors used by E8.

Hydra head analysis uses Veronika's actual state tensor layout. The CPU command
accepts an explicit reviewed mapping, so it does not guess whether a packed
tensor dimension indexes heads or LoRA rank:

```bash
source /data/ugrip2026/miniconda3/etc/profile.d/conda.sh
conda activate flare-blackwell-v5
python code/analysis/representations.py heads \
  --state /path/to/run/method_state.safetensors \
  --mapping /path/to/hydra_head_mapping.json \
  --out /path/to/head_similarity.json
```

Mapping schema: `{"mode":"hydraT","layers":{"layer_name":{"classification":
[{"parameter":"actual_B_tensor_name","axis":0,"index":0}], ...}}}`.
Every layer must map all six full task names. A separate tensor per head can
instead be referenced by its string name. For Hydra-T select B heads; for
Hydra-R select A heads. Run on both backbones and all three seeds; retain the
state and mapping hashes. Final mappings depend on the actual Hydra plugin,
which is not yet present in this branch.

## 5. Alex: E8 synthesis

After E3/E7 analysis and the Phase A holdout/CRIMSON artifacts exist:

```bash
bash code/train/phase_b.sh e8-prepare &&
bash code/train/phase_b.sh e8-analyze
```

`e8_results/synthesis.json` and `cells.csv` contain all 19 cells for both
backbones, TaskLoRA/Hydra-T/Hydra-R versus Shared, candidate predictors,
permutation Spearman tests, Holm-adjusted values and cross-backbone sign
agreement. The primary cross-cell outcome uses PGF (PLAN §1.8) so MAE units are
not ranked directly against accuracy; raw oriented differences are retained.
ZS-bare sensitivity recomputes outcome normalization and E3 aggregates.
Categorical output structure/modality are one-hot encoded, not arbitrary ranks.

## Statistical choices and boundaries

- E3 follows [Sachdeva et al.](https://arxiv.org/abs/2511.18787) as recorded in
  `references/papers/METHOD_DIGEST.md`: signed transferability/malleability use
  `(1-exp(-p/N))*mean(signed PGF)`. Primary excludes the diagonal (N=5); N=6
  including-diagonal sensitivity is also saved. Perfect anchors make PGF
  undefined. Nonfinite numeric metrics are recorded as null with diagnostics,
  never removed to improve an average.
- E3 covariates use all 720 joint task-label permutations, preserving dependence
  among the 30 pairs. All six output-structure labels differ, so equality is a
  constant off-diagonal predictor and is explicitly reported as untestable.
- E7 implements unbiased linear CKA following [Kornblith et al.](https://proceedings.mlr.press/v97/kornblith19a.html).
  Negative debiased estimates are retained. The primary input measure is centroid
  cosine; supplementary RSA compares source/cell similarity profiles to fixed
  cell anchors. Adapter-pair comparisons remain descriptive. Same-task input
  groups share probe observations; this self-overlap must be stated in reporting.
- Router recipes are fixed in code before evaluation: GLiClass 3 epochs,
  batch 16, AdamW 2e-5; ResNet18 5 epochs, batch 16, 2048-square, AdamW 1e-4,
  class-weighted CE with smoothing .05 and horizontal flips. Multi-image
  modality predictions average softmax probabilities before argmax.
  GLiClass training uses the project's [documented dataset/collator API](https://docs.knowledgator.com/docs/frameworks/gliclass/training/).
- Output directories are fresh-only. To rerun an analysis, use its Python CLI
  with a new `--out`. `BOOTSTRAPS` changes analysis resamples, not training or
  evaluation length; published analyses use the default 10,000.

## Verification

Run CPU tests in a local test environment that already contains pytest:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  code/tests/test_group_decks.py code/tests/test_group_workflow.py \
  code/tests/test_backbones.py code/tests/test_transfer_matrix.py \
  code/tests/test_phase_b.py code/tests/test_phase_b_analysis.py code/tests/test_crimson_layouts.py
```

Coverage includes full synthetic E3 collection, all 164 E7 configurations,
19-cell E7/E8 analyses, actual tensor-hook pooling with a fake backbone,
predicted-route evaluator execution with a fake model, and integrity failures.
Real GLiClass/ResNet training, both VLM feature passes and Hydra integration
still require Blackwell validation. The workstation lacks transformers and
the real decks/checkpoints; CPU passes do not close those runtime gates.
