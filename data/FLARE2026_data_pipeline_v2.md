# FLARE 2026 Task 3 — Internal Dataset Pipeline (v2)
### Scoring-aware cleaning, splitting, storage, and multi-image handling

*Supersedes v1 (`FLARE2026_data_pipeline.md`, "Internal Data Cleaning & Splitting"). The v1
three-tier split (`train/dev/test_internal/experiment`) is replaced here by a 50/50
cross-validation geometry, the storage is made lossless, and multi-image packing becomes a tested
knob rather than a baked-in default. **Status:** dataset/split layer design-locked and **built**
(`internal_splits_v2/` live); §8 training-setup and §9 results **filled with Phase-1 run #1** (classification
packing sweep — ran end-to-end, pipeline works, result inconclusive pending an epoch-control rerun); §10
inference still a placeholder. Date: 2026-06-11.*

---

## 1. Overview & philosophy

We get **one real test submission** and **four people iterating in parallel**. To pick the best
method without burning that submission — and without four of us silently measuring on four different
cuts of the data — we build **one** frozen, cleaned, leak-free dataset that everyone trains and
evaluates on. This document describes how it is produced, why each decision was made, and how to use
it.

What changed since v1, and why each change happened, is the spine of this document:

1. **The split geometry is now 50/50 cross-validation** (§4) — a direct directive from Prof. Yaqub
   (2026-06-10 meeting): the way to know a method *generalizes* (rather than overfits a single cut)
   is to train on one half, test on the other, then flip, and check the two results agree.
2. **Evaluation is now per (dataset × task)** (§2) — forced by how the challenge actually ranks
   teams; the v1 habit of pooling one metric per task type does not match it.
3. **Storage is lossless and concern-separated** (§5) — v1 destroyed image paths past the cap *in
   the data files*. v2 keeps the dataset pure (every image, no cap) and pushes all
   tile/token/packing logic downstream where it belongs.
4. **Multi-image packing is a knob we test, not a default we assume** (§6) — Dr. Mai's report that
   multi-image training *hurt* last year means our K=6 grouping has to be *measured* against the
   flatten baseline, not adopted on faith.

The organizing principle behind 3 and 4 is **separation of concerns**: the dataset is the ground
truth of *what each question is* (its images, its answer); how those images get packed into training
rows or tiled into tokens is the job of the layers above it, not the data.

**Where everything lives.** Shared outputs sit in the team dir on the cluster (read/write for all
four of us); code is in our repo on branch `Data_Set_Dev`.

| What | Where |
|---|---|
| Raw 2D dataset | `…/shared/datasets/FLARE-Task5-MLLM-2D/` |
| **The splits (v2)** | `…/shared/internal_splits_v2/` *(v1 preserved alongside)* |
| Audit report | `…/shared/data_audit_v1/` |
| Training file(s) | `…/shared/sharegpt_v2/…` *(one per K-arm — see §6)* |
| Code | `internal_eval/`, `llama_factory_internvl/scripts/` |

---

## 2. How the competition scores us (and why it shapes the design)

Every design choice below is downstream of how FLARE actually computes a score, so it's worth being
precise — especially because the public description and the reference code appear to disagree, and
resolving that disagreement is what tells us how to evaluate internally.

**The per-task metrics.** Each task type has its own metric: Classification → Balanced Accuracy;
Multi-label → micro-F1; Detection → F1 at IoU > 0.5; Counting → MAE; Regression → MAE; Report
Generation → CRIMSON.

**The ranking is rank-based, not score-based.** Per the EVAL page, FLARE follows ChallengeR: for each
testing case and each metric, *rank all participants*, then *average the ranks*. Lowest mean rank
wins. You are not scored on your raw accuracy; you are scored on what *place* you come in, averaged
over many little contests.

**The apparent contradiction, and its resolution.** The page says ranking is "per testing case," but
last year's reference `evaluation.py` computes each metric **pooled per task type** (all
classification samples across all datasets → one balanced accuracy). These can't both be literally
true at the image level, because balanced accuracy and micro-F1 are **set-level** metrics —
mathematically degenerate on a single sample. The resolution is that two *different stages* are being
conflated:

- **Stage 1 — metric computation** (what `evaluation.py` does, what produces your numbers): set-level
  metrics over groups of samples.
- **Stage 2 — cross-team ranking** (ChallengeR, server-side): ranks teams on those values and
  averages ranks.

So "case" in the ranking text cannot mean a single image; it denotes an *evaluation unit* (a set).
**What that unit is — per task type (~6 units) or per dataset (~20 units) — is still unconfirmed.**
The reference code implies per-task-type; our prior working note assumed per-dataset; we can't close
it from the public materials. It will be settled by the validation-leaderboard submission (planned
next week) and Dr. Tausifa's question to the organizers.

**Decision: we evaluate internally at the finest well-defined level — per (dataset × task) — and
aggregate up.** This is strictly more informative than per-task-type pooling (you can always re-pool;
you can't un-pool), it is robust whichever way the unit question resolves, and it matches the real
ranking better than v1's pooled metrics. It also explains Mai's "**classification first**" guidance
mechanically: classification spans the most datasets, so it contributes the most ranking units —
winning those moves your mean rank the most.

---

## 3. Cleaning the data (carried from v1, unchanged)

The auditor (`audit_data.py`) defines what a *valid* sample looks like for every task type and flags
anything that violates the schema, plus subtler problems that are schema-valid but still wrong
(cross-split leakage, contradictory labels, coordinate-convention drift, outlier labels). Crucially
it **never deletes anything**: it produces a candidate list with reason codes, and the splitter
(`build_splits.py`) removes only the categories we agreed on. Over-cleaning (tossing hard-but-correct
samples) hurts more than it helps, so the bias is toward keeping data.

**What we remove from training.** After triage, **73 samples** are quarantined into
`flagged_removed.jsonl` and never enter any training-derived split:

| Dataset | Problem | # | Why it is a bug |
|---|---|---|---|
| endo | `'T'`=Tumor as the answer | 36 | answer is option *text*, not the offered letter |
| endo | multi-letter "select-all" mislabel | 11 | e.g. `A B` on a single-label question |
| BUSI-det | inverted / zero-area boxes | 17 | geometrically invalid, cannot produce a valid IoU |
| periapical | contradictory labels | 7 | same image+question, conflicting answers |
| periapical | exact duplicate | 1 | byte-identical repeated sample |
| boneresorption | regression outlier | 1 | label 2085 vs median 28 (≈75×) — bad label / unit error |
| **Total removed from training** | | **73** | |

The regression-outlier rule carries a **10× median guard** (a real unit error is orders of magnitude
off; iugc's wide-but-real 144–151 mm tail, only ≈1.8× median, is correctly *not* flagged).

**Cross-split leakage (Strategy A).** 229 images are byte-identical between `training` and
`validation-public`, hidden behind renumbered filenames (concentrated in the retino / BUSI / BUS-UCLM
cells). Policy: **keep** training on them, but **drop them from the pseudo-test** so we only score
images the model has never seen. The excluded images are listed in `validation-public-leaked.json`.

**Metadata we don't trust: the `Modality` field.** It is inconsistently cased (`Dermatology` vs
`Dermoscopy`, `X-Ray` vs `x-ray`), so we derive **modality and dataset from the folder path**
(`<split>/<Modality>/<dataset>/`), the reliable source. **Task** comes from the per-sample `TaskType`
field, and the `task_shape_mismatch` audit (which checks the declared task matches the answer's
shape) came back clean.

**One change in spirit from v1:** cleaning removes bad *records*, but it **never truncates images**
(see §5). A record is either kept whole or removed; we no longer silently amputate part of a sample.

---

## 4. Splitting the data (v2)

### 4.1 The protocol: 50/50 cross-validation

Prof. Yaqub's directive (2026-06-10): the way to tell whether a method *generalizes* rather than
overfits a particular cut of the data is to **train on one half, test on the other, then flip the
halves and repeat.** If the two runs agree, the method is generalizable and we can confidently scale
it to the full dataset; if one half scores well and the other poorly, we're overfitting and likely to
fail on the hidden test. For the internship we cap training at ~50% of the data (favoring *more
experiments* over *more data*); only the 1–2 finalist pipelines get scaled to the full set before the
real submission.

### 4.2 The geometry

We carve the cleaned data into a small untouched test plus two symmetric halves:

```
full cleaned data (image-grouped, stratified by dataset × task)
├── true_test       10%   — untouched, never trained on (in-distribution objective test)
├── Set A           45%   = A_train (90%) + A_dev (10%)
│      └── A'  = 30% of A     (interchangeable quick set)
├── Set B           45%   = B_train (90%) + B_dev (10%)
│      └── B'  = 30% of B     (interchangeable quick set)
└── validation-public      — untouched (competition distribution); raw + clean + leaked views
```

| File | Tier | Use it for |
|---|---|---|
| `true_test.jsonl` | TEST (in-distribution) | objective final internal check; never trained on |
| `A.jsonl` / `A_dev.jsonl` | half A (train / dev) | fold-1 training; dev = checkpoint selection |
| `B.jsonl` / `B_dev.jsonl` | half B (train / dev) | fold-2 training; dev = checkpoint selection |
| `A_prime.jsonl` / `B_prime.jsonl` | 30% quick sets | fast iteration; interchangeable train/test |
| `validation-public.jsonl` | PSEUDO-TEST (competition dist.) | 2nd honest check; mirrors real eval |
| `validation-public-clean.jsonl` | (bugs removed) | the achievable ceiling; shows label-noise cost |
| `flagged_removed.jsonl` | (quarantined) | bad samples — do *not* use |

**Why each piece exists:**

- **`true_test` (10%, untouched).** The A↔B flip trains on *both* halves across the two folds, so
  neither half is a pristine never-trained set. `true_test` is carved out *first* and never trained
  on in either fold, giving one objective in-distribution yardstick both folds report on.
- **`A` / `B` (45% each).** The two CV halves. 10% + 45% + 45% = 100%; training on one half is ~45%
  of all data, respecting the "cap at ~50%" guidance.
- **`A_dev` / `B_dev` (10% within each half).** Checkpoint selection / early stopping has to happen
  on data that is *not* the test half — otherwise the test half is contaminated. So each half holds
  out a small internal dev slice. "Train on A" means "fit on A_train, choose the model on A_dev,
  score on B."
- **`A'` / `B'` (30% each, interchangeable).** Miniatures of the main cycle for fast overnight
  iteration: train on A', test on B' (or flip). Because A ∩ B = ∅ at the image level, A' and B' are
  automatically leak-free against each other.
- **Two untouched holdouts.** `true_test` (training distribution) and `validation-public`
  (competition distribution) are complementary: the first answers "how do we do on unseen
  in-distribution data," the second "how do we do on the organizers' distribution." This satisfies
  both Yaqub's "keep a diverse final test" and the team's "objective true test."

### 4.3 Image-level, stratified, reproducible

We split by the **image group**, not the row: some datasets (e.g. iugc) reuse the same image across
classification/detection/regression, and some samples are multi-image, so splitting per row would
leak an image across A and B. All questions on one image therefore stay in the same split. The split
is **stratified by (dataset × task-combo)** at image-group granularity with a fixed seed, so every
cell is proportionally represented and the result is identical for all four of us.

*(Note: stratification works at the QA-pair level and is completely independent of the multi-image
packing in §6 — K only changes how many training *rows* a pair becomes downstream, never which split
it lands in. This is why the split can be built and frozen before the multi-image questions are
settled.)*

### 4.4 The rules that keep it honest

- Never train on `*_dev`, `true_test`, or `validation-public`.
- Tune freely on `*_dev`; touch `true_test` / `validation-public` *rarely* (milestones only) — every
  peek contaminates them a little.
- Never use anything in `flagged_removed.jsonl`.
- Only at the very end, for the final production model, do we fold `validation-public` back into
  training (no more decisions left to protect).

### 4.5 Dropped from v1

The standalone `test_internal` and `experiment` tiers are gone (the A↔B flip + A'/B' replace them).
The **rare-cell floor** discussed during design was dropped: at 30% the quick sets are already
reasonably populated, and the floor's value hinges on the still-unconfirmed per-dataset ranking unit
(§2), so it's not worth the added machinery yet.

---

## 5. Storage & separation of concerns

### 5.1 The three layers

The single most important structural idea in v2: **each layer is ignorant of the one below it.**

1. **Dataset / splits** (`build_splits.py`) — *pure content.* A QA pair → `{set, question, answer,
   ALL image paths, metadata}`. No cap, no chunking, no tiles, no token budget. A pair with 18 images
   simply *has* 18 images. This is the source of truth.
2. **Training-prep / export** (`export_sharegpt.py`) — takes a split file **+ K** and emits the
   ShareGPT rows, applying the `ceil(X/K)` packing *here* (§6). K exists only at this layer, and only
   for training.
3. **Runtime** (train + inference) — the tile cap (`max_num`) and context length live here, applied
   by the model's image processor when images become tokens. Never written to a data file.

### 5.2 The lossless fix (a real v1 bug)

v1's `build_splits.py` applied the image cap *while materializing the split files*
(`r["images"][:6]`), so any QA pair with more than 6 images had the extra image paths **destroyed on
disk**. The split files were lossy. v2 stores **every** image path and caps nothing; the cap (now a
*chunk size*, §6) moves to the export layer. This both honors the separation above and removes silent
data loss from the source of truth.

### 5.3 Schema and identity

Each split is a **self-contained JSONL** — every line carries its content inline; no index, no
manifest lookup, no re-walking the raw tree:

```python
import json
for line in open("…/internal_splits_v2/A.jsonl"):
    r = json.loads(line)
    r["question"]    # the question text
    r["image"]       # list of ALL absolute image paths, ordered, UNCAPPED
    r["answer"]      # the ground-truth target
    r["task"]        # classification / detection / counting / …
    r["dataset"]     # e.g. BUSI, bcn20000   (per-dataset scoring)
    r["modality"]    # e.g. Ultrasound       (from the folder path)
    r["sample_key"]  # str(ImageName)||Question — the identity; matches evaluation.py
    r["flagged_bug"], r["leaked"]   # cleaning flags
```

`sample_key` is the only identity — it equals the key `evaluation.py` already uses, so predictions
join 1:1 to ground truth, and it's recomputable from the data (no lookup table). Cleaning verdicts
ride along as fields, so filtering to "clean training" is just dropping flagged rows.

---

## 6. Multi-image handling — the K knob

### 6.1 Why this is tested, not assumed

32% of training and 65% of validation-public samples are multi-image, dominated by n=2 (paired
radiology views: IU_XRay frontal+lateral, CMMD mammography), concentrated in classification and
report generation. v1 chose grouped multi-image (K=6). But Dr. Mai reported that feeding multiple
images per question at *training* gave **worse** inference results last year (single-image flatten let
the visual encoder focus). So grouping is no longer a settled default — it is a hypothesis we
**measure** against the flatten baseline.

### 6.2 The knob

K = **chunk size**. A QA pair with X images becomes **ceil(X / K)** training rows, each row fed up to
K images at once, **no image ever dropped**:

- **K=1** → X rows of 1 image each = **last year's flatten** (the incumbent baseline).
- **K=2**, X=2 → 1 row, both images fed together.
- **K=6**, X=12 → 2 rows of 6.

K lives only in the export step and only affects **training**. At **inference** a case has X images
and exactly one expected answer, so it is fed its images once (up to a runtime budget, §7) and
produces one answer — there is no chunking at test time.

### 6.3 The experiment (as run)

Train on **A'** classification, evaluate balanced accuracy **per dataset** on **B'**. Arms: **k1**=flatten
(K=1), **k2**=group(K=2), **mix**=grouped+individual (MANTIS-style). K=6 was dropped — the §3 census shows
classification multi-image is X=2 pairs (CMMD) + X=3 (neojaundice), so K>3 has no data to act on. **endo**
excluded (33 samples, X→105 multi-frame video, label-noisy + row-multiplication). Markers ON for all arms
(per-image `Image-N:`, the §0a fix). Results — run #1 (inconclusive, epoch confound) + run #2 (corrected,
conclusive: neojaundice grouping +0.13, per-dataset-K confirmed): **§9**.

### 6.4 The real conclusion — K should be **per-dataset**, set from image redundancy

Task-dependent K was a proxy; the variable that actually decides flatten-vs-group is **how a dataset's own
images relate** (Phase-0C): high intra-sample redundancy ⇒ flatten loses nothing (CMMD's 2 views ~0.90
similar); complementary images ⇒ grouping should help (neojaundice's 3 regions ~0.65). Same task
(classification), opposite optimal K — so a single global K is wrong. The export layer already supports
this (K is a per-record lookup) and it's **free at inference** (packing is training-only; inference feeds
the case's images once regardless). Plan: read run #1 (and the flip) **per dataset** → a per-dataset K
table; the deliverable is a single "per-dataset-K" model confirmed to beat both uniform arms (Phase-1.5).

---

## 7. Hardware & runtime constraints (measured)

A GPU capacity probe (`mem_probe.py`) reframed what the multi-image constraint actually *is*. Sweep:
image-count {1,2,4,6,8,12,16} × tile-cap `max_num` {12,6,1} on InternVL3-8B (4-bit nf4, matching
`inference.py`). Weight footprint 5.34 GB. Cell = peak GB / latency s:

| N images | max_num=12 | max_num=6 | max_num=1 |
|---|---|---|---|
| 1  | 6.02 / 1.33 | 5.56 / 0.40 | 5.52 / 0.39 |
| 2  | 6.51 / 0.88 | 5.73 / 0.47 | 5.56 / 0.40 |
| 4  | 7.83 / 1.54 | 5.97 / 0.61 | 5.65 / 0.45 |
| 6  | 9.00 / 2.21 | 6.46 / 0.85 | 5.73 / 0.47 |
| 8  | 10.16 / 2.97 | 6.77 / 1.00 | 5.81 / 0.52 |
| 12 | 12.64 / 4.89 | 7.58 / 1.42 | 5.98 / 0.61 |
| 16 | 15.12 / 7.12 | 8.19 / 1.75 | 6.16 / 0.71 |

**What it means:**

- **VRAM is a non-issue.** The 4-bit 8B model is tiny: even 16 images at full tiling peaks at 15 GB,
  against the **48 GB** eval budget (A6000 Ada). There is no memory cliff anywhere.
- **The binding constraints are latency and context length.** Latency: 16 images @ max_num=12 = 7.1
  s/case — the 1.5× baseline-runtime rule bites long before memory. Context: N≥4 @ max_num=12 already
  overflows ~8k visual+text tokens, at which point attention degrades and answers garble. This —
  *not* VRAM — is the likely reason multi-image hurt last year.
- **Memory scales with total *tiles*, not image count; `max_num` is the lever.** Each image is tiled
  by resolution (up to ~12 tiles, ~256 tokens each), so resolution and count trade against a shared
  token budget. At N=6, max_num 12→6→1 cuts latency 2.2→0.85→0.47 s.

**Consequences for the pipeline:**

1. **High-K arms need a reduced tile cap** to stay inside the context window — tile-cap and K are
   coupled, set per arm.
2. **The real context length must be confirmed** — the probe assumed 8192; the v1 `build_splits`
   comment assumed 2048 (a 4× disagreement that changes how aggressive the tile cap must be). See §12.
3. **Not crashing the 48 GB eval machine on the worst-case input is our responsibility and is
   explicitly rewarded** (the "≤4 GB peak ⇒ perfect GPU-Time score" tip). Tile-capping, downscaling,
   and limiting images-per-pass are expected engineering, not workarounds.

---

## 8. Training setup

**Base + method (held constant across arms so packing is the only variable):** InternVL3-8B-hf, **QLoRA**
4-bit nf4, LoRA rank 8 / target all, via **LLaMA-Factory** (`examples/train_qlora/flare_cls_phase1.yaml`),
template `intern_vl`, effective batch 8 (2 × grad-accum 4), lr 2e-4 linear, **seed 1337**, `report_to: none`
(headless batch has no WandB key). One classification LoRA adapter per arm (no GLiClass routing at train
time — Phase-1 is classification-only).

**`cutoff_len: 8192`** (raised from 2048→4096→8192). Phase-0C measured a CMMD pair = ~6,720 tokens at full
tiling; 4096 would truncate and *break image-token alignment*. 8192 fits CMMD pairs + typical bcn/neojaundice
rows at full `max_num`. **Tile caps are NOT needed for Phase-1**: K-chunking bounds images-per-row to ≤K, so
the scary per-*sample* token counts (bcn/endo, §7/§0C) never appear per *row*. The tile-budget packer stays
deferred until report-gen (high-res X-ray pairs) / 3D.

**The control — match training effort across arms.** Flattening multiplies rows (a pair → 2 rows), so at
equal epochs the flatten arm gets more gradient updates *for free*. Run #1 matched **steps** (`max_steps=1020`)
to neutralize that — **but that backfired**: equal steps over unequal row counts = **unequal epochs**
(k1 2.1 / k2 3.0 / mix 1.6), and the result tracked epochs, not packing (§9). **Corrected control for the
rerun: match EPOCHS** (3/arm, drop `max_steps`) so each arm learns its data the same number of times and the
single-image control datasets become comparable; add 2–3 seeds for error bars.

**Open (Yaqub):** re-evaluate the base VLM (newer ≲8B efficient VLMs) before scaling; task-token injection
(Alex's thread) is separate from this packing study.

---

## 9. Experiments & results

### 9.1 Phase-1 run #1 — classification packing (2026-06-11)

Full per-dataset table + numbers live in **`STATS.md` §11**. Headline:

- **All 3 arms trained + scored end-to-end** (~53–66 min each) — the pipeline (export → QLoRA → per-dataset
  balanced-accuracy eval) works. This was the real win of the run; first training ever on this cluster.
- **Result is INCONCLUSIVE** because of the §8 epoch confound: macro means (k2 0.548 > k1 0.541 > mix 0.517)
  track *epochs trained* (3.0 > 2.1 > 1.6), not packing. The **single-image control datasets** — which
  packing cannot affect (their rows are identical across arms) — **swing up to 0.20** (dental 0.534→0.334),
  a noise floor bigger than the effect. So no per-dataset delta is trustworthy yet.
- **Directional signal matches the §6.4 hypothesis** (but within the noise): **neojaundice** (complementary)
  grouping 0.53→0.66→0.69; **CMMD** (near-dup) marginal 0.35→0.37; **bcn20000** (redundant) flat ~0.35.

### 9.2 Phase-1 run #2 — CORRECTED (2026-06-12) ✅ conclusive & positive

Reran §9.1 **epoch-matched** (3 epochs/arm, `max_steps` dropped) with **3 seeds** {1337, 42, 7} = 9 jobs.
Full per-dataset table (mean ± half-range) in **`STATS.md` §11b**. Headline:

- **Epoch fix worked — controls now stable.** The single-image control datasets' across-arm spread
  collapsed from run #1's ~0.20 to seed-noise (dental 0.200→0.032, bone_marrow 0.179→0.020). So multi-image
  arm deltas are now real packing effects, not training-duration artifacts.
- **neojaundice (complementary): clean +0.13–0.14 from grouping** — k1 0.543 → k2 0.674 → mix 0.687,
  monotone + seed-tight (≤0.04). **CMMD (near-dup) and bcn20000 (redundant): flat** — grouping inert.
  Both halves of the §6.4 hypothesis confirmed.
- **macro means barely move** (0.559/0.557/0.572) because the effect is *local* — multi-image is a minority
  of cls mass. **Conclusion: optimal K is per-dataset (from intra-sample redundancy), not global.** `mix`
  is the safe global default.
- **CV flip (BA) skipped** — controls-stability + 3-seed tightness make the result trustworthy without it.

*(Pending: comparison vs the frozen ME-VLIP baseline; report-gen / 3D packing; thin-cell caveats —
neojaundice n=80.)*

---

## 10. Inference pipeline

**Task routing.** GLiClass reads the question text → task type; if score ≥ 0.3, set the matching
task-specific LoRA adapter, else the general adapter (unchanged from last year / `docker_internvl/
inference.py`).

**Multi-image feeding rule (VERIFIED inference-safe, 2026-06-12 — job 783, PROGRESS 2026-06-12).**
At the Docker test the model is **blind to the dataset**, so one global rule must cover every multi-image
case. Verified on the `neo_k3` policy adapter over B's three (and only) multi-image classification
datasets:

| dataset | first-image-all | **group-all (cap 6)** | count-gated (neo=3 else first) |
|---|---|---|---|
| neojaundice | 0.597 | **0.770** | 0.770 |
| CMMD | 0.423 | 0.425 | 0.423 |
| bcn20000 | 0.386 | 0.399 | 0.386 |

- **Rule = group-all-up-to-cap with per-image `Image-i:` markers.** Group every multi-image case (feed up
  to `MAX_IMAGES` = 6 images), each prefixed `Image-1:`/`Image-2:`/… exactly as in the `--markers`
  training export. This recovers neojaundice's full **+0.173** grouped gain and leaves CMMD/bcn unchanged
  (+0.002 / +0.012, both inside the ~0.05–0.07 noise floor). **No count-based feed logic and no "mix"
  training are needed** — markers are what make grouping the flatten-trained datasets harmless.
- **⚠️ Required fix:** `docker_internvl/inference.py:405` currently builds the block *without* markers
  (`"\n".join([IMAGE_TOKEN]*len)`) — a train/inference format mismatch (production adapters train *with*
  markers). **Add the `Image-i:` markers**; the existing group-all cap-6 feeding is already correct.
- *Optional, not required:* the count-gated rule (group only X=3 cases, else first image) is provably
  zero-risk on CMMD/bcn but gains nothing over group-all here; keep it in reserve if a future dataset
  shows grouping degrades it.
- **endo:** subsample any >6-image case to **M=4 uniform frames** (blind, count-keyed; required anyway so
  a raw 105-frame clip doesn't overflow context / 1.5× runtime — see §11e / endo decision).
- **IU_XRay report-gen:** feeding TBD (predict group K=2 frontal+lateral) — needs the CRIMSON/GREEN
  harness; §11d placeholder.

**Per-modality tile caps / worst-case defense.** `max_num` tiles is the lever under the 48 GB / 1.5×
budget; the GPU probe showed VRAM is a non-issue (16 imgs @ full tiling ≈ 15 GB) — the real limits are
latency + context length. Cap images per case (`MAX_IMAGES`) and frames (M=4) so no test case OOMs or
times out; high-image cases need a reduced tile cap if they approach the 8192 cutoff (the 3-image
neojaundice rows sit at ~832 tokens, so the cap rarely binds for classification).

---

## 11. Code & file map (v2)

| File | Change |
|---|---|
| `internal_eval/build_splits.py` | **v2 geometry** (true_test/A/B + dev + A'/B'); **stores all images, caps nothing** (lossless); drops `test_internal`/`experiment` tiers and the `--max_images` truncation |
| `internal_eval/audit_data.py` | unchanged (regression-outlier 10× guard already in) |
| `llama_factory_internvl/scripts/export_sharegpt.py` | **built**: `--max_images K` (`ceil(X/K)` chunking, K=1=flatten), `--markers` (per-image `Image-N:`), `--mix` (MANTIS single+multi), always-drop `instance_detection`, `--exclude_dataset` |
| `llama_factory_internvl/examples/train_qlora/flare_cls_phase1.yaml` | **new**: Phase-1 QLoRA base config (cutoff_len 8192, seed 1337, report_to none); per-arm overrides via CLI |
| `llama_factory_internvl/scripts/eval_cls_arm.py` | **new**: per-dataset balanced-accuracy scorer on B′; arm-consistent feeding (k1 first-image, k2/mix grouped) |
| `llama_factory_internvl/scripts/train_cls_arm.sh` | **new**: sbatch wrapper (ugrip/prod/1-GPU), export→train→score per arm; `sbatch train_cls_arm.sh <k1\|k2\|mix> <max_steps>` |
| `docker_internvl/inference.py` | runtime image/token budget cap (per §7); no chunking *(Phase-1 eval uses `eval_cls_arm.py` instead)* |
| evaluation (Veronika's branch) | report **per (dataset × task)**, not pooled per task type |

**Cluster setup gotchas (first training run):** training env needs a one-time `pip install -e . --no-deps`
in `llama_factory_internvl/` to get `llamafactory-cli`; headless batch needs `report_to: none`; SLURM
`--output` + input JSONL must be on the shared home FS, not compute-node `/tmp`.

---

## 12. Open questions / to confirm

- **Phase-1 rerun** — match epochs + seeds + the A′↔B′ flip (§9.2), then read per-dataset K.
- **Per-dataset K table** — derive K from an image-redundancy score (upgrade aHash → a real embedding;
  aHash misses CMMD microcalcifications); validate, don't blind-sweep 19 datasets (§6.4).
- **Ranking unit** — per task type or per dataset? (val-leaderboard submission + organizer question.)
- **Context length** — `cutoff_len` 8192 works for classification; confirm the real ceiling for report-gen
  / 3D where the tile-budget packer becomes necessary.
- **Base VLM** — keep InternVL3-8B or adopt a newer efficient VLM (Yaqub).
- **Priority** — classification-first (Mai) vs the detection/counting work in
  `detection-counting-paths.md`.
