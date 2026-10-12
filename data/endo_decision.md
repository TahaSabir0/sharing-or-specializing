# endo (Endoscopy) — decision doc

*2026-06-12. Brainstorm/decision for the one open `endo` row in `STATS.md` §11d. No training run; this
is a recommendation + a minimal inference-only experiment. Companion: `STATS.md` §4 (census) +
PROGRESS 2026-06-12 endo deep-dive + [[project-instance-detection-dead-task]].*

---

## TL;DR

**The ranking is per-CASE, not per-dataset (EVAL.pdf, confirmed) — so endo is ~0.06% of the score, not a
full modality slot.** Combined with only ~15 trainable samples (can't learn a modality from 15), endo is
not worth a training investment. **Keep it excluded from training; handle it at inference with a global,
modality-blind frame cap (uniform-subsample any case with >B images to M frames); treat its internal metric
as a sanity check only.** The single cheap experiment is an **inference-only M∈{1,4,8} sweep on the existing
fine-tuned model** over 2 true_test + 16 (normalized) val-public endo cases — no retraining.

---

## 1. The unknown that flipped the decision — RESOLVED

The brief flagged a material unknown: *does FLARE rank per-DATASET (endo = a full scored slot, dropping it
drags the average) or per-CASE (endo = 33/51k, negligible)?*

**EVAL.pdf → Ranking Scheme (ChallengeR) resolves it to per-CASE:**

> Step 1. For each testing case, compute all M metrics.
> Step 2. Rank participants for each of the N testing cases and each metric → M·N rankings.
> Step 3. Average all these rankings.

There is **no per-dataset normalization** — a dataset's influence on the final rank is exactly its number
of test cases. Endo is **33 cases out of ~51k 2D cases (~0.06%)**. Dropping/mis-scoring it moves the
averaged rank by essentially nothing. The "drop endo ⇒ lose a whole scored modality" fear is **refuted**.

**Residual risk (flag, don't block on it):** the *hidden* test set's modality mix is unpublished. If FLARE
deliberately rebalanced it to over-weight rare modalities, endo's weight could rise. But the *stated* scheme
is per-case and nothing in the docs implies rebalancing, so we assume proportional. Cheap to revisit if the
April→August validation leaderboard ever shows endo-sensitive movement.

## 2. The constraints that bound any choice

- **~15 trainable samples** (A=8 / B=7; rest in true_test + val-public). You cannot *learn* an imaging
  modality from 15 examples. Any value endo training could add is **output-format conditioning + inference
  feeding**, not a learned visual skill — and format is already solved by the main fine-tune
  ([[project-format-not-bottleneck]]).
- **endo = video** (33 clips, 4–105 frames, contiguous `Endo_<case>-<frame>.png`, one label/clip). So
  chunk-K is the wrong lever; the lever is a **frame budget M** (M=1 = last year's first-frame-only).
- **At inference you can't "skip" a case.** Blind inference means every test case — including endo clips —
  arrives and must be answered. "Drop endo" can only ever mean *drop from training / internal eval*, never
  *skip at test time*. So an inference policy for high-frame-count cases is **mandatory regardless**.
- **A raw endo case will blow the budget.** STATS §7: endo median ≈ **143k tokens**, max ≈ 349k (vs
  cutoff 8192) — a 105-frame clip overflows context *and* the 1.5× runtime rule. So an inference frame cap
  is **required for robustness**, independent of any accuracy argument.
- **Blind-inference-legal:** the cap must key on something observable. Image **count** is observable, and
  only endo + the bcn20000 derm tail ever exceed ~6 images — so a single global "cap at >B images" rule
  catches endo without ever needing to *detect* "this is endoscopy."

## 3. The options, weighed

| option | what it is | verdict |
|---|---|---|
| **(a) Drop entirely** | exclude from train + eval, and... do nothing at inference | **Impossible at inference** — endo cases still arrive and, uncapped, blow context/runtime. "Drop" only validly applies to train+eval. |
| **(b) Inference frame-cap M** | exclude from train; at inference uniformly subsample any >B-image case to M frames | **✅ Recommended.** Mandatory for robustness anyway; M is a cheap knob, not a learned skill. |
| **(c) Keyframe dedup** | drop near-duplicate adjacent frames before sampling | Over-engineered for 0.06% weight. Uniform M already covers the clip; dedup adds CPU + code for no measurable rank gain. Park it. |
| **Include in training** | K-chunk the clips into rows and fine-tune | **No.** 15 samples can't teach the modality; a 105-frame clip flattened at K=1 emits hundreds of near-dup single-frame rows that pollute the cls mix. Status quo (excluded) is correct. |

## 4. Recommendation

1. **Training: keep endo EXCLUDED** (status quo — `train_full_sft.sh` already excludes it). 15 samples
   can't teach endoscopy; the 105-frame clips would pollute training.
2. **Inference: add a global, modality-blind frame cap.** *If a case has > B images, uniformly subsample to
   M frames* (evenly spaced over the clip so a localized lesion in the middle isn't systematically missed).
   This is the image-budget cap the pipeline needs anyway; endo is just its main trigger. Start **M = 4**
   — chosen as the cheapest *robustness* setting (covers the clip, stays well inside context/runtime), not
   as an accuracy play, because endo's weight doesn't justify optimizing it.
3. **Internal eval: endo = sanity check, not a tracked metric.** With 2 true_test + 16 val-public it's far
   too small to be a reliable signal — but normalize the 16 (below) and eyeball that the model emits *clean
   classification letters*, not garbage. If it does, endo is "handled" and we move on.
4. **STATS §11d endo row → fill as:** `inference frame-cap M=4 (global >B-image rule); excluded from
   training; sanity-eval only.` (After Taha approves + the M-sweep confirms M.)

## 5. The minimal decisive experiment (inference-only, no retraining)

Cheap because it reuses the **existing fine-tuned model + cls adapter** — no GPU training, ~tens of
inferences total.

1. **Normalize the val-public endo GT** (step 6) → 16 usable cases; add the 2 true_test endo cases → **n=18**.
2. **Run the existing model on those 18 cases at M ∈ {1, 4, 8}** (uniform frame subsample). 18 × 3 = 54
   inferences — a few minutes on one GPU.
3. **Score balanced accuracy** per M, and **inspect outputs** for format cleanliness + whether more frames
   ever flip a wrong→right (this subsumes the "is the lesion only in some frames?" redundancy question —
   if M=8 never beats M=4, frames are redundant enough that M=4 is safe).
4. **Decide M** = smallest M with no accuracy loss and no context/runtime blowup. Default to **M=4** unless
   M=8 shows a real, repeated gain (unlikely to matter at n=18 / 0.06% weight).

*Note on power:* n=18 cannot give a tight estimate — and it doesn't need to. The experiment's real job is
(a) confirm the model doesn't emit garbage on multi-frame endo input, and (b) rule out M=1 being
catastrophically worse. Both are answerable at n=18. Don't over-invest past that.

## 6. The val-public normalization step (required before any endo metric is trustworthy)

val-public endo GT is **uncleaned — 8/16 corrupt**. Before scoring:
- **`'T'` → `'D'`** (Tumor mislabeled as the letter `T`; the option letter for Tumor is `D`).
- **Multi-letter (e.g. `'A B'`) → parse into a set** and score as multi-label/any-correct against the
  posed "select all" framing (Ulcer/Erosion/Polyp/Tumor/None = A/B/C/D/E).
- Apply the **same map** the training cleaner used (PROGRESS 2026-06-12: 36 `'T'` + 11 multi-letter pulled
  from train) so train and eval normalization are consistent.

Until this is applied, *any* endo number off val-public is noise — half the labels are wrong.

---

## 7. Result — experiment ran, decision CONFIRMED (2026-06-12)

The §5 M-sweep ran inference-only (job 782, n=18, 54 inferences, ~12 min; full table in `STATS.md` §11e):

- **Accuracy is invariant to M** — BA 0.183 strict / ~0.22 lenient at every M∈{1,4,8}; plain-acc ~0.39.
- **Zero wrong→right flips** at any M transition; one lenient-only *regression* at M≥4 ⇒ extra frames never
  help and once mildly hurt. **Frames are redundant → low M is free.**
- **Format 100% clean** (all 54 outputs valid option letters, 0 garbage) → the low score is a *value*
  problem (wrong class), not format — consistent with [[project-format-not-bottleneck]].
- Router sent all 18 endo questions to `adapter_classification` (score ≥0.3, no general fallback).
- Normalization worked: **7 `'T'`→`'D'` + 1 `'A B'`→{A,B}** = the ~8/16 corrupt val-public GTs. The model
  *also* emits literal `'T'`, so predictions need the same T→D map before scoring (now baked into the scorer).

**Net:** every assumption held. **M=4 locked** as the endo frame-cap (M=1 is accuracy-equal and cheaper if
we later optimize runtime). endo stays train-excluded, sanity-eval only. `STATS.md` §11d row updated to ✅.

## One-line recommendation

> **Keep endo out of training; handle it at inference with a global, modality-blind "subsample any >B-image
> case to M=4 uniform frames" cap (mandatory for context/runtime robustness anyway); treat endo's internal
> metric as a sanity check only — because the ranking is per-case, endo is ~0.06% of the score, not a
> modality slot. Confirm M with a 54-inference, no-retrain M∈{1,4,8} sweep on the 18 (normalized) endo cases.**
