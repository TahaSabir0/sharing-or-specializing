# Challenge-era results (read-only reference)

These are ledgers copied on 2026-10-11 from the FLARE 2026 challenge repo
(`C:/Users/tahas/UGRIP/stats/`). They hold numbers we measured during the challenge.

**Rules:**
- **Never mix these numbers into journal results tables.** They used a different recipe:
  - ranks 8/32/64, dropout 0;
  - token-weighted loss;
  - challenge prompts (`medical_cot`, counting CoT);
  - one-row-per-image training decks;
  - mostly 2048²;
  - some adapters trained on holdouts.
- Use them for three things only:
  1. expectations ("prior evidence" lines in PLAN.md §4);
  2. compute anchors (PLAN.md §2);
  3. cited background, explicitly labelled as challenge-era.
- Don't edit them here. If something needs correcting, correct it in the UGRIP repo.

## Index

| Ledger | What it holds | Used by |
|---|---|---|
| `26.Qwen35Sprint_Stats.md` | Qwen3.5-9B: speed/memory anchors, general vs specialist (§1k), zero-shot protocols (§6, §8), resolution twins (§1g–§1j, §9) | §2, S0, S1, S9.1 |
| `12.TaskTransfer_Stats.md` | Task-transfer PGF matrix on Qwen3-VL-8B; format lock-in | S2 |
| `25.AdapterComposition_Stats.md` | Adapter composition and learned routers (Alex), with the review | S4b, S4c |
| `15.ModalityAdapters_Stats.md` | Modality adapters, task+modality merge, ResNet-18 modality classifier | S5, S4c |
| `11.Backbone_Bakeoff_Stats.md` | MedGemma vs Qwen3 bake-off, MedSigLIP probe, MedGemma speed | S10, §2 |
| `21.TrainingRecipe_Stats.md` | Recipe facts (frozen vision tower and projector by default), rank screens | §1.2, S9.2 |
| `17.Preprocessing_Stats.md` | Border crop, histogram equalisation | S9.4 |
| `23.Augmentation_Stats.md` | Modality-aware augmentation | S9.5 |
| `3.Packing_Stats.md` | Multi-image packing and flattening | S9.6 |
| `4.Veronica_Alex_Stats.md` | CoT prompts and the task token | S9.7 |
| `18.Regression_Stats.md` | Regression data mixtures, degenerate-zero answers, greedy-decoding fix | S3, S9.8 |

The dataset census and holdout composition (`stats/1`, `stats/22`) are in `../../data/`.

## Findings carried over from the challenge

Condensed from the challenge repo's `docs/RESEARCH_CONTEXT.md`. Each is scoped to the
challenge setup.

1. **Format and perception are different failures.** Fine-tuning fixes output format
   quickly; the remaining errors on counting, detection and the hard classification sets
   (CMMD, bcn20000, dental) were wrong values in clean outputs. Inspect raw generations
   before blaming the parser or the model.
2. **Routing helped some tasks, not all.** On Qwen3.5-9B at 768², specialists beat the
   general adapter on regression (significantly), classification and detection; multi-label
   and counting tied; the general adapter was slightly better on reports (`26 §1k`).
   Negative transfer between task adapters was mostly output-format lock-in (`12`).
3. **Resolution and data effects are task-dependent.** Resolution helped counting (−1.7
   MAE) and slightly helped classification and multi-label; detection and regression were
   null. Regression donor mixtures refuted simple "same modality" and "more data" stories
   (`18`).
4. **Naive adapter merging was destructive.** Summing task or modality adapter deltas at
   λ=1 hurt detection badly (`25`, `15 §5`). The journal's λ grid tests whether the weight or
   the mechanism was to blame.
