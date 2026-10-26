# analysis/: statistics and figures (mostly CPU)

Cap threads when you run these on the box (`OMP_NUM_THREADS=4`). Inputs are the `metrics.json`
/ prediction files of finished runs (`/data/ugrip2026/journal/runs/<run_id>/`).

| Script | What | Owner | Spec (PLAN.md v2) | State |
|---|---|---|---|---|
| `bootstrap.py` | per-task CIs, paired arm differences, two-level seed bootstrap, Holm | Taha | §1.11 | done |
| `group_results.py` | validates and joins E4/E5 oracle slices; normalizes saved baseline predictions and CRIMSON | Alex | E4/E5 | implemented; CPU integration tested |
| `compare_groups.py`, `analyze_e4_e5.py` | task/cell comparisons, three-seed CIs, Holm, E5 size/difficulty correlations | Alex | E4/E5 | implemented; see `../E4_E5_RUNBOOK.md` |
| `transfer_matrix.py` | C_ij, PGF μ, transferability/malleability, covariate correlations with permutation tests, heatmap | Alex | E3 | to build |
| `representations.py` | hidden-state extraction, debiased CKA, RSA, permutation tests; Hydra head similarity | Alex (Hydra heads with Veronika) | E7 | to build |
| `rq4_synthesis.py` | per-cell specialization benefit vs task characteristics | Alex | E8 | to build |
| `scaling_model.py` | mixed-effects interaction model, crossover p*, curves | Alberto | E9 | to build (Phase C) |
