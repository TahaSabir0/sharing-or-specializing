# analysis/: statistics and figures (mostly CPU)

Cap threads when you run these on the box (`OMP_NUM_THREADS=4`). Inputs are the `metrics.json`
/ prediction files of finished runs (`/data/ugrip2026/journal/runs/<run_id>/`).

| Script | What | Owner | Spec (PLAN.md v2) | State |
|---|---|---|---|---|
| `bootstrap.py` | per-task CIs, paired arm differences, two-level seed bootstrap, Holm | Taha | §1.11 | done |
| `transfer_matrix.py` | C_ij, PGF μ, transferability/malleability, covariate correlations with permutation tests, heatmap | Alex | E3 | to build |
| `representations.py` | hidden-state extraction, debiased CKA, RSA, permutation tests; Hydra head similarity | Alex (Hydra heads with Veronika) | E7 | to build |
| `hydra_heads.py` | per-target cosine similarity of B_t (Hydra-T) or A_t (Reversed Hydra), from saved adapters on CPU; zero-norm heads marked undefined | Veronika | E7 | implemented, CPU-tested; experimental outputs pending |
| `rq4_synthesis.py` | per-cell specialization benefit vs task characteristics | Alex | E8 | to build |
| `scaling_model.py` | mixed-effects interaction model, crossover p*, curves | Alberto | E9 | to build (Phase C) |
