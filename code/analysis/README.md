# analysis/: statistics and figures (mostly CPU)

Cap threads when you run these on the box (`OMP_NUM_THREADS=4`). Inputs are the `metrics.json`
/ prediction files of finished runs (`/data/ugrip2026/journal/runs/<run_id>/`).

| Script | What | Owner | Spec (PLAN.md v2) | State |
|---|---|---|---|---|
| `bootstrap.py` | per-task CIs, paired arm differences, two-level seed bootstrap, Holm | Taha | §1.11 | done |
| `group_results.py` | validates and joins E4/E5 oracle slices; normalizes saved baseline predictions and CRIMSON | Alex | E4/E5 | implemented; CPU integration tested |
| `compare_groups.py`, `analyze_e4_e5.py` | task/cell comparisons, three-seed CIs, Holm, E5 size/difficulty correlations | Alex | E4/E5 | implemented; see `../E4_E5_RUNBOOK.md` |
| `transfer_matrix.py` | C_ij, PGF μ, transferability/malleability, covariate correlations with permutation tests, heatmap | Alex | E3 | implemented; CPU synthetic integration tested |
| `routing_results.py` | predicted routing accuracy and metric drop versus oracle | Alex | E6 | implemented; real router/model gates pending |
| `representations.py`, `plan_e7.py` | hidden-state extraction, debiased CKA, RSA, permutation tests; Hydra head similarity | Alex (Hydra heads with Veronika) | E7 | implemented; actual probe IDs/head maps and model validation pending |
| `rq4_synthesis.py` | 19-cell specialization benefit vs task characteristics, both anchors/backbones | Alex | E8 | implemented; CPU synthetic integration tested |
| `scaling_model.py` | mixed-effects interaction model, crossover p*, curves | Alberto | E9 | to build (Phase C) |

Full commands and ownership: [Phase B runbook](../PHASE_B_RUNBOOK.md).
