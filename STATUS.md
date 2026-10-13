# Status

**Updated 2026-10-13.**

## Where we are
- The experiment plan (`PLAN.md`) was **confirmed by Tausifa and the team on 2026-10-11 with
  no changes**. The working budget is the trimmed plan in PLAN.md §8.3 (≈1,450 GPU-h on
  ws, plus ~109 h on Blackwell for 2048²).
- The team assignment is PLAN.md §7.
- **Build first, interface note last** (2026-10-13, PLAN.md changelog): Taha builds the whole
  foundation, launches S1 seed 1337 once W2 passes, builds W3 while it trains, then hands the
  team a half-page plug-in note (`code/INTERFACE.md`) plus a reference plug-in.
- Foundation work started 2026-10-13. Step 0 and W1 done; W2 in progress.

## Foundation progress (Taha)
| Step | State |
|---|---|
| 0 Setup: ws clone, env, 4-bit load, LF dry run, LF version recorded | **done** 2026-10-13 (`code/env/LF_VERSION.md`; P8 = 21,639,168) |
| W1 Data: decks, dev-select, holdout file, subsets, md5 manifest | **done** 2026-10-13 (decks on ws `/share/sda/ugrip/journal/data`; `data/DECK_MANIFEST.json`, `data/subsets/`) |
| W2 Training: `journal_train`, per-example loss, plug-in plumbing, registry; 50-step gate | — |
| S1 seed 1337 launch (Shared + 6 TaskLoRA) | — (ask Taha first) |
| W3 Eval: ZS-bare/ZS-prompt, adapters, oracle sets, composition, plug-ins, scoring, bootstrap | — |
| S0: full zero-shot on holdout + dev; real U | — |
| Handoff: `code/INTERFACE.md`, reference plug-in, gate + launch/eval commands, queue rules | — |

## Next steps
1. **Taha:** finish step 0, then W1 → W2 → launch S1 s1337 → W3 → S0 → handoff note.
2. **Teammates (until the handoff note lands):** read your method papers and design your
   modules. Veronika: HydraLoRA oracle mode first. Alex: W6, W7. Alberto: W10, W9, and D12
   (regression vs counting as the low-resource scaling target) with Tausifa.

## Decisions to relay
- **Tell Tausifa:** A/B include dev; we train on A∪B (40,094) and make selection decisions on
  the holdout (PLAN.md changelog 2026-10-13). Option: split holdout into select/report halves.

## Open checks before the first runs
- Verify the per-example-loss implementation with a unit test (W2).
- Measure U (the cost of one full run) with packed one-row-per-question decks (the S1 Shared
  s1337 run gives this).

## Open requests
- Ask Dr. Yaqub, through Tausifa, for Blackwell GPU 0 for the whole window (PLAN.md §8.5).
  It is the largest available lever on the budget.

## Active runs
None. (ws GPUs 0, 2, 3 were busy with other users' jobs on 2026-10-13; GPU 1 free.)
