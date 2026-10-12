# Status

**Updated 2026-10-11.**

## Where we are
- The experiment plan (`PLAN.md`) was **confirmed by Tausifa and the team on 2026-10-11 with
  no changes**. The working budget is the trimmed plan in PLAN.md §8.3 (≈1,450 GPU-h on
  ws, plus ~109 h on Blackwell for 2048²).
- The team assignment is PLAN.md §7.
- The repo holds docs and code together. `code/` is seeded with the reusable challenge code
  (scorer, eval harness, deck builder, LF patch, recipe template; see `code/README.md`). No
  journal-specific code has been written and no runs have started.

## Next steps (week 1)
1. **Taha:** write and circulate the plug-in interface doc (`code/INTERFACE.md`, PLAN.md
   §3.2) on day 1.
2. **Taha:** build the foundation (W1 data, W2 training harness, W3 eval harness) in `code/`,
   starting from what is already there (`code/README.md`). Run S0 (smoke/calibration,
   ZS-bare and ZS-prompt).
3. **Veronika:** HydraLoRA module, **oracle mode first** (S3, S5 and S10 wait on it).
4. **Alex:** composition and MoLE-gate module (W6) and location target sets (W7).
5. **Alberto:** practical-factor switches (W10) and MedGemma port (W9). Raise D12
   (regression vs counting as the low-resource scaling target) with Tausifa.
6. When W1–W3 pass the smoke gate, launch S1 seed 1337 (Shared + TaskLoRA).

## Open checks before the first runs
- Confirm that `A.jsonl`/`B.jsonl` exclude `A_dev`/`B_dev`, and get exact counts (W1).
- Read the exact Qwen3.5-9B LoRA target-module list and measure P8 (W2).
- Verify the per-example-loss implementation with a unit test (W2).
- Measure U (the cost of one full run) with packed one-row-per-question decks in S0.

## Open requests
- Ask Dr. Yaqub, through Tausifa, for Blackwell GPU 0 for the whole window (PLAN.md §8.5).
  It is the largest available lever on the budget.

## Active runs
None.
