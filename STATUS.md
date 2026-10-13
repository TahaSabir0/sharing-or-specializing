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
- Foundation work started 2026-10-13. Step 0, W1, W2 done. S1 seed 1337 launch awaiting Taha's OK; W3 code written, testing next.

## Foundation progress (Taha)
| Step | State |
|---|---|
| 0 Setup: ws clone, env, 4-bit load, LF dry run, LF version recorded | **done** 2026-10-13 (`code/env/LF_VERSION.md`; P8 = 21,639,168) |
| W1 Data: decks, dev-select, holdout file, subsets, md5 manifest | **done** 2026-10-13 (decks on ws `/share/sda/ugrip/journal/data`; `data/DECK_MANIFEST.json`, `data/subsets/`) |
| W2 Training: `journal_train`, per-example loss, plug-in plumbing, registry; 50-step gate | **done** 2026-10-13: unit test PASS, template gate 14/14, smoke loss 1.57→0.63, trainable = P8, 1.665 s/example, 15.5 GB peak |
| S1 seed 1337 launch (Shared + 6 TaskLoRA) | — (ask Taha first) |
| W3 Eval: ZS-bare/ZS-prompt, adapters, oracle sets, composition, plug-ins, scoring, bootstrap | code written (branch `taha/w3-eval`), untested |
| S0: full zero-shot on holdout + dev; real U | — |
| Handoff: `code/INTERFACE.md`, reference plug-in, gate + launch/eval commands, queue rules | — |

## Next steps
1. **Taha decides the S1 seed 1337 launch** (GPU layout below).
2. While S1 trains: test W3 (both zero-shot protocols on ~50 holdout rows per task, scored),
   merge `taha/w3-eval`, then S0 (full ZS-bare + ZS-prompt on holdout and dev, ~5 GPU-h).
3. Handoff: `code/INTERFACE.md` + reference plug-in (`code/methods/lora_ref.py`, written,
   untested) + gate/launch/eval commands; then send to the team.
4. When S1 finishes: holdout eval, ledger, S2.
5. **Teammates (until the handoff note lands):** read your method papers and design your
   modules. Veronika: HydraLoRA oracle mode first. Alex: W6, W7. Alberto: W10, W9, and D12
   (regression vs counting as the low-resource scaling target) with Tausifa.

## Blocker: GPUs (2026-10-13)
- An outside user (`mostafasalem`, not on the team) has jobs on **all 4 ws GPUs** (on GPU 1
  since 11:55, right after our smoke run freed it). GPUs 2/3 have ~9 GB free (too little;
  a training run needs ~17 GB reserved). GPUs 0/1 can each fit one of our runs, shared.
- S1 s1337 estimate (measured 1.665 s/example, uncontended): Shared ≈ 58 h; the 6 TaskLoRAs
  back to back ≈ 56 h (cls 21, report 10, ml 9, det 7, reg 7, count 1.3). Sharing a card adds
  ~30–100%.
- Proposed: Shared on GPU 0, TaskLoRA chain on GPU 1 (`code/train/run_chain.sh`); done ~16–18 Oct.
- S1 seeds 2026/42 (~230 GPU-h) and teammates' runs do not fit by 31 Oct on 1–2 shared cards.

## Decisions to relay (Tausifa)
- A/B include dev; we train on A∪B (40,094) and make selection decisions on the holdout
  (PLAN.md changelog 2026-10-13). Option: split holdout into select/report halves.
- GPU access: ws is fully occupied by an outside user; ask for cards to be freed and/or
  Blackwell GPU 0 for the window (PLAN.md §8.5).

## Open checks
- Measure U with the packed decks: the S1 Shared s1337 run gives it (smoke implies ≈ 56 GPU-h
  vs the planned 68).
- Full-deck tokenization takes ~35–40 min (cached per deck after the first run).

## Active runs
None.
