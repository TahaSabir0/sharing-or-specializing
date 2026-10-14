# Status

**Updated 2026-10-14.**

## Where we are
- The experiment plan (`PLAN.md`) was **confirmed by Tausifa and the team on 2026-10-11 with
  no changes**. Its budget (PLAN.md §8.3, ≈1,450 GPU-h on ws) **no longer fits**: since
  2026-10-14 we have **one card, Blackwell `mbz-yaqub` GPU 0**, for training and eval; `ws` is
  not ours. A cut is needed (next steps).
- The team assignment is PLAN.md §7.
- **Build first, interface note last** (2026-10-13, PLAN.md changelog): Taha builds the whole
  foundation, launches S1 seed 1337 once W2 passes, builds W3 while it trains, then hands the
  team a half-page plug-in note (`code/INTERFACE.md`) plus a reference plug-in.
- Foundation work started 2026-10-13. Step 0, W1, W2 done on ws. **Move to Blackwell done (no-GPU part) 2026-10-14**; the box smoke gate waits for GPU 0.

## Foundation progress (Taha)
| Step | State |
|---|---|
| 0 Setup: ws clone, env, 4-bit load, LF dry run, LF version recorded | **done** 2026-10-13 (`code/env/LF_VERSION.md`; P8 = 21,639,168) |
| W1 Data: decks, dev-select, holdout file, subsets, md5 manifest | **done** 2026-10-13 (decks on ws `/share/sda/ugrip/journal/data`; `data/DECK_MANIFEST.json`, `data/subsets/`) |
| W2 Training: `journal_train`, per-example loss, plug-in plumbing, registry; 50-step gate | **done** 2026-10-13: unit test PASS, template gate 14/14, smoke loss 1.57→0.63, trainable = P8, 1.665 s/example, 15.5 GB peak |
| Move to Blackwell (no-GPU part): env/LF/splits/decks verified identical to ws, code defaults → box, repo clone, docs | **done** 2026-10-14 (branch `taha/blackwell-move`; PLAN.md changelog) |
| Smoke gate on Blackwell GPU 0 (50 steps; measures s/example on the box) | — waits for GPU 0 |
| S1 seed 1337 launch (Shared + 6 TaskLoRA) | — (ask Taha first) |
| W3 Eval: ZS-bare/ZS-prompt, adapters, oracle sets, composition, plug-ins, scoring, bootstrap | code written (branch `taha/w3-eval`), untested |
| S0: full zero-shot on holdout + dev; real U | — |
| Handoff: `code/INTERFACE.md`, reference plug-in, gate + launch/eval commands, queue rules | — |

## Next steps
1. **GPU 0:** ask Ahmed to have `mostafasalem`'s job (PID 16366, ~12.4 GB, running since
   ~10 Oct) moved off GPU 0. We fit alongside it (~17 GB per run), just slower.
2. **Smoke gate on the box** (needs GPU 0, ~15 min): 50-step `journal_train` run on the mix
   deck + the ~50-question `eval_arm.py` check. Compare loss with the ws smoke run (1.57→0.63)
   and measure s/example and peak memory.
3. **Re-plan with Tausifa** using the measured speed: one card ≤ ~400 h until 31 Oct, vs
   ~1,450 A5000 GPU-h planned. Options: one seed for most arms, three only for the headline
   comparison; subsets for some studies; merge or drop studies.
4. S1 seed 1337 (Shared, then the 6 TaskLoRAs, back to back on GPU 0) after Taha's OK.
5. W3 test and S0 share GPU 0 with training: one queue for everyone (`run_chain.sh`).
6. Handoff: `code/INTERFACE.md` + reference plug-in, rewritten for the box.
7. **Teammates:** read papers and design modules. All GPU jobs go through the GPU 0 queue.

## Compute (2026-10-14)
- Only box: `mbz-yaqub`, **GPU 0** (our assignment). Setup and paths: banner of
  `ops/cluster-config.md`. Launch with `code/train/run_chain.sh` (defaults to GPU 0).
- On 2026-10-14, GPU 0 also had an outside job (`mostafasalem`, ~12.4 GB, 25–0% util).
- ws estimate for S1 s1337 was 1.665 s/example (Shared ≈ 58 h, 6 TaskLoRAs ≈ 56 h); the
  challenge-era Blackwell speedup was ~4.1× at 2048². The box's 768² speed is measured in the smoke gate.
- Fixed 2026-10-14: `run_chain.sh` logged `rc=0` for every run, failures included (the `$?`
  came after a `$(date)` substitution). It now records the real exit code.

## Decisions to relay (Tausifa)
- A/B include dev; we train on A∪B (40,094) and make selection decisions on the holdout
  (PLAN.md changelog 2026-10-13). Option: split holdout into select/report halves.
- Compute is one Blackwell card: the plan needs a cut (step 3 above).

## Open checks
- Measure U with the packed decks: the S1 Shared s1337 run gives it (smoke implies ≈ 56 GPU-h
  vs the planned 68).
- Full-deck tokenization takes ~35–40 min (cached per deck after the first run).

## Active runs
None.
