# Status

**Updated 2026-10-14.**

## Where we are
- The experiment plan (`PLAN.md`) was **confirmed by Tausifa and the team on 2026-10-11 with
  no changes**. Its budget (PLAN.md §8.3, ≈1,450 GPU-h on ws) **no longer fits**: since
  2026-10-14 we have **one card, Blackwell `mbz-yaqub` GPU 1**, for training and eval; `ws` is
  not ours. A cut is needed (next steps).
- The team assignment is PLAN.md §7.
- **Build first, interface note last** (2026-10-13, PLAN.md changelog): Taha builds the whole
  foundation, launches S1 seed 1337 once W2 passes, builds W3 while it trains, then hands the
  team a half-page plug-in note (`code/INTERFACE.md`) plus a reference plug-in.
- Foundation work started 2026-10-13. Step 0, W1, W2 done on ws. **Move to Blackwell done (no-GPU part) 2026-10-14**; smoke gate passed on GPU 1.

## Foundation progress (Taha)
| Step | State |
|---|---|
| 0 Setup: ws clone, env, 4-bit load, LF dry run, LF version recorded | **done** 2026-10-13 (`code/env/LF_VERSION.md`; P8 = 21,639,168) |
| W1 Data: decks, dev-select, holdout file, subsets, md5 manifest | **done** 2026-10-13 (decks on ws `/share/sda/ugrip/journal/data`; `data/DECK_MANIFEST.json`, `data/subsets/`) |
| W2 Training: `journal_train`, per-example loss, plug-in plumbing, registry; 50-step gate | **done** 2026-10-13: unit test PASS, template gate 14/14, smoke loss 1.57→0.63, trainable = P8, 1.665 s/example, 15.5 GB peak |
| Move to Blackwell (no-GPU part): env/LF/splits/decks verified identical to ws, code defaults → box, repo clone, docs | **done** 2026-10-14 (branch `taha/blackwell-move`; PLAN.md changelog) |
| Smoke gate on Blackwell GPU 1 (50 steps; measures s/example on the box) | **passed** 2026-10-14: loss 1.563→0.642 (ws 1.573→0.634), **1.196 s/example** (ws 1.665, ≈1.4×), peak 15.5 GB, trainable = P8; holdout50 eval 0 truncated / 0 malformed |
| S1 seed 1337 launch (Shared + 6 TaskLoRA) | — (ask Taha first) |
| W3 Eval: ZS-bare/ZS-prompt, adapters, oracle sets, composition, plug-ins, scoring, bootstrap | code written (branch `taha/w3-eval`), untested |
| S0: full zero-shot on holdout + dev; real U | — |
| Handoff: `code/INTERFACE.md`, reference plug-in, gate + launch/eval commands, queue rules | — |

## Next steps
1. **GPU:** we use **GPU 1** (free on 2026-10-14). Check `nvidia-smi -i 1` before each launch.
2. ~~Smoke gate on the box~~ passed 2026-10-14; 3-at-once test done (2.0× throughput). Use 3 slots on GPU 1.
3. **Scope review (coming, with Tausifa)**: the fundamental experiments stay. Budget input:
   one card until **13 Nov** (≤ ~720 h ≈ 1,000 A5000-equivalent h at 1.4×, minus final evals
   before the ~15 Nov submission), vs ~1,450 A5000 GPU-h planned. At 1.196 s/example, S1 s1337
   (Shared + 6 TaskLoRAs) ≈ 80 h on its own. Options: one seed for most arms, three only for the headline
   comparison; subsets for some studies; merge or drop studies.
4. S1 seed 1337 (Shared, then the 6 TaskLoRAs, back to back on GPU 1) after Taha's OK.
5. W3 test and S0 share GPU 1 with training: one queue for everyone (`run_chain.sh`).
6. Handoff: `code/INTERFACE.md` + reference plug-in, rewritten for the box.
7. **Teammates:** read papers and design modules. All GPU jobs go through the GPU 1 queue.

## Compute (2026-10-14)
- GPU access until **13 Nov 2026**.
- Only box: `mbz-yaqub`, **GPU 1** (first assigned GPU 0; switched to the free GPU 1 the same day). Setup and paths: banner of
  `ops/cluster-config.md`. Launch with `code/train/run_chain.sh` (defaults to GPU 1).
- On 2026-10-14, GPU 1 was empty; `mostafasalem` has jobs on GPUs 0, 2 and 5.
- ws estimate for S1 s1337 was 1.665 s/example (Shared ≈ 58 h, 6 TaskLoRAs ≈ 56 h); the
  challenge-era Blackwell speedup was ~4.1× at 2048². The box's 768² speed is measured in the smoke gate.
- Fixed 2026-10-14: `run_chain.sh` logged `rc=0` for every run, failures included (the `$?`
  came after a `$(date)` substitution). It now records the real exit code.

## Finding: ws and Blackwell are not bit-reproducible (2026-10-14)
- Same code, data, decks, packages and seed, but the 50-step smoke adapters give different greedy
  outputs on holdout50: identical raw outputs per task are cls 39/50, ML 36/50, report 41/50,
  count 23/50, reg 15/50, det 0/50. GPU kernels differ (sm_120 vs sm_86), so the numerics differ
  slightly and greedy decoding diverges. Counting MAE 38.2 (ws) vs 76.2 (box), almost all from
  three degenerate answers ("1000" for 41 and 13, "105" for 7) of a barely trained adapter.
- Consequence: **every arm we compare must be trained and evaluated on the same box.**
  All journal runs from now on are on Blackwell GPU 1. The ws smoke numbers are not comparable
  to box numbers. (Holdout50 here is a format check, not a result.)
- Throughput: one run alone ≈ 1.4 A5000s (1.196 s/example, GPU ~53% busy). **3 runs at once
  (tested 2026-10-14): 1.75–1.76 s/example each, ≈ 1.70 examples/s in total = 2.0× one run ≈
  2.8 A5000s.** Peak 16.95 GB reserved each (~51 GB of 96). Budget to 13 Nov at 3 slots ≈
  2,000 A5000-equivalent GPU-h. 4 at once is untested.
- The 3 runs had the same config and seed, but their final losses differed slightly (0.6376 /
  0.6424 / 0.6401; alone: 0.6417). So training isn't bit-deterministic even on one box: run-to-run
  noise at the ~1% loss level, on top of seed variance.

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
