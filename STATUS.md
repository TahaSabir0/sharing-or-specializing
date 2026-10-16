# Status

**Updated 2026-10-16.**

## Where we are
- **New plan, v2 (`PLAN.md`, 2026-10-16)**, from the 14 Oct research update
  (`references/plan_updates/research_update_2026-10-14.pdf`). Five main arms (Shared, Hydra-T,
  Reversed Hydra, TaskLoRA, Shared r48) × 3 seeds × Qwen and MedGemma run first (Phase A); the
  additional experiments follow (Phase B); scaling (RQ2) is last and over budget (Phase C).
  v1 is archived in `PLAN_v1_2026-10-11.md`.
- Team roles under v2: to be agreed separately (PLAN.md §7). v1's assignment no longer applies.
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
| S1 seed 1337 (Shared + 6 TaskLoRA; v2 E1) | 6 TaskLoRAs **done** 2026-10-15 (all rc=0); Shared at 89% on 16 Oct 10:41, ETA ~15:00 box time; see Active runs |
| W3 Eval: ZS-bare/ZS-prompt, adapters, oracle sets, composition, plug-ins, scoring, bootstrap | code written (branch `taha/w3-eval`), untested |
| S0: full zero-shot on holdout + dev; real U | — |
| Handoff: `code/INTERFACE.md`, reference plug-in, gate + launch/eval commands, queue rules | — |

## Next steps (order: PLAN.md §5.1)
1. **Fill the 2 idle slots on GPU 1** (idle since the TaskLoRAs finished, 15 Oct ~22:00): E0
   Qwen zero-shot, holdout evals of the s1337 TaskLoRAs (W3's first test), Shared r48 s1337
   after its smoke gate.
2. Seeds 2026 and 42 of Shared, TaskLoRA and Shared r48.
3. W4 HydraLoRA module (`hydraT` + `hydraR` modes) → gate → Hydra arms × 3 seeds.
4. W9 MedGemma port → gate → E0 MedGemma + E2.
5. Handoff: `code/INTERFACE.md` + reference plug-in.
6. Agree v2 team roles.

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
  (v1 changelog 2026-10-13; PLAN.md v2 §1.7). v2 needs no selection decisions.
- v2 decisions N1–N8 (PLAN.md §6): total-parameter counting with Shared r48 (α 96) as control;
  3 seeds and MedGemma everywhere; additional experiments after the main ones; RQ2 last.

## Open checks
- Measure U with the packed decks: the S1 Shared s1337 run gives it (smoke implies ≈ 56 GPU-h
  vs the planned 68).
- Full-deck tokenization takes ~35–40 min (cached per deck after the first run).

## Active runs
S1 seed 1337, launched 2026-10-14 14:27 (box time, +04), `mbz-yaqub` GPU 1, commit `5199a58`,
repo `/data/ugrip2026/journal/repo`, logs in `/data/ugrip2026/journal/runs/`:
| Chain | PID | Runs, in order | State (16 Oct 10:41) |
|---|---|---|---|
| A | 670039 | `j26_q35_shared_full_768_s1337` | 89% (13,321/15,036 steps), ETA ~15:00 16 Oct |
| B | 670040 | `taskcls` (26.9 h) → `taskdet` (4.5 h) | **done** 15 Oct 21:52, rc=0 |
| C | 670041 | `taskcount` (1.2 h) → `taskreport` (10.0 h) → `taskml` (9.0 h) → `taskreg` (6.0 h) | **done** 15 Oct 16:38, rc=0 |

Collect: `ssh blackwell 'cat /data/ugrip2026/journal/runs/chain_{A,B,C}.log'`; per run:
`<run_id>.log` and `<run_id>/runtime.json`.

Incident (2026-10-14 14:11): the first launch had no thread cap. Each of the 48 tokenizer workers
started ~one thread per core (512 cores) → ~9k threads, ~400 cores, box load 2,800. Stopped by
Taha at ~14:25 (still tokenizing, nothing lost; logs `chain_*_aborted_threads.log`). Relaunched
with `run_chain.sh` capping OMP/MKL/BLAS at 4 threads per process: 168 threads, ~41 cores.
