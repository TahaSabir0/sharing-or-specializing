# sharing-or-specializing (local folder: IEEE) — instructions for agents

Journal study for **IEEE Transactions on Medical Imaging**: "Sharing or Specializing? A
Systematic Study of Parameter-Efficient VLM Adaptation for Heterogeneous Medical Image
Analysis". Mentor: Dr. Tausifa Jan Saleem (PI: Prof. Mohammad Yaqub, MBZUAI). GPU access
(Blackwell GPU 1) runs until 13 Nov 2026; submission target ~15 Nov 2026.

Read **STATUS.md** first, then the parts of **PLAN.md** your task touches. PLAN.md §1 is
the fixed protocol every experiment follows.

## Who's who
- **Taha Sabir:** lead; foundation code, S0–S2, GPU queue, records. Explain ML reasoning
  plainly: Taha is comfortable with code but still building depth in ML.
- **Veronika Dmitrenko:** HydraLoRA (S4a) and MoE-style experts (S6).
- **Thinh Quang Ly ("Alex"):** composition, routing, modality (S4b, S4c, S5), location (S8),
  representations (S7).
- **Alberto Montero Molina:** scaling (S3), practical factors (S9), MedGemma (S10).
- **Mentors:** Dr. Tausifa Jan Saleem (this paper), Prof. Mohammad Yaqub, Ahmed Aly.

## Folder map

| Path | Contents |
|---|---|
| `STATUS.md` | current state, active runs, next steps (keep it current) |
| `PLAN.md` | the confirmed experiment plan; changelog at the top; Tausifa's 12 answers in Appendix A |
| `paper/` | `main.tex` (the manuscript we edit), `proposal_original.tex` (Tausifa's draft, frozen), `paper.bib` |
| `references/papers/` | the 11 method papers + `METHOD_DIGEST.md` |
| `references/challenge_results/` | challenge-era ledgers: read-only, never mixed into new results (see its README) |
| `references/FLARE2026_Hexapus_paper_submitted.pdf` | our challenge paper (cite it) |
| `data/` | split design (`FLARE2026_data_pipeline_v2.md`), dataset census, holdout composition; `subsets/` for the scaling subsets |
| `ops/` | compute guide for `ws` and Blackwell; `cluster-executor` agent definition |
| `code/` | all journal code: env, data, train, methods, eval, analysis, configs (see `code/README.md`) |
| `results/` | `LEDGER.md` (numbers of record), `RUNS.csv` (run registry) and `predictions/` |
| `transcripts/` | meeting transcripts + `SUMMARY.md` |

**Code** lives in `code/` of this repo; `code/README.md` explains what each part is, where
it came from, and who builds what. It was seeded from the challenge code repo
(`TahaSabir0/MICCAI-FLARE-2026-Task-3`, branch `Taha/qwen35-sprint`, commit `bec2d1f`).
Look things up there only for history; don't develop in it. Use one topic branch (and one
worktree) per piece of work, and don't switch a shared checkout's branch.

The old challenge workspace `C:/Users/tahas/research/UGRIP/` is history: look things up there only
when a specific question needs it (for example its `PROGRESS.md`). Don't load it by
default.

## Research rules
1. **Follow PLAN.md §1 exactly.** Bare prompts; 768² cap (Qwen); rank 8 / α 16 / dropout
   0.05; per-example loss with batch 1 × grad-accum 8; 3 epochs, lr 2e-4; greedy decoding,
   EOS pinned, `enable_thinking=False`, `max_new_tokens=512`. Any change is a logged
   deviation.
2. **Pin every framework default explicitly.** Our biggest past bugs came from unset
   defaults: sampling leaking in, a thinking-mode default, a missing EOS.
3. **Match before comparing:** same data, resolution, prompt, decoding and scorer. Use the
   existing scorer (`score_single.py`); don't re-implement parsers.
4. **Per-dataset metrics, then the macro within each task.** Never pool across datasets.
   Label every evaluation surface (holdout, dev, validation-public).
5. **Holdout is touched once per finished arm.** Every selection decision uses dev-select.
6. **Challenge numbers are not journal numbers** (`references/challenge_results/README.md`).
7. **Keep what's needed to reproduce a claim:** config, commit, deck md5, adapter, raw
   predictions, metric definition.
8. **Preserve negative results** and uncertainties; don't invent completion.

## Compute and execution
- Since 2026-10-14 all training and evaluation run on `mbz-yaqub` (Blackwell, **GPU 1 only**).
  `ws` is used by others: don't run jobs there. See the banner in `ops/cluster-config.md`.
  Check `nvidia-smi -i 1` before launching.
- Delegate remote GPU/data work to the `cluster-executor` agent (`ops/cluster-executor.toml`)
  with a concrete task, branch/commit, GPU and budget.
- Long jobs: launch durably (nohup + log), report PID, log path, ETA and the collection
  command, then return. Don't poll for minutes.
- Before any full run, a method must pass the smoke gate (PLAN.md §3.3).

## Keeping the record
- After substantial work, update `STATUS.md`. Add new numbers to `results/LEDGER.md`; add
  plan changes to the PLAN.md changelog.
- After each meeting, summarise it into `transcripts/SUMMARY.md` (the `UGRIP-transcript`
  skill).
- Never commit credentials, datasets, weights or large build artifacts. Passwords are not
  stored in this repo.
- Commit focused changes; don't force-push.
