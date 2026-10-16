# Experiment plan v2: "Sharing or Specializing?"

The working plan for the IEEE TMI paper (submission ~15 Nov 2026; GPU access until 13 Nov).
It says what we run, in what order, with which settings, and what each run is for. It follows
the research update of 14 Oct 2026 (`references/plan_updates/research_update_2026-10-14.pdf`).

**v1 is archived** in `PLAN_v1_2026-10-11.md` (frozen). It holds Tausifa's 12 protocol answers
(its Appendix A) and the full designs of the studies v2 dropped. §1 below is v1's protocol with
the changes marked **(v2)**.

**Changelog**
- 2026-10-16: v2 written from the 14 Oct research update and Taha's decisions of 16 Oct:
  - the five arms of the update's table are the **main experiments**, run first;
  - 3 seeds and both backbones (Qwen3.5-9B, MedGemma) for **every** experiment;
  - the **additional experiments** (transfer matrix, modality and dataset adapters,
    predicted routing, representations) run only **after** the main experiments' training
    runs are done;
  - RQ2 (scaling) runs **last**, as designed in v1 S3. It does not fit the budget; its scope
    is decided once the earlier phases are done;
  - parameters are counted as **total stored parameters**. The arms are deliberately
    unbalanced, and Shared r48 is the control for that;
  - dropped: adapter composition (fixed and learned), Hydra-L, latent experts (v1 S6),
    location (v1 S8), practical factors (v1 S9), method tuning (v1 S11);
  - team roles are to be agreed separately (§7).

---

## The study in one page

**Question.** When one VLM is adapted to many heterogeneous medical tasks, which adaptation
parameters should be shared, and which should be specialized?

**Research questions**
- **RQ1.** Does task-specific adaptation outperform shared adaptation, and for which task
  families?
- **RQ2.** How does the amount of task-specific supervision change the optimal degree of
  sharing?
- **RQ3.** What determines whether tasks help or hurt each other: output format, imaging
  modality, or dataset?
- **RQ4.** Which measurable task characteristics predict whether tasks benefit from shared or
  specialized adaptation?

**Backbones:** Qwen3.5-9B and MedGemma-1.5-4B. **Data:** FLARE 2026 Task 3 (17 datasets,
7 modalities, 6 task families).

**Main arms: the degree of sharing.** LoRA learns ΔW = B·A. A (rank × input width) projects
the input down; B (output width × rank) maps it back up.

| Arm | Run-ID tag | A (input projection) | B (output projection) | Sharing | Parameters per layer | What it tells us |
|---|---|---|---|---|---|---|
| Shared LoRA | `shared` | shared | shared | full | 1A + 1B | Can one adapter serve all tasks? |
| HydraLoRA, task heads | `hydraT` | shared | per task | partial | 1A + 6B | Do tasks share input features but need their own output mapping? |
| Reversed HydraLoRA | `hydraR` | per task | shared | partial | 6A + 1B | Does it matter *which* matrix is shared, or only how much? |
| TaskLoRA | `task{cls,…}` | per task | per task | none | 6A + 6B | Is full specialization worth six times the parameters? |
| Shared LoRA, rank 48 (control) | `shared48` | shared | shared | full | ≈ 6A + 6B | Is a TaskLoRA gain due to specialization, or only to more parameters? |

**Plan at a glance (execution order, §5)**

| Phase | What | Answers |
|---|---|---|
| **A. Main experiments** | E0 zero-shot anchors; E1 the five arms on Qwen; E2 the five arms on MedGemma | RQ1 |
| **B. Additional experiments** (after Phase A training) | E3 transfer matrix; E4 modality adapters; E5 dataset adapters; E6 predicted routing; E7 representations; E8 RQ4 synthesis | RQ3, RQ4, RQ1 robustness |
| **C. Supervision scaling** (last; over budget) | E9 = v1 S3 on both backbones | RQ2 |

**Glossary**

| Term | Meaning |
|---|---|
| A∪B | the training pool, `internal_splits_v2/A.jsonl ∪ B.jsonl` (40,094 questions; contains dev, §1.7) |
| Holdout | `internal_splits_v2/true_test.jsonl`, 4,447 scored questions; the reporting surface |
| Question / row | one question–answer pair with all its images; the unit of training and evaluation |
| Cell | one (task, dataset) pair of the holdout; there are 19 |
| Oracle route | picking the head/adapter from the known task (or modality, dataset) label |
| ZS-bare / ZS-prompt | zero-shot base model without / with one generic system prompt (§1.6) |
| P8 | parameter count of one rank-8 LoRA on the Qwen target modules: 21,639,168 |
| Slot-hour | one hour of one of the 3 concurrent run slots on GPU 1 (§2) |
| C_ij | transfer score: how task i's adapter changes task j's metric relative to the base model (E3) |
| μ / PGF | normalised task score, Sachdeva et al.'s "perfection gap factor" (§1.8) |
| LF | LLaMA-Factory v0.9.5, our training framework |
| iugc | the ultrasound dataset whose images carry cls, detection and regression questions |
| W-numbers | code work packages (§3), named as in v1 |

---

## 1. Fixed protocol

Every experiment follows this protocol. An experiment may change **one** setting, and only
when that setting is its subject (the grouping level in E4/E5, the data amount in E9). Any
other change is a logged deviation.

### 1.1 Backbones and loading
- **Qwen3.5-9B** (`/data/ugrip2026/models/Qwen3.5-9B` on the box; paths in
  `ops/cluster-config.md`).
  - Template: LF `qwen3_5`, `enable_thinking: false` (the empty `<think>` block sits in the
    prompt, not the target).
  - Image budget: **768² area cap** (`image_max_pixels: 589824`).
- **MedGemma-1.5-4B** (`google/medgemma-1.5-4b-it`).
  - Template: LF `gemma3`.
  - Image budget: **native fixed 896² (256 tokens)**. The SigLIP input size is fixed, so the
    768² cap cannot apply. This is a backbone-intrinsic difference, stated in the paper.
- **Both:**
  - 4-bit bnb NF4, BF16 compute, `double_quantization: false`;
  - frozen: all base weights, the vision encoder, the multimodal projector;
  - trainable: LoRA parameters (and Hydra heads) only.
- Qwen3.5 traps, pinned in training **and** evaluation:
  - `enable_thinking=False`;
  - `eos_token_id=[<|im_end|>, <|endoftext|>]` (without it the model babbles a fake user turn);
  - `do_sample=False`.

### 1.2 LoRA defaults
- `r = 8`, `alpha = 16` (scale α/r = 2), `lora_dropout = 0.05`, for every arm **except
  Shared r48** (§1.3).
- **Target modules:** every linear of the language model, i.e. what LF `lora_target: all`
  resolves to. Frozen list: `code/train/target_modules_qwen35.json` (248 linears). Vision
  tower and projector excluded. Every arm wraps exactly this set.
- **MedGemma:** the equivalent set on the Gemma3 language model, frozen into a JSON file by the
  MedGemma port (W9). Gemma 4 needed an explicit regex because PEFT could not wrap its vision
  layers; check whether Gemma3 does too.

### 1.3 Parameter counting (v2: replaces v1's active-parameter matching)
- Arms are compared **as designed**, with their total stored parameters as in the table
  below. The imbalance is intended: the degree of sharing *is* the variable.
- **Shared r48 is the control** for TaskLoRA's 6× parameters: it has about as many
  parameters as six r8 adapters (48 = 6 × 8), but they are all shared.
  - **α = 96** for r48, so that α/r = 2 as in every other arm. With α = 16 the update would be
    scaled by 0.33 instead of 2, i.e. 6× weaker, and the control could lose for that reason
    alone.
- Every results table lists, per arm, both the **stored** parameters and the parameters
  **active per example**.

| Arm | Configuration | Active per example | Stored |
|---|---|---|---|
| Shared | A r8, B r8 | P8 | P8 |
| Hydra-T | shared A r8; 6 heads B_t r8; oracle task picks B_t | P8 | 1A + 6B |
| Reversed Hydra | 6 heads A_t r8; shared B r8; oracle task picks A_t | P8 | 6A + 1B |
| TaskLoRA | 6 independent r8 adapters; oracle task picks one | P8 | 6 · P8 |
| Shared r48 | A r48, B r48, α 96 | 6 · P8 | 6 · P8 |
| Modality adapters (E4) | 7 independent r8 adapters; oracle modality picks one | P8 | 7 · P8 |
| Dataset adapters (E5) | 10 independent r8 cls adapters; oracle dataset picks one | P8 | 10 · P8 |

For Qwen, A and B are close in size on most target modules, so 1A + 6B and 6A + 1B are each
about 3.5 · P8. The exact counts are printed at load and recorded in `runtime.json`.

### 1.4 Common training recipe
- AdamW, weight decay 0; lr **2e-4**, linear decay, 3% warmup; **3 epochs**.
- Effective batch 8 as `per_device_train_batch_size=1` × `gradient_accumulation_steps=8`
  (needed for the per-example loss, §1.5).
- bf16, gradient checkpointing, `cutoff_len 8192`.
- The **final** checkpoint after 3 epochs; no early stopping. End-of-epoch checkpoints are kept
  for diagnostics only.
- Equal epochs = equal exposure to every example. Step counts therefore differ per adapter;
  that is expected and recorded.
- **Seeds: 1337, 2026, 42, for every arm of every experiment, on both backbones.** The seed
  sets the LoRA initialisation and the data order. Data subsets are fixed separately (§1.9).
- No per-method tuning: every arm uses this recipe.

### 1.5 Data unit, sampling and loss
- **One row = one question** with all its images (up to 4). Multi-image questions are not split.
- **Sampler:** the natural A∪B distribution: concatenate the task decks and shuffle uniformly.
  No task or dataset balancing.

| Task | A∪B questions | Share | Holdout questions | Holdout datasets |
|---|---:|---:|---:|---:|
| Classification | 15,218 | 38.0% | 1,689 | 10 |
| Report generation | 7,016 | 17.5% | 781 | 1 |
| Multi-label | 6,415 | 16.0% | 703 | 2 |
| Detection | 5,280 | 13.2% | 588 | 3 |
| Regression | 5,265 | 13.1% | 586 | 2 |
| Counting | 900 | 2.2% | 100 | 1 |
| **Total** | **40,094** | | **4,447** | **19** |

- **Loss:** token cross-entropy averaged within each example, then over the 8 examples of the
  effective batch. A long report weighs the same as a one-token label. Implemented in W2 by
  ignoring HF's `num_items_in_batch`; unit-tested.
- Decks: `/data/ugrip2026/journal/data` on the box; md5s in `data/DECK_MANIFEST_blackwell.json`.

### 1.6 Prompts, decoding, parsing
- **Bare prompts** for every task and arm, in training and evaluation: the dataset's question
  as given, no system prompt, no added instruction. The target is the dataset's answer string.
- **Zero-shot, two protocols** (inference only, both backbones; nothing else differs):
  - **ZS-bare:** the bare question.
  - **ZS-prompt:** the same plus one task-agnostic system prompt: *"Answer the question about
    the medical image(s). Return only the requested answer, without an explanation."* It is the
    anchor `M_base` for μ and C_ij. It is never used in training.
  - MedGemma folds the system prompt into the first user turn; the text stays identical.
- **Decoding:** greedy, `do_sample=False`, EOS pinned, `enable_thinking=False`,
  `max_new_tokens=512` for every task, arm and backbone. Log the truncation rate.
- **Parsers:** the existing scorer, `score_single.py`. Do not re-implement them.
- **Images:** up to 4 per question, each preceded by `Image-i:`.
- **Excluded:** instance detection (unscored); endoscopy (out of the decks and the cls macro).

### 1.7 Data splits (v2: updated by the 2026-10-13 W1 finding)
- **Train: A∪B (40,094).** `A.jsonl` = `A_train ∪ A_dev` (likewise B), so dev is inside the
  training pool and cannot serve as an untouched selection set.
- **Report: holdout, 4,447 questions.** Each finished arm is evaluated on it once.
- **Selection decisions:** the experiments in this plan need none (no λ grids, no router
  tuning, no early stopping). If one comes up, it is made on the holdout and flagged in the
  paper as optimistically biased; the alternative is splitting the holdout into a selection
  half and a reporting half.
- **Optional second surface:** `validation-public-clean.jsonl` (5,340 questions, no
  regression), main arms only, in a separate labelled table.
- v1's A→B / B→A cross-fold design stays dropped (Tausifa Q7). Seeds and bootstrap CIs partly
  replace it; state this in Limitations.

### 1.8 Metrics and aggregation
- **Per-dataset metric, then the unweighted macro over datasets within the task.** Never pool
  across datasets.
  - cls: balanced accuracy; ML: micro-F1 (sample-F1 as secondary); det: F1 at IoU > 0.5;
    counting: MAE; regression: MAE per dataset, then macro, plus boneresorption alone;
    report: CRIMSON.
- **Every run also logs:** malformed-output rate per task, truncation rate, report omission
  rate (`code/eval/reportgen_omission.py`).
- **μ (PGF), for aggregate plots and E8 only, never as a headline:** against ZS-prompt of the
  same backbone. Higher-is-better metrics: `μ = (M − M_base)/(1 − M_base)`; MAE:
  `μ = (MAE_base − MAE)/MAE_base`. Recompute against ZS-bare and report what changes.
- **Power warning:** counting n=100, boneresorption n=70, BUS-UCLM-det n=22. Wide CIs there.

### 1.9 Subsets for scaling (used only by E9)
Already built (W1, `data/subsets/`): nested subsets of the target task's image groups,
stratified by dataset (and label for cls), seed `20261009`, p ∈ {1, 5, 10, 25, 50, 100}%, plus
sensitivity draws `20261010`, `20261011` at {5, 10, 25}%. Only target-task questions are
dropped; on shared iugc images the other tasks keep theirs (v1 D5).

### 1.10 Run registry and provenance
- **Run ID:** `j26_{backbone}_{arm}_{data}_{res}_s{seed}`; backbone `q35` or `mg15`; e.g.
  `j26_q35_hydraR_full_768_s2026`, `j26_mg15_shared48_full_896_s42`,
  `j26_q35_modxray_full_768_s1337`, `j26_q35_dsbonemarrow_full_768_s1337`.
- **Outputs** under `/data/ugrip2026/journal/runs/<run_id>/`: `config.yaml`,
  `git_commit.txt`, deck md5s, `trainer_log.jsonl`, adapter, `predictions_holdout.jsonl`,
  `metrics.json`, `runtime.json` (s/it, peak memory, GPU id, co-tenants, parameter counts).
- **Registry:** one row per run in `results/RUNS.csv`. **Numbers of record:** `results/LEDGER.md`.
  Raw predictions are archived into `results/predictions/` when each experiment closes.

### 1.11 Statistics
- **Per task:** 95% CI from 10k bootstrap resamples of holdout questions, stratified by
  dataset, with the macro recomputed on each resample.
- **Arm differences:** paired bootstrap on the same questions.
- **3 seeds:** mean ± SD over seeds; CI from a two-level bootstrap (seeds, then questions).
- **Multiplicity:** Holm across the 6 tasks within each pairwise comparison. Report adjusted
  and unadjusted values.

### 1.12 Same box for everything (v2)
Training is not bit-reproducible across machines: ws and Blackwell gave different greedy
outputs from the same adapter recipe (STATUS, 2026-10-14). **Every run that enters a
comparison is trained and evaluated on `mbz-yaqub` GPU 1.** Same-box run-to-run noise exists
too (~1% in final loss); seeds cover it.

---

## 2. Compute model

### 2.1 Measured on GPU 1 (3 runs at once, Qwen, 768²)

| Run (seed 1337) | Wall time, one slot |
|---|---:|
| Shared, full A∪B | ≈ 48.5 h (finishing 16 Oct) |
| TaskLoRA cls | 26.9 h |
| TaskLoRA report | 10.0 h |
| TaskLoRA ML | 9.0 h |
| TaskLoRA reg | 6.0 h |
| TaskLoRA det | 4.5 h |
| TaskLoRA count | 1.2 h |
| **6 TaskLoRAs, total** | **57.6 h** |

- 3 runs at once: 1.75 s/example each, 2.0× the throughput of one run alone. Peak ~17 GB
  reserved per run (~51 of 96 GB). 4 at once is untested.
- Each run's first pass tokenizes its deck (~35–40 min for the full deck), then caches it.

### 2.2 Planning constants (slot-hours)

| Item | Estimate | Basis |
|---|---:|---|
| Qwen, one full-mixture run (Shared, Hydra-T, Reversed Hydra) | ≈ 50 | measured Shared; Hydra overhead ×1.05 assumed |
| Qwen, Shared r48 | ≈ 55 | rank 48 adds a little compute; measure in its smoke gate |
| Qwen, 6 TaskLoRAs | ≈ 58 | measured |
| **Qwen, the 5 main arms, one seed** | **≈ 265** | |
| MedGemma, one full-mixture run | ≈ 8 | v1 estimate (7.2 ws GPU-h); **unmeasured on the box** |
| Qwen, one holdout eval (all tasks + CRIMSON) | ≈ 2.5 | v1: 2.1 h on ws |
| MedGemma, one holdout eval | ≈ 1 | estimate |

### 2.3 Capacity
- 16 Oct → 11 Nov (12–13 Nov kept for final evals and slack): 26 days × 24 h × 3 slots ≈
  **1,870 slot-hours**; at ~85% usable, **≈ 1,600**.
- Evaluation, zero-shot and smoke runs share the same three slots (one queue,
  `code/train/run_chain.sh`). Check `nvidia-smi -i 1` before each launch.

---

## 3. Foundation and code to build

### 3.1 State of the foundation

| Package | What | State |
|---|---|---|
| W1 Data | decks, holdout file, subsets, md5 manifest, task/modality/dataset labels on every row | **done** 2026-10-13 |
| W2 Training | `journal_train`, per-example loss, plug-in hook, registry | **done** 2026-10-13; smoke gate passed on GPU 1, 2026-10-14 |
| W3 Eval | holdout eval of single adapters and oracle-routed adapter sets, plug-in methods, scoring, CRIMSON, omission rule, bootstrap | written (branch `taha/w3-eval`), **untested** |
| Interface note | `code/INTERFACE.md` + a reference plug-in | to write once W3 works |

### 3.2 Code to build for v2 (in order of need)

| Package | For | What |
|---|---|---|
| **W4 HydraLoRA module** | E1, E2 (and E9) | One plug-in (`code/methods/hydra.py`) replacing each target `Linear4bit` with `y = W0·x + s·B·(A·x)`, where the oracle `task_id` picks the head. Two modes: **`hydraT`** (shared A, 6 B_t heads) and **`hydraR`** (6 A_t heads, shared B). Standard LoRA init in both (A random, B zeros). Prints the stored and active parameter counts at load. Learned routing (Hydra-L) is not needed. |
| Shared r48 config | E1, E2 | A config only: `lora_rank: 48`, `lora_alpha: 96`. |
| **W9 MedGemma port** | E0, E2 and every later experiment | Gemma3 template and the train/inference byte-identity gate; the language-model target list (no vision modules); detection coordinates at 896² (targets in original pixels, as for Qwen; check on a smoke run that the model learns the mapping); W4 ported to Gemma3 layer names. |
| W11a Transfer analysis | E3, E8 | `code/analysis/transfer_matrix.py`: C_ij, PGF μ_{i→j}, transferability/malleability (Sachdeva et al., Eqs. 1–3), malformed/truncation table, covariate correlations with permutation tests, heatmap. CPU only. |
| Routers | E6 | Task route: the question-only GLiClass router; confirm it never saw holdout data, else retrain on A∪B questions (minutes). Modality route: the ResNet-18 modality classifier, trained on A∪B images. |
| W8 Representation tooling | E7 | `code/analysis/representations.py`: forward-only hooks, debiased linear CKA, RSA, permutation tests; head-similarity for Hydra. |
| W11b Scaling model | E9 | `code/analysis/scaling_model.py`: the mixed-effects interaction model, crossover p*, curves. CPU only. |

### 3.3 Smoke gate (every method and backbone, before a full GPU slot)
1. Loss decreases over 50 training steps.
2. Stored and active parameter counts match §1.3.
3. Train/inference prompt render is byte-identical (`template_gate.py`).
4. Eval runs end to end on ~50 holdout questions with a sensible malformed and truncation
   rate.
5. `runtime.json` records s/it, peak memory and parameter counts (these replace the §2.2
   estimates).

The foundation owner reviews the gate output, not the whole implementation.

---

## 4. Experiments

### Phase A — Main experiments

#### E0. Zero-shot anchors
- ZS-bare and ZS-prompt (§1.6) on the holdout, for Qwen and MedGemma.
- Per task: metric, malformed rate, truncation rate, and ZS-prompt − ZS-bare (how much of the
  zero-shot gap is format).
- Needed for μ (§1.8) and C_ij (E3). Challenge-era zero-shot runs are not reusable
  (different resolution, prompts and protocol).
- **Cost:** ≈ 5 (Qwen) + 3 (MedGemma) slot-h.
- **Also:** W3's first real test is evaluating the S1 seed-1337 Shared and TaskLoRA adapters.

#### E1. The five arms on Qwen3.5-9B (RQ1)
- **Arms:** Shared, Hydra-T, Reversed Hydra, TaskLoRA, Shared r48 (§1.3), each × seeds 1337,
  2026, 42, on the full A∪B mixture. Hydra-T and Reversed Hydra train on the full mixture with
  the oracle task picking the head; each TaskLoRA trains on its task deck only.
- **Evaluation:** holdout, all tasks, oracle task route where the arm needs one.
- **Contrasts (each per task, paired bootstrap, Holm over 6):**

  | Contrast | Question it answers |
  |---|---|
  | TaskLoRA − Shared | the headline: does specialization help, and for which task families? |
  | TaskLoRA − Shared r48 | is that gain specialization, or just 6× the parameters? |
  | Hydra-T − Reversed Hydra | does it matter which matrix is shared? |
  | Hydra-T, Reversed Hydra − Shared and − TaskLoRA | where partial sharing falls between the extremes |

- **Also report:** per-dataset deltas grouped by modality; malformed, truncation and report
  omission rates; an output-format error table (does any shared arm emit the wrong task's
  format?); task-mean μ.
- **Prior evidence** (challenge, different recipe, not reused as data): specialists were better
  on regression, cls and detection; ML and counting tied; the general adapter was slightly
  better on reports.
- **Done so far:** seed 1337 TaskLoRA (all 6) trained; seed 1337 Shared finishing 16 Oct.
- **Cost:** 3 × 265 ≈ 800 slot-h (≈ 105 already spent) + 15 evals ≈ 40.

#### E2. The five arms on MedGemma-1.5-4B (RQ1, cross-backbone)
- Same arms, seeds and evaluation as E1, after the W9 port passes the gate. Run IDs use
  `mg15` and `896`.
- **Analysis:** `Δ_spec = M_TaskLoRA − M_Shared` per task for each backbone, compared with
  bootstrap CIs. Read it alongside the differences in size, resolution and pretraining (4B,
  fixed 896², medical SigLIP).
- **Risk:** MedGemma was very weak on detection in the organiser baselines (0.092 F1). Once the
  coordinate convention is verified, low detection numbers are a finding, not a bug.
- **Cost:** 15 runs × ≈ 8 + 15 evals ≈ 135 slot-h (MedGemma speed unmeasured).

### Phase B — Additional experiments

> **Phase B GPU work starts only after every Phase A training run has finished** (E1 and E2,
> all seeds). Code, routers and analysis scripts for Phase B can be built at any time. Every
> Phase B experiment runs on both backbones with 3 seeds.

#### E3. Transfer matrix (RQ3, RQ4)
- **Design:** evaluate each TaskLoRA_i on every other task j's holdout slice: a 6 × 6 grid per
  seed and backbone (the diagonal comes from E1/E2), plus the base rows from E0.
  - `C_ij = M_j(TaskLoRA_i) − M_j(base)`, metric-oriented, base = ZS-prompt. Recompute against
    ZS-bare and list the cells whose sign flips.
  - Also PGF `μ_{i→j}`, transferability Δ(i)± and malleability Θ(j)±.
  - Malformed and truncation rate per cell: format lock-in was the main negative-transfer
    mechanism in the challenge.
  - CRIMSON scores from non-report adapters will mostly be junk. Keep and flag them.
- **RQ3 analysis:** relate C_ij to pair covariates computed from the W1 labels: output-structure
  similarity (hand-coded: label / label-set / boxes / integer / scalar / free text), modality
  overlap (Jaccard), dataset overlap, source and target data size. 30 off-diagonal pairs:
  Spearman correlations with permutation tests, not a multiple regression. Sign stability across
  the 3 seeds (Wilcoxon).
- **Natural experiment:** iugc carries cls, detection and regression questions on the *same*
  ultrasound images, so transfer between those three holds modality and dataset fixed and
  varies only the output format.
- **Cost:** Qwen ≈ 12 per seed → 36; MedGemma ≈ 6 per seed → 18. **≈ 54 slot-h**, all
  inference.

#### E4. Modality adapters (RQ3)
- **Design:** one r8 adapter per imaging modality, trained on all of that modality's A∪B
  questions, whatever the task. Modalities: X-ray, Ultrasound, Mammography, Microscopy,
  Dermatology, Retinography, Clinical. Endoscopy skipped (15 questions).
- **Evaluation:** each modality adapter on its modality's holdout questions (oracle modality
  route), compared on the same questions with Shared (one group), TaskLoRA (grouped by task)
  and the five E1/E2 arms.
- **Question:** is specializing by *modality* better or worse than specializing by *task*? Put
  together with E5, this asks at which level sharing should be cut: task, modality or dataset.
- Composition (task + modality adapters combined) is out of scope (§6, N5).
- **Prior evidence** (challenge, 8B): modality adapters lost to task specialists on cls but beat
  the general adapter; X-ray ML was the one slice where modality was competitive.
- **Cost:** Qwen ≈ 50 per seed (the 7 adapters together cover the mixture) + evals → ≈ 165;
  MedGemma ≈ 30. **≈ 195 slot-h.**

#### E5. Dataset adapters within classification (RQ3)
- **Design:** one r8 adapter per cls dataset (10), each trained on that dataset's A∪B cls
  questions, evaluated on its holdout slice (oracle dataset route), compared with TaskLoRA-cls
  and Shared on the same questions.
- **Analysis:** the per-dataset gain over TaskLoRA-cls against dataset size and zero-shot
  difficulty. Does specializing below the task level help only for large or visually distinct
  datasets?
- **Cost:** Qwen ≈ 30 per seed (about one cls TaskLoRA) + evals → ≈ 95; MedGemma ≈ 15.
  **≈ 110 slot-h.**

#### E6. Predicted routing (RQ1 robustness)
- Every specialized arm needs a route at test time; E1–E5 use the oracle label. Here the label
  is predicted:
  - task route (TaskLoRA, Hydra-T, Reversed Hydra): the question-only GLiClass router;
  - modality route (E4 adapters): the ResNet-18 modality classifier.
- **Report:** routing accuracy per task/modality, and the end-to-end metric drop versus the
  oracle route. Does the specialization gain survive without labels?
- **Cost:** ≈ 24 evals → **≈ 45 slot-h**.

#### E7. Representations (RQ4)
- **Probe set:** a fixed, stratified holdout subset of 30 questions per cell (≈ 570); commit
  the IDs.
- **Models:** base, Shared, Shared r48, TaskLoRA (each adapter on all probe questions), Hydra-T,
  Reversed Hydra, modality and dataset adapters; 3 seeds give the seed-to-seed noise ceiling.
- **Features:** mean-pooled image tokens and question tokens, plus the last prompt token, at a
  fixed set of LLM layers (proposed for Qwen: 1, 8, 16, 24, 32, 40, 48, 56, 64; owner sets the
  MedGemma list).
- **Measures:**
  - **input similarity:** RSA/centroid similarity between base-model representations of
    source task i's inputs and target cell c's inputs; correlated with transfer C_i,c (≈ 110
    task–cell pairs; permutation test shuffling target cells; Holm across layers). This is the
    main test;
  - **adapter similarity:** debiased linear CKA between TaskLoRA_i and TaskLoRA_j on the same
    inputs, per layer, against C_ij (descriptive only: 6 tasks are too few for a test);
  - **shift from base:** CKA(model, base) per layer: where does specialization appear?
  - **Hydra heads:** pairwise cosine similarity of the B_t heads (Hydra-T) and of the A_t heads
    (Reversed Hydra) across tasks. Which matrix actually diverges between tasks?
- **Cost:** forward passes only, **≈ 30 slot-h**; storage ≈ 200 MB per configuration.

#### E8. RQ4 synthesis (CPU)
- **Unit:** the 19 cells, per backbone. **Outcome:** the specialization benefit per cell,
  TaskLoRA − Shared (and Hydra-T − Shared, Reversed Hydra − Shared), averaged over seeds.
- **Candidate predictors:** training size of the cell and of its task, number of datasets in
  the task, output-structure type, modality, zero-shot score (difficulty), malleability Θ(j)
  and transferability Δ(i) from E3, input and adapter similarity from E7.
- **Method:** Spearman correlations with permutation tests, reported per predictor (19 cells
  are too few for a joint regression). Check that the sign agrees across backbones.

### Phase C — Supervision scaling

#### E9. Supervision scaling (RQ2)
- **Runs last.** Its scope is decided once Phases A and B are done, from the compute left.
  As designed it does **not** fit (§5.2). Until then it stands as in v1:
- **Design (v1 S3, unchanged):**
  - targets: cls (15,218 questions) and regression (5,265; caveat: its holdout macro is
    iugc-dominated; report boneresorption separately);
  - fractions p ∈ {1, 5, 10, 25, 50, 100}% of the target task's image groups (§1.9); other tasks
    stay at 100%; the 100% point is reused from E1/E2;
  - arms: TaskLoRA (target subset only), Shared and Hydra-T (mixture with the subsetted target);
  - 3 seeds on the same nested subsets, plus the 2 sensitivity draws at {5, 10, 25}%, 1 seed;
  - both backbones (v2: MedGemma gets the full design, not v1's reduced one).
- **Analysis:** `μ = β0 + β1 log N + β2 S + β3 (log N × S) + u_dataset + ε` per (task,
  dataset, arm, p, seed), with TaskLoRA as the reference arm. Main test: β3. Also the crossover
  p* where TaskLoRA ≥ Shared, with a bootstrap CI; per-dataset fits for cls.
- **Expected issue:** at 1%, cls has ~150 questions and regression ~53. Keep the malformed-rate
  column.
- **Cost as designed:** ≈ 4,640 (Qwen, v1 estimate) + ≈ 500 (MedGemma) slot-h.

---

## 5. Order of execution and budget

### 5.1 Order
The order is for organisation. Runs inside a step can share the three slots; a step can start
while the previous one is finishing, **except** that Phase B GPU work waits for all Phase A
training runs. Code for later steps is built in parallel with earlier runs.

1. **Now (Phase A):**
   - E0 Qwen zero-shot; holdout evals of the seed-1337 Shared and TaskLoRA (W3's first test);
   - Shared r48 seed 1337 (config only, smoke gate first).
2. **Phase A, no new code needed:** Shared, TaskLoRA and Shared r48, seeds 2026 and 42.
3. **Phase A, after W4 passes the gate:** Hydra-T and Reversed Hydra × 3 seeds.
4. **Phase A, after W9 passes the gate:** E0 MedGemma; the 5 arms × 3 seeds (E2). MedGemma runs
   are short: slot them in whenever one is free.
5. Holdout evals of every Phase A arm; RQ1 analysis.
6. **Phase B** (all Phase A training done): E3 → E4 → E5 → E6 → E7 → E8. E6 needs E4's adapters;
   E7 needs every adapter; E8 needs E3 and E7.
7. **Phase C:** E9, scoped to the compute left.

### 5.2 Budget (slot-hours on GPU 1)

| Item | Slot-h |
|---|---:|
| E0 zero-shot | ≈ 8 |
| E1 Qwen main, 3 seeds, remaining | ≈ 695 |
| E1 evals | ≈ 40 |
| E2 MedGemma main, 3 seeds, incl. evals | ≈ 135 |
| **Phase A remaining** | **≈ 880** |
| E3 transfer matrix | ≈ 54 |
| E4 modality adapters | ≈ 195 |
| E5 dataset adapters | ≈ 110 |
| E6 predicted routing | ≈ 45 |
| E7 representations | ≈ 30 |
| **Phase B** | **≈ 435** |
| **A + B** | **≈ 1,315** of ≈ 1,600 usable |
| Left for Phase C | **≈ 285** |
| E9 as designed | ≈ 5,100 |

- At ~60 usable slot-hours a day, Phase A ends around **31 Oct** and Phase B around **7 Nov**.
- The estimates for Hydra, r48 and MedGemma are unmeasured; their smoke gates replace them.
  Testing 4 runs at once could add capacity.
- **Phase C as designed is ~18× what is left.** Options, for when we get there: one target
  (cls), fewer fractions (e.g. 5, 25, 50), fewer seeds, no sensitivity draws, MedGemma first
  (≈ 9× cheaper).

---

## 6. Decisions

**v2 decisions (Taha, 2026-10-16)**

| # | Decision |
|---|---|
| N1 | The five arms of the 14 Oct update are the main experiments and run first. |
| N2 | Parameters are counted as total stored parameters. Arms are unbalanced by design; Shared r48 (α 96) is the parameter control. |
| N3 | 3 seeds (1337, 2026, 42) for every arm of every experiment. |
| N4 | MedGemma-1.5-4B runs every experiment, not only the main ones. |
| N5 | Kept from v1 as additional experiments, after Phase A: transfer matrix (S2 → E3), modality adapters (S5a → E4), dataset adapters (S5b → E5), predicted routing (S4c → E6), representations (S7 → E7). No adapter composition anywhere: v1 S4b and the composed arms of S5a/S5b are dropped. |
| N6 | Dropped: Hydra-L, latent experts (S6), location (S8), practical factors (S9), method tuning (S11), v1's Med-MoE-style arm. |
| N7 | RQ2 (v1 S3) runs last and is scoped to the compute left. |
| N8 | All compared runs are trained and evaluated on `mbz-yaqub` GPU 1 (§1.12). |

**Carried over from v1** (numbers as in `PLAN_v1_2026-10-11.md` §6)
- D1: the final 3-epoch checkpoint for every arm; no early stopping.
- D5: scaling on shared iugc images drops only target-task questions.
- D6: validation-public as an optional second surface, main arms only.
- D7: bare prompts everywhere; ZS-bare and ZS-prompt for zero-shot, ZS-prompt as anchor.
- D10: MedGemma at native 896².
- D11: seeds 1337 / 2026 / 42.
- D12: regression stays a scaling target; boneresorption and iugc reported separately.
- 2026-10-13: train on A∪B (dev included); selection decisions, if any, on the holdout (§1.7).

---

## 7. Team

To be agreed separately, after this plan. Until then, every GPU job goes through the one
GPU 1 queue (`code/train/run_chain.sh`).

---

## Appendix. Where v1's studies went

| v1 | v2 |
|---|---|
| S0 zero-shot | E0 |
| S1 Shared vs TaskLoRA | E1 (with Hydra-T, Reversed Hydra and Shared r48 added) |
| S2 transfer matrix | E3 |
| S3 scaling | E9 (last) |
| S4a Hydra-T | E1/E2 arm; Hydra-L dropped |
| S4b composition | dropped |
| S4c predicted routing | E6 |
| S5a modality | E4 (modality adapters only) |
| S5b dataset | E5 (dataset adapters only) |
| S6 latent experts | dropped |
| S7 representations | E7 |
| S8 location | dropped |
| S9 practical factors | dropped |
| S10 MedGemma | E2, and MedGemma in every other experiment |
| S11 method tuning | dropped |
