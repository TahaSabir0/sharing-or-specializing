# Format analysis plan (v2, fixed 2026-11-22)

Revision of Tausifa's protocol (after the 21 Nov meeting). Changes from her version are marked **[changed]**
or **[new]**, and the list at the bottom says why.

**Question.** The Discussion currently says, without testing it, that "locking into the source task's output
format is a likely mechanism of negative transfer" (`main.tex`, E3 paragraph). Off-task transfer (E3, Table VI)
is mostly zero or negative. Is that mainly because a single-task adapter answers in its own task's format
(format lock-in), and does joint training avoid it because the question tells the model which format to use?

## General rules
- Data: the existing E3 off-task holdout predictions (both backbones, 3 seeds) and the E1/E2 diagonal.
  No new training.
- Cells: all 30 off-diagonal (source i, target j) pairs per backbone, with these exceptions:
  - Detection as target: format rates only (μ is 0 everywhere).
  - Report as target: format rates only, no CRIMSON interpretation.
  - MedGemma counting as target: format rates only, no μ (as in E3).
- **[changed]** Fix in advance: before computing any lock-in rate, commit the format rules, their unit tests
  (F0) and this plan to the repo. In the paper, say "rules fixed before the lock-in rates were computed", not
  "pre-registered", since the E3 predictions and μ values were already known.
- Run order: F0 → F1 → F7 → F3 → F6 → F5 (all CPU). Run F4 (GPU) only if its gate passes (see F4).

## F0. Format classifier (CPU)
Assign every answer exactly one format. Check in this order and stop at the first match:

1. **class label**: exactly one item, and it matches one of *that question's* options (letter or label
   text). Check this first, because some labels are numbers (DR grades 0–4).
2. **label set**: 2 or more comma-separated items, all from that question's label list.
3. **boxes**: the whole answer is a list of 4-number lists, e.g. `[[x1, y1, x2, y2], ...]`.
4. **integer**: only an integer.
5. **decimal**: only a number with a decimal point.
6. **[new] foreign label**: a short answer (1–3 items) whose items are labels from the *source adapter's*
   training label vocabulary but not from this question's options. This is lock-in to the source's
   vocabulary, not only to its format.
7. **free text**: prose. 4 or more words, 2 or more sentences, or one sentence closed by a full stop
   ("Cardiomegaly.", "Heart size normal."). Words are tokens with 2+ letters, so box lists and numbers
   never count. **[changed after F0 validation, before any lock-in rate]** The original ">15 words or >1
   sentence" rule marked 4–6% of the report adapter's own reports as unparseable.
8. **other**: empty, or none of the above.

Sub-flags (reported, they don't change the format):
- **non-canonical class label** (classification targets): a valid option given in a form the scorer
  marks wrong ("C. Moderate NPDR" or "Moderate NPDR" when the gold answer is "C"). F7c rescoring maps
  these to the letter.
- **flat box** (detection targets): `[x1, y1, x2, y2]` without the outer list. Counted as boxes but not as
  valid for detection, since the scorer needs nested lists.

Implementation: `code/analysis/format_rules.py`, tests `code/analysis/tests/test_format_rules.py`, label
vocabularies `code/analysis/format_vocab.json` (built from the training decks by `format_lockin.py vocab`).
Option parsing: lettered questions use their letters (canonical) and texts; unlettered classification
questions (CMMD) use the dataset's answer vocabulary as named in the question; multi-label questions use
the list after "from the following".

Notes:
- Don't reuse the scorer's parsers: they are lenient (the number parser pulls a number out of a box list).
- The source label vocabulary for rule 6 comes from each TaskLoRA's training data (the union of option
  lists across its training datasets). Commit it as a fixed file next to the rules.

Validation (all must pass before F1):
- Unit tests for every format and these edge cases: numeric class labels (DR 0–4); a single label given to a
  multi-label question; an integer given to a regression question; boxes with extra spaces or a trailing
  comma; a number inside a sentence; an empty answer; **[new]** a label that is valid for another dataset of
  the same task but not for this question; **[new]** a label that appears both in the question's options and
  in the source vocabulary (it must resolve to class label).
- Ambiguity check: count the answers that match more than one rule before precedence is applied, and say
  how they were resolved. Two overlaps are by design and resolved by the order: a label long enough to be
  prose (label rule wins), and a label that is both an option of this question and a source label (class
  label wins). Any other overlap is a real conflict and should be close to zero.
- Consistency check: run the rules on the diagonal (each TaskLoRA on its own task). Almost every answer
  should come out in its own task's format, consistent with the ~0% malformed rate already reported. List
  any exceptions.

## F1. Lock-in rates (CPU)
For each cell, seed and backbone:
- **valid-for-target**: the answer is in the target task's format. Overlaps count as valid: a single label
  is a valid answer to a multi-label question, and an integer is a valid answer to a regression question.
- **locked-in**: the answer is in the SOURCE task's format AND is not valid for the target, *or* it is a
  foreign label. Source formats: cls → class label, ML → label set, det → boxes, count → integer,
  reg → decimal, report → free text.
- **[new] under-cardinality (multi-label targets)**: the answer is a single label but the gold answer has 2
  or more labels. Report it next to the share of multi-label questions whose gold answer has 2+ labels
  (some gold answers have only one label, so a single label isn't always wrong). This is the likely
  mechanism behind cls→ml (Qwen μ = −0.397), which the overlap rule would otherwise score as 0% lock-in.
- **[new] integer-on-regression rate (regression targets)**: descriptive only. An integer may be a legitimate
  value, so it isn't counted as lock-in.

Report per cell (mean ± SD over seeds): lock-in rate, under-cardinality rate (ML targets), target-format
rate, and the full 8-format distribution.

Outputs:
- 6×6 lock-in heatmap per backbone, same row and column order as Table VI.
- Appendix table: the full format distribution per cell.

## F2. Lock-in vs transfer (CPU, descriptive) [changed]
- Pairs: off-diagonal pairs, excluding detection and report targets (and the counting target on MedGemma):
  Qwen 20 pairs (5 sources × 4 targets), MedGemma 15 (5 × 3).
- Scatter plot of lock-in rate against μ (both averaged over seeds), one dot per pair, with the most
  negative pairs labelled. Report Spearman ρ as a description only, **with no p-value**.
- Why no test: answers in the wrong format score about zero by construction, so lock-in lowers μ
  mechanically and a correlation is close to guaranteed. With 15–20 pairs the test would also be
  underpowered, like the RQ3 covariate test. F7, F3 and F4 are the real tests.
- Appendix only. In the paper, write "associated with", not "caused by".

## F3. Content on well-formed answers (CPU)
For each cell, keep only the questions the off-task adapter answered in the target format (Q_valid), and
compute on Q_valid:
- the off-task adapter's score, the ZS-prompt base score, and μ_valid (dataset macro over the datasets with
  at least one valid answer);
- the target TaskLoRA's own score on Q_valid vs on all questions. This is the bias check: is Q_valid simply
  the easier questions?
- **[new]** for ML targets, a second version of μ_valid with under-cardinality answers also excluded.

Report n and the % of questions retained. Cells with fewer than 30 questions or under 20% retained are
descriptive only. For classification with small subsets, report plain accuracy next to balanced accuracy.

## F4. Format-forced re-evaluation (GPU, inference only) [changed: gated and narrowed]
**Gate.** Run F4 only if F7a shows a sizeable format effect: |μ_F7a − μ| ≥ 0.05 in at least 3 eligible
cells on at least one backbone. Otherwise report F7 and F3 and stop there.

**Scope.** Targets: classification, multi-label, counting and regression. Not detection: base detection F1 is
0, and adapters that never learned boxes would output arbitrary boxes when forced. Not report.

Forcing, applied identically to (a) the off-task adapter, (b) the ZS-prompt base model and (c) the target
TaskLoRA on its own task. Everything else stays identical (greedy, pinned EOS, thinking off, same images).
- Classification: option scoring. Take the log-likelihood of each option string after the prompt (averaged
  per token when options are label texts of different lengths) and pick the maximum.
- Multi-label: constrain the output to comma-separated labels from the question's own list.
- Counting: constrained decoding, digits only, at least one digit, at most 8 tokens.
- Regression: digits plus at most one '.', at most 8 tokens.

Compute per cell:
- μ_forced = forced adapter vs forced base (content transfer);
- format effect = μ (unforced, Table VI) − μ_forced.

Uncertainty: the same two-level paired bootstrap as E3 (adapter vs base on the same questions), with Holm
across the targets of each source.

Sanity checks (report all three):
- Forcing the target TaskLoRA on its own task changes its score by less than ~0.01 (or ~2% of MAE). A
  bigger change means a forcing bug.
- 100% of forced outputs parse with the scorer.
- The mean log-probability of each forced answer, as a measure of how far forcing pushed the model from
  what it wanted to say.

Implementation note: forcing must be added to both harnesses (Qwen `eval_arm.py`, MedGemma
`eval_harness.py`) and needs its own smoke test on dev-select before the holdout runs.

## F4 v2. Forced decoding as run (Tausifa, 25 Nov; replaces the F4 scope above)
**Cells.** All 14 negative-transfer cells of Table VIII (μ < −0.05), so there is no selection question.
Qwen: Det→Reg, Det→Count, Det→ML, Cls→ML, Report→ML, Det→Cls, Reg→ML. MedGemma: Det→Reg, Cls→Reg,
Report→ML, Reg→ML, Det→ML, Count→ML, Report→Reg.

**Arms, same forcing for all.** The source TaskLoRA (3 seeds), the ZS-prompt base, and Shared (3 seeds, the
reference) on the same targets. μ_forced = (adapter_forced − ZS_forced) / (perfect − ZS_forced).

**Forcing** (`code/eval/forced_format.py`, harness `code/eval/forced_eval.py`, jobs `code/eval/jobs/f4/`):
- Count: regex `\d{1,5}`; Reg: `\d{1,5}(\.\d{1,3})?`; max_new_tokens 12.
- ML: `(o1|…)(SEP(o1|…))*` from the question's own options, SEP the question's own separator (`, ` chestdr;
  `; ` periapical, whose last option contains commas). Duplicates removed afterwards; the number of labels is
  the model's. max_new_tokens 64. If the cap cuts the last label, that partial label is dropped (seen in the
  smoke: MedGemma ZS-prompt loops through the list); truncation is counted per cell.
- Cls: no generation; each option is scored by log p(option + end-of-turn | prompt); the options are the
  letters for lettered questions (the gold answers are letters) and, for CMMD, the answer texts as the question
  writes them. **Primary: length-normalised** (mean per token); summed reported beside it. For lettered
  questions both give the same choice (every option is one token + end-of-turn); they differ only on CMMD.
- Regex enforcement without a new package (shared envs frozen; lm-format-enforcer not installed): a logits
  processor allows token t when generated-text + text(t) is a prefix of the language (`regex` partial match);
  end-of-turn (`<|im_end|>` Qwen, `<end_of_turn>` MedGemma) only on a complete match.
- Unchanged: bare prompt (ZS-prompt system prompt for the base), image budget, greedy, thinking off (Qwen),
  pinned stop tokens, batch 1.

**Statistics.** Table VIII's two-level paired bootstrap (`format_lockin.boot_job`, B = 10,000) for μ_forced,
and for the change μ_forced − μ (same resamples, each side against its own base). Holm across the forced targets
of each source adapter per backbone (Table VIII's family restricted to the forced cells), plus Holm across all
forced cells of a backbone as a sensitivity. Analysis: `code/analysis/forced_mu.py`.

**Checks.** Before the holdout: dev-select smoke, 5 rows per dataset per cell, outputs read by hand. In the
results: errors, truncation, incomplete answers, dead ends, scorer-unparseable answers, mean unconstrained
log-prob of forced answers, and Shared forced vs unforced answer agreement (Shared is 0% wrong-format, so
forcing should leave it nearly unchanged: a bug check).

## F5. Matched-prompt check (CPU) [changed: no GPU]
The adapters are evaluated with the bare prompt but the E3 base is ZS-prompt. A matched comparison already
exists: μ recomputed against ZS-bare (bare vs bare), whose sign flips are listed in the LEDGER's E3
diagnostics. Report those cells next to the lock-in heatmap. Don't re-run adapters with a system prompt:
they were trained without one, so adding it is a distribution shift and a PLAN §1 deviation.

## F6. Joint-training contrast (CPU)
Run the F0 classifier on all holdout predictions of Shared, Hydra-T, Hydra-R and Shared r48 (both backbones,
3 seeds). Report the cross-task format error rate (answers in another task's format, plus **[new]** foreign
labels and under-cardinality on ML), next to the TaskLoRAs' off-task rates from F1. Expected: ~0%. One small
table or one sentence beside the lock-in heatmap.

## F7. Counterfactual estimate (CPU) [changed: now the primary test, before F4]
In each cell, replace each locked-in answer with the ZS-prompt base model's answer to the same question,
scored normally even when that answer is itself malformed, and recompute μ (**F7a**). The change in μ
approximates the loss due to lock-in. Same bootstrap as E3 (two-level, B = 10,000), also on the difference
μ_F7a − μ. F7a decides whether F4 runs. Sensitivity, reported beside it: **F7b** swaps every answer not in
the target format (locked-in or not); **F7c** (classification targets) maps non-canonical class labels to
their letter instead of swapping.

## Figure (one per backbone) [changed]
- Main paper: **A** μ (Table VI as now) | **B** lock-in rate (F1, with under-cardinality marked on ML
  targets). Then **C**, either μ_forced (F4) or μ after the F7 counterfactual if F4 doesn't run, on the same
  colour scale as A.
- Diagonal blank, report column greyed, detection column hatched.
- Appendix: the F2 scatter and the full format distributions.

## How we will read it
- High lock-in and μ_forced (or μ_F7) ≈ 0: negative transfer is mainly format lock-in, and joint training
  avoids it. Best case; this becomes a main finding.
- High lock-in but μ_forced still negative: format explains part of it, and content is also damaged.
- Low lock-in, low under-cardinality and few foreign labels: format isn't the cause. Report that, and keep
  the Discussion sentence as a rejected hypothesis.
- cls→ml driven by under-cardinality: report it as a separate mechanism (the adapter keeps the source's
  answer *shape*, a single label, even when the format is technically valid).

## What changed from the original and why
1. **Under-cardinality metric**: the overlap rule made the worst cell (cls→ml) read as 0% lock-in.
2. **Foreign-label format**: answers from the source's label vocabulary were being filed as unparseable
   instead of lock-in.
3. **Precedence order, plus two new tests**: makes "exactly one format" true by construction.
4. **F2 descriptive, no p-value**: lock-in and μ are mechanically linked, and there are only 15–20 pairs.
5. **F5 moved to the existing ZS-bare μ**: a matched comparison already exists, and adding a prompt to
   bare-trained adapters is a deviation.
6. **F7 before F4, with a gate on F4**: F4 needs new code in two harnesses plus a smoke test, with roughly
   a week of GPU access left.
7. **No forced detection**: base F1 is 0 and forced boxes from non-detection adapters would carry no
   information.
8. **"Fixed in advance", not "pre-registered"**: E3 had already been seen.
9. **Figure narrowed**: TMI space.
10. **After F0 validation, before any lock-in rate (22 Nov)**: free-text rule widened to short report
    sentences; "unparseable" renamed "other"; non-canonical class label and flat-box sub-flags; F7b/F7c
    sensitivity; gate pinned to 3 cells.

## Changes after the first lock-in run (22 Nov, logged deviations)
1. **Label fragments.** 27% of the Qwen classification adapter's answers to multi-label questions were
   "Endodontic Treatment", a comma-piece of the option "Endodontic Treatment, Restorative Treatment, and
   Complications". The committed rule (`656b644`) filed them as "other". The scorer splits labels on
   commas and credits the piece, so the rule now counts a fragment as that option (sub-flag `fragment`,
   reported per cell). Lock-in rates are unchanged by this. Before the change, cls→ml validity was
   70.0% (Qwen) and 74.3% (MedGemma); after it, 97.3% and 95.5%.
2. **F7c canonicalises the base as well.** The ZS-prompt base gives non-canonical class labels too (44.6%
   Qwen, 53.3% MedGemma), so mapping only the adapter's answers would be an unmatched comparison.

## F0 validation result (22 Nov, before any lock-in rate)
- Diagonal: every TaskLoRA's own answers are in its own format, 100% on both backbones, except 1 MedGemma
  regression answer (`A(105.92`, s42).
- Ambiguity: no real conflicts among 13,341 own-task and 66,705 off-task answers per backbone. Designed
  overlaps only (a label that is also prose, or also a source label).
