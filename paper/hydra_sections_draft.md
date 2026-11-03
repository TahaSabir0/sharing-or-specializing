# Hydra sections for the v2 manuscript (Veronika)

Integration draft for Taha; not yet inserted into the v1 manuscript. Full-run
results and E7 interpretation remain pending. The paragraphs below describe
the implemented study arms, rather than the historical two-stage Hydra workflow.

## Methods: oracle-routed partial sharing

We compare two forms of partial parameter sharing between six task families.
For a frozen linear map \(W_0\), a rank \(r\) adapter, and a task label \(t\),
Hydra-T uses a shared input factor \(A\) and task-specific output factors \(B_t\):

\[
y=W_0x+\frac{\alpha}{r}B_t A\,\mathrm{Dropout}(x).
\]

Reversed Hydra shares the output factor \(B\) and specializes the input factors
\(A_t\):

\[
y=W_0x+\frac{\alpha}{r}B A_t\,\mathrm{Dropout}(x).
\]

The known task label selects exactly one of six heads for each example, in both
training and inference. No learned router or auxiliary balancing loss is used.
All adapter factors train jointly on the natural mixture of task examples. The
input factors use Kaiming-uniform initialization and the output factors are
initialized to zero, so the initial adapted model equals the frozen backbone.
The base model, vision encoder and multimodal projector remain frozen.

Both variants use rank 8, scaling coefficient alpha 16 and adapter dropout 0.05.
They wrap the same frozen language-model target set as the other study arms:
248 linear modules on Qwen3.5-9B and 238 on MedGemma-1.5-4B. They inherit the
common per-example loss, optimizer, natural-frequency sampling, training schedule
and decoding protocol. Seeds are 1337, 2026 and 42; results must be reported
across seeds after completed training and holdout evaluation.

| Backbone | Arm | Stored/trainable parameters | Active per example |
|---|---|---:|---:|
| Qwen3.5-9B | Hydra-T | 78,716,928 | 21,639,168 |
| Qwen3.5-9B | Reversed Hydra | 72,757,248 | 21,639,168 |
| MedGemma-1.5-4B | Hydra-T | 55,287,808 | 14,901,248 |
| MedGemma-1.5-4B | Reversed Hydra | 49,020,928 | 14,901,248 |

Counts follow from the frozen target shapes; Qwen stored counts were also checked
in the completed smoke tests. MedGemma Hydra requires its own real-model smoke.
The two partial-sharing arms have the same active parameter budget within each
backbone, while their stored budgets differ because input/output widths differ.

## Methods: E7 head similarity

For every adapted target matrix, we flatten each task-specific factor and compute
pairwise cosine similarities between tasks: \(B_t\) for Hydra-T and \(A_t\) for
Reversed Hydra. We retain the frozen task order and report the six-by-six matrices
and factor norms separately for each target and seed. Cosines involving a
zero-norm factor are undefined and are recorded as missing values. An optional
decoder-block summary concatenates the selected factors in a fixed target order;
this is a norm-weighted summary, not an unweighted average of target cosines.

Raw factor similarities depend on initialization and parameter coordinates.
In particular, the independently initialized input heads in Reversed Hydra can
already differ at step zero. We therefore treat these quantities as descriptive
and do not interpret low cosine alone as learned functional specialization or
evidence of task transfer. Comparisons across seeds and with downstream analyses
must retain this limitation.

## Development validation (exclude from main task-performance results)

Both Qwen arms completed a 50-step training smoke and 10 dev-select questions per
task on 3 November 2026. Hydra-T loss decreased from 1.534 to 0.641 and Reversed
Hydra loss from 1.611 to 0.630. Both produced 60 predictions without malformed
outputs, truncations or evaluation errors, with valid routing statistics.
The dev-select examples are training-seen and these checks establish operation,
not comparative medical task quality. Full three-seed holdout results, confidence
intervals and trained-head E7 findings have not yet been obtained for these arms.

## Results to complete after Taha's runs

- E1/E2: Hydra-T versus Reversed Hydra and each versus Shared/TaskLoRA, using
  the planned paired contrasts, per-dataset task macros, seed variability and CIs.
- Report omission, malformed and truncation rates from completed full evals.
- E7: per-target/per-layer head matrices and descriptive cross-seed patterns.
- Retain failed development runs in the engineering record; do not report their
  smoke scores as paper results or infer a method ranking from them.
