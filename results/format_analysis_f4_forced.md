# F4 v2 forced-format decoding (holdout, B = 10000, code `03fd179`)

μ_forced = (adapter_forced − ZS_forced)/(perfect − ZS_forced), mean ± SD over 3 seeds; CI from the Table VIII two-level paired bootstrap; * Holm p < 0.05 across the forced targets of each source. Change = μ_forced − μ (same resamples). Cls: length-normalised option log-prob (summed in brackets).

## q35

| cell | μ (Table VIII) | μ_forced [95% CI] | Holm p | change [95% CI] | Holm p | trunc/incomplete/err |
|---|---:|---:|---:|---:|---:|---|
| det->reg | -1.279 ± 0.250 | +0.057 ± 0.057 [-0.010, +0.124] | 0.307 | +1.337 [+1.010, +1.645] | 0.0004* | 0/0/0 |
| det->count | -1.194 ± 0.268 | -0.158 ± 0.631 [-1.660, +0.314] | 0.889 | +1.036 [-0.340, +2.023] | 0.166 | 0/0/0 |
| det->ml | -0.401 ± 0.030 | -0.023 ± 0.004 [-0.057, +0.009] | 0.319 | +0.378 [+0.331, +0.430] | 0.0004* | 92/90/0 |
| cls->ml | -0.397 ± 0.046 | -0.130 ± 0.029 [-0.187, -0.080] | 0.0001* | +0.266 [+0.223, +0.315] | 0.0001* | 0/0/0 |
| report->ml | -0.128 ± 0.092 | -0.069 ± 0.008 [-0.111, -0.028] | 0.0013* | +0.059 [-0.003, +0.152] | 0.0977 | 4/3/0 |
| det->cls | -0.123 ± 0.019 | -0.089 ± 0.042 (-0.091) [-0.160, -0.025] | 0.0156* | +0.033 [-0.029, +0.099] | 0.282 | 0/0/0 |
| reg->ml | -0.076 ± 0.045 | -0.090 ± 0.048 [-0.158, -0.033] | 0.0007* | -0.015 [-0.026, -0.004] | 0.0063* | 4/4/0 |

Shared (reference), same targets:

| target | μ Shared unforced | μ Shared forced [95% CI] | macro unforced → forced | forced = unforced answers |
|---|---:|---:|---:|---:|
| cls | +0.701 ± 0.010 | +0.552 ± 0.013 (+0.550) [+0.511, +0.607] | 0.737 → 0.737 | 96.1% |
| ml | +0.218 ± 0.016 | +0.205 ± 0.017 [+0.152, +0.254] | 0.609 → 0.609 | 100.0% |
| count | +0.875 ± 0.013 | +0.875 ± 0.013 [+0.834, +0.909] | 12.963 → 12.963 | 100.0% |
| reg | +0.809 ± 0.012 | +0.809 ± 0.012 [+0.776, +0.841] | 11.349 → 11.349 | 99.7% |

ZS-prompt forced checks: {"cls": {"n": 1689, "errors": 0, "dead_end": 0, "mean_model_logprob_per_token": -0.9926138389286987, "sum_vs_mean_disagree": 14}, "ml": {"n": 703, "errors": 0, "dead_end": 0, "mean_model_logprob_per_token": -0.14163979288096107, "truncated": 19, "incomplete": 19, "mean_labels": 1.9971550497866288}, "count": {"n": 100, "errors": 0, "dead_end": 0, "mean_model_logprob_per_token": -0.9436206802803403, "truncated": 0, "incomplete": 0, "scorer_unparseable": 0}, "reg": {"n": 586, "errors": 0, "dead_end": 0, "mean_model_logprob_per_token": -0.9587011501555106, "truncated": 0, "incomplete": 0, "scorer_unparseable": 0}}

## mg15

| cell | μ (Table VIII) | μ_forced [95% CI] | Holm p | change [95% CI] | Holm p | trunc/incomplete/err |
|---|---:|---:|---:|---:|---:|---|
| det->reg | -1.419 ± 0.112 | -0.302 ± 0.097 [-0.441, -0.153] | 0.0002* | +1.117 [+0.910, +1.336] | 0.0002* | 0/0/0 |
| cls->reg | -0.440 ± 0.440 | -0.481 ± 0.440 [-0.953, -0.158] | 0.0001* | -0.041 [-0.139, +0.056] | 0.406 | 0/0/0 |
| report->ml | -0.275 ± 0.004 | -0.051 ± 0.024 [-0.085, -0.020] | 0.0042* | +0.224 [+0.181, +0.267] | 0.0002* | 521/460/0 |
| reg->ml | -0.260 ± 0.029 | +0.037 ± 0.024 [+0.015, +0.064] | 0.0001* | +0.298 [+0.247, +0.349] | 0.0001* | 850/676/0 |
| det->ml | -0.247 ± 0.055 | -0.035 ± 0.058 [-0.098, +0.021] | 0.254 | +0.212 [+0.098, +0.306] | 0.0002* | 735/586/0 |
| count->ml | -0.081 ± 0.146 | -0.038 ± 0.018 [-0.075, -0.005] | 0.0235* | +0.042 [-0.085, +0.184] | 0.581 | 244/189/0 |
| report->reg | -0.056 ± 0.019 | -0.082 ± 0.011 [-0.154, -0.012] | 0.0219* | -0.026 [-0.100, +0.049] | 0.482 | 0/0/0 |

Shared (reference), same targets:

| target | μ Shared unforced | μ Shared forced [95% CI] | macro unforced → forced | forced = unforced answers |
|---|---:|---:|---:|---:|
| ml | +0.448 ± 0.009 | +0.360 ± 0.010 [+0.318, +0.398] | 0.569 → 0.569 | 100.0% |
| reg | +0.765 ± 0.008 | +0.759 ± 0.008 [+0.715, +0.802] | 12.160 → 12.160 | 100.0% |

ZS-prompt forced checks: {"ml": {"n": 703, "errors": 0, "dead_end": 0, "mean_model_logprob_per_token": -0.13057496364782906, "truncated": 393, "incomplete": 361, "mean_labels": 10.58605974395448}, "reg": {"n": 586, "errors": 0, "dead_end": 0, "mean_model_logprob_per_token": -3.459212105981422, "truncated": 0, "incomplete": 0, "scorer_unparseable": 0}}

