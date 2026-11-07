#!/usr/bin/env python3
"""E3: six-by-six transfer, task/cell PGF, diagnostics and RQ3 covariates.

Input is the index produced by plan_e3.py collect. Reuses E1/E2 diagonal predictions.
Raw oriented C is primary; PGF is ancillary. A perfect zero-shot baseline makes PGF
undefined (null), never an epsilon-amplified claim. Sachdeva Eqs 1/2 use the off-diagonal
N=5 as primary, with N=6 including-diagonal sensitivity explicitly reported.
"""
import argparse
import itertools
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, wilcoxon

import bootstrap as bs
from compare_groups import subset_cell
from phase_b_common import (TASKS, SHORT, STRUCTURE, STRUCTURE_FEATURES, SEEDS, BACKBONES, read_json, read_jsonl,
                            write_json, fresh_dir, provenance, commit, finite, summary, load_eval, require_input_hash)


def oriented_difference(score, base, task):
    return (score - base) * (1 if bs.HIGHER_IS_BETTER[task] else -1)


def pgf(score, base, task):
    gap = 1 - base if bs.HIGHER_IS_BETTER[task] else base
    return finite(oriented_difference(score, base, task) / gap) if gap > 0 else None


def signed_aggregate(values):
    """(1-exp(-p/N))/p * sum(signed PGF); empty sign class is zero."""
    values = list(values)
    if not values or any(v is None or not np.isfinite(v) for v in values):
        return {'positive': None, 'negative': None}
    n = len(values)
    out = {}
    for key, selected in [('positive', [v for v in values if v > 0]),
                           ('negative', [v for v in values if v < 0])]:
        p = len(selected)
        out[key] = float(-np.expm1(-p / n) * np.mean(selected)) if p else 0.0
    return out


def diagnostics(rows, task):
    from score_single import is_malformed
    ps = [p for p in rows if p['TaskType'] == task]
    result = {'n': len(ps), 'truncation_rate': sum(p['truncated'] for p in ps) / len(ps),
              'malformed_rate': sum(not p.get('raw_output', '').strip() if task == bs.REPORT
                                    else is_malformed(task, p['Answer']) for p in ps) / len(ps)}
    # Omission is already recorded by score_run for every assembled report arm.
    if task == bs.REPORT:
        result['report_note'] = 'Non-report source outputs remain scored, including empty or unrelated reports.'
    return result


def seed_stability(values):
    if any(v is None for v in values):
        return {'p_wilcoxon': None, 'reason': 'undefined metric'}
    if all(v == 0 for v in values):
        p = 1.0
    else:
        p = float(wilcoxon(values, alternative='two-sided', zero_method='wilcox', method='auto').pvalue)
    return {'p_wilcoxon': p, 'same_nonzero_sign': all(v > 0 for v in values) or all(v < 0 for v in values),
            'note': 'Three seeds: a two-sided exact test cannot reach p < .05.'}


def pair_covariates(train):
    grouped = {t: [r for r in train if r['task'] == t] for t in TASKS}
    if any(not rows for rows in grouped.values()):
        raise ValueError('training deck must include all six tasks')
    def jaccard(a, b):
        return len(a & b) / len(a | b)
    result = {}
    for i, ti in enumerate(TASKS):
        for j, tj in enumerate(TASKS):
            if i == j:
                continue
            a, b = grouped[ti], grouped[tj]
            result[(i, j)] = dict(source_n=len(a), target_n=len(b),
                output_structure_similarity=jaccard(set(STRUCTURE_FEATURES[ti]),
                                                    set(STRUCTURE_FEATURES[tj])),
                modality_jaccard=jaccard({r['modality'] for r in a}, {r['modality'] for r in b}),
                dataset_jaccard=jaccard({r['dataset'] for r in a}, {r['dataset'] for r in b}))
    return result


def covariate_tests(matrix, covariates):
    """Exact 6! node-label permutations preserve dependence among the 30 task pairs."""
    pairs = list(covariates)
    y = np.array([matrix[i, j] for i, j in pairs])
    result = {}
    for name in next(iter(covariates.values())):
        x = np.array([covariates[p][name] for p in pairs], float)
        if not np.isfinite(y).all() or np.ptp(x) == 0 or np.ptp(y) == 0:
            result[name] = {'rho': None, 'p_permutation': None,
                            'reason': 'constant covariate/outcome or undefined metric'}
            continue
        xr, yr = rankdata(x), rankdata(y)
        xr, yr = xr - xr.mean(), yr - yr.mean()
        den = np.linalg.norm(xr) * np.linalg.norm(yr)
        rho = float(xr @ yr / den)
        hits = 0
        for perm in itertools.permutations(range(6)):
            yp = rankdata([matrix[perm[i], perm[j]] for i, j in pairs])
            hits += abs(xr @ (yp - yp.mean()) / den) >= abs(rho) - 1e-12
        result[name] = {'rho': rho, 'p_permutation': hits / 720, 'n_pairs': 30,
                        'permutations': 720, 'unit': 'task labels permuted jointly on both axes'}
    adjusted = bs.holm({k: v['p_permutation'] for k, v in result.items() if v['p_permutation'] is not None})
    for key, p in adjusted.items():
        result[key]['p_holm'] = p
    return result


def transfer_ci(specialists, base, task, B, seed):
    """Three trained seeds, one fixed base; paired dataset-stratified question resampling."""
    rng = np.random.default_rng(seed)
    st = bs.strata(base)
    for t in specialists:
        bs.align(t, base)
    if not np.isfinite([bs.macro(t, task, st) for t in [base, *specialists]]).all():
        return {'ci95': None, 'B': B, 'reason': 'nonfinite full-sample metric; inspect malformed outputs'}
    draws = []
    for _ in range(B):
        sample = bs.resample(st, rng)
        chosen = rng.integers(0, len(specialists), len(specialists))
        score = np.mean([bs.macro(specialists[i], task, sample) for i in chosen])
        draws.append(oriented_difference(score, bs.macro(base, task, sample), task))
    if not np.isfinite(draws).all():
        return {'ci95': None, 'B': B, 'reason': 'nonfinite resampled metric; inspect malformed outputs'}
    point = oriented_difference(np.mean([bs.macro(t, task, st) for t in specialists]), bs.macro(base, task, st), task)
    return {'ci95': [float(x) for x in np.quantile(draws, [.025, .975])],
            'p': bs.p_two_sided(np.asarray(draws), point), 'B': B}


def analyze(index_path, train_path, holdout_path, out_path, B=10000, random_seed=20261013, plots=True):
    if B < 1:
        raise ValueError('B must be positive')
    index, train, holdout = read_json(index_path), read_jsonl(train_path), read_jsonl(holdout_path)
    require_input_hash(index, train_path)
    require_input_hash(index, holdout_path)
    out = fresh_dir(out_path)
    covariates = pair_covariates(train)
    result = {'study': 'E3', 'surface': 'holdout', 'seeds': list(SEEDS), 'B': B,
              'random_seed': random_seed, 'code_commit': commit(),
              'inputs': provenance([index_path, train_path, holdout_path]), 'backbones': {},
              'definitions': {'C': 'metric-oriented difference from ZS-prompt',
                  'PGF': 'PLAN 1.8; null if baseline gap is zero or metric nonfinite',
                  'aggregate': 'Sachdeva Eq 1/2; primary excludes diagonal, N=5',
                  'output_structure': STRUCTURE,
                  'output_structure_features': STRUCTURE_FEATURES,
                  'structure_note': 'Graded Jaccard overlap of pre-result answer-format features.'}}
    for backbone in BACKBONES:
        spec = index['backbones'][backbone]
        base, _ = load_eval(spec['zsprompt'], holdout, backbone, arm='zsprompt')
        bare, _ = load_eval(spec['zsbare'], holdout, backbone, arm='zsbare')
        rows = {}
        raw = {}
        for source in TASKS:
            rows[source], raw[source] = [], []
            for seed in SEEDS:
                run, predictions = load_eval(spec['rows'][SHORT[source]][str(seed)], holdout,
                                              backbone, seed, 'transfer_' + SHORT[source])
                rows[source].append(run)
                raw[source].append(predictions)
        matrices = np.full((3, 6, 6), np.nan)
        mus = np.full_like(matrices, np.nan)
        mus_bare = np.full_like(matrices, np.nan)
        output = {'pairs': {}, 'cells': {}, 'sign_flips_vs_zsbare': []}
        for i, source in enumerate(TASKS):
            for j, target in enumerate(TASKS):
                st = bs.strata(base[target])
                baseline, baseline_bare = bs.macro(base[target], target, st), bs.macro(bare[target], target, st)
                scores = [bs.macro(r[target], target, st) for r in rows[source]]
                c = [finite(oriented_difference(v, baseline, target)) for v in scores]
                mu = [pgf(v, baseline, target) for v in scores]
                cb = [finite(oriented_difference(v, baseline_bare, target)) for v in scores]
                matrices[:, i, j] = [np.nan if v is None else v for v in c]
                mus[:, i, j] = [np.nan if v is None else v for v in mu]
                mus_bare[:, i, j] = [np.nan if (v := pgf(score, baseline_bare, target)) is None else v for score in scores]
                key = f'{SHORT[source]}->{SHORT[target]}'
                output['pairs'][key] = {'source': source, 'target': target, 'M': summary(scores),
                    'base': finite(baseline), 'base_bare': finite(baseline_bare), 'C': summary(c),
                    'C_bare': summary(cb), 'PGF': summary(mu),
                    'PGF_bare': summary([pgf(v, baseline_bare, target) for v in scores]),
                    'diagnostics': [diagnostics(p, target) for p in raw[source]],
                    'stability': seed_stability(c),
                    'bootstrap': transfer_ci([r[target] for r in rows[source]], base[target], target, B, random_seed)}
                for s, cp, cbp in zip(SEEDS, c, cb):
                    if cp is not None and cbp is not None and cp * cbp < 0:
                        output['sign_flips_vs_zsbare'].append({'pair': key, 'seed': s})
                for dataset in sorted(st):
                    br = subset_cell(base, target, dataset)[target]
                    bbr = subset_cell(bare, target, dataset)[target]
                    ss = bs.strata(br)
                    bv = bs.macro(br, target, ss)
                    bbv = bs.macro(bbr, target, ss)
                    ts = [subset_cell(r, target, dataset)[target] for r in rows[source]]
                    values = [bs.macro(t, target, ss) for t in ts]
                    output['cells'][f'{SHORT[source]}->{target}/{dataset}'] = {
                        'source': source, 'target': target, 'dataset': dataset,
                        'C': summary([finite(oriented_difference(v, bv, target)) for v in values]),
                        'PGF': summary([pgf(v, bv, target) for v in values]),
                        'PGF_bare': summary([pgf(v, bbv, target) for v in values]),
                        'bootstrap': transfer_ci(ts, br, target, B, random_seed)}
        for aggregate_key, values in [('aggregates', mus), ('aggregates_bare', mus_bare)]:
            output[aggregate_key] = {}
            for included in (False, True):
                label = 'including_diagonal' if included else 'off_diagonal'
                output[aggregate_key][label] = {}
                for i, task in enumerate(TASKS):
                    idx = [j for j in range(6) if included or j != i]
                    record = {}
                    for name, series in [('transferability', values[:, i, idx]), ('malleability', values[:, idx, i])]:
                        aggregates = [signed_aggregate(v) for v in series]
                        record[name] = {sign: summary([a[sign] for a in aggregates]) for sign in ('positive', 'negative')}
                    output[aggregate_key][label][task] = record
        for source in TASKS:
            family = {key: pair['bootstrap']['p'] for key, pair in output['pairs'].items()
                      if pair['source'] == source and 'p' in pair['bootstrap']}
            for key, p in bs.holm(family).items():
                output['pairs'][key]['bootstrap']['p_holm_targets'] = p
        output['covariates'] = [{'source': TASKS[i], 'target': TASKS[j], **v} for (i, j), v in covariates.items()]
        output['covariate_tests'] = covariate_tests(matrices.mean(axis=0), covariates)
        output['iugc_natural_experiment'] = {k: v for k, v in output['cells'].items()
            if v['dataset'] == 'iugc' and v['source'] in ('classification', 'detection', 'regression')
            and v['target'] != v['source']}
        result['backbones'][backbone] = output
        if plots:
            heatmap(matrices.mean(axis=0), out / f'{backbone}_transfer.pdf', backbone)
    write_json(out / 'transfer.json', result)
    return result


def heatmap(matrix, path, backbone):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 6))
    limit = max(float(np.nanmax(np.abs(matrix))) if np.isfinite(matrix).any() else 1, 1e-8)
    im = ax.imshow(matrix, cmap='coolwarm', vmin=-limit, vmax=limit)
    labels = [SHORT[t] for t in TASKS]
    ax.set(xticks=range(6), yticks=range(6), xticklabels=labels, yticklabels=labels,
           xlabel='Target task', ylabel='Source adapter', title=f'{backbone}: oriented transfer C (3-seed mean)')
    for i, j in itertools.product(range(6), repeat=2):
        ax.text(j, i, f'{matrix[i,j]:.3g}' if np.isfinite(matrix[i,j]) else 'undefined', ha='center', va='center')
    fig.colorbar(im, ax=ax, label='Raw metric units differ by target column; compare within columns')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index', required=True)
    p.add_argument('--train', required=True)
    p.add_argument('--holdout', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--B', type=int, default=10000)
    p.add_argument('--random-seed', type=int, default=20261013)
    p.add_argument('--no-plots', action='store_true')
    a = p.parse_args()
    analyze(a.index, a.train, a.holdout, a.out, a.B, a.random_seed, not a.no_plots)


if __name__ == '__main__':
    main()
