#!/usr/bin/env python3
"""CPU E4/E5 contrasts, using the repository scorer and bootstrap implementation.

Three seeded assembled directories per arm, in the same seed order. Writes task-level
two-level bootstrap contrasts (Holm across tasks), per-dataset contrasts, and a CSV of
per-cell mean/SD and oriented benefit. Optional ZS-prompt predicts E5 benefit together
with training size using exploratory Spearman permutation tests over the ten datasets.
"""
import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import bootstrap as bs
from group_results import read_json, read_jsonl, validate_predictions


def subset_cell(run, task, dataset):
    t = run[task]
    idx = [i for i, d in enumerate(t['dataset']) if d == dataset]
    return {task: {k: (v[idx] if isinstance(v, np.ndarray) else [v[i] for i in idx])
                   for k, v in t.items()}}


def validate_runs(paths_a, paths_b):
    all_paths = list(paths_a) + list(paths_b)
    metas = [read_json(Path(d) / 'run_meta.json') for d in all_paths]
    if len(paths_a) != 3 or len(paths_b) != 3:
        raise ValueError('each arm needs all three seeds')
    if {m['seed'] for m in metas[:3]} != {1337, 2026, 42}:
        raise ValueError('expected seeds 1337, 2026, 42')
    if [m['seed'] for m in metas[:3]] != [m['seed'] for m in metas[3:]]:
        raise ValueError('seed order differs across arms')
    if len({m['arm'] for m in metas[:3]}) != 1 or len({m['arm'] for m in metas[3:]}) != 1:
        raise ValueError('each side must contain the same arm across seeds')
    if len({m['backbone'] for m in metas}) != 1 or any(m['surface'] != 'holdout' for m in metas):
        raise ValueError('comparisons require one backbone and holdout surface')
    rows = read_json(Path(paths_a[0]) / 'all_predictions.json')
    deck = [dict(id=r['id'], task=r['TaskType'], dataset=r['dataset'], answer=r['gt_answer'],
                 question=r['Question'], images=r['ImageName'], modality=r['modality']) for r in rows]
    for d in all_paths:
        validate_predictions(read_json(Path(d) / 'all_predictions.json'), deck)
    runs = [bs.load_run(d) for d in all_paths]
    for run in runs:
        for task, t in run.items():
            if not bs.usable(t, task):
                raise ValueError('CRIMSON scores required before comparison')
            point = bs.macro(t, task, bs.strata(t))
            if not math.isfinite(point):
                raise ValueError(f'{task}: non-finite metric (e.g. malformed numeric output); inspect raw results')
    return runs[:3], runs[3:], metas


def correlation(x, y, B, seed):
    from scipy.stats import spearmanr
    if len(set(x)) < 2 or len(set(y)) < 2:
        return {'rho': None, 'p_permutation': None, 'note': 'constant predictor/outcome'}
    rho = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    hits = sum(abs(spearmanr(x, rng.permutation(y)).statistic) >= abs(rho) - 1e-12 for _ in range(B))
    return {'rho': rho, 'p_permutation': (hits + 1) / (B + 1), 'n': len(x), 'B': B}


def arm_summary(runs, B, seed):
    """Two-level seed/question CI for each arm, using the existing dataset macro."""
    rng = np.random.default_rng(seed)
    result = {}
    for task in sorted(runs[0]):
        ts = [r[task] for r in runs]
        strata = bs.strata(ts[0])
        points = [bs.macro(t, task, strata) for t in ts]
        draws = []
        for _ in range(B):
            chosen = rng.integers(0, len(ts), len(ts))
            sample = bs.resample(strata, rng)
            draws.append(float(np.mean([bs.macro(ts[i], task, sample) for i in chosen])))
        result[task] = {'mean': float(np.mean(points)), 'sd': float(np.std(points, ddof=1)),
                        'ci95': [float(x) for x in np.quantile(draws, [0.025, 0.975])],
                        'n': len(ts[0]['ids']), 'B': B}
    return result


def compare(paths_a, paths_b, train, out, B=10000, seed=20261013, zs=None):
    if B <= 0:
        raise ValueError('B must be positive')
    ra, rb, metas = validate_runs(paths_a, paths_b)
    result = {'surface': 'holdout', 'backbone': metas[0]['backbone'], 'a': list(paths_a),
              'b': list(paths_b), 'B': B, 'random_seed': seed,
              'per_task': bs.seeds(ra, rb, B, seed), 'per_cell': {},
              'arm_a': arm_summary(ra, B, seed), 'arm_b': arm_summary(rb, B, seed)}
    train_rows = read_jsonl(train)
    counts = Counter((r['task'], r['dataset']) for r in train_rows)
    modalities = {(r['task'], r['dataset']): r['modality'] for r in train_rows}
    records = []
    zs_run = bs.load_run(zs) if zs else None
    if zs:
        zm = read_json(Path(zs) / 'run_meta.json')
        if zm['backbone'] != metas[0]['backbone'] or zm['surface'] != 'holdout' or zm['arm'] != 'zsprompt':
            raise ValueError('zero-shot difficulty must use same-backbone ZS-prompt holdout')
    for task, t in sorted(ra[0].items()):
        for ds in sorted(set(t['dataset'])):
            a = [subset_cell(r, task, ds) for r in ra]
            b = [subset_cell(r, task, ds) for r in rb]
            stats = bs.seeds(a, b, B, seed)[task]
            stats.pop('p_holm', None)  # Apply Holm across all cells below, not separately per cell.
            direction = 1 if bs.HIGHER_IS_BETTER[task] else -1
            stats['benefit_b_minus_a'] = direction * stats['diff_b_minus_a']
            result['per_cell'][f'{task}/{ds}'] = stats
            row = {'task': task, 'dataset': ds, 'modality': modalities.get((task, ds)),
                   'train_n': counts[(task, ds)],
                   'mean_a': stats['mean_a'], 'sd_a': stats['sd_a'], 'mean_b': stats['mean_b'],
                   'sd_b': stats['sd_b'], 'benefit_b_minus_a': stats['benefit_b_minus_a']}
            if not row['train_n']:
                raise ValueError(f'no training rows for {task}/{ds}')
            if zs_run:
                z = subset_cell(zs_run, task, ds)[task]
                bs.align(a[0][task], z)
                if z['gt'] != a[0][task]['gt']:
                    raise ValueError('zero-shot ground truth differs')
                row['zs_score'] = bs.macro(z, task, bs.strata(z))
                row['zs_difficulty'] = -row['zs_score'] if direction > 0 else row['zs_score']
            records.append(row)
    for key, p in bs.holm({k: v['p'] for k, v in result['per_cell'].items()}).items():
        result['per_cell'][key]['p_holm_cells'] = p
    if set(ra[0]) == {'classification'}:
        result['e5_exploratory_correlations'] = {
            predictor: correlation([r[predictor] for r in records],
                                   [r['benefit_b_minus_a'] for r in records], B, seed)
            for predictor in (['train_n', 'zs_difficulty'] if zs else ['train_n'])}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'contrasts.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    with open(out / 'cells.csv', 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--a', nargs=3, required=True, help='reference arm: 1337, 2026, 42')
    p.add_argument('--b', nargs=3, required=True, help='specialized arm: 1337, 2026, 42')
    p.add_argument('--train', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--zs', help='assembled ZS-prompt directory; E5 difficulty predictor')
    p.add_argument('--B', type=int, default=10000)
    p.add_argument('--seed', type=int, default=20261013)
    a = p.parse_args()
    compare(a.a, a.b, a.train, a.out, a.B, a.seed, a.zs)
    print(f'Contrasts written to {a.out}')


if __name__ == '__main__':
    main()
