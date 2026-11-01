#!/usr/bin/env python3
"""E8: 19-cell RQ4 synthesis, per backbone, without an underpowered joint regression.

Primary benefit is the PGF difference specialized minus Shared (PLAN 1.8); raw oriented
metric differences are retained. Recompute using ZS-bare. Each categorical predictor is
explicitly one-hot encoded, never treated as an ordinal code. No undefined cell is dropped.
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np

import bootstrap as bs
from compare_groups import subset_cell
from group_results import assemble, baseline_sources
from phase_b_common import (SEEDS, BACKBONES, STRUCTURE, read_json, read_jsonl, write_json, fresh_dir,
                            provenance, load_eval, summary, rank_correlation, finite, require_input_hash)
from transfer_matrix import oriented_difference, pgf

ARMS = ('task', 'hydraT', 'hydraR')


def prepare(runs, holdout, out, score=True):
    deck = read_jsonl(holdout)
    out = fresh_dir(out)
    index = {'study': 'E8', 'holdout': str(Path(holdout).resolve()), 'backbones': {}}
    for backbone in BACKBONES:
        spec = {}
        for arm in ('shared', *ARMS, 'zsprompt', 'zsbare'):
            if arm.startswith('zs'):
                dest = out / f'{backbone}_{arm}'
                assemble(baseline_sources(runs, arm, backbone, None), deck, dest, backbone, None, arm, score=score)
                spec[arm] = str(dest.resolve())
            else:
                spec[arm] = {}
                for seed in SEEDS:
                    dest = out / f'{backbone}_{arm}_s{seed}'
                    assemble(baseline_sources(runs, arm, backbone, seed), deck, dest, backbone, seed, arm, score=score)
                    spec[arm][str(seed)] = str(dest.resolve())
        index['backbones'][backbone] = spec
    write_json(out / 'index.json', index)
    return index


def predictors(train, cells, transfer, representations, anchor='zsprompt'):
    cell_n = Counter((r['task'], r['dataset']) for r in train)
    task_n = Counter(r['task'] for r in train)
    datasets = {t: {r['dataset'] for r in train if r['task'] == t} for t, _ in cells}
    modalities = {(t, d): {r['modality'] for r in train if (r['task'], r['dataset']) == (t, d)} for t, d in cells}
    if any(len(m) != 1 for m in modalities.values()):
        raise ValueError('each task/dataset cell must have exactly one modality')
    all_mods = sorted(set.union(*modalities.values()))
    result = {}
    for task, ds in cells:
        if cell_n[(task, ds)] == 0:
            raise ValueError(f'no training rows for {task}/{ds}')
        record = {'train_cell_n': cell_n[(task, ds)], 'train_task_n': task_n[task],
                  'task_dataset_n': len(datasets[task])}
        record.update({'output_' + x: int(STRUCTURE[task] == x) for x in sorted(set(STRUCTURE.values()))})
        record.update({'modality_' + x: int(x in modalities[(task, ds)]) for x in all_mods})
        agg = transfer['aggregates_bare' if anchor == 'zsbare' else 'aggregates']['off_diagonal'][task]
        for kind in ('transferability', 'malleability'):
            for sign in ('positive', 'negative'):
                record[kind + '_' + sign] = agg[kind][sign]['mean']
        for layer_pool, values in representations['cell_predictors'][f'{task}/{ds}'].items():
            for kind, value in values.items():
                record[f'{kind}_{layer_pool}'] = value
        result[f'{task}/{ds}'] = record
    return result


def pgf_benefit(a, b, base, task):
    pa, pb = pgf(a, base, task), pgf(b, base, task)
    return pb - pa if pa is not None and pb is not None else None


def analyze(index_path, train_path, transfer_path, representations_path, out, B=10000, seed=20261013):
    if B < 1:
        raise ValueError('B must be positive')
    index = read_json(index_path)
    holdout, train = read_jsonl(index['holdout']), read_jsonl(train_path)
    cells = sorted({(r['task'], r['dataset']) for r in holdout})
    if len(cells) != 19:
        raise ValueError(f'E8 reporting requires all 19 cells, found {len(cells)}')
    transfer, representations = read_json(transfer_path), read_json(representations_path)
    require_input_hash(transfer, index['holdout'])
    require_input_hash(transfer, train_path)
    require_input_hash(representations, index['holdout'])
    result = {'study': 'E8', 'n_cells': 19, 'B': B, 'random_seed': seed,
              'inputs': provenance([index_path, train_path, transfer_path, representations_path, index['holdout']]),
              'primary_outcome': 'mean-seed PGF(specialized)-PGF(Shared), ZS-prompt anchor',
              'sensitivity': 'same tests using ZS-bare; raw metric differences retained per cell',
              'categorical_encoding': 'one-hot indicators', 'permutation_unit': 'cell',
              'backbones': {}, 'cross_backbone_signs': {}}
    all_csv = []
    for backbone in BACKBONES:
        spec = index['backbones'][backbone]
        runs = {arm: [load_eval(spec[arm][str(s)], holdout, backbone, s, arm)[0] for s in SEEDS]
                for arm in ('shared', *ARMS)}
        anchors = {arm: load_eval(spec[arm], holdout, backbone, arm=arm)[0] for arm in ('zsprompt', 'zsbare')}
        output = {'outcomes': {}}
        for arm in ARMS:
            for anchor, base in anchors.items():
                covariates = predictors(train, cells, transfer['backbones'][backbone], representations['backbones'][backbone], anchor)
                cell_results = {}
                for task, ds in cells:
                    key = f'{task}/{ds}'
                    aa, bb = ([subset_cell(r, task, ds) for r in runs[a]] for a in ('shared', arm))
                    bt = subset_cell(base, task, ds)[task]
                    st = bs.strata(bt)
                    av = [bs.macro(r[task], task, st) for r in aa]
                    bv = [bs.macro(r[task], task, st) for r in bb]
                    zv = bs.macro(bt, task, st)
                    raw = [finite(oriented_difference(b, a, task)) for a, b in zip(av, bv)]
                    norm = [pgf_benefit(a, b, zv, task) for a, b in zip(av, bv)]
                    pred = dict(covariates[key], zs_difficulty=finite(1-zv if bs.HIGHER_IS_BETTER[task] else zv))
                    record = {'predictors': pred, 'raw_benefit': summary(raw), 'PGF_benefit': summary(norm)}
                    cell_results[key] = record
                    all_csv.append({'backbone': backbone, 'arm': arm, 'anchor': anchor, 'task': task, 'dataset': ds,
                                    'benefit': record['PGF_benefit']['mean'], 'raw_benefit': record['raw_benefit']['mean'], **pred})
                tests = {}
                y = [r['PGF_benefit']['mean'] for r in cell_results.values()]
                for name in next(iter(cell_results.values()))['predictors']:
                    x = [r['predictors'][name] for r in cell_results.values()]
                    if any(v is None or not np.isfinite(v) for v in x + y):
                        tests[name] = {'rho': None, 'p_permutation': None,
                                       'reason': 'at least one undefined cell; all 19 cells required'}
                    else:
                        tests[name] = rank_correlation(x, y, B, seed)
                for name, p in bs.holm({k: v['p_permutation'] for k, v in tests.items() if v['p_permutation'] is not None}).items():
                    tests[name]['p_holm_predictors'] = p
                output['outcomes'][arm + '/' + anchor] = {'cells': cell_results, 'tests': tests}
        result['backbones'][backbone] = output
    for outcome, q in result['backbones']['q35']['outcomes'].items():
        m = result['backbones']['mg15']['outcomes'][outcome]
        # Backbones have different depths; only identical predictor names are directly paired.
        result['cross_backbone_signs'][outcome] = {}
        for name in sorted(set(q['tests']) | set(m['tests'])):
            a, b = q['tests'].get(name, {}).get('rho'), m['tests'].get(name, {}).get('rho')
            result['cross_backbone_signs'][outcome][name] = {'q35': a, 'mg15': b,
                'same_nonzero_sign': bool(a * b > 0) if a is not None and b is not None else None}
    out = fresh_dir(out)
    write_json(out / 'synthesis.json', result)
    with open(out / 'cells.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=sorted(set().union(*(r.keys() for r in all_csv))))
        writer.writeheader()
        writer.writerows(all_csv)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='cmd', required=True)
    prep = sub.add_parser('prepare')
    for name in ('runs', 'holdout', 'out'):
        prep.add_argument('--' + name, required=True)
    analysis = sub.add_parser('analyze')
    for name in ('index', 'train', 'transfer', 'representations', 'out'):
        analysis.add_argument('--' + name, required=True)
    analysis.add_argument('--B', type=int, default=10000)
    a = p.parse_args()
    if a.cmd == 'prepare':
        prepare(a.runs, a.holdout, a.out)
    else:
        analyze(a.index, a.train, a.transfer, a.representations, a.out, a.B)
