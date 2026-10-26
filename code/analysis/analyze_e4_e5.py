#!/usr/bin/env python3
"""Complete CPU E4/E5 report from finished GPU runs and existing E0/E1/E2 predictions.

Requires a fresh output directory. Full default: E4 vs all five Phase-A arms; E5 vs
Shared and TaskLoRA-cls; ZS-prompt difficulty correlations; 10k bootstrap resamples.
--arms can restrict E4 comparisons explicitly when an arm has not been evaluated yet.
No GPU inference is performed by this script.
"""
import argparse
from pathlib import Path

from group_results import (assemble, baseline_sources, collect, prediction_path,
                           read_json, read_jsonl)
from compare_groups import compare

SEEDS = (1337, 2026, 42)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', required=True)
    p.add_argument('--holdout', required=True)
    p.add_argument('--train', required=True)
    p.add_argument('--baseline-runs', default='/data/ugrip2026/journal/runs')
    p.add_argument('--out', required=True)
    p.add_argument('--arms', nargs='+', default=['shared', 'task', 'shared48', 'hydraT', 'hydraR'],
                   choices=['shared', 'task', 'shared48', 'hydraT', 'hydraR'])
    p.add_argument('--B', type=int, default=10000)
    a = p.parse_args()
    root = Path(a.out)
    if root.exists() and any(root.iterdir()):
        raise SystemExit('choose a fresh --out directory; previous analysis is preserved')
    arms = list(dict.fromkeys(a.arms + ['shared', 'task']))
    # Fail before assembly if an expected baseline artifact is absent.
    for backbone in ('q35', 'mg15'):
        for arm in arms + ['zsprompt']:
            for seed in ((1337,) if arm == 'zsprompt' else SEEDS):
                for source in baseline_sources(a.baseline_runs, arm, backbone, seed):
                    prediction_path(source)
    plan = read_json(a.plan)
    collect(plan, a.holdout, root / 'grouped')
    deck = read_jsonl(a.holdout)
    cls = [r for r in deck if r['task'] == 'classification']
    for backbone in ('q35', 'mg15'):
        zs = root / f'zsprompt_{backbone}_cls'
        assemble(baseline_sources(a.baseline_runs, 'zsprompt', backbone, 1337), cls,
                 zs, backbone, None, 'zsprompt', subset=True)
        for arm in arms:
            reference = []
            reference_cls = []
            for seed in SEEDS:
                dest = root / f'{arm}_{backbone}_s{seed}'
                assemble(baseline_sources(a.baseline_runs, arm, backbone, seed), deck,
                         dest, backbone, seed, arm, subset=True)
                reference.append(str(dest))
                if arm in ('shared', 'task'):
                    dest_cls = root / f'{arm}_{backbone}_cls_s{seed}'
                    assemble([dest], cls, dest_cls, backbone, seed, arm, subset=True)
                    reference_cls.append(str(dest_cls))
            if arm in a.arms:
                compare(reference, [str(root / f'grouped/E4_{backbone}_s{s}') for s in SEEDS],
                        a.train, root / f'E4_vs_{arm}_{backbone}', B=a.B)
            if arm in ('shared', 'task'):
                compare(reference_cls, [str(root / f'grouped/E5_{backbone}_s{s}') for s in SEEDS],
                        a.train, root / f'E5_vs_{arm}_{backbone}', B=a.B, zs=str(zs))
    print(f'ANALYSIS COMPLETE: {root}; selected E4 reference arms: {a.arms}')


if __name__ == '__main__':
    main()
