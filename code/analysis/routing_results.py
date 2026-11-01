#!/usr/bin/env python3
"""E6 routing accuracy and end-to-end metric drop, paired against each oracle arm."""
import argparse
from pathlib import Path
import numpy as np

import bootstrap as bs
from phase_b_common import (read_json, read_jsonl, write_json, fresh_dir, SEEDS, BACKBONES,
                            load_eval, summary, provenance)
from transfer_matrix import diagnostics
from predicted_routes import accuracy, load_routes


def analyze(index_path, out, B=10000, seed=20261013):
    if B < 1:
        raise ValueError('B must be positive')
    index = read_json(index_path)
    deck = read_jsonl(index['holdout'])
    result = {'study': 'E6', 'inputs': provenance([index_path, index['holdout']]), 'backbones': {},
              'definition': 'drop > 0 means predicted routing is worse than oracle; raw task metric units'}
    for backbone in BACKBONES:
        result['backbones'][backbone] = {}
        for arm in ('task', 'hydraT', 'hydraR', 'modality'):
            oracle, predicted, raw, routes = [], [], [], []
            axis = 'modality' if arm == 'modality' else 'task'
            for s in SEEDS:
                spec = index['backbones'][backbone][arm][str(s)]
                ro, _ = load_eval(spec['oracle'], deck, backbone, s, arm + '_oracle')
                rp, ps = load_eval(spec['predicted'], deck, backbone, s, arm + '_predicted')
                oracle.append(ro)
                predicted.append(rp)
                raw.append(ps)
                _, route_file = load_routes(spec['routes'], index['holdout'], deck, axis)
                routes.append(accuracy(deck, route_file['predictions'], axis))
            stats = {}
            for task in oracle[0]:
                st = bs.strata(oracle[0][task])
                op = [bs.macro(r[task], task, st) for r in oracle]
                pp = [bs.macro(r[task], task, st) for r in predicted]
                direction = 1 if bs.HIGHER_IS_BETTER[task] else -1
                rec = {'oracle': summary(op), 'predicted': summary(pp),
                       'drop': summary([direction * (o - p) for o, p in zip(op, pp)]),
                       'diagnostics': [diagnostics(r, task) for r in raw]}
                if np.isfinite(op + pp).all():
                    boot = bs.seeds([{task: r[task]} for r in oracle], [{task: r[task]} for r in predicted], B, seed)[task]
                    ci = sorted(-direction * x for x in boot['ci95'])
                    rec.update(drop_ci95=ci, p=boot['p'], B=B)
                else:
                    rec.update(drop_ci95=None, p=None, note='undefined metric; malformed predictions retained')
                stats[task] = rec
            for task, p in bs.holm({t: r['p'] for t, r in stats.items() if r['p'] is not None}).items():
                stats[task]['p_holm'] = p
            result['backbones'][backbone][arm] = {'routing': routes, 'tasks': stats}
    out = fresh_dir(out)
    write_json(out / 'routing.json', result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--B', type=int, default=10000)
    a = p.parse_args()
    analyze(a.index, a.out, a.B)
