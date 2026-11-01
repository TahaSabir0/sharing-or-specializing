#!/usr/bin/env python3
"""Prepare the 24 E6 predicted-route evaluations (4 arms x 2 backbones x 3 seeds)."""
import argparse
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / 'analysis'))
sys.path.insert(0, str(CODE / 'data'))
from phase_b_common import (TASKS, SHORT, SEEDS, BACKBONES, read_json, read_jsonl,
                            write_json, fresh_dir, provenance, digest, commit)
from predicted_routes import LABELS, load_routes
from grouping import MODALITY_SLUGS
from phase_b_jobs import check_plan, evaluated
from group_results import assemble, baseline_sources


def prepare(data, groups, runs, routes, outputs, out):
    data, groups, runs, routes, outputs, out = [Path(p).resolve() for p in (data, groups, runs, routes, outputs, out)]
    holdout = data / 'j26_holdout.jsonl'
    rows = read_jsonl(holdout)
    paths = [holdout, data / 'j26_train_mix.jsonl']
    plan = {'schema': 1, 'study': 'E6', 'holdout': str(holdout), 'runs': str(runs),
            'code_commit': commit(), 'jobs': []}
    out = fresh_dir(out)
    for backbone in BACKBONES:
        res = 768 if backbone == 'q35' else 896
        for si, seed in enumerate(SEEDS):
            for arm in ('task', 'hydraT', 'hydraR', 'modality'):
                axis = 'modality' if arm == 'modality' else 'task'
                route_file = routes / f'{axis}_s{seed}.json'
                _, route_meta = load_routes(route_file, holdout, rows, axis)
                if route_meta['seed'] != seed:
                    raise ValueError('router seed differs from adapter seed')
                paths.append(route_file)
                labels = list(LABELS[axis])
                adapters, mapping = [], {}
                if arm == 'task':
                    tags = {t: 'task' + SHORT[t] for t in labels}
                    decks = {t: data / f'j26_train_{SHORT[t]}.jsonl' for t in labels}
                elif arm == 'modality':
                    tags = {t: 'mod' + MODALITY_SLUGS[t] for t in labels}
                    decks = {t: groups / f'j26_train_mod{MODALITY_SLUGS[t]}.jsonl' for t in labels}
                else:
                    tags = {arm: arm}
                    decks = {arm: data / 'j26_train_mix.jsonl'}
                for label, tag in tags.items():
                    path = runs / f'j26_{backbone}_{tag}_full_{res}_s{seed}'
                    mapping[label] = str(path)
                    paths.append(decks[label])
                    adapters.append({'path': str(path), 'backbone': backbone, 'seed': seed,
                                     'plugin': arm in ('hydraT', 'hydraR'),
                                     'train_deck': decks[label].stem, 'train_md5': digest(decks[label])})
                key = f'e6_{backbone}_{arm}_s{seed}'
                dest = outputs / key
                mode = 'plugin' if arm in ('hydraT', 'hydraR') else 'routed'
                args = ['--mode', mode, '--backbone', backbone, '--data', str(holdout), '--surface', 'holdout',
                        '--batch_size', '1', '--routes', str(route_file), '--route_axis', axis, '--out_dir', str(dest)]
                if mode == 'plugin':
                    args += ['--run_dir', adapters[0]['path']]
                else:
                    mapping_path = out / f'{key}_adapters.json'
                    write_json(mapping_path, mapping)
                    paths.append(mapping_path)
                    args += ['--adapter_set', str(mapping_path)]
                plan['jobs'].append({'key': key, 'backbone': backbone, 'seed': seed, 'arm': arm,
                    'chain': si + 1, 'mode': mode, 'out': str(dest), 'adapters': adapters,
                    'routes': str(route_file), 'route_axis': axis, 'args': args})
    plan['inputs'] = provenance(sorted(set(paths)))
    write_json(out / 'plan.json', plan)
    check_plan(plan)
    return plan


def collect(plan, out, score=True):
    check_plan(plan)
    deck = read_jsonl(plan['holdout'])
    for job in plan['jobs']:
        if not evaluated(plan, job):
            raise ValueError(f"{job['key']}: evaluation incomplete")
    out = fresh_dir(out)
    index = {'study': 'E6', 'holdout': plan['holdout'], 'backbones': {b: {} for b in BACKBONES}}
    for job in plan['jobs']:
        backbone, arm, seed = job['backbone'], job['arm'], job['seed']
        pred = out / f'{backbone}_{arm}_predicted_s{seed}'
        oracle = out / f'{backbone}_{arm}_oracle_s{seed}'
        assemble([job['out']], deck, pred, backbone, seed, arm + '_predicted', score=score)
        sources = ([Path(a['path']) / 'eval_holdout' for a in job['adapters']] if arm == 'modality'
                   else baseline_sources(plan['runs'], arm, backbone, seed))
        assemble(sources, deck, oracle, backbone, seed, arm + '_oracle', score=score)
        index['backbones'][backbone].setdefault(arm, {})[str(seed)] = {
            'predicted': str(pred), 'oracle': str(oracle), 'routes': job['routes']}
    write_json(out / 'index.json', index)
    return index


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='cmd', required=True)
    prep = sub.add_parser('prepare')
    for name in ('data', 'groups', 'runs', 'routes', 'outputs', 'out'):
        prep.add_argument('--' + name, required=True)
    col = sub.add_parser('collect')
    col.add_argument('--plan', required=True)
    col.add_argument('--out', required=True)
    a = p.parse_args()
    if a.cmd == 'prepare':
        prepare(a.data, a.groups, a.runs, a.routes, a.outputs, a.out)
    else:
        collect(read_json(a.plan), a.out)


if __name__ == '__main__':
    main()
