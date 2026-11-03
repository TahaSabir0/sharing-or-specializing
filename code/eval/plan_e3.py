#!/usr/bin/env python3
"""Prepare E3 off-diagonal jobs, then collect full transfer rows using existing E1/E2 diagonals."""
import argparse
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / 'analysis'))
from phase_b_common import (TASKS, SHORT, SEEDS, BACKBONES, read_json, read_jsonl, write_json,
                            digest, provenance, fresh_dir, commit)
from group_results import assemble, baseline_sources
from phase_b_jobs import check_plan, evaluated


def prepare(data, runs, outputs, out):
    data, runs, outputs, out = [Path(p).resolve() for p in (data, runs, outputs, out)]
    holdout = data / 'j26_holdout.jsonl'
    train = data / 'j26_train_mix.jsonl'
    rows = read_jsonl(holdout)
    if set(r['task'] for r in rows) != set(TASKS):
        raise ValueError('holdout must cover all six tasks')
    inputs = [holdout, train] + [data / f'j26_train_{s}.jsonl' for s in SHORT.values()]
    plan = {'schema': 1, 'study': 'E3', 'holdout': str(holdout), 'train': str(train),
            'runs': str(runs), 'inputs': provenance(inputs), 'code_commit': commit(), 'jobs': []}
    for backbone in BACKBONES:
        resolution = 768 if backbone == 'q35' else 896
        for si, seed in enumerate(SEEDS):
            for source in TASKS:
                short = SHORT[source]
                adapter = runs / f'j26_{backbone}_task{short}_full_{resolution}_s{seed}'
                key = f'e3_{backbone}_{short}_s{seed}'
                dest = outputs / key
                tasks = [t for t in TASKS if t != source]
                args = ['--mode', 'adapter', '--adapter', str(adapter), '--backbone', backbone,
                        '--data', str(holdout), '--surface', 'holdout', '--batch_size', '1',
                        '--tasks', ','.join(SHORT[t] for t in tasks), '--out_dir', str(dest)]
                plan['jobs'].append({'key': key, 'chain': si + 1, 'backbone': backbone, 'seed': seed,
                    'source': source, 'tasks': tasks, 'mode': 'adapter', 'out': str(dest), 'args': args,
                    'adapters': [{'path': str(adapter), 'seed': seed, 'backbone': backbone,
                        'train_deck': f'j26_train_{short}', 'train_md5': digest(data / f'j26_train_{short}.jsonl')} ]})
    fresh_dir(out)
    write_json(out / 'plan.json', plan)
    check_plan(plan)
    print(f"Prepared {len(plan['jobs'])} off-diagonal row jobs (180 task slices), 30 CRIMSON jobs. Diagonals reused.")
    return plan


def collect(plan, out, score=True):
    check_plan(plan)
    holdout = read_jsonl(plan['holdout'])
    for job in plan['jobs']:
        if not evaluated(plan, job):
            raise ValueError(f"{job['key']}: evaluate before collection")
    out = fresh_dir(out)
    index = {'study': 'E3', 'surface': 'holdout', 'inputs': plan['inputs'], 'backbones': {}}
    for backbone in BACKBONES:
        spec = {'rows': {s: {} for s in SHORT.values()}}
        for arm in ('zsprompt', 'zsbare'):
            dest = out / backbone / arm
            assemble(baseline_sources(plan['runs'], arm, backbone, None), holdout,
                     dest, backbone, None, arm, score=score)
            spec[arm] = str(dest.resolve())
        for job in [j for j in plan['jobs'] if j['backbone'] == backbone]:
            short, seed = SHORT[job['source']], job['seed']
            diag = out / backbone / f'diagonal_{short}_s{seed}'
            assemble([job['adapters'][0]['path']], [r for r in holdout if r['task'] == job['source']],
                     diag, backbone, seed, 'task' + short, subset=True, score=score)
            dest = out / backbone / f'transfer_{short}_s{seed}'
            assemble([diag, job['out']], holdout, dest, backbone, seed, 'transfer_' + short, score=score)
            spec['rows'][short][str(seed)] = str(dest.resolve())
        index['backbones'][backbone] = spec
    write_json(out / 'index.json', index)
    return index


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='stage', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--data', required=True)
    prep.add_argument('--runs', required=True)
    prep.add_argument('--outputs', required=True)
    prep.add_argument('--out', required=True)
    c = sub.add_parser('collect')
    c.add_argument('--plan', required=True)
    c.add_argument('--out', required=True)
    a = p.parse_args()
    if a.stage == 'prepare':
        prepare(a.data, a.runs, a.outputs, a.out)
    else:
        collect(read_json(a.plan), a.out)


if __name__ == '__main__':
    main()
