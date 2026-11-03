#!/usr/bin/env python3
"""Prepare/run all E7 feature configurations, including each TaskLoRA on ALL probe rows."""
import argparse
import os
import sys
from pathlib import Path

from phase_b_common import (CODE, TASKS, SHORT, SEEDS, BACKBONES, read_json, read_jsonl, write_json,
                            digest, provenance, fresh_dir, commit)
sys.path.insert(0, str(CODE / 'data'))
from grouping import MODALITY_SLUGS, CLASSIFICATION_DATASET_SLUGS
from representations import probe_rows, load_features, extract


def prepare(data, groups, runs, probe, outputs, out):
    data, groups, runs, probe, outputs, out = [Path(p).resolve() for p in (data, groups, runs, probe, outputs, out)]
    holdout = data / 'j26_holdout.jsonl'
    probe_rows(holdout, read_json(probe))
    paths = [holdout, probe, data / 'j26_train_mix.jsonl']
    out = fresh_dir(out)
    plan = {'schema': 1, 'study': 'E7', 'holdout': str(holdout), 'probe': str(probe),
            'code_commit': commit(), 'jobs': []}
    index = {'backbones': {}}
    for backbone in BACKBONES:
        res = 768 if backbone == 'q35' else 896
        entries = [('base', None, None)]
        for seed in SEEDS:
            entries += [(arm, seed, data / 'j26_train_mix.jsonl') for arm in ('shared', 'shared48', 'hydraT', 'hydraR')]
            entries += [('task' + SHORT[t], seed, data / f'j26_train_{SHORT[t]}.jsonl') for t in TASKS]
            entries += [('mod' + s, seed, groups / f'j26_train_mod{s}.jsonl') for s in MODALITY_SLUGS.values()]
            entries += [('ds' + s, seed, groups / f'j26_train_ds{s}.jsonl') for s in CLASSIFICATION_DATASET_SLUGS.values()]
        index['backbones'][backbone] = {'models': {}}
        for name, seed, deck in entries:
            key = f'e7_{backbone}_{name}' + (f'_s{seed}' if seed is not None else '')
            dest, spec_path = outputs / key, out / (key + '.json')
            spec = {'name': name, 'backbone': backbone, 'seed': seed, 'adapters': []}
            if name == 'base':
                spec['mode'] = 'zs'
                index['backbones'][backbone]['base'] = str(dest)
            else:
                adapter = runs / f'j26_{backbone}_{name}_full_{res}_s{seed}'
                plugin = name in ('hydraT', 'hydraR')
                spec.update(mode='plugin' if plugin else 'adapter')
                spec['run_dir' if plugin else 'adapter'] = str(adapter)
                spec['adapters'] = [{'path': str(adapter), 'backbone': backbone, 'seed': seed,
                    'plugin': plugin, 'train_deck': deck.stem, 'train_md5': digest(deck)}]
                paths.append(deck)
                index['backbones'][backbone]['models'].setdefault(name, {})[str(seed)] = str(dest)
            write_json(spec_path, spec)
            paths.append(spec_path)
            plan['jobs'].append({'key': key, 'spec': str(spec_path), 'out': str(dest),
                                 'chain': SEEDS.index(seed) + 1 if seed is not None else 1})
    plan['inputs'] = provenance(sorted(set(paths)))
    write_json(out / 'plan.json', plan)
    write_json(out / 'index.json', index)
    return plan


def run(plan, chain=None, check=False):
    for item in plan['inputs']:
        if digest(item['path']) != item['md5']:
            raise ValueError(f"E7 input changed: {item['path']}")
    jobs = [j for j in plan['jobs'] if chain is None or j['chain'] == chain]
    if not jobs:
        raise ValueError('empty queue')
    if check:
        print(f'CHECK PASS: {len(jobs)} feature extraction jobs')
        return
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
        raise ValueError('GPU 1 only')
    from phase_b_jobs import adapter_identity
    import fcntl
    for job in jobs:
        out = Path(job['out'])
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out.parent / (out.name + '.lock'), 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            spec = read_json(job['spec'])
            if (out / 'features.json').exists():
                meta, _ = load_features(out, plan['probe'])
                if meta['spec'] != spec or meta['adapter_files'] != [f for a in spec['adapters'] for f in adapter_identity(a)]:
                    raise ValueError('completed feature checkpoint/spec changed')
                print(f"{job['key']}: validated complete; skipping", flush=True)
                continue
            extract(plan['holdout'], plan['probe'], job['spec'], out)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='cmd', required=True)
    prep = sub.add_parser('prepare')
    for name in ('data', 'groups', 'runs', 'probe', 'outputs', 'out'):
        prep.add_argument('--' + name, required=True)
    for name in ('run', 'check'):
        q = sub.add_parser(name)
        q.add_argument('--plan', required=True)
        q.add_argument('--chain', type=int, choices=[1, 2, 3])
    a = p.parse_args()
    if a.cmd == 'prepare':
        prepare(a.data, a.groups, a.runs, a.probe, a.outputs, a.out)
    else:
        run(read_json(a.plan), a.chain, a.cmd == 'check')
