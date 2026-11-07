#!/usr/bin/env python3
"""Prepare E3 off-diagonal jobs, then collect full transfer rows using existing E1/E2 diagonals."""
import argparse
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / 'analysis'))
from phase_b_common import (TASKS, SHORT, SEEDS, BACKBONES, read_json, read_jsonl, write_json,
                            digest, provenance, fresh_dir, commit)
from group_results import assemble, baseline_sources, prediction_path
from group_artifacts import validate_predictions
from phase_b_jobs import check_plan, evaluated, adapter_identity


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


def chain_output(plan, job, output_root=None):
    """Location written by Taha's run_eval_chain E3 job files."""
    root = Path(output_root).resolve() if output_root else Path(plan['runs']).resolve() / 'e3'
    return root / f"e3_{job['backbone']}_task{SHORT[job['source']]}_s{job['seed']}"


def collection_source(plan, job, holdout, output_root=None):
    """Prefer reviewed chain-runner output; retain old receipt layout for compatibility."""
    chain = chain_output(plan, job, output_root)
    if chain.exists():
        path = prediction_path(chain)  # fails closed on an interrupted/empty chain directory
        validate_predictions(read_json(path), holdout)
        meta_path = path.parent / 'run_meta.json'
        if meta_path.exists():  # MedGemma eval_harness output
            meta = read_json(meta_path)
            expected_adapter = str(Path(job['adapters'][0]['path']).resolve())
            if (meta.get('surface') != 'holdout' or meta.get('backbone') != job['backbone']
                    or meta.get('mode') != 'adapter' or not meta.get('finished')
                    or meta.get('batch_size') != 1 or meta.get('per_task', 0) != 0
                    or str(Path(meta.get('adapter', '')).resolve()) != expected_adapter):
                raise ValueError(f'{chain}: incompatible E3 eval_harness metadata')
        elif path.name != 'predictions_holdout.json' or job['backbone'] != 'q35':
            raise ValueError(f'{chain}: only Qwen eval_arm output may omit run_meta.json')
        return chain, 'chain-full-holdout'
    if not evaluated(plan, job):
        raise ValueError(f"{job['key']}: evaluate before collection")
    return Path(job['out']), 'phase-b-off-diagonal'


def collect(plan, out, score=True, output_root=None):
    check_plan(plan)
    holdout = read_jsonl(plan['holdout'])
    sources = {}
    for job in plan['jobs']:
        adapter_identity(job['adapters'][0])
        sources[job['key']] = collection_source(plan, job, holdout, output_root)
    out = fresh_dir(out)
    index = {'study': 'E3', 'surface': 'holdout', 'inputs': plan['inputs'], 'backbones': {}}
    for backbone in BACKBONES:
        spec = {'rows': {s: {} for s in SHORT.values()}, 'evaluation_sources': {}}
        for arm in ('zsprompt', 'zsbare'):
            dest = out / backbone / arm
            # allow_unscored: Qwen ZS-bare has 1 report CRIMSON returned no score for (LEDGER n=780)
            assemble(baseline_sources(plan['runs'], arm, backbone, None), holdout,
                     dest, backbone, None, arm, score=score, allow_unscored=True)
            spec[arm] = str(dest.resolve())
        for job in [j for j in plan['jobs'] if j['backbone'] == backbone]:
            short, seed = SHORT[job['source']], job['seed']
            diag = out / backbone / f'diagonal_{short}_s{seed}'
            assemble([job['adapters'][0]['path']], [r for r in holdout if r['task'] == job['source']],
                     diag, backbone, seed, 'task' + short, subset=True, score=score)
            source, layout = sources[job['key']]
            # Non-report adapters answering report questions: a report the CRIMSON judge returned no
            # score for is left out of the mean (same rule as LEDGER/score_crimson.py; Taha 7 Nov),
            # counted per cell in transfer.json. The report adapter's own cells stay strict.
            junk = job['source'] != 'report_generation'
            if layout == 'chain-full-holdout':
                offdiag = out / backbone / f'offdiagonal_{short}_s{seed}'
                assemble([source], [r for r in holdout if r['task'] != job['source']], offdiag,
                         backbone, seed, 'offdiagonal_' + short, subset=True, score=score,
                         allow_unscored=junk)
            else:
                offdiag = source
            dest = out / backbone / f'transfer_{short}_s{seed}'
            assemble([diag, offdiag], holdout, dest, backbone, seed, 'transfer_' + short, score=score,
                     allow_unscored=junk)
            spec['rows'][short][str(seed)] = str(dest.resolve())
            spec['evaluation_sources'][f'{short}_s{seed}'] = {
                'directory': str(source.resolve()), 'layout': layout}
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
    c.add_argument('--e3-outputs', help='chain-runner output root (default: <runs>/e3)')
    a = p.parse_args()
    if a.stage == 'prepare':
        prepare(a.data, a.runs, a.outputs, a.out)
    else:
        collect(read_json(a.plan), a.out, output_root=a.e3_outputs)


if __name__ == '__main__':
    main()
