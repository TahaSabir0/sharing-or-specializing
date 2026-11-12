#!/usr/bin/env python3
"""Validated, resumable queues for Phase B inference. Real GPU stages belong to Taha.

A job contains eval_harness arguments as a list, an exact expected deck slice, trained
adapter identities, and a fresh output directory. Never invokes a shell with plan text.
Complete artifacts are validated before skipping; interrupted artifacts are preserved.
"""
import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / 'analysis'))
from phase_b_common import read_json, read_jsonl, write_json, digest, provenance
from group_artifacts import validate_predictions, crimson_scores


def check_plan(plan):
    if plan['schema'] != 1:
        raise ValueError('unsupported plan schema')
    for item in plan['inputs']:
        if digest(item['path']) != item['md5']:
            raise ValueError(f"input changed since planning: {item['path']}")
    keys = [j['key'] for j in plan['jobs']]
    paths = [str(Path(j['out']).resolve()) for j in plan['jobs']]
    if len(keys) != len(set(keys)) or len(paths) != len(set(paths)):
        raise ValueError('duplicate job identity or output path')
    return True


def adapter_identity(spec):
    import yaml
    run = Path(spec['path'])
    cfg = yaml.safe_load((run / 'config.yaml').read_text(encoding='utf-8'))
    rt = read_json(run / 'runtime.json')
    with open(run / 'registry_row.csv', encoding='utf-8') as f:
        record = next(csv.DictReader(f))
    if (cfg['run_id'] != run.name or cfg['seed'] != spec['seed'] or cfg['backbone'] != spec['backbone']
            or cfg.get('smoke') or record['status'] != 'done'
            # the only allowed deviation is the r48 arm's declared rank/alpha (the arm's definition: r48, alpha 96)
            or (cfg.get('deviations') and not ('_shared48_' in run.name
                                               and set(cfg['deviations']) <= {'lora_rank', 'lora_alpha'}))
            or rt['optimizer_steps'] <= 0):
        raise ValueError(f'{run}: expected a completed protocol adapter')
    lf = cfg['llamafactory_args']
    if lf['num_train_epochs'] != 3.0 or lf.get('max_steps', -1) > 0:
        raise ValueError(f'{run}: wrong training duration')
    # Real run configs record the deck as llamafactory_args.dataset; test fixtures use `deck`.
    if cfg.get('deck', lf.get('dataset')) != spec['train_deck']:
        raise ValueError(f'{run}: wrong training deck')
    deck = read_json(run / 'decks.json')['train']
    if deck['name'] != spec['train_deck'] or deck['md5'] != spec['train_md5']:
        raise ValueError(f'{run}: training deck hash mismatch')
    files = ['config.yaml', 'runtime.json', 'decks.json', 'git_commit.txt']
    if spec.get('plugin'):
        files += ['method_config.json', 'method_state.safetensors']
    else:
        files += ['adapter_config.json', 'adapter_model.safetensors']
    return provenance([run / f for f in files])


def expected_rows(plan, job):
    rows = read_jsonl(plan['holdout'])
    if job.get('tasks'):
        rows = [r for r in rows if r['task'] in job['tasks']]
    return rows


def evaluated(plan, job):
    out = Path(job['out'])
    if not (out / 'all_predictions.json').exists():
        return False
    meta = read_json(out / 'run_meta.json')
    if (meta.get('surface') != 'holdout' or meta.get('backbone') != job['backbone']
            or meta.get('mode') != job['mode'] or not meta.get('finished')
            or meta.get('batch_size') != 1 or meta.get('per_task', 0) != 0):
        raise ValueError(f'{out}: incompatible evaluation metadata')
    if job['mode'] == 'adapter' and str(Path(meta['adapter']).resolve()) != str(Path(job['adapters'][0]['path']).resolve()):
        raise ValueError(f'{out}: wrong adapter')
    predictions = validate_predictions(read_json(out / 'all_predictions.json'), expected_rows(plan, job))
    if job.get('routes'):
        from predicted_routes import load_routes
        routes, _ = load_routes(job['routes'], plan['holdout'], read_jsonl(plan['holdout']), job['route_axis'])
        if meta.get('routes_md5') != digest(job['routes']) or meta.get('route_axis') != job['route_axis']:
            raise ValueError(f'{out}: wrong predicted routes')
        for pred in predictions:
            if pred.get('predicted_route') != routes[pred['id']]['label']:
                raise ValueError(f'{out}: recorded prediction route differs')
        if job['mode'] == 'plugin':
            if Path(meta['run_dir']).resolve() != Path(job['adapters'][0]['path']).resolve():
                raise ValueError(f'{out}: wrong plugin checkpoint')
        else:
            expected = read_json(job['args'][job['args'].index('--adapter_set') + 1])
            if meta.get('adapter_set') != expected:
                raise ValueError(f'{out}: wrong routed adapter set')
    receipt = read_json(out / 'phase_b_receipt.json')
    if receipt['job'] != job or receipt['inputs'] != plan['inputs']:
        raise ValueError(f'{out}: job or input provenance differs')
    actual = [f for adapter in job['adapters'] for f in adapter_identity(adapter)]
    if receipt['adapter_files'] != actual:
        raise ValueError(f'{out}: adapter files changed or provenance is incomplete')
    return True


def run(plan, stage, chain=None):
    check_plan(plan)
    jobs = [j for j in plan['jobs'] if chain is None or j['chain'] == chain]
    if not jobs:
        raise ValueError('empty job selection')
    if stage == 'check':
        print(f"CHECK PASS: {len(jobs)} {plan['study']} inference jobs; input hashes match")
        return
    if stage == 'status':
        for job in jobs:
            print(job['key'], 'validated complete' if evaluated(plan, job) else 'pending')
        return
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
        raise ValueError('Phase B GPU stages require CUDA_VISIBLE_DEVICES=1')
    subprocess.run(['nvidia-smi', '-i', '1'], check=True)
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '4'
    import fcntl
    for job in jobs:
        out = Path(job['out'])
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out.parent / (out.name + '.lock'), 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            print(f"{stage}: {job['key']}", flush=True)
            identities = [f for adapter in job['adapters'] for f in adapter_identity(adapter)]
            done = evaluated(plan, job)
            if stage == 'eval':
                if not done:
                    if out.exists() and any(out.iterdir()):
                        raise ValueError(f'{out}: interrupted output preserved; inspect/archive before retry')
                    write_json(out / 'phase_b_receipt.json', {'job': job, 'inputs': plan['inputs'],
                                                            'adapter_files': identities})
                    subprocess.run([sys.executable, str(CODE / 'eval/eval_harness.py'), *job['args']], check=True)
                    if not evaluated(plan, job):
                        raise ValueError(f'{out}: evaluator did not produce complete results')
                subprocess.run([sys.executable, str(CODE / 'eval/score_run.py'), '--out_dir', str(out)], check=True)
            elif stage == 'crimson':
                if not done:
                    raise ValueError(f'{out}: evaluation is incomplete')
                ids = [r['id'] for r in expected_rows(plan, job) if r['task'] == 'report_generation']
                if not ids:
                    continue
                cp = out / 'crimson_per_sample.jsonl'
                if not cp.exists():
                    subprocess.run(['bash', str(CODE / 'eval/crimson_box.sh'), str(out)], check=True)
                crimson_scores([cp], ids)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['check', 'status', 'eval', 'crimson'])
    p.add_argument('--plan', required=True)
    p.add_argument('--chain', type=int, choices=[1, 2, 3])
    a = p.parse_args()
    run(read_json(a.plan), a.stage, a.chain)


if __name__ == '__main__':
    main()
