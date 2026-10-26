#!/usr/bin/env python3
"""Execute or audit an E4/E5 plan. GPU stages: designated operator only, after Phase A.

Stages: check (CPU, verify data/config hashes), status (CPU), train, eval, crimson.
Each GPU stage stops on failure. Finished training/evaluation is validated and skipped;
partial training is preserved and rejected, never silently overwritten. Resume a stopped
queue by rerunning the stage after resolving its reported failure.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
REPO = CODE.parent
sys.path.insert(0, str(CODE / 'eval'))
from group_artifacts import read_json, read_jsonl, digest, validate_predictions, crimson_scores


def check_plan(plan):
    if digest(plan['manifest']) != plan['manifest_md5']:
        raise ValueError('group manifest changed since planning; regenerate and review the plan')
    manifest = read_json(plan['manifest'])
    import yaml
    for job in plan['jobs']:
        path = REPO / job['config']
        if digest(path) != job['config_md5']:
            raise ValueError(f'config changed: {path}')
        cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
        for name in (job['train_deck'], cfg['eval_deck'], job['holdout_deck']):
            deck_path = Path(plan['data_dir']) / f'{name}.jsonl'
            if digest(deck_path) != manifest['decks'][name]['md5']:
                raise ValueError(f'deck hash mismatch: {deck_path}')
        info = read_json(Path(plan['data_dir']) / 'dataset_info.json')
        for name in (job['train_deck'], cfg['eval_deck']):
            if info.get(name, {}).get('file_name') != f'{name}.jsonl':
                raise ValueError(f'dataset_info missing or wrong for {name}')


def trained(run, job):
    import yaml
    if not (run / 'runtime.json').exists():
        return False
    cfg = yaml.safe_load((run / 'config.yaml').read_text(encoding='utf-8'))
    runtime = read_json(run / 'runtime.json')
    with open(run / 'registry_row.csv', encoding='utf-8') as f:
        registry = next(csv.DictReader(f))
    if (cfg['run_id'] != job['run_id'] or cfg.get('smoke') or cfg.get('deviations')
            or registry['status'] != 'done' or runtime['optimizer_steps'] <= 0
            or cfg['backbone'] != job['backbone'] or cfg['seed'] != job['seed']):
        raise ValueError(f'{run}: not a completed protocol run')
    for name in ('adapter_config.json', 'adapter_model.safetensors', 'decks.json', 'git_commit.txt'):
        if not (run / name).is_file():
            raise ValueError(f'{run}: missing {name}')
    if read_json(run / 'decks.json')['train']['name'] != job['train_deck']:
        raise ValueError(f'{run}: trained on the wrong deck')
    source_cfg = yaml.safe_load((REPO / job['config']).read_text(encoding='utf-8'))
    lf = cfg['llamafactory_args']
    if (lf['dataset'] != source_cfg['deck'] or lf['num_train_epochs'] != 3.0
            or lf.get('max_steps', -1) > 0 or not runtime.get('trainable_matches_P8')):
        raise ValueError(f'{run}: incomplete or wrong training recipe')
    return True


def evaluated(out, job, deck):
    if not (out / 'all_predictions.json').exists():
        return False
    meta = read_json(out / 'run_meta.json')
    if (meta['surface'] != 'holdout' or meta['backbone'] != job['backbone']
            or meta['mode'] != 'adapter' or meta.get('per_task', 0) != 0
            or meta.get('batch_size') != 1 or not meta.get('finished')
            or Path(meta['adapter']).name != job['run_id']):
        raise ValueError(f'{out}: incompatible evaluation provenance')
    validate_predictions(read_json(out / 'all_predictions.json'), read_jsonl(deck))
    return True


def run_stage(plan, stage, chain=None):
    jobs = [j for j in plan['jobs'] if chain is None or j['chain'] == chain]
    if not jobs:
        raise ValueError('no jobs selected')
    if stage == 'status':
        for job in jobs:
            run = Path(plan['runs_dir']) / job['run_id']
            print(job['run_id'], 'runtime=' + str((run / 'runtime.json').exists()),
                  'predictions=' + str((run / 'eval_holdout/all_predictions.json').exists()))
        return
    check_plan(plan)
    if stage == 'check':
        print(f"CHECK PASS: {len(plan['jobs'])} configs and their train/dev/holdout hashes")
        return
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
        raise ValueError('GPU stages require CUDA_VISIBLE_DEVICES=1')
    subprocess.run(['nvidia-smi', '-i', '1'], check=True)
    os.environ.update(JOURNAL_DATA=plan['data_dir'], JOURNAL_RUNS=plan['runs_dir'])
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '4'
    for job in jobs:
        run = Path(plan['runs_dir']) / job['run_id']
        out = run / 'eval_holdout'
        deck = Path(plan['data_dir']) / (job['holdout_deck'] + '.jsonl')
        print(f"{stage}: {job['run_id']}", flush=True)
        # The account is shared: prevent two workers from mutating the same run.
        import fcntl
        lock_root = Path(plan['runs_dir']) / '.e4e5_locks'
        lock_root.mkdir(parents=True, exist_ok=True)
        lock = open(lock_root / job['run_id'], 'a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if stage == 'train':
            if trained(run, job):
                print('validated completed training; skipping', flush=True)
                lock.close()
                continue
            if run.exists() and any(run.iterdir()):
                raise ValueError(f'{run}: partial training preserved. Operator must inspect before retrying.')
            cmd = [sys.executable, str(CODE / 'train/journal_train.py'), '--config', job['config']]
        elif stage == 'eval':
            if not trained(run, job):
                raise ValueError(f'{run}: training not complete')
            if read_json(run / 'decks.json')['train']['md5'] != digest(
                    Path(plan['data_dir']) / (job['train_deck'] + '.jsonl')):
                raise ValueError(f'{run}: trained deck hash differs from plan')
            if not evaluated(out, job, deck):
                if out.exists() and any(out.iterdir()):
                    raise ValueError(f'{out}: partial evaluation preserved; inspect before resuming')
                cmd = [sys.executable, str(CODE / 'eval/eval_harness.py'), '--mode', 'adapter',
                       '--adapter', str(run), '--backbone', job['backbone'], '--batch_size', '1',
                       '--data', str(deck), '--surface', 'holdout', '--out_dir', str(out)]
            else:
                cmd = None
        else:
            if not job['reports']:
                lock.close()
                continue
            if not evaluated(out, job, deck):
                raise ValueError(f'{out}: evaluate before CRIMSON')
            report_ids = [r['id'] for r in read_jsonl(deck) if r['task'] == 'report_generation']
            scores = out / 'crimson_per_sample.jsonl'
            if scores.exists():
                crimson_scores([scores], report_ids)
                print('validated completed CRIMSON; skipping', flush=True)
                lock.close()
                continue
            cmd = ['bash', str(CODE / 'eval/crimson_box.sh'), str(out)]
        if cmd:
            subprocess.run(cmd, cwd=REPO, check=True)
        if stage == 'train' and not trained(run, job):
            raise ValueError(f'{run}: training returned without complete artifacts')
        if stage == 'eval':
            if not evaluated(out, job, deck):
                raise ValueError(f'{out}: incomplete evaluation')
            subprocess.run([sys.executable, str(CODE / 'eval/score_run.py'), '--out_dir', str(out)], check=True)
        if stage == 'crimson':
            crimson_scores([out / 'crimson_per_sample.jsonl'], report_ids)
        lock.close()
    print(f'{stage.upper()} COMPLETE', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['check', 'status', 'train', 'eval', 'crimson'])
    p.add_argument('--plan', required=True)
    p.add_argument('--chain', type=int)
    args = p.parse_args()
    run_stage(read_json(args.plan), args.stage, args.chain)


if __name__ == '__main__':
    main()
