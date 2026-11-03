#!/usr/bin/env python3
"""Audit an E4/E5 plan (CPU only; it never launches anything).

Stages:
  check   before launch: every config and train/dev/holdout deck still matches the plan's hashes
  status  which runs have a runtime.json and holdout predictions
  verify  after the chains: each run is a complete protocol run, its holdout predictions cover
          its deck exactly, and X-ray report runs have CRIMSON scores. Exit 1 if anything is
          missing or wrong.
Real runs go through code/train/run_chain.sh and code/eval/run_eval_chain.sh with the job files
from plan_e4_e5.py (Taha only, PLAN.md section 7.3).
"""
import argparse
import csv
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
    info = read_json(Path(plan['data_dir']) / 'dataset_info.json')
    for job in plan['jobs']:
        path = REPO / job['config']
        if digest(path) != job['config_md5']:
            raise ValueError(f'config changed: {path}')
        cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
        for name in (job['train_deck'], cfg['eval_deck'], job['holdout_deck']):
            deck_path = Path(plan['data_dir']) / f'{name}.jsonl'
            if digest(deck_path) != manifest['decks'][name]['md5']:
                raise ValueError(f'deck hash mismatch: {deck_path}')
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


def predictions_file(out):
    """eval_arm.py (q35) leaves predictions_holdout.json; eval_harness.py (mg15) all_predictions.json."""
    for name in ('predictions_holdout.json', 'all_predictions.json'):
        if (out / name).is_file():
            return out / name
    return None


def crimson_file(out):
    """run_crimson_chain.sh scores into <pred_dir>/crimson/; crimson_box.sh on a folder writes in place."""
    for path in (out / 'crimson' / 'crimson_per_sample.jsonl', out / 'crimson_per_sample.jsonl'):
        if path.is_file():
            return path
    return None


def evaluated(out, job, deck):
    path = predictions_file(out)
    if path is None:
        return False
    if (out / 'run_meta.json').exists():
        meta = read_json(out / 'run_meta.json')
        if (meta['surface'] != 'holdout' or meta['backbone'] != job['backbone']
                or meta['mode'] != 'adapter' or meta.get('per_task', 0) != 0
                or meta.get('batch_size') != 1 or not meta.get('finished')
                or Path(meta['adapter']).name != job['run_id']):
            raise ValueError(f'{out}: incompatible evaluation provenance')
    elif job['backbone'] != 'q35':
        raise ValueError(f'{out}: harness evaluation without run_meta.json')
    validate_predictions(read_json(path), read_jsonl(deck))
    return True


def run_stage(plan, stage, chain=None):
    jobs = [j for j in plan['jobs'] if chain is None or j['chain'] == chain]
    if not jobs:
        raise ValueError('no jobs selected')
    if stage == 'status':
        for job in jobs:
            run, out = Path(plan['runs_dir']) / job['run_id'], Path(job['pred_dir'])
            print(job['run_id'], 'runtime=' + str((run / 'runtime.json').exists()),
                  'predictions=' + str(predictions_file(out) is not None))
        return
    check_plan(plan)
    if stage == 'check':
        print(f"CHECK PASS: {len(plan['jobs'])} E4/E5 slots ({len({j['run_id'] for j in plan['jobs']})} runs): configs and train/dev/holdout hashes")
        return
    problems = []
    for job in jobs:
        run, out = Path(plan['runs_dir']) / job['run_id'], Path(job['pred_dir'])
        deck = Path(plan['data_dir']) / (job['holdout_deck'] + '.jsonl')
        try:
            if not trained(run, job):
                raise ValueError('not trained')
            if read_json(run / 'decks.json')['train']['md5'] != digest(
                    Path(plan['data_dir']) / (job['train_deck'] + '.jsonl')):
                raise ValueError('trained deck hash differs from plan')
            if not evaluated(out, job, deck):
                raise ValueError('no holdout predictions')
            if job['reports']:
                scores = crimson_file(out)
                if scores is None:
                    raise ValueError('no CRIMSON scores')
                crimson_scores([scores], [r['id'] for r in read_jsonl(deck) if r['task'] == 'report_generation'])
        except (ValueError, KeyError, FileNotFoundError) as e:
            problems.append(f"{job['run_id']}: {e}")
    for p in problems:
        print('FAIL', p)
    print(f"VERIFY: {len(jobs) - len(problems)}/{len(jobs)} runs complete")
    if problems:
        raise SystemExit(1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('stage', choices=['check', 'status', 'verify'])
    p.add_argument('--plan', required=True)
    p.add_argument('--chain', type=int)
    args = p.parse_args()
    run_stage(read_json(args.plan), args.stage, args.chain)


if __name__ == '__main__':
    main()
