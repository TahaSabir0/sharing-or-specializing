#!/usr/bin/env python3
"""Assemble E4/E5 oracle predictions and existing comparison arms without new inference.

collect: join all 7 E4 or 10 E5 slices for each backbone/seed, checking exact holdout coverage.
baseline: normalize saved E0/E1/E2 predictions (legacy or harness) to the same output format.
No score parser is reimplemented: each assembled directory is scored by score_run.py.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / 'eval'))
from group_artifacts import (read_json, read_jsonl, digest, index_rows,
                             validate_predictions, crimson_scores, REPORT)


def prediction_path(source):
    source = Path(source)
    if source.is_file():
        return source
    for name in ('all_predictions.json', 'predictions_holdout.json', 'eval_holdout/all_predictions.json'):
        if (source / name).is_file():
            return source / name
    raise FileNotFoundError(f'no saved holdout predictions in {source}')


def assemble(sources, deck, out, backbone, seed, arm, subset=False, score=True):
    """subset=True permits a full baseline to be sliced to E5 classification questions."""
    predictions, score_paths, provenance = [], [], []
    for source in sources:
        path = prediction_path(source)
        rows = read_json(path)
        predictions.extend(rows)
        meta_path = path.parent / 'run_meta.json'
        if meta_path.exists():
            meta = read_json(meta_path)
            if meta.get('surface') != 'holdout' or meta.get('backbone') != backbone:
                raise ValueError(f'{path}: surface/backbone mismatch')
            if 'seed' in meta and meta['seed'] != seed:
                raise ValueError(f'{path}: seed mismatch')
        cfg_path = path.parent / 'config.yaml'
        if not cfg_path.exists():
            cfg_path = path.parent.parent / 'config.yaml'
        if cfg_path.exists():
            import yaml
            cfg = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
            if cfg.get('backbone') != backbone or cfg.get('seed') != seed or cfg.get('smoke'):
                raise ValueError(f'{path}: training backbone/seed/smoke mismatch')
        for cp in (path.parent / 'crimson_per_sample.jsonl',
                   path.parent / 'crimson/crimson_per_sample.jsonl'):
            if cp.exists():
                score_paths.append(cp)
                break
        provenance.append({'path': str(path.resolve()), 'md5': digest(path),
                           'metadata': read_json(meta_path) if meta_path.exists() else None,
                           'training_config': str(cfg_path.resolve()) if cfg_path.exists() else None})
    selected = validate_predictions(predictions, deck, exact=not subset)
    report_ids = [r['id'] for r in deck if r['task'] == REPORT]
    scores = crimson_scores(score_paths, report_ids) if report_ids else {}
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'{out}: output exists; use a fresh analysis directory')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'all_predictions.json').write_text(json.dumps(selected, ensure_ascii=False) + '\n', encoding='utf-8')
    meta = {'surface': 'holdout', 'mode': 'assembled', 'backbone': backbone, 'seed': seed,
            'arm': arm, 'data': 'canonical holdout IDs validated during assembly',
            'sources': provenance, 'code_commit': subprocess.run(
                ['git', '-C', str(CODE.parent), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip(),
            'n_predictions': len(selected)}
    (out / 'run_meta.json').write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')
    if report_ids:
        (out / 'crimson_per_sample.jsonl').write_text(''.join(
            json.dumps({'sample_key': k, 'crimson_score': scores[k]}) + '\n'
            for k in sorted(scores)), encoding='utf-8')
        (out / 'crimson_metrics.json').write_text(json.dumps({
            'crimson_score_mean': sum(scores.values()) / len(scores),
            'n': len(scores), 'n_scored': len(scores)}), encoding='utf-8')
    if score:
        subprocess.run([sys.executable, str(CODE / 'eval/score_run.py'), '--out_dir', str(out)], check=True)
    return meta


def collect(plan, holdout, out, score=True):
    from collections import defaultdict
    sys.path.insert(0, str(CODE / 'train'))
    from group_pipeline import check_plan, evaluated, trained
    check_plan(plan)
    deck = read_jsonl(holdout)
    manifest = read_json(plan['manifest'])
    if digest(holdout) != manifest['sources']['j26_holdout']['md5']:
        raise ValueError('canonical holdout hash differs from source used for grouped decks')
    groups = defaultdict(list)
    for job in plan['jobs']:
        groups[(job['study'], job['backbone'], job['seed'])].append(job)
    if len(groups) != 12:
        raise ValueError('expected E4/E5 x 2 backbones x 3 seeds')
    # Preflight all inputs before writing any assembled result.
    for jobs in groups.values():
        for job in jobs:
            run = Path(plan['runs_dir']) / job['run_id']
            group_deck = Path(plan['data_dir']) / (job['holdout_deck'] + '.jsonl')
            if not trained(run, job) or not evaluated(run / 'eval_holdout', job, group_deck):
                raise ValueError(f'{run}: incomplete training/evaluation')
            if read_json(run / 'decks.json')['train']['md5'] != manifest['decks'][job['train_deck']]['md5']:
                raise ValueError(f'{run}: training deck hash mismatch')
            if job['reports']:
                crimson_scores([run / 'eval_holdout/crimson_per_sample.jsonl'],
                               [r['id'] for r in read_jsonl(group_deck) if r['task'] == REPORT])
    index = []
    for (study, backbone, seed), jobs in sorted(groups.items()):
        selected = deck if study == 'E4' else [r for r in deck if r['task'] == 'classification']
        dest = Path(out) / f'{study}_{backbone}_s{seed}'
        assemble([Path(plan['runs_dir']) / j['run_id'] / 'eval_holdout' for j in jobs],
                 selected, dest, backbone, seed, study, score=score)
        index.append({'study': study, 'backbone': backbone, 'seed': seed, 'directory': str(dest.resolve())})
    (Path(out) / 'index.json').write_text(json.dumps(index, indent=2) + '\n', encoding='utf-8')
    return index


def baseline_sources(runs, arm, backbone, seed):
    resolution = {'q35': 768, 'mg15': 896}[backbone]
    if arm in ('zsbare', 'zsprompt'):
        return [Path(runs) / f'j26_{backbone}_{arm}_holdout_{resolution}']
    arms = [f'task{x}' for x in ('cls', 'ml', 'det', 'count', 'reg', 'report')] if arm == 'task' else [arm]
    return [Path(runs) / f'j26_{backbone}_{a}_full_{resolution}_s{seed}' for a in arms]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('collect')
    c.add_argument('--plan', required=True)
    c.add_argument('--holdout', required=True)
    c.add_argument('--out', required=True)
    b = sub.add_parser('baseline')
    b.add_argument('--runs', default='/data/ugrip2026/journal/runs')
    b.add_argument('--arm', required=True, choices=['shared', 'shared48', 'task', 'hydraT', 'hydraR', 'zsbare', 'zsprompt'])
    b.add_argument('--backbone', required=True, choices=['q35', 'mg15'])
    b.add_argument('--seed', type=int, choices=[1337, 2026, 42], default=1337)
    b.add_argument('--holdout', required=True)
    b.add_argument('--cls-only', action='store_true')
    b.add_argument('--sources', nargs='+', help='explicit saved files/folders if historical names differ')
    b.add_argument('--out', required=True)
    args = p.parse_args()
    if args.cmd == 'collect':
        collect(read_json(args.plan), args.holdout, args.out)
    else:
        deck = read_jsonl(args.holdout)
        if args.cls_only:
            deck = [r for r in deck if r['task'] == 'classification']
        sources = args.sources or baseline_sources(args.runs, args.arm, args.backbone, args.seed)
        if args.cls_only and args.arm == 'task' and not args.sources:
            sources = sources[:1]
        assemble(sources, deck, args.out, args.backbone,
                 None if args.arm.startswith('zs') else args.seed, args.arm, subset=True)


if __name__ == '__main__':
    main()
