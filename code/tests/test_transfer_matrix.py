"""E3 CPU contract tests. Synthetic predictions are not GPU-validation evidence."""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

CODE = Path(__file__).resolve().parents[1]
for folder in ('analysis', 'eval'):
    sys.path.insert(0, str(CODE / folder))
from phase_b_common import TASKS, SHORT, SEEDS, BACKBONES, write_json, read_json, digest
from transfer_matrix import (pgf, signed_aggregate, analyze, covariate_tests, pair_covariates,
                             oriented_difference, seed_stability)
from phase_b_jobs import adapter_identity, evaluated, check_plan, run
from plan_e3 import prepare, collect
from group_results import baseline_sources


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')


def preds(rows, base=False):
    result = []
    for r in rows:
        ans = r['answer']
        if base:
            if r['task'] in ('counting', 'regression'):
                ans = str(float(ans) + 2)
            elif r['task'] == 'classification':
                ans = 'a'
        result.append(dict(id=r['id'], TaskType=r['task'], dataset=r['dataset'], modality=r['modality'],
                           gt_answer=r['answer'], Answer=ans, Question=r['question'], ImageName=r['images'],
                           raw_output=ans, truncated=False, n_new_tokens=2))
    return result


def artifact(path, rows, backbone, seed=None, base=False, **meta):
    write_json(path / 'all_predictions.json', preds(rows, base))
    write_json(path / 'run_meta.json', dict(surface='holdout', backbone=backbone, seed=seed,
                                           finished='synthetic', per_task=0, batch_size=1, **meta))
    reports = [r for r in rows if r['task'] == 'report_generation']
    if reports:
        write_rows(path / 'crimson_per_sample.jsonl', [dict(sample_key=r['id'], crimson_score=.4 if base else .7)
                                                     for r in reports])


@pytest.fixture
def e3(tmp_path):
    data, runs = tmp_path / 'data', tmp_path / 'runs'
    answers = ['a', 'a;b', '[[0,0,2,2]]', '2', '2', 'Normal.']
    holdout, train = [], []
    for i, task in enumerate(TASKS):
        for n in range(4):
            row = dict(id=f'{i}-{n}', task=task, dataset='iugc' if i in (0, 2, 4) else f'ds{i}',
                       question='Question?', answer='b' if i == 0 and n % 2 else answers[i],
                       images=[f'/hold/{i}-{n}.png'], modality='Ultrasound' if i in (0, 2, 4) else 'Xray')
            holdout.append(row)
            train.append(dict(row, id='train-' + row['id'], images=['/train/' + row['id'] + '.png']))
    write_rows(data / 'j26_holdout.jsonl', holdout)
    write_rows(data / 'j26_train_mix.jsonl', train)
    for task in TASKS:
        write_rows(data / f'j26_train_{SHORT[task]}.jsonl', [r for r in train if r['task'] == task])
    plan = prepare(data, runs, tmp_path / 'gpu', tmp_path / 'plan')
    for b in BACKBONES:
        for arm in ('zsprompt', 'zsbare'):
            artifact(baseline_sources(runs, arm, b, None)[0], holdout, b, base=True, mode='zs', arm=arm)
    for job in plan['jobs']:
        spec = job['adapters'][0]
        adapter = Path(spec['path'])
        adapter.mkdir(parents=True)
        cfg = dict(run_id=adapter.name, seed=job['seed'], backbone=job['backbone'], deck=spec['train_deck'],
                   llamafactory_args=dict(num_train_epochs=3.0))
        (adapter / 'config.yaml').write_text(yaml.safe_dump(cfg), encoding='utf-8')
        write_json(adapter / 'runtime.json', dict(optimizer_steps=3))
        write_json(adapter / 'decks.json', dict(train=dict(name=spec['train_deck'], md5=spec['train_md5'])))
        (adapter / 'registry_row.csv').write_text('status\ndone\n', encoding='utf-8')
        (adapter / 'git_commit.txt').write_text('synthetic-only', encoding='utf-8')
        write_json(adapter / 'adapter_config.json', {})
        (adapter / 'adapter_model.safetensors').write_bytes(b'not-real-weights')
        diag = [r for r in holdout if r['task'] == job['source']]
        artifact(adapter / 'eval_holdout', diag, job['backbone'], job['seed'], mode='adapter', adapter=str(adapter))
        dest = Path(job['out'])
        artifact(dest, [r for r in holdout if r['task'] != job['source']], job['backbone'], job['seed'],
                 mode='adapter', adapter=str(adapter))
        write_json(dest / 'phase_b_receipt.json', dict(job=job, inputs=plan['inputs'], adapter_files=adapter_identity(spec)))
    return tmp_path, data, plan


def test_metric_orientation_pgf_and_sachdeva_equations():
    assert oriented_difference(3, 5, 'regression') == 2
    assert pgf(3, 5, 'regression') == .4
    assert pgf(.75, .5, 'classification') == .5
    assert pgf(1, 1, 'classification') is None
    assert pgf(0, 0, 'counting') is None
    assert pgf(float('inf'), 5, 'regression') is None
    agg = signed_aggregate([1, .5, -.5, 0, 0])
    assert agg['positive'] == pytest.approx((1 - np.exp(-2/5)) * .75)
    assert agg['negative'] == pytest.approx((1 - np.exp(-1/5)) * -.5)
    assert signed_aggregate([0, 0]) == dict(positive=0, negative=0)
    assert seed_stability([1, 1, 1])['p_wilcoxon'] == .25


def test_plan_reuses_diagonals_and_covers_both_backbones_all_seeds(e3):
    _, _, plan = e3
    assert len(plan['jobs']) == 36
    assert sum('report_generation' in j['tasks'] for j in plan['jobs']) == 30
    assert {j['chain'] for j in plan['jobs']} == {1, 2, 3}
    assert all(len(j['tasks']) == 5 and j['source'] not in j['tasks'] for j in plan['jobs'])
    assert all('--max_steps' not in j['args'] for j in plan['jobs'])
    assert evaluated(plan, plan['jobs'][0])


@pytest.mark.filterwarnings('ignore:.*single label.*:UserWarning')
def test_collection_to_transfer_and_cell_results(e3):
    tmp, data, plan = e3
    index = collect(plan, tmp / 'collected', score=False)
    result = analyze(tmp / 'collected/index.json', data / 'j26_train_mix.jsonl', data / 'j26_holdout.jsonl',
                     tmp / 'analysis', B=8, plots=True)
    assert set(index['backbones']) == set(BACKBONES)
    for backbone in BACKBONES:
        r = result['backbones'][backbone]
        assert len(r['pairs']) == 36
        assert r['pairs']['cls->reg']['C']['mean'] == 2
        assert r['pairs']['cls->reg']['PGF']['mean'] == 1
        assert r['pairs']['cls->reg']['bootstrap']['ci95'] == [2, 2]
        assert len(r['iugc_natural_experiment']) == 6
        cov = r['covariate_tests']['output_structure_similarity']
        assert cov['excluded_targets'] == (['counting'] if backbone == 'mg15' else [])
        assert cov['n_pairs'] == (25 if backbone == 'mg15' else 30)
        assert (tmp / 'analysis' / f'{backbone}_transfer.pdf').exists()
    assert (tmp / 'analysis/transfer.json').exists()


def test_collects_reviewed_chain_outputs_including_legacy_qwen(e3):
    tmp, _, plan = e3
    holdout = read_json(Path(plan['holdout'])) if str(plan['holdout']).endswith('.json') else [
        json.loads(line) for line in Path(plan['holdout']).read_text(encoding='utf-8').splitlines()]
    root = Path(plan['runs']) / 'e3'
    for job in plan['jobs']:
        dest = root / f"e3_{job['backbone']}_task{SHORT[job['source']]}_s{job['seed']}"
        if job['backbone'] == 'q35':
            write_json(dest / 'predictions_holdout.json', preds(holdout))
            reports = [r for r in holdout if r['task'] == 'report_generation']
            write_rows(dest / 'crimson/crimson_per_sample.jsonl', [
                {'sample_key': r['id'], 'crimson_score': .7} for r in reports])
        else:
            artifact(dest, holdout, 'mg15', job['seed'], mode='adapter',
                     adapter=job['adapters'][0]['path'])
    index = collect(plan, tmp / 'chain-collected', score=False)
    qwen_source = index['backbones']['q35']['evaluation_sources']['cls_s1337']
    med_source = index['backbones']['mg15']['evaluation_sources']['cls_s1337']
    assert qwen_source['layout'] == med_source['layout'] == 'chain-full-holdout'
    assert qwen_source['directory'].endswith('e3_q35_taskcls_s1337')
    full = read_json(index['backbones']['q35']['rows']['cls']['1337'] + '/all_predictions.json')
    assert len(full) == len(holdout)


def test_output_structure_similarity_is_graded(e3):
    _, data, _ = e3
    train = [json.loads(line) for line in (data / 'j26_train_mix.jsonl').read_text().splitlines()]
    values = pair_covariates(train)
    sims = {round(v['output_structure_similarity'], 6) for v in values.values()}
    assert len(sims) >= 3
    assert values[(0, 1)]['output_structure_similarity'] > values[(0, 5)]['output_structure_similarity']


@pytest.mark.parametrize('corruption', ['weights', 'deck', 'partial', 'adapter', 'receipt'])
def test_fail_closed_on_changed_or_incomplete_artifacts(e3, corruption):
    tmp, data, plan = e3
    job = plan['jobs'][0]
    dest = Path(job['out'])
    if corruption == 'weights':
        (Path(job['adapters'][0]['path']) / 'adapter_model.safetensors').write_bytes(b'changed')
    elif corruption == 'deck':
        (data / 'j26_holdout.jsonl').write_text('[]', encoding='utf-8')
        with pytest.raises(ValueError, match='input changed'):
            check_plan(plan)
        return
    elif corruption == 'partial':
        rows = read_json(dest / 'all_predictions.json')
        write_json(dest / 'all_predictions.json', rows[:-1])
    elif corruption == 'adapter':
        m = read_json(dest / 'run_meta.json')
        m['adapter'] = str(tmp / 'wrong-adapter')
        write_json(dest / 'run_meta.json', m)
    else:
        receipt = read_json(dest / 'phase_b_receipt.json')
        receipt['adapter_files'] = []
        write_json(dest / 'phase_b_receipt.json', receipt)
    with pytest.raises(ValueError):
        evaluated(plan, job)


def test_nonfinite_transfer_is_reported_not_discarded(e3):
    tmp, data, plan = e3
    collect(plan, tmp / 'collected', score=False)
    index = read_json(tmp / 'collected/index.json')
    dest = Path(index['backbones']['q35']['rows']['cls']['1337'])
    rows = read_json(dest / 'all_predictions.json')
    for r in rows:
        if r['TaskType'] == 'regression':
            r['Answer'] = 'not a number'
    write_json(dest / 'all_predictions.json', rows)
    result = analyze(tmp / 'collected/index.json', data / 'j26_train_mix.jsonl', data / 'j26_holdout.jsonl',
                     tmp / 'analysis', B=2, plots=False)
    pair = result['backbones']['q35']['pairs']['cls->reg']
    assert pair['C']['mean'] is None
    assert pair['C']['values'][0] is None
    assert pair['diagnostics'][0]['malformed_rate'] == 1
    assert pair['bootstrap']['ci95'] is None


def test_covariate_test_on_mu_with_excluded_target_keeps_its_label():
    import numpy as np
    from transfer_matrix import covariate_tests
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(6, 6))
    matrix[:, 3] = np.nan                     # counting column undefined/excluded
    cov = {(i, j): {'x': float(i * 6 + j)} for i in range(6) for j in range(6) if i != j}
    r = covariate_tests(matrix, cov, ('counting',))['x']
    assert r['n_pairs'] == 25 and r['permutations'] == 120 and r['rho'] is not None
    full = covariate_tests(np.where(np.isnan(matrix), 0.0, matrix), cov)['x']
    assert full['n_pairs'] == 30 and full['permutations'] == 720
