"""CPU integration tests: synthetic decks -> 102-job plan -> artifacts -> statistics.

Synthetic adapters are fixtures, never model weights or evidence of a GPU pass.
"""
import json
import sys
from pathlib import Path

import pytest
import yaml

CODE = Path(__file__).resolve().parents[1]
for folder in ('data', 'train', 'eval', 'analysis'):
    sys.path.insert(0, str(CODE / folder))
from build_group_decks import build_group_decks
from grouping import CLASSIFICATION_DATASET_SLUGS, MODALITY_SLUGS
from plan_e4_e5 import load_configs, schedule, write_plan
from group_artifacts import digest, read_json, read_jsonl, validate_predictions, crimson_scores
from group_pipeline import check_plan, trained, evaluated
from group_results import assemble, collect
from compare_groups import compare

pytestmark = pytest.mark.filterwarnings("ignore:A single label was found:UserWarning")


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding='utf-8')


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')


def row(key, modality, dataset, task='classification', answer='a'):
    return dict(id=key, modality=modality, dataset=dataset, task=task, answer=answer,
                question='Question?', images=[f'/images/{key}.png'], image_group=key,
                modality_id=0, dataset_id=0, task_id=0, conversations=[])


@pytest.fixture
def workflow(tmp_path):
    source, grouped = tmp_path / 'source', tmp_path / 'groups'
    rows = []
    mods = list(MODALITY_SLUGS)
    for i, ds in enumerate(CLASSIFICATION_DATASET_SLUGS):
        for j, answer in enumerate(('a', 'b')):
            rows.append(row(f'{ds}-{j}', mods[i % 7], ds, answer=answer))
    rows += [row('report', 'Xray', 'IU_XRay', 'report_generation', 'Normal.'),
             row('reg', 'Xray', 'boneresorption', 'regression', '2'),
             row('det', 'Ultrasound', 'BUSI-det', 'detection', '[[0,0,2,2]]'),
             row('count', 'Microscopy', 'neurips22cell', 'counting', '2'),
             row('ml', 'Xray', 'chestdr', 'multi-label classification', 'a;b')]
    hold = [dict(r, id='hold-' + r['id'], image_group='hold-' + r['image_group'],
                 images=['/hold/' + r['id'] + '.png']) for r in rows]
    write_rows(source / 'j26_train_mix.jsonl', rows)
    write_rows(source / 'j26_devselect.jsonl', rows)
    write_rows(source / 'j26_holdout.jsonl', hold)
    manifest = build_group_decks(source, grouped)
    configs = load_configs()
    chains = schedule(configs, manifest)
    plan = write_plan(chains, grouped / 'GROUP_MANIFEST.json', tmp_path / 'jobs',
                      str(grouped), str(tmp_path / 'runs'))
    return dict(source=source, grouped=grouped, manifest=manifest, configs=configs,
                chains=chains, plan=plan, hold=hold, tmp=tmp_path)


def predictions(deck):
    return [dict(id=r['id'], TaskType=r['task'], dataset=r['dataset'], modality=r['modality'],
                 gt_answer=r['answer'], Answer=r['answer'], Question=r['question'],
                 ImageName=r['images'], raw_output=r['answer'], truncated=False, n_new_tokens=2) for r in deck]


def fake_run(w, job):
    run = Path(w['plan']['runs_dir']) / job['run_id']
    run.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((CODE.parent / job['config']).read_text(encoding='utf-8'))
    cfg.update(smoke={}, deviations={}, llamafactory_args=dict(dataset=cfg['deck'], num_train_epochs=3.0))
    (run / 'config.yaml').write_text(yaml.safe_dump(cfg), encoding='utf-8')
    write_json(run / 'runtime.json', dict(optimizer_steps=3, trainable_matches_P8=True))
    (run / 'registry_row.csv').write_text('status\ndone\n', encoding='utf-8')
    (run / 'adapter_model.safetensors').write_bytes(b'synthetic-test-only')
    write_json(run / 'adapter_config.json', {})
    (run / 'git_commit.txt').write_text('synthetic-fixture', encoding='utf-8')
    write_json(run / 'decks.json', {'train': {'name': job['train_deck'],
               'md5': digest(w['grouped'] / (job['train_deck'] + '.jsonl'))}})
    out = run / 'eval_holdout'
    deck = read_jsonl(w['grouped'] / (job['holdout_deck'] + '.jsonl'))
    write_json(out / 'all_predictions.json', predictions(deck))
    write_json(out / 'run_meta.json', dict(surface='holdout', backbone=job['backbone'], mode='adapter',
               per_task=0, batch_size=1, finished='fixture', adapter=str(run)))
    reports = [r for r in deck if r['task'] == 'report_generation']
    if reports:
        write_rows(out / 'crimson_per_sample.jsonl', [dict(sample_key=r['id'], crimson_score=0.75) for r in reports])
    return run, out, deck


def test_plan_covers_102_jobs_once_and_six_report_judges(workflow):
    w = workflow
    check_plan(w['plan'])
    jobs = w['plan']['jobs']
    assert len(jobs) == len({j['run_id'] for j in jobs}) == 102
    assert sum(j['study'] == 'E4' for j in jobs) == 42
    assert sum(j['reports'] for j in jobs) == 6
    assert {j['chain'] for j in jobs} == {1, 2, 3}
    assert max(c['weight'] for c in w['chains']) - min(c['weight'] for c in w['chains']) <= 4
    for i in (1, 2, 3):
        lines = (w['tmp'] / f'jobs/eval_chain_{i}.txt').read_text().splitlines()[1:]
        assert all('--batch_size 1' in s and '--surface holdout' in s for s in lines)
    assert len((w['tmp'] / 'jobs/crimson.txt').read_text().splitlines()) == 7


def test_deck_hash_changes_fail_preflight(workflow):
    w = workflow
    p = w['grouped'] / (w['plan']['jobs'][0]['train_deck'] + '.jsonl')
    p.write_text(p.read_text() + '\n')
    with pytest.raises(ValueError, match='hash mismatch'):
        check_plan(w['plan'])


@pytest.mark.parametrize('corruption', ['duplicate', 'missing', 'wrong_gt', 'error'])
def test_prediction_integrity_failures(workflow, corruption):
    deck = workflow['hold']
    ps = predictions(deck)
    if corruption == 'duplicate':
        ps.append(ps[0])
    elif corruption == 'missing':
        ps.pop()
    elif corruption == 'wrong_gt':
        ps[0]['gt_answer'] = 'wrong'
    else:
        ps[0]['Answer'] = 'Error: out of memory'
    with pytest.raises(ValueError):
        validate_predictions(ps, deck)


def test_completed_stage_validation_rejects_smoke_and_wrong_adapter(workflow):
    job = workflow['plan']['jobs'][0]
    run, out, deck = fake_run(workflow, job)
    assert trained(run, job)
    assert evaluated(out, job, workflow['grouped'] / (job['holdout_deck'] + '.jsonl'))
    cfg = yaml.safe_load((run / 'config.yaml').read_text())
    cfg['smoke'] = {'max_steps': 50}
    (run / 'config.yaml').write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match='protocol'):
        trained(run, job)
    meta = read_json(out / 'run_meta.json')
    meta['adapter'] = 'different_adapter'
    write_json(out / 'run_meta.json', meta)
    with pytest.raises(ValueError, match='provenance'):
        evaluated(out, job, workflow['grouped'] / (job['holdout_deck'] + '.jsonl'))


def test_all_102_artifacts_collect_into_12_arms_and_compare(workflow):
    w = workflow
    for job in w['plan']['jobs']:
        fake_run(w, job)
    out = w['tmp'] / 'collected'
    index = collect(w['plan'], w['source'] / 'j26_holdout.jsonl', out, score=False)
    assert len(index) == 12
    assert len(read_json(out / 'E4_q35_s1337/all_predictions.json')) == len(w['hold'])
    paths = [str(out / f'E5_q35_s{s}') for s in (1337, 2026, 42)]
    result = compare(paths, paths, w['source'] / 'j26_train_mix.jsonl', w['tmp'] / 'stats', B=10)
    assert result['per_task']['classification']['diff_b_minus_a'] == 0
    assert len(result['per_cell']) == 10
    assert (w['tmp'] / 'stats/cells.csv').is_file()


def test_legacy_baseline_slice_and_missing_crimson(workflow):
    w = workflow
    source = w['tmp'] / 'legacy'
    write_json(source / 'predictions_holdout.json', predictions(w['hold']))
    cls = [r for r in w['hold'] if r['task'] == 'classification']
    assemble([source], cls, w['tmp'] / 'base_cls', 'q35', 1337, 'shared', subset=True, score=False)
    with pytest.raises(ValueError, match='CRIMSON'):
        assemble([source], w['hold'], w['tmp'] / 'base_all', 'q35', 1337, 'shared', score=False)


def test_reports_keep_per_question_scores(workflow):
    w = workflow
    source = w['tmp'] / 'baseline'
    write_json(source / 'all_predictions.json', predictions(w['hold']))
    report = next(r for r in w['hold'] if r['task'] == 'report_generation')
    write_rows(source / 'crimson_per_sample.jsonl', [dict(sample_key=report['id'], crimson_score=0.6)])
    out = w['tmp'] / 'assembled'
    assemble([source], w['hold'], out, 'q35', 1337, 'shared', score=True)
    assert read_json(out / 'crimson_metrics.json')['crimson_score_mean'] == 0.6
    assert crimson_scores([out / 'crimson_per_sample.jsonl'], [report['id']])[report['id']] == 0.6
    metrics = read_json(out / 'metrics.json')['tasks']
    assert metrics['detection']['macro'] == 1.0  # singleton F1 must not disappear
    assert metrics['report_generation']['macro'] == 0.6
    assert metrics['classification']['macro'] == 1.0


def test_holdout_leakage_rejected_before_writing(workflow):
    w = workflow
    train = read_jsonl(w['source'] / 'j26_train_mix.jsonl')
    write_rows(w['source'] / 'j26_holdout.jsonl', train)
    with pytest.raises(ValueError, match='leakage'):
        build_group_decks(w['source'], w['tmp'] / 'bad_output')
    assert not (w['tmp'] / 'bad_output').exists()


def test_mae_benefit_orientation_and_different_seed_rejected(workflow):
    w = workflow
    deck = [r for r in w['hold'] if r['task'] == 'regression']
    paths_a, paths_b = [], []
    for seed in (1337, 2026, 42):
        for arm, answer, paths in [('a', '4', paths_a), ('b', '2', paths_b)]:
            source = w['tmp'] / f'raw_{arm}_{seed}'
            ps = predictions(deck)
            ps[0]['Answer'] = answer
            write_json(source / 'all_predictions.json', ps)
            out = w['tmp'] / f'{arm}_{seed}'
            assemble([source], deck, out, 'q35', seed, arm, score=False)
            paths.append(str(out))
    res = compare(paths_a, paths_b, w['source'] / 'j26_train_mix.jsonl', w['tmp'] / 'mae', B=10)
    cell = res['per_cell']['regression/boneresorption']
    assert cell['diff_b_minus_a'] == -2
    assert cell['benefit_b_minus_a'] == 2
    assert res['arm_b']['regression']['ci95'] == [0.0, 0.0]
    from compare_groups import validate_runs
    with pytest.raises(ValueError, match='seed order'):
        validate_runs(paths_a, paths_b[::-1])


def test_incomplete_crimson_and_nonfinite_scores_fail(tmp_path):
    path = tmp_path / 'crimson.jsonl'
    write_rows(path, [dict(sample_key='x', crimson_score=None)])
    with pytest.raises(ValueError, match='invalid'):
        crimson_scores([path], ['x'])
    write_rows(path, [dict(sample_key='x', crimson_score=0.5)])
    with pytest.raises(ValueError, match='coverage'):
        crimson_scores([path], ['x', 'y'])


def test_plan_rejects_recipe_override(tmp_path):
    import shutil
    cfg_root = tmp_path / 'configs'
    for folder in ('e4', 'e5'):
        shutil.copytree(CODE / 'configs' / folder, cfg_root / folder)
    path = next((cfg_root / 'e4').glob('*.yaml'))
    cfg = yaml.safe_load(path.read_text())
    cfg['overrides'] = {'num_train_epochs': 0.01}
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match='fixed E4/E5 matrix'):
        load_configs(cfg_root)


def test_real_stage_commands_and_completed_run_skips(workflow, monkeypatch):
    """Exercise orchestration with simulated subprocesses, never a model/GPU."""
    from types import SimpleNamespace
    import group_pipeline as pipeline
    w = workflow
    job = next(j for j in w['plan']['jobs'] if j['reports'])
    plan = dict(w['plan'], jobs=[job])
    monkeypatch.setitem(sys.modules, 'fcntl', SimpleNamespace(LOCK_EX=1, LOCK_NB=2,
                                                            flock=lambda *args: None))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1')
    # run_stage sets these variables; ensure monkeypatch restores the calling environment.
    for key in ('JOURNAL_DATA', 'JOURNAL_RUNS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        monkeypatch.setenv(key, 'test')
    commands = []

    def fake_process(cmd, **kwargs):
        commands.append(cmd)
        if any(str(c).endswith(('journal_train.py', 'eval_harness.py')) for c in cmd):
            fake_run(w, job)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(pipeline.subprocess, 'run', fake_process)
    pipeline.run_stage(plan, 'train')
    train_commands = [c for c in commands if any(str(x).endswith('journal_train.py') for x in c)]
    assert len(train_commands) == 1 and '--max_steps' not in train_commands[0]
    pipeline.run_stage(plan, 'train')
    assert sum(any(str(x).endswith('journal_train.py') for x in c) for c in commands) == 1
    run = Path(plan['runs_dir']) / job['run_id']
    (run / 'eval_holdout').rename(run / 'saved_fixture_eval')
    pipeline.run_stage(plan, 'eval')
    eval_command = next(c for c in commands if any(str(x).endswith('eval_harness.py') for x in c))
    assert eval_command[eval_command.index('--batch_size') + 1] == '1'
    assert eval_command[eval_command.index('--surface') + 1] == 'holdout'
    assert '--per_task' not in eval_command
    pipeline.run_stage(plan, 'eval')
    assert sum(any(str(x).endswith('eval_harness.py') for x in c) for c in commands) == 1
    pipeline.run_stage(plan, 'crimson')  # valid existing scores must not call judge again
    assert not any('crimson_box.sh' in str(c) for c in commands)
