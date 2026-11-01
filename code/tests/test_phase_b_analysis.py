"""Full synthetic E7 and E8 surfaces (19 cells, both backbones, every arm and seed)."""
import sys
from pathlib import Path

import numpy as np
import pytest

CODE = Path(__file__).resolve().parents[1]
for folder in ('analysis', 'eval', 'data'):
    sys.path.insert(0, str(CODE / folder))
from phase_b_common import TASKS, SHORT, SEEDS, BACKBONES, write_json, read_json, digest, provenance
from grouping import CLASSIFICATION_DATASET_SLUGS, MODALITY_SLUGS
from representations import make_probe, analyze as analyze_representations
from rq4_synthesis import analyze as analyze_synthesis
from test_transfer_matrix import write_rows, artifact


@pytest.fixture
def surface(tmp_path):
    cells = [('classification', ds) for ds in CLASSIFICATION_DATASET_SLUGS]
    cells += [('multi-label classification', ds) for ds in ('chestdr', 'periapical')]
    cells += [('detection', ds) for ds in ('BUSI-det', 'BUS-UCLM-det', 'iugc')]
    cells += [('regression', ds) for ds in ('boneresorption', 'iugc')]
    cells += [('counting', 'neurips22cell'), ('report_generation', 'IU_XRay')]
    answers = dict(zip(TASKS, ['a', 'a;b', '[[0,0,2,2]]', '2', '2', 'Normal.']))
    rows = []
    for i, (task, ds) in enumerate(cells):
        for j in range(4):
            rows.append(dict(id=f'{i}-{j}', task=task, dataset=ds, modality=list(MODALITY_SLUGS)[i % 7],
                images=[f'/hold/{i}-{j}.png'], question='Question?',
                answer='b' if task == 'classification' and j % 2 else answers[task]))
    train = [dict(r, id='train-' + r['id'], images=['/train/' + r['id'] + '.png']) for r in rows]
    holdout, train_path, probe = tmp_path / 'holdout.jsonl', tmp_path / 'train.jsonl', tmp_path / 'probe.json'
    write_rows(holdout, rows)
    write_rows(train_path, train)
    manifest = make_probe(holdout, probe)
    transfer = {'study': 'E3', 'backbones': {}, 'inputs': provenance([holdout, train_path])}
    for b in BACKBONES:
        t = {'cells': {}, 'pairs': {}, 'aggregates': {'off_diagonal': {}}}
        for i, task in enumerate(TASKS):
            t['aggregates']['off_diagonal'][task] = {
                kind: {sign: {'mean': (i+1)*.01*(1 if sign == 'positive' else -1)} for sign in ('positive', 'negative')}
                for kind in ('transferability', 'malleability')}
            for j, (target, ds) in enumerate(cells):
                t['cells'][f'{SHORT[task]}->{target}/{ds}'] = {'C': {'mean': (j-i)*.01}}
            for j, target in enumerate(TASKS):
                t['pairs'][f'{SHORT[task]}->{SHORT[target]}'] = {'C': {'mean': (j-i)*.01}}
        t['aggregates_bare'] = t['aggregates']
        transfer['backbones'][b] = t
    transfer_path = tmp_path / 'transfer.json'
    write_json(transfer_path, transfer)
    index = {'backbones': {}}
    rng = np.random.default_rng(42)
    names = ['shared', 'shared48', 'hydraT', 'hydraR'] + ['task'+s for s in SHORT.values()]
    names += ['mod'+s for s in MODALITY_SLUGS.values()] + ['ds'+s for s in CLASSIFICATION_DATASET_SLUGS.values()]
    for b in BACKBONES:
        spec = {'models': {}}
        for name in ['base', *names]:
            for seed in ([None] if name == 'base' else SEEDS):
                path = tmp_path / f'{b}_{name}_{seed}'
                path.mkdir()
                features = {pool + '_L1': rng.normal(size=(len(rows), 8)).astype(np.float32)
                            for pool in ('image', 'question', 'last')}
                np.savez_compressed(path / 'features.npz', ids=np.array(manifest['ids']), **features)
                write_json(path / 'features.json', {'spec': {'backbone': b, 'name': name, 'seed': seed},
                    'features_md5': digest(path / 'features.npz'), 'probe_md5': digest(probe), 'layers': [1]})
                if name == 'base':
                    spec['base'] = str(path)
                else:
                    spec['models'].setdefault(name, {})[str(seed)] = str(path)
        index['backbones'][b] = spec
    index_path = tmp_path / 'feature_index.json'
    write_json(index_path, index)
    return dict(tmp=tmp_path, train=train_path, holdout=holdout, probe=probe, transfer=transfer_path,
                feature_index=index_path, rows=rows)


def test_all_representation_arms_feed_19_cell_synthesis(surface):
    w = surface
    rep = analyze_representations(w['feature_index'], w['probe'], w['holdout'], w['transfer'],
                                  w['tmp'] / 'e7', B=9)
    for b in BACKBONES:
        result = rep['backbones'][b]
        assert len(result['shift_from_base']) == 27
        assert len(result['adapter_pairs']) == 15
        assert len(result['cell_predictors']) == 19
        assert result['input_similarity']['image_L1']['centroid']['n_pairs'] == 114
        assert 'p_holm_layers' in result['input_similarity']['image_L1']['centroid']
    index = {'holdout': str(w['holdout']), 'backbones': {}}
    for b in BACKBONES:
        spec = {}
        for arm in ('shared', 'task', 'hydraT', 'hydraR', 'zsprompt', 'zsbare'):
            for seed in ([None] if arm.startswith('zs') else SEEDS):
                path = w['tmp'] / f'e8_{b}_{arm}_{seed}'
                artifact(path, w['rows'], b, seed, base=arm.startswith('zs') or arm == 'shared', mode='assembled', arm=arm)
                if arm.startswith('zs'):
                    spec[arm] = str(path)
                else:
                    spec.setdefault(arm, {})[str(seed)] = str(path)
        index['backbones'][b] = spec
    write_json(w['tmp'] / 'e8_index.json', index)
    syn = analyze_synthesis(w['tmp'] / 'e8_index.json', w['train'], w['transfer'],
                            w['tmp'] / 'e7/representations.json', w['tmp'] / 'e8', B=9)
    assert syn['n_cells'] == 19
    assert len(syn['backbones']['q35']['outcomes']) == 6
    cells = syn['backbones']['q35']['outcomes']['task/zsprompt']['cells']
    assert cells['regression/iugc']['raw_benefit']['mean'] == 2
    assert cells['regression/iugc']['PGF_benefit']['mean'] == 1
    assert len(syn['cross_backbone_signs']) == 6
    assert (w['tmp'] / 'e8/cells.csv').exists()


def test_representation_analysis_rejects_missing_model_seed(surface):
    w = surface
    index = read_json(w['feature_index'])
    del index['backbones']['q35']['models']['shared']['42']
    write_json(w['feature_index'], index)
    with pytest.raises(ValueError, match='three feature seeds'):
        analyze_representations(w['feature_index'], w['probe'], w['holdout'], w['transfer'], w['tmp'] / 'bad', B=2)


def test_routing_analysis_reports_accuracy_and_positive_drop_for_worse_predictions(surface):
    from routing_results import analyze
    w = surface
    index = {'holdout': str(w['holdout']), 'backbones': {}}
    for b in BACKBONES:
        index['backbones'][b] = {}
        for arm in ('task', 'hydraT', 'hydraR', 'modality'):
            index['backbones'][b][arm] = {}
            axis = 'modality' if arm == 'modality' else 'task'
            for seed in SEEDS:
                op, pp = (w['tmp'] / f'{b}_{arm}_{route}_{seed}' for route in ('oracle', 'predicted'))
                artifact(op, w['rows'], b, seed, mode='assembled', arm=arm + '_oracle')
                artifact(pp, w['rows'], b, seed, base=True, mode='assembled', arm=arm + '_predicted')
                routes = w['tmp'] / f'routes_{b}_{arm}_{seed}.json'
                write_json(routes, dict(axis=axis, input='question only' if axis == 'task' else 'image pixels only',
                    data_md5=digest(w['holdout']), predictions=[dict(id=r['id'], label=r[axis], confidence=1) for r in w['rows']]))
                index['backbones'][b][arm][str(seed)] = dict(oracle=str(op), predicted=str(pp), routes=str(routes))
    write_json(w['tmp'] / 'routing_index.json', index)
    result = analyze(w['tmp'] / 'routing_index.json', w['tmp'] / 'routing_results', B=3)
    r = result['backbones']['q35']['task']
    assert all(v['accuracy'] == 1 for v in r['routing'])
    assert r['tasks']['regression']['drop']['mean'] == 2
    assert r['tasks']['regression']['drop_ci95'] == [2, 2]
