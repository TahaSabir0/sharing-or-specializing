"""E6-E8 CPU tests, including routed harness execution with a deterministic fake model."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
for folder in ('analysis', 'eval', 'train', 'data'):
    sys.path.insert(0, str(CODE / folder))
from phase_b_common import write_json, read_json, digest, TASKS, SEEDS, BACKBONES
from predicted_routes import load_routes, routing_ids, accuracy, LABELS
from routers import audit_split, unique_images, task_collator
from representations import (cka, rsa, u_center, cell_permutation, question_mask, make_probe,
                              probe_rows, layer_selection, load_features)
from rq4_synthesis import predictors, pgf_benefit
from test_group_workflow import workflow, write_rows


def test_unbiased_cka_matches_independent_hsic_estimator():
    rng = np.random.default_rng(42)
    x, y = rng.normal(size=(21, 8)), rng.normal(size=(21, 11))
    def hsic(a, b):
        k, l = a @ a.T, b @ b.T
        np.fill_diagonal(k, 0)
        np.fill_diagonal(l, 0)
        n = len(a)
        return (np.trace(k @ l) + k.sum()*l.sum()/((n-1)*(n-2))
                - 2*np.dot(k.sum(0), l.sum(0))/(n-2)) / (n*(n-3))
    expected = hsic(x, y) / np.sqrt(hsic(x, x)*hsic(y, y))
    assert cka(x, y) == pytest.approx(expected, abs=1e-12)
    q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    assert cka(x, 3*x @ q + 10) == pytest.approx(1, abs=1e-12)
    assert cka(x, np.ones((21, 3))) is None
    with pytest.raises(ValueError):
        cka(x[:3], y[:3])
    assert rsa(x, 3*x @ q) == pytest.approx(1)


def test_cell_permutations_preserve_source_structure():
    x = np.tile(np.arange(19), (6, 1))
    result = cell_permutation(x, x, B=99, seed=8)
    assert result['rho'] == pytest.approx(1)
    assert result['n_cells'] == 19 and result['n_pairs'] == 114
    assert result['p_permutation'] == .01
    assert cell_permutation(np.ones_like(x), x)['rho'] is None


def test_question_mask_excludes_image_prefix_and_chat_suffix():
    original = 'Image-1: <image>\nWhat is this?<end>\n'
    expanded = original.replace('<image>', '<image_soft_token>' * 4)
    offsets = [[i, i+1] for i in range(len(expanded))]
    mask = question_mask(offsets, original, 'What is this?')
    assert ''.join(c for c, keep in zip(expanded, mask) if keep) == 'What is this?'
    assert layer_selection('q35', 32) == [1, 8, 16, 24, 32]
    assert layer_selection('mg15', 34)[-1] == 34


def test_probe_selection_is_stable_stratified_and_uses_small_cells(tmp_path):
    rows = [dict(id=f'{ds}-{i}', task='classification', dataset=ds) for ds, n in [('small', 22), ('big', 40)] for i in range(n)]
    deck = tmp_path / 'holdout.jsonl'
    write_rows(deck, rows)
    a = make_probe(deck, tmp_path / 'a.json')
    assert len(a['ids']) == 52
    assert a['cells']['classification/small']['selected'] == 22
    with pytest.raises(ValueError, match='expected 53 IDs'):
        make_probe(deck, tmp_path / 'wrong.json', expected_total=53)
    write_rows(deck, list(reversed(rows)))
    b = make_probe(deck, tmp_path / 'b.json')
    assert a['ids'] == b['ids']
    with pytest.raises(ValueError, match='belong'):
        probe_rows(deck, a)
    assert len(probe_rows(deck, b)) == 52


def test_router_audit_detects_partial_multiimage_leakage(workflow):
    w = workflow
    train = [json.loads(s) for s in (w['source'] / 'j26_train_mix.jsonl').read_text().splitlines()]
    assert audit_split(train, w['hold'])['overlapping_images'] == 0
    hold = [dict(r) for r in w['hold']]
    hold[0]['images'] = ['/different.png', train[0]['images'][0]]
    with pytest.raises(ValueError, match='overlap'):
        audit_split(train, hold)
    assert len(unique_images(train + train)) == len(unique_images(train))
    with pytest.raises(ValueError, match='inconsistent'):
        unique_images([train[0], dict(train[0], modality='different')])


def test_gliclass_scalar_targets_survive_public_collator():
    import torch
    def old_collator(items):
        # Reproduce public v0.1.20 behavior: tensor scalars are not padded/stacked.
        labels = [] if isinstance(items[0]['labels'], torch.Tensor) else torch.tensor([r['labels'] for r in items])
        return {'labels': labels, 'input_ids': torch.tensor([[1], [2]]),
                'labels_text': ['a', 'b'], 'input_texts': ['question A', 'question B']}
    items = [{'labels': torch.tensor(0)}, {'labels': torch.tensor(1)}]
    result = task_collator(old_collator)(items)
    assert result['labels'].tolist() == [0, 1]
    assert set(result) == {'labels', 'input_ids'}
    assert isinstance(items[0]['labels'], torch.Tensor)


def test_task_router_training_uses_questions_only_and_completes_fixed_epochs(tmp_path, monkeypatch):
    import torch
    import types
    import importlib.machinery
    from routers import train_task
    # Initialize torch's optimizer imports before replacing the optional transformers module.
    torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))])
    observed = {}
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.config = SimpleNamespace(architecture_type='uni-encoder', prompt_first=False)
        @classmethod
        def from_pretrained(cls, path, local_files_only):
            assert local_files_only
            return cls()
        def forward(self, input_ids, labels):
            assert labels.dtype == torch.int64 and labels.ndim == 1
            return SimpleNamespace(loss=self.weight.square().mean())
        def save_pretrained(self, path):
            torch.save(self.state_dict(), Path(path) / 'fake.pt')
    class Dataset:
        def __init__(self, records, tokenizer, augment_config, max_length, problem_type,
                     shuffle_labels=True, prompt_first=False, add_description=True):
            observed.update(records=records, augmentation=augment_config.enabled, shuffle=shuffle_labels,
                            problem_type=problem_type, max_length=max_length)
            self.records = records
        def __len__(self):
            return len(self.records)
        def __getitem__(self, i):
            return {'input_ids': [i], 'labels': torch.tensor(i % 6)}
    class Collator:
        def __init__(self, device):
            assert device == 'cpu'
        def __call__(self, items):
            assert all(isinstance(r['labels'], int) for r in items)
            return {k: torch.tensor([r[k] for r in items]) for k in items[0]}
    gliclass = types.ModuleType('gliclass')
    gliclass.__spec__ = importlib.machinery.ModuleSpec('gliclass', loader=None)
    gliclass.GLiClassModel = Model
    gliclass.data_processing = SimpleNamespace(GLiClassDataset=Dataset, DataCollatorWithPadding=Collator,
                                               AugmentationConfig=lambda enabled: SimpleNamespace(enabled=enabled))
    transformers = types.ModuleType('transformers')
    transformers.__spec__ = importlib.machinery.ModuleSpec('transformers', loader=None)
    transformers.AutoTokenizer = SimpleNamespace(from_pretrained=lambda *a, **kw:
        SimpleNamespace(save_pretrained=lambda path: write_json(Path(path) / 'tokenizer.json', {})))
    monkeypatch.setitem(sys.modules, 'gliclass', gliclass)
    monkeypatch.setitem(sys.modules, 'transformers', transformers)
    rows = [dict(question='Only question ' + str(i), task=t, answer='secret answer', dataset='secret dataset')
            for i, t in enumerate(TASKS)]
    recipe = train_task(rows, 'fake-initial', tmp_path, 1337, 'cpu')
    assert len(recipe['loss_by_epoch']) == 3
    assert recipe['loss_by_epoch'][-1] < recipe['loss_by_epoch'][0]
    assert all(set(r) == {'text', 'all_labels', 'true_labels'} for r in observed['records'])
    assert all(r['text'] == rows[i]['question'] for i, r in enumerate(observed['records']))
    assert observed['augmentation'] is False and observed['shuffle'] is False


def test_predicted_route_never_relabels_scoring_task(tmp_path):
    row = dict(id='a', task='regression', task_id=4, modality_id=5, dataset_id=12)
    deck = tmp_path / 'deck.jsonl'
    write_rows(deck, [row])
    path = tmp_path / 'routes.json'
    value = dict(axis='task', input='question only', data_md5=digest(deck),
                 predictions=[dict(id='a', label='classification', confidence=.8)])
    write_json(path, value)
    routes, _ = load_routes(path, deck, [row], 'task')
    ids = routing_ids(row, routes['a'], 'task')
    assert ids['task_id'] == 0 and row['task_id'] == 4
    assert row['task'] == 'regression'
    assert accuracy([row], value['predictions'], 'task')['accuracy'] == 0
    value['predictions'][0]['label'] = 'wrong-label'
    write_json(path, value)
    with pytest.raises(ValueError, match='unknown'):
        load_routes(path, deck, [row], 'task')


def test_routed_harness_selects_each_prediction_and_parses_true_task(tmp_path, monkeypatch):
    import torch
    from methods import base as mb
    module = SimpleNamespace(AutoModelForImageTextToText=object, AutoProcessor=object, BitsAndBytesConfig=object)
    monkeypatch.setitem(sys.modules, 'transformers', module)
    monkeypatch.setitem(sys.modules, 'infer_single_adapter', SimpleNamespace(parse_answer=lambda raw, task: 'parsed:' + task))
    spec = importlib.util.spec_from_file_location('harness_test', CODE / 'eval/eval_harness.py')
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    rows = [dict(id=str(i), task=task, task_id=LABELS['task'][task], modality_id=5, dataset_id=12,
                 images=['x.png'], question='Question?', answer='2', dataset='iugc', modality='Ultrasound')
            for i, task in enumerate(('classification', 'regression'))]
    data, routes = tmp_path / 'data.jsonl', tmp_path / 'routes.json'
    write_rows(data, rows)
    write_json(routes, dict(axis='task', input='question only', data_md5=digest(data), predictions=[
        dict(id='0', label='regression', confidence=.7), dict(id='1', label='classification', confidence=.9)]))
    selected, seen = [], []
    class Model:
        config = SimpleNamespace(model_type='qwen3_5')
        def generate(self, **kwargs):
            seen.append(mb.routing.get('task_id').item())
            return torch.tensor([[8, 8, 5, 2]])
    class Inputs(dict):
        def to(self, device):
            return self
    proc = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0, decode=lambda ids, **kw: 'raw'))
    harness.AutoProcessor = SimpleNamespace(from_pretrained=lambda *a, **kw: proc)
    harness.build_model = lambda a: (Model(), selected.append, {'mode': 'routed'})
    harness.chat_text = lambda *a, **kw: 'prompt'
    harness.load_images = lambda *a, **kw: []
    harness.encode_batch = lambda *a, **kw: Inputs(input_ids=torch.tensor([[8, 8]]))
    harness.pinned_eos = lambda *a: [2]
    harness.torch = SimpleNamespace(tensor=lambda v, **kw: torch.tensor(v), inference_mode=torch.inference_mode,
        cuda=SimpleNamespace(max_memory_allocated=lambda: 0, empty_cache=lambda: None),
        OutOfMemoryError=torch.OutOfMemoryError)
    monkeypatch.setattr(sys, 'argv', ['eval_harness.py', '--mode', 'routed', '--routes', str(routes),
        '--route_axis', 'task', '--adapter_set', 'unused.json', '--data', str(data), '--surface', 'holdout',
        '--out_dir', str(tmp_path / 'output'), '--batch_size', '1'])
    harness.main()
    assert selected == ['classification', 'regression']  # Harness evaluates regression first.
    assert seen == [0, 4]
    preds = {r['id']: r for r in read_json(tmp_path / 'output/all_predictions.json')}
    assert preds['0']['TaskType'] == 'classification' and preds['0']['Answer'] == 'parsed:classification'
    assert preds['1']['TaskType'] == 'regression' and preds['1']['predicted_route'] == 'classification'
    assert mb.routing.get('task_id') is None


def test_e7_plan_covers_every_model_and_all_probes(workflow):
    from plan_e7 import prepare, run
    w = workflow
    from phase_b_common import SHORT, read_jsonl
    train = read_jsonl(w['source'] / 'j26_train_mix.jsonl')
    for task, short in SHORT.items():
        write_rows(w['source'] / f'j26_train_{short}.jsonl', [r for r in train if r['task'] == task])
    probe = w['tmp'] / 'probe.json'
    make_probe(w['source'] / 'j26_holdout.jsonl', probe)
    plan = prepare(w['source'], w['grouped'], w['tmp'] / 'runs', probe, w['tmp'] / 'features', w['tmp'] / 'e7')
    assert len(plan['jobs']) == 164
    assert sum(read_json(j['spec'])['name'] == 'base' for j in plan['jobs']) == 2
    assert all('tasks' not in read_json(j['spec']) for j in plan['jobs'])
    assert run(plan, check=True) is None


def test_e6_plan_has_24_full_holdout_jobs_and_correct_routes(workflow):
    from plan_e6 import prepare
    from phase_b_common import SHORT, read_jsonl
    w = workflow
    train = read_jsonl(w['source'] / 'j26_train_mix.jsonl')
    for task, short in SHORT.items():
        write_rows(w['source'] / f'j26_train_{short}.jsonl', [r for r in train if r['task'] == task])
    for axis in ('task', 'modality'):
        for seed in SEEDS:
            write_json(w['tmp'] / f'routes/{axis}_s{seed}.json', dict(axis=axis, seed=seed,
                input='question only' if axis == 'task' else 'image pixels only',
                data_md5=digest(w['source'] / 'j26_holdout.jsonl'),
                predictions=[dict(id=r['id'], label=r[axis], confidence=1) for r in w['hold']]))
    plan = prepare(w['source'], w['grouped'], w['tmp'] / 'runs', w['tmp'] / 'routes',
                   w['tmp'] / 'e6_outputs', w['tmp'] / 'e6_jobs')
    assert len(plan['jobs']) == 24
    assert sum(j['mode'] == 'plugin' for j in plan['jobs']) == 12
    assert all('--tasks' not in j['args'] for j in plan['jobs'])
    assert all(j['route_axis'] == ('modality' if j['arm'] == 'modality' else 'task') for j in plan['jobs'])


def test_e8_normalization_keeps_mae_direction_and_perfect_baseline_undefined():
    assert pgf_benefit(4, 2, 5, 'regression') == pytest.approx(.4)
    assert pgf_benefit(.6, .8, .5, 'classification') == pytest.approx(.4)
    assert pgf_benefit(.6, .8, 1, 'classification') is None


def test_extraction_pools_real_hook_tensors_without_answer_tokens(tmp_path, monkeypatch):
    import torch
    import journal_render
    from representations import extract
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(image_token_id=9)
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2, bias=False) for _ in range(2)])
            with torch.no_grad():
                self.model.layers[0].weight.copy_(torch.eye(2))
                self.model.layers[1].weight.copy_(2*torch.eye(2))
        def forward(self, input_ids, attention_mask, use_cache):
            assert use_cache is False and not torch.is_grad_enabled()
            hidden = input_ids.float().unsqueeze(-1).expand(-1, -1, 2)
            for layer in self.model.layers:
                hidden = layer(hidden)
            return hidden
    model = Model()
    monkeypatch.setitem(sys.modules, 'eval_harness', SimpleNamespace(
        build_model=lambda a: (model, lambda t: None, {}), DEFAULT_BASE={'q35': 'fake-base'}))
    proc = SimpleNamespace(tokenizer=SimpleNamespace(is_fast=True))
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        AutoProcessor=SimpleNamespace(from_pretrained=lambda *a, **kw: proc)))
    class Inputs(dict):
        def to(self, device):
            return self
    real_tensor = torch.tensor
    monkeypatch.setattr(torch, 'tensor', lambda *a, **kw: real_tensor(*a, **{k: v for k, v in kw.items() if k != 'device'}))
    monkeypatch.setattr(journal_render, 'chat_text', lambda *a, **kw: 'XXQ!Y')
    monkeypatch.setattr(journal_render, 'load_images', lambda *a, **kw: [])
    monkeypatch.setattr(journal_render, 'encode_batch', lambda *a, **kw: Inputs(
        input_ids=real_tensor([[9, 9, 3, 4, 5]]), attention_mask=real_tensor([[1, 1, 1, 1, 1]]),
        offset_mapping=real_tensor([[[i, i+1] for i in range(5)]])))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1')
    row = dict(id='x', task='classification', dataset='ds', question='Q!', answer='MUST NEVER ENTER MODEL',
               images=['x.png'], task_id=0, modality_id=0, dataset_id=0)
    deck, probe, spec = tmp_path / 'deck.jsonl', tmp_path / 'probe.json', tmp_path / 'spec.json'
    write_rows(deck, [row])
    make_probe(deck, probe)
    write_json(spec, dict(mode='zs', name='base', backbone='q35', seed=None, adapters=[]))
    extract(deck, probe, spec, tmp_path / 'features')
    meta, features = load_features(tmp_path / 'features', probe)
    np.testing.assert_allclose(features['image_L1'], [[9, 9]])
    np.testing.assert_allclose(features['question_L2'], [[7, 7]])
    np.testing.assert_allclose(features['last_L2'], [[10, 10]])
    assert meta['token_counts'] == [dict(image=2, question=2, last=1)]
    assert all(not layer._forward_hooks for layer in model.model.layers)
