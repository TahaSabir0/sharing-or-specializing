#!/usr/bin/env python3
"""E7 fixed probes, forward-only token pooling, debiased linear CKA and RSA.

CKA uses unbiased (U-centered) linear Gram matrices, as in Kornblith et al. 2019:
https://proceedings.mlr.press/v97/kornblith19a.html . Negative debiased estimates are retained.
Base input similarity is centroid cosine and RSA of similarities to the same cell anchors.
The primary transfer test permutes target cells jointly across sources, not independent pairs.
"""
import argparse
import hashlib
import itertools
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.stats import rankdata

import bootstrap as bs
from phase_b_common import (CODE, TASKS, SHORT, SEEDS, BACKBONES, read_json, read_jsonl, write_json,
                            digest, provenance, fresh_dir, commit, summary, finite, require_input_hash)
sys.path.insert(0, str(CODE / 'data'))

POOLS = ('image', 'question', 'last')


def make_probe(holdout_path, out, n=30, seed=20261019):
    """Stable to row ordering: select by salted ID hash, up to 30 from every cell."""
    if n <= 0:
        raise ValueError('probe size must be positive')
    from group_artifacts import index_rows
    rows = read_jsonl(holdout_path)
    index_rows(rows)
    cells = sorted({(r['task'], r['dataset']) for r in rows})
    ids, counts = [], {}
    for task, ds in cells:
        candidates = [r for r in rows if (r['task'], r['dataset']) == (task, ds)]
        candidates.sort(key=lambda r: hashlib.sha256(f"{seed}:{r['id']}".encode()).hexdigest())
        selected = candidates[:n]
        ids += [r['id'] for r in selected]
        counts[f'{task}/{ds}'] = {'available': len(candidates), 'selected': len(selected)}
    manifest = {'surface': 'holdout', 'holdout_md5': digest(holdout_path), 'seed': seed,
                'per_cell': n, 'ids': ids, 'cells': counts,
                'selection': 'SHA256(seed:question_id); cells smaller than n use all rows'}
    if Path(out).exists():
        raise ValueError(f'{out}: probe IDs are frozen; use a new path for a different probe')
    write_json(out, manifest)
    return manifest


def probe_rows(holdout, probe):
    from group_artifacts import index_rows
    if digest(holdout) != probe['holdout_md5'] or probe['surface'] != 'holdout':
        raise ValueError('probe does not belong to this holdout')
    rows = index_rows(read_jsonl(holdout))
    if len(probe['ids']) != len(set(probe['ids'])) or not set(probe['ids']) <= rows.keys():
        raise ValueError('duplicate or unknown probe IDs')
    return [rows[i] for i in probe['ids']]


def u_center(gram):
    gram = np.array(gram, dtype=np.float64, copy=True)
    n = len(gram)
    if gram.shape != (n, n) or n < 4 or not np.isfinite(gram).all():
        raise ValueError('unbiased CKA needs a finite square Gram matrix with n >= 4')
    np.fill_diagonal(gram, 0)
    sums = gram.sum(axis=0)
    gram -= sums[None, :] / (n - 2)
    gram -= sums[:, None] / (n - 2)
    gram += sums.sum() / ((n - 1) * (n - 2))
    np.fill_diagonal(gram, 0)
    return gram


def cka(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y):
        raise ValueError('CKA requires paired rows')
    # Center features first for numerical stability; U-centering removes this translation.
    x, y = x - x.mean(0), y - y.mean(0)
    k, l = u_center(x @ x.T), u_center(y @ y.T)
    denom = np.linalg.norm(k) * np.linalg.norm(l)
    return float(np.sum(k * l) / denom) if denom > 1e-20 else None


def cosine(x, y):
    den = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / den) if den > 1e-20 else None


def rsa(x, y):
    """Spearman correlation of pairwise cosine-distance RDMs on matched observations."""
    if len(x) != len(y) or len(x) < 3:
        raise ValueError('RSA requires >=3 matched observations')
    def rdm(z):
        norm = np.linalg.norm(z, axis=1, keepdims=True)
        if (norm <= 1e-20).any():
            return None
        z = z / norm
        return (1 - z @ z.T)[np.triu_indices(len(z), 1)]
    a, b = rdm(np.asarray(x, float)), rdm(np.asarray(y, float))
    if a is None or b is None:
        return None
    return spearman(a, b)


def spearman(x, y):
    x, y = rankdata(x), rankdata(y)
    return cosine(x - x.mean(), y - y.mean())


def cell_permutation(similarities, transfer, B=10000, seed=20261013):
    """Matrices [source task, target cell]. All sources share one permuted cell order."""
    x, y = np.asarray(similarities, float), np.asarray(transfer, float)
    if x.shape != y.shape or x.ndim != 2 or B < 1:
        raise ValueError('expected equal source-by-cell matrices and positive B')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return {'rho': None, 'p_permutation': None, 'reason': 'undefined input/transfer; no pairs dropped'}
    xr, yr = rankdata(x.ravel()), rankdata(y.ravel()).reshape(y.shape)
    xr, yr = xr - xr.mean(), yr - yr.mean()
    den = np.linalg.norm(xr) * np.linalg.norm(yr)
    if den <= 1e-20:
        return {'rho': None, 'p_permutation': None, 'reason': 'constant variable'}
    rho = float(xr @ yr.ravel() / den)
    rng = np.random.default_rng(seed)
    hits = sum(abs(xr @ yr[:, rng.permutation(y.shape[1])].ravel() / den) >= abs(rho) - 1e-12 for _ in range(B))
    return {'rho': rho, 'p_permutation': (hits + 1) / (B + 1), 'B': B,
            'n_cells': x.shape[1], 'n_pairs': x.size, 'unit': 'target cells, jointly across source tasks'}


def question_mask(offsets, rendered, question):
    """Processor expands image placeholders before tokenization, only BEFORE question.

    Locate question characters backwards from the unchanged prompt suffix; offsets are
    returned by the same processor call as model IDs. This avoids a second tokenization.
    """
    offsets = np.asarray(offsets)
    start = rendered.rfind(question)
    if not question or start < 0:
        raise ValueError('empty or absent question')
    suffix = len(rendered) - (start + len(question))
    end = int(offsets[:, 1].max()) - suffix
    begin = end - len(question)
    mask = (offsets[:, 1] > begin) & (offsets[:, 0] < end) & (offsets[:, 1] > offsets[:, 0])
    if not mask.any():
        raise ValueError('processor offsets do not cover question')
    return mask


def decoder_layers(model):
    import torch
    candidates = [(name, module) for name, module in model.named_modules()
                  if name.endswith('.layers') and isinstance(module, torch.nn.ModuleList)
                  and not any(x in name.lower() for x in ('vision', 'visual'))]
    if len(candidates) != 1:
        raise ValueError(f'expected one language decoder layer list; found {[n for n, _ in candidates]}')
    return candidates[0]


def layer_selection(backbone, depth):
    proposed = [1, 8, 16, 24, 32, 40, 48, 56, 64] if backbone == 'q35' else [1, 4, 8, 12, 16, 20, 24, 28, 32]
    return sorted(set([n for n in proposed if n <= depth] + [depth]))


def extract(holdout, probe_path, spec_path, out, device='cuda:0'):
    """One forward per probe, no answer token or ground truth enters the model."""
    if device != 'cuda:0' or os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
        raise ValueError('feature extraction requires CUDA_VISIBLE_DEVICES=1 and cuda:0')
    import torch
    from transformers import AutoProcessor
    from eval_harness import build_model, DEFAULT_BASE
    from journal_render import chat_text, encode_batch, load_images, image_rules
    from methods import base as mb
    from phase_b_jobs import adapter_identity
    spec = read_json(spec_path)
    rows = probe_rows(holdout, read_json(probe_path))
    identities = [f for a in spec.get('adapters', []) for f in adapter_identity(a)]
    out = fresh_dir(out)
    backbone = spec['backbone']
    base_model = spec.get('base_model', DEFAULT_BASE[backbone])
    a = SimpleNamespace(mode=spec['mode'], base_model=base_model, protocol='bare',
                        adapter=spec.get('adapter'), adapter_set=spec.get('adapter_set'),
                        run_dir=spec.get('run_dir'))
    model, select, _ = build_model(a)
    model.requires_grad_(False)
    proc = AutoProcessor.from_pretrained(base_model, local_files_only=True)
    if not proc.tokenizer.is_fast:
        raise ValueError('E7 needs a fast tokenizer with character offsets')
    layer_name, layers = decoder_layers(model)
    selected = layer_selection(backbone, len(layers))
    config = model.config
    image_id = getattr(config, 'image_token_id', None)
    if image_id is None:
        image_id = getattr(config, 'image_token_index', None)
    if image_id is None:
        raise ValueError('model config does not declare its image token ID')
    storage = {f'{pool}_L{layer}': [] for pool in POOLS for layer in selected}
    masks, captured = {}, {}
    handles = []
    def hook(layer):
        def capture(module, inputs, output):
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            hidden = hidden[0].detach().float()
            for pool in POOLS:
                mask = masks[pool].to(hidden.device)
                captured[f'{pool}_L{layer}'] = hidden[mask].mean(0).cpu().numpy()
        return capture
    for layer in selected:
        handles.append(layers[layer - 1].register_forward_hook(hook(layer)))
    token_counts = []
    try:
        for i, row in enumerate(rows):
            text = chat_text(proc, row['question'], len(row['images'][:4]), backbone=backbone)
            images = load_images(row['images'][:4], **image_rules(backbone))
            inputs = encode_batch(proc, [text], [images], backbone, return_offsets_mapping=True)
            offsets = inputs.pop('offset_mapping')[0].numpy()
            ids = inputs['input_ids'][0]
            qm = torch.from_numpy(question_mask(offsets, text, row['question']))
            im = ids == image_id
            attn = inputs['attention_mask'][0].bool()
            qm &= attn & ~im
            last = torch.zeros_like(attn)
            last[torch.where(attn)[0][-1]] = True
            masks.update(image=im & attn, question=qm, last=last)
            if any(not mask.any() for mask in masks.values()):
                raise ValueError(f"{row['id']}: empty token pooling mask")
            token_counts.append({k: int(v.sum()) for k, v in masks.items()})
            select(row['task'])
            routing = {k: torch.tensor([row[k]], device=device) for k in mb.ROUTING_KEYS}
            captured.clear()
            with mb.routing.use(**routing), torch.inference_mode():
                model(**inputs.to(device), use_cache=False)
            if set(captured) != set(storage):
                raise ValueError('not all requested decoder hooks ran')
            for key, value in captured.items():
                if not np.isfinite(value).all():
                    raise ValueError('nonfinite representation')
                storage[key].append(value)
            if (i + 1) % 20 == 0:
                print(f"{spec['name']}: {i + 1}/{len(rows)} probes", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    np.savez_compressed(out / 'features.npz', ids=np.array([r['id'] for r in rows]),
                        **{k: np.stack(v).astype(np.float32) for k, v in storage.items()})
    write_json(out / 'features.json', {'spec': spec, 'probe_md5': digest(probe_path), 'inputs': provenance([holdout, probe_path, spec_path]),
        'adapter_files': identities, 'code_commit': commit(), 'decoder': layer_name, 'depth': len(layers),
        'layers': selected, 'pooling': list(POOLS), 'token_counts': token_counts,
        'features_md5': digest(out / 'features.npz'), 'prompt': 'bare, forward only, no answer'})


def load_features(path, probe_path):
    path = Path(path)
    meta = read_json(path / 'features.json')
    if meta['probe_md5'] != digest(probe_path) or meta['features_md5'] != digest(path / 'features.npz'):
        raise ValueError(f'{path}: probe or feature hash mismatch')
    z = np.load(path / 'features.npz', allow_pickle=False)
    if z['ids'].tolist() != read_json(probe_path)['ids']:
        raise ValueError(f'{path}: probe ID ordering differs')
    features = {k: z[k].astype(float) for k in z.files if k != 'ids'}
    expected = {f'{pool}_L{layer}' for pool in POOLS for layer in meta['layers']}
    if set(features) != expected:
        raise ValueError('feature layer/pool coverage differs from metadata')
    if any(v.ndim != 2 or len(v) != len(z['ids']) or not np.isfinite(v).all() for v in features.values()):
        raise ValueError('invalid feature matrix')
    return meta, features


def analyze(index_path, probe_path, holdout, transfer_path, out, B=10000, seed=20261013):
    index, transfer = read_json(index_path), read_json(transfer_path)
    require_input_hash(transfer, holdout)
    rows = probe_rows(holdout, read_json(probe_path))
    cells = sorted({(r['task'], r['dataset']) for r in rows})
    out = fresh_dir(out)
    result = {'study': 'E7', 'inputs': provenance([index_path, probe_path, holdout, transfer_path]),
              'backbones': {}, 'primary': 'centroid input similarity vs raw C, target-cell permutations',
              'rsa_definition': 'Spearman of source/target centroid similarity profiles to fixed cell anchors; supplementary'}
    for backbone in BACKBONES:
        spec = index['backbones'][backbone]
        base_meta, base = load_features(spec['base'], probe_path)
        if base_meta['spec']['backbone'] != backbone or base_meta['spec']['name'] != 'base':
            raise ValueError('wrong base features')
        loaded = {}
        for name, paths in spec['models'].items():
            if set(paths) != {str(s) for s in SEEDS}:
                raise ValueError(f'{name}: three feature seeds required')
            loaded[name] = []
            for s in SEEDS:
                meta, f = load_features(paths[str(s)], probe_path)
                if (meta['spec']['backbone'] != backbone or meta['spec']['seed'] != s
                        or meta['spec']['name'] != name or set(f) != set(base)):
                    raise ValueError('feature arm, seed, backbone or layer/pool mismatch')
                loaded[name].append(f)
        required = {'shared', 'shared48', 'hydraT', 'hydraR'} | {'task' + s for s in SHORT.values()}
        from grouping import MODALITY_SLUGS, CLASSIFICATION_DATASET_SLUGS
        required |= {'mod' + s for s in MODALITY_SLUGS.values()} | {'ds' + s for s in CLASSIFICATION_DATASET_SLUGS.values()}
        if set(loaded) != required:
            raise ValueError(f'E7 model coverage differs: missing={required - set(loaded)}, extra={set(loaded) - required}')
        output = {'input_similarity': {}, 'shift_from_base': {}, 'seed_noise_ceiling': {},
                  'adapter_pairs': {}, 'cell_predictors': {f'{t}/{d}': {} for t, d in cells}}
        c = np.array([[transfer['backbones'][backbone]['cells'][f'{SHORT[t]}->{ct}/{ds}']['C']['mean']
                       for ct, ds in cells] for t in TASKS], dtype=float)
        for key, features in base.items():
            anchors = np.stack([features[[i for i, r in enumerate(rows) if (r['task'], r['dataset']) == cell]].mean(0)
                                for cell in cells])
            sources = np.stack([features[[i for i, r in enumerate(rows) if r['task'] == t]].mean(0) for t in TASKS])
            sim = np.array([[cosine(x, y) for y in anchors] for x in sources], float)
            anchor_sim = np.array([[cosine(x, y) for y in anchors] for x in anchors], float)
            rsa_sim = np.array([[spearman(x, y) for y in anchor_sim] for x in sim], float)
            output['input_similarity'][key] = {'centroid': cell_permutation(sim, c, B, seed),
                'rsa_profile': cell_permutation(rsa_sim, c, B, seed),
                'sources': list(TASKS), 'cells': [f'{t}/{d}' for t, d in cells],
                'centroid_values': [[finite(v) for v in row] for row in sim],
                'rsa_values': [[finite(v) for v in row] for row in rsa_sim]}
            for name, runs in loaded.items():
                output['shift_from_base'].setdefault(name, {})[key] = {
                    'cka': summary([cka(r[key], features) for r in runs]),
                    'rsa': summary([rsa(r[key], features) for r in runs])}
                output['seed_noise_ceiling'].setdefault(name, {})[key] = summary(
                    [cka(runs[i][key], runs[j][key]) for i, j in itertools.combinations(range(3), 2)])
            for i, j in itertools.combinations(range(6), 2):
                a, b = 'task' + SHORT[TASKS[i]], 'task' + SHORT[TASKS[j]]
                output['adapter_pairs'].setdefault(a + '/' + b, {})[key] = {
                    'cka': summary([cka(x[key], y[key]) for x, y in zip(loaded[a], loaded[b])]),
                    'C_forward': transfer['backbones'][backbone]['pairs'][SHORT[TASKS[i]] + '->' + SHORT[TASKS[j]]]['C'],
                    'C_reverse': transfer['backbones'][backbone]['pairs'][SHORT[TASKS[j]] + '->' + SHORT[TASKS[i]]]['C'],
                    'inference': 'descriptive only; six tasks are too few for an adapter-pair test'}
            for ci, (task, ds) in enumerate(cells):
                own = TASKS.index(task)
                indices = [i for i, r in enumerate(rows) if (r['task'], r['dataset']) == (task, ds)]
                other = [i for i in range(6) if i != own]
                adapter_values = [cka(loaded['task' + SHORT[task]][s][key][indices],
                                      loaded['task' + SHORT[TASKS[j]]][s][key][indices])
                                  for s in range(3) for j in other]
                output['cell_predictors'][f'{task}/{ds}'][key] = {
                    'input_similarity': finite(np.mean(sim[other, ci])),
                    'adapter_similarity': summary(adapter_values)['mean']}
        for pool in POOLS:
            for measure in ('centroid', 'rsa_profile'):
                family = {k: v[measure]['p_permutation'] for k, v in output['input_similarity'].items()
                          if k.startswith(pool + '_') and v[measure]['p_permutation'] is not None}
                for key, p in bs.holm(family).items():
                    output['input_similarity'][key][measure]['p_holm_layers'] = p
        result['backbones'][backbone] = output
    write_json(out / 'representations.json', result)
    return result


def head_similarity(state_path, mapping_path, out):
    """Explicit mapping avoids guessing Veronika's checkpoint parameter names.

    Mapping: {mode: hydraT|hydraR, layers: {layer: {task: [parameter_names]}}}.
    Only B_t heads for hydraT and A_t heads for hydraR are selected by the reviewed map.
    """
    from safetensors.numpy import load_file
    state, mapping = load_file(state_path), read_json(mapping_path)
    if mapping['mode'] not in ('hydraT', 'hydraR'):
        raise ValueError('expected hydraT or hydraR')
    result = {'mode': mapping['mode'], 'factor': 'B' if mapping['mode'] == 'hydraT' else 'A',
              'inputs': provenance([state_path, mapping_path]), 'layers': {}}
    for layer, tasks in mapping['layers'].items():
        if set(tasks) != set(TASKS):
            raise ValueError('head map must cover all six tasks')
        def select(entry):
            if isinstance(entry, str):
                return state[entry].astype(float).ravel()
            value = state[entry['parameter']]
            return np.take(value, entry['index'], axis=entry.get('axis', 0)).astype(float).ravel()
        vectors = {t: np.concatenate([select(k) for k in keys]) for t, keys in tasks.items()}
        if len({len(v) for v in vectors.values()}) != 1:
            raise ValueError('heads differ in size')
        result['layers'][layer] = {SHORT[a] + '/' + SHORT[b]: cosine(vectors[a], vectors[b])
                                   for a, b in itertools.combinations(TASKS, 2)}
    write_json(out, result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='cmd', required=True)
    probe = sub.add_parser('probe')
    probe.add_argument('--holdout', required=True)
    probe.add_argument('--out', required=True)
    extract_p = sub.add_parser('extract')
    for name in ('holdout', 'probe', 'spec', 'out'):
        extract_p.add_argument('--' + name, required=True)
    analysis = sub.add_parser('analyze')
    for name in ('index', 'probe', 'holdout', 'transfer', 'out'):
        analysis.add_argument('--' + name, required=True)
    analysis.add_argument('--B', type=int, default=10000)
    head = sub.add_parser('heads')
    for name in ('state', 'mapping', 'out'):
        head.add_argument('--' + name, required=True)
    a = p.parse_args()
    if a.cmd == 'probe':
        make_probe(a.holdout, a.out)
    elif a.cmd == 'extract':
        extract(a.holdout, a.probe, a.spec, a.out)
    elif a.cmd == 'analyze':
        analyze(a.index, a.probe, a.holdout, a.transfer, a.out, a.B)
    else:
        head_similarity(a.state, a.mapping, a.out)


if __name__ == '__main__':
    main()
