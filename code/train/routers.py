#!/usr/bin/env python3
"""E6 question-only GLiClass and image-only ResNet18 routers.

Training uses only frozen A-union-B; holdout is read solely for leakage checks.
No selection/early stopping. Every seed trains its own routers, shared by both backbones.
Weights must already be local: this script never installs packages or downloads models.
GLiClass API: https://docs.knowledgator.com/docs/frameworks/gliclass/training/
"""
import argparse
import inspect
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / 'analysis'))
sys.path.insert(0, str(CODE / 'eval'))
from phase_b_common import read_json, read_jsonl, write_json, fresh_dir, provenance, digest, commit, SEEDS
from predicted_routes import LABELS, accuracy


def image_key(path):
    return str(path).replace('\\', '/').split('FLARE-MLLM-2D/', 1)[-1]


def audit_split(train, holdout):
    from group_artifacts import index_rows
    ti, hi = index_rows(train), index_rows(holdout)
    train_images = {image_key(p) for r in train for p in r['images']}
    hold_images = {image_key(p) for r in holdout for p in r['images']}
    if ti.keys() & hi.keys() or train_images & hold_images:
        raise ValueError('router training/holdout ID or image overlap')
    for axis in ('task', 'modality'):
        if set(r[axis] for r in train) != set(LABELS[axis]):
            raise ValueError(f'training data must cover all {axis} labels')
    return {'train_questions': len(train), 'train_unique_images': len(train_images),
            'holdout_questions': len(holdout), 'overlapping_images': 0, 'overlapping_ids': 0}


def unique_images(rows):
    images = {}
    for row in rows:
        for path in row['images']:
            key = image_key(path)
            if key in images and images[key][1] != row['modality']:
                raise ValueError(f'{path}: inconsistent modality labels')
            images[key] = (path, row['modality'])
    return [images[k] for k in sorted(images)]


def initialize(seed, device):
    import torch
    if device.startswith('cuda') and os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
        raise ValueError('GPU router training/inference requires CUDA_VISIBLE_DEVICES=1')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return torch


def task_collator(collator):
    """Older GLiClass collators omit 0-D tensor labels; provide Python scalars.

    Keep the public collator for token padding. Metadata strings are not model
    inputs: some versions forward unknown kwargs directly into the encoder.
    """
    def collate(items):
        prepared = []
        for item in items:
            value = dict(item)
            label = value['labels']
            if hasattr(label, 'ndim') and label.ndim == 0:
                value['labels'] = int(label.item())
            prepared.append(value)
        batch = collator(prepared)
        batch.pop('labels_text', None)
        batch.pop('input_texts', None)
        return batch
    return collate


def train_task(train, initial, out, seed, device, epochs=3):
    """Public GLiClass dataset/collator, explicit PyTorch loop, all six labels each time."""
    torch = initialize(seed, device)
    from gliclass import GLiClassModel
    from gliclass import data_processing as gd
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader
    model = GLiClassModel.from_pretrained(str(initial), local_files_only=True).to(device)
    model.config.problem_type = 'single_label_classification'
    if getattr(model.config, 'architecture_type', 'uni-encoder') != 'uni-encoder':
        raise ValueError('the pinned task-router recipe requires a GLiClass uni-encoder initial model')
    tokenizer = AutoTokenizer.from_pretrained(str(initial), local_files_only=True)
    labels = list(LABELS['task'])
    records = [{'text': r['question'], 'all_labels': list(labels), 'true_labels': [r['task']]} for r in train]
    params = inspect.signature(gd.GLiClassDataset).parameters
    kwargs = dict(max_length=512, problem_type='single_label_classification')
    if 'augmentation_config' in params:
        kwargs['augmentation_config'] = gd.AugmentationConfig(enabled=False)
    if 'augment_config' in params:
        kwargs['augment_config'] = gd.AugmentationConfig(enabled=False)
    if 'shuffle_labels' in params:
        kwargs['shuffle_labels'] = False
    if 'prompt_first' in params:
        kwargs['prompt_first'] = getattr(model.config, 'prompt_first', False)
    if 'add_description' in params:
        kwargs['add_description'] = False
    dataset = gd.GLiClassDataset(records, tokenizer, **kwargs)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0,
                        collate_fn=task_collator(gd.DataCollatorWithPadding(device=device)))
    optim = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=.01)
    losses = []
    model.train()
    for epoch in range(epochs):
        total, n = 0., 0
        for batch in loader:
            batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}
            optim.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise ValueError('nonfinite router training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optim.step()
            total += float(loss.detach())
            n += 1
        losses.append(total / n)
        print(f'GLiClass epoch {epoch + 1}/{epochs}: loss={losses[-1]:.6f}', flush=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    return {'architecture': 'GLiClass', 'epochs': epochs, 'batch_size': 16, 'lr': 2e-5,
            'weight_decay': .01, 'max_length': 512, 'loss_by_epoch': losses}


def image_transform(size, augment=False):
    from torchvision import transforms as T
    return T.Compose([T.Resize((size, size))] + ([T.RandomHorizontalFlip()] if augment else []) +
                     [T.ToTensor(), T.Normalize([.485, .456, .406], [.229, .224, .225])])


def train_modality(train, initial, out, seed, device, epochs=5, size=2048):
    torch = initialize(seed, device)
    from torchvision.models import resnet18
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image
    images = unique_images(train)
    transform = image_transform(size, True)
    class Images(Dataset):
        def __len__(self):
            return len(images)
        def __getitem__(self, i):
            path, label = images[i]
            with Image.open(path) as im:
                pixels = transform(im.convert('RGB'))
            return pixels, LABELS['modality'][label]
    model = resnet18(weights=None)
    model.load_state_dict(torch.load(initial, map_location='cpu', weights_only=True))
    model.fc = torch.nn.Linear(model.fc.in_features, len(LABELS['modality']))
    model.to(device)
    counts = np.bincount([LABELS['modality'][label] for _, label in images], minlength=7)
    weights = torch.tensor(len(images) / (7 * counts), dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=.05)
    loader = DataLoader(Images(), batch_size=16, shuffle=True, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
    losses = []
    model.train()
    for epoch in range(epochs):
        total, n = 0., 0
        for pixels, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=device.startswith('cuda')):
                logits = model(pixels.to(device))
                loss = criterion(logits.float(), targets.to(device))
            if not torch.isfinite(loss):
                raise ValueError('nonfinite modality training loss')
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            n += 1
        losses.append(total / n)
        print(f'ResNet18 epoch {epoch + 1}/{epochs}: loss={losses[-1]:.6f}', flush=True)
    torch.save(model.cpu().state_dict(), out / 'resnet18.pt')
    return {'architecture': 'ResNet18', 'epochs': epochs, 'image_size': size, 'batch_size': 16,
            'lr': 1e-4, 'weight_decay': .01, 'label_smoothing': .05,
            'class_weights': weights.cpu().tolist(), 'loss_by_epoch': losses,
            'multi_image_rule': 'mean softmax probabilities over first four images, then argmax'}


def train_router(axis, train_path, holdout_path, initial, out, seed, device):
    train, holdout = read_jsonl(train_path), read_jsonl(holdout_path)
    audit = audit_split(train, holdout)
    if not Path(initial).exists():
        raise FileNotFoundError(f'local pretrained weights required: {initial}')
    out = fresh_dir(out)
    recipe = (train_task if axis == 'task' else train_modality)(train, initial, out, seed, device)
    files = [p for p in sorted(out.rglob('*')) if p.is_file()]
    from importlib.metadata import version
    versions = {n: version(n) for n in (['torch', 'transformers', 'gliclass'] if axis == 'task' else ['torch', 'torchvision'])}
    write_json(out / 'router.json', {'axis': axis, 'seed': seed, 'labels': LABELS[axis], 'recipe': recipe,
        'split_audit': audit, 'inputs': provenance([train_path, holdout_path]),
        'initial_weights': provenance([initial] if Path(initial).is_file() else
                                      [p for p in sorted(Path(initial).rglob('*')) if p.is_file()]),
        'trained_files': provenance(files), 'versions': versions, 'code_commit': commit(),
        'input': 'question only' if axis == 'task' else 'image pixels only'})


def predict(router, data, out, device='cpu'):
    router, data = Path(router), Path(data)
    meta = read_json(router / 'router.json')
    torch = initialize(meta['seed'], device)
    for item in meta['trained_files']:
        # Allow the whole artifact directory to move to another machine.
        local = router / Path(item['path']).name
        if digest(local) != item['md5']:
            raise ValueError('router checkpoint changed')
    rows = read_jsonl(data)
    predictions = []
    axis = meta['axis']
    if axis == 'task':
        from gliclass import GLiClassModel, ZeroShotClassificationPipeline
        from transformers import AutoTokenizer
        model = GLiClassModel.from_pretrained(str(router), local_files_only=True)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(str(router), local_files_only=True)
        pipe = ZeroShotClassificationPipeline(model, tokenizer, classification_type='single-label', device=device)
        for start in range(0, len(rows), 16):
            batch = rows[start:start + 16]
            output = pipe([r['question'] for r in batch], list(LABELS['task']))
            for r, candidates in zip(batch, output):
                if isinstance(candidates, dict):
                    candidates = [candidates]
                best = max(candidates, key=lambda v: v['score'])
                predictions.append({'id': r['id'], 'label': best['label'], 'confidence': float(best['score'])})
    else:
        from torchvision.models import resnet18
        from PIL import Image
        model = resnet18(weights=None, num_classes=7)
        model.load_state_dict(torch.load(router / 'resnet18.pt', map_location='cpu', weights_only=True))
        model.to(device).eval()
        transform = image_transform(meta['recipe']['image_size'])
        labels = {v: k for k, v in LABELS['modality'].items()}
        cache = {}
        for r in rows:
            probabilities = []
            for path in r['images'][:4]:
                if path not in cache:
                    with Image.open(path) as im:
                        pixels = transform(im.convert('RGB')).unsqueeze(0).to(device)
                    with torch.inference_mode():
                        cache[path] = model(pixels).float().softmax(-1)[0].cpu()
                probabilities.append(cache[path])
            prob = torch.stack(probabilities).mean(0)
            confidence, idx = prob.max(0)
            predictions.append({'id': r['id'], 'label': labels[int(idx)], 'confidence': float(confidence)})
    if Path(out).exists():
        raise ValueError(f'{out}: routes already exist; use a fresh output file')
    write_json(out, {'axis': axis, 'input': meta['input'], 'seed': meta['seed'], 'data_md5': digest(data),
                     'router': provenance([router / 'router.json']), 'predictions': predictions,
                     'accuracy': accuracy(rows, predictions, axis)})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('train')
    t.add_argument('--axis', choices=['task', 'modality'], required=True)
    t.add_argument('--train', required=True)
    t.add_argument('--holdout', required=True)
    t.add_argument('--initial', required=True)
    t.add_argument('--out', required=True)
    t.add_argument('--seed', type=int, choices=SEEDS, required=True)
    t.add_argument('--device', default='cuda:0')
    pr = sub.add_parser('predict')
    pr.add_argument('--router', required=True)
    pr.add_argument('--data', required=True)
    pr.add_argument('--out', required=True)
    pr.add_argument('--device', default='cpu')
    a = p.parse_args()
    if a.cmd == 'train':
        train_router(a.axis, a.train, a.holdout, a.initial, a.out, a.seed, a.device)
    else:
        predict(a.router, a.data, a.out, a.device)


if __name__ == '__main__':
    main()
