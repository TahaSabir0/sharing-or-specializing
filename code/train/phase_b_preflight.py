#!/usr/bin/env python3
"""Read-only CPU dependency/asset audit. Does not install packages or start model/GPU jobs."""
import argparse
import importlib.util
import importlib.metadata
import json
from pathlib import Path


def check(data, groups, initial_task=None, initial_modality=None, experiment='all'):
    failures, versions = [], {}
    packages = {'numpy': 'numpy', 'scipy': 'scipy', 'sklearn': 'scikit-learn', 'matplotlib': 'matplotlib',
                'yaml': 'PyYAML', 'PIL': 'Pillow', 'torch': 'torch', 'torchvision': 'torchvision',
                'transformers': 'transformers', 'peft': 'peft', 'bitsandbytes': 'bitsandbytes',
                'gliclass': 'gliclass', 'safetensors': 'safetensors'}
    if experiment == 'e3':
        packages.pop('gliclass')
        packages.pop('torchvision')
    for module, package in packages.items():
        if importlib.util.find_spec(module) is None:
            failures.append('missing package: ' + package)
        else:
            versions[package] = importlib.metadata.version(package)
    for path in [Path(data) / n for n in ('j26_train_mix.jsonl', 'j26_holdout.jsonl', 'dataset_info.json')]:
        if not path.is_file():
            failures.append('missing source: ' + str(path))
    for name in (() if experiment == 'e3' else ('dataset_info.json', 'GROUP_MANIFEST.json')):
        if not (Path(groups) / name).is_file():
            failures.append('missing grouped deck metadata: ' + str(Path(groups) / name))
    if experiment != 'e3' and initial_task and not (Path(initial_task) / 'config.json').is_file():
        failures.append('GLiClass initial directory has no config.json')
    if experiment != 'e3' and initial_modality and not Path(initial_modality).is_file():
        failures.append('ResNet ImageNet state dictionary is missing')
    print(json.dumps({'experiment': experiment, 'versions': versions, 'failures': failures,
                      'scope': 'CPU package/asset presence only; no real-model compatibility claim'}, indent=2))
    return not failures


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--groups', required=True)
    p.add_argument('--gliclass-initial')
    p.add_argument('--resnet-initial')
    p.add_argument('--experiment', choices=('all', 'e3'), default='all')
    a = p.parse_args()
    raise SystemExit(0 if check(a.data, a.groups, a.gliclass_initial, a.resnet_initial, a.experiment) else 1)
