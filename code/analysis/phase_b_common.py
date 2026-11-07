"""Shared CPU utilities for E3, E6, E7 and E8. No model or parser dependencies."""
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / 'eval'))
from group_artifacts import read_json, read_jsonl, digest, validate_predictions, crimson_scores

SEEDS = (1337, 2026, 42)
BACKBONES = ('q35', 'mg15')
TASKS = ('classification', 'multi-label classification', 'detection', 'counting',
         'regression', 'report_generation')
SHORT = dict(zip(TASKS, ('cls', 'ml', 'det', 'count', 'reg', 'report')))
STRUCTURE = dict(zip(TASKS, ('label', 'label-set', 'boxes', 'integer', 'scalar', 'free text')))
# Transparent, pre-result descriptors of the answer representation.  Pairwise output-structure
# similarity is Jaccard overlap on these sets; this is graded while avoiding an arbitrary ordinal
# rank over otherwise incomparable formats.
STRUCTURE_FEATURES = {
    'classification': ('categorical', 'atomic', 'discrete', 'short'),
    'multi-label classification': ('categorical', 'set', 'discrete', 'short'),
    'detection': ('numeric', 'set', 'continuous', 'spatial', 'short'),
    'counting': ('numeric', 'atomic', 'discrete', 'short'),
    'regression': ('numeric', 'atomic', 'continuous', 'short'),
    'report_generation': ('language', 'sequence', 'free-form', 'long'),
}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def fresh_dir(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f'{path}: use a fresh output directory; existing work is preserved')
    path.mkdir(parents=True, exist_ok=True)
    return path


def provenance(paths):
    return [{'path': str(Path(p).resolve()), 'md5': digest(p)} for p in paths]


def require_input_hash(artifact, path):
    """Permit relocation, but never mix analyses of different frozen decks."""
    expected = digest(path)
    if not any(item['md5'] == expected for item in artifact.get('inputs', [])):
        raise ValueError(f'{path}: input hash is absent from prerequisite artifact provenance')


def commit():
    return subprocess.run(['git', '-C', str(CODE.parent), 'rev-parse', 'HEAD'],
                          capture_output=True, text=True).stdout.strip()


def finite(value):
    return float(value) if value is not None and math.isfinite(value) else None


def summary(values):
    """Never silently drop a failed/nonfinite seed."""
    values = [finite(v) for v in values]
    if None in values:
        return {'values': values, 'mean': None, 'sd': None, 'reason': 'undefined metric in a seed'}
    return {'values': values, 'mean': float(np.mean(values)),
            'sd': float(np.std(values, ddof=1)) if len(values) > 1 else None}


def rank_correlation(x, y, B=10000, seed=20261013):
    from scipy.stats import rankdata
    x, y = np.asarray(x, float), np.asarray(y, float)
    if B < 1 or len(x) != len(y) or len(x) < 3 or not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError('correlation needs >=3 paired finite observations and B > 0')
    x, y = rankdata(x), rankdata(y)
    x, y = x - x.mean(), y - y.mean()
    den = np.linalg.norm(x) * np.linalg.norm(y)
    if den == 0:
        return {'rho': None, 'p_permutation': None, 'n': len(x), 'reason': 'constant variable'}
    rho = float(x @ y / den)
    rng = np.random.default_rng(seed)
    hits = sum(abs(x @ rng.permutation(y) / den) >= abs(rho) - 1e-12 for _ in range(B))
    return {'rho': rho, 'p_permutation': (hits + 1) / (B + 1), 'n': len(x), 'B': B}


def load_eval(path, deck, backbone, seed=None, arm=None):
    """Analysis inputs must be normalized and match the canonical questions, not just counts."""
    import bootstrap as bs
    path = Path(path)
    meta = read_json(path / 'run_meta.json')
    if meta.get('surface') != 'holdout' or meta.get('backbone') != backbone:
        raise ValueError(f'{path}: wrong surface/backbone')
    if seed is not None and meta.get('seed') != seed:
        raise ValueError(f'{path}: wrong seed')
    if arm is not None and meta.get('arm') != arm:
        raise ValueError(f'{path}: wrong arm (expected {arm})')
    rows = validate_predictions(read_json(path / 'all_predictions.json'), deck)
    reports = [r['id'] for r in deck if r['task'] == 'report_generation']
    if reports:
        crimson_scores([path / 'crimson_per_sample.jsonl'], reports,
                       allow_unscored=meta.get('arm') in ('zsbare', 'zsprompt') or (
                           str(meta.get('arm', '')).startswith('transfer_')
                           and meta.get('arm') != 'transfer_report'))
    return bs.load_run(path), rows
