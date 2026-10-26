"""CPU-only integrity checks shared by E4/E5 execution and analysis."""
import hashlib
import json
import math
from pathlib import Path

REPORT = 'report_generation'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]


def digest(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def index_rows(rows):
    result = {}
    for row in rows:
        if row['id'] in result:
            raise ValueError(f"duplicate question id: {row['id']}")
        result[row['id']] = row
    return result


def validate_predictions(predictions, deck, exact=True):
    """Reject missing/duplicate/wrong-GT rows; retain malformed model answers as results."""
    predictions, reference = index_rows(predictions), index_rows(deck)
    missing = set(reference) - set(predictions)
    extra = set(predictions) - set(reference)
    if missing or (exact and extra):
        raise ValueError(f'question coverage mismatch: {len(missing)} missing, {len(extra)} extra')
    for key, row in reference.items():
        p = predictions[key]
        for pk, dk in [('TaskType', 'task'), ('dataset', 'dataset'), ('gt_answer', 'answer'),
                       ('Question', 'question'), ('ImageName', 'images')]:
            if p[pk] != row[dk]:
                raise ValueError(f'{key}: {pk} does not match canonical deck')
        if str(p.get('Answer', '')).startswith('Error:'):
            raise ValueError(f'{key}: inference error; cannot mark this evaluation complete')
        if 'truncated' not in p or 'n_new_tokens' not in p:
            raise ValueError(f'{key}: missing decoding diagnostics')
    return [dict(predictions[k], modality=r['modality']) for k, r in sorted(reference.items())]


def crimson_scores(paths, expected_ids):
    scores = {}
    for path in paths:
        for r in read_jsonl(path):
            key, score = r['sample_key'], r['crimson_score']
            if key in scores:
                raise ValueError(f'duplicate CRIMSON id {key}')
            if not isinstance(score, (float, int)) or not math.isfinite(score):
                raise ValueError(f'invalid CRIMSON score for {key}')
            scores[key] = score
    if set(scores) != set(expected_ids):
        raise ValueError('CRIMSON question coverage differs from report predictions')
    return scores
