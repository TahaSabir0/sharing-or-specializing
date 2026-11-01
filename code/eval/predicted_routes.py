"""E6 route-file contract. True labels remain scoring targets, never router inputs."""
from pathlib import Path
from group_artifacts import read_json, digest, index_rows

LABELS = read_json(Path(__file__).resolve().parents[1] / 'data/label_ids.json')


def load_routes(path, data_path, rows, axis):
    value = read_json(path)
    if value['axis'] != axis or value['data_md5'] != digest(data_path):
        raise ValueError('route axis or evaluation deck hash mismatch')
    if value.get('input') != ('question only' if axis == 'task' else 'image pixels only'):
        raise ValueError('router must use only question text or only image pixels')
    mapping = index_rows(value['predictions'])
    if set(mapping) != {r['id'] for r in rows}:
        raise ValueError('route IDs must match evaluation deck exactly')
    for row in mapping.values():
        if row['label'] not in LABELS[axis]:
            raise ValueError(f"unknown predicted {axis}: {row['label']}")
        if not 0 <= row['confidence'] <= 1:
            raise ValueError('invalid route confidence')
    return mapping, value


def routing_ids(row, route, axis):
    result = {k: row[k] for k in ('task_id', 'modality_id', 'dataset_id')}
    result[axis + '_id'] = LABELS[axis][route['label']]
    return result


def accuracy(rows, predictions, axis):
    mapping = index_rows(predictions)
    if set(mapping) != {r['id'] for r in rows}:
        raise ValueError('route IDs differ')
    labels = sorted({r[axis] for r in rows})
    confusion = {a: {b: 0 for b in LABELS[axis]} for a in labels}
    for r in rows:
        confusion[r[axis]][mapping[r['id']]['label']] += 1
    per = {a: {'n': sum(confusion[a].values()),
               'accuracy': confusion[a][a] / sum(confusion[a].values())} for a in labels}
    return {'accuracy': sum(confusion[a][a] for a in labels) / len(rows),
            'balanced_accuracy': sum(v['accuracy'] for v in per.values()) / len(per),
            'per_label': per, 'confusion': confusion}
