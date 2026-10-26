import importlib.util
import json
import sys
from pathlib import Path


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "data"))
from build_group_decks import build_group_decks  # noqa: E402
from grouping import CLASSIFICATION_DATASET_SLUGS, MODALITY_SLUGS  # noqa: E402


def _row(i, task, modality, dataset):
    return {
        "id": f"id-{i}",
        "task": task,
        "modality": modality,
        "dataset": dataset,
        "modality_id": list(MODALITY_SLUGS).index(modality),
        "dataset_id": i,
        "question": "q",
        "answer": "a",
        "images": [f"/images/{i}.png"],
        "conversations": [],
    }


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_group_decks_partition_frozen_rows(tmp_path):
    source, out = tmp_path / "source", tmp_path / "out"
    source.mkdir()
    train = []
    i = 0
    # Cover all ten classification datasets and all seven modalities. Some modality
    # groups also get a non-classification row to verify E4/E5 differ as designed.
    modalities = list(MODALITY_SLUGS)
    for dataset in CLASSIFICATION_DATASET_SLUGS:
        train.append(_row(i, "classification", modalities[i % len(modalities)], dataset))
        i += 1
    for modality in modalities:
        train.append(_row(i, "regression", modality, "iugc"))
        i += 1
    dev = [dict(row) for row in train]
    holdout = [dict(row, id=f"holdout-{row['id']}") for row in train]
    _write(source / "j26_train_mix.jsonl", train)
    _write(source / "j26_devselect.jsonl", dev)
    _write(source / "j26_holdout.jsonl", holdout)

    manifest = build_group_decks(source, out)

    assert len(manifest["decks"]) == 3 * (len(MODALITY_SLUGS) + len(CLASSIFICATION_DATASET_SLUGS))
    mod_train = [v for v in manifest["decks"].values()
                 if v["kind"] == "train" and v["group_type"] == "modality"]
    ds_train = [v for v in manifest["decks"].values()
                if v["kind"] == "train" and v["group_type"] == "classification_dataset"]
    assert sum(v["rows"] for v in mod_train) == len(train)
    assert sum(v["rows"] for v in ds_train) == len(CLASSIFICATION_DATASET_SLUGS)
    assert all(set(v["task_counts"]) == {"classification"} for v in ds_train)
    mod_holdout = [v for v in manifest["decks"].values()
                   if v["kind"] == "holdout" and v["group_type"] == "modality"]
    assert sum(v["rows"] for v in mod_holdout) == len(holdout)
    info = json.loads((out / "dataset_info.json").read_text(encoding="utf-8"))
    assert set(info) == set(manifest["decks"])


def test_generated_config_matrix_is_complete_and_importable():
    spec = importlib.util.spec_from_file_location(
        "generate_e4_e5", CODE / "configs" / "generate_e4_e5.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    configs = list(module.expected_configs())
    assert len(configs) == (7 + 10) * 2 * 3
    texts = [text for _, text in configs]
    run_ids = [next(line for line in text.splitlines() if line.startswith("run_id: ")) for text in texts]
    assert len(run_ids) == len(set(run_ids))
    assert all("owner: alex" in text and "overrides: {}" in text for text in texts)
