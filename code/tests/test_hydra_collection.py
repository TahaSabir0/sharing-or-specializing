"""Small smoke-evidence collection must exclude training data and model weights."""
import json
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
from collect_hydra_smoke import REQUIRED, collect


def test_collect_only_small_evidence_with_hashes_and_preserve_existing_output(tmp_path):
    rid = "j26_q35_hydraT_full_768_s1337_smoke_veronika_test"
    root, out = tmp_path / "runs", tmp_path / "evidence.tar.gz"
    run = root / rid
    for rel in REQUIRED:
        path = run / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture evidence")
    for rel in ("method_state.safetensors", "eval_smoke/all_predictions.json", "train.jsonl"):
        (run / rel).write_text("must not be collected")
    manifest = collect(root, [rid], out)
    with tarfile.open(out) as archive:
        assert set(archive.getnames()) == {f"{rid}/{rel}" for rel in REQUIRED} | {"collection.json"}
        assert json.load(archive.extractfile("collection.json")) == manifest
    original = out.read_bytes()
    with pytest.raises(ValueError, match="archive exists"):
        collect(root, [rid], out)
    assert out.read_bytes() == original


def test_collection_requires_complete_evidence_before_writing_archive(tmp_path):
    out = tmp_path / "missing.tar.gz"
    with pytest.raises(ValueError, match="missing smoke evidence"):
        collect(tmp_path, ["j26_q35_hydraT_smoke_missing"], out)
    assert not out.exists()
