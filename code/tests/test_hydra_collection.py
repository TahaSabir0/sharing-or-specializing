"""Small smoke-evidence collection must exclude training data and model weights."""
import json
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
from collect_hydra_smoke import REQUIRED, collect
from hydra_smoke_queue import CODE, verify


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


def queue_fixture(tmp_path, backbone="q35", mode="hydraT"):
    stored, active, layers = {
        ("q35", "hydraT"): (78716928, 21639168, 248),
        ("q35", "hydraR"): (72757248, 21639168, 248),
        ("mg15", "hydraT"): (55287808, 14901248, 238),
        ("mg15", "hydraR"): (49020928, 14901248, 238)}[backbone, mode]
    rid = f"j26_{backbone}_{mode}_full_smoke_veronika_test"
    run = tmp_path / rid
    (run / "eval_smoke").mkdir(parents=True)
    (run / "runtime.json").write_text(json.dumps(dict(optimizer_steps=50,
        trainable_params=stored, stored_params=stored, active_params_per_example=active,
        s_per_it=15, peak_mem_reserved_gb=18)))
    (run / "method_config.json").write_text(json.dumps(dict(method="hydra", cfg=dict(mode=mode), n_params=stored)))
    (run / "trainer_log.jsonl").write_text('{"loss": 1.5}\n{"loss": 0.7}\n')
    tasks = json.loads((CODE / "data/label_ids.json").read_text())["task"]
    (run / "eval_smoke/metrics.json").write_text(json.dumps(dict(tasks={task: dict(n=10,
        malformed_rate=0, truncation_rate=0, errors=0) for task in tasks})))
    summary = tmp_path / "summary.log"
    summary.write_text(f"[smoke] veronika: {rid}\ntraining exit code: 0   eval: 0\n"
        f"5. routing_stats.npz: {{'sequences': 60, 'layers': {layers}, 'experts': 6}}\n")
    return rid, run, summary


@pytest.mark.parametrize("backbone", ["q35", "mg15"])
@pytest.mark.parametrize("mode", ["hydraT", "hydraR"])
def test_smoke_queue_accepts_complete_gates_for_both_backbones(tmp_path, backbone, mode):
    rid, _, summary = queue_fixture(tmp_path, backbone, mode)
    assert verify(summary, tmp_path, backbone, mode) == rid


@pytest.mark.parametrize("failure", ["eval", "params", "loss", "tasks", "format", "routing"])
def test_smoke_queue_refuses_to_advance_on_incomplete_gate(tmp_path, failure):
    _, run, summary = queue_fixture(tmp_path)
    if failure == "eval":
        summary.write_text(summary.read_text().replace("eval: 0", "eval: 1"))
    elif failure == "params":
        (run / "runtime.json").write_text('{}')
    elif failure == "loss":
        (run / "trainer_log.jsonl").write_text('{"loss": 1.5}\n{"loss": 2.0}\n')
    elif failure in ("tasks", "format"):
        path = run / "eval_smoke/metrics.json"
        met = json.loads(path.read_text())
        if failure == "tasks":
            met["tasks"].pop("counting")
        else:
            met["tasks"]["counting"]["errors"] = 1
        path.write_text(json.dumps(met))
    else:
        summary.write_text(summary.read_text().replace("'experts': 6", "'experts': 1"))
    with pytest.raises(ValueError):
        verify(summary, tmp_path, "q35", "hydraT")
