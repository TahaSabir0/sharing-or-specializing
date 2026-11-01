"""E6-only dependencies must not block the E3 audit."""
import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    'phase_b_preflight', Path(__file__).parents[1] / 'train/phase_b_preflight.py')
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)


def test_missing_router_package_only_blocks_full_audit(tmp_path, monkeypatch, capsys):
    for name in ('j26_train_mix.jsonl', 'j26_holdout.jsonl', 'dataset_info.json',
                 'GROUP_MANIFEST.json'):
        (tmp_path / name).touch()
    monkeypatch.setattr(preflight.importlib.util, 'find_spec',
                        lambda name: None if name == 'gliclass' else object())
    monkeypatch.setattr(preflight.importlib.metadata, 'version', lambda name: 'test')
    assert preflight.check(tmp_path, tmp_path, experiment='e3')
    assert not preflight.check(tmp_path, tmp_path)
    assert 'missing package: gliclass' in capsys.readouterr().out


def test_e3_still_rejects_missing_core_dependency(tmp_path, monkeypatch):
    for name in ('j26_train_mix.jsonl', 'j26_holdout.jsonl', 'dataset_info.json'):
        (tmp_path / name).touch()
    monkeypatch.setattr(preflight.importlib.util, 'find_spec',
                        lambda name: None if name == 'transformers' else object())
    monkeypatch.setattr(preflight.importlib.metadata, 'version', lambda name: 'test')
    assert not preflight.check(tmp_path, tmp_path / 'absent_groups', experiment='e3')
