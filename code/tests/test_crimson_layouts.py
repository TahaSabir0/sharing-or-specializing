"""Exercise the merged shell launcher's layout selection without running a judge."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest


def test_existing_and_new_crimson_layouts_are_preserved(tmp_path):
    bash = Path('C:/Program Files/Git/bin/bash.exe') if os.name == 'nt' else Path(shutil.which('bash') or '/missing')
    if not bash.is_file():
        pytest.skip('Bash required for launcher integration test')
    code = Path(__file__).resolve().parents[1]
    scripts = tmp_path / 'code/eval'
    scripts.mkdir(parents=True)
    shutil.copyfile(code / 'eval/run_crimson_chain.sh', scripts / 'run_crimson_chain.sh')
    (scripts / 'crimson_box.sh').write_text('#!/bin/bash\nprintf "%s\\n" "$1" >> "$CAPTURE"\n', encoding='utf-8')
    old, fresh, pairs_only = [tmp_path / x for x in ('existing', 'fresh layout', 'pairs only')]
    (old / 'crimson').mkdir(parents=True)
    fresh.mkdir()
    pairs_only.mkdir()
    (old / 'crimson/reportgen_pairs.jsonl').write_text('{}\n')
    (old / 'reportgen_pairs.jsonl').write_text('{}\n')
    (old / 'run_meta.json').write_text('{}')
    (fresh / 'reportgen_pairs.jsonl').write_text('{}\n')
    (fresh / 'run_meta.json').write_text('{}')
    (pairs_only / 'reportgen_pairs.jsonl').write_text('{}\n')
    jobs = tmp_path / 'jobs.txt'
    # The launcher changes to this fixture's repo. Relative paths also avoid
    # MSYS mkdir -p probing protected Windows profile ancestors in sandboxed tests.
    jobs.write_text(''.join(f'job{i} {p.name}\n' for i, p in enumerate((old, fresh, pairs_only))), encoding='utf-8')
    capture = tmp_path / 'capture.txt'
    env = dict(os.environ, CAPTURE=capture.name, JOURNAL_RUNS='.', GPU='1')
    result = subprocess.run([str(bash), str(scripts / 'run_crimson_chain.sh'), str(jobs)],
                            env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert capture.read_text().splitlines() == ['existing/crimson/reportgen_pairs.jsonl',
                                               'fresh layout', 'pairs only/crimson/reportgen_pairs.jsonl'], result.stdout + result.stderr
    assert (pairs_only / 'crimson/reportgen_pairs.jsonl').is_file()
