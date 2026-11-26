#!/usr/bin/env python3
"""Job files for F4 v2 forced-format decoding (results/format_analysis_plan.md, F4 v2).

The 14 negative-transfer cells of Table VIII (mu < -0.05), each forced for its source TaskLoRA, the ZS-prompt
base and Shared, three seeds. One run per (backbone, arm, seed) covering all its forced targets.
  python code/eval/jobs/f4/plan_f4.py        writes smoke.txt and holdout_s{seed}.txt next to this file
Run with EVAL_PY=code/eval/forced_eval.py bash code/eval/run_eval_chain.sh <file> (launch_f4.sh).
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
J = '/data/ugrip2026/journal'
SEEDS = (1337, 2026, 42)
RES = {'q35': 768, 'mg15': 896}
# Table VIII negative cells: source -> forced targets
CELLS = {
    'q35': {'det': ['reg', 'count', 'ml', 'cls'], 'cls': ['ml'], 'report': ['ml'], 'reg': ['ml']},
    'mg15': {'det': ['reg', 'ml'], 'cls': ['reg'], 'report': ['ml', 'reg'], 'reg': ['ml'], 'count': ['ml']},
}
ORDER = ['cls', 'ml', 'count', 'reg']


def targets(bb):
    return [t for t in ORDER if any(t in v for v in CELLS[bb].values())]


def job(name, bb, tasks, data, surface, out, adapter=None, extra=''):
    mode = f'--mode adapter --adapter {adapter}' if adapter else '--mode zs --protocol prompt'
    return (f'{name} --backbone {bb} {mode} --tasks {",".join(tasks)} --data {data} --surface {surface} '
            f'--out_dir {out}{extra}')


def arm_jobs(bb, seed, data, surface, root, extra='', tag=''):
    out = []
    for src, tg in CELLS[bb].items():
        ad = f'{J}/runs/j26_{bb}_task{src}_full_{RES[bb]}_s{seed}'
        name = f'f4{tag}_{bb}_task{src}_s{seed}'
        out.append(job(name, bb, [t for t in ORDER if t in tg], data, surface, f'{root}/{name}', ad, extra))
    name = f'f4{tag}_{bb}_shared_s{seed}'
    out.append(job(name, bb, targets(bb), data, surface, f'{root}/{name}', f'{J}/runs/j26_{bb}_shared_full_{RES[bb]}_s{seed}', extra))
    return out


def main():
    hold, dev = f'{J}/data/j26_holdout.jsonl', f'{J}/data/j26_devselect.jsonl'
    root = f'{J}/runs/f4'
    smoke = ['# F4 v2 smoke: dev-select, 5 rows per dataset, seed 1337 (format check, not a result)']
    for bb in ('q35', 'mg15'):
        smoke.append(job(f'f4smoke_{bb}_zsprompt', bb, targets(bb), dev, 'devselect', f'{root}/smoke/f4smoke_{bb}_zsprompt',
                         extra=' --per_dataset 5'))
        smoke += arm_jobs(bb, 1337, dev, 'devselect', f'{root}/smoke', ' --per_dataset 5', 'smoke')
    (HERE / 'smoke.txt').write_text('\n'.join(smoke) + '\n')
    for i, seed in enumerate(SEEDS):
        lines = [f'# F4 v2 holdout, seed {seed}: forced TaskLoRA sources + Shared (+ ZS-prompt base in this chain: {i < 2})']
        for bb in ('q35', 'mg15'):
            if i == ('q35', 'mg15').index(bb):  # ZS-prompt base once per backbone: q35 in chain 1, mg15 in chain 2
                lines.append(job(f'f4_{bb}_zsprompt', bb, targets(bb), hold, 'holdout', f'{root}/f4_{bb}_zsprompt'))
            lines += arm_jobs(bb, seed, hold, 'holdout', root)
        (HERE / f'holdout_s{seed}.txt').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
