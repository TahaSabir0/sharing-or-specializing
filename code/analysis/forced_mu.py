#!/usr/bin/env python3
"""F4 v2: transfer under forced format, for the 14 negative cells of Table VIII (format_analysis_plan.md).

mu_forced = (adapter_forced - ZS_forced) / (perfect - ZS_forced), per seed, mean over 3 seeds, with the
Table VIII bootstrap (format_lockin.boot_job: two-level paired, seeds then questions stratified by dataset,
B resamples) and Holm across the forced targets of each source adapter (per backbone). Also:
  - mu (unforced, as Table VIII) and the change mu_forced - mu (same resamples; each side its own base);
  - Shared forced vs ZS forced (the reference) and Shared unforced vs ZS unforced;
  - cls: length-normalised option score (primary) and summed (sensitivity);
  - checks: errors, truncation, incomplete answers, dead ends, scorer-unparseable answers, mean
    unconstrained log-prob of the forced answers, and Shared forced = Shared unforced agreement.

CPU. Runs on the box (paths under /data/ugrip2026/journal), or on a mirror with --root.
  python code/analysis/forced_mu.py --root /data/ugrip2026/journal --out OUT [--B 10000] [--workers 16]
"""
import argparse
import copy
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'eval'))
sys.path.insert(0, str(HERE.parent / 'eval' / 'jobs' / 'f4'))
import bootstrap as bs  # noqa: E402
import format_lockin as FL  # noqa: E402
from plan_f4 import CELLS  # noqa: E402
from score_single import overflows  # noqa: E402

SEEDS = FL.SEEDS
LONG = {v: k for k, v in FL.SHORT.items()}


def load_forced(path, sum_variant=False):
    """bs.load_run of a forced run; sum_variant: cls answers from the summed option score."""
    path = Path(path)
    preds = json.loads((path / 'all_predictions.json').read_text(encoding='utf-8'))
    if sum_variant:
        preds = [dict(p, Answer=p.get('Answer_sum', p['Answer'])) for p in preds]
    import tempfile
    import shutil
    d = Path(tempfile.mkdtemp())
    (d / 'all_predictions.json').write_text(json.dumps(preds), encoding='utf-8')
    run = bs.load_run(str(d))
    shutil.rmtree(d, ignore_errors=True)
    return run, {p['id']: p for p in preds}


def diff_job(job):
    """mean-over-seeds mu_b - mu_a on the same resamples (seeds, then questions); each side its own base."""
    ts_a, base_a, ts_b, base_b, target, B, seed = job
    rng = np.random.default_rng(seed)
    st = bs.strata(base_a)
    draws = []
    for _ in range(B):
        s = bs.resample(st, rng)
        chosen = rng.integers(0, len(ts_a), len(ts_a))
        ba, bb = bs.macro(base_a, target, s), bs.macro(base_b, target, s)
        va = [FL.pgf(bs.macro(ts_a[i], target, s), ba, target) for i in chosen]
        vb = [FL.pgf(bs.macro(ts_b[i], target, s), bb, target) for i in chosen]
        draws.append(np.nan if None in va or None in vb else np.mean(vb) - np.mean(va))
    draws = np.asarray(draws, float)
    draws = draws[np.isfinite(draws)]
    if len(draws) < 0.95 * B:
        return {'ci95': None, 'B': B, 'reason': f'only {len(draws)} finite draws'}
    return {'ci95': [float(x) for x in np.quantile(draws, [.025, .975])], 'B': B,
            'p_vs_zero': float(min(1.0, 2 * min((draws <= 0).mean(), (draws >= 0).mean()) + 1 / (B + 1)))}


def dispatch(job):
    return diff_job(job[1:]) if job[0] == 'diff' else FL.boot_job(job)


def checks(rows, target):
    """Diagnostics of one forced task slice (list of prediction dicts)."""
    n = len(rows)
    out = {'n': n, 'errors': sum(str(r['Answer']).startswith('Error:') for r in rows),
           'dead_end': sum(bool(r.get('dead_end')) for r in rows),
           'mean_model_logprob_per_token': float(np.mean([r.get('model_logprob_mean', np.nan) for r in rows]))}
    if target != FL.CLS:
        out['truncated'] = sum(bool(r.get('truncated')) for r in rows)
        out['incomplete'] = sum(r.get('complete_match') is False for r in rows)
    if target in (FL.CNT, FL.REG):
        out['scorer_unparseable'] = sum(overflows(r['Answer']) or not str(r['Answer']).strip() for r in rows)
    if target == FL.ML:
        out['mean_labels'] = float(np.mean([len([x for x in r['Answer'].replace(';', ',').split(',') if x.strip()])
                                            if r['Answer'] else 0 for r in rows]))
    if target == FL.CLS:
        out['sum_vs_mean_disagree'] = sum(r.get('Answer_sum') != r['Answer'] for r in rows)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default=FL.BOX_ROOT)
    ap.add_argument('--out', required=True)
    ap.add_argument('--B', type=int, default=10000)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--random_seed', type=int, default=20261125)
    a = ap.parse_args()
    root = Path(a.root)
    store = FL.Store(root)
    f4 = root / 'runs' / 'f4'
    res = {'study': 'F4 v2 forced-format decoding (Tausifa 25 Nov)', 'surface': 'holdout', 'B': a.B,
           'random_seed': a.random_seed, 'code_commit': FL.git_commit(),
           'holm': 'across the forced targets of each source adapter, per backbone (and, as a sensitivity, '
                   'across all forced cells of the backbone)', 'backbones': {}}
    jobs, keys = [], []

    def add(job, key):
        jobs.append(job)
        keys.append(key)

    for bb, cells in CELLS.items():
        base_u, _ = store.base(bb, 'zsprompt')
        variants = {'mean': load_forced(f4 / f'f4_{bb}_zsprompt')}
        if any(FL.CLS in [LONG[t] for t in tg] for tg in cells.values()):
            variants['sum'] = load_forced(f4 / f'f4_{bb}_zsprompt', sum_variant=True)
        shared_f = {s: load_forced(f4 / f'f4_{bb}_shared_s{s}') for s in SEEDS}
        shared_f_sum = {s: load_forced(f4 / f'f4_{bb}_shared_s{s}', sum_variant=True) for s in SEEDS}
        shared_u = {s: store.joint(bb, 'shared', s)[0] for s in SEEDS}
        out_bb = {'cells': {}, 'shared': {}, 'zs_forced_checks': {}}
        targets = sorted({LONG[t] for tg in cells.values() for t in tg}, key=FL.TASKS.index)
        for tgt in targets:
            base_f = variants['mean'][0][tgt]
            out_bb['zs_forced_checks'][FL.SHORT[tgt]] = checks([variants['mean'][1][i] for i in base_f['ids']], tgt)
            # Shared reference
            sf = [shared_f[s][0][tgt] for s in SEEDS]
            su = [shared_u[s][tgt] for s in SEEDS]
            for t in sf + su:
                bs.align(t, base_f)
            st = bs.strata(base_f)
            agree = [float(np.mean([str(x).strip().lower() == str(y).strip().lower() for x, y in zip(f['pred'], u['pred'])]))
                     for f, u in zip(sf, su)]
            sh = {'mu_forced': FL.summary(FL.mu_of(sf, base_f, tgt, st)),
                  'mu_unforced': FL.summary(FL.mu_of(su, base_u[tgt], tgt, st)),
                  'macro_forced': FL.summary([bs.macro(t, tgt, st) for t in sf]),
                  'macro_unforced': FL.summary([bs.macro(t, tgt, st) for t in su]),
                  'agreement_forced_vs_unforced': FL.summary(agree),
                  'checks': [checks([shared_f[s][1][i] for i in base_f['ids']], tgt) for s in SEEDS]}
            if tgt == FL.CLS:
                sfs = [shared_f_sum[s][0][tgt] for s in SEEDS]
                sh['mu_forced_sum'] = FL.summary(FL.mu_of(sfs, variants['sum'][0][tgt], tgt, st))
            out_bb['shared'][FL.SHORT[tgt]] = sh
            if a.B:
                add(('full', sf, base_f, tgt, None, a.B, a.random_seed + len(jobs)), (bb, 'shared', FL.SHORT[tgt], 'mu_forced_bootstrap'))
        for src_s, tgs in cells.items():
            src = LONG[src_s]
            runs = {s: load_forced(f4 / f'f4_{bb}_task{src_s}_s{s}') for s in SEEDS}
            runs_sum = {s: load_forced(f4 / f'f4_{bb}_task{src_s}_s{s}', sum_variant=True) for s in SEEDS} if 'cls' in tgs else None
            for tgt_s in tgs:
                tgt = LONG[tgt_s]
                key = f'{src_s}->{tgt_s}'
                base_f = variants['mean'][0][tgt]
                tf = [runs[s][0][tgt] for s in SEEDS]
                tu = [store.transfer_run(bb, src, s)[0][tgt] for s in SEEDS]
                for t in tf + tu:
                    bs.align(t, base_f)
                bs.align(base_u[tgt], base_f)
                st = bs.strata(base_f)
                mu_u = FL.mu_of(tu, base_u[tgt], tgt, st)
                mu_e3 = store.transfer['backbones'][bb]['pairs'][key]['PGF']['values']
                cell = {'source': src, 'target': tgt,
                        'mu_forced': FL.summary(FL.mu_of(tf, base_f, tgt, st)),
                        'mu_unforced': FL.summary(mu_u), 'mu_unforced_matches_table': bool(np.allclose(mu_u, mu_e3, atol=1e-9)),
                        'macro_forced': FL.summary([bs.macro(t, tgt, st) for t in tf]),
                        'macro_unforced': FL.summary([bs.macro(t, tgt, st) for t in tu]),
                        'zs_macro_forced': bs.macro(base_f, tgt, st), 'zs_macro_unforced': bs.macro(base_u[tgt], tgt, st),
                        'checks': [checks([runs[s][1][i] for i in base_f['ids']], tgt) for s in SEEDS]}
                cell['change'] = cell['mu_forced']['mean'] - cell['mu_unforced']['mean']
                if tgt == FL.CLS:
                    tfs = [runs_sum[s][0][tgt] for s in SEEDS]
                    cell['mu_forced_sum'] = FL.summary(FL.mu_of(tfs, variants['sum'][0][tgt], tgt, st))
                out_bb['cells'][key] = cell
                if a.B:
                    k = a.random_seed + 1000 + len(jobs)
                    add(('full', tf, base_f, tgt, None, a.B, k), (bb, 'cells', key, 'mu_forced_bootstrap'))
                    add(('diff', tu, base_u[tgt], tf, base_f, tgt, a.B, k + 1), (bb, 'cells', key, 'change_bootstrap'))
        res['backbones'][bb] = out_bb

    if jobs:
        print(f'bootstrap: {len(jobs)} jobs, B={a.B}, workers={a.workers}', flush=True)
        with Pool(a.workers) as pool:
            done = pool.map(dispatch, jobs, chunksize=1)
        for (bb, part, key, name), r in zip(keys, done):
            res['backbones'][bb][part][key][name] = r
        for bb, cells in CELLS.items():
            b = res['backbones'][bb]['cells']
            for stat in ('mu_forced_bootstrap', 'change_bootstrap'):
                fam_all = {k: c[stat]['p_vs_zero'] for k, c in b.items() if c[stat].get('p_vs_zero') is not None}
                for k, p in bs.holm(fam_all).items():
                    b[k][stat]['p_holm_backbone'] = p
                for src_s in cells:
                    fam = {k: p for k, p in fam_all.items() if k.startswith(src_s + '->')}
                    for k, p in bs.holm(fam).items():
                        b[k][stat]['p_holm_targets'] = p
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'forced_mu.json').write_text(json.dumps(res, indent=1, default=FL._default) + '\n', encoding='utf-8')
    write_md(res, out / 'forced_mu.md')
    print((out / 'forced_mu.md').read_text(encoding='utf-8'))


def f(v, d=3):
    return '—' if v is None else f'{v:+.{d}f}'


def ms(s):
    return '—' if s['mean'] is None else f"{s['mean']:+.3f} ± {s['sd']:.3f}"


def write_md(res, path):
    L = [f"# F4 v2 forced-format decoding (holdout, B = {res['B']}, code `{res['code_commit'][:7]}`)", '',
         'μ_forced = (adapter_forced − ZS_forced)/(perfect − ZS_forced), mean ± SD over 3 seeds; CI from the '
         'Table VIII two-level paired bootstrap; * Holm p < 0.05 across the forced targets of each source. '
         'Change = μ_forced − μ (same resamples). Cls: length-normalised option log-prob (summed in brackets).', '']
    for bb, b in res['backbones'].items():
        L += [f'## {bb}', '', '| cell | μ (Table VIII) | μ_forced [95% CI] | Holm p | change [95% CI] | Holm p | trunc/incomplete/err |',
              '|---|---:|---:|---:|---:|---:|---|']
        for k, c in sorted(b['cells'].items(), key=lambda kc: kc[1]['mu_unforced']['mean']):
            bf, bd = c.get('mu_forced_bootstrap', {}), c.get('change_bootstrap', {})
            ci = lambda x: f"[{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}]" if x.get('ci95') else ''  # noqa: E731
            ph = lambda x: f"{x['p_holm_targets']:.3g}{'*' if x['p_holm_targets'] < .05 else ''}" if 'p_holm_targets' in x else ''  # noqa: E731
            extra = f" ({c['mu_forced_sum']['mean']:+.3f})" if 'mu_forced_sum' in c else ''
            ck = c['checks']
            tie = '/'.join(str(sum(x.get(q, 0) for x in ck)) for q in ('truncated', 'incomplete', 'errors'))
            L.append(f"| {k} | {ms(c['mu_unforced'])} | {ms(c['mu_forced'])}{extra} {ci(bf)} | {ph(bf)} | "
                     f"{f(c['change'])} {ci(bd)} | {ph(bd)} | {tie} |")
        L += ['', 'Shared (reference), same targets:', '',
              '| target | μ Shared unforced | μ Shared forced [95% CI] | macro unforced → forced | forced = unforced answers |',
              '|---|---:|---:|---:|---:|']
        for t, s in b['shared'].items():
            bf = s.get('mu_forced_bootstrap', {})
            ci = f"[{bf['ci95'][0]:+.3f}, {bf['ci95'][1]:+.3f}]" if bf.get('ci95') else ''
            extra = f" ({s['mu_forced_sum']['mean']:+.3f})" if 'mu_forced_sum' in s else ''
            L.append(f"| {t} | {ms(s['mu_unforced'])} | {ms(s['mu_forced'])}{extra} {ci} | "
                     f"{s['macro_unforced']['mean']:.3f} → {s['macro_forced']['mean']:.3f} | "
                     f"{100 * s['agreement_forced_vs_unforced']['mean']:.1f}% |")
        L += ['', 'ZS-prompt forced checks: ' + json.dumps(b['zs_forced_checks']), '']
    path.write_text('\n'.join(L) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
