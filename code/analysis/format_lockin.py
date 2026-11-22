#!/usr/bin/env python3
"""Format lock-in analysis on the E3 transfer predictions (results/format_analysis_plan.md).

CPU only. Stages:
  vocab     build code/analysis/format_vocab.json (label vocabularies) from the training decks
  validate  F0 checks: ambiguity counts + consistency on the diagonal (each TaskLoRA on its own task)
  analyze   F1 lock-in rates, F2 scatter, F3 well-formed subset, F5 ZS-bare, F6 joint arms, F7 swap
            (point estimates; --B > 0 adds the two-level paired bootstrap for F3/F7)

Paths: --root is a local mirror of /data/ugrip2026/journal (phase_b/e3_collected, phase_b/e3_results,
phase_b/e8_collected, runs/j26_*_shared48_*, data/). Box paths in the E3 index are remapped onto it.

  python code/analysis/format_lockin.py vocab    --root R
  python code/analysis/format_lockin.py validate --root R --out OUT
  python code/analysis/format_lockin.py analyze  --root R --out OUT [--B 10000] [--workers 16]
"""
import argparse
import collections
import copy
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'eval'))
import bootstrap as bs  # noqa: E402
import format_rules as F  # noqa: E402

BOX_ROOT = '/data/ugrip2026/journal'
VOCAB_PATH = HERE / 'format_vocab.json'
SEEDS = (1337, 2026, 42)
BACKBONES = ('q35', 'mg15')
TASKS = F.TASKS
SHORT = F.SHORT
CLS, ML, DET, CNT, REG, REP = TASKS
# Targets with a usable transfer score mu (plan: detection mu is 0 everywhere, report is not
# interpreted, MedGemma's ZS-prompt counting MAE is 75,269 so mu ~ 1 for any sane output).
MU_TARGETS = {'q35': (CLS, ML, CNT, REG), 'mg15': (CLS, ML, REG)}
GATE_EFFECT = 0.05   # plan F4 gate: |mu_F7 - mu| >= 0.05 ...
GATE_CELLS = 3       # ... in >= 3 eligible cells on at least one backbone
JOINT_ARMS = ('shared', 'hydraT', 'hydraR', 'shared48')


# ----------------------------------------------------------------------------- helpers
def local(root, path):
    p = str(path)
    return Path(root) / p[len(BOX_ROOT) + 1:] if p.startswith(BOX_ROOT) else Path(p)


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=1, allow_nan=False, default=_default) + '\n', encoding='utf-8')


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def finite(v):
    return float(v) if v is not None and np.isfinite(v) else None


def summary(values):
    vals = [finite(v) for v in values]
    if None in vals or not vals:
        return {'values': vals, 'mean': None, 'sd': None}
    return {'values': vals, 'mean': float(np.mean(vals)), 'sd': float(np.std(vals, ddof=1)) if len(vals) > 1 else None}


def oriented(score, base, task):
    return (score - base) * (1 if bs.HIGHER_IS_BETTER[task] else -1)


def pgf(score, base, task):
    """Same definition as transfer_matrix.pgf (PLAN 1.8)."""
    gap = 1 - base if bs.HIGHER_IS_BETTER[task] else base
    if not (np.isfinite(score) and np.isfinite(base)) or gap <= 0:
        return None
    return finite(oriented(score, base, task) / gap)


def git_commit():
    import subprocess
    return subprocess.run(['git', '-C', str(HERE), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()


# ----------------------------------------------------------------------------- vocab
def build_vocab(root):
    data = Path(root) / 'data'
    out = {'dataset_vocab': {}, 'source_vocab': {}, 'source_letters': {},
           'built_from': [f'j26_train_{s}.jsonl' for s in SHORT.values()]}
    cls = read_jsonl(data / 'j26_train_cls.jsonl')
    by_ds = collections.defaultdict(set)
    texts = set()
    for r in cls:
        a = F.norm(r['answer'])
        by_ds[r['dataset']].add(a)
        for t in F.letter_options(r['question']).values():
            texts.add(t)
        if len(a) > 1:
            texts.add(a)
    out['dataset_vocab'] = {ds: sorted(v) for ds, v in sorted(by_ds.items())}
    out['source_vocab'][CLS] = sorted(texts)
    out['source_letters'][CLS] = True
    ml = read_jsonl(data / 'j26_train_ml.jsonl')
    labels = set()
    for r in ml:
        opts = F.ml_options(r['question'])
        labels.update(opts)
        got = F.parse_labels(r['answer'], opts)
        labels.update(got or [])
    out['source_vocab'][ML] = sorted(labels)
    out['source_letters'][ML] = False
    for t in (DET, CNT, REG, REP):
        out['source_vocab'][t] = []
        out['source_letters'][t] = False
    write_json(VOCAB_PATH, out)
    print(f'wrote {VOCAB_PATH}: cls source labels {len(out["source_vocab"][CLS])}, '
          f'ml source labels {len(out["source_vocab"][ML])}, datasets {len(out["dataset_vocab"])}')
    return out


def load_vocab():
    v = json.loads(VOCAB_PATH.read_text(encoding='utf-8'))
    v['source_vocab'] = {k: frozenset(x) for k, x in v['source_vocab'].items()}
    return v


# ----------------------------------------------------------------------------- loading
class Store:
    """Everything one analysis needs, read once from the local mirror."""

    def __init__(self, root):
        self.root = Path(root)
        self.deck = {r['id']: r for r in read_jsonl(self.root / 'data' / 'j26_holdout.jsonl')}
        self.index = json.loads((self.root / 'phase_b/e3_collected/index.json').read_text())
        self.transfer = json.loads((self.root / 'phase_b/e3_results/transfer.json').read_text())
        self.vocab = load_vocab()
        self._cache = {}

    def run(self, path):
        """-> (bs.load_run dict, {id: raw_output})"""
        path = local(self.root, path)
        if path not in self._cache:
            pred_file = path / 'all_predictions.json'
            if not pred_file.exists():
                pred_file = path / 'predictions_holdout.json'
            preds = json.loads(pred_file.read_text(encoding='utf-8'))
            raw = {p['id']: p.get('raw_output', '') for p in preds}
            if pred_file.name != 'all_predictions.json':
                run = load_run_file(pred_file)
            else:
                run = bs.load_run(str(path))
            self._cache[path] = (run, raw)
        return self._cache[path]

    def transfer_run(self, bb, source, seed):
        return self.run(self.index['backbones'][bb]['rows'][SHORT[source]][str(seed)])

    def base(self, bb, arm='zsprompt'):
        return self.run(self.index['backbones'][bb][arm])

    def joint(self, bb, arm, seed):
        if arm == 'shared48':
            res = 768 if bb == 'q35' else 896
            d = self.root / 'runs' / f'j26_{bb}_shared48_full_{res}_s{seed}'
            return self.run(d / 'eval_holdout' if (d / 'eval_holdout').exists() else d)
        return self.run(self.root / 'phase_b/e8_collected' / f'{bb}_{arm}_s{seed}')

    def assess_slice(self, run_raw, target, source):
        """F.assess for every question of `target`, in bs.load_run order."""
        run, raw = run_raw
        t = run[target]
        vocab = self.vocab
        offtask = source and source != target  # foreign labels exist only off-task
        sv = vocab['source_vocab'].get(source, frozenset()) if offtask else frozenset()
        sl = vocab['source_letters'].get(source, False) if offtask else False
        out = []
        for qid in t['ids']:
            row = self.deck[qid]
            out.append(F.assess(raw[qid], row, source or target,
                                dataset_vocab=vocab['dataset_vocab'].get(row['dataset'], ()),
                                source_vocab=sv, source_letters=sl))
        return out


def load_run_file(pred_file):
    """bs.load_run for a predictions file without CRIMSON beside it (Qwen r48 run dirs)."""
    import tempfile
    import shutil
    d = Path(tempfile.mkdtemp())
    shutil.copy(pred_file, d / 'all_predictions.json')
    crim = Path(pred_file).parent / 'crimson' / 'crimson_per_sample.jsonl'
    if crim.exists():
        shutil.copy(crim, d / 'crimson_per_sample.jsonl')
    run = bs.load_run(str(d))
    shutil.rmtree(d, ignore_errors=True)
    return run


# ----------------------------------------------------------------------------- F0 validate
LABEL_RULES = {'class_label', 'label_set', 'foreign_label'}


def is_conflict(key):
    """Designed overlaps: a label also being long enough to be prose, or a label of this question also
    being one of the source adapter's labels (resolved by precedence, plan F0). Anything else, e.g. a
    shape rule and a label rule at once, is a real conflict."""
    m = set(key.split(' + '))
    return not (m <= LABEL_RULES | {'free_text'} and m & LABEL_RULES)
def validate(store, out):
    """Ambiguity + diagonal consistency. Uses only diagonal (own-task) and zero-shot answers, so no
    lock-in rate is computed before the rules are committed."""
    result = {'surface': 'holdout', 'code_commit': git_commit(), 'backbones': {}}
    for bb in BACKBONES:
        res = {'diagonal': {}, 'ambiguity': collections.Counter(), 'n': 0}
        for task in TASKS:
            per_seed = []
            for seed in SEEDS:
                a = store.assess_slice(store.transfer_run(bb, task, seed), task, task)
                fmts = collections.Counter(x['format'] for x in a)
                bad = [(qid, store.transfer_run(bb, task, seed)[1][qid][:120], x['format'])
                       for qid, x in zip(store.transfer_run(bb, task, seed)[0][task]['ids'], a) if not x['valid']]
                for x in a:
                    res['n'] += 1
                    if x['n_matches'] > 1:
                        res['ambiguity'][' + '.join(x['matches'])] += 1
                extra = {}
                if task == CLS:
                    extra['noncanonical_rate'] = float(np.mean([x['noncanonical'] for x in a]))
                if task == ML:
                    extra['under_cardinality_rate'] = float(np.mean([x['under_cardinality'] for x in a]))
                per_seed.append({'seed': seed, 'n': len(a), 'valid_rate': float(np.mean([x['valid'] for x in a])),
                                 'formats': dict(fmts), 'invalid_examples': bad[:15], 'n_invalid': len(bad), **extra})
            res['diagonal'][task] = per_seed
        res['ambiguity'] = dict(res['ambiguity'].most_common())
        res['ambiguity_conflicts'] = {k: v for k, v in res['ambiguity'].items() if is_conflict(k)}
        # Off-task answers too: format counts only (no rates computed or written).
        off = collections.Counter()
        n_off = 0
        for source in TASKS:
            for target in TASKS:
                if source != target:
                    for seed in SEEDS:
                        for x in store.assess_slice(store.transfer_run(bb, source, seed), target, source):
                            n_off += 1
                            if x['n_matches'] > 1:
                                off[' + '.join(x['matches'])] += 1
        res['ambiguity_offtask'] = dict(off.most_common())
        res['ambiguity_offtask_conflicts'] = {k: v for k, v in off.items() if is_conflict(k)}
        res['n_offtask'] = n_off
        result['backbones'][bb] = res
        print(f'== {bb}')
        for task, rows in res['diagonal'].items():
            print(f'  {SHORT[task]:6s} own-task valid: ' + ', '.join(f"{r['valid_rate']:.4f} ({r['n_invalid']} bad)" for r in rows))
        print(f'  own-task answers matching >1 rule: {sum(res["ambiguity"].values())} of {res["n"]}: {res["ambiguity"]}')
        print(f'    of which real conflicts: {res["ambiguity_conflicts"]}')
        print(f'  off-task answers matching >1 rule: {sum(res["ambiguity_offtask"].values())} of {res["n_offtask"]}')
        print(f'    of which real conflicts: {res["ambiguity_offtask_conflicts"]}')
    write_json(Path(out) / 'f0_validation.json', result)
    return result


# ----------------------------------------------------------------------------- F1/F3/F7 core
def cell_assessments(store, bb, source, target):
    return [store.assess_slice(store.transfer_run(bb, source, seed), target, source) for seed in SEEDS]


def rates(a, target):
    n = len(a)
    r = {'n': n, 'lock_in': np.mean([x['locked_in'] for x in a]), 'valid': np.mean([x['valid'] for x in a]),
         'formats': {f: sum(x['format'] == f for x in a) / n for f in F.FORMATS}}
    if target == CLS:
        r['noncanonical'] = np.mean([x['noncanonical'] for x in a])
    if target == ML:
        r['under_cardinality'] = np.mean([x['under_cardinality'] for x in a])
        r['gold_multi'] = np.mean([x['gold_multi'] for x in a])
        r['fragment'] = np.mean([x['fragment'] for x in a])
    if target == REG:
        r['integer_on_regression'] = np.mean([x['integer_on_regression'] for x in a])
    if target == DET:
        r['flat_boxes'] = np.mean([x['flat_boxes'] for x in a])
    return r


def swapped(t, base_t, mask):
    """Copy of a task slice with the masked answers replaced by the base model's answers."""
    new = copy.copy(t)
    new['pred'] = [b if m else p for p, b, m in zip(t['pred'], base_t['pred'], mask)]
    return new


def canonicalised(t, a):
    """Classification: map non-canonical class labels ('C. Moderate NPDR', 'Malignant lesion') to the
    option letter the scorer expects."""
    new = copy.copy(t)
    new['pred'] = [x['class_key'] if x.get('noncanonical') else p for p, x in zip(t['pred'], a)]
    return new


def subset_strata(st, mask):
    return {ds: idx[mask[idx]] for ds, idx in st.items() if mask[idx].any()}


def accuracy_macro(t, st):
    return float(np.mean([np.mean([str(t['pred'][i]).strip().lower() == str(t['gt'][i]).strip().lower() for i in idx])
                          for idx in st.values()]))


def mu_of(ts, base_t, target, st):
    b = bs.macro(base_t, target, st)
    return [pgf(bs.macro(t, target, st), b, target) for t in ts]


# ----------------------------------------------------------------------------- bootstrap jobs
def boot_job(job):
    """Two-level paired bootstrap of a mean-over-seeds transfer score (seeds, then questions stratified
    by dataset), as transfer_matrix.transfer_ci, on the PGF scale. kind='full': all questions;
    kind='subset': each seed's own Q_valid mask applied inside every resample."""
    kind, ts, base_t, target, masks, B, seed = job
    rng = np.random.default_rng(seed)
    st = bs.strata(base_t)
    draws = []
    for _ in range(B):
        sample = bs.resample(st, rng)
        chosen = rng.integers(0, len(ts), len(ts))
        vals = []
        for i in chosen:
            s = sample if kind == 'full' else subset_strata(sample, masks[i])
            if not s:
                continue
            vals.append(pgf(bs.macro(ts[i], target, s), bs.macro(base_t, target, s), target))
        vals = [v for v in vals if v is not None]
        draws.append(np.mean(vals) if vals else np.nan)
    draws = np.asarray(draws, float)
    draws = draws[np.isfinite(draws)]
    if len(draws) < 0.95 * B:
        return {'ci95': None, 'B': B, 'reason': f'only {len(draws)} finite draws'}
    return {'ci95': [float(x) for x in np.quantile(draws, [.025, .975])], 'B': B, 'n_finite': int(len(draws)),
            'p_vs_zero': float(min(1.0, 2 * min((draws <= 0).mean(), (draws >= 0).mean()) + 1 / (B + 1)))}


def diff_job(job):
    """Bootstrap of the format effect mu_swap - mu (same resamples for both)."""
    ts_a, ts_b, base_t, target, B, seed = job
    rng = np.random.default_rng(seed)
    st = bs.strata(base_t)
    draws = []
    for _ in range(B):
        sample = bs.resample(st, rng)
        chosen = rng.integers(0, len(ts_a), len(ts_a))
        b = bs.macro(base_t, target, sample)
        va = [pgf(bs.macro(ts_a[i], target, sample), b, target) for i in chosen]
        vb = [pgf(bs.macro(ts_b[i], target, sample), b, target) for i in chosen]
        if None in va or None in vb:
            draws.append(np.nan)
            continue
        draws.append(np.mean(vb) - np.mean(va))
    draws = np.asarray(draws, float)
    draws = draws[np.isfinite(draws)]
    if len(draws) < 0.95 * B:
        return {'ci95': None, 'B': B, 'reason': f'only {len(draws)} finite draws'}
    return {'ci95': [float(x) for x in np.quantile(draws, [.025, .975])], 'B': B,
            'p_vs_zero': float(min(1.0, 2 * min((draws <= 0).mean(), (draws >= 0).mean()) + 1 / (B + 1)))}


# ----------------------------------------------------------------------------- analyze
def analyze(store, out, B=0, workers=8, random_seed=20261013):
    out = Path(out)
    result = {'study': 'E3 format analysis', 'surface': 'holdout', 'plan': 'results/format_analysis_plan.md',
              'code_commit': git_commit(), 'B': B, 'random_seed': random_seed,
              'vocab_md5': _md5(VOCAB_PATH), 'backbones': {}}
    jobs, job_keys = [], []
    for bb in BACKBONES:
        base_run, _ = store.base(bb, 'zsprompt')
        bare_run, _ = store.base(bb, 'zsbare')
        tr = store.transfer['backbones'][bb]
        base_cls_assess = store.assess_slice(store.base(bb, 'zsprompt'), CLS, None)
        base_formats = {t: rates(store.assess_slice(store.base(bb, arm), t, None), t)
                        for arm in ('zsprompt',) for t in TASKS}
        cells = {}
        for source in TASKS:
            for target in TASKS:
                if source == target:
                    continue
                key = f'{SHORT[source]}->{SHORT[target]}'
                a = cell_assessments(store, bb, source, target)
                per_seed = [rates(x, target) for x in a]
                cell = {'source': source, 'target': target,
                        'lock_in': summary([r['lock_in'] for r in per_seed]),
                        'valid': summary([r['valid'] for r in per_seed]),
                        'formats': {f: summary([r['formats'][f] for r in per_seed]) for f in F.FORMATS},
                        'n_questions': per_seed[0]['n']}
                for extra in ('noncanonical', 'under_cardinality', 'gold_multi', 'fragment', 'integer_on_regression', 'flat_boxes'):
                    if extra in per_seed[0]:
                        cell[extra] = summary([r[extra] for r in per_seed])
                pair = tr['pairs'][key]
                cell['mu_e3'] = pair['PGF']['mean']
                cell['mu_e3_bare'] = pair['PGF_bare']['mean']  # F5: matched bare-vs-bare comparison
                if target in MU_TARGETS[bb]:
                    ts = [store.transfer_run(bb, source, s)[0][target] for s in SEEDS]
                    base_t = base_run[target]
                    for t in ts:
                        bs.align(t, base_t)
                    st = bs.strata(base_t)
                    mu = mu_of(ts, base_t, target, st)
                    cell['mu'] = summary(mu)
                    cell['mu_matches_e3'] = bool(np.allclose(mu, pair['PGF']['values'], atol=1e-9))
                    # F7a: swap locked-in answers; F7b: swap every answer not in the target format
                    masks_lock = [np.array([x['locked_in'] for x in aa]) for aa in a]
                    masks_invalid = [np.array([not x['valid'] for x in aa]) for aa in a]
                    ts_a = [swapped(t, base_t, m) for t, m in zip(ts, masks_lock)]
                    ts_b = [swapped(t, base_t, m) for t, m in zip(ts, masks_invalid)]
                    cell['f7a_mu_swap_locked'] = summary(mu_of(ts_a, base_t, target, st))
                    cell['f7b_mu_swap_invalid'] = summary(mu_of(ts_b, base_t, target, st))
                    for name, mu_s in (('f7a_effect', cell['f7a_mu_swap_locked']), ('f7b_effect', cell['f7b_mu_swap_invalid'])):
                        cell[name] = (mu_s['mean'] - cell['mu']['mean']) if None not in (mu_s['mean'], cell['mu']['mean']) else None
                    if target == CLS:
                        # F7c: canonicalise BOTH sides (the base gives 'C. Moderate NPDR' too), so the
                        # comparison stays matched (CLAUDE.md rule 3)
                        ts_c = [canonicalised(t, aa) for t, aa in zip(ts, a)]
                        base_c = canonicalised(base_t, base_cls_assess)
                        cell['mu_canonicalised'] = summary(mu_of(ts_c, base_c, target, st))
                    # F3: content on well-formed answers
                    valid_masks = [np.array([x['valid'] for x in aa]) for aa in a]
                    f3 = f3_cell(store, bb, source, target, ts, base_t, st, valid_masks)
                    cell['f3'] = f3
                    if target == ML:
                        strict = [np.array([x['valid'] and not x['under_cardinality'] for x in aa]) for aa in a]
                        cell['f3_strict_ml'] = f3_cell(store, bb, source, target, ts, base_t, st, strict)
                    if B > 0:
                        seed_k = random_seed + len(jobs)
                        jobs += [('full', ts_a, base_t, target, None, B, seed_k),
                                 ('subset', ts, base_t, target, valid_masks, B, seed_k + 1)]
                        job_keys += [(bb, key, 'f7a_bootstrap'), (bb, key, 'f3_bootstrap')]
                        jobs.append(('diff', ts, ts_a, base_t, target, B, seed_k + 2))
                        job_keys.append((bb, key, 'f7a_effect_bootstrap'))
                cells[key] = cell
        result['backbones'][bb] = {'cells': cells, 'f2': f2(cells, bb), 'gate': gate(cells, bb),
                                   'zsprompt_base_formats': base_formats,
                                   'f5_sign_flips_vs_zsbare': tr.get('sign_flips_vs_zsbare'),
                                   'f6_joint': f6(store, bb)}
    if jobs:
        print(f'bootstrap: {len(jobs)} jobs, B={B}, workers={workers}', flush=True)
        with Pool(workers) as pool:
            res = pool.map(dispatch, jobs, chunksize=1)
        for (bb, key, name), r in zip(job_keys, res):
            result['backbones'][bb]['cells'][key][name] = r
    passed = any(result['backbones'][bb]['gate']['passed'] for bb in BACKBONES)
    result['f4_gate_passed'] = passed
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'format_analysis.json', result)
    write_tables(result, out)
    plots(result, out)
    return result


def dispatch(job):
    if job[0] == 'diff':
        return diff_job(job[1:])
    return boot_job(job)


def f3_cell(store, bb, source, target, ts, base_t, st, masks):
    own = [store.transfer_run(bb, target, s)[0][target] for s in SEEDS]  # target TaskLoRA, same seed
    per = []
    for t, o, m in zip(ts, own, masks):
        sub = subset_strata(st, m)
        n = int(m.sum())
        rec = {'n_valid': n, 'pct_valid': n / len(m), 'datasets_kept': len(sub), 'datasets_total': len(st)}
        if sub:
            sa, sb = bs.macro(t, target, sub), bs.macro(base_t, target, sub)
            rec.update(adapter=finite(sa), base=finite(sb), mu_valid=pgf(sa, sb, target),
                       tasklora_on_valid=finite(bs.macro(o, target, sub)), tasklora_all=finite(bs.macro(o, target, st)))
            if target == CLS:
                rec.update(acc_adapter=accuracy_macro(t, sub), acc_base=accuracy_macro(base_t, sub))
        per.append(rec)
    out = {'per_seed': per}
    for k in ('n_valid', 'pct_valid', 'adapter', 'base', 'mu_valid', 'tasklora_on_valid', 'tasklora_all',
              'acc_adapter', 'acc_base'):
        if all(k in r for r in per):
            out[k] = summary([r[k] for r in per])
    n_mean = np.mean([r['n_valid'] for r in per])
    pct_mean = np.mean([r['pct_valid'] for r in per])
    out['descriptive_only'] = bool(n_mean < 30 or pct_mean < 0.20)
    return out


def f2(cells, bb):
    from scipy.stats import spearmanr
    pairs = [(k, c) for k, c in cells.items() if c['target'] in MU_TARGETS[bb] and c.get('mu', {}).get('mean') is not None]
    x = [c['lock_in']['mean'] for _, c in pairs]
    xu = [c['lock_in']['mean'] + (c['under_cardinality']['mean'] if 'under_cardinality' in c else 0) for _, c in pairs]
    y = [c['mu']['mean'] for _, c in pairs]
    rho = spearmanr(x, y).statistic if np.ptp(x) > 0 else None
    rho_u = spearmanr(xu, y).statistic if np.ptp(xu) > 0 else None
    return {'n_pairs': len(pairs), 'pairs': [k for k, _ in pairs], 'spearman_lockin_mu': finite(rho),
            'spearman_lockin_plus_undercard_mu': finite(rho_u),
            'note': 'descriptive only (plan F2): wrong-format answers score ~0 by construction'}


def gate(cells, bb):
    eff = {k: c['f7a_effect'] for k, c in cells.items() if c['target'] in MU_TARGETS[bb] and c.get('f7a_effect') is not None}
    big = {k: v for k, v in eff.items() if abs(v) >= GATE_EFFECT}
    return {'rule': f'|mu_F7a - mu| >= {GATE_EFFECT} in >= {GATE_CELLS} eligible cells', 'cells_over': big,
            'n_over': len(big), 'n_eligible': len(eff), 'passed': len(big) >= GATE_CELLS}


def f6(store, bb):
    """Cross-task format errors of the joint arms (and the TaskLoRA diagonal, for contrast)."""
    out = {}
    for arm in JOINT_ARMS + ('task',):
        per_task = collections.defaultdict(list)
        for seed in SEEDS:
            try:
                rr = store.joint(bb, arm, seed) if arm != 'task' else None
            except FileNotFoundError:
                rr = None
            for task in TASKS:
                if arm == 'task':
                    a = store.assess_slice(store.transfer_run(bb, task, seed), task, task)
                elif rr is None:
                    continue
                else:
                    a = store.assess_slice(rr, task, None)
                rec = {'n': len(a), 'invalid': np.mean([not x['valid'] for x in a]),
                       'cross_task_format': np.mean([(not x['valid']) and x['format'] not in ('other', 'foreign_label') for x in a]),
                       'other': np.mean([x['format'] == 'other' for x in a])}
                if task == CLS:
                    rec['noncanonical'] = np.mean([x['noncanonical'] for x in a])
                if task == ML:
                    rec['under_cardinality'] = np.mean([x['under_cardinality'] for x in a])
                per_task[task].append(rec)
        out[arm] = {t: {k: summary([r[k] for r in v]) for k in v[0]} for t, v in per_task.items() if v}
    return out


def _md5(p):
    import hashlib
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


# ----------------------------------------------------------------------------- outputs
def fmt_ms(s, pct=False):
    if not s or s.get('mean') is None:
        return '—'
    m, sd = s['mean'], s.get('sd') or 0
    return f'{100*m:.1f} ± {100*sd:.1f}' if pct else f'{m:+.3f} ± {sd:.3f}'


def write_tables(result, out):
    lines = ['# Format analysis tables (generated by format_lockin.py)', '',
             f'code `{result["code_commit"][:7]}`, B = {result["B"]}, surface: holdout. Rates in % (mean ± SD over 3 seeds).', '']
    for bb, r in result['backbones'].items():
        lines += [f'## {bb}', '', '### Lock-in rate (%) — row = source adapter, column = target', '',
                  '| src \\ tgt | ' + ' | '.join(SHORT[t] for t in TASKS) + ' |', '|---|' + '---:|' * 6]
        for s in TASKS:
            row = [('—' if s == t else fmt_ms(r['cells'][f'{SHORT[s]}->{SHORT[t]}']['lock_in'], True)) for t in TASKS]
            lines.append(f'| {SHORT[s]} | ' + ' | '.join(row) + ' |')
        lines += ['', '### Per-cell summary (μ targets)', '',
                  '| cell | lock-in % | valid % | extra | μ (E3) | μ F7a swap | effect | μ F7b | F3 μ_valid (n%) | F3 TaskLoRA valid/all |',
                  '|---|---:|---:|---|---:|---:|---:|---:|---|---|']
        for k, c in r['cells'].items():
            if 'mu' not in c:
                continue
            extra = ''
            if 'under_cardinality' in c:
                extra = (f'under-card {fmt_ms(c["under_cardinality"], True)} (gold 2+: {fmt_ms(c["gold_multi"], True)}); '
                         f'fragment {fmt_ms(c["fragment"], True)}')
            if 'noncanonical' in c:
                extra = f'non-canon {fmt_ms(c["noncanonical"], True)}; μ canon {fmt_ms(c["mu_canonicalised"])}'
            if 'integer_on_regression' in c:
                extra = f'int-on-reg {fmt_ms(c["integer_on_regression"], True)}'
            f3 = c['f3']
            f3s = (f'{fmt_ms(f3.get("mu_valid"))} ({fmt_ms(f3.get("pct_valid"), True)})'
                   + (' *desc*' if f3['descriptive_only'] else '')) if 'mu_valid' in f3 else '—'
            tl = (f'{f3["tasklora_on_valid"]["mean"]:.3f}/{f3["tasklora_all"]["mean"]:.3f}'
                  if f3.get('tasklora_on_valid', {}).get('mean') is not None else '—')
            eff = f'{c["f7a_effect"]:+.3f}' if c.get('f7a_effect') is not None else '—'
            ci = c.get('f7a_effect_bootstrap', {}).get('ci95')
            if ci:
                eff += f' [{ci[0]:+.3f}, {ci[1]:+.3f}]'
            lines.append(f'| {k} | {fmt_ms(c["lock_in"], True)} | {fmt_ms(c["valid"], True)} | {extra} | '
                         f'{fmt_ms(c["mu"])} | {fmt_ms(c["f7a_mu_swap_locked"])} | {eff} | {fmt_ms(c["f7b_mu_swap_invalid"])} | {f3s} | {tl} |')
        g = r['gate']
        bf = r['zsprompt_base_formats']
        lines += ['', 'ZS-prompt base, own format (%): ' + ', '.join(f'{SHORT[t]} {100*bf[t]["valid"]:.1f}' for t in TASKS)
                  + f'; cls non-canonical {100*bf[CLS]["noncanonical"]:.1f}']
        lines += ['', f'F4 gate ({g["rule"]}): {g["n_over"]}/{g["n_eligible"]} cells over → **{"PASS" if g["passed"] else "fail"}**',
                  f'F2 (descriptive): Spearman ρ(lock-in, μ) = {r["f2"]["spearman_lockin_mu"]}, '
                  f'with under-cardinality = {r["f2"]["spearman_lockin_plus_undercard_mu"]}, {r["f2"]["n_pairs"]} pairs', '']
        lines += ['### F6: answers not in their own task\'s format (%)', '', '| arm | ' + ' | '.join(SHORT[t] for t in TASKS) + ' |',
                  '|---|' + '---:|' * 6]
        for arm, d in r['f6_joint'].items():
            lines.append(f'| {arm} | ' + ' | '.join(fmt_ms(d[t]['invalid'], True) if t in d else '—' for t in TASKS) + ' |')
        lines += ['', '### Full format distribution per cell (%)', '', '| cell | ' + ' | '.join(F.FORMATS) + ' |',
                  '|---|' + '---:|' * len(F.FORMATS)]
        for k, c in r['cells'].items():
            lines.append(f'| {k} | ' + ' | '.join(f'{100*c["formats"][f]["mean"]:.1f}' for f in F.FORMATS) + ' |')
        lines.append('')
    (out / 'format_tables.md').write_text('\n'.join(lines), encoding='utf-8')


def plots(result, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for bb, r in result['backbones'].items():
        fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
        mats = {}
        for name, getter in [('mu', lambda c: c.get('mu', {}).get('mean', c.get('mu_e3'))),
                             ('lock', lambda c: c['lock_in']['mean']),
                             ('swap', lambda c: c.get('f7a_mu_swap_locked', {}).get('mean'))]:
            m = np.full((6, 6), np.nan)
            for i, s in enumerate(TASKS):
                for j, t in enumerate(TASKS):
                    if i != j:
                        v = getter(r['cells'][f'{SHORT[s]}->{SHORT[t]}'])
                        m[i, j] = np.nan if v is None else v
            mats[name] = m
        lim = np.nanmax(np.abs(np.concatenate([mats['mu'][:, :5].ravel(), mats['swap'][:, :5].ravel()])))
        labels = [SHORT[t] for t in TASKS]
        for ax, (name, title, cmap, vmin, vmax) in zip(axes, [
                ('mu', 'A. μ (E3, Table VI)', 'RdBu', -lim, lim),
                ('lock', 'B. lock-in rate', 'Oranges', 0, 1),
                ('swap', 'C. μ after swapping locked-in answers (F7)', 'RdBu', -lim, lim)]):
            m = mats[name].copy()
            if name != 'lock':
                m[:, 5] = np.nan  # report column not interpreted
            im = ax.imshow(m, cmap=cmap, vmin=vmin, vmax=vmax)
            for i in range(6):
                for j in range(6):
                    if i == j:
                        continue
                    v = m[i, j]
                    ax.text(j, i, '' if np.isnan(v) else (f'{100*v:.0f}%' if name == 'lock' else f'{v:+.2f}'),
                            ha='center', va='center', fontsize=8)
            ax.set(xticks=range(6), yticks=range(6), xticklabels=labels, yticklabels=labels, title=title,
                   xlabel='target task', ylabel='source adapter')
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(f'{bb}: format lock-in (holdout, mean of 3 seeds)')
        fig.tight_layout()
        fig.savefig(out / f'{bb}_format_panels.pdf')
        fig.savefig(out / f'{bb}_format_panels.png', dpi=130)
        plt.close(fig)
        # F2 scatter
        pts = [(k, c['lock_in']['mean'], c['mu']['mean']) for k, c in r['cells'].items() if 'mu' in c and c['mu']['mean'] is not None]
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.scatter([p[1] for p in pts], [p[2] for p in pts])
        for k, x, y in sorted(pts, key=lambda p: p[2])[:5]:
            ax.annotate(k, (x, y), fontsize=8)
        ax.axhline(0, color='grey', lw=0.5)
        ax.set(xlabel='lock-in rate', ylabel='μ (E3)', title=f'{bb}: lock-in vs transfer (descriptive)')
        fig.tight_layout()
        fig.savefig(out / f'{bb}_lockin_scatter.pdf')
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('stage', choices=['vocab', 'validate', 'analyze'])
    p.add_argument('--root', required=True)
    p.add_argument('--out')
    p.add_argument('--B', type=int, default=0)
    p.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = p.parse_args()
    if not sys.flags.utf8_mode:  # bootstrap.load_run uses open() without an encoding (cp1252 on Windows)
        raise SystemExit('run with python -X utf8 (or PYTHONUTF8=1): question ids contain non-ASCII text')
    if a.stage == 'vocab':
        build_vocab(a.root)
        return
    store = Store(a.root)
    if a.stage == 'validate':
        validate(store, a.out)
    else:
        analyze(store, a.out, a.B, a.workers)


if __name__ == '__main__':
    main()
