#!/usr/bin/env python3
"""Holm correction + paper tables for the format analysis (results/format_analysis_plan.md).

Reads format_analysis.json (format_lockin.py analyze, B > 0), adds Holm-adjusted p-values (across the
mu targets of each source adapter, per backbone and statistic, as in E3), and writes:
  format_analysis_holm.json   the input plus p_holm fields
  format_tables.tex           three LaTeX tables: lock-in heatmap, negative cells, full format distribution

  python code/analysis/format_paper.py --in OUT/format_analysis.json --out OUT
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap as bs  # noqa: E402
from format_lockin import MU_TARGETS, TASKS, SHORT  # noqa: E402
from format_rules import FORMATS  # noqa: E402

BOOT = ('f7a_bootstrap', 'f7a_effect_bootstrap', 'f3_bootstrap')
NAME = {'q35': 'Qwen3.5-9B', 'mg15': 'MedGemma-1.5-4B'}
CAP = dict(zip(TASKS, ('Cls', 'ML', 'Det', 'Count', 'Reg', 'Report')))
FMT_CAP = dict(zip(FORMATS, ('Label', 'Label set', 'Boxes', 'Integer', 'Decimal', 'Source label', 'Prose', 'Other')))
NEG_THRESHOLD = -0.05  # cells listed in the negative-cells table: mean mu below this


def add_holm(result):
    for bb, b in result['backbones'].items():
        for source in TASKS:
            for stat in BOOT:
                fam = {}
                for target in MU_TARGETS[bb]:
                    if target == source:
                        continue
                    c = b['cells'][f'{SHORT[source]}->{SHORT[target]}']
                    p = c.get(stat, {}).get('p_vs_zero')
                    if p is not None:
                        fam[target] = p
                for target, p in bs.holm(fam).items():
                    b['cells'][f'{SHORT[source]}->{SHORT[target]}'][stat]['p_holm_targets'] = p
    result['holm'] = 'across the mu targets of each source adapter, per backbone and statistic'
    return result


def num(v, d=2):
    if abs(v) < 0.5 * 10 ** -d:  # rounds to zero: keep the sign of a negative value
        return ('$-$' if v < 0 else '') + f'{0:.{d}f}'
    s = f'{v:+.{d}f}'
    return s.replace('+', '$+$').replace('-', '$-$')


def ci(b):
    if not b or not b.get('ci95'):
        return '---'
    lo, hi = b['ci95']
    return f'[{num(lo)}, {num(hi)}]'


def star(b):
    """Zero-width star so starred and plain numbers stay aligned in right-aligned columns."""
    return '\\rlap{$^*$}' if b and b.get('p_holm_targets') is not None and b['p_holm_targets'] < 0.05 else ''


def lock_color(v):
    """White to dark orange."""
    r, g, bl = 1 - 0.12 * v, 1 - 0.55 * v, 1 - 0.85 * v
    text = f'{100 * v:.0f}'
    if v > 0.7:
        text = '{\\color{white}' + text + '}'
    return f'\\cellcolor[rgb]{{{r:.3f},{g:.3f},{bl:.3f}}}{text}'


def lockin_table(result):
    rows = []
    for i, s in enumerate(TASKS):
        cells = []
        for bb in ('q35', 'mg15'):
            part = [CAP[s]]
            for t in TASKS:
                if s == t:
                    part.append('\\cellcolor[gray]{0.85}')
                else:
                    part.append(lock_color(result['backbones'][bb]['cells'][f'{SHORT[s]}->{SHORT[t]}']['lock_in']['mean']))
            cells.append(' & '.join(part))
        rows.append(cells[0] + ' & & ' + cells[1] + ' \\\\')
    head = ' & '.join(CAP[t] for t in TASKS)
    return '\n'.join([
        '\\begin{table*}[!t]',
        '\\caption{Format lock-in in cross-task transfer: percentage of an adapter\'s answers to task $j$ that are',
        'in its own training task\'s output format (and not valid for $j$), or are labels from its own training',
        'data (mean over three seeds). Joint-training arms (Shared, Hydra-T, Hydra-R, Shared r48) answer 0\\% of',
        'questions outside their own task\'s format on both backbones.}',
        '\\label{tab:lockin}', '\\centering', '\\small', '\\setlength{\\tabcolsep}{6pt}', '\\renewcommand{\\arraystretch}{1.25}',
        '\\begin{tabular}{l*{6}{c}c l*{6}{c}}', '\\toprule',
        f'\\multicolumn{{7}}{{c}}{{{NAME["q35"]}}} & & \\multicolumn{{7}}{{c}}{{{NAME["mg15"]}}} \\\\',
        '\\cmidrule(lr){1-7}\\cmidrule(lr){9-15}',
        f'Adapter & {head} & & Adapter & {head} \\\\', '\\midrule', *rows, '\\bottomrule', '\\end{tabular}',
        '\\end{table*}'])


def negative_table(result):
    lines = []
    for bb in ('q35', 'mg15'):
        b = result['backbones'][bb]
        cells = [(k, c) for k, c in b['cells'].items()
                 if 'mu' in c and c['mu']['mean'] is not None and c['mu']['mean'] < NEG_THRESHOLD]
        cells.sort(key=lambda kc: kc[1]['mu']['mean'])
        lines.append(f'\\multicolumn{{8}}{{l}}{{\\textit{{{NAME[bb]}}}}} \\\\')
        for k, c in cells:
            s, t = k.split('->')
            src = next(x for x in TASKS if SHORT[x] == s)
            tgt = next(x for x in TASKS if SHORT[x] == t)
            under = f'{100 * c["under_cardinality"]["mean"]:.0f}' if 'under_cardinality' in c else ''
            f3 = c['f3']
            if 'mu_valid' in f3 and f3['mu_valid']['mean'] is not None and not f3['descriptive_only']:
                b3 = c.get('f3_bootstrap')
                f3s = f'{num(f3["mu_valid"]["mean"])}{star(b3)} & {{\\footnotesize {ci(b3)}}}'
            else:
                f3s = '--- & '
            swap = f'{num(c["f7a_mu_swap_locked"]["mean"])}{star(c.get("f7a_bootstrap"))}'
            lines.append(f'{CAP[src]} $\\to$ {CAP[tgt]} & {100 * c["lock_in"]["mean"]:.0f} & {under} & '
                         f'{num(c["mu"]["mean"])} & {swap} & {f3s} & {100 * f3["pct_valid"]["mean"]:.0f} \\\\')
        lines.append('\\midrule' if bb == 'q35' else '')
    return '\n'.join([
        '\\begin{table*}[!t]',
        '\\caption{Negative transfer cells ($\\mu<-0.05$) and how much of each is format (mean over three seeds).',
        'Lock-in and under-cardinality in \\% of answers; $\\mu$ after the swap and on well-formed answers as defined',
        'in Section~\\ref{sec:e3}, with the share of questions kept. $^*$: differs from 0 after Holm correction',
        'across the targets of each adapter (two-level paired bootstrap, $B=10{,}000$).}',
        '\\label{tab:lockin_cells}', '\\centering', '\\small', '\\setlength{\\tabcolsep}{6pt}',
        '\\renewcommand{\\arraystretch}{1.1}',
        '\\begin{tabular}{lrrrrr@{\\hspace{16pt}}lr}', '\\toprule',
        ' & \\multicolumn{2}{c}{Answers (\\%)} & \\multicolumn{4}{c}{Transfer $\\mu$} & \\\\',
        '\\cmidrule(lr){2-3}\\cmidrule(lr){4-7}',
        'Pair & Lock-in & Under-card. & All & After swap & Well-formed & 95\\% CI & Kept (\\%) \\\\',
        '\\midrule', *[l for l in lines if l], '\\bottomrule', '\\end{tabular}',
        '',
        '\\vspace{2pt}',
        '\\parbox{\\textwidth}{\\footnotesize Under-card.: one label where the gold answer has two or more',
        '(multi-label targets only). ---: fewer than 30 questions, or under 20\\%, answered in the target format.}',
        '\\end{table*}'])


def distribution_table(result):
    lines = []
    for bb in ('q35', 'mg15'):
        lines.append(f'\\multicolumn{{{len(FORMATS) + 1}}}{{l}}{{\\textit{{{NAME[bb]}}}}} \\\\')
        for s in TASKS:
            for t in TASKS:
                if s == t:
                    continue
                c = result['backbones'][bb]['cells'][f'{SHORT[s]}->{SHORT[t]}']
                vals = ' & '.join(f'{100 * c["formats"][f]["mean"]:.1f}' for f in FORMATS)
                lines.append(f'{CAP[s]} $\\to$ {CAP[t]} & {vals} \\\\')
        if bb == 'q35':
            lines.append('\\midrule')
    return '\n'.join([
        '\\begin{table*}[!t]',
        '\\caption{Format of every off-task answer (\\% of answers, mean over three seeds). Label: one option of',
        'the question; Label set: two or more; Source label: a label from the adapter\'s own training data that',
        'the question does not offer; Prose: a sentence or four or more words; Other: none of these.}',
        '\\label{tab:formats}', '\\centering', '\\scriptsize', '\\setlength{\\tabcolsep}{4pt}',
        '\\begin{tabular}{l' + 'r' * len(FORMATS) + '}', '\\toprule',
        'Pair & ' + ' & '.join(FMT_CAP[f] for f in FORMATS) + ' \\\\', '\\midrule', *lines,
        '\\bottomrule', '\\end{tabular}', '\\end{table*}'])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--in', dest='inp', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    result = add_holm(json.loads(Path(a.inp).read_text(encoding='utf-8')))
    out = Path(a.out)
    (out / 'format_analysis_holm.json').write_text(json.dumps(result, indent=1) + '\n', encoding='utf-8')
    tex = '% Generated by code/analysis/format_paper.py from format_analysis.json\n\n' + '\n\n'.join(
        [lockin_table(result), negative_table(result), distribution_table(result)]) + '\n'
    (out / 'format_tables.tex').write_text(tex, encoding='utf-8')
    for bb, b in result['backbones'].items():
        print(f'== {bb}')
        for k, c in b['cells'].items():
            if 'mu' in c:
                ps = [c.get(s, {}).get('p_holm_targets') for s in BOOT]
                print(f'  {k:14s} Holm p  F7a {ps[0]}  effect {ps[1]}  F3 {ps[2]}')


if __name__ == '__main__':
    main()
