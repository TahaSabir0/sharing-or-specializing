"""Unit tests for code/eval/forced_format.py (CPU, no model): python -m pytest code/tests/test_forced_format.py"""
import os
import random
import sys

import regex

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'eval'))
import forced_format as FF  # noqa: E402

CHEST = ('What pathologies are present in this chest X-ray? Select all that apply from the following: '
         'Pleural Effusion, Nodule, Pneumonia, Cardiomegaly, Normal')
PERI = ('As a dental imaging specialist, what abnormalities would you report on tooth 11? Select all that apply '
        'from the following, separate multiple selections with a semicolon: Dental Caries; Periradicular Lesions; '
        'Endodontic Treatment, Restorative Treatment, and Complications.')

# a toy vocabulary: single characters, digits, some words with and without a leading space, junk
VOCAB = (list('0123456789.,; abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
         + ['12', '.5', 'Nod', 'ule', ' Nodule', 'Pleural', ' Effusion', ', ', '; ', 'Dental', ' Caries',
            'Endodontic', ' Treatment', ', Restorative', 'Complications', ' and', 'Pneumonia', 'Normal', '\n', 'xyz!'])
EOT = len(VOCAB)
STRS = VOCAB + [None]  # EOT has no text


def walk(constraint, rng, max_new):
    """Random walk through the allowed tokens (stands in for any logits): returns (text, ended)."""
    ids = []
    for _ in range(max_new):
        t = rng.choice(constraint.allowed(ids))
        if t == EOT:
            return constraint.text(ids), True
        ids.append(t)
    return constraint.text(ids), False


def test_ml_options():
    opts, sep = FF.ml_options(CHEST)
    assert sep == ', ' and opts == ['Pleural Effusion', 'Nodule', 'Pneumonia', 'Cardiomegaly', 'Normal']
    opts, sep = FF.ml_options(PERI)
    assert sep == '; '
    assert opts[-1] == 'Endodontic Treatment, Restorative Treatment, and Complications'


def test_numbers_only_valid_strings():
    rng = random.Random(0)
    cands = FF.number_candidates(STRS)
    assert {STRS[i] for i in cands} == set('0123456789.') | {'12', '.5'}
    for pat in (FF.COUNT_RE, FF.REG_RE):
        for _ in range(300):
            c = FF.RegexConstraint(pat, cands, STRS, EOT)
            text, ended = walk(c, rng, 12)
            assert ended, text  # 12 tokens always suffice to finish
            assert regex.fullmatch(pat, text), (pat, text)
            assert not c.dead_end


def test_eot_only_on_complete_match():
    c = FF.RegexConstraint(FF.REG_RE, FF.number_candidates(STRS), STRS, EOT)
    one = STRS.index('1')
    dot = STRS.index('.')
    assert EOT not in c.allowed([])            # empty answer not allowed
    assert EOT in c.allowed([one])             # '1' complete
    assert EOT not in c.allowed([one, dot])    # '1.' incomplete
    assert dot not in c.allowed([one, dot])    # no second point
    five = [one] * 5
    assert set(STRS[t] for t in c.allowed(five) if t != EOT) <= {'.', '.5'}  # at most 5 integer digits


def test_ml_language_and_dedupe():
    rng = random.Random(1)
    idx = FF.index_strings(STRS)
    for q in (CHEST, PERI):
        opts, sep = FF.ml_options(q)
        pat = FF.ml_regex(opts, sep)
        cands = FF.ml_candidates(idx, opts, sep)
        assert STRS.index('xyz!') not in cands and STRS.index('\n') not in cands
        n_ended = 0
        for _ in range(200):
            c = FF.RegexConstraint(pat, cands, STRS, EOT)
            text, ended = walk(c, rng, 200)
            if not ended:
                continue
            n_ended += 1
            assert regex.fullmatch(pat, text), text
            labels = FF.dedupe_labels(text, sep).split(sep)
            assert len(set(labels)) == len(labels) and set(labels) <= set(opts)
        assert n_ended > 50
    assert FF.dedupe_labels('Nodule, Normal, Nodule', ', ') == 'Nodule, Normal'
    opts, _ = FF.ml_options(CHEST)
    assert FF.dedupe_labels('Nodule, Normal, Nodule, Pneum', ', ', opts) == 'Nodule, Normal'  # cut mid-label
    assert FF.dedupe_labels('Nodule, Normal', ', ', opts) == 'Nodule, Normal'
    peri_last = 'Endodontic Treatment, Restorative Treatment, and Complications'
    assert FF.dedupe_labels(f'{peri_last}; Dental Caries; {peri_last}', '; ') == f'{peri_last}; Dental Caries'


def test_cls_candidates():
    assert FF.cls_candidates('Grade? A. No DR B. Mild NPDR C. Moderate NPDR', ()) == ['A', 'B', 'C']
    assert FF.cls_candidates('Options: A: Normal bone marrow B: Myelofibrosis', ()) == ['A', 'B']
    assert FF.cls_candidates('Options: A) Benign lesion B) Malignant lesion C) No lesion', ()) == ['A', 'B', 'C']
    assert FF.cls_candidates('Select one: is the lesion a Calcification, a Mass, or Both?',
                             {'Calcification', 'Mass', 'Both'}) == ['Both', 'Calcification', 'Mass']
