"""Forced-format decoding: the constraints, torch-free (results/format_analysis_plan.md, F4 v2).

Tausifa's spec (25 Nov): rerun the negative-transfer cells with the output restricted to the target
task's format, identically for the adapter, the ZS-prompt base and Shared.
  counting     regex  \\d{1,5}
  regression   regex  \\d{1,5}(\\.\\d{1,3})?
  multi-label  regex  (o1|o2|...)(SEP(o1|o2|...))*  built from the question's own options, SEP the
               question's own separator (', ' for chestdr; '; ' for periapical, whose last option itself
               contains commas). Duplicate labels removed afterwards; the model chooses how many.
  cls          no generation: each option scored by its log-probability as the whole answer
               (option tokens + end-of-turn), summed and length-normalised both kept.

Regex enforcement: a token t is allowed after the text generated so far, s, when s + str(t) is a prefix
of some string in the language (`regex` module partial matching); end-of-turn is allowed only when s is
a complete match. str(t) is each token's exact text contribution, measured by decoding it after a fixed
anchor token (so leading spaces of SentencePiece / byte-level BPE tokens are kept).
"""
import re

import regex

COUNT_RE = r'\d{1,5}'
REG_RE = r'\d{1,5}(?:\.\d{1,3})?'
MAX_NEW = {'counting': 12, 'regression': 12, 'multi-label classification': 64}
NUMBER_CHARS = frozenset('0123456789.')
MAX_TOKEN_CHARS = 32  # longest token text considered for multi-label candidates

_LETTER_MARK = re.compile(r'(?:^|(?<=[\s,;(]))\(?([A-K])\s*[.):]\s+')


# ------------------------------------------------------------------------------ question options
def ml_options(question):
    """(options in their original case, separator) from 'Select all that apply from the following: a, b'
    or '..., separate multiple selections with a semicolon: a; b; c.' Same parsing as
    analysis/format_rules.ml_options, without lower-casing."""
    i = question.lower().rfind('from the following')
    if i < 0:
        raise ValueError(f'no option list in multi-label question: {question[:120]!r}')
    rest = question[i + len('from the following'):]
    rest = re.sub(r'^\s*,?\s*separate multiple selections with a semicolon', '', rest, flags=re.I)
    rest = rest.lstrip(' :').strip().rstrip('.')
    sep = '; ' if ';' in rest else ', '
    opts = [x.strip() for x in rest.split(sep.strip()) if x.strip()]
    if len(opts) < 2 or len(set(opts)) != len(opts):
        raise ValueError(f'bad option list {opts!r}')
    return opts, sep


def ml_regex(options, sep):
    alt = '(?:' + '|'.join(regex.escape(o) for o in options) + ')'
    return f'{alt}(?:{regex.escape(sep)}{alt})*'


def dedupe_labels(text, sep):
    """'A, B, A' -> 'A, B' (first occurrence kept, order kept)."""
    out = []
    for part in text.split(sep):
        if part and part not in out:
            out.append(part)
    return sep.join(out)


def cls_candidates(question, dataset_answers):
    """Answer strings to score: the option letters for lettered questions (the gold answers are letters),
    else the dataset's answers named in the question (CMMD: Calcification / Mass / Both)."""
    marks, want = [], 'A'
    for m in _LETTER_MARK.finditer(question):
        if m.group(1) == want:
            marks.append(m.group(1))
            want = chr(ord(want) + 1)
    if len(marks) >= 2:
        return marks
    # the option as the question writes it ('Luminal A', 'HER2-enriched'; the scorer lower-cases)
    found = []
    for a in sorted(set(dataset_answers)):
        m = re.search(r'(?<![A-Za-z0-9])' + re.escape(a) + r'(?![A-Za-z0-9])', question, flags=re.I)
        if m:
            found.append(m.group(0))
    if len(found) < 2:
        raise ValueError(f'no options found in {question[:120]!r} (vocab {sorted(set(dataset_answers))})')
    return found


# ------------------------------------------------------------------------------ token tables
def token_strings(tok):
    """[str | None] per token id: the exact text a token adds after an anchor; None for special tokens
    and tokens that are not whole characters (byte pieces decode to U+FFFD)."""
    anchor = tok.encode('0', add_special_tokens=False)
    base = tok.decode(anchor)
    special = set(tok.all_special_ids) | set(getattr(tok, 'added_tokens_decoder', {}) or {})
    n = len(tok)
    out = [None] * n
    step = 8192
    for lo in range(0, n, step):
        ids = list(range(lo, min(n, lo + step)))
        dec = tok.batch_decode([anchor + [i] for i in ids], skip_special_tokens=False,
                               clean_up_tokenization_spaces=False)
        for i, d in zip(ids, dec):
            if i in special or not d.startswith(base):
                continue
            s = d[len(base):]
            if s and '�' not in s:
                out[i] = s
    return out


def number_candidates(strs):
    return [i for i, s in enumerate(strs) if s and set(s) <= NUMBER_CHARS]


def ml_candidates(strs_index, options, sep):
    """Ids of tokens whose text is a substring of 'o_i', or of 'o_i SEP o_j' (a token may span a
    separator). strs_index: {text: [ids]} for texts up to MAX_TOKEN_CHARS."""
    pieces = set(options) | {a + sep + b for a in options for b in options}
    subs = set()
    for p in pieces:
        n = len(p)
        for i in range(n):
            for j in range(i + 1, min(n, i + MAX_TOKEN_CHARS) + 1):
                subs.add(p[i:j])
    return sorted({t for s in subs for t in strs_index.get(s, ())})


def index_strings(strs):
    idx = {}
    for i, s in enumerate(strs):
        if s and len(s) <= MAX_TOKEN_CHARS:
            idx.setdefault(s, []).append(i)
    return idx


# ------------------------------------------------------------------------------ the constraint
class RegexConstraint:
    """allowed(generated_ids) -> token ids allowed next. Torch-free; the harness wraps it."""

    def __init__(self, pattern, candidates, strs, eot_id):
        self.pattern = pattern
        self.rx = regex.compile(pattern)
        self.candidates = list(candidates)
        self.strs = strs
        self.eot = eot_id
        self.dead_end = False
        self._cache = {}

    def text(self, ids):
        return ''.join(self.strs[i] for i in ids)

    def complete(self, text):
        return self.rx.fullmatch(text) is not None

    def allowed(self, ids):
        text = self.text(ids)
        if text in self._cache:
            return self._cache[text]
        ok = [t for t in self.candidates if self.rx.fullmatch(text + self.strs[t], partial=True)]
        if self.complete(text):
            ok.append(self.eot)
        if not ok:  # cannot happen for these languages (single characters are tokens); recorded if it does
            self.dead_end = True
            ok = [self.eot]
        self._cache[text] = ok
        return ok
