"""Format classifier for the format lock-in analysis (results/format_analysis_plan.md, F0).

Assigns every model answer exactly one of eight formats, checking the rules in a fixed order and
stopping at the first match:

  class_label   exactly one item matching one of THIS question's options (letter or label text)
  label_set     2+ items, all from this question's options
  boxes         the whole answer is a list of 4-number lists (or one flat 4-number list, flagged)
  integer       only an integer
  decimal       only a number with a decimal point
  foreign_label 1-3 items from the SOURCE adapter's training label vocabulary, not this question's
  free_text     prose: 4+ words, 2+ sentences, or one sentence closed by a full stop
  other         empty, or none of the above

Deliberately independent of the scorer's parsers (score_single.py, infer_single_adapter.py), which
are lenient: they pull a number out of a box list, or take the last line of a paragraph.
Pure Python, no model or numpy dependency.
"""
import ast
import re

FORMATS = ('class_label', 'label_set', 'boxes', 'integer', 'decimal', 'foreign_label', 'free_text', 'other')
TASKS = ('classification', 'multi-label classification', 'detection', 'counting', 'regression',
         'report_generation')
SHORT = dict(zip(TASKS, ('cls', 'ml', 'det', 'count', 'reg', 'report')))
# The format each single-task adapter was trained to produce.
SOURCE_FORMAT = {'classification': 'class_label', 'multi-label classification': 'label_set',
                 'detection': 'boxes', 'counting': 'integer', 'regression': 'decimal',
                 'report_generation': 'free_text'}
# Formats that count as a valid answer to each target task. Overlaps are deliberate: a single label
# is a valid multi-label answer, an integer a valid regression answer (plan F1).
VALID = {'classification': {'class_label'}, 'multi-label classification': {'class_label', 'label_set'},
         'detection': {'boxes'}, 'counting': {'integer'}, 'regression': {'integer', 'decimal'},
         'report_generation': {'free_text'}}

_THINK_RE = re.compile(r'<think>.*?</think>', re.S)
_INT_RE = re.compile(r'[-+]?\d+')
_DEC_RE = re.compile(r'[-+]?\d*\.\d+')
_LETTER_MARK = re.compile(r'(?:^|(?<=[\s,;(]))\(?([A-K])\s*[.):]\s+')
_LETTER_ANSWER = re.compile(r'\(?([a-k])\)?(?:\s*[.):]\s*(.*))?')
# A sentence boundary: end punctuation after a 2+ letter word, then a capitalised word. Excludes
# 'C. Moderate NPDR' (one letter before the stop) and decimals.
_SENTENCE_END = re.compile(r'(?<=[A-Za-z]{2})[.!?](?=\s+[A-Z])')


def strip_think(text):
    """Same effect as vlm_prompt.strip_think_block: drop reasoning-channel tags."""
    if not isinstance(text, str):
        return ''
    text = _THINK_RE.sub('', text)
    return re.sub(r'</?think>', '', text).strip()


def norm(text):
    """Lower case, collapse whitespace, drop surrounding quotes and a trailing full stop."""
    s = re.sub(r'\s+', ' ', str(text)).strip().strip('"\'`').strip()
    s = s.rstrip('.').strip()
    return s.lower()


# ----------------------------------------------------------------------------- question options
def letter_options(question):
    """{'a': 'benign lesion', ...} from 'A) Benign lesion B) ...' / 'A: x, B: y' / 'A. x B. y'.
    Markers must run A, B, C, ... in order; fewer than two -> {}."""
    marks = list(_LETTER_MARK.finditer(question))
    chain, want = [], 'A'
    for m in marks:
        if m.group(1) == want:
            chain.append(m)
            want = chr(ord(want) + 1)
    if len(chain) < 2:
        return {}
    out = {}
    for k, m in enumerate(chain):
        end = chain[k + 1].start() if k + 1 < len(chain) else len(question)
        out[m.group(1).lower()] = norm(question[m.end():end].strip().rstrip(',;').strip())
    return out


def class_options(question, dataset_vocab=()):
    """Options of a classification question. Lettered questions: the letters (canonical, as the gold
    answers are letters) and their texts. Otherwise the dataset's answer vocabulary as it appears in
    the question ('Benign or Malignant?'); if fewer than two appear, the whole dataset vocabulary
    (e.g. numeric grades '0'-'4' named as a range)."""
    letters = letter_options(question)
    if letters:
        return {'letters': letters, 'texts': set()}
    ql = question.lower()
    vocab = {norm(v) for v in dataset_vocab}
    if vocab and all(_INT_RE.fullmatch(v) for v in vocab):
        return {'letters': {}, 'texts': vocab}  # numeric grades: a question names only the range
    found = {v for v in vocab if re.search(r'(?<![a-z0-9])' + re.escape(v) + r'(?![a-z0-9])', ql)}
    return {'letters': {}, 'texts': found if len(found) >= 2 else vocab}


def ml_options(question):
    """Label list after the last 'from the following'; ';'-separated when the list uses ';'."""
    i = question.lower().rfind('from the following')
    if i < 0:
        return []
    rest = question[i + len('from the following'):]
    rest = re.sub(r'^\s*,?\s*separate multiple selections with a semicolon', '', rest, flags=re.I)
    rest = rest.lstrip(' :').strip().rstrip('.')
    sep = ';' if ';' in rest else ','
    return [norm(x) for x in rest.split(sep) if x.strip()]


def parse_labels(answer, options):
    """Split an answer into labels from `options` (a set of normalised labels), or None if any part is
    not an option. Splits on ';' first, then on ',' while re-joining consecutive parts so a label
    that itself contains commas ('Endodontic Treatment, Restorative Treatment, and Complications')
    is read as one label. Returns the distinct labels in order."""
    options = set(options)
    s = re.sub(r'\s+', ' ', str(answer)).strip()
    if not s:
        return None
    labels = []
    for piece in s.split(';'):
        p = norm(piece)
        if not p:
            continue
        if p in options:
            labels.append(p)
            continue
        parts = [norm(x) for x in p.split(',')]
        i = 0
        while i < len(parts):
            for j in range(len(parts), i, -1):
                cand = ', '.join(x for x in parts[i:j])
                if cand in options:
                    labels.append(cand)
                    i = j
                    break
            else:
                return None
    out = list(dict.fromkeys(labels))
    return out or None


def match_class(answer, opts):
    """-> ('canonical'|'noncanonical', option key) or None. Canonical = the exact form the scorer
    accepts (bare letter for lettered questions, label text otherwise)."""
    s = norm(answer)
    if not s:
        return None
    if opts['letters']:
        m = _LETTER_ANSWER.fullmatch(s)
        if m and m.group(1) in opts['letters']:
            rest = (m.group(2) or '').strip()
            text = opts['letters'][m.group(1)]
            if s == m.group(1):
                return 'canonical', m.group(1)
            if not rest or rest == text or text.startswith(rest):
                return 'noncanonical', m.group(1)
        for k, text in opts['letters'].items():
            if s == text:
                return 'noncanonical', k
        return None
    if s in opts['texts']:
        return 'canonical', s
    return None


# ----------------------------------------------------------------------------- shape rules
def box_kind(text):
    t = text.strip()
    if not t.startswith('['):
        return None
    try:
        v = ast.literal_eval(t)
    except (ValueError, SyntaxError, MemoryError, RecursionError, TypeError):
        return None

    def num(x):
        return isinstance(x, (int, float)) and not isinstance(x, bool)

    if not isinstance(v, (list, tuple)):
        return None
    if len(v) == 4 and all(num(x) for x in v):
        return 'flat'
    if all(isinstance(b, (list, tuple)) and len(b) == 4 and all(num(x) for x in b) for b in v):
        return 'nested'
    return None


def is_free_text(text):
    """Prose: 4+ words, or 2+ sentences, or one sentence closed by a full stop ('Cardiomegaly.',
    'Heart size normal.'), the style of the report adapter. Words are tokens with 2+ letters, so a
    box list or a number never counts."""
    s = re.sub(r'\s+', ' ', text).strip()
    words = [w for w in s.split() if len(re.findall(r'[A-Za-z]', w)) >= 2]
    if len(words) >= 4 or len(_SENTENCE_END.findall(s)) >= 1:
        return True
    return bool(re.search(r'[A-Za-z]{3,}[^\s\d]*[.!?]$', s))


# ----------------------------------------------------------------------------- classifier
def classify(answer, task, question, dataset_vocab=(), source_vocab=frozenset(), source_letters=False):
    """Classify one raw answer to a question of `task`.

    dataset_vocab: the answer vocabulary of the question's dataset (classification, unlettered).
    source_vocab: normalised labels the SOURCE adapter was trained to output (empty for non-label
    sources); source_letters: whether bare option letters were among them (classification source).
    Returns dict(format, matches = every rule that matched before precedence, detail flags)."""
    a = strip_think(answer)
    s = re.sub(r'\s+', ' ', a).strip()
    detail = {}
    matches = []
    if task == 'classification':
        opts = class_options(question, dataset_vocab)
        hit = match_class(s, opts)
        if hit:
            matches.append('class_label')
            detail['class_form'] = hit[0]
            detail['class_key'] = hit[1]
        pieces = [x for x in re.split(r'[;,]', s) if x.strip()]
        if len(pieces) >= 2 and all(match_class(x, opts) for x in pieces):
            matches.append('label_set')
    elif task == 'multi-label classification':
        labels = parse_labels(s, ml_options(question))
        if labels:
            matches.append('class_label' if len(labels) == 1 else 'label_set')
            detail['n_labels'] = len(labels)
    kind = box_kind(s) if s else None
    if kind:
        matches.append('boxes')
        detail['box_kind'] = kind
    if _INT_RE.fullmatch(s):
        matches.append('integer')
    if _DEC_RE.fullmatch(s):
        matches.append('decimal')
    if s and source_vocab is not None:
        bare = norm(s)
        foreign = source_letters and _LETTER_ANSWER.fullmatch(bare) is not None and len(bare) <= 3
        if not foreign and source_vocab:
            stripped = re.sub(r'^\(?[a-k]\)?\s*[.):]\s*', '', bare) if source_letters else bare
            labels = parse_labels(stripped, source_vocab)
            foreign = labels is not None and 1 <= len(labels) <= 3
        if foreign:
            matches.append('foreign_label')
    if s and is_free_text(s):
        matches.append('free_text')
    fmt = next((f for f in FORMATS if f in matches), 'other')
    return {'format': fmt, 'matches': matches, **detail}


def gold_cardinality(gold, question):
    labels = parse_labels(gold, ml_options(question))
    if labels is None:
        return len([x for x in re.split(r';' if ';' in str(gold) else r',', str(gold)) if x.strip()])
    return len(labels)


def assess(answer, row, source, dataset_vocab=(), source_vocab=frozenset(), source_letters=False):
    """F1 verdict for one off-task answer: format, valid-for-target, locked-in, and the extra flags
    (non-canonical class label, under-cardinality on multi-label, integer on regression, flat boxes)."""
    target = row['task']
    c = classify(answer, target, row['question'], dataset_vocab, source_vocab, source_letters)
    fmt = c['format']
    valid = fmt in VALID[target]
    if target == 'detection' and c.get('box_kind') == 'flat':
        valid = False  # the scorer needs nested boxes; flat lists are reported separately
    locked = (fmt == SOURCE_FORMAT[source] and not valid) or fmt == 'foreign_label'
    out = {'format': fmt, 'valid': valid, 'locked_in': locked, 'n_matches': len(c['matches']),
           'matches': c['matches']}
    if target == 'classification':
        out['noncanonical'] = fmt == 'class_label' and c.get('class_form') == 'noncanonical'
        out['class_key'] = c.get('class_key')
    if target == 'multi-label classification':
        gold_n = gold_cardinality(row['answer'], row['question'])
        out['gold_multi'] = gold_n >= 2
        out['under_cardinality'] = fmt == 'class_label' and gold_n >= 2
    if target == 'regression':
        out['integer_on_regression'] = fmt == 'integer'
    if target == 'detection':
        out['flat_boxes'] = c.get('box_kind') == 'flat'
    return out
