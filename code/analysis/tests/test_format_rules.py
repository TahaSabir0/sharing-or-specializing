"""F0 unit tests (results/format_analysis_plan.md). Run: python -m pytest code/analysis/tests -q"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import format_rules as F  # noqa: E402

CLS_Q = ('Question: What is the most likely diagnosis based on the ultrasound findings? '
         'Options: A) Benign lesion B) Malignant lesion C) No lesion')
CLS_Q2 = ('How can the tooth loss pattern in the mandibular jaw be classified? A: No loss, '
          'B: Single tooth loss, C: Consecutive multiple loss, D: Edentulism')
CLS_Q3 = ('Which of the following is the correct diabetic retinopathy grade for this image? A. No DR '
          '(No Diabetic Retinopathy) B. Mild NPDR (Non-Proliferative Diabetic Retinopathy) C. Moderate NPDR '
          'D. Severe NPDR E. PDR (Proliferative Diabetic Retinopathy)')
CMMD_Q = 'According to the visual evidence, is this tumor Benign or Malignant?'
CMMD_VOCAB = ['Benign', 'Malignant', 'Mass', 'Calcification', 'Both', 'Luminal a']
DR_NUM_Q = 'Grade the diabetic retinopathy in this image on a scale from 0 to 4.'
DR_NUM_VOCAB = ['0', '1', '2', '3', '4']
ML_Q = ('Which of the following conditions are visible in this chest radiograph? Select all that apply '
        'from the following: Pleural Effusion, Nodule, Pneumonia, Cardiomegaly, Normal')
PERI_Q = ('What abnormalities does tooth 36 show in this radiograph? Select all that apply from the '
          'following, separate multiple selections with a semicolon: Dental Caries; Periradicular Lesions; '
          'Pulp Diseases; Endodontic Treatment, Restorative Treatment, and Complications.')
DET_Q = 'For the ultrasound image shown, what are the coordinates of fetal head?'
REG_Q = 'Find the Angle of Progression (AoP) value in degrees for this ultrasound scan.'
CNT_Q = 'Compute the aggregate number of cells in this microscopic image'
CLS_VOCAB = frozenset({'benign lesion', 'malignant lesion', 'no lesion', 'benign', 'malignant', 'mass'})
ML_VOCAB = frozenset({'pleural effusion', 'nodule', 'cardiomegaly', 'dental caries', 'normal'})


def fmt(answer, task, q, **kw):
    return F.classify(answer, task, q, **kw)['format']


# ------------------------------------------------------------------ the eight formats
def test_class_label_letter_canonical():
    c = F.classify('B', 'classification', CLS_Q)
    assert c['format'] == 'class_label' and c['class_form'] == 'canonical' and c['class_key'] == 'b'


def test_class_label_letter_plus_text_is_noncanonical():
    c = F.classify('C. Moderate NPDR', 'classification', CLS_Q3)
    assert c['format'] == 'class_label' and c['class_form'] == 'noncanonical' and c['class_key'] == 'c'
    c = F.classify('C: Consecutive multiple loss', 'classification', CLS_Q2)
    assert c['format'] == 'class_label' and c['class_key'] == 'c'


def test_class_label_text_only_is_noncanonical():
    c = F.classify('Malignant lesion', 'classification', CLS_Q)
    assert c['format'] == 'class_label' and c['class_form'] == 'noncanonical' and c['class_key'] == 'b'


def test_class_label_unlettered_dataset():
    c = F.classify('Malignant', 'classification', CMMD_Q, dataset_vocab=CMMD_VOCAB)
    assert c['format'] == 'class_label' and c['class_form'] == 'canonical'
    # a label of the same dataset that the question does not offer is not an option here
    assert fmt('Mass', 'classification', CMMD_Q, dataset_vocab=CMMD_VOCAB) != 'class_label'


def test_label_text_not_an_option_is_not_class_label():
    assert fmt('Benign', 'classification', CLS_Q) != 'class_label'
    assert fmt('F', 'classification', CLS_Q) != 'class_label'


def test_label_set_multilabel():
    c = F.classify('Pleural Effusion, Cardiomegaly', 'multi-label classification', ML_Q)
    assert c['format'] == 'label_set' and c['n_labels'] == 2


def test_label_set_semicolons_and_commas_inside_label():
    c = F.classify('Periradicular Lesions; Endodontic Treatment, Restorative Treatment, and Complications',
                   'multi-label classification', PERI_Q)
    assert c['format'] == 'label_set' and c['n_labels'] == 2


def test_label_with_internal_commas_is_one_label():
    c = F.classify('Endodontic Treatment, Restorative Treatment, and Complications',
                   'multi-label classification', PERI_Q)
    assert c['format'] == 'class_label' and c['n_labels'] == 1


def test_label_set_on_classification_question():
    assert fmt('A, C', 'classification', CLS_Q) == 'label_set'


def test_boxes():
    assert fmt('[[237,35,313,76]]', 'detection', DET_Q) == 'boxes'
    assert fmt('[[28, 107, 98, 157], [198, 126, 298, 221]]', 'detection', DET_Q) == 'boxes'


def test_integer_decimal():
    assert fmt('14', 'counting', CNT_Q) == 'integer'
    assert fmt('122.32', 'regression', REG_Q) == 'decimal'
    assert fmt('.5', 'regression', REG_Q) == 'decimal'


def test_foreign_label():
    # classification adapter answering a counting question with one of its own labels / letters
    assert fmt('Malignant lesion', 'counting', CNT_Q, source_vocab=CLS_VOCAB, source_letters=True) == 'foreign_label'
    assert fmt('B', 'regression', REG_Q, source_vocab=CLS_VOCAB, source_letters=True) == 'foreign_label'
    assert fmt('C. Moderate NPDR', 'counting', CNT_Q,
               source_vocab=CLS_VOCAB | {'moderate npdr'}, source_letters=True) == 'foreign_label'
    # multi-label adapter answering a classification question with its own labels
    assert fmt('Cardiomegaly, Nodule', 'classification', CLS_Q, source_vocab=ML_VOCAB) == 'foreign_label'


def test_free_text():
    assert fmt('The image shows numerous cells, some of which are in clusters.', 'counting', CNT_Q) == 'free_text'
    assert fmt('The HSD is 2.2 cm.', 'regression', REG_Q) == 'free_text'
    long = ' '.join(['word'] * 16)
    assert fmt(long, 'classification', CLS_Q) == 'free_text'


def test_short_report_sentences_are_free_text():
    for s in ('Heart size normal.', 'Cardiomegaly.', 'Lungs are clear.', 'No acute findings.'):
        assert fmt(s, 'counting', CNT_Q) == 'free_text'
        assert F.assess(s, {'task': 'report_generation', 'question': 'x', 'answer': 'y'}, 'report_generation')['valid']


def test_box_list_is_not_free_text():
    c = F.classify('[[28, 107, 98, 157], [198, 126, 298, 221]]', 'detection', DET_Q)
    assert c['matches'] == ['boxes']
    assert 'free_text' not in F.classify('100.66', 'regression', REG_Q)['matches']


def test_other():
    assert fmt('Angle of Progression', 'regression', REG_Q) == 'other'
    assert fmt('fetal head', 'detection', DET_Q) == 'other'


# ------------------------------------------------------------------ edge cases from the plan
def test_numeric_class_labels_dr_grades():
    c = F.classify('2', 'classification', DR_NUM_Q, dataset_vocab=DR_NUM_VOCAB)
    assert c['format'] == 'class_label' and 'integer' in c['matches']  # option check wins
    assert F.assess('2', {'task': 'classification', 'question': DR_NUM_Q, 'answer': '2'}, 'counting',
                    dataset_vocab=DR_NUM_VOCAB)['valid']


def test_single_label_on_multilabel_question_is_valid_and_under_cardinality():
    row = {'task': 'multi-label classification', 'question': ML_Q, 'answer': 'Pleural Effusion, Cardiomegaly'}
    a = F.assess('Cardiomegaly', row, 'classification')
    assert a['format'] == 'class_label' and a['valid'] and not a['locked_in']
    assert a['gold_multi'] and a['under_cardinality']
    row1 = dict(row, answer='Cardiomegaly')
    a1 = F.assess('Cardiomegaly', row1, 'classification')
    assert not a1['gold_multi'] and not a1['under_cardinality']


def test_integer_on_regression_is_valid_and_flagged():
    row = {'task': 'regression', 'question': REG_Q, 'answer': '122.32'}
    a = F.assess('100', row, 'counting')
    assert a['format'] == 'integer' and a['valid'] and not a['locked_in'] and a['integer_on_regression']


def test_decimal_on_counting_is_locked_in_for_regression_source():
    row = {'task': 'counting', 'question': CNT_Q, 'answer': '7'}
    a = F.assess('80.0', row, 'regression')
    assert a['format'] == 'decimal' and not a['valid'] and a['locked_in']


def test_boxes_with_extra_spaces_and_trailing_comma():
    assert fmt('  [ [ 1, 2, 3, 4 ] , [5,6,7,8], ]  ', 'detection', DET_Q) == 'boxes'
    assert fmt('[[1, 2, 3, 4],]', 'detection', DET_Q) == 'boxes'


def test_flat_box_is_boxes_but_not_valid_detection():
    c = F.classify('[389, 283, 508, 335]', 'detection', DET_Q)
    assert c['format'] == 'boxes' and c['box_kind'] == 'flat'
    a = F.assess('[389, 283, 508, 335]', {'task': 'detection', 'question': DET_Q, 'answer': '[[1,2,3,4]]'},
                 'classification')
    assert not a['valid'] and a['flat_boxes']


def test_box_list_is_not_a_number():
    # the scorer's number parser would read 162 out of this; we must not
    c = F.classify('[[162,118,360,370]]', 'regression', REG_Q)
    assert c['format'] == 'boxes' and 'integer' not in c['matches'] and 'decimal' not in c['matches']
    a = F.assess('[[162,118,360,370]]', {'task': 'regression', 'question': REG_Q, 'answer': '1.0'}, 'detection')
    assert a['locked_in'] and not a['valid']


def test_number_inside_sentence():
    assert fmt('There are 14 cells.', 'counting', CNT_Q) == 'free_text'
    assert fmt('14 cells', 'counting', CNT_Q) == 'other'
    assert fmt('10.6cm', 'regression', REG_Q) == 'other'
    assert fmt('50%', 'regression', REG_Q) == 'other'


def test_empty_answers():
    for empty in ('', '   ', None, '<think>\n\n</think>\n\n'):
        assert fmt(empty, 'counting', CNT_Q) == 'other'
        assert fmt(empty, 'classification', CLS_Q) == 'other'


def test_think_block_stripped():
    assert fmt('<think>\n\n</think>\n\nB', 'classification', CLS_Q) == 'class_label'


def test_label_from_same_task_other_dataset_is_not_class_label():
    # a valid label of another classification dataset (BUSI), not an option of this lettered question
    assert fmt('Standard plane', 'classification', CLS_Q) != 'class_label'


def test_label_both_option_and_source_vocab_resolves_to_class_label():
    c = F.classify('Malignant lesion', 'classification', CLS_Q, source_vocab=CLS_VOCAB, source_letters=True)
    assert c['format'] == 'class_label' and 'foreign_label' in c['matches']


def test_report_source_free_text_locked_in_on_counting():
    a = F.assess('The number of cells is within normal limits.', {'task': 'counting', 'question': CNT_Q,
                 'answer': '7'}, 'report_generation')
    assert a['format'] == 'free_text' and a['locked_in']


def test_free_text_valid_for_report():
    a = F.assess('The lungs are clear. No effusion.', {'task': 'report_generation', 'question': 'x',
                 'answer': 'y'}, 'classification')
    assert a['valid'] and not a['locked_in']


def test_letter_options_parsing():
    assert F.letter_options(CLS_Q) == {'a': 'benign lesion', 'b': 'malignant lesion', 'c': 'no lesion'}
    assert F.letter_options(CLS_Q2)['d'] == 'edentulism'
    assert F.letter_options('Kennedy class? A: Class I, B: Class II, C: Class III')['a'] == 'class i'
    assert F.letter_options(CMMD_Q) == {}


def test_ml_options_parsing():
    assert F.ml_options(PERI_Q)[-1] == 'endodontic treatment, restorative treatment, and complications'
    assert F.ml_options(ML_Q) == ['pleural effusion', 'nodule', 'pneumonia', 'cardiomegaly', 'normal']


def test_formats_are_exhaustive_and_exclusive():
    for ans in ('B', '[[1,2,3,4]]', '3', '3.5', 'x', '', 'A, B', 'Some words in a row here'):
        r = F.classify(ans, 'classification', CLS_Q)
        assert r['format'] in F.FORMATS
