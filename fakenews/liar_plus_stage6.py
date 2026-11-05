"""Stage 6 (research): LIAR + LIAR-PLUS justifications.

LIAR-PLUS (Alhindi et al., 2018) adds to each LIAR claim the justification
sentences extracted from the PolitiFact verdict article. The cleaned stage-1
splits (reports/liar/split_manifest.csv) are kept unchanged; the justification
is joined by claim id. Hyperparameters are chosen on validation macro-F1 only,
and the test split is evaluated once per model.

Caveat: the justification is written by the fact-checker *after* checking the
claim, so these models are not usable for brand-new claims. They measure how
much the evidence text helps, like stage 5 does for social context.

    python -m fakenews.liar_plus_stage6     # writes reports/v6/
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
PLUS = ROOT / 'data/liar_plus'
OUT = ROOT / 'reports/v6'
LABELS = ['pants-fire', 'false', 'barely-true', 'half-true', 'mostly-true', 'true']
COLUMNS = ['row', 'id', 'label', 'statement', 'subject', 'speaker', 'job', 'state', 'party',
           'barely_true_c', 'false_c', 'half_true_c', 'mostly_true_c', 'pants_fire_c', 'context', 'justification']
C_GRID = [0.25, 0.5, 1, 2, 4, 8]


def load():
    manifest = pd.read_csv(ROOT / 'reports/liar/split_manifest.csv')
    plus = pd.concat([pd.read_csv(PLUS / f'{name}.tsv', sep='\t', header=None, names=COLUMNS,
                                  quoting=3, dtype=str, keep_default_na=False)
                      for name in ('train2', 'val2', 'test2')])
    plus = plus.drop_duplicates('id').set_index('id')
    frame = manifest.join(plus[['justification', 'subject', 'speaker', 'job', 'state', 'party', 'context']], on='id')
    for column in ['justification', 'subject', 'speaker', 'job', 'state', 'party', 'context']:
        frame[column] = frame[column].fillna('').astype(str)
    return frame


def meta_text(frame):
    # Metadata as tokens; credit-history counts are deliberately not used (they include the test label).
    return ('subj_' + frame.subject.str.replace(',', ' subj_') + ' spk_' + frame.speaker + ' party_' + frame.party
            + ' state_' + frame.state.str.replace(' ', '_') + ' ctx ' + frame.context + ' job ' + frame.job)


FEATURES = {
    'claim': ['text'],
    'claim + metadata': ['text', 'meta'],
    'claim + justification': ['text', 'justification'],
    'claim + metadata + justification': ['text', 'meta', 'justification'],
}


def build(train, frames, parts):
    blocks = {'text': TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
              'meta': TfidfVectorizer(token_pattern=r'[^\s]+', min_df=2, sublinear_tf=True),
              'justification': TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200000, sublinear_tf=True)}
    columns = {'text': lambda f: f.text, 'meta': meta_text, 'justification': lambda f: f.justification}
    fitted = {p: blocks[p].fit(columns[p](train)) for p in parts}
    return [sparse.hstack([fitted[p].transform(columns[p](f)) for p in parts], format='csr') for f in frames]


def scores(y, pred):
    return {'accuracy': round(accuracy_score(y, pred), 4), 'macro_f1': round(f1_score(y, pred, average='macro'), 4)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frame = load()
    train, valid, test = (frame[frame.split == s] for s in ('train', 'valid', 'test'))
    coverage = {s: round(float((frame[frame.split == s].justification != '').mean()), 4) for s in ('train', 'valid', 'test')}
    results, predictions = [], {'id': test.id.values, 'label': test.label.values}
    for name, parts in FEATURES.items():
        Xtr, Xva, Xte = build(train, [train, valid, test], parts)
        best = max(C_GRID, key=lambda C: f1_score(valid.label, LogisticRegression(C=C, max_iter=3000, class_weight='balanced')
                                                     .fit(Xtr, train.label).predict(Xva), average='macro'))
        model = LogisticRegression(C=best, max_iter=3000, class_weight='balanced').fit(Xtr, train.label)
        val_pred, test_pred = model.predict(Xva), model.predict(Xte)
        binary = lambda labels: np.isin(labels, ['true', 'mostly-true', 'half-true'])
        row = {'model': name, 'C': best, 'val_macro_f1': scores(valid.label, val_pred)['macro_f1'],
               **{f'test_{k}': v for k, v in scores(test.label, test_pred).items()},
               'test_binary_accuracy': round(float((binary(test.label.values) == binary(test_pred)).mean()), 4)}
        results.append(row)
        predictions[name] = test_pred
        print(row, flush=True)
    best = max(results, key=lambda r: r['val_macro_f1'])
    matrix = confusion_matrix(test.label, predictions[best['model']], labels=LABELS).tolist()
    summary = {'selected_on_validation': best['model'], 'labels': LABELS, 'confusion_matrix_selected': matrix,
               'justification_coverage': coverage, 'rows': {s: int((frame.split == s).sum()) for s in ('train', 'valid', 'test')},
               'results': results,
               'note': 'Justifications come from the fact-check article; research result, not usable for new claims.'}
    pd.DataFrame(results).to_csv(OUT / 'liar_plus_results.csv', index=False)
    pd.DataFrame(predictions).to_csv(OUT / 'liar_plus_test_predictions.csv', index=False)
    (OUT / 'liar_plus_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps({'coverage': coverage, 'selected': best}, indent=1))


if __name__ == '__main__':
    main()
