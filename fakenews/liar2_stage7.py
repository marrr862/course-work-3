"""Stage 7: more and newer fact-checks from LIAR2 (PolitiFact, up to July 2023).

LIAR2 (Xu & Kechadi, 2024; github.com/chengxuphd/liar2) has ~23k PolitiFact claims with
dates from 2007 to 2023-07. About 9.6k of them are the original LIAR claims under new ids,
so claims are matched to our LIAR splits by normalised text, never by id.

Two experiments, both TF-IDF + logistic regression with C chosen on validation macro-F1 only:

  E1 "more data": train on LIAR train + every LIAR2 claim that is not in our LIAR
     validation/test; evaluate on our unchanged LIAR test (comparable with stages 1-6).
  E2 "recent claims": split LIAR2 by date (train <= 2021, validation 2022-01..2022-06,
     test 2022-07..2023-07) and compare a model trained on old LIAR only with one
     trained on newer data, on the same recent test set.

Labels: LIAR2 0..5 = pants-fire, false, barely-true, half-true, mostly-true, true
(verified against the shared LIAR claims). The credit-history counts are not used: they
include the claim's own label. Justifications are not used: they are written after the check.

    python -m fakenews.liar2_stage7          # writes reports/v7/
    python -m fakenews.liar2_stage7 --build  # also saves models/liar2_stage7.joblib
"""
import argparse
import json
import re
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/v7'
LABELS = ['pants-fire', 'false', 'barely-true', 'half-true', 'mostly-true', 'true']
TRUEISH = {'half-true', 'mostly-true', 'true'}
C_GRID = [0.25, 0.5, 1, 2, 4]
LIAR_COLUMNS = ['id', 'label', 'statement', 'subject', 'speaker', 'job', 'state', 'party',
                'bt', 'f', 'ht', 'mt', 'pof', 'context']


def norm_key(text):
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9 ]', ' ', str(text).lower())).strip()


def load_liar():
    manifest = pd.read_csv(ROOT / 'reports/liar/split_manifest.csv')
    raw = pd.concat([pd.read_csv(ROOT / f'data/raw/liar/{s}.tsv', sep='\t', header=None, names=LIAR_COLUMNS,
                                 quoting=3, dtype=str, keep_default_na=False) for s in ['train', 'valid', 'test']])
    frame = manifest.merge(raw[['id', 'subject', 'speaker', 'context']].drop_duplicates('id'), on='id', how='left')
    frame = frame.fillna('')
    frame['subject'] = frame.subject.str.replace(',', ';')
    frame['date'] = pd.NaT
    frame['origin'] = 'liar'
    return frame[['text', 'label', 'split', 'subject', 'speaker', 'context', 'date', 'origin']]


def load_liar2():
    frames = []
    for split in ['train', 'valid', 'test']:
        d = pd.read_csv(ROOT / f'data/liar2/{split}.csv')
        frames.append(pd.DataFrame({'text': d.statement.astype(str), 'label': d.label.map(dict(enumerate(LABELS))),
                                    'split': 'liar2_' + split, 'subject': d.subject.fillna('').astype(str),
                                    'speaker': d.speaker.fillna('').astype(str), 'context': d.context.fillna('').astype(str),
                                    'date': pd.to_datetime(d.date, errors='coerce'), 'origin': 'liar2'}))
    return pd.concat(frames, ignore_index=True)


def meta_text(frame):
    return ('spk_' + frame.speaker.str.replace(' ', '-') + ' ' + frame.subject.str.replace(' ', '_').str.replace(';', ' ')
            + ' ctx ' + frame.context)


def features(train, frames, with_meta):
    text = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(train.text)
    blocks = [lambda f: text.transform(f.text)]
    if with_meta:
        meta = TfidfVectorizer(token_pattern=r'[^\s]+', min_df=2, sublinear_tf=True).fit(meta_text(train))
        blocks.append(lambda f: meta.transform(meta_text(f)))
    vectorizers = [text] + ([meta] if with_meta else [])
    return [sparse.hstack([b(f) for b in blocks], format='csr') for f in frames], vectorizers


def scores(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    return {'accuracy': round(accuracy_score(y, pred), 4), 'macro_f1': round(f1_score(y, pred, average='macro'), 4),
            'binary_accuracy': round(float((np.isin(y, list(TRUEISH)) == np.isin(pred, list(TRUEISH))).mean()), 4)}


def fit_select(train, valid, test, with_meta):
    (Xtr, Xva, Xte), vectorizers = features(train, [train, valid, test], with_meta)
    best = max(C_GRID, key=lambda C: f1_score(valid.label, LogisticRegression(C=C, max_iter=3000, class_weight='balanced')
                                                 .fit(Xtr, train.label).predict(Xva), average='macro'))
    model = LogisticRegression(C=best, max_iter=3000, class_weight='balanced').fit(Xtr, train.label)
    return {'C': best, 'train_rows': len(train), 'test_rows': len(test), 'val': scores(valid.label, model.predict(Xva)),
            'test': scores(test.label, model.predict(Xte))}, model, vectorizers


def experiments():
    liar, liar2 = load_liar(), load_liar2()
    liar['key'], liar2['key'] = liar.text.map(norm_key), liar2.text.map(norm_key)
    liar2 = liar2.drop_duplicates('key')
    heldout = set(liar.key[liar.split != 'train'])
    known = set(liar.key)
    results, notes = [], {}

    # E1: more data, original LIAR test.
    train_l = liar[liar.split == 'train']
    valid_l, test_l = liar[liar.split == 'valid'], liar[liar.split == 'test']
    extra = liar2[~liar2.key.isin(known)]                       # claims LIAR does not have at all
    notes['E1'] = {'liar2_rows_removed_as_heldout_overlap': int(liar2.key.isin(heldout).sum()),
                   'liar2_new_claims_added': len(extra)}
    for with_meta in (False, True):
        name = 'claim + metadata' if with_meta else 'claim'
        base, _, _ = fit_select(train_l, valid_l, test_l, with_meta)
        more, _, _ = fit_select(pd.concat([train_l, extra]), valid_l, test_l, with_meta)
        results.append({'experiment': 'E1 original LIAR test', 'features': name, 'training': 'LIAR train', **flat(base)})
        results.append({'experiment': 'E1 original LIAR test', 'features': name, 'training': 'LIAR train + LIAR2 new', **flat(more)})
        print(results[-2], '\n', results[-1], flush=True)

    # E2: recent claims (LIAR2 by date). Old LIAR (2007-2016) has no dates and is used only as the "old data" baseline.
    dated = liar2[liar2.date.notna()]
    train_r = dated[dated.date < '2022-01-01']
    valid_r = dated[(dated.date >= '2022-01-01') & (dated.date < '2022-07-01')]
    test_r = dated[dated.date >= '2022-07-01']
    old_only = liar[(liar.split == 'train') & ~liar.key.isin(set(valid_r.key) | set(test_r.key))]
    notes['E2'] = {'train_until': '2021-12-31', 'valid': '2022-01..2022-06', 'test': '2022-07..2023-07',
                   'rows': {'train': len(train_r), 'valid': len(valid_r), 'test': len(test_r)}}
    for with_meta in (False, True):
        name = 'claim + metadata' if with_meta else 'claim'
        old, _, _ = fit_select(old_only, valid_r, test_r, with_meta)
        new, _, _ = fit_select(train_r, valid_r, test_r, with_meta)
        results.append({'experiment': 'E2 recent claims 2022-07..2023-07', 'features': name, 'training': 'old LIAR only (2007-2016)', **flat(old)})
        results.append({'experiment': 'E2 recent claims 2022-07..2023-07', 'features': name, 'training': 'LIAR2 until 2021', **flat(new)})
        print(results[-2], '\n', results[-1], flush=True)
    return results, notes, (liar, liar2, extra, heldout)


def flat(run):
    return {'C': run['C'], 'train_rows': run['train_rows'], 'test_rows': run['test_rows'],
            'val_macro_f1': run['val']['macro_f1'], **{f'test_{k}': v for k, v in run['test'].items()}}


def build(data):
    """Deployable model: claim text only, trained on every claim outside our LIAR validation/test."""
    liar, liar2, extra, heldout = data
    pool = pd.concat([liar[liar.split == 'train'], extra])
    valid = liar[liar.split == 'valid']
    run, model, vectorizers = fit_select(pool, valid, liar[liar.split == 'test'], with_meta=False)
    artifact = {'vectorizer': vectorizers[0], 'model': model, 'labels': list(model.classes_),
                'trained_on': 'LIAR train + LIAR2 claims not in LIAR valid/test (PolitiFact, up to 2023-07)',
                'test_on_original_liar': run['test'], 'C': run['C']}
    joblib.dump(artifact, ROOT / 'models/liar2_stage7.joblib', compress=3)
    print('saved models/liar2_stage7.joblib', run['test'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', action='store_true', help='also save models/liar2_stage7.joblib')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    results, notes, data = experiments()
    pd.DataFrame(results).to_csv(OUT / 'liar2_results.csv', index=False)
    (OUT / 'liar2_summary.json').write_text(json.dumps({'results': results, 'notes': notes}, indent=2, default=str))
    print(json.dumps(notes, indent=1, default=str))
    if args.build:
        build(data)


if __name__ == '__main__':
    main()
