"""Stage 5 (research): social-context features from FakeNewsNet tweet IDs, without Twitter API.

Every tweet ID after November 2010 is a Snowflake ID whose upper bits store the creation time:
t_ms = (id >> 22) + 1288834974657. From the list of IDs we derive propagation features
(how many tweets, how fast and how long the headline was shared). Absolute dates are NOT used:
real and fake items were collected over different years, so the year would leak collection
artefacts instead of propagation behaviour (reported separately as a leakage check).

A gradient-boosting model is trained on train, tuned on validation; its P(fake) is added as a
component to the stage-4 meta-learner (5-fold CV on validation). Test is evaluated once.
The "early" variant uses only tweets within the first 24 hours after the first tweet.

    python -m fakenews.social          # writes reports/v5/
"""
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from .improve_v2 import ROOT, M, paired_ci, best_threshold
from .stack_v4 import fnn_components, cv_meta, logit, NOT_DEPLOYABLE

EPOCH_MS = 1288834974657
FIRST_SNOWFLAKE = 29700859247  # IDs below this are sequential and carry no timestamp
OUT = ROOT / 'reports/v5'
HOURS = [1, 6, 24, 24 * 7]


def tweet_times(ids):
    """Creation times in hours (relative to the first timestamped tweet), sorted."""
    ts = np.sort([((i >> 22) + EPOCH_MS) / 3.6e6 for i in ids if i > FIRST_SNOWFLAKE])
    return ts - ts[0] if len(ts) else ts


def propagation_features(tweet_ids, horizon=None):
    ids = [int(x) for x in str(tweet_ids).split()] if isinstance(tweet_ids, str) and tweet_ids.strip() else []
    t = tweet_times(ids)
    if horizon is not None:
        t = t[t <= horizon]
        n = len(t)
    else:
        n = len(ids)
    f = {'log_tweets': np.log1p(n), 'has_tweets': float(n > 0), 'untimed_share': 0.0 if horizon else (1 - len(t) / n if n else 0.0)}
    if len(t):
        gaps = np.diff(t)
        span = t[-1]
        f.update({'log_span_h': np.log1p(span),
                  'log_median_gap_h': np.log1p(np.median(gaps)) if len(gaps) else 0.0,
                  'burstiness': (gaps.std() - gaps.mean()) / (gaps.std() + gaps.mean()) if len(gaps) > 1 and gaps.std() + gaps.mean() > 0 else 0.0,
                  'log_t50_h': np.log1p(np.quantile(t, .5)), 'log_t90_h': np.log1p(np.quantile(t, .9))})
        for h in HOURS:
            f[f'share_first_{h}h'] = float((t <= h).mean())
            f[f'log_tweets_first_{h}h'] = np.log1p((t <= h).sum())
    else:
        f.update({k: 0.0 for k in ['log_span_h', 'log_median_gap_h', 'burstiness', 'log_t50_h', 'log_t90_h']})
        for h in HOURS:
            f[f'share_first_{h}h'] = 0.0; f[f'log_tweets_first_{h}h'] = 0.0
    return f


def first_year(tweet_ids):
    ids = [int(x) for x in str(tweet_ids).split()] if isinstance(tweet_ids, str) else []
    ts = [(i >> 22) + EPOCH_MS for i in ids if i > FIRST_SNOWFLAKE]
    return pd.to_datetime(min(ts), unit='ms').year if ts else 0


def load_frame():
    man = pd.read_csv(ROOT / 'reports/fakenewsnet/split_manifest.csv')
    tw = pd.concat([pd.read_csv(ROOT / f'data/raw/fakenewsnet/{s}_{l}.csv', usecols=['id', 'tweet_ids'], dtype=str)
                    for s in ['politifact', 'gossipcop'] for l in ['fake', 'real']]).drop_duplicates('id')
    df = man.merge(tw, on='id', how='left'); assert len(df) == len(man)
    return df


def feature_matrix(df, horizon=None, with_year=False):
    X = pd.DataFrame([propagation_features(t, horizon) for t in df.tweet_ids])
    X['is_gossipcop'] = (df.source == 'gossipcop').astype(float).values
    if with_year:
        X['first_year'] = df.tweet_ids.map(first_year).values
    return X


def fit_social(Xtr, ytr, Xva, yva):
    """Small grid on validation; returns (val macro-F1, model, threshold)."""
    best = None
    for lr in [0.03, 0.1]:
        for leaves in [15, 31]:
            for l2 in [0.0, 1.0]:
                m = HistGradientBoostingClassifier(learning_rate=lr, max_leaf_nodes=leaves, l2_regularization=l2,
                                                   max_iter=300, class_weight='balanced', random_state=42).fit(Xtr, ytr)
                p = m.predict_proba(Xva)[:, list(m.classes_).index('fake')]
                t, f = best_threshold(yva, p)
                if best is None or f > best[0]:
                    best = (f, m, t)
    return best


def proba_fake(m, X):
    return m.predict_proba(X)[:, list(m.classes_).index('fake')]


def stack(yva, yte, comps, names):
    Xva = np.column_stack([logit(comps[n][0]) for n in names]); Xte = np.column_stack([logit(comps[n][1]) for n in names])
    f, C, t, _ = cv_meta(Xva, yva)
    meta = LogisticRegression(C=C, max_iter=2000).fit(Xva, yva)
    p = meta.predict_proba(Xte)[:, list(meta.classes_).index('fake')]
    coef = dict(zip(names, np.round(meta.coef_[0] * (1 if list(meta.classes_).index('fake') == 1 else -1), 3).tolist()))
    return f, np.where(p >= t, 'fake', 'real'), p, {'C': C, 'threshold': t, 'coef': coef}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_frame()
    tr, va, te = [df[df.split == s].reset_index(drop=True) for s in ['train', 'valid', 'test']]
    ytr, yva, yte = tr.label.values, va.label.values, te.label.values
    rows, preds, social = [], {}, {}

    for name, horizon, year in [('social', None, False), ('social_24h', 24, False), ('social_with_year', None, True)]:
        Xtr, Xva, Xte = (feature_matrix(d, horizon, year) for d in (tr, va, te))
        f, m, t = fit_social(Xtr, ytr, Xva, yva)
        pva, pte = proba_fake(m, Xva), proba_fake(m, Xte)
        social[name] = (pva, pte)
        pred = np.where(pte >= t, 'fake', 'real'); preds[f'single:{name}'] = pred
        rows.append({'model': f'{name} only (gradient boosting)', 'val_macro_f1': round(f, 4), **M(yte, pred, 'fake')})
        np.save(OUT / f'fakenewsnet_{name}_valid.npy', pva); np.save(OUT / f'fakenewsnet_{name}_test.npy', pte)
        if name == 'social':
            from sklearn.inspection import permutation_importance
            imp = permutation_importance(m, Xva, yva, scoring='f1_macro', n_repeats=5, random_state=42)
            pd.Series(imp.importances_mean, index=Xva.columns).sort_values(ascending=False).round(4) \
                .to_csv(OUT / 'social_feature_importance.csv', header=['macro_f1_drop'])

    _, _, comps = fnn_components()
    comps = {n: c for n, c in comps.items() if not n.startswith(NOT_DEPLOYABLE)}
    text = [n for n, (_, _, u) in comps.items() if not u]
    url = list(comps)
    allc = {**comps, **{k: (v[0], v[1], False) for k, v in social.items()}}
    configs = [('stage4 text (reference)', text), ('stage4 text + url (reference)', url),
               ('text + social', text + ['social']), ('text + url + social', url + ['social']),
               ('text + social 24h', text + ['social_24h']), ('text + url + social 24h', url + ['social_24h'])]
    info = {}
    for label, names in configs:
        f, pred, p, extra = stack(yva, yte, allc, names)
        preds[label] = pred; info[label] = {'components': names, **extra}
        rows.append({'model': label, 'val_macro_f1': round(f, 4), **M(yte, pred, 'fake')})
        if label == 'text + url + social':
            np.save(OUT / 'fakenewsnet_v5_test_fake_proba.npy', p)

    res = pd.DataFrame(rows)
    res.to_csv(OUT / 'fakenewsnet_v5_results.csv', index=False)
    ci = {'text + social vs stage4 text': paired_ci(yte, preds['stage4 text (reference)'], preds['text + social']),
          'text + url + social vs stage4 text + url': paired_ci(yte, preds['stage4 text + url (reference)'], preds['text + url + social']),
          'text + url + social 24h vs stage4 text + url': paired_ci(yte, preds['stage4 text + url (reference)'], preds['text + url + social 24h'])}
    by_source = {}
    for label in ['stage4 text + url (reference)', 'text + url + social']:
        by_source[label] = {s: M(yte[te.source == s], preds[label][te.source.values == s], 'fake') for s in ['politifact', 'gossipcop']}
    te.assign(pred_stage4=preds['stage4 text + url (reference)'], pred_v5=preds['text + url + social'],
              n_tweets=te.tweet_ids.fillna('').str.split().str.len()) \
        .drop(columns=['tweet_ids', 'key']).to_csv(OUT / 'fakenewsnet_v5_test_predictions.csv', index=False)
    summary = {'results': rows, 'meta': info, 'macro_f1_diff_95ci': ci, 'by_source': by_source,
               'note': 'Social features: propagation statistics from Snowflake tweet IDs, no absolute dates. '
                       'social_with_year is a leakage check only. Thresholds/weights chosen on validation; test evaluated once.'}
    json.dump(summary, open(OUT / 'fakenewsnet_v5_summary.json', 'w'), indent=2, ensure_ascii=False)
    pd.set_option('display.width', 220)
    print(res.to_string(index=False)); print(json.dumps(ci, indent=1)); print(json.dumps(by_source, indent=1))


if __name__ == '__main__':
    main()
