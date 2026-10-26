"""Stage 4: combine all available models with a meta-learner chosen on the validation split.

Components are probability outputs already saved on validation and test. The meta-learner
(logistic regression on log-probabilities) is tuned with 5-fold cross-validation inside the
validation split, refit on the whole validation split and applied once to test. A simple
average and the best single component are evaluated the same way; the method with the best
cross-validated validation macro-F1 is selected.

    python -m fakenews.stack_v4 [--dataset fakenewsnet|liar]
"""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from .improve_v2 import ROOT, M, paired_ci, best_threshold, THRESHOLDS
from .stacking import domain

V4 = ROOT / 'reports/v4'


def logit(p):
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def fnn_components():
    man = pd.read_csv(ROOT / 'reports/fakenewsnet/split_manifest.csv')
    urls = pd.concat([pd.read_csv(ROOT / f'data/raw/fakenewsnet/{s}_{l}.csv', usecols=['id', 'news_url'])
                      for s in ['politifact', 'gossipcop'] for l in ['fake', 'real']]).drop_duplicates('id')
    df = man.merge(urls, on='id', how='left')
    tr, va, te = [df[df.split == s].reset_index(drop=True) for s in ['train', 'valid', 'test']]
    comps = {'stage1': (np.load(ROOT / 'reports/optimization/fakenewsnet/selected_validation_probabilities.npy')[:, 0],
                        np.load(ROOT / 'reports/optimization/fakenewsnet/test_probabilities.npy')[:, 0], False)}
    if (ROOT / 'reports/v3/fakenewsnet_large_valid.npy').exists():
        comps['roberta'] = (np.load(ROOT / 'reports/v3/fakenewsnet_large_valid.npy')[:, 0], np.load(ROOT / 'reports/v3/fakenewsnet_large_test.npy')[:, 0], False)
    for f in sorted(V4.glob('fakenewsnet_*_valid.npy')):
        tag = f.name[len('fakenewsnet_'):-len('_valid.npy')]
        uses_url = 'url' in tag
        comps[tag] = (np.load(f)[:, 0], np.load(V4 / f'fakenewsnet_{tag}_test.npy')[:, 0], uses_url)
    v2 = joblib.load(ROOT / 'models/fakenewsnet_v2.joblib')
    for kind in ['text', 'domain']:
        st = v2['models'][kind]
        comps[f'lr_{kind}'] = (st.lr_proba(st.features.frame(va.text.tolist(), va.news_url.tolist()))[:, 0],
                               st.lr_proba(st.features.frame(te.text.tolist(), te.news_url.tolist()))[:, 0], kind == 'domain')
    pva, pte, bundle = url_lr(tr, va, te)
    comps['lr_url'] = (pva, pte, True)
    fnn_components.lr_url_bundle = bundle
    return va.label.values, te.label.values, comps


def url_lr(tr, va, te):
    """Word+char TF-IDF of the headline, domain one-hot and TF-IDF of URL path words (train only)."""
    import scipy.sparse as sp
    from sklearn.feature_extraction.text import TfidfVectorizer
    from .stacking import FakeNewsFeatures
    from .stacking import url_words
    for d in (tr, va, te):
        d['key'] = d['key'] if 'key' in d else d.text
        d['domain'] = d.news_url.map(domain); d['path'] = d.news_url.map(url_words)
    feats = FakeNewsFeatures(True).fit(tr)
    pv = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(tr.path)
    X = lambda d: sp.hstack([feats.transform(d), pv.transform(d.path)]).tocsr()
    best = None
    for C in [1, 3, 10]:
        m = LogisticRegression(C=C, max_iter=4000, class_weight='balanced').fit(X(tr), tr.label)
        f = f1_score(va.label, m.predict(X(va)), average='macro')
        if best is None or f > best[0]:
            best = (f, m)
    m = best[1]; k = list(m.classes_).index('fake')
    from .stacking import UrlLR
    return m.predict_proba(X(va))[:, k], m.predict_proba(X(te))[:, k], UrlLR(feats, pv, m)


def cv_meta(Xva, yva, seed=42):
    """Out-of-fold P(fake) on validation for each C; returns best C and its OOF predictions."""
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    best = None
    for C in [0.03, 0.1, 0.3, 1, 3]:
        oof = np.zeros(len(yva))
        for a, b in skf.split(Xva, yva):
            m = LogisticRegression(C=C, max_iter=2000).fit(Xva[a], yva[a])
            oof[b] = m.predict_proba(Xva[b])[:, list(m.classes_).index('fake')]
        t, f = best_threshold(yva, oof)
        if best is None or f > best[0]:
            best = (f, C, t, oof)
    return best


def run_fnn(out):
    yva, yte, comps = fnn_components()
    rows, preds, chosen = [], {}, {}
    for mode in ['text', 'url']:
        names = [n for n, (_, _, u) in comps.items() if mode == 'url' or not u]
        Xva = np.column_stack([logit(comps[n][0]) for n in names]); Xte = np.column_stack([logit(comps[n][1]) for n in names])
        cands = []
        # best single component (threshold chosen on validation; CV score = same threshold search)
        for n in names:
            t, f = best_threshold(yva, comps[n][0])
            cands.append((f, f'single:{n}', lambda n=n, t=t: np.where(comps[n][1] >= t, 'fake', 'real'), {'component': n, 'threshold': t}))
        # simple average of probabilities
        avg_va = np.mean([comps[n][0] for n in names], 0); avg_te = np.mean([comps[n][1] for n in names], 0)
        t, f = best_threshold(yva, avg_va)
        cands.append((f, 'average', lambda t=t: np.where(avg_te >= t, 'fake', 'real'), {'threshold': t}))
        # cross-validated logistic meta-learner
        f, C, t, _ = cv_meta(Xva, yva)
        meta = LogisticRegression(C=C, max_iter=2000).fit(Xva, yva)
        k = list(meta.classes_).index('fake')
        cands.append((f, 'meta_lr', lambda: np.where(meta.predict_proba(Xte)[:, k] >= t, 'fake', 'real'),
                      {'C': C, 'threshold': t, 'coef': dict(zip(names, np.round(meta.coef_[0] * (1 if k == 1 else -1), 3).tolist()))}))
        for f, label, fn, info in cands:
            p = fn(); key = f'{mode}:{label}'; preds[key] = p
            rows.append({'mode': mode, 'method': label, 'val_score': round(f, 4), **M(yte, p, 'fake')})
        winner = max(cands, key=lambda c: c[0])
        chosen[mode] = {'method': winner[1], 'val_score': round(winner[0], 4), 'components': names, **winner[3]}
    res = pd.DataFrame(rows).sort_values(['mode', 'val_score'], ascending=[True, False])
    res.to_csv(out / 'fakenewsnet_v4_results.csv', index=False)
    base = preds.get('url:single:lr_domain')
    summary = {'selected': chosen, 'note': 'val_score: macro-F1 on validation (5-fold out-of-fold for meta_lr); test evaluated once.'}
    for mode in ['text', 'url']:
        key = f"{mode}:{chosen[mode]['method']}"
        summary[f'{mode}_selected_test'] = M(yte, preds[key], 'fake')
    stage3 = json.loads((ROOT / 'reports/v3/fakenewsnet_v3_summary.json').read_text()) if (ROOT / 'reports/v3/fakenewsnet_v3_summary.json').exists() else None
    json.dump(summary, open(out / 'fakenewsnet_v4_summary.json', 'w'), indent=2)
    return res, summary


def run_liar(out):
    man = pd.read_csv(ROOT / 'reports/liar/split_manifest.csv')
    va, te = [man[man.split == s].reset_index(drop=True) for s in ['valid', 'test']]
    yva, yte = va.label.values, te.label.values
    classes = np.array(sorted(set(yva)))
    comps = {'stage1': (np.load(ROOT / 'reports/optimization/liar/selected_validation_probabilities.npy'), np.load(ROOT / 'reports/optimization/liar/test_probabilities.npy'))}
    for f in sorted(V4.glob('liar_*_valid.npy')):
        tag = f.name[len('liar_'):-len('_valid.npy')]
        comps[tag] = (np.load(f), np.load(V4 / f'liar_{tag}_test.npy'))
    raw_cols = ['id', 'label', 'statement', 'subject', 'speaker', 'job', 'state', 'party', 'bt', 'f', 'ht', 'mt', 'pof', 'context']
    raw = pd.concat([pd.read_csv(ROOT / f'data/raw/liar/{s}.tsv', sep='\t', header=None, names=raw_cols, quoting=3, dtype=str) for s in ['train', 'valid', 'test']]).fillna('NA')
    v2 = joblib.load(ROOT / 'models/liar_v2.joblib')['models']['metadata']
    for name, frame in [('va', va), ('te', te)]:
        pass
    fva = va.merge(raw.drop(columns=['label']).drop_duplicates('id'), on='id', how='left'); fte = te.merge(raw.drop(columns=['label']).drop_duplicates('id'), on='id', how='left')
    comps['lr_metadata'] = (v2.lr_proba(fva), v2.lr_proba(fte))
    names = list(comps)
    Xva = np.column_stack([np.log(np.clip(comps[n][0], 1e-5, 1)) for n in names]); Xte = np.column_stack([np.log(np.clip(comps[n][1], 1e-5, 1)) for n in names])
    rows, preds, cands = [], {}, []
    for n in names:
        f = f1_score(yva, classes[comps[n][0].argmax(1)], average='macro')
        cands.append((f, f'single:{n}', lambda n=n: classes[comps[n][1].argmax(1)]))
    avg_va = np.mean([comps[n][0] for n in names], 0); avg_te = np.mean([comps[n][1] for n in names], 0)
    cands.append((f1_score(yva, classes[avg_va.argmax(1)], average='macro'), 'average', lambda: classes[avg_te.argmax(1)]))
    skf = StratifiedKFold(5, shuffle=True, random_state=42); best = None
    for C in [0.01, 0.03, 0.1, 0.3, 1]:
        oof = np.empty(len(yva), dtype=object)
        for a, b in skf.split(Xva, yva):
            m = LogisticRegression(C=C, max_iter=3000, class_weight='balanced').fit(Xva[a], yva[a]); oof[b] = m.predict(Xva[b])
        f = f1_score(yva, oof.astype(str), average='macro')
        if best is None or f > best[0]:
            best = (f, C)
    meta = LogisticRegression(C=best[1], max_iter=3000, class_weight='balanced').fit(Xva, yva)
    cands.append((best[0], f'meta_lr(C={best[1]})', lambda: meta.predict(Xte)))
    for f, label, fn in cands:
        p = fn(); preds[label] = p
        rows.append({'method': label, 'val_score': round(f, 4), **M(yte, p)})
    res = pd.DataFrame(rows).sort_values('val_score', ascending=False)
    res.to_csv(out / 'liar_v4_results.csv', index=False)
    winner = max(cands, key=lambda c: c[0])
    summary = {'selected': winner[1], 'components': names, 'selected_test': M(yte, preds[winner[1]])}
    json.dump(summary, open(out / 'liar_v4_summary.json', 'w'), indent=2)
    return res, summary


# components that can run at inference time without the training matrix (kNN) or a
# 1.3 GB model that was stopped after one epoch (roberta-large)
NOT_DEPLOYABLE = ('knn', 'rl_')


def build_artifact():
    """Refit the selected meta-learner on deployable components and save models/fakenewsnet_v4.joblib."""
    from .stacking import Stage4Classifier
    yva, yte, comps = fnn_components()
    comps = {n: c for n, c in comps.items() if not n.startswith(NOT_DEPLOYABLE)}
    models, metrics = {}, {}
    for kind, mode in [('text', 'text'), ('domain', 'url')]:
        names = [n for n, (_, _, u) in comps.items() if mode == 'url' or not u]
        Xva = np.column_stack([logit(comps[n][0]) for n in names]); Xte = np.column_stack([logit(comps[n][1]) for n in names])
        f, C, t, _ = cv_meta(Xva, yva)
        meta = LogisticRegression(C=C, max_iter=2000).fit(Xva, yva)
        model = Stage4Classifier(names, meta, t, fnn_components.lr_url_bundle if 'lr_url' in names else None)
        pred = model.decide(model.meta_proba(Xte))
        models[kind] = model
        metrics[kind] = {'components': names, 'C': C, 'threshold': t, 'val_macro_f1_oof': round(f, 4), **M(yte, pred, 'fake')}
        np.save(V4 / f'fakenewsnet_v4_{kind}_test_fake_proba.npy', model.meta_proba(Xte)[:, 0])
    artifact = {'dataset': 'fakenewsnet', 'models': models, 'metrics': metrics,
                'note': 'Stage 4: logistic meta-learner over component P(fake) logits; tuned by 5-fold CV on validation, test evaluated once.'}
    joblib.dump(artifact, ROOT / 'models/fakenewsnet_v4.joblib', compress=3)
    reg_path = ROOT / 'models/active_models.json'
    reg = json.loads(reg_path.read_text()) if reg_path.exists() else {}
    reg['fakenewsnet_v4'] = 'models/fakenewsnet_v4.joblib'; reg_path.write_text(json.dumps(reg, indent=2))
    json.dump(metrics, open(V4 / 'fakenewsnet_v4_deployed_metrics.json', 'w'), indent=2)
    return metrics


def build_liar_artifact():
    """LIAR stage 4 (needs speaker metadata): average of stage 1, RoBERTa(statement + metadata) and
    the stage-2 metadata LR. Kept only if it beats the stage-2 metadata model on validation."""
    from .stacking import LiarStage4Classifier
    man = pd.read_csv(ROOT / 'reports/liar/split_manifest.csv')
    va, te = [man[man.split == s].reset_index(drop=True) for s in ['valid', 'test']]
    raw_cols = ['id', 'label', 'statement', 'subject', 'speaker', 'job', 'state', 'party', 'bt', 'f', 'ht', 'mt', 'pof', 'context']
    raw = pd.concat([pd.read_csv(ROOT / f'data/raw/liar/{s}.tsv', sep='\t', header=None, names=raw_cols, quoting=3, dtype=str) for s in ['train', 'valid', 'test']]).fillna('NA')
    v2 = joblib.load(ROOT / 'models/liar_v2.joblib')['models']['metadata']
    classes = v2.classes_
    fva = va.merge(raw.drop(columns=['label']).drop_duplicates('id'), on='id', how='left')
    fte = te.merge(raw.drop(columns=['label']).drop_duplicates('id'), on='id', how='left')
    s1 = [np.load(ROOT / f'reports/optimization/liar/{n}.npy') for n in ['selected_validation_probabilities', 'test_probabilities']]
    rb = [np.load(V4 / f'liar_rb_meta_{n}.npy') for n in ['valid', 'test']]
    lr = [v2.lr_proba(fva), v2.lr_proba(fte)]
    cand = {'stage2_metadata': [v2.alpha * s1[i] + (1 - v2.alpha) * lr[i] for i in (0, 1)],
            'stage4_average': [(s1[i] + rb[i] + lr[i]) / 3 for i in (0, 1)]}
    rows = {}
    for name, (pv, pt) in cand.items():
        rows[name] = {'val_macro_f1': round(f1_score(va.label, classes[pv.argmax(1)], average='macro'), 4), **M(te.label.values, classes[pt.argmax(1)])}
    print(json.dumps(rows, indent=1))
    if rows['stage4_average']['val_macro_f1'] <= rows['stage2_metadata']['val_macro_f1']:
        print('stage 4 is not better on validation; artifact not saved'); return rows
    np.save(V4 / 'liar_v4_metadata_test_proba.npy', cand['stage4_average'][1])
    artifact = {'dataset': 'liar', 'models': {'metadata': LiarStage4Classifier(classes)}, 'metrics': {'metadata': rows['stage4_average'], 'stage2_reference': rows['stage2_metadata']},
                'note': 'LIAR stage 4: mean of stage-1, RoBERTa(statement + metadata) and stage-2 metadata LR probabilities; selected on validation.'}
    joblib.dump(artifact, ROOT / 'models/liar_v4.joblib', compress=3)
    reg_path = ROOT / 'models/active_models.json'
    reg = json.loads(reg_path.read_text()) if reg_path.exists() else {}
    reg['liar_v4'] = 'models/liar_v4.joblib'; reg_path.write_text(json.dumps(reg, indent=2))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['fakenewsnet', 'liar', 'both'], default='both')
    ap.add_argument('--build', action='store_true', help='save the deployable FakeNewsNet stage-4 artifact')
    a = ap.parse_args()
    if a.build:
        if a.dataset in ('fakenewsnet', 'both'):
            print(json.dumps(build_artifact(), indent=1))
        if a.dataset in ('liar', 'both'):
            build_liar_artifact()
        return
    V4.mkdir(parents=True, exist_ok=True)
    pd.set_option('display.width', 220)
    if a.dataset in ('fakenewsnet', 'both'):
        res, summary = run_fnn(V4)
        print(res.to_string(index=False)); print(json.dumps(summary['selected'], indent=1))
    if a.dataset in ('liar', 'both') and list(V4.glob('liar_*_valid.npy')):
        res, summary = run_liar(V4)
        print(res.to_string(index=False)); print(summary['selected'], summary['selected_test'])


if __name__ == '__main__':
    main()
