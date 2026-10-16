"""Stage 2: char n-grams, source domain, LIAR metadata and stacking with the stage-1 ensemble.

All choices (C, stacking weight alpha, decision threshold) use the validation split only;
the test split is evaluated once. With --save (default) the selected models are written to
models/<dataset>_v2.joblib and registered in models/active_models.json.

Usage: python -m fakenews.improve_v2 [--no-save]
"""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from .stacking import FakeNewsFeatures, LiarFeatures, StackedClassifier, domain
from .data import LABELS

ROOT = Path(__file__).resolve().parents[1]
ALPHAS = np.round(np.arange(0, 1.01, .1), 1)
THRESHOLDS = np.round(np.arange(.20, .81, .01), 2)


def M(y, p, pos=None):
    r = dict(accuracy=accuracy_score(y, p), macro_f1=f1_score(y, p, average='macro'),
             macro_precision=precision_score(y, p, average='macro', zero_division=0),
             macro_recall=recall_score(y, p, average='macro', zero_division=0))
    if pos is not None:
        r[f'{pos}_precision'] = precision_score(y, p, pos_label=pos, average='binary', zero_division=0)
        r[f'{pos}_recall'] = recall_score(y, p, pos_label=pos, average='binary', zero_division=0)
    return {k: round(float(v), 4) for k, v in r.items()}


def paired_ci(y, a, b, n=1000):
    rng = np.random.default_rng(42); y, a, b = map(np.asarray, (y, a, b)); d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        d.append(f1_score(y[i], b[i], average='macro') - f1_score(y[i], a[i], average='macro'))
    return [round(float(x), 4) for x in np.quantile(d, [.025, .975])]


def fit_lr(features, tr, ytr, va, yva, Cs):
    Xtr, Xva = features.transform(tr, loo=True), features.transform(va)
    best = None
    for C in Cs:
        m = LogisticRegression(C=C, max_iter=4000, class_weight='balanced').fit(Xtr, ytr)
        f = f1_score(yva, m.predict(Xva), average='macro')
        if best is None or f > best[0]:
            best = (f, C, m)
    return best


def best_threshold(y, p_fake):
    fs = [f1_score(y, np.where(p_fake >= t, 'fake', 'real'), average='macro') for t in THRESHOLDS]
    i = int(np.argmax(fs)); return float(THRESHOLDS[i]), fs[i]


def registry():
    path = ROOT / 'models/active_models.json'
    return json.loads(path.read_text()) if path.exists() else {}


def save(name, artifact):
    path = ROOT / 'models' / f'{name}_v2.joblib'
    joblib.dump(artifact, path)
    reg = registry(); reg[f'{name}_v2'] = str(path.relative_to(ROOT))
    tmp = ROOT / 'models/active_models.tmp'; tmp.write_text(json.dumps(reg, indent=2)); tmp.replace(ROOT / 'models/active_models.json')
    print('saved', path)


def fakenewsnet(raw, out, do_save):
    man = pd.read_csv(ROOT / 'reports/fakenewsnet/split_manifest.csv')
    urls = pd.concat([pd.read_csv(raw / f'{s}_{l}.csv', usecols=['id', 'news_url'])
                      for s in ['politifact', 'gossipcop'] for l in ['fake', 'real']]).drop_duplicates('id')
    df = man.merge(urls, on='id', how='left'); assert len(df) == len(man)
    df['domain'] = df.news_url.map(domain)
    tr, va, te = [df[df.split == s].reset_index(drop=True) for s in ['train', 'valid', 'test']]
    base_path = registry().get('fakenewsnet', 'models/fakenewsnet.joblib')
    neural_va = np.load(ROOT / 'reports/optimization/fakenewsnet/selected_validation_probabilities.npy')
    neural_te = np.load(ROOT / 'reports/optimization/fakenewsnet/test_probabilities.npy')
    classes = np.array(['fake', 'real'])  # stage-1 column order (sorted labels)
    old = pd.read_csv(ROOT / 'reports/optimization/fakenewsnet/test_predictions.csv')
    assert (old.id.values == te.id.values).all(), 'test order mismatch'
    yva, yte = va.label.values, te.label.values
    rows, preds, sel, artifacts = [], {}, {}, {}

    def add(name, pva, pte, note):
        t, f = best_threshold(yva, pva)
        p = np.where(pte >= t, 'fake', 'real')
        preds[name] = p; sel[name] = dict(threshold=t, val_macro_f1=round(f, 4), note=note)
        rows.append(dict(model=name, threshold=t, val_macro_f1=round(f, 4), **M(yte, p, 'fake'), note=note))
        return t

    add('stage1_neural_ensemble', neural_va[:, 0], neural_te[:, 0], 'previous best (MiniLM fine-tuned + TF-IDF, 50/50)')
    for use_domain in [False, True]:
        feats = FakeNewsFeatures(use_domain).fit(tr)
        _, C, lr = fit_lr(feats, tr, tr.label, va, va.label, (1, 3, 10))
        tag = 'word+char' + ('+domain' if use_domain else '')
        stack = StackedClassifier(base_path, feats, lr, 0, classes, positive='fake', name=tag)
        pva, pte = stack.lr_proba(va)[:, 0], stack.lr_proba(te)[:, 0]
        add(f'lr_{tag}', pva, pte, f'logistic regression C={C}')
        best = max((best_threshold(yva, a * neural_va[:, 0] + (1 - a) * pva)[1], a) for a in ALPHAS)
        stack.alpha = float(best[1])
        t = add(f'stack_neural+{tag}', stack.combine(neural_va, va)[:, 0], stack.combine(neural_te, te)[:, 0],
                f'weight neural={stack.alpha}, LR={round(1 - stack.alpha, 1)} (chosen on validation)')
        stack.threshold = t
        # artifact must reproduce the evaluated predictions exactly
        assert (stack.decide(stack.combine(neural_te, te)) == preds[f'stack_neural+{tag}']).all()
        artifacts['domain' if use_domain else 'text'] = stack
    res = pd.DataFrame(rows); res.to_csv(out / 'fakenewsnet_v2_results.csv', index=False)
    base = preds['stage1_neural_ensemble']
    ci = {n: paired_ci(yte, base, p) for n, p in preds.items() if n != 'stage1_neural_ensemble'}
    cm = {n: confusion_matrix(yte, preds[n], labels=['fake', 'real']).tolist() for n in ['stack_neural+word+char', 'stack_neural+word+char+domain']}
    pd.DataFrame({'id': te.id, 'source': te.source, 'text': te.text, 'label': yte, **preds}).to_csv(out / 'fakenewsnet_v2_test_predictions.csv', index=False)
    per_src = {n: {s: M(yte[(te.source == s).values], p[(te.source == s).values]) for s in ['gossipcop', 'politifact']} for n, p in preds.items()}
    summary = dict(selection=sel, paired_macro_f1_delta_ci_vs_stage1=ci, confusion_matrices=cm, per_source=per_src,
                   test_domains_seen_in_train=round(float(te.domain.isin(set(tr.domain)).mean()), 4),
                   note='Selection on validation only; test previously reported (not blind).')
    json.dump(summary, open(out / 'fakenewsnet_v2_summary.json', 'w'), indent=2)
    errs = te.assign(pred=preds['stack_neural+word+char'])
    errs[errs.label != errs.pred].head(40)[['source', 'text', 'label', 'pred']].to_csv(out / 'fakenewsnet_v2_error_sample.csv', index=False)
    if do_save:
        metrics = {k: rows[[r['model'] for r in rows].index(f'stack_neural+{n}')] for k, n in [('text', 'word+char'), ('domain', 'word+char+domain')]}
        save('fakenewsnet', {'dataset': 'fakenewsnet', 'models': artifacts, 'metrics': metrics, 'training_keys': set(tr.key),
                             'base_path': base_path, 'note': 'domain model is used only when the article URL is supplied'})
    return res


def liar(out, do_save):
    cols = ['id', 'label', 'statement', 'subject', 'speaker', 'job', 'state', 'party', 'bt', 'f', 'ht', 'mt', 'pof', 'context']
    raw = pd.concat([pd.read_csv(ROOT / f'data/raw/liar/{s}.tsv', sep='\t', header=None, names=cols, quoting=3, dtype=str) for s in ['train', 'valid', 'test']]).fillna('NA')
    man = pd.read_csv(ROOT / 'reports/liar/split_manifest.csv')
    df = man.merge(raw.drop(columns=['label']).drop_duplicates('id'), on='id', how='left'); assert len(df) == len(man)
    tr, va, te = [df[df.split == s].reset_index(drop=True) for s in ['train', 'valid', 'test']]
    base_path = registry().get('liar', 'models/liar.joblib')
    nva = np.load(ROOT / 'reports/optimization/liar/selected_validation_probabilities.npy')
    nte = np.load(ROOT / 'reports/optimization/liar/test_probabilities.npy')
    classes = np.array(sorted(LABELS))
    rows, saved = [], {}
    for binary in [False, True]:
        task = 'binary' if binary else '6-class'
        lab = (lambda s: np.where(pd.Series(s).isin(LABELS[3:]), 'true-ish', 'false-ish')) if binary else (lambda s: np.asarray(s))
        yva, yte = lab(va.label), lab(te.label)
        if binary:
            t_idx = [list(classes).index(x) for x in LABELS[3:]]; f_idx = [list(classes).index(x) for x in LABELS[:3]]
            collapse = lambda P: np.c_[P[:, f_idx].sum(1), P[:, t_idx].sum(1)]; cl = np.array(['false-ish', 'true-ish'])
        else:
            collapse = lambda P: P; cl = classes
        Pva, Pte = collapse(nva), collapse(nte)
        rows.append(dict(task=task, model='stage1_neural_ensemble', **M(yte, cl[Pte.argmax(1)])))
        for use_meta in [False, True]:
            kind = 'text+metadata' if use_meta else 'text'
            feats = LiarFeatures(use_meta).fit(tr)
            _, C, lr = fit_lr(feats, tr, lab(tr.label), va, yva, (.3, 1, 3))
            stack = StackedClassifier(base_path, feats, lr, 0, cl, name=kind)
            pva, pte = stack.lr_proba(va), stack.lr_proba(te)
            rows.append(dict(task=task, model=f'lr_{kind}', **M(yte, cl[pte.argmax(1)])))
            a = max((f1_score(yva, cl[(a * Pva + (1 - a) * pva).argmax(1)], average='macro'), a) for a in ALPHAS)[1]
            stack.alpha = float(a)
            p = stack.decide(stack.combine(Pte, te))
            rows.append(dict(task=task, model=f'stack_neural+{kind} (w={a})', **M(yte, p)))
            if not binary and use_meta:
                saved = {'metadata': stack, 'metrics': rows[-1]}
    res = pd.DataFrame(rows); res.to_csv(out / 'liar_v2_results.csv', index=False)
    if do_save and saved:
        save('liar', {'dataset': 'liar', 'models': {'metadata': saved['metadata']}, 'metrics': {'metadata': saved['metrics']},
                      'training_keys': set(tr.key), 'base_path': base_path, 'note': 'used only when speaker metadata is supplied'})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', type=Path, default=None, help='folder with FakeNewsNet *_fake.csv / *_real.csv')
    ap.add_argument('--no-save', action='store_true')
    a = ap.parse_args()
    raw = a.raw or next(p for p in [ROOT / 'data/raw/fakenewsnet', ROOT.parent / 'dataset'] if (p / 'politifact_fake.csv').exists())
    out = ROOT / 'reports/v2'; out.mkdir(parents=True, exist_ok=True)
    pd.set_option('display.width', 200)
    print(fakenewsnet(raw, out, not a.no_save).to_string(index=False))
    print(liar(out, not a.no_save).to_string(index=False))


if __name__ == '__main__':
    main()
