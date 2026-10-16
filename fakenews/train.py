import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_score, recall_score
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from .data import clean_text, load_liar, load_fakenews, assert_disjoint

ROOT = Path(__file__).resolve().parents[1]


def candidates():
    return {
        'majority': DummyClassifier(strategy='most_frequent'),
        'complement_nb': ComplementNB(alpha=1.0),
        'logistic_regression': LogisticRegression(C=2, class_weight='balanced', max_iter=1500, random_state=42),
        'linear_svm': LinearSVC(C=1, class_weight='balanced', random_state=42, max_iter=5000),
    }


def pipeline(model):
    return Pipeline([('tfidf', TfidfVectorizer(preprocessor=clean_text, ngram_range=(1, 2), min_df=2, max_df=.98, max_features=60000, sublinear_tf=True)), ('classifier', model)])


def metrics(y, pred):
    return dict(accuracy=accuracy_score(y, pred), balanced_accuracy=balanced_accuracy_score(y, pred), macro_precision=precision_score(y, pred, average='macro', zero_division=0), macro_recall=recall_score(y, pred, average='macro', zero_division=0), macro_f1=f1_score(y, pred, average='macro', zero_division=0), weighted_f1=f1_score(y, pred, average='weighted', zero_division=0))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def bootstrap(y, pred):
    rng = np.random.default_rng(42)
    y, pred = np.asarray(y), np.asarray(pred)
    labels = np.unique(y)
    values = []
    for _ in range(500):
        idx = rng.integers(0, len(y), len(y))
        values.append(f1_score(y[idx], pred[idx], labels=labels, average='macro', zero_division=0))
    return np.quantile(values, [.025, .975]).tolist()


def run(name, df, audit):
    folder = ROOT / 'reports' / name
    folder.mkdir(parents=True, exist_ok=True)
    assert_disjoint(df)
    audit['split_counts'] = pd.crosstab(df.split, df.label).to_dict()
    audit['text_overlap_between_splits'] = 0
    write_json(folder / 'data_audit.json', audit)
    df.to_csv(folder / 'split_manifest.csv', index=False)
    train, valid, test = [df[df.split == s] for s in ['train', 'valid', 'test']]
    fitted, rows = {}, []
    for model_name, estimator in candidates().items():
        start = time.perf_counter()
        model = pipeline(estimator).fit(train.text, train.label)
        fitted[model_name] = model
        rows.append({'model': model_name, **metrics(valid.label, model.predict(valid.text)), 'fit_seconds': time.perf_counter() - start})
        print(name, model_name, 'validation macro F1', round(rows[-1]['macro_f1'], 4), flush=True)
    comparison = pd.DataFrame(rows).sort_values('macro_f1', ascending=False)
    comparison.to_csv(folder / 'validation_comparison.csv', index=False)
    winner = comparison.iloc[0]['model']
    model = fitted[winner]  # Test never participates in model selection.
    pred = model.predict(test.text)
    result = {'dataset': name, 'selected_model': winner, 'selection': 'highest validation macro-F1; no train+valid refit', 'test_rows': len(test), **metrics(test.label, pred), 'macro_f1_95pct_bootstrap_ci': bootstrap(test.label, pred)}
    write_json(folder / 'test_metrics.json', result)
    write_json(folder / 'classification_report.json', classification_report(test.label, pred, output_dict=True, zero_division=0))
    predictions = test[['id', 'source', 'text', 'label']].copy()
    predictions['prediction'] = pred
    predictions.to_csv(folder / 'test_predictions.csv', index=False)
    predictions[predictions.label != predictions.prediction].to_csv(folder / 'errors.csv', index=False)
    labels = list(model.classes_)
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(confusion_matrix(test.label, pred, labels=labels), display_labels=labels).plot(ax=ax, cmap='Blues', colorbar=False, xticks_rotation=35)
    ax.set_title(name + ': held-out test')
    fig.tight_layout(); fig.savefig(folder / 'confusion_matrix.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(comparison.model, comparison.macro_f1, color='#247d91')
    ax.set_xlim(0, 1); ax.set_xlabel('Validation macro F1'); ax.set_title(name)
    fig.tight_layout(); fig.savefig(folder / 'model_comparison.png', dpi=160); plt.close(fig)
    clf = model.named_steps['classifier']
    if hasattr(clf, 'coef_'):
        names = model.named_steps['tfidf'].get_feature_names_out()
        coefs = clf.coef_
        if len(labels) == 2:
            coefs = np.vstack([-coefs[0], coefs[0]])
        write_json(folder / 'top_features.json', {label: names[np.argsort(coef)[-20:][::-1]].tolist() for label, coef in zip(labels, coefs)})
    subgroups = {source: metrics(group.label, model.predict(group.text)) for source, group in test.groupby('source')}
    write_json(folder / 'source_metrics.json', subgroups)
    joblib.dump({'model': model, 'dataset': name, 'metrics': result, 'training_keys': set(train.key)}, ROOT / 'models' / f'{name}.joblib')
    if name == 'fakenewsnet':
        transfer = {}
        for source, target in [('gossipcop', 'politifact'), ('politifact', 'gossipcop')]:
            source_train = train[train.source == source]
            target_test = test[test.source == target]
            other = clone(model).fit(source_train.text, source_train.label)
            transfer[f'{source}_to_{target}'] = {'train_rows': len(source_train), 'test_rows': len(target_test), **metrics(target_test.label, other.predict(target_test.text))}
        write_json(folder / 'cross_domain_metrics.json', transfer)
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--liar', type=Path, default=(ROOT / 'data/raw/liar' if (ROOT / 'data/raw/liar/train.tsv').exists() else ROOT.parents[1] / 'liar_dataset'))
    parser.add_argument('--fakenews', type=Path, default=(ROOT / 'data/raw/fakenewsnet' if (ROOT / 'data/raw/fakenewsnet/politifact_fake.csv').exists() else ROOT.parent / 'dataset'))
    args = parser.parse_args()
    (ROOT / 'models').mkdir(exist_ok=True)
    (ROOT / 'reports').mkdir(exist_ok=True)
    results = []
    for name, loader, path in [('liar', load_liar, args.liar), ('fakenewsnet', load_fakenews, args.fakenews)]:
        df, audit = loader(path)
        results.append(run(name, df, audit))
    pd.DataFrame(results).to_csv(ROOT / 'reports' / 'summary.csv', index=False)
    inputs = list(args.liar.glob('*.tsv')) + [args.fakenews / f'{s}_{l}.csv' for s in ['politifact', 'gossipcop'] for l in ['fake', 'real']]
    write_json(ROOT / 'reports' / 'reproducibility.json', {'seed': 42, 'python': platform.python_version(), 'sklearn': sklearn.__version__, 'input_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}})

if __name__ == '__main__':
    main()
