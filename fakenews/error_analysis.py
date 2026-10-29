"""Error analysis of the FakeNewsNet test predictions (stage 4 text + URL and stage 5 with social context).

Breaks errors down by source, headline length and number of tweets, lists typical false positives
and false negatives and draws the figures used in chapter 5.

    python -m fakenews.error_analysis      # after python -m fakenews.social; writes reports/v5/
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
from .improve_v2 import ROOT

OUT = ROOT / 'reports/v5'
MODELS = {'pred_stage4': 'Փուլ 4 (տեքստ + հղում)', 'pred_v5': 'Փուլ 5 (+ սոցիալական համատեքստ)'}


def bucket_report(df, col, bins, labels):
    df = df.assign(bucket=pd.cut(df[col], bins=bins, labels=labels, right=False))
    rows = []
    for b, g in df.groupby('bucket', observed=True):
        r = {'bucket': str(b), 'rows': len(g), 'fake_share': round((g.label == 'fake').mean(), 4)}
        for p in MODELS:
            r[f'{p}_error_rate'] = round((g[p] != g.label).mean(), 4)
            r[f'{p}_macro_f1'] = round(f1_score(g.label, g[p], average='macro'), 4) if g.label.nunique() > 1 else None
        rows.append(r)
    return rows


def main():
    df = pd.read_csv(OUT / 'fakenewsnet_v5_test_predictions.csv')
    df['words'] = df.text.str.split().str.len()
    report = {'rows': len(df)}
    for p in MODELS:
        cm = confusion_matrix(df.label, df[p], labels=['fake', 'real'])
        report[p] = {'confusion_fake_real': cm.tolist(), 'errors': int((df[p] != df.label).sum()),
                     'false_negative': int(cm[0, 1]), 'false_positive': int(cm[1, 0])}
    both = (df.pred_stage4 != df.label) & (df.pred_v5 != df.label)
    report['fixed_by_social'] = int(((df.pred_stage4 != df.label) & (df.pred_v5 == df.label)).sum())
    report['broken_by_social'] = int(((df.pred_stage4 == df.label) & (df.pred_v5 != df.label)).sum())
    report['errors_in_both'] = int(both.sum())
    report['by_source'] = {s: {p: round((g[p] != g.label).mean(), 4) for p in MODELS} | {'rows': len(g)} for s, g in df.groupby('source')}
    report['by_tweets'] = bucket_report(df, 'n_tweets', [0, 1, 10, 100, 1e9], ['0', '1–9', '10–99', '≥100'])
    report['by_words'] = bucket_report(df, 'words', [0, 6, 10, 15, 1e9], ['≤5', '6–9', '10–14', '≥15'])
    cols = ['id', 'source', 'label', 'n_tweets', 'text']
    fn = df[(df.label == 'fake') & (df.pred_v5 == 'real')]
    fp = df[(df.label == 'real') & (df.pred_v5 == 'fake')]
    fixed = df[(df.label == 'fake') & (df.pred_stage4 == 'real') & (df.pred_v5 == 'fake')]
    rng = 42
    report['examples'] = {'false_negative': fn.sample(min(8, len(fn)), random_state=rng)[cols].to_dict('records'),
                          'false_positive': fp.sample(min(8, len(fp)), random_state=rng)[cols].to_dict('records'),
                          'fixed_fake_by_social': fixed.sample(min(6, len(fixed)), random_state=rng)[cols].to_dict('records')}
    y = (df.label == 'fake').values.astype(float)
    report['calibration'] = {}
    for p, path in [('pred_stage4', ROOT / 'reports/v4/fakenewsnet_v4_domain_test_fake_proba.npy'),
                    ('pred_v5', OUT / 'fakenewsnet_v5_test_fake_proba.npy')]:
        prob = np.load(path)
        bins = np.minimum((prob * 10).astype(int), 9)
        ece = sum(abs(prob[bins == b].mean() - y[bins == b].mean()) * (bins == b).mean() for b in range(10) if (bins == b).any())
        report['calibration'][p] = {'brier': round(float(np.mean((prob - y) ** 2)), 4), 'ece_10_bins': round(float(ece), 4)}
    json.dump(report, open(OUT / 'error_analysis.json', 'w'), indent=2, ensure_ascii=False, default=int)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, (p, title) in zip(axes, MODELS.items()):
        cm = np.array(report[p]['confusion_fake_real'])
        ax.imshow(cm, cmap='Blues')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{cm[i, j]:,}'.replace(',', ' '), ha='center', va='center', fontsize=13,
                        color='white' if cm[i, j] > cm.max() / 2 else 'black')
        ax.set_xticks([0, 1], ['կեղծ', 'իրական']); ax.set_yticks([0, 1], ['կեղծ', 'իրական'])
        ax.set_xlabel('Կանխատեսում'); ax.set_ylabel('Իրական պիտակ'); ax.set_title(title, fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / 'fig_confusion_v4_v5.png', dpi=200); plt.close(fig)

    imp = pd.read_csv(OUT / 'social_feature_importance.csv', index_col=0).iloc[:, 0].head(10)[::-1]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.barh(imp.index, imp.values * 100, color='#3b6ea5')
    ax.set_xlabel('Macro-F1-ի նվազում (տոկոսային կետ) հատկանիշը խառնելիս'); fig.tight_layout()
    fig.savefig(OUT / 'fig_social_importance.png', dpi=200); plt.close(fig)

    tw = pd.DataFrame(report['by_tweets'])
    fig, ax = plt.subplots(figsize=(7, 3.4)); x = np.arange(len(tw)); w = .38
    for k, (p, title) in enumerate(MODELS.items()):
        ax.bar(x + (k - .5) * w, tw[f'{p}_error_rate'] * 100, w, label=title, color=['#9db4cf', '#3b6ea5'][k])
    ax.set_xticks(x, [f'{b}\n(n={n})' for b, n in zip(tw.bucket, tw.rows)])
    ax.set_xlabel('Թվիթների քանակ'); ax.set_ylabel('Սխալների մասնաբաժին, %'); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / 'fig_errors_by_tweets.png', dpi=200); plt.close(fig)
    print(json.dumps({k: v for k, v in report.items() if k != 'examples'}, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
