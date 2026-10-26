"""How much does machine translation cost? Round-trip evaluation on FakeNewsNet test headlines.

FakeNewsNet has no Armenian or Russian headlines, so a stratified sample of English test headlines
is translated to Armenian / Russian with NLLB-200 and back to English, exactly as the API does for
non-English input, and then classified by the deployed stage-4 text-only model. The round trip adds
two translation errors, so the drop is a pessimistic estimate for text written natively in hy/ru.

    python -m fakenews.eval_translation [--n 300]
Outputs: reports/translation/round_trip_metrics.json, round_trip_examples.csv
"""
import argparse
import json
import time
import joblib
import numpy as np
import pandas as pd
from .improve_v2 import ROOT, M
from .translate import SUPPORTED, generate

OUT = ROOT / 'reports/translation'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=300)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(ROOT / 'reports/fakenewsnet/split_manifest.csv')
    te = man[man.split == 'test']
    sample = pd.concat([g.sample(round(a.n * len(g) / len(te)), random_state=42) for _, g in te.groupby('label')]).reset_index(drop=True)
    model = joblib.load(ROOT / 'models/fakenewsnet_v4.joblib')['models']['text']
    model.set_base(joblib.load(ROOT / 'reports/optimization/fakenewsnet/selected.joblib')['model'])
    texts = sample.text.tolist()
    p_en = model.predict_proba(texts)
    frame = sample[['id', 'label', 'text']].copy()
    frame['pred_en'] = model.decide(p_en)
    metrics = {'n': len(sample), 'english': M(sample.label.values, frame.pred_en.values, 'fake')}
    for lang, code in SUPPORTED.items():
        start = time.monotonic()
        foreign = generate(texts, 'eng_Latn', code)
        back = generate(foreign, code, 'eng_Latn')
        p = model.predict_proba(back)
        frame[f'{lang}'] = foreign; frame[f'{lang}_back'] = back; frame[f'pred_{lang}'] = model.decide(p)
        metrics[lang] = {**M(sample.label.values, frame[f'pred_{lang}'].values, 'fake'),
                         'agreement_with_english': round(float((frame[f'pred_{lang}'] == frame.pred_en).mean()), 4),
                         'mean_abs_change_p_fake': round(float(np.abs(p[:, 0] - p_en[:, 0]).mean()), 4),
                         'seconds': round(time.monotonic() - start, 1)}
        print(lang, json.dumps(metrics[lang]), flush=True)
    frame.to_csv(OUT / 'round_trip_examples.csv', index=False)
    (OUT / 'round_trip_metrics.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(json.dumps(metrics, indent=1))


if __name__ == '__main__':
    main()
