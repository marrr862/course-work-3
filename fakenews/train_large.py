"""Stage 3: fine-tune a larger transformer (default microsoft/deberta-v3-base) and stack it.

Same splits as before; early stopping, the stacking weights and the decision threshold
are all chosen on the validation split only. The test split is evaluated once at the end.

    python -m fakenews.train_large --dataset fakenewsnet [--model microsoft/deberta-v3-base]
"""
import argparse
import gc
import json
import time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from .improve_v2 import M, paired_ci, best_threshold, ROOT

DEVICE = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')


def predict(model, tok, texts, max_len, bs=64):
    model.eval(); out = []
    with torch.inference_mode():
        for i in range(0, len(texts), bs):
            batch = tok(texts[i:i + bs], padding=True, truncation=True, max_length=max_len, return_tensors='pt').to(DEVICE)
            out.append(model(**batch).logits.float().softmax(-1).cpu().numpy())
    return np.concatenate(out)


def finetune(name, model_id, train, valid, classes, epochs, lr, bs, max_len, out_dir):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
    torch.manual_seed(42); np.random.seed(42)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=len(classes)).to(DEVICE)
    labelmap = {c: i for i, c in enumerate(classes)}
    y = torch.tensor([labelmap[v] for v in train.label])
    counts = np.bincount(y.numpy(), minlength=len(classes))
    weights = torch.tensor(np.sqrt(len(y) / (len(classes) * counts)), dtype=torch.float32, device=DEVICE)
    texts = train.text.tolist()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.01)
    steps = (len(y) + bs - 1) // bs
    sched = get_linear_schedule_with_warmup(opt, int(steps * .1), steps * epochs)
    best, stale, history, best_prob = -1, 0, [], None
    for epoch in range(1, epochs + 1):
        model.train(); perm = torch.randperm(len(y)); total = 0; start = time.monotonic()
        for k, off in enumerate(range(0, len(y), bs)):
            idx = perm[off:off + bs]
            batch = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=max_len, return_tensors='pt').to(DEVICE)
            loss = torch.nn.functional.cross_entropy(model(**batch).logits.float(), y[idx].to(DEVICE), weight=weights)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True); total += loss.item()
            if k % 100 == 0:
                print(name, 'epoch', epoch, 'step', k, '/', steps, 'loss', round(loss.item(), 4), flush=True)
        prob = predict(model, tok, valid.text.tolist(), max_len)
        pred = np.asarray(classes)[prob.argmax(1)]
        row = {'epoch': epoch, 'val_macro_f1': round(float(f1_score(valid.label, pred, average='macro')), 4),
               'mean_loss': round(total / steps, 4), 'seconds': round(time.monotonic() - start, 1)}
        history.append(row); print(name, 'EPOCH', json.dumps(row), flush=True)
        if row['val_macro_f1'] > best:
            best, stale, best_prob = row['val_macro_f1'], 0, prob
            model.save_pretrained(out_dir); tok.save_pretrained(out_dir)
        else:
            stale += 1
        (ROOT / 'reports/v3' / f'{name}_training_history.json').write_text(json.dumps(history, indent=2))
        if stale >= 2:
            break
    del model, opt; gc.collect()
    if DEVICE == 'mps':
        torch.mps.empty_cache()
    return best_prob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['fakenewsnet', 'liar'], default='fakenewsnet')
    ap.add_argument('--model', default='microsoft/deberta-v3-base')
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--max-len', type=int, default=64)
    ap.add_argument('--build-only', action='store_true', help='reuse the saved fine-tuned model and probabilities; only rebuild the stage-3 artifact')
    a = ap.parse_args()
    name = a.dataset
    if a.build_only:
        return build_artifact(name, a.max_len)
    (ROOT / 'reports/v3').mkdir(parents=True, exist_ok=True)
    out_dir = ROOT / 'models' / f'{name}_large'
    df = pd.read_csv(ROOT / f'reports/{name}/split_manifest.csv')
    tr, va, te = [df[df.split == s].reset_index(drop=True) for s in ['train', 'valid', 'test']]
    classes = np.sort(tr.label.unique())
    print('device', DEVICE, 'model', a.model, 'train', len(tr), flush=True)
    pva = finetune(name, a.model, tr, va, classes, a.epochs, a.lr, a.batch, a.max_len, out_dir)
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(out_dir)
    model = AutoModelForSequenceClassification.from_pretrained(out_dir).to(DEVICE)
    pte = predict(model, tok, te.text.tolist(), a.max_len)
    np.save(ROOT / f'reports/v3/{name}_large_valid.npy', pva); np.save(ROOT / f'reports/v3/{name}_large_test.npy', pte)

    # stacking with the saved stage-2 components (validation only)
    import joblib
    v2 = joblib.load(ROOT / f'models/{name}_v2.joblib')
    s1_va = np.load(ROOT / f'reports/optimization/{name}/selected_validation_probabilities.npy')
    s1_te = np.load(ROOT / f'reports/optimization/{name}/test_probabilities.npy')
    rows, preds = [], {}
    if name == 'fakenewsnet':
        urls = pd.concat([pd.read_csv(ROOT / f'data/raw/fakenewsnet/{s}_{l}.csv', usecols=['id', 'news_url'])
                          for s in ['politifact', 'gossipcop'] for l in ['fake', 'real']]).drop_duplicates('id')
        va = va.merge(urls, on='id', how='left'); te = te.merge(urls, on='id', how='left')
        y_va, y_te = va.label.values, te.label.values

        def evaluate(tag, q_va, q_te):
            t, f = best_threshold(y_va, q_va)
            p = np.where(q_te >= t, 'fake', 'real'); preds[tag] = p
            rows.append({'model': tag, 'threshold': t, 'val_macro_f1': round(f, 4), **M(y_te, p, 'fake')})
        evaluate('large_alone', pva[:, 0], pte[:, 0])
        for kind in ['text', 'domain']:
            st = v2['models'][kind]
            lr_va = st.lr_proba(st.features.frame(va.text.tolist(), va.news_url.tolist()))[:, 0]
            lr_te = st.lr_proba(st.features.frame(te.text.tolist(), te.news_url.tolist()))[:, 0]
            evaluate(f'stage2_{kind}', st.alpha * s1_va[:, 0] + (1 - st.alpha) * lr_va, st.alpha * s1_te[:, 0] + (1 - st.alpha) * lr_te)
            best = None
            for a1 in np.round(np.arange(0, 1.01, .1), 1):          # weight of the large model
                for a2 in np.round(np.arange(0, 1.01 - a1, .1), 1):  # weight of stage-1 ensemble
                    q = a1 * pva[:, 0] + a2 * s1_va[:, 0] + (1 - a1 - a2) * lr_va
                    t, f = best_threshold(y_va, q)
                    if best is None or f > best[0]:
                        best = (f, a1, a2)
            _, a1, a2 = best
            evaluate(f'stage3_{kind} (large={a1}, stage1={a2})', a1 * pva[:, 0] + a2 * s1_va[:, 0] + (1 - a1 - a2) * lr_va,
                     a1 * pte[:, 0] + a2 * s1_te[:, 0] + (1 - a1 - a2) * lr_te)
        base = preds['stage2_domain']
        ci = {k: paired_ci(y_te, base, v) for k, v in preds.items() if k != 'stage2_domain'}
    else:
        y_va, y_te = va.label.values, te.label.values
        p = classes[pte.argmax(1)]; preds['large_alone'] = p
        rows.append({'model': 'large_alone', **M(y_te, p)})
        ci = {}
    res = pd.DataFrame(rows); res.to_csv(ROOT / f'reports/v3/{name}_v3_results.csv', index=False)
    json.dump({'model_id': a.model, 'device': DEVICE, 'paired_macro_f1_delta_ci_vs_stage2_domain': ci,
               'note': 'All weights and thresholds chosen on validation; test evaluated once.'},
              open(ROOT / f'reports/v3/{name}_v3_summary.json', 'w'), indent=2)
    pd.set_option('display.width', 200)
    print(res.to_string(index=False))


def build_artifact(name, max_len=64):
    """Recreate the validation-selected stage-3 combination as models/<name>_v3.joblib."""
    import joblib
    from .stacking import Stage3Classifier
    if name != 'fakenewsnet':
        raise SystemExit('stage 3 artifact is only built for FakeNewsNet')
    df = pd.read_csv(ROOT / f'reports/{name}/split_manifest.csv')
    va, te = [df[df.split == s].reset_index(drop=True) for s in ['valid', 'test']]
    urls = pd.concat([pd.read_csv(ROOT / f'data/raw/fakenewsnet/{s}_{l}.csv', usecols=['id', 'news_url'])
                      for s in ['politifact', 'gossipcop'] for l in ['fake', 'real']]).drop_duplicates('id')
    va = va.merge(urls, on='id', how='left'); te = te.merge(urls, on='id', how='left')
    pva = np.load(ROOT / f'reports/v3/{name}_large_valid.npy'); pte = np.load(ROOT / f'reports/v3/{name}_large_test.npy')
    s1_va = np.load(ROOT / f'reports/optimization/{name}/selected_validation_probabilities.npy')
    s1_te = np.load(ROOT / f'reports/optimization/{name}/test_probabilities.npy')
    v2 = joblib.load(ROOT / f'models/{name}_v2.joblib')
    models, metrics = {}, {}
    for kind in ['text', 'domain']:
        st = v2['models'][kind]
        fva = st.features.frame(va.text.tolist(), va.news_url.tolist()); fte = st.features.frame(te.text.tolist(), te.news_url.tolist())
        lr_va = st.lr_proba(fva)[:, 0]
        best = None
        for a1 in np.round(np.arange(0, 1.01, .1), 1):
            for a2 in np.round(np.arange(0, 1.01 - a1, .1), 1):
                t, f = best_threshold(va.label.values, a1 * pva[:, 0] + a2 * s1_va[:, 0] + (1 - a1 - a2) * lr_va)
                if best is None or f > best[0]:
                    best = (f, a1, a2, t)
        _, a1, a2, t = best
        model = Stage3Classifier(st, f'models/{name}_large', a1, a2, t, max_len)
        pred = model.decide(model.combine(pte, s1_te, fte))
        metrics[kind] = {'w_large': a1, 'w_stage1': a2, 'threshold': t, **M(te.label.values, pred, 'fake')}
        models[kind] = model
    path = ROOT / f'models/{name}_v3.joblib'
    joblib.dump({'dataset': name, 'models': models, 'metrics': metrics, 'note': 'stage 3: RoBERTa-base + stage 1 + stage 2 LR'}, path)
    reg_path = ROOT / 'models/active_models.json'; reg = json.loads(reg_path.read_text()); reg[f'{name}_v3'] = str(path.relative_to(ROOT))
    reg_path.write_text(json.dumps(reg, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
