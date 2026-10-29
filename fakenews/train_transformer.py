"""Fine-tune one transformer on the fixed splits and save validation/test probabilities.

    python -m fakenews.train_transformer --dataset fakenewsnet --model roberta-base --input url_title --tag rb_url

--input:
  title       headline / statement only
  url_title   FakeNewsNet: headline + "domain + URL path words" as a second segment
  liar_meta   LIAR: statement + speaker/party/state/job/context/subject as a second segment
Outputs: models/<dataset>_<tag>/ (best checkpoint by validation macro-F1),
         reports/v4/<dataset>_<tag>_{valid,test}.npy and _history.json
"""
import argparse
import gc
import json
import time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from .improve_v2 import ROOT
from .stacking import url_words

DEVICE = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
OUT = ROOT / 'reports/v4'


def load(dataset, kind):
    df = pd.read_csv(ROOT / f'reports/{dataset}/split_manifest.csv')
    if dataset == 'fakenewsnet':
        urls = pd.concat([pd.read_csv(ROOT / f'data/raw/fakenewsnet/{s}_{l}.csv', usecols=['id', 'news_url'])
                          for s in ['politifact', 'gossipcop'] for l in ['fake', 'real']]).drop_duplicates('id')
        df = df.merge(urls, on='id', how='left')
        df['second'] = df.news_url.map(url_words) if kind == 'url_title' else None
    else:
        cols = ['id', 'label', 'statement', 'subject', 'speaker', 'job', 'state', 'party', 'bt', 'f', 'ht', 'mt', 'pof', 'context']
        raw = pd.concat([pd.read_csv(ROOT / f'data/raw/liar/{s}.tsv', sep='\t', header=None, names=cols, quoting=3, dtype=str)
                         for s in ['train', 'valid', 'test']]).fillna('')
        df = df.merge(raw[['id', 'subject', 'speaker', 'job', 'state', 'party', 'context']].drop_duplicates('id'), on='id', how='left').fillna('')
        df['second'] = ('speaker: ' + df.speaker.str.replace('-', ' ') + '; party: ' + df.party + '; state: ' + df.state + '; job: ' + df.job +
                        '; context: ' + df.context + '; subject: ' + df.subject.str.replace(',', ', ')) if kind == 'liar_meta' else None
    return [df[df.split == s].reset_index(drop=True) for s in ['train', 'valid', 'test']]


def encode(tok, frame, idx, max_len):
    first = frame.text.iloc[idx].tolist()
    second = frame.second.iloc[idx].tolist() if frame.second.notna().all() else None
    return tok(first, second, padding=True, truncation='longest_first' if second else True, max_length=max_len, return_tensors='pt')


def predict(model, tok, frame, max_len, bs=64):
    model.eval(); out = []
    with torch.inference_mode():
        for i in range(0, len(frame), bs):
            batch = encode(tok, frame, np.arange(i, min(i + bs, len(frame))), max_len).to(DEVICE)
            out.append(model(**batch).logits.float().softmax(-1).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['fakenewsnet', 'liar'], default='fakenewsnet')
    ap.add_argument('--model', default='roberta-base')
    ap.add_argument('--input', choices=['title', 'url_title', 'liar_meta'], default='title')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--max-len', type=int, default=96)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--limit', type=int, default=0, help='smoke test: use only the first N rows of each split')
    a = ap.parse_args()
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
    OUT.mkdir(parents=True, exist_ok=True)
    name = f'{a.dataset}_{a.tag}'
    out_dir = ROOT / 'models' / name
    tr, va, te = load(a.dataset, a.input)
    if a.limit:
        tr, va, te = tr.head(a.limit), va.head(a.limit), te.head(a.limit)
        name += '_smoke'; out_dir = ROOT / 'models' / name
    classes = np.sort(tr.label.unique())
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=len(classes)).to(DEVICE)
    y = torch.tensor([list(classes).index(v) for v in tr.label])
    counts = np.bincount(y.numpy(), minlength=len(classes))
    weights = torch.tensor(np.sqrt(len(y) / (len(classes) * counts)), dtype=torch.float32, device=DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=.01)
    steps = (len(y) + a.batch - 1) // a.batch
    sched = get_linear_schedule_with_warmup(opt, int(steps * .1), steps * a.epochs)
    print(name, 'device', DEVICE, 'model', a.model, 'input', a.input, 'train', len(tr), flush=True)
    best, stale, history = -1, 0, []
    for epoch in range(1, a.epochs + 1):
        model.train(); perm = torch.randperm(len(y)).numpy(); total = 0; start = time.monotonic()
        for k, off in enumerate(range(0, len(y), a.batch)):
            idx = perm[off:off + a.batch]
            batch = encode(tok, tr, idx, a.max_len).to(DEVICE)
            loss = torch.nn.functional.cross_entropy(model(**batch).logits.float(), y[idx].to(DEVICE), weight=weights)
            if not torch.isfinite(loss):
                raise SystemExit(f'non-finite loss at epoch {epoch} step {k}')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True); total += loss.item()
            if k % 100 == 0:
                print(name, 'epoch', epoch, 'step', k, '/', steps, 'loss', round(loss.item(), 4), flush=True)
        prob = predict(model, tok, va, a.max_len)
        f = float(f1_score(va.label, classes[prob.argmax(1)], average='macro'))
        history.append({'epoch': epoch, 'val_macro_f1': round(f, 4), 'mean_loss': round(total / steps, 4), 'seconds': round(time.monotonic() - start, 1)})
        print(name, 'EPOCH', json.dumps(history[-1]), flush=True)
        if f > best:
            best, stale = f, 0
            model.save_pretrained(out_dir); tok.save_pretrained(out_dir)
            np.save(OUT / f'{name}_valid.npy', prob)
            np.save(OUT / f'{name}_test.npy', predict(model, tok, te, a.max_len))  # stored only; selection never looks at it
        else:
            stale += 1
        (OUT / f'{name}_history.json').write_text(json.dumps({'model': a.model, 'input': a.input, 'seed': a.seed, 'classes': classes.tolist(), 'history': history}, indent=2))
        if stale >= 2:
            break
    print(name, 'BEST val macro-F1', round(best, 4), flush=True)
    del model, opt; gc.collect()
    if DEVICE == 'mps':
        torch.mps.empty_cache()


if __name__ == '__main__':
    main()
