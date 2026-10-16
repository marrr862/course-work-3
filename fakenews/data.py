import html
import re
import unicodedata
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

LABELS = ['pants-fire', 'false', 'barely-true', 'half-true', 'mostly-true', 'true']


def clean_text(text):
    text = unicodedata.normalize('NFKC', html.unescape(str(text)))
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' URL ', text)
    return re.sub(r'\s+', ' ', text).strip().lower()


def text_key(text):
    return re.sub(r'[^\w]+', ' ', clean_text(text)).strip()


def deduplicate(df):
    df = df.copy()
    audit = {'raw_rows': len(df)}
    df['text'] = df['text'].fillna('').astype(str)
    df['key'] = df.text.map(text_key)
    nonempty = df.key.str.contains(r'\w', regex=True)
    audit['empty_removed'] = int((~nonempty).sum())
    df = df[nonempty].copy()
    conflicts = df.groupby('key').label.nunique()
    bad = set(conflicts[conflicts > 1].index)
    audit['conflicting_rows_removed'] = int(df.key.isin(bad).sum())
    df = df[~df.key.isin(bad)]
    before = len(df)
    df = df.drop_duplicates('key', keep='first').reset_index(drop=True)
    audit['duplicates_removed'] = before - len(df)
    audit['retained_rows'] = len(df)
    return df, audit


def load_liar(root):
    frames = []
    for split, filename in [('train', 'train.tsv'), ('valid', 'valid.tsv'), ('test', 'test.tsv')]:
        raw = pd.read_csv(Path(root) / filename, sep='\t', header=None, dtype=str, keep_default_na=False)
        if raw.shape[1] != 14:
            raise ValueError(f'Expected 14 LIAR columns, got {raw.shape[1]}')
        frames.append(pd.DataFrame({'id': raw[0], 'label': raw[1], 'text': raw[2], 'source': 'liar', 'split': split}))
    df, audit = deduplicate(pd.concat(frames, ignore_index=True))
    if not set(df.label) <= set(LABELS):
        raise ValueError('Unexpected LIAR label')
    return df, audit


def load_fakenews(root):
    frames = []
    for source in ['politifact', 'gossipcop']:
        for label in ['fake', 'real']:
            raw = pd.read_csv(Path(root) / f'{source}_{label}.csv', usecols=['id', 'title'], dtype=str)
            frames.append(pd.DataFrame({'id': raw.id, 'text': raw.title, 'label': label, 'source': source}))
    df, audit = deduplicate(pd.concat(frames, ignore_index=True))
    train, rest = train_test_split(df.index, test_size=.30, random_state=42, stratify=df.source + ':' + df.label)
    valid, test = train_test_split(rest, test_size=.50, random_state=42, stratify=df.loc[rest].source + ':' + df.loc[rest].label)
    df['split'] = 'train'
    df.loc[valid, 'split'] = 'valid'
    df.loc[test, 'split'] = 'test'
    return df, audit


def assert_disjoint(df):
    keys = {s: set(df.loc[df.split == s, 'key']) for s in ['train', 'valid', 'test']}
    assert not keys['train'] & keys['valid']
    assert not keys['train'] & keys['test']
    assert not keys['valid'] & keys['test']
