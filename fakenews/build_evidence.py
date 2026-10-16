"""Build a provenance-tracked archive from LIAR-PLUS training rows only."""
import hashlib
import json
from pathlib import Path
import httpx
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from .data import text_key

ROOT = Path(__file__).resolve().parents[1]
COMMIT = '992bfd97555631e48b6bddb1a87aa61c51ee1197'
SOURCE = f'https://raw.githubusercontent.com/Tariq60/LIAR-PLUS/{COMMIT}/dataset/tsv/train2.tsv'


def main():
    folder = ROOT / 'data/liar_plus'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'train2.tsv'
    if not path.exists():
        response = httpx.get(SOURCE, timeout=60)
        response.raise_for_status()
        path.write_bytes(response.content)
    raw = pd.read_csv(path, sep='\t', header=None, dtype=str, keep_default_na=False)
    if raw.shape[1] != 16:
        raise ValueError('Expected exported row index plus 15 LIAR-PLUS fields')
    original = pd.read_csv(ROOT / 'reports/liar/split_manifest.csv', dtype=str, keep_default_na=False)
    allowed = original[original.split == 'train'].set_index('id')
    records = []
    excluded = {'not_in_clean_train': 0, 'mismatched_claim_or_label': 0, 'missing_explanation': 0}
    for line, row in raw.iterrows():
        ident, label, claim, explanation = row[1], row[2], row[3], row[15].strip()
        if ident not in allowed.index:
            excluded['not_in_clean_train'] += 1
            continue
        orig = allowed.loc[ident]
        if orig.label != label or text_key(orig.text) != text_key(claim):
            excluded['mismatched_claim_or_label'] += 1
            continue
        if len(explanation.split()) < 8:
            excluded['missing_explanation'] += 1
            continue
        records.append({'id': ident, 'claim': claim, 'verdict': label, 'explanation': explanation,
                        'speaker': row[5], 'context': row[14], 'subject': row[4],
                        'publisher': 'PolitiFact via LIAR-PLUS research archive',
                        'url': f'https://github.com/Tariq60/LIAR-PLUS/blob/{COMMIT}/dataset/tsv/train2.tsv#L{line+1}',
                        'source_type': 'archived_extracted_justification', 'split': 'train'})
    claims = [r['claim'] for r in records]
    word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=100000)
    char = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), sublinear_tf=True, max_features=100000)
    artifact = {'records': records, 'word_vectorizer': word, 'char_vectorizer': char,
                'word_matrix': word.fit_transform(claims), 'char_matrix': char.fit_transform(claims)}
    joblib.dump(artifact, ROOT / 'models/evidence_archive.joblib', compress=3)
    manifest = {'source_url': SOURCE, 'commit': COMMIT, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'raw_rows': len(raw), 'indexed_records': len(records), 'excluded': excluded,
                'classifier_training_changed': False, 'allowed_split': 'cleaned LIAR train only',
                'retrieval': '0.65 word TF-IDF cosine + 0.35 character TF-IDF cosine',
                'automatic_application': 'exact archived claim match only; all other candidates require review',
                'license': 'Research use only; original sources retain copyright'}
    (ROOT / 'reports/evidence_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))

if __name__ == '__main__':
    main()
