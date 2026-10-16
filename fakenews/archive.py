"""Archive lookup, never an assertion that lexical similarity proves a claim."""
from pathlib import Path
import re
import unicodedata
import joblib


def exact_key(text):
    # Keep URLs, numbers, punctuation inside the claim, and negation unchanged.
    text = unicodedata.normalize('NFKC', str(text)).casefold()
    return re.sub(r'\s+', ' ', text).strip().removesuffix('.').strip()


class Archive:
    def __init__(self, path=None):
        path = path or Path(__file__).resolve().parents[1] / 'models/evidence_archive.joblib'
        self.data = joblib.load(path) if path.exists() else None
        self.exact = {}
        if self.data:
            for record in self.data['records']:
                self.exact.setdefault(exact_key(record['claim']), []).append(record)

    def search(self, text, limit=5):
        if not self.data:
            return {'available': False, 'match': None, 'candidates': []}
        matches = self.exact.get(exact_key(text), [])
        match = matches[0] if len(matches) == 1 else None
        d = self.data
        scores = (.65 * (d['word_matrix'] @ d['word_vectorizer'].transform([text]).T)
                  + .35 * (d['char_matrix'] @ d['char_vectorizer'].transform([text]).T)).toarray().ravel()
        candidates = [{**d['records'][i], 'similarity': round(float(scores[i]), 5),
                       'match_type': 'exact' if exact_key(text) == exact_key(d['records'][i]['claim']) else 'related_only'}
                      for i in scores.argsort()[::-1][:limit] if scores[i] >= .12]
        return {'available': True, 'match': match, 'candidates': candidates}
