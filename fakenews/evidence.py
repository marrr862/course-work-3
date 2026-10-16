"""Source-grounded responses with explicit abstention for unverified claims."""
import json
import os
from pathlib import Path
import httpx
from sklearn.feature_extraction.text import TfidfVectorizer
from .archive import Archive, exact_key


class EvidenceLookup:
    def __init__(self, path=None, archive=None):
        self.entries = json.loads((path or Path(__file__).with_name('evidence.json')).read_text())
        self.exact = {exact_key(claim): entry for entry in self.entries for claim in entry['claims']}
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform([' '.join(e['claims']) for e in self.entries])
        self.archive = archive or Archive()

    def search(self, text, online=False):
        match = self.exact.get(exact_key(text))
        if match:
            return {'status': 'verified_local_match', 'verdict': match['verdict'], 'correction': match['correction'], 'sources': [match], 'online_status': 'not_needed', 'method': 'exact match to a manually curated claim', 'correction_type': 'curated_source_based_correction'}
        found = self.archive.search(text)
        if found['match']:
            record = found['match']
            return {'status': 'archived_claim_match', 'verdict': record['verdict'],
                    'correction': record['explanation'], 'correction_type': 'archived_explanation_not_rewritten_claim',
                    'sources': [record], 'archive_candidates': found['candidates'], 'online_status': 'not_needed',
                    'method': 'exact match to a LIAR-PLUS training claim; returns the archived rating and extracted explanation',
                    'caveat': 'Historical assessment tied to the displayed speaker and context. This is an archived excerpt, not current verification or an independently generated corrected sentence.'}
        scores = (self.matrix @ self.vectorizer.transform([text]).T).toarray().ravel()
        suggestions = [{**self.entries[i], 'similarity': round(float(scores[i]), 4)} for i in scores.argsort()[::-1][:3] if scores[i] >= .25]
        result = {'status': 'insufficient_evidence', 'verdict': None, 'correction': None, 'correction_type': None,
                  'sources': [], 'related_local_records': suggestions, 'archive_candidates': found['candidates'],
                  'archive_available': found['available'], 'online_status': 'not_requested',
                  'method': 'TF-IDF ranks related archived claims; lexical similarity never transfers a verdict or correction'}
        if not online:
            return result
        key = os.environ.get('GOOGLE_FACT_CHECK_API_KEY')
        if not key:
            result['online_status'] = 'api_key_missing'
            return result
        try:
            response = httpx.get('https://factchecktools.googleapis.com/v1alpha1/claims:search', params={'query': text, 'languageCode': 'en', 'pageSize': 5, 'key': key}, timeout=10)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict) or not isinstance(payload.get('claims', []), list):
                raise ValueError('Malformed response')
            records = []
            for claim in payload.get('claims', []):
                if not isinstance(claim, dict) or not isinstance(claim.get('claimReview', []), list):
                    continue
                for review in claim.get('claimReview', []):
                    if not isinstance(review, dict):
                        continue
                    url = review.get('url', '')
                    if not isinstance(url, str) or not url.startswith('https://'):
                        continue
                    publisher = review.get('publisher') or {}
                    records.append({'claim': claim.get('text'), 'publisher': publisher.get('name') if isinstance(publisher, dict) else None, 'url': url, 'rating': review.get('textualRating'), 'review_date': review.get('reviewDate')})
            result['sources'] = records
            result['online_status'] = 'results_found' if records else 'no_results'
            if records:
                result['status'] = 'review_required'
        except (httpx.HTTPError, ValueError, TypeError, AttributeError):
            result['online_status'] = 'service_unavailable'
        return result
