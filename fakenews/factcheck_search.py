"""Published fact-checks for a claim, up to today (Google Fact Check Tools API).

The API indexes ClaimReview markup from fact-checkers such as PolitiFact, Snopes, Reuters,
AFP, Full Fact and others, updated daily. It needs a free API key in the environment or in
.env as GOOGLE_FACT_CHECK_API_KEY (Google Cloud console -> enable "Fact Check Tools API" ->
Credentials -> API key).

Only reviews whose checked claim shares most key words with the user's text are kept, and each
fact-checker's own rating is shown next to its link; the module never invents a verdict.
"""
import math
import os
from pathlib import Path
import httpx
from .news_search import clean_headline, key_terms, relevance

ROOT = Path(__file__).resolve().parents[1]
API = 'https://factchecktools.googleapis.com/v1alpha1/claims:search'

FALSE_WORDS = ('false', 'fake', 'pants on fire', 'incorrect', 'wrong', 'fabricated', 'hoax', 'not true', 'baseless',
               'no evidence', 'misattributed', 'scam', 'debunked', 'mostly false', 'four pinocchios')
MIXED_WORDS = ('half', 'mixture', 'mixed', 'partly', 'partially', 'misleading', 'missing context', 'needs context',
               'exaggerat', 'unproven', 'unsupported', 'distort', 'out of context', 'satire')
TRUE_WORDS = ('true', 'correct', 'accurate', 'mostly true', 'verified', 'legit')


def api_key():
    key = os.environ.get('GOOGLE_FACT_CHECK_API_KEY')
    env = ROOT / '.env'
    if not key and env.exists():
        for line in env.read_text().splitlines():
            if line.strip().startswith('GOOGLE_FACT_CHECK_API_KEY='):
                key = line.split('=', 1)[1].strip().strip('"\'')
    return key or None


def rating_class(text):
    """Coarse class of a fact-checker's own rating: false, mixed, true or unknown."""
    t = (text or '').lower()
    if any(w in t for w in MIXED_WORDS):
        return 'mixed'
    if any(w in t for w in FALSE_WORDS):
        return 'false'
    if any(w in t for w in TRUE_WORDS):
        return 'true'
    return 'unknown'


def factcheck_search(text, limit=3, min_relevance=0.4, client=None, timeout=10):
    """Return {'available', 'reviews', 'summary', 'error'} for published fact-checks of `text`."""
    terms = key_terms(text)[:8]
    result = {'available': False, 'reviews': [], 'summary': None, 'error': None}
    if os.environ.get('FACTCHECK_SEARCH', '1') == '0':
        return {**result, 'error': 'disabled'}
    key = api_key()
    if not key:
        return {**result, 'error': 'api_key_missing'}
    if len(terms) < 2:
        return {**result, 'error': 'too_few_words'}
    try:
        http = client or httpx.Client(timeout=timeout)
        response = http.get(API, params={'query': clean_headline(text)[:300], 'languageCode': 'en', 'pageSize': 10, 'key': key})
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError) as error:
        return {**result, 'error': type(error).__name__}
    reviews = []
    needed = max(2, math.ceil(min_relevance * len(terms)))
    for claim in payload.get('claims', []) if isinstance(payload, dict) else []:
        if not isinstance(claim, dict):
            continue
        checked = str(claim.get('text') or '')
        score = relevance(terms, checked)
        if score * len(terms) < needed:
            continue
        for review in claim.get('claimReview') or []:
            if not isinstance(review, dict):
                continue
            url = str(review.get('url') or '')
            if not url.startswith('https://'):
                continue
            publisher = review.get('publisher') if isinstance(review.get('publisher'), dict) else {}
            rating = str(review.get('textualRating') or '')
            reviews.append({'claim': checked, 'rating': rating, 'rating_class': rating_class(rating),
                            'publisher': str(publisher.get('name') or publisher.get('site') or ''),
                            'date': str(review.get('reviewDate') or '')[:10], 'url': url, 'relevance': round(score, 2)})
    reviews.sort(key=lambda r: (r['relevance'], r['date']), reverse=True)
    reviews = reviews[:limit]
    classes = [r['rating_class'] for r in reviews if r['rating_class'] != 'unknown']
    summary = None
    if classes:
        top = max(set(classes), key=classes.count)
        summary = top if classes.count(top) == len(classes) else 'disagree'
    return {**result, 'available': True, 'reviews': reviews, 'summary': summary}
