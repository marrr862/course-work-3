"""Recent news search (Google News RSS, no API key) with a strict relevance filter.

The classifiers were trained on 2010–2018 headlines and cannot know today's events.
This module only answers "do current news outlets report something with the same
key words?" It never turns a search result into a verdict: an article can repeat a
false story, and a true story can be missing from the feed.

    from fakenews.news_search import recent_news
    recent_news('NASA delays Artemis moon mission')
"""
import math
import os
import re
from email.utils import parsedate_to_datetime
from xml.etree import ElementTree
import httpx
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

FEED = 'https://news.google.com/rss/search'
HEADERS = {'User-Agent': 'FakeNewsResearch/1.0 (local academic fact verification prototype)'}
MAX_BYTES = 2_000_000
EXTRA_STOP = {'says', 'said', 'claim', 'claims', 'breaking', 'report', 'reports', 'new', 'news', 'just', 'video', 'watch'}
# Common headline words that say little about the topic: kept for matching, skipped for the short query.
GENERIC = {'thousand', 'thousands', 'million', 'millions', 'hundred', 'hundreds', 'people', 'year', 'years', 'day', 'days',
           'week', 'today', 'first', 'last', 'secret', 'secretly', 'nationwide', 'operation', 'official', 'officials',
           'major', 'huge', 'big', 'latest', 'update', 'shocking', 'amazing', 'world', 'time', 'times', 'man', 'woman'}


def clean_headline(text):
    """Drop a cut-off first word after a leading ellipsis ('...ens of thousands' -> 'of thousands')."""
    return re.sub(r'^\s*(?:\.{2,}|…)\S*\s*', '', text or '')


def is_acronym(word):
    return 2 <= len(word) <= 5 and word.isupper() and word.lower() not in EXTRA_STOP


def key_terms(text, stem=True):
    """Content words, lower-cased. Acronyms such as UK, US, EU are kept although short.

    With stem=True a plural 's' is removed so 'moons' matches 'moon'.
    """
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", clean_headline(text))
    shouting = sum(w.isupper() for w in words) > len(words) / 2  # an all-caps headline has no real acronyms
    terms = []
    for word in words:
        word = word.strip("'-")
        low = word.lower()
        acronym = is_acronym(word) and not shouting
        if not acronym and (len(low) < 3 or low in ENGLISH_STOP_WORDS or low in EXTRA_STOP):
            continue
        if stem and not acronym and low.endswith('s') and len(low) > 3:
            low = low[:-1]
        terms.append(low)
    return list(dict.fromkeys(terms))


def distinctive_terms(text, n=4):
    """The n most specific words (names, acronyms, long words), kept in their original order."""
    found = re.findall(r"[A-Za-z][A-Za-z'-]*", clean_headline(text))
    shouting = sum(w.isupper() for w in found) > len(found) / 2
    words = []
    for word in found:
        low = word.lower()
        if (is_acronym(word) and not shouting) or (len(low) >= 3 and low not in ENGLISH_STOP_WORDS and low not in EXTRA_STOP
                                                   and low not in GENERIC):
            score = 2 * (word[0].isupper() and not shouting) + len(word) / 10
            words.append((low, score))
    unique = list(dict.fromkeys(w for w, _ in words))
    best = set(sorted(unique, key=lambda w: -max(sc for x, sc in words if x == w))[:n])
    return [w for w in unique if w in best]


def relevance(query_terms, title):
    """Share of the query's key terms that appear in the article title."""
    if not query_terms:
        return 0.0
    found = set(key_terms(title))
    return sum(term in found for term in query_terms) / len(query_terms)


def parse_feed(xml_text):
    root = ElementTree.fromstring(xml_text)
    items = []
    for item in root.iter('item'):
        title = (item.findtext('title') or '').strip()
        source = item.find('source')
        publisher = (source.text or '').strip() if source is not None else ''
        if publisher and title.endswith(' - ' + publisher):
            title = title[: -len(' - ' + publisher)].strip()
        link = (item.findtext('link') or '').strip()
        try:
            published = parsedate_to_datetime(item.findtext('pubDate') or '').date().isoformat()
        except (TypeError, ValueError):
            published = None
        if title and link.startswith('https://'):
            items.append({'title': title, 'publisher': publisher, 'url': link, 'published': published})
    return items


def recent_news(text, limit=3, min_relevance=0.4, client=None, timeout=8):
    """Return up to `limit` recent articles whose titles share key terms with `text`.

    Two searches are tried (the first six key words, then the four most specific ones) and
    their results merged. An article is kept when its title contains at least 40% of the
    claim's key terms, and never fewer than two. Result: {'available', 'articles', 'query',
    'error'}; `available` is False when the search could not run, and `error` says why.
    """
    terms = key_terms(text)[:8]
    raw = key_terms(text, stem=False)
    queries = list(dict.fromkeys(q for q in [' '.join(raw[:6]), ' '.join(distinctive_terms(text))] if q))
    result = {'available': False, 'articles': [], 'query': queries[0] if queries else '', 'error': None}
    if os.environ.get('NEWS_SEARCH', '1') == '0':
        return {**result, 'error': 'disabled'}
    if len(terms) < 2:
        return {**result, 'error': 'too_few_words'}
    items, errors = [], []
    http = client or httpx.Client(headers=HEADERS, timeout=timeout, follow_redirects=True)
    for query in queries:
        try:
            response = http.get(FEED, params={'q': query, 'hl': 'en-US', 'gl': 'US', 'ceid': 'US:en'})
            response.raise_for_status()
            if len(response.content) > MAX_BYTES:
                raise ValueError('feed too large')
            items.extend(parse_feed(response.text))
        except (httpx.HTTPError, ElementTree.ParseError, ValueError) as error:
            errors.append(type(error).__name__)
    if errors and len(errors) == len(queries):
        return {**result, 'error': errors[0]}
    needed = max(2, math.ceil(min_relevance * len(terms))) if min_relevance > 0 else 0
    scored, seen = [], set()
    for item in items:
        key = item['title'].lower()
        if key in seen:
            continue
        seen.add(key)
        score = relevance(terms, item['title'])
        if score * len(terms) >= needed:
            scored.append({**item, 'relevance': round(score, 2)})
    scored.sort(key=lambda a: (a['relevance'], a['published'] or ''), reverse=True)
    return {**result, 'available': True, 'articles': scored[:limit]}
