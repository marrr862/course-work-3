"""Recent news search (Google News RSS, no API key) with a strict relevance filter.

The classifiers were trained on 2010–2018 headlines and cannot know today's events.
This module only answers "do current news outlets report something with the same
key words?" It never turns a search result into a verdict: an article can repeat a
false story, and a true story can be missing from the feed.

    from fakenews.news_search import recent_news
    recent_news('NASA delays Artemis moon mission')
"""
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


def key_terms(text, stem=True):
    """Content words, lower-cased; with stem=True a plural 's' is removed so 'moons' matches 'moon'."""
    words = re.findall(r"[A-Za-z][A-Za-z'-]+", text.lower())
    terms = []
    for word in words:
        word = word.strip("'-")
        if len(word) < 3 or word in ENGLISH_STOP_WORDS or word in EXTRA_STOP:
            continue
        terms.append(word[:-1] if stem and word.endswith('s') and len(word) > 3 else word)
    return list(dict.fromkeys(terms))


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


def recent_news(text, limit=3, min_relevance=0.6, client=None, timeout=8):
    """Return up to `limit` recent articles whose titles share most key terms with `text`.

    Result: {'available': bool, 'articles': [...], 'query': str}. `available` is False when
    the search could not run (no network, disabled with NEWS_SEARCH=0, too few key terms).
    """
    terms = key_terms(text)
    query = ' '.join(key_terms(text, stem=False)[:8])  # search with the words as written
    if os.environ.get('NEWS_SEARCH', '1') == '0' or len(terms) < 2:
        return {'available': False, 'articles': [], 'query': query}
    try:
        http = client or httpx.Client(headers=HEADERS, timeout=timeout, follow_redirects=True)
        response = http.get(FEED, params={'q': query, 'hl': 'en-US', 'gl': 'US', 'ceid': 'US:en'})
        response.raise_for_status()
        if len(response.content) > MAX_BYTES:
            return {'available': False, 'articles': [], 'query': query}
        items = parse_feed(response.text)
    except (httpx.HTTPError, ElementTree.ParseError, ValueError):
        return {'available': False, 'articles': [], 'query': query}
    needed = max(2, round(min_relevance * len(terms[:8])))
    scored = []
    for item in items:
        score = relevance(terms[:8], item['title'])
        if score * len(terms[:8]) >= needed:
            scored.append({**item, 'relevance': round(score, 2)})
    scored.sort(key=lambda a: (a['relevance'], a['published'] or ''), reverse=True)
    unique, seen = [], set()
    for article in scored:
        key = article['title'].lower()
        if key not in seen:
            seen.add(key)
            unique.append(article)
    return {'available': True, 'articles': unique[:limit], 'query': query}
