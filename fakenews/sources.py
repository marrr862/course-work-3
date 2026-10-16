"""Keyless source acquisition with dated snapshots and bounded HTTPS origins."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlparse
import httpx
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'data/source_cache'
HEADERS = {'User-Agent': 'FakeNewsResearch/1.0 (local academic fact verification prototype)'}
ALLOWED_HOSTS = {'science.nasa.gov', 'www.nasa.gov', 'en.wikipedia.org'}


def sentences(text):
    # Protect decimal points and abbreviations by splitting only after punctuation + whitespace.
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"“])|\n+', text)
    return [re.sub(r'\s+', ' ', p).strip() for p in parts if 6 <= len(p.split()) <= 100]


def valid_url(url):
    parsed = urlparse(url)
    if parsed.scheme != 'https' or parsed.hostname not in ALLOWED_HOSTS or parsed.username or parsed.password or parsed.port not in (None, 443):
        raise ValueError('Use an HTTPS NASA or English Wikipedia article URL.')
    if not parsed.path or parsed.path == '/':
        raise ValueError('Provide an article URL, not a homepage.')
    return parsed


def _cache_file(url, cache_dir=None):
    return (Path(cache_dir) if cache_dir is not None else CACHE) / (hashlib.sha256(url.encode()).hexdigest() + '.json')


def fetch_article(url, refresh=False, cache_dir=None):
    parsed = valid_url(url)
    path = _cache_file(url, cache_dir)
    if path.exists() and not refresh:
        cached = json.loads(path.read_text())
        cached['delivery'] = 'cached_snapshot'
        return cached
    with httpx.Client(headers=HEADERS, timeout=15, follow_redirects=False) as client:
        current = url
        for _ in range(4):
            response = client.get(current)
            if response.is_redirect:
                from urllib.parse import urljoin
                current = urljoin(current, response.headers['location'])
                valid_url(current)
                continue
            response.raise_for_status()
            break
        else:
            raise ValueError('Too many redirects')
    if len(response.content) > 3_000_000:
        raise ValueError('Source page exceeds size limit')
    soup = BeautifulSoup(response.text, 'html.parser')
    title = soup.title.get_text(' ', strip=True) if soup.title else url
    for node in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'table', 'figure']):
        node.decompose()
    main = soup.find('main') or soup.find('article') or soup
    paragraphs = [p.get_text(' ', strip=True) for p in main.find_all('p')]
    paragraphs = [p for p in paragraphs if len(p.split()) >= 3]
    text = '\n'.join(dict.fromkeys(paragraphs))[:100000]
    if len(text.split()) < 30:
        raise ValueError('No readable article content')
    item = {'id': hashlib.sha256(url.encode()).hexdigest()[:16], 'url': current, 'requested_url': url,
            'title': title, 'publisher': parsed.hostname, 'text': text,
            'retrieved_at': datetime.now(timezone.utc).isoformat(), 'delivery': 'live_fetch',
            'sha256': hashlib.sha256(text.encode()).hexdigest(),
            'source_type': 'primary_agency' if 'nasa.gov' in parsed.hostname else 'encyclopedia'}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(item, ensure_ascii=False, indent=2))
    return item


def wikipedia_search(claim):
    """Search two short topic queries; obtain page introductions through MediaWiki."""
    tokens = re.findall(r'[A-Za-z][A-Za-z-]+', claim)
    keywords = [t for t in tokens if t.lower() not in ENGLISH_STOP_WORDS and t.lower() not in {'says','said','claim','claims','true','false'}]
    entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', claim)
    entities = [e for e in entities if e.lower() not in ENGLISH_STOP_WORDS]
    queries = list(dict.fromkeys([' '.join(keywords[:7])] + entities[:2]))[:3]
    documents, errors = {}, []
    def fetch_query(query):
        response = httpx.get('https://en.wikipedia.org/w/api.php', params={
            'action':'query','format':'json','generator':'search','gsrsearch':query,'gsrlimit':2,
            'prop':'extracts|info|revisions','explaintext':1,'exintro':1,'inprop':'url','rvprop':'ids'},
            headers=HEADERS, timeout=12)
        response.raise_for_status()
        return response.json().get('query', {}).get('pages', {})
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(fetch_query,q) for q in queries if q]
        for future in futures:
            try:
                pages = future.result()
                for page in pages.values():
                    text = page.get('extract', '')[:16000]
                    if len(text.split()) < 20: continue
                    revision = (page.get('revisions') or [{}])[0].get('revid')
                    url = f'https://en.wikipedia.org/w/index.php?oldid={revision}' if revision else page.get('fullurl')
                    if not url: continue
                    documents[str(page['pageid'])] = {'id': 'wiki-'+str(page['pageid']), 'title': page['title'], 'url': url,
                        'publisher': 'Wikipedia', 'text': text, 'source_type': 'encyclopedia', 'delivery': 'live_search',
                        'retrieved_at': datetime.now(timezone.utc).isoformat(), 'sha256': hashlib.sha256(text.encode()).hexdigest()}
            except (httpx.HTTPError, ValueError, TypeError, KeyError, AttributeError):
                errors.append('wikipedia_search_unavailable')
    return list(documents.values()), sorted(set(errors))


def cached_sources():
    if not CACHE.exists(): return []
    return [{**json.loads(p.read_text()), 'delivery':'cached_snapshot'} for p in sorted(CACHE.glob('*.json'))]
