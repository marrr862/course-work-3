"""Recent-news search: feed parsing and the relevance filter (no network)."""
import pytest
from fakenews.news_search import distinctive_terms, key_terms, parse_feed, recent_news

FEED = """<?xml version="1.0" encoding="UTF-8"?><rss><channel>
<item><title>NASA delays Artemis III moon landing to 2028 - Reuters</title><link>https://news.google.com/rss/articles/a</link>
<pubDate>Wed, 19 Nov 2026 10:00:00 GMT</pubDate><source url="https://www.reuters.com">Reuters</source></item>
<item><title>Artemis moon mission delayed again, NASA says - AP News</title><link>https://news.google.com/rss/articles/b</link>
<pubDate>Tue, 18 Nov 2026 09:00:00 GMT</pubDate><source url="https://apnews.com">AP News</source></item>
<item><title>Best moon photos of the year - Some Blog</title><link>https://news.google.com/rss/articles/c</link>
<pubDate>Mon, 17 Nov 2026 09:00:00 GMT</pubDate><source url="https://blog.example">Some Blog</source></item>
<item><title>Bad link item - X</title><link>javascript:alert(1)</link><source url="https://x">X</source></item>
</channel></rss>"""


class Resp:
    def __init__(self, text, code=200): self.text, self.status_code, self.content = text, code, text.encode()
    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx
            raise httpx.HTTPStatusError('err', request=None, response=None)


class Client:
    def __init__(self, resp): self.resp, self.queries = resp, []
    def get(self, url, params): self.queries.append(params['q']); return self.resp


@pytest.fixture(autouse=True)
def search_on(monkeypatch):
    monkeypatch.setenv('NEWS_SEARCH', '1')


def test_key_terms_drop_stop_words_and_plurals():
    assert key_terms('BREAKING: Mars has no moons!') == ['mar', 'moon']


def test_parse_feed_strips_publisher_and_rejects_non_https():
    items = parse_feed(FEED)
    assert items[0] == {'title': 'NASA delays Artemis III moon landing to 2028', 'publisher': 'Reuters',
                        'url': 'https://news.google.com/rss/articles/a', 'published': '2026-11-19'}
    assert all(i['url'].startswith('https://') for i in items) and len(items) == 3


def test_only_relevant_articles_are_kept():
    client = Client(Resp(FEED))
    found = recent_news('NASA delays Artemis moon mission', client=client)
    titles = [a['title'] for a in found['articles']]
    assert found['available'] and 'Best moon photos of the year' not in titles
    assert titles[0].startswith('Artemis moon mission delayed') or titles[0].startswith('NASA delays')
    assert client.queries == ['nasa delays artemis moon mission', 'nasa delays artemis mission']


def test_network_failure_is_not_available():
    found = recent_news('NASA delays Artemis moon mission', client=Client(Resp('', 503)))
    assert found['available'] is False and found['articles'] == [] and found['error'] == 'HTTPStatusError'


def test_disabled_or_too_short(monkeypatch):
    assert recent_news('Hello')['available'] is False
    monkeypatch.setenv('NEWS_SEARCH', '0')
    assert recent_news('NASA delays Artemis moon mission')['available'] is False


def test_cut_off_first_word_and_acronyms():
    h = '...ens of thousands of UK fridges secured from hackers in secret nationwide operation'
    assert key_terms(h)[:3] == ['thousand', 'uk', 'fridge']
    assert distinctive_terms(h) == ['uk', 'fridges', 'secured', 'hackers']


def test_reworded_article_still_matches():
    feed = FEED.replace('Best moon photos of the year - Some Blog', 'Thousands of UK smart fridges protected from hackers - BBC').replace('>Some Blog<', '>BBC<')
    found = recent_news('...ens of thousands of UK fridges secured from hackers in secret nationwide operation', client=Client(Resp(feed)))
    assert [a['publisher'] for a in found['articles']] == ['BBC']


def test_all_caps_headline_is_not_read_as_acronyms():
    assert key_terms('BREAKING: MARS HAS NO MOONS') == ['mar', 'moon']
