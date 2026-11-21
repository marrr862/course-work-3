"""Published fact-check search: relevance filter, rating classes, bot formatting (no network)."""
import pytest
from fakenews.factcheck_search import factcheck_search, rating_class
from fakenews.telegram_bot import Analyzer, format_factchecks, reply_for

PAYLOAD = {'claims': [
    {'text': 'Mars has no moons', 'claimReview': [
        {'publisher': {'name': 'Snopes'}, 'url': 'https://www.snopes.com/fact-check/mars-moons/', 'textualRating': 'False',
         'reviewDate': '2026-03-02T00:00:00Z'},
        {'publisher': {'name': 'AFP'}, 'url': 'https://factcheck.afp.com/mars', 'textualRating': 'Pants on Fire',
         'reviewDate': '2025-11-01T00:00:00Z'}]},
    {'text': 'Vaccines cause autism', 'claimReview': [
        {'publisher': {'name': 'X'}, 'url': 'https://x.example/v', 'textualRating': 'False', 'reviewDate': '2024-01-01'}]},
    {'text': 'Mars moons are fake news', 'claimReview': [
        {'publisher': {'name': 'Bad'}, 'url': 'javascript:alert(1)', 'textualRating': 'False'}]},
]}


class Resp:
    def __init__(self, data, code=200): self.data, self.status_code = data, code
    def json(self): return self.data
    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx
            raise httpx.HTTPStatusError('err', request=None, response=None)


class Client:
    def __init__(self, resp): self.resp, self.params = resp, None
    def get(self, url, params): self.params = params; return self.resp


@pytest.fixture
def key(monkeypatch):
    monkeypatch.setenv('FACTCHECK_SEARCH', '1')
    monkeypatch.setenv('GOOGLE_FACT_CHECK_API_KEY', 'test-key')


@pytest.mark.parametrize('rating,cls', [('False', 'false'), ('Pants on Fire!', 'false'), ('Mostly True', 'true'),
                                        ('Half True', 'mixed'), ('Misleading', 'mixed'), ('Not true', 'false'),
                                        ('Four Pinocchios', 'false'), ('Unrated', 'unknown')])
def test_rating_class(rating, cls):
    assert rating_class(rating) == cls


def test_only_relevant_https_reviews_are_kept(key):
    client = Client(Resp(PAYLOAD))
    found = factcheck_search('Mars has no moons.', client=client)
    assert client.params['key'] == 'test-key'
    assert [r['publisher'] for r in found['reviews']] == ['Snopes', 'AFP']
    assert found['summary'] == 'false' and found['reviews'][0]['date'] == '2026-03-02'


def test_missing_key_and_errors(monkeypatch, key):
    assert factcheck_search('Mars has no moons.', client=Client(Resp({}, 403)))['error'] == 'HTTPStatusError'
    monkeypatch.setenv('GOOGLE_FACT_CHECK_API_KEY', '')
    monkeypatch.setattr('fakenews.factcheck_search.ROOT', __import__('pathlib').Path('/nonexistent'))
    assert factcheck_search('Mars has no moons.')['error'] == 'api_key_missing'


def test_bot_formatting_and_escaping():
    text = format_factchecks({'available': True, 'summary': 'false', 'reviews': [
        {'claim': 'Mars <b>has</b> no moons', 'rating': 'False', 'rating_class': 'false', 'publisher': 'Snopes',
         'date': '2026-03-02', 'url': 'https://www.snopes.com/x'}]})
    assert 'Snopes' in text and 'կեղծ' in text and '&lt;b&gt;' in text and '2026-03-02' in text
    assert format_factchecks({'available': False, 'reviews': [], 'error': 'api_key_missing'}) == ''
    assert 'չի գտնվել' in format_factchecks({'available': True, 'reviews': []})


def test_bot_calls_factcheck_for_news():
    from tests.test_telegram_bot import StubClient
    seen = []
    def fake(text):
        seen.append(text)
        return {'available': True, 'reviews': [], 'summary': None}
    answer = reply_for('Mars has no moons.', Analyzer(StubClient(), news_search=lambda t: {'available': False}, factcheck=fake))
    assert seen == ['Mars has no moons.'] and 'Փաստաստուգողներ' in answer
