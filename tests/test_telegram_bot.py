"""Telegram bot message handling (no network, stub API client)."""
import pytest
from fakenews.telegram_bot import Analyzer, HELP, parse_message, reply_for


class Resp:
    def __init__(self, code, data): self.status_code, self._data = code, data
    def json(self): return self._data
    def raise_for_status(self):
        if self.status_code >= 400: raise RuntimeError(self.status_code)


class StubClient:
    def __init__(self): self.calls = []
    def post(self, path, json):
        self.calls.append((path, json))
        if path == '/analyze':
            if json.get('source_url') == 'https://bad':
                return Resp(422, {})
            pred = 'fake' if json['dataset'] == 'fakenewsnet' else 'half-true'
            return Resp(200, {'model_prediction': pred, 'model_scores': {pred: .83}, 'evidence': {'status': 'insufficient_evidence'},
                              'model_version': 'stage2_domain' if json.get('source_url') else 'stage2_text', 'seen_in_training': False})
        return Resp(200, {'status': 'correction_proposed', 'claims': [{'correction': 'Mars has two small moons, Phobos and Deimos.',
                          'evidence': [{'source': {'title': 'Mars: Facts', 'url': 'https://science.nasa.gov/mars/facts/'}}]}]})


@pytest.mark.parametrize('text,headline,url', [
    ('BREAKING: Couple married!!! https://hollywoodlife.com/a/b', 'BREAKING: Couple married!!!', 'https://hollywoodlife.com/a/b'),
    ('https://variety.com/x - Oscar list', 'Oscar list', 'https://variety.com/x'),
    ('Mars has no moons.', 'Mars has no moons.', None),
])
def test_parse_message(text, headline, url):
    assert parse_message(text) == (headline, url)


def test_news_reply_uses_url_and_shows_correction():
    client = StubClient()
    answer = reply_for('Mars has no moons. https://example.com/story', Analyzer(client))
    assert client.calls[0] == ('/analyze', {'text': 'Mars has no moons.', 'dataset': 'fakenewsnet', 'source_url': 'https://example.com/story'})
    assert 'Կեղծ' in answer and '83%' in answer and 'կայքի տիրույթ' in answer
    assert 'Phobos' in answer and 'science.nasa.gov' in answer


def test_bad_url_falls_back_to_text_only():
    client = StubClient()
    reply_for('Some headline here https://bad', Analyzer(client))
    assert client.calls[1] == ('/analyze', {'text': 'Some headline here', 'dataset': 'fakenewsnet'})


@pytest.mark.parametrize('text', ['这是中文新闻标题', 'https://only-a-link.com', 'ok', 'x' * 1200])
def test_rejected_inputs_never_call_the_api(text):
    client = StubClient()
    answer = reply_for(text, Analyzer(client))
    assert client.calls == [] and answer


def test_commands():
    client = StubClient()
    assert reply_for('/start', Analyzer(client)) == HELP
    assert 'Մասամբ ճիշտ' in reply_for('/liar Says unemployment fell to 4 percent.', Analyzer(client))
    assert client.calls[-1][1]['dataset'] == 'liar'
    assert 'Անհայտ' in reply_for('/foo', Analyzer(client))


def test_html_is_escaped():
    class Evil(StubClient):
        def post(self, path, json):
            r = super().post(path, json)
            if path == '/verify':
                r._data['claims'][0]['correction'] = '<script>x</script>'
            return r
    assert '<script>' not in reply_for('Mars has no moons.', Analyzer(Evil()))


def test_armenian_input_is_sent_and_translation_shown():
    class Translating(StubClient):
        def post(self, path, json):
            r = super().post(path, json)
            if path == '/analyze':
                r._data.update(detected_language='hy', translated_text='Mars has no moons.', analyzed_text='Mars has no moons.')
            return r
    client = Translating()
    answer = reply_for('Մարսը արբանյակներ չունի', Analyzer(client))
    assert client.calls[0][1]['text'] == 'Մարսը արբանյակներ չունի'
    assert client.calls[1] == ('/verify', {'text': 'Mars has no moons.', 'live': True})
    assert 'Թարգմանություն հայերենից' in answer and 'Mars has no moons.' in answer
