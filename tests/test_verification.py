from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from fakenews.app import app
from fakenews.sources import valid_url
from fakenews.verify import retrieve, verify_claim, verify_text, split_claims

client=TestClient(app)


def doc(text,title='Mars facts',url='https://science.nasa.gov/mars/facts/'):
    return {'text':text,'title':title,'url':url,'retrieved_at':'2026-09-29','publisher':'NASA'}


class StubNLI:
    def __init__(self, scores):self.scores=scores
    def score(self,pairs):return self.scores[:len(pairs)]


def test_unrelated_evidence_excluded_before_nli():
    assert retrieve('The mayor bought a new car yesterday.',[doc('Mars has two small moons and a dusty surface.')])==[]


def test_refutation_quotes_source_without_generation():
    source=doc('Mars has two small moons called Phobos and Deimos.')
    result=verify_claim('Mars has no moons.',[source],StubNLI([{'contradiction':.99,'entailment':.005,'neutral':.005}]))
    assert result['status']=='refuted_by_sources'
    assert result['correction']==source['text']
    assert result['correction_source']['url']==source['url']


def test_conflict_abstains():
    sources=[doc('Mars has two small moons called Phobos and Deimos.'),doc('Mars has no moons in this alternative account.',url='https://en.wikipedia.org/wiki/Mars')]
    scores=[{'contradiction':.99,'entailment':.001,'neutral':.009},{'contradiction':.001,'entailment':.99,'neutral':.009}]
    result=verify_claim('Mars has no moons.',sources,StubNLI(scores))
    assert result['status']=='conflicting_evidence'
    assert result['correction'] is None


@pytest.mark.parametrize('url',['http://science.nasa.gov/mars/facts/','https://127.0.0.1/secret','https://en.wikipedia.org.evil.test/wiki/Mars','https://user:pass@en.wikipedia.org/wiki/Mars','https://en.wikipedia.org:8443/wiki/Mars','file:///etc/passwd'])
def test_source_origin_restriction(url):
    with pytest.raises(ValueError):valid_url(url)
    assert client.post('/verify',json={'text':'Mars has no moons.','source_urls':[url]}).status_code==422


def test_compound_claim_requires_split():
    result=verify_text('Mars has moons and the mayor won the election.')
    assert result['status']=='needs_shorter_claims'
    assert result['corrected_text'] is None


def test_claim_limit():
    assert verify_text('Mars has moons. '*6)['status']=='needs_shorter_claims'


def test_decimal_sentence_split():
    assert len(split_claims('Earth takes 365.25 days to orbit the Sun. Mars has moons.'))==2


def test_live_fetch_error_is_explicit(monkeypatch):
    monkeypatch.setattr('fakenews.verify.wikipedia_search',lambda text:([],['wikipedia_search_unavailable']))
    monkeypatch.setattr('fakenews.verify.cached_sources',lambda:[])
    result=verify_text('A village mayor resigned today.',live=True)
    assert result['status']=='needs_review'
    assert result['provider_errors']


def test_live_model_supports_new_wording():
    result=client.post('/verify',json={'text':'Mars does not have any moons.'}).json()
    assert result['status']=='correction_proposed'
    assert result['corrected_text']
    assert result['claims'][0]['correction_source']['url'].startswith('https://')


def test_unknown_claim_abstains():
    result=client.post('/verify',json={'text':'An unnamed local politician resigned today.'}).json()
    assert result['corrected_text'] is None
    assert result['status']=='needs_review'


def test_partial_evidence_does_not_rewrite_entire_article():
    result=client.post('/verify',json={'text':'Mars has no moons. An unnamed senator resigned today.'}).json()
    assert result['status']=='needs_review'
    assert result['corrected_text'] is None
    assert result['claims'][0]['correction']
    assert result['claims'][1]['correction'] is None


def test_multi_claim_correction_preserves_supported_statement():
    result=client.post('/verify',json={'text':'The Sun is a star. Mars has no moons.'}).json()
    assert result['status']=='correction_proposed'
    assert result['corrected_text'].startswith('The Sun is a star.')
    assert 'https://science.nasa.gov/mars/facts/' in result['corrected_text']
