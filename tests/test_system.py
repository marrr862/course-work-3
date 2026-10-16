import json
from pathlib import Path
import httpx
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from fakenews.app import app
from fakenews.data import clean_text, deduplicate, assert_disjoint
from fakenews.evidence import EvidenceLookup

client = TestClient(app)
ROOT = Path(__file__).resolve().parents[1]


def test_preprocess_preserves_negation_numbers():
    assert clean_text('<b>NOT</b> 30 &amp; 365.25') == 'not 30 & 365.25'


def test_conflicts_and_cross_split_duplicates():
    frame = pd.DataFrame({'text': ['A claim.', 'a CLAIM!', 'Other', 'Other', None], 'label': ['true', 'true', 'true', 'false', 'true'], 'split': ['train', 'test', 'train', 'test', 'valid']})
    cleaned, audit = deduplicate(frame)
    assert len(cleaned) == 1
    assert cleaned.iloc[0]['split'] == 'train'
    assert audit['conflicting_rows_removed'] == 2
    assert audit['empty_removed'] == 1
    assert_disjoint(cleaned)


@pytest.mark.parametrize('dataset', ['liar', 'fakenewsnet'])
def test_real_split_integrity_and_metrics(dataset):
    frame = pd.read_csv(ROOT / 'reports' / dataset / 'split_manifest.csv')
    assert_disjoint(frame)
    predictions = pd.read_csv(ROOT / 'reports' / dataset / 'test_predictions.csv')
    metrics = json.loads((ROOT / 'reports' / dataset / 'test_metrics.json').read_text())
    assert len(predictions) == metrics['test_rows']
    assert (predictions.label == predictions.prediction).mean() == pytest.approx(metrics['accuracy'])


@pytest.mark.parametrize('text', ['The Sun is a planet.', '  THE SUN IS A PLANET  '])
def test_sourced_correction(text):
    result = client.post('/analyze', json={'text': text}).json()
    assert result['evidence']['verdict'] == 'false'
    assert result['evidence']['correction']
    assert result['evidence']['sources'][0]['url'].startswith('https://science.nasa.gov/')


@pytest.mark.parametrize('text', ['The Sun is not a planet.', 'The Sun is a planet and Earth has no oceans.', 'Earth takes about 36525 days to orbit the Sun.', 'This is a new claim about the city budget.'])
def test_similar_or_unknown_claim_never_auto_corrected(text):
    result = EvidenceLookup().search(text)
    assert result['status'] == 'insufficient_evidence'
    assert result['correction'] is None


@pytest.mark.parametrize('text', ['', '   ', '123456', 'Սա հայերեն պնդում է', 'Это русское утверждение', 'x'*10001])
def test_invalid_input(text):
    assert client.post('/analyze', json={'text': text}).status_code == 422


def test_invalid_dataset():
    assert client.post('/analyze', json={'text': 'Some claim', 'dataset': '../../etc/passwd'}).status_code == 422


def test_out_of_vocabulary_abstention():
    result = client.post('/analyze', json={'text': 'zxqwvbbnmm qqqzxv'}).json()
    assert result['model_prediction'] is None


def test_key_missing(monkeypatch):
    monkeypatch.delenv('GOOGLE_FACT_CHECK_API_KEY', raising=False)
    assert EvidenceLookup().search('A new unknown claim', online=True)['online_status'] == 'api_key_missing'


def test_online_failure(monkeypatch):
    monkeypatch.setenv('GOOGLE_FACT_CHECK_API_KEY', 'test-key')
    def fail(*args, **kwargs):
        raise httpx.ConnectError('unavailable')
    monkeypatch.setattr('fakenews.evidence.httpx.get', fail)
    result = EvidenceLookup().search('A new unknown claim', online=True)
    assert result['online_status'] == 'service_unavailable'
    assert result['correction'] is None


def test_online_results_require_review(monkeypatch):
    monkeypatch.setenv('GOOGLE_FACT_CHECK_API_KEY', 'test-key')
    def fake_get(*args, **kwargs):
        return httpx.Response(200, request=httpx.Request('GET', 'https://example.org'), json={'claims': [{'text': 'A related claim', 'claimReview': [{'url': 'https://example.org/check', 'publisher': {'name': 'Test fixture'}, 'textualRating': 'False'}]}]})
    monkeypatch.setattr('fakenews.evidence.httpx.get', fake_get)
    result = EvidenceLookup().search('A new unknown claim', online=True)
    assert result['status'] == 'review_required'
    assert result['verdict'] is None
    assert result['correction'] is None
    assert len(result['sources']) == 1


def test_service_routes():
    assert client.get('/').status_code == 200
    assert client.get('/health').json()['ready']
    assert client.get('/metrics').status_code == 200
    assert client.get('/docs').status_code == 200


def test_archive_is_train_only():
    from fakenews.archive import Archive
    archive = Archive()
    frame = pd.read_csv(ROOT / 'reports/liar/split_manifest.csv')
    train_ids = set(frame.loc[frame.split == 'train', 'id'])
    heldout_ids = set(frame.loc[frame.split != 'train', 'id'])
    ids = {r['id'] for r in archive.data['records']}
    assert ids <= train_ids
    assert not ids & heldout_ids
    assert len(ids) > 10000


def test_archived_explanation_has_provenance():
    text = 'In Georgia, women earn 82 cents for every dollar earned by men.'
    result = client.post('/analyze', json={'text': text}).json()
    evidence = result['evidence']
    assert evidence['status'] == 'archived_claim_match'
    assert evidence['correction_type'] == 'archived_explanation_not_rewritten_claim'
    assert evidence['sources'][0]['id'] == '13077.json'
    assert '#L' in evidence['sources'][0]['url']
    assert evidence['sources'][0]['context']


def test_paraphrase_returns_candidate_not_verdict():
    result = client.post('/analyze', json={'text': 'The idea of renting the Lincoln Bedroom originated with Terry McAuliffe.'}).json()['evidence']
    assert result['verdict'] is None
    assert result['correction'] is None
    assert any(r['id'] == '8187.json' for r in result['archive_candidates'])


@pytest.mark.parametrize('text', [
 'In Georgia, women earn 99 cents for every dollar earned by men.',
 'It is not true that In Georgia, women earn 82 cents for every dollar earned by men.',
 'In Georgia, women earn 82 cents for every dollar earned by men. This is still true today.',
 'The Sun is a planet. https://example.com',
])
def test_changed_context_never_inherits_correction(text):
    result = client.post('/analyze', json={'text': text}).json()['evidence']
    assert result['correction'] is None
    assert result['verdict'] is None


def test_missing_archive_still_works(tmp_path):
    from fakenews.archive import Archive
    archive = Archive(tmp_path / 'missing.joblib')
    result = EvidenceLookup(archive=archive).search('Unseen claim')
    assert result['archive_available'] is False
    assert result['correction'] is None


@pytest.mark.parametrize('payload', [[], {'claims': None}, {'claims': [None, {}, {'claimReview': [None, {'url': None}]}]}])
def test_malformed_online_response_is_safe(monkeypatch, payload):
    monkeypatch.setenv('GOOGLE_FACT_CHECK_API_KEY', 'test-key')
    monkeypatch.setattr('fakenews.evidence.httpx.get', lambda *a, **k: httpx.Response(200, request=httpx.Request('GET', 'https://example.org'), json=payload))
    result = EvidenceLookup().search('A new unknown claim', online=True)
    assert result['correction'] is None
    assert result['online_status'] in ['service_unavailable', 'no_results']
