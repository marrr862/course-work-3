"""Stage-2 stacked models: feature helpers, stacking arithmetic, saved artifacts and API routing."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from fastapi.testclient import TestClient
import fakenews.app as app_module
from fakenews.stacking import FakeNewsFeatures, StackedClassifier, domain, normalize_meta

ROOT = Path(__file__).resolve().parents[1]
client = TestClient(app_module.app)


class FixedBase:
    """Stand-in for the stage-1 neural ensemble (no torch needed)."""
    def __init__(self, classes, row):
        self.classes_ = np.asarray(classes); self.row = np.asarray(row, dtype=float)
    def predict_proba(self, texts):
        return np.tile(self.row, (len(list(texts)), 1))


@pytest.mark.parametrize('url,expected', [
    ('https://www.dailymail.co.uk/tvshowbiz/article-1.html', 'dailymail.co.uk'),
    ('variety.com/2017/biz/news/tax-march', 'variety.com'),
    ('HTTP://WWW2.Example.com:8080/a?b=c', 'example.com'),
    ('https://user@site.org/x', 'site.org'),
    (None, 'NONE'), ('', 'NONE'), (float('nan'), 'NONE'),
])
def test_domain(url, expected):
    assert domain(url) == expected


def test_metadata_normalization_matches_liar_conventions():
    assert normalize_meta('speaker', 'Barack Obama') == 'barack-obama'
    assert normalize_meta('party', ' Democrat ') == 'democrat'
    assert normalize_meta('subject', 'Health care, Taxes') == 'health-care,taxes'
    assert normalize_meta('state', '') == 'NA'
    assert normalize_meta('job', None) == 'NA'


def tiny_stack(alpha=.5, threshold=.5):
    titles = ['shocking secret revealed', 'shocking truth exposed', 'you will not believe this', 'secret exposed now',
              'senate passes budget bill', 'court rules on budget', 'senate votes on bill', 'court passes new rule'] * 3
    labels = (['fake'] * 4 + ['real'] * 4) * 3
    urls = ['https://hoax.example/a'] * 4 + ['https://news.example/b'] * 4
    frame = pd.DataFrame({'key': titles, 'domain': [domain(u) for u in urls * 3], 'label': labels})
    feats = FakeNewsFeatures(True).fit(frame)
    lr = LogisticRegression(max_iter=1000).fit(feats.transform(frame), frame.label)
    return StackedClassifier('unused.joblib', feats, lr, alpha, ['fake', 'real'], threshold=threshold, positive='fake').set_base(FixedBase(['real', 'fake'], [.4, .6]))


def test_stacking_is_weighted_average_with_aligned_classes():
    stack = tiny_stack(alpha=.5)
    texts, urls = ['shocking secret exposed'], ['https://hoax.example/x']
    lr = stack.lr_proba(stack.features.frame(texts, urls))
    # base returns columns [real, fake] -> must be realigned to [fake, real]
    np.testing.assert_allclose(stack.predict_proba(texts, urls), .5 * np.array([[.6, .4]]) + .5 * lr)
    assert stack.predict(texts, urls)[0] == 'fake'


def test_alpha_one_equals_stage1_and_threshold_is_used():
    stack = tiny_stack(alpha=1., threshold=.65)
    np.testing.assert_allclose(stack.predict_proba(['anything']), [[.6, .4]])
    assert stack.predict(['anything'])[0] == 'real'  # 0.6 < 0.65
    stack.threshold = .55
    assert stack.predict(['anything'])[0] == 'fake'


def test_pickle_never_contains_the_neural_base(tmp_path):
    stack = tiny_stack()
    path = tmp_path / 's.joblib'; joblib.dump(stack, path)
    loaded = joblib.load(path)
    assert loaded._base is None and loaded.base_path == 'unused.joblib'


V2 = ROOT / 'models/fakenewsnet_v2.joblib'


@pytest.mark.skipif(not V2.exists(), reason='run python -m fakenews.improve_v2 first')
def test_saved_fakenewsnet_v2_reproduces_reported_test_predictions():
    artifact = joblib.load(V2)
    man = pd.read_csv(ROOT / 'reports/fakenewsnet/split_manifest.csv')
    te = man[man.split == 'test'].reset_index(drop=True)
    saved = pd.read_csv(ROOT / 'reports/v2/fakenewsnet_v2_test_predictions.csv')
    assert (saved.id.values == te.id.values).all()
    urls = pd.concat([pd.read_csv(p, usecols=['id', 'news_url']) for p in [ROOT / f'data/raw/fakenewsnet/{s}_{l}.csv' for s in ['politifact', 'gossipcop'] for l in ['fake', 'real']]]).drop_duplicates('id')
    te = te.merge(urls, on='id', how='left')
    base = np.load(ROOT / 'reports/optimization/fakenewsnet/test_probabilities.npy')
    for kind, column in [('text', 'stack_neural+word+char'), ('domain', 'stack_neural+word+char+domain')]:
        model = artifact['models'][kind]
        frame = model.features.frame(te.text.tolist(), te.news_url.tolist())
        pred = model.decide(model.combine(base, frame))
        assert (pred == saved[column].values).all()
        assert (pred == te.label.values).mean() == pytest.approx(artifact['metrics'][kind]['accuracy'], abs=5e-5)
    assert artifact['metrics']['domain']['macro_f1'] > artifact['metrics']['text']['macro_f1']


@pytest.fixture
def stub_models(monkeypatch):
    """Route the API through stage-2 logic with stub stage-1 models (no torch)."""
    class Stage1(FixedBase):
        pass
    vocabulary = type('V', (), {'transform': staticmethod(lambda texts: type('X', (), {'nnz': 1})())})()
    stage1 = {'model': Stage1(['fake', 'real'], [.3, .7]), 'vocabulary_gate': vocabulary, 'training_keys': set(), 'metrics': {}}
    stack = tiny_stack(alpha=.5)
    v2 = {'models': {'text': stack, 'domain': stack, 'metadata': stack}, 'metrics': {'text': {}, 'domain': {}, 'metadata': {}}}
    monkeypatch.setattr(app_module, 'load_model', lambda dataset: stage1)
    monkeypatch.setattr(app_module, 'load_v2', lambda dataset: v2)
    monkeypatch.setattr(app_module, 'load_v3', lambda dataset: None)
    monkeypatch.setattr(app_module, 'load_v4', lambda dataset: None)
    return stack


def analyze(**body):
    return client.post('/analyze', json={'text': 'Shocking secret exposed', **body})


def test_api_uses_domain_model_only_with_url(stub_models):
    with_url = analyze(dataset='fakenewsnet', source_url='https://hoax.example/story').json()
    assert with_url['model_version'] == 'stage2_domain' and with_url['source_domain_used']
    without = analyze(dataset='fakenewsnet').json()
    assert without['model_version'] == 'stage2_text' and not without['source_domain_used']


def test_api_keeps_evaluated_stage1_thresholds_for_other_modes(stub_models):
    result = analyze(dataset='fakenewsnet', decision_mode='high_recall', source_url='https://hoax.example/x').json()
    assert result['model_version'] == 'stage1'


def test_api_explicit_stage1_request(stub_models):
    assert analyze(dataset='fakenewsnet', source_url='https://hoax.example/x', model='stage1').json()['model_version'] == 'stage1'


def test_api_liar_metadata_switches_to_stage2(stub_models):
    assert analyze(dataset='liar').json()['model_version'] == 'stage1'
    assert analyze(dataset='liar', metadata={'speaker': '  '}).json()['model_version'] == 'stage1'
    assert analyze(dataset='liar', metadata={'speaker': 'Barack Obama', 'party': 'democrat'}).json()['model_version'] == 'stage2_metadata'


@pytest.mark.parametrize('url', ['not a url', 'javascript:alert(1)', 'http://', 'x' * 2001])
def test_api_rejects_invalid_source_url(stub_models, url):
    assert analyze(dataset='fakenewsnet', source_url=url).status_code == 422


def test_api_prefers_stage3_when_available(stub_models, monkeypatch):
    v3 = {'models': {'text': stub_models, 'domain': stub_models}, 'metrics': {}}
    monkeypatch.setattr(app_module, 'load_v3', lambda dataset: v3)
    assert analyze(dataset='fakenewsnet').json()['model_version'] == 'stage3_text'
    r = analyze(dataset='fakenewsnet', source_url='https://hoax.example/x').json()
    assert r['model_version'] == 'stage3_domain' and r['source_domain_used']
    assert analyze(dataset='liar', metadata={'speaker': 'x'}).json()['model_version'] == 'stage2_metadata'


V3 = ROOT / 'models/fakenewsnet_v3.joblib'


@pytest.mark.skipif(not V3.exists(), reason='run python -m fakenews.train_large first')
def test_saved_stage3_reproduces_reported_metrics():
    artifact = joblib.load(V3)
    reported = pd.read_csv(ROOT / 'reports/v3/fakenewsnet_v3_results.csv')
    for kind in ['text', 'domain']:
        row = reported[reported.model.str.startswith(f'stage3_{kind}')].iloc[0]
        assert artifact['metrics'][kind]['accuracy'] == pytest.approx(row.accuracy, abs=5e-5)
        assert artifact['metrics'][kind]['macro_f1'] == pytest.approx(row.macro_f1, abs=5e-5)


def test_api_prefers_stage4_over_stage3(stub_models, monkeypatch):
    both = {'models': {'text': stub_models, 'domain': stub_models}, 'metrics': {}}
    monkeypatch.setattr(app_module, 'load_v3', lambda dataset: both)
    monkeypatch.setattr(app_module, 'load_v4', lambda dataset: both)
    assert analyze(dataset='fakenewsnet').json()['model_version'] == 'stage4_text'
    assert analyze(dataset='fakenewsnet', source_url='https://hoax.example/x').json()['model_version'] == 'stage4_domain'


def test_api_translates_armenian_and_russian(stub_models, monkeypatch):
    import fakenews.translate as tr
    monkeypatch.setattr(tr, 'available', lambda lang: True)
    monkeypatch.setattr(tr, 'translate', lambda text, lang: 'Shocking secret exposed')
    r = client.post('/analyze', json={'text': 'Ցնցող գաղտնիք է բացահայտվել', 'dataset': 'fakenewsnet'}).json()
    assert r['detected_language'] == 'hy' and r['translated_text'] == 'Shocking secret exposed' and r['analyzed_text'] == 'Shocking secret exposed'
    assert client.post('/analyze', json={'text': 'Шокирующая тайна раскрыта', 'dataset': 'fakenewsnet'}).json()['detected_language'] == 'ru'
    assert analyze(dataset='fakenewsnet').json()['detected_language'] == 'en'


def test_api_without_translation_model_returns_422(stub_models, monkeypatch):
    import fakenews.translate as tr
    monkeypatch.setattr(tr, 'available', lambda lang: lang == 'en')
    assert client.post('/analyze', json={'text': 'Шокирующая тайна раскрыта', 'dataset': 'fakenewsnet'}).status_code == 422


@pytest.mark.parametrize('text,lang', [('Mars has no moons.', 'en'), ('Մարսը արբանյակներ չունի', 'hy'), ('У Марса нет спутников', 'ru'),
                                       ('火星没有卫星', 'other'), ('12345', 'other')])
def test_detect_language(text, lang):
    from fakenews.translate import detect_language
    assert detect_language(text) == lang


V4 = ROOT / 'models/fakenewsnet_v4.joblib'


@pytest.mark.skipif(not V4.exists(), reason='run python -m fakenews.stack_v4 --build first')
def test_saved_stage4_reproduces_reported_test_metrics():
    from fakenews.improve_v2 import M
    artifact = joblib.load(V4)
    man = pd.read_csv(ROOT / 'reports/fakenewsnet/split_manifest.csv')
    y = man[man.split == 'test'].label.values
    for kind in ['text', 'domain']:
        model = artifact['models'][kind]
        p = np.load(ROOT / f'reports/v4/fakenewsnet_v4_{kind}_test_fake_proba.npy')
        pred = model.decide(np.column_stack([p, 1 - p]))
        assert M(y, pred, 'fake')['macro_f1'] == pytest.approx(artifact['metrics'][kind]['macro_f1'], abs=5e-5)
    assert artifact['metrics']['domain']['macro_f1'] > artifact['metrics']['text']['macro_f1']


def test_occlusion_finds_the_decisive_word():
    from fakenews.explain import occlusion
    predict = lambda texts: np.array([[.9, .1] if 'BREAKING' in t else [.3, .7] for t in texts])
    result = occlusion(predict, 'BREAKING celebrity couple married', 0)
    assert result['words'][0] == {'word': 'BREAKING', 'position': 0, 'effect': .6} and len(result['words']) == 1
    assert occlusion(predict, 'single', 0) is None and occlusion(predict, 'w ' * 61, 0) is None


def test_api_explanation_uses_the_routed_model(stub_models):
    r = analyze(dataset='fakenewsnet', source_url='https://hoax.example/x', explain=True).json()
    assert r['explanation']['method'].startswith('leave-one-word-out')
    assert {w['word'] for w in r['explanation']['words']} <= {'Shocking', 'secret', 'exposed'}
    assert analyze(dataset='fakenewsnet').json()['explanation'] is None


def test_api_liar_prefers_stage4_with_metadata(stub_models, monkeypatch):
    v4 = {'models': {'metadata': stub_models}, 'metrics': {}}
    monkeypatch.setattr(app_module, 'load_v4', lambda dataset: v4 if dataset == 'liar' else None)
    assert analyze(dataset='liar', metadata={'speaker': 'Barack Obama'}).json()['model_version'] == 'stage4_metadata'
    assert analyze(dataset='liar').json()['model_version'] == 'stage1'


def test_liar_pair_text_matches_training_format():
    from fakenews.stacking import liar_pair_text
    assert liar_pair_text({'speaker': 'Barack Obama', 'party': 'Democrat', 'subject': 'health-care,taxes'}) == \
        'speaker: barack obama; party: democrat; state: ; job: ; context: ; subject: health-care, taxes'


V4L = ROOT / 'models/liar_v4.joblib'


@pytest.mark.skipif(not V4L.exists(), reason='run python -m fakenews.stack_v4 --build --dataset liar first')
def test_saved_liar_stage4_reproduces_reported_metrics():
    from fakenews.improve_v2 import M
    artifact = joblib.load(V4L)
    man = pd.read_csv(ROOT / 'reports/liar/split_manifest.csv')
    y = man[man.split == 'test'].label.values
    model = artifact['models']['metadata']
    p = np.load(ROOT / 'reports/v4/liar_v4_metadata_test_proba.npy')
    assert M(y, model.decide(p))['macro_f1'] == pytest.approx(artifact['metrics']['metadata']['macro_f1'], abs=5e-5)
    assert artifact['metrics']['metadata']['val_macro_f1'] > artifact['metrics']['stage2_reference']['val_macro_f1']
