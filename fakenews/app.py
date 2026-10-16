from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional
import re
import json
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator
from .data import clean_text, text_key
from .evidence import EvidenceLookup

ROOT = Path(__file__).resolve().parents[1]
app = FastAPI(title='Fake News Research Demo', version='4.0.0')
evidence = EvidenceLookup()


class LiarMetadata(BaseModel):
    speaker: Optional[str] = Field(None, max_length=200)
    party: Optional[str] = Field(None, max_length=100)
    state: Optional[str] = Field(None, max_length=100)
    job: Optional[str] = Field(None, max_length=200)
    context: Optional[str] = Field(None, max_length=300)
    subject: Optional[str] = Field(None, max_length=300)

    def provided(self):
        return any((getattr(self, f) or '').strip() for f in type(self).model_fields)


class Request(BaseModel):
    text: str = Field(min_length=3, max_length=10000)
    dataset: Literal['liar', 'fakenewsnet'] = 'liar'
    online: bool = False
    decision_mode: Literal['standard', 'balanced', 'high_recall'] = 'standard'
    source_url: Optional[str] = Field(None, max_length=2000, description='FakeNewsNet: article URL; only its domain is used, nothing is fetched.')
    metadata: Optional[LiarMetadata] = Field(None, description='LIAR: optional speaker metadata.')
    model: Literal['auto', 'stage1'] = Field('auto', description='auto = best available model for the supplied inputs; stage1 = text-only stage-1 ensemble.')

    @field_validator('source_url')
    @classmethod
    def plausible_url(cls, value):
        if value is None or not value.strip():
            return None
        from .stacking import domain
        value = value.strip()
        if not re.match(r'^(https?://)?[a-z0-9.-]+\.[a-z]{2,}([/:?#].*)?$', value, re.I) or domain(value) == 'NONE':
            raise ValueError('Enter a valid news article URL, for example https://www.example.com/article.')
        return value

    @field_validator('text')
    @classmethod
    def meaningful(cls, value):
        if len(re.findall(r'[A-Za-z]', clean_text(value))) < 3 or re.search(r'[^\x00-\x7f]', value) and re.search(r'[\u0400-\u058f\u4e00-\u9fff]', value):
            raise ValueError('This model supports English text only. Enter an English claim or headline.')
        return value.strip()


@lru_cache(maxsize=2)
def load_model(dataset):
    registry_path = ROOT / 'models/active_models.json'
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else {}
    path = ROOT / registry.get(dataset, f'models/{dataset}.joblib')
    if not path.exists():
        raise HTTPException(503, 'Models are not trained. Run python -m fakenews.train')
    return joblib.load(path)


@lru_cache(maxsize=2)
def load_v2(dataset):
    """Stage-2 stacked models, or None when they have not been trained."""
    registry_path = ROOT / 'models/active_models.json'
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else {}
    path = ROOT / registry.get(f'{dataset}_v2', f'models/{dataset}_v2.joblib')
    return joblib.load(path) if path.exists() else None


def stage2(request, artifact):
    """Pick the stage-2 model that fits the supplied inputs; None means use stage 1."""
    v2 = load_v2(request.dataset)
    if v2 is None or request.model == 'stage1' or request.decision_mode != 'standard':
        return None
    if request.dataset == 'fakenewsnet':
        kind = 'domain' if request.source_url else 'text'
        model = v2['models'][kind]
        model.set_base(artifact['model'])
        return f'stage2_{kind}', model, model.predict_proba([request.text], urls=[request.source_url])[0]
    if request.metadata is not None and request.metadata.provided():
        model = v2['models']['metadata']
        model.set_base(artifact['model'])
        return 'stage2_metadata', model, model.predict_proba([request.text], metadata=[request.metadata.model_dump()])[0]
    return None


@app.get('/')
def home():
    return FileResponse(ROOT / 'fakenews' / 'index.html')


@app.get('/health')
def health():
    ready = {name: (ROOT / 'models' / f'{name}.joblib').exists() for name in ['liar', 'fakenewsnet']}
    stage2_ready = {name: load_v2(name) is not None for name in ['liar', 'fakenewsnet']}
    return {'ready': all(ready.values()), 'models': ready, 'stage2_models': stage2_ready, 'language': 'en', 'archive_records': len(evidence.archive.data['records']) if evidence.archive.data else 0, 'nli_model_ready': (ROOT / 'models/nli_deberta/model.safetensors').exists(), 'science_ready': (ROOT/'models/science_index.joblib').exists() and (ROOT/'reports/reliability/scifact/holdout_metrics.json').exists()}


@app.get('/metrics')
def get_metrics():
    result = {name: load_model(name)['metrics'] for name in ['liar', 'fakenewsnet']}
    for name in ['liar', 'fakenewsnet']:
        v2 = load_v2(name)
        if v2 is not None:
            result[f'{name}_stage2'] = v2['metrics']
    return result


@app.post('/analyze')
def analyze(request: Request):
    artifact = load_model(request.dataset)
    model = artifact['model']
    vocabulary = artifact.get('vocabulary_gate')
    if vocabulary is None:
        vocabulary = model.named_steps['tfidf']
    vector = vocabulary.transform([request.text])
    seen = text_key(request.text) in artifact['training_keys']
    if request.dataset != 'fakenewsnet' and request.decision_mode != 'standard':
        raise HTTPException(422, 'Decision modes are supported for FakeNewsNet only.')
    scores = model.predict_proba([request.text])[0] if vector.nnz else None
    threshold = None
    prediction = str(model.classes_[scores.argmax()]) if scores is not None else None
    if scores is not None and request.dataset == 'fakenewsnet':
        mode_path = ROOT / 'reports/reliability/decision_modes.json'
        if request.decision_mode == 'standard':
            threshold = .5
        elif not mode_path.exists():
            raise HTTPException(503, 'Decision modes have not been evaluated.')
        else:
            threshold = json.loads(mode_path.read_text())[request.decision_mode]['threshold']
        fake_score = float(scores[list(model.classes_).index('fake')])
        prediction = 'fake' if fake_score >= threshold else 'real'
    model_version = 'stage1'
    classes = model.classes_
    if scores is not None:
        chosen = stage2(request, artifact)
        if chosen is not None:
            model_version, stacked, scores = chosen
            classes = stacked.classes_
            prediction = str(stacked.decide(scores[None, :])[0])
            threshold = stacked.threshold
    fact_check = evidence.search(request.text, online=request.online)
    return {'model_version': model_version, 'source_domain_used': model_version == 'stage2_domain', 'verification_status': fact_check['status'], 'dataset': request.dataset, 'model_prediction': prediction, 'decision_mode': request.decision_mode, 'decision_threshold': threshold, 'model_scores': dict(zip(map(str, classes), map(float, scores))) if scores is not None else {}, 'score_note': 'Model scores are not calibrated probabilities of truth.', 'prediction_status': 'statistical_prediction' if prediction else 'out_of_vocabulary', 'seen_in_training': seen, 'warning': 'A text classifier prediction is not a factual verdict. English only. LIAR uses six labels; FakeNewsNet uses headline-only fake/real labels.', 'evidence': fact_check}


class VerifyRequest(Request):
    live: bool = False
    source_urls: list[str] = Field(default_factory=list, max_length=3)

    @field_validator('source_urls')
    @classmethod
    def public_sources(cls, values):
        from .sources import valid_url
        for value in values:
            valid_url(value)
        return values


@app.post('/verify')
def verify(request: VerifyRequest):
    from .verify import verify_text
    return verify_text(request.text, live=request.live, source_urls=request.source_urls)


@app.get('/verification-metrics')
def verification_metrics():
    import json
    path=ROOT / 'reports/verification/test_metrics.json'
    return json.loads(path.read_text()) if path.exists() else {'status':'evaluation_pending'}


@app.get("/reliability-metrics")
def reliability_metrics():
    names={"decision_modes":"decision_modes_test.json","coverage":"coverage_curves.json","science":"scifact/holdout_metrics.json"}
    return {name:json.loads((ROOT/"reports/reliability"/path).read_text()) if (ROOT/"reports/reliability"/path).exists() else {"status":"pending"} for name,path in names.items()}


@app.post('/verify-science')
def verify_science_route(request: Request):
    from .verify import split_claims
    if len(split_claims(request.text)) != 1 or len(request.text.split()) > 100:
        raise HTTPException(422, 'Enter one scientific claim of at most 100 words.')
    if not (ROOT/'reports/reliability/scifact/holdout_metrics.json').exists():
        raise HTTPException(503, 'Scientific verifier evaluation is not complete.')
    from .science_service import verify_science
    return verify_science(request.text)


@app.get('/science-source/{doc_id}')
def science_source(doc_id: int):
    from .science import get_index
    from fastapi.responses import HTMLResponse
    from html import escape
    document = next((d for d in get_index().data['documents'] if d['doc_id'] == doc_id), None)
    if document is None:
        raise HTTPException(404, 'Source not found.')
    title = escape(document['title'])
    paragraphs = ''.join('<p>'+escape(s)+'</p>' for s in document['abstract'])
    return HTMLResponse('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><main style="max-width:800px;margin:40px auto;padding:20px;font:18px/1.6 system-ui"><h1>'+title+'</h1><p>SciFact corpus · document '+str(doc_id)+'. Publication date is not supplied in this dataset.</p>'+paragraphs+'<p><a href="https://github.com/allenai/scifact">Dataset provenance and documentation</a></p><p>This abstract is provided for research verification, not clinical advice.</p></main></html>')
