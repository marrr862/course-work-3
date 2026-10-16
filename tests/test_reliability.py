import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from fakenews.app import app
from fakenews.sources import fetch_article
from fakenews.science import decide
from fakenews.train import ROOT
client=TestClient(app)


def test_high_recall_mode_changes_only_threshold():
    folder=ROOT/'reports/optimization/fakenewsnet'
    probs=np.load(folder/'test_probabilities.npy')[:,0]
    predictions=pd.read_csv(folder/'test_predictions.csv')
    idx=np.flatnonzero((probs>.32)&(probs<.45))[0]
    text=predictions.iloc[int(idx)].text
    standard=client.post('/analyze',json={'text':text,'dataset':'fakenewsnet','model':'stage1'}).json()
    sensitive=client.post('/analyze',json={'text':text,'dataset':'fakenewsnet','decision_mode':'high_recall'}).json()
    assert standard['model_prediction']=='real'
    assert sensitive['model_prediction']=='fake'
    assert standard['model_scores']==sensitive['model_scores']
    assert sensitive['decision_threshold']==pytest.approx(.3)


def test_invalid_or_inapplicable_decision_modes():
    assert client.post('/analyze',json={'text':'An example headline.','decision_mode':'high_recall','dataset':'liar'}).status_code==422
    assert client.post('/analyze',json={'text':'An example headline.','decision_mode':'perfect','dataset':'fakenewsnet'}).status_code==422


def test_source_extraction_retains_short_facts(monkeypatch,tmp_path):
    import httpx
    html='<html><title>Mercury facts</title><main><p>Mercury does not have moons.</p><p>Mercury is a rocky planet near the Sun and has a surface marked by numerous impact craters from ancient collisions with other objects in the solar system.</p></main></html>'
    monkeypatch.setattr(httpx.Client,'get',lambda self,url:httpx.Response(200,request=httpx.Request('GET',url),text=html))
    source=fetch_article('https://science.nasa.gov/mercury/facts/',cache_dir=tmp_path)
    assert 'Mercury does not have moons.' in source['text']
    assert len(list(tmp_path.glob('*.json')))==1


def test_scientific_conflicts_and_low_relevance_abstain():
    passages=[{'doc_id':'a','text':'one','semantic_similarity':.8,'lexical_similarity':.2,'nli_scores':{'entailment':.99,'contradiction':.001}}, {'doc_id':'b','text':'two','semantic_similarity':.8,'lexical_similarity':.2,'nli_scores':{'entailment':.001,'contradiction':.99}}]
    assert decide(passages)[0]=='conflicting_evidence'
    assert decide([{**passages[0],'semantic_similarity':.1}])[0]=='insufficient_evidence'


def test_frozen_threshold_metrics_recompute():
    folder=ROOT/'reports/optimization/fakenewsnet'
    probabilities=np.load(folder/'test_probabilities.npy')[:,0]
    frame=pd.read_csv(folder/'test_predictions.csv');gold=(frame.label=='fake').to_numpy()
    modes=json.loads((ROOT/'reports/reliability/decision_modes_test.json').read_text())
    for values in modes.values():
        predicted=probabilities>=values['threshold']
        assert float((predicted==gold).mean())==pytest.approx(values['accuracy'])
        assert int((gold&~predicted).sum())==values['false_negatives']


def test_scientific_holdout_recomputes_and_is_disjoint():
    from fakenews.benchmark_science import evaluate
    folder=ROOT/'reports/reliability/scifact'
    selection=json.loads((folder/'selection.json').read_text())
    rows=json.loads((folder/'holdout_scored.json').read_text())
    recorded=json.loads((folder/'holdout_metrics.json').read_text())
    tuning=set(json.loads((folder/'tuning_ids.json').read_text()))
    train={json.loads(line)['id'] for line in (ROOT/'data/scifact/claims_train.jsonl').read_text().splitlines()}
    assert len(rows)==300
    assert not {r['id'] for r in rows}&train
    assert tuning <= train and len(train-tuning)==659
    actual,_=evaluate(rows,selection['selected']['configuration'])
    for key in ['accuracy','macro_f1_three_gold_classes','coverage','refutation_precision']:
        assert actual[key]==pytest.approx(recorded[key])


def test_scientific_api_matches_evaluation_and_exposes_source():
    rows=json.loads((ROOT/'reports/reliability/scifact/holdout_predictions.json').read_text())
    sample=next(r for r in rows if r['predicted']=='refuted_by_sources')
    response=client.post('/verify-science',json={'text':sample['claim']})
    assert response.status_code==200,response.text
    result=response.json()
    assert result['status']==sample['predicted']
    assert result['corrected_text'] is None
    assert result['claims'][0]['correction'] is None
    source=result['claims'][0]['evidence'][0]['source']
    assert source['doc_id']==sample['evidence_doc_id']
    page=client.get(source['url'])
    assert page.status_code==200 and 'Dataset provenance' in page.text
    assert client.get('/science-source/999999999999').status_code==404
    assert client.post('/verify-science',json={'text':'First scientific assertion. Second scientific assertion.'}).status_code==422
