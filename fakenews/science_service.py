"""Serve the evaluated scientific verifier with dataset-level provenance."""
from functools import lru_cache
import json,time
from .science import get_index,BatchNLI,decide
from .train import ROOT

@lru_cache(maxsize=1)
def get_configuration():
    return json.loads((ROOT/'reports/reliability/scifact/selection.json').read_text())

@lru_cache(maxsize=1)
def get_scorer():
    return BatchNLI(model_path=get_configuration().get('model_path','models/nli_deberta'))


def verify_science(claim):
    start=time.perf_counter();selection=get_configuration();config=selection['selected']['configuration']
    passages=get_index().retrieve(claim,weight=config['weight'],limit=config['limit'])
    scores=get_scorer().score([(p['title']+'. '+p['text'],claim) for p in passages])
    for p,s in zip(passages,scores):p['nli_scores']=s
    status,best=decide(passages,**config)
    if best:passages=[best]+[p for p in passages if p is not best]
    evidence=[{'text':p['text'],'nli_scores':p['nli_scores'],'similarity':p['retrieval_score'],'source':{'title':p['title'],'url':'/science-source/'+p['doc_id'],'publisher':'SciFact corpus · '+p['doc_id'],'source_type':'research_abstract','retrieved_at':'Dataset snapshot; publication date unavailable','dataset_url':'https://github.com/allenai/scifact','doc_id':p['doc_id']}} for p in passages]
    return {'status':status,'claims':[{'claim':claim,'status':status,'correction':None,'evidence':evidence}], 'corrected_text':None,'provider_errors':[],'source_count':len(get_index().data['documents']),'elapsed_seconds':round(time.perf_counter()-start,3),'mode':'scientific_corpus','method':'Evidence retrieval over SciFact abstracts, followed by scientific NLI.','caveat':'Research prototype over a fixed abstract corpus, not medical advice or a current-news truth guarantee. Shows source evidence rather than generating a replacement scientific claim.'}
