"""Context-preserving evidence verification for NASA/Wikipedia articles."""
from functools import lru_cache
import json,re
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS,TfidfVectorizer
from .advanced_models import embed
from .verify import get_nli,verify_text as legacy_verify_text
from .train import ROOT


def passages(documents):
    result=[]
    for d in documents:
        for paragraph in d['text'].splitlines():
            paragraph=paragraph.strip()
            if len(paragraph.split())<3:continue
            parts=re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])',paragraph)
            windows=[paragraph] if len(paragraph.split())<=140 else [' '.join(parts[i:i+3]) for i in range(0,len(parts),2)]
            for text in windows:
                if 3<=len(text.split())<=200:
                    result.append({'text':text,'source':{k:v for k,v in d.items() if k!='text'}})
    return result

@lru_cache(maxsize=3)
def prepare(serialized):
    ps=passages(json.loads(serialized))
    if not ps:return [],None,None,None
    tfidf=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,stop_words='english')
    matrix=tfidf.fit_transform([p['text'] for p in ps])
    vectors=embed([p['source']['title']+'. '+p['text'] for p in ps])
    return ps,tfidf,matrix,vectors


def retrieve(claim,documents,limit=5):
    ps,tfidf,matrix,vectors=prepare(json.dumps(documents,sort_keys=True))
    if not ps:return []
    words=[w for w in re.findall(r'[a-z]+',claim.lower()) if w not in ENGLISH_STOP_WORDS]
    names=[w.lower() for w in re.findall(r'\b[A-Z][a-z]+\b',claim) if w.lower() not in ENGLISH_STOP_WORDS]
    anchors=set(names or words[:1]);semantic=vectors@embed([claim])[0]
    lexical=(matrix@tfidf.transform([claim]).T).toarray().ravel()
    ranked=[]
    for i,p in enumerate(ps):
        title=set(re.findall(r'[a-z]+',p['source']['title'].lower()))
        text=set(re.findall(r'[a-z]+',p['text'].lower()))
        # Entity/topic agreement supplements semantic similarity to reduce off-topic NLI errors.
        if anchors and not anchors&(title|text):continue
        if lexical[i]<=0:continue
        ranked.append({**p,'similarity':float(lexical[i]),'semantic_similarity':float(semantic[i])})
    return sorted(ranked,key=lambda p:p['semantic_similarity'],reverse=True)[:limit]


def decide(claim,candidates,threshold=.92,minimum_similarity=.45,limit=1):
    selected=[p for p in candidates[:limit] if p['semantic_similarity']>=minimum_similarity]
    support=[p for p in selected if p['nli_scores']['entailment']>=threshold]
    refute=[p for p in selected if p['nli_scores']['contradiction']>=threshold]
    status='insufficient_evidence';best=None
    if support and refute:status='conflicting_evidence'
    elif refute:status='refuted_by_sources';best=max(refute,key=lambda p:p['nli_scores']['contradiction'])
    elif support:status='supported_by_sources';best=max(support,key=lambda p:p['nli_scores']['entailment'])
    result={'claim':claim,'status':status,'correction':best['text'] if refute and not support else None,'evidence':([best]+[p for p in selected if p is not best]) if best else selected}
    if result['correction']:result.update(correction_method='verbatim_source_passage',correction_source=best['source'])
    return result


def config():
    path=ROOT/'reports/reliability/context/selection.json'
    return json.loads(path.read_text())['configuration'] if path.exists() else {'threshold':.92,'minimum_similarity':.45,'limit':1}


def verify_claim(claim,documents):
    candidates=retrieve(claim,documents)
    scores=get_nli().score([(p['source']['title']+'. '+p['text'],claim) for p in candidates])
    for p,s in zip(candidates,scores):p['nli_scores']=s
    return decide(claim,candidates,**config())


def verify_text(text,live=False,source_urls=None):
    from .sources import cached_sources
    paths=list((ROOT/'data/source_cache_v2').glob('*.json'))
    docs=[{**json.loads(p.read_text()),'delivery':'cached_snapshot'} for p in paths] if paths else cached_sources()
    result=legacy_verify_text(text,live,source_urls,verifier=verify_claim,initial_documents=docs)
    result['method']='Context-preserving passages, semantic ranking, topic overlap and NLI; quoted corrections retain attribution.'
    result['verifier_version']='context-v2'
    return result
