"""Scientific abstract retrieval and evidence-relative verification.

Corpus documents are public SciFact abstracts; never treat the output as clinical advice.
"""
from functools import lru_cache
from pathlib import Path
from threading import Lock
import json
import hashlib
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from .advanced_models import embed
from .train import ROOT

INDEX=ROOT/'models/science_index.joblib'
LABELS=['supported_by_sources','refuted_by_sources','insufficient_evidence']

class ScientificIndex:
    def __init__(self,data):self.data=data
    @classmethod
    def build(cls,device='cpu'):
        path=ROOT/'data/scifact/corpus.jsonl';docs=[json.loads(line) for line in path.read_text().splitlines()]
        texts=[d['title']+'. '+' '.join(d['abstract']) for d in docs]
        vectorizer=TfidfVectorizer(ngram_range=(1,2),min_df=1,max_df=.98,max_features=120000,sublinear_tf=True,stop_words='english')
        matrix=vectorizer.fit_transform(texts)
        vectors=embed(texts,device=device)
        data={'documents':docs,'vectorizer':vectorizer,'matrix':matrix,'vectors':vectors,'corpus_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        joblib.dump(data,INDEX)
        return cls(data)
    def scores(self,claim):
        lexical=(self.data['matrix']@self.data['vectorizer'].transform([claim]).T).toarray().ravel()
        semantic=self.data['vectors']@embed([claim])[0]
        return lexical,semantic
    def retrieve(self,claim,weight=.5,limit=3,scores=None):
        lexical,semantic=self.scores(claim) if scores is None else scores
        # Weighted reciprocal-rank fusion avoids equating incompatible cosine scales.
        lr=np.empty(len(lexical),dtype=int);lr[np.argsort(-lexical)]=np.arange(len(lexical))
        sr=np.empty(len(semantic),dtype=int);sr[np.argsort(-semantic)]=np.arange(len(semantic))
        ranking=(1-weight)/(60+lr)+weight/(60+sr)
        result=[]
        for i in np.argsort(-ranking)[:limit]:
            d=self.data['documents'][int(i)]
            result.append({'doc_id':str(d['doc_id']),'title':d['title'],'text':' '.join(d['abstract']),'sentences':d['abstract'],'lexical_similarity':float(lexical[i]),'semantic_similarity':float(semantic[i]),'retrieval_score':float(ranking[i])})
        return result

@lru_cache(maxsize=1)
def get_index():return ScientificIndex(joblib.load(INDEX))

class BatchNLI:
    def __init__(self,device='cpu',model_path='models/nli_deberta'):
        import torch
        from transformers import AutoTokenizer,AutoModelForSequenceClassification
        self.torch=torch;torch.set_num_threads(4);self.device=device;self.lock=Lock()
        self.tokenizer=AutoTokenizer.from_pretrained(ROOT/model_path,local_files_only=True)
        self.model=AutoModelForSequenceClassification.from_pretrained(ROOT/model_path,local_files_only=True).to(device).eval()
    def score(self,pairs):
        rows=[]
        with self.lock,self.torch.inference_mode():
            for start in range(0,len(pairs),4):
                chunk=pairs[start:start+4]
                encoded=self.tokenizer([p[0] for p in chunk],[p[1] for p in chunk],padding=True,truncation='only_first',max_length=512,return_tensors='pt').to(self.device)
                probabilities=self.model(**encoded).logits.softmax(-1).cpu().numpy()
                rows.extend([dict(zip(['contradiction','entailment','neutral'],map(float,row))) for row in probabilities])
        return rows


def decide(passages,threshold=.9,minimum_similarity=.3,limit=3,**ignored):
    considered=[p for p in passages[:limit] if p['semantic_similarity']>=minimum_similarity and p['lexical_similarity']>0]
    supports=[p for p in considered if p['nli_scores']['entailment']>=threshold]
    refutes=[p for p in considered if p['nli_scores']['contradiction']>=threshold]
    if supports and refutes:return 'conflicting_evidence',None
    if refutes:return 'refuted_by_sources',max(refutes,key=lambda p:p['nli_scores']['contradiction'])
    if supports:return 'supported_by_sources',max(supports,key=lambda p:p['nli_scores']['entailment'])
    return 'insufficient_evidence',None
