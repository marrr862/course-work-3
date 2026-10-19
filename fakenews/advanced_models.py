"""Portable inference for validation-selected semantic/lexical classifiers."""
from functools import lru_cache
from pathlib import Path
from threading import Lock
import numpy as np
from scipy import sparse
ROOT=Path(__file__).resolve().parents[1]

@lru_cache(maxsize=3)
def encoder(path, classification=False, device='cpu'):
    import torch
    from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
    torch.set_num_threads(4)
    absolute=ROOT/path
    tokenizer=AutoTokenizer.from_pretrained(absolute,local_files_only=True)
    cls=AutoModelForSequenceClassification if classification else AutoModel
    model=cls.from_pretrained(absolute,local_files_only=True).to(device).eval()
    return tokenizer,model,Lock()


def embed(texts, device='cpu', batch_size=48):
    import torch
    tok,model,lock=encoder('models/sentence_minilm',device=device)
    result=[]
    with lock,torch.inference_mode():
        for start in range(0,len(texts),batch_size):
            inputs=tok(list(texts[start:start+batch_size]),padding=True,truncation=True,max_length=128,return_tensors='pt').to(device)
            hidden=model(**inputs).last_hidden_state
            mask=inputs['attention_mask'].unsqueeze(-1)
            pooled=(hidden*mask).sum(1)/mask.sum(1).clamp(min=1)
            result.append(torch.nn.functional.normalize(pooled,p=2,dim=1).cpu().numpy())
    return np.concatenate(result)

class SemanticClassifier:
    def __init__(self,classifier,lexical=None,weight=1.):
        self.classifier=classifier;self.lexical=lexical;self.weight=weight;self.classes_=classifier.classes_
    def predict_proba(self,texts):
        texts=list(texts);values=embed(texts)
        if self.lexical is not None:
            values=sparse.hstack([self.lexical.transform(texts),sparse.csr_matrix(values*self.weight)],format='csr')
        return self.classifier.predict_proba(values)
    def predict(self,texts):return self.classes_[self.predict_proba(texts).argmax(1)]

class TransformerClassifier:
    def __init__(self,path,classes):self.path=path;self.classes_=np.asarray(classes)
    def predict_proba(self,texts,device='cpu'):
        import torch
        texts=list(texts);tok,model,lock=encoder(self.path,classification=True,device=device);rows=[]
        with lock,torch.inference_mode():
            for start in range(0,len(texts),48):
                inputs=tok(texts[start:start+48],padding=True,truncation=True,max_length=getattr(self,'max_length',128),return_tensors='pt').to(device)
                rows.append(model(**inputs).logits.softmax(-1).cpu().numpy())
        return np.concatenate(rows)
    def predict(self,texts):return self.classes_[self.predict_proba(texts).argmax(1)]

class ProbabilityEnsemble:
    def __init__(self,models,weights):self.models=models;self.weights=weights;self.classes_=models[0].classes_
    def predict_proba(self,texts):
        texts=list(texts)
        return sum(w*m.predict_proba(texts)[:,[list(m.classes_).index(c) for c in self.classes_]] for w,m in zip(self.weights,self.models))
    def predict(self,texts):return self.classes_[self.predict_proba(texts).argmax(1)]
