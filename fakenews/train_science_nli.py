"""Adapt NLI on official SciFact train only, excluding the 150 tuning claims."""
import json,time,gc
import numpy as np
import torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification,get_linear_schedule_with_warmup
from .science import get_index
from .train import ROOT,write_json
OUT=ROOT/'reports/reliability/scifact'


def main():
    torch.set_num_threads(4);torch.manual_seed(2026);np.random.seed(2026)
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    excluded=set(json.loads((OUT/'tuning_ids.json').read_text()))
    cases=[json.loads(s) for s in (ROOT/'data/scifact/claims_train.jsonl').read_text().splitlines()]
    index=get_index();docs={str(d['doc_id']):d for d in index.data['documents']};pairs=[];tuning=[]
    for c in cases:
        target=tuning if c['id'] in excluded else pairs
        evidence=c['evidence']
        for doc_id,items in evidence.items():
            d=docs[doc_id];label=0 if items[0]['label']=='CONTRADICT' else 1
            target.append({'id':c['id'],'premise':d['title']+'. '+' '.join(d['abstract']),'claim':c['claim'],'label':label})
            # Rationale views are training-only augmentations of the same training claim.
            if c['id'] not in excluded:
                ids=sorted({i for item in items for i in item['sentences']})
                target.append({'id':c['id'],'premise':d['title']+'. '+' '.join(d['abstract'][i] for i in ids),'claim':c['claim'],'label':label})
        neutral=[str(i) for i in c['cited_doc_ids'] if str(i) not in evidence]
        if not neutral:
            scores=(index.data['matrix']@index.data['vectorizer'].transform([c['claim']]).T).toarray().ravel()
            neutral=[str(index.data['documents'][i]['doc_id']) for i in np.argsort(-scores)[:10] if str(index.data['documents'][i]['doc_id']) not in evidence][:1]
        for doc_id in neutral[:1]:
            d=docs[doc_id];target.append({'id':c['id'],'premise':d['title']+'. '+' '.join(d['abstract']),'claim':c['claim'],'label':2})
    assert not {p['id'] for p in pairs}&{p['id'] for p in tuning}
    write_json(OUT/'adaptation_manifest.json',{'training_claims':len({p['id'] for p in pairs}),'training_pairs':len(pairs),'tuning_claims':len(excluded),'tuning_pairs':len(tuning),'seed':2026,'device':device,'epochs':3,'learning_rate':1e-5,'batch_size':4,'accumulation':4,'max_tokens':384,'selection':'tuning pair macro-F1; official dev not read','neutral_caveat':'Non-gold retrieved documents are assumed neutral; unannotated evidence can introduce label noise.'})
    tok=AutoTokenizer.from_pretrained(ROOT/'models/nli_deberta',local_files_only=True)
    model=AutoModelForSequenceClassification.from_pretrained(ROOT/'models/nli_deberta',local_files_only=True).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=.01)
    steps=(len(pairs)+15)//16;scheduler=get_linear_schedule_with_warmup(opt,int(steps*.3),steps*3)
    labels=np.array([p['label'] for p in pairs]);counts=np.bincount(labels,minlength=3)
    weights=torch.tensor(np.sqrt(len(labels)/(3*counts)),dtype=torch.float32,device=device)
    best=-1;history=[]
    from sklearn.metrics import f1_score,accuracy_score
    for epoch in range(1,4):
        model.train();order=np.random.permutation(len(pairs));opt.zero_grad(set_to_none=True);start=time.monotonic()
        for batchno,offset in enumerate(range(0,len(order),4)):
            batch=[pairs[i] for i in order[offset:offset+4]]
            encoded=tok([p['premise'] for p in batch],[p['claim'] for p in batch],padding=True,truncation='only_first',max_length=384,return_tensors='pt').to(device)
            y=torch.tensor([p['label'] for p in batch],device=device)
            loss=torch.nn.functional.cross_entropy(model(**encoded).logits,y,weight=weights)/4
            loss.backward()
            if (batchno+1)%4==0 or offset+4>=len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                opt.step();scheduler.step();opt.zero_grad(set_to_none=True)
            if batchno%80==0:print('epoch',epoch,'pairs',offset,'/',len(pairs),'loss',round(loss.item()*4,4),flush=True)
        model.eval();pred=[]
        with torch.inference_mode():
            for offset in range(0,len(tuning),4):
                batch=tuning[offset:offset+4]
                encoded=tok([p['premise'] for p in batch],[p['claim'] for p in batch],padding=True,truncation='only_first',max_length=384,return_tensors='pt').to(device)
                pred.extend(model(**encoded).logits.argmax(-1).cpu().tolist())
        y=[p['label'] for p in tuning];score=f1_score(y,pred,average='macro',zero_division=0)
        row={'epoch':epoch,'tuning_pair_macro_f1':score,'tuning_pair_accuracy':accuracy_score(y,pred),'seconds':time.monotonic()-start};history.append(row);print(json.dumps(row),flush=True)
        if score>best:
            best=score;model.save_pretrained(ROOT/'models/nli_science');tok.save_pretrained(ROOT/'models/nli_science')
        write_json(OUT/'adaptation_history.json',history)

if __name__=='__main__':main()
