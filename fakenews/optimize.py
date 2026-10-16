"""Finite, predeclared validation search. Evaluation is a separate command."""
import gc
import json
import hashlib
import time
import joblib
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from .train import ROOT,metrics,write_json
from .improve import features
from .advanced_models import embed,encoder,SemanticClassifier,TransformerClassifier,ProbabilityEnsemble

torch.set_num_threads(4)
DEVICE='mps' if torch.backends.mps.is_available() else 'cpu'
OUT=ROOT/'reports/optimization'


def finetune(name,train,valid,classes):
    from transformers import AutoTokenizer,AutoModelForSequenceClassification,get_linear_schedule_with_warmup
    torch.manual_seed(42);np.random.seed(42)
    tok=AutoTokenizer.from_pretrained(ROOT/'models/sentence_minilm',local_files_only=True)
    model=AutoModelForSequenceClassification.from_pretrained(ROOT/'models/sentence_minilm',num_labels=len(classes),local_files_only=True).to(DEVICE)
    labelmap={v:i for i,v in enumerate(classes)}
    inputs=tok(train.text.tolist(),padding='max_length',truncation=True,max_length=128,return_tensors='pt')
    y=torch.tensor([labelmap[v] for v in train.label])
    # Mild inverse-frequency weighting reduces majority collapse without extreme reweighting.
    counts=np.bincount(y.numpy(),minlength=len(classes));weights=torch.tensor(np.sqrt(len(y)/(len(classes)*counts)),dtype=torch.float32,device=DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=.01)
    steps=(len(y)+31)//32
    schedule=get_linear_schedule_with_warmup(opt,num_warmup_steps=int(steps*.5),num_training_steps=steps*5)
    destination=ROOT/'models'/f'{name}_finetuned'
    best=-1;stale=0;history=[];best_prob=None
    for epoch in range(1,6):
        model.train();permutation=torch.randperm(len(y));loss_total=0;start=time.monotonic()
        for offset in range(0,len(y),32):
            idx=permutation[offset:offset+32]
            batch={k:v[idx].to(DEVICE) for k,v in inputs.items()}
            logits=model(**batch).logits
            loss=torch.nn.functional.cross_entropy(logits,y[idx].to(DEVICE),weight=weights)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();schedule.step();opt.zero_grad(set_to_none=True)
            loss_total+=loss.item()
            if offset%3200==0:print(name,'epoch',epoch,'rows',offset,'loss',round(loss.item(),4),flush=True)
        model.eval();prob=[]
        with torch.inference_mode():
            for offset in range(0,len(valid),48):
                batch=tok(valid.text.iloc[offset:offset+48].tolist(),padding=True,truncation=True,max_length=128,return_tensors='pt').to(DEVICE)
                prob.append(model(**batch).logits.softmax(-1).cpu().numpy())
        prob=np.concatenate(prob);result={'epoch':epoch,**metrics(valid.label,np.asarray(classes)[prob.argmax(1)]),'seconds':time.monotonic()-start,'mean_loss':loss_total/steps}
        history.append(result);print(name,'EPOCH RESULT',json.dumps(result),flush=True)
        if result['macro_f1']>best:
            best=result['macro_f1'];stale=0;best_prob=prob
            model.save_pretrained(destination);tok.save_pretrained(destination)
        else:stale+=1
        write_json(OUT/name/'training_history.json',history)
        if stale>=2:break
    del model,opt,inputs;gc.collect()
    if DEVICE=='mps':torch.mps.empty_cache()
    return TransformerClassifier(str(destination.relative_to(ROOT)),classes),best_prob


def main():
    for name in ['liar','fakenewsnet']:
        folder=OUT/name;folder.mkdir(exist_ok=True,parents=True)
        path=ROOT/'reports'/name/'split_manifest.csv';df=pd.read_csv(path)
        train=df[df.split=='train'];valid=df[df.split=='valid'];classes=np.sort(train.label.unique())
        models={};probabilities={};rows=[]
        def register(key,model,probs):
            row={'name':key,**metrics(valid.label,classes[probs.argmax(1)])}
            rows.append(row);models[key]=model;probabilities[key]=probs
            print(name,'VALID',key,round(row['accuracy'],4),round(row['macro_f1'],4),flush=True)
        baseline=joblib.load(ROOT/'models'/f'{name}.joblib')['model']
        register('original',baseline,baseline.predict_proba(valid.text))
        previous=joblib.load(ROOT/'reports/improvements'/name/'candidate.joblib')['model']
        register('previous_search',previous,previous.predict_proba(valid.text))
        for split,frame in [('train',train),('valid',valid)]:
            cache=folder/f'{split}_embeddings.npy'
            if not cache.exists():
                print(name,'encoding',split,len(frame),flush=True)
                np.save(cache,embed(frame.text.tolist(),device=DEVICE))
        x=np.load(folder/'train_embeddings.npy');v=np.load(folder/'valid_embeddings.npy')
        lexical=features('word_char');lx=lexical.fit_transform(train.text);lv=lexical.transform(valid.text)
        for kind in ['semantic','combined']:
            for weight in ([1.] if kind=='semantic' else [.5,1.,2.]):
                tx=x if kind=='semantic' else sparse.hstack([lx,sparse.csr_matrix(x*weight)],format='csr')
                vx=v if kind=='semantic' else sparse.hstack([lv,sparse.csr_matrix(v*weight)],format='csr')
                for c in [.1,1.,10.]:
                    for balance in [None,'balanced']:
                        clf=LogisticRegression(C=c,class_weight=balance,max_iter=1500,random_state=42).fit(tx,train.label)
                        key=f'{kind}_w{weight}_C{c}_{balance}'
                        register(key,SemanticClassifier(clf,None if kind=='semantic' else lexical,weight),clf.predict_proba(vx))
        encoder.cache_clear();gc.collect()
        if DEVICE=='mps':torch.mps.empty_cache()
        neural,probs=finetune(name,train,valid,classes)
        register('finetuned_minilm',neural,probs)
        # Ensembles use the best independently validated lexical/semantic and neural models.
        best_non_neural=max([r for r in rows if r['name']!='finetuned_minilm'],key=lambda r:r['macro_f1'])['name']
        for alpha in [.25,.5,.75]:
            register(f'ensemble_neural_{alpha}',ProbabilityEnsemble([models[best_non_neural],neural],[1-alpha,alpha]),(1-alpha)*probabilities[best_non_neural]+alpha*probs)
        winner=max(rows,key=lambda r:r['macro_f1'])
        pd.DataFrame(rows).sort_values('macro_f1',ascending=False).to_csv(folder/'validation_comparison.csv',index=False)
        write_json(folder/'selection.json',{'winner':winner,'criterion':'validation macro-F1 only','manifest_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'device':DEVICE,'candidates':len(rows),'test_used':False})
        joblib.dump({'model':models[winner['name']],'dataset':name,'training_keys':set(train.key),'selection':winner},folder/'selected.joblib')
        np.save(folder/'selected_validation_probabilities.npy',probabilities[winner['name']])
        print(name,'SELECTED',winner,flush=True)
        del models,probabilities,x,v,lx,lv;gc.collect()

if __name__=='__main__':main()
