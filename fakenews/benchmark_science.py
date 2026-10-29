"""Tune on SciFact train subset, freeze, evaluate official dev once as project holdout."""
import argparse
import hashlib
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,classification_report
from .science import ScientificIndex,get_index,BatchNLI,decide,LABELS,INDEX
from .train import ROOT,write_json
OUT=ROOT/'reports/reliability/scifact'


def gold(case):
    labels={e['label'] for values in case['evidence'].values() for e in values}
    if len(labels)>1:return 'conflicting_evidence'
    return {'SUPPORT':LABELS[0],'CONTRADICT':LABELS[1]}.get(next(iter(labels),None),LABELS[2])


def score_cases(cases,weights,filename,model_path="models/nli_deberta"):
    path=OUT/filename
    if path.exists():return json.loads(path.read_text())
    index=get_index();scorer=BatchNLI(device='cpu',model_path=model_path);rows=[]
    for n,case in enumerate(cases):
        similarities=index.scores(case['claim']);retrieved={str(w):index.retrieve(case['claim'],weight=w,scores=similarities) for w in weights}
        unique={p['doc_id']:p for passages in retrieved.values() for p in passages}
        values=list(unique.values());scores=scorer.score([(p['title']+'. '+p['text'],case['claim']) for p in values])
        for p,s in zip(values,scores):p['nli_scores']=s
        row={'id':case['id'],'claim':case['claim'],'expected':gold(case),'gold_doc_ids':list(case['evidence']),'retrieved':{w:[unique[p['doc_id']] for p in ps] for w,ps in retrieved.items()}}
        rows.append(row)
        if n%10==0:print(filename,n+1,'/',len(cases),flush=True)
    write_json(path,rows);return rows


def evaluate(rows,config):
    results=[];hits=[]
    for row in rows:
        ps=row['retrieved'][str(config['weight'])]
        status,evidence=decide(ps,**config)
        if row['gold_doc_ids']:hits.append(bool(set(row['gold_doc_ids'])&{p['doc_id'] for p in ps[:config['limit']]}))
        results.append({'id':row['id'],'claim':row['claim'],'expected':row['expected'],'predicted':status,'evidence_doc_id':evidence['doc_id'] if evidence else None,'evidence_text':evidence['text'] if evidence else None})
    y=[r['expected'] for r in results];p=[r['predicted'] for r in results];decisive=[r for r in results if r['predicted'] in LABELS[:2]];refuted=[r for r in results if r['predicted']==LABELS[1]]
    return {'count':len(rows),'accuracy':accuracy_score(y,p),'macro_f1_three_gold_classes':f1_score(y,p,labels=LABELS,average='macro',zero_division=0),'decisive_count':len(decisive),'coverage':len(decisive)/len(rows),'accuracy_on_decisive':sum(r['predicted']==r['expected'] for r in decisive)/len(decisive) if decisive else None,'refutation_precision':sum(r['expected']==LABELS[1] for r in refuted)/len(refuted) if refuted else None,'evidence_document_recall_at_k':float(np.mean(hits)) if hits else None,'retrieval_evaluable_claims':len(hits),'configuration':config,'classification_report':classification_report(y,p,zero_division=0,output_dict=True)},results


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['tune','evaluate'],required=True);parser.add_argument('--model-path',default='models/nli_deberta');args=parser.parse_args()
    OUT.mkdir(exist_ok=True,parents=True)
    if not INDEX.exists():
        import torch
        print('BUILDING SCIENTIFIC INDEX',flush=True)
        ScientificIndex.build(device='mps' if torch.backends.mps.is_available() else 'cpu')
    if args.stage=='tune':
        cases=[json.loads(s) for s in (ROOT/'data/scifact/claims_train.jsonl').read_text().splitlines()]
        _,selected=train_test_split(cases,test_size=150,stratify=[gold(c) for c in cases],random_state=2026)
        write_json(OUT/'tuning_ids.json',[c['id'] for c in selected])
        suffix='_adapted' if args.model_path!='models/nli_deberta' else ''
        rows=score_cases(selected,[0.,.5,1.],f'tuning_scored{suffix}.json',args.model_path);comparisons=[]
        for weight in [0.,.5,1.]:
            for limit in [1,3]:
                for threshold in [.8,.9,.95]:
                    for similarity in [.3,.45]:
                        config={'weight':weight,'limit':limit,'threshold':threshold,'minimum_similarity':similarity}
                        metrics,_=evaluate(rows,config);comparisons.append(metrics)
        winner=max(comparisons,key=lambda m:(m['macro_f1_three_gold_classes'],m['accuracy']))
        write_json(OUT/'tuning_comparison.json',comparisons)
        write_json(OUT/'selection.json',{'selected':winner,'criterion':'150 official train claims; maximize macro-F1 over three gold classes, then accuracy','official_dev_labels_used':False,'model_path':args.model_path,'science_code_sha256':hashlib.sha256((ROOT/'fakenews/science.py').read_bytes()).hexdigest(),'limitations':'New-to-project public benchmark; pretraining overlap unknown. Simplified claim-level scoring, not the official SciFact evidence-set score.'})
        print('FROZEN SELECTION',json.dumps(winner),flush=True)
    else:
        selection=json.loads((OUT/'selection.json').read_text());assert selection['science_code_sha256']==hashlib.sha256((ROOT/'fakenews/science.py').read_bytes()).hexdigest()
        config=selection['selected']['configuration']
        cases=[json.loads(s) for s in (ROOT/'data/scifact/claims_dev.jsonl').read_text().splitlines()]
        tuning_ids=set(json.loads((OUT/'tuning_ids.json').read_text()));assert not tuning_ids&{c['id'] for c in cases}
        rows=score_cases(cases,[config['weight']],'holdout_scored.json',selection.get('model_path','models/nli_deberta'));metrics,predictions=evaluate(rows,config)
        metrics['split']='official SciFact dev, reserved as project final holdout';metrics['selection_sha256']=hashlib.sha256((OUT/'selection.json').read_bytes()).hexdigest()
        write_json(OUT/'holdout_metrics.json',metrics);write_json(OUT/'holdout_predictions.json',predictions)
        print('FINAL HOLDOUT',json.dumps(metrics),flush=True)

if __name__=='__main__':main()
