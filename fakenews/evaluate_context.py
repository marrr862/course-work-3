"""Select context verifier on old development set; report old test as follow-up only."""
import json
from sklearn.metrics import accuracy_score,f1_score
from .context_verify import retrieve,decide
from .verify import get_nli
from .train import ROOT,write_json
OUT=ROOT/'reports/reliability/context'


def collect(split):
    docs=[json.loads(p.read_text()) for p in (ROOT/'data/source_cache_v2').glob('*.json')]
    cases=json.loads((ROOT/f'data/verification/{split}.json').read_text())
    for c in cases:
        ps=retrieve(c['claim'],docs)
        scores=get_nli().score([(p['source']['title']+'. '+p['text'],c['claim']) for p in ps])
        for p,s in zip(ps,scores):p['nli_scores']=s
        c['candidates']=ps
    return cases


def evaluate(cases,config):
    outputs=[{**decide(c['claim'],c['candidates'],**config),'expected':c['expected']} for c in cases]
    y=[c['expected'] for c in cases];p=[o['status'] for o in outputs]
    return {'accuracy':accuracy_score(y,p),'macro_f1':f1_score(y,p,average='macro',zero_division=0),'false_corrections':sum(bool(o['correction']) and o['expected']!='refuted_by_sources' for o in outputs),'count':len(cases),'configuration':config},outputs


def main():
    OUT.mkdir(exist_ok=True,parents=True);dev=collect('development');comparisons=[]
    for threshold in [.92,.97]:
        for similarity in [.35,.45,.55]:
            for limit in [1,3,5]:
                result,_=evaluate(dev,{'threshold':threshold,'minimum_similarity':similarity,'limit':limit});comparisons.append(result)
    best=max(comparisons,key=lambda r:(-r['false_corrections'],r['macro_f1'],r['accuracy']))
    write_json(OUT/'selection.json',best);write_json(OUT/'development_comparison.json',comparisons)
    print('SELECTED',best,flush=True)
    test=collect('test');metrics,pred=evaluate(test,best['configuration'])
    metrics['caveat']='Repeated 24-example science challenge, not independent evidence of generalization. Sources refreshed with improved extraction; not a controlled same-source ablation.'
    write_json(OUT/'test_metrics.json',metrics);write_json(OUT/'test_predictions.json',pred)
    print('TEST',metrics,flush=True)

if __name__=='__main__':main()
