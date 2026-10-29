"""Development-only evidence ranking search; original verifier remains unchanged."""
import json
from sklearn.metrics import accuracy_score,f1_score
from .advanced_models import embed
from .sources import cached_sources
from .verify import retrieve,get_nli
from .train import ROOT,write_json
OUT=ROOT/'reports/optimization/verifier'


def decide(candidates,threshold,limit,ranking):
    selected=sorted(candidates,key=lambda p:p['semantic_similarity'] if ranking=='semantic' else p['similarity'],reverse=True)[:limit]
    support=[p for p in selected if p['nli_scores']['entailment']>=threshold]
    contradict=[p for p in selected if p['nli_scores']['contradiction']>=threshold]
    if support and contradict:return 'conflicting_evidence',None
    if contradict:return 'refuted_by_sources',max(contradict,key=lambda p:p['nli_scores']['contradiction'])['text']
    if support:return 'supported_by_sources',None
    return 'insufficient_evidence',None


def prepare(split):
    cache=OUT/f'{split}_scored.json'
    if cache.exists():return json.loads(cache.read_text())
    cases=json.loads((ROOT/f'data/verification/{split}.json').read_text())
    manifest=json.loads((ROOT/'reports/verifier_manifest.json').read_text());allowed={s['sha256'] for s in manifest['source_snapshots']}
    sources=[s for s in cached_sources() if s['sha256'] in allowed]
    for case in cases:
        candidates=retrieve(case['claim'],sources)
        if candidates:
            vectors=embed([case['claim']]+[p['source']['title']+'. '+p['text'] for p in candidates])
            scores=get_nli().score([(p['source']['title']+'. '+p['text'],case['claim']) for p in candidates])
            for p,v,s in zip(candidates,vectors[1:],scores):p.update(semantic_similarity=float(v@vectors[0]),nli_scores=s)
        case['candidates']=candidates
        print(split,case['claim'],len(candidates),flush=True)
    write_json(cache,cases);return cases


def evaluate(cases,config):
    results=[]
    for c in cases:
        predicted,correction=decide(c['candidates'],**config)
        results.append({'claim':c['claim'],'expected':c['expected'],'predicted':predicted,'correction':correction})
    expected=[r['expected'] for r in results];pred=[r['predicted'] for r in results]
    wrong_corrections=sum(bool(r['correction']) and r['expected']!='refuted_by_sources' for r in results)
    return {'config':config,'count':len(cases),'accuracy':accuracy_score(expected,pred),'macro_f1':f1_score(expected,pred,average='macro',zero_division=0),'wrong_corrections':wrong_corrections,'corrections':sum(bool(r['correction']) for r in results)},results


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    dev=prepare('development');rows=[]
    # Prefer the original configuration on exact ties; require no development false corrections.
    configs=[{'ranking':'lexical','threshold':.92,'limit':16}]+[{'ranking':ranking,'threshold':threshold,'limit':limit} for ranking in ['lexical','semantic'] for threshold in [.85,.92,.97] for limit in [1,3,5,16]]
    for config in configs:
        result,_=evaluate(dev,config);rows.append(result)
    best=max(rows,key=lambda r:(-r['wrong_corrections'],r['macro_f1'],r['accuracy']))
    write_json(OUT/'development_comparison.json',rows)
    write_json(OUT/'selection.json',{'selected':best,'selection':'development only; minimize false corrections then maximize macro-F1; original wins ties','limitation':'Only 12 manually authored development cases; high risk of selection overfitting. Old test is a follow-up diagnostic.'})
    print('SELECTED',best,flush=True)
    test=prepare('test');result,pred=evaluate(test,best['config'])
    write_json(OUT/'test_metrics.json',result);write_json(OUT/'test_predictions.json',pred)
    print('TEST',result,flush=True)

if __name__=='__main__':main()
