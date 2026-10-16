"""Separate development/test evaluation; never modify thresholds on test outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from .sources import cached_sources
from .verify import verify_claim, THRESHOLD, MIN_SIMILARITY
ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--split',choices=['development','test'],required=True);args=parser.parse_args()
    path=ROOT/'data/verification'/f'{args.split}.json'
    cases=json.loads(path.read_text());manifest=json.loads((ROOT/'reports/verifier_manifest.json').read_text());allowed={r['sha256'] for r in manifest['source_snapshots']};sources=[d for d in cached_sources() if d['sha256'] in allowed];results=[]
    for case in cases:
        start=time.perf_counter();out=verify_claim(case['claim'],sources)
        result={**case,'predicted':out['status'],'seconds':time.perf_counter()-start,'correction':out['correction'],'evidence':out['evidence'][:3]}
        if out['correction']:
            result['extractive_integrity']=any(out['correction'] in d['text'] for d in sources)
        results.append(result)
        print(case['claim'],case['expected'],'->',out['status'],flush=True)
    y=[r['expected'] for r in results];pred=[r['predicted'] for r in results]
    corrections=[r for r in results if r['correction']]
    labels=['supported_by_sources','refuted_by_sources','insufficient_evidence','conflicting_evidence']
    metrics={'split':args.split,'count':len(cases),'accuracy':accuracy_score(y,pred),'macro_f1':f1_score(y,pred,average='macro',zero_division=0),
      'corrections_proposed':len(corrections),'correct_refutation_precision':sum(r['expected']=='refuted_by_sources' for r in corrections)/len(corrections) if corrections else None,
      'verbatim_source_integrity':sum(r['extractive_integrity'] for r in corrections)/len(corrections) if corrections else None,
      'threshold':THRESHOLD,'minimum_retrieval_similarity':MIN_SIMILARITY,'query_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'model_id':manifest['model_id'],'model_revision':manifest['revision'],'source_hashes':sorted(allowed),
      'classification_report':classification_report(y,pred,zero_division=0,output_dict=True),
      'confusion_labels':labels,'confusion_matrix':confusion_matrix(y,pred,labels=labels).tolist(),
      'limitations':'Small manually authored English science/unknown-claim challenge; not a broad news benchmark. Verbatim integrity measures provenance, not semantic correctness. Pretrained NLI model was not trained here.'}
    folder=ROOT/'reports/verification';folder.mkdir(exist_ok=True)
    (folder/f'{args.split}_predictions.json').write_text(json.dumps(results,indent=2,ensure_ascii=False))
    (folder/f'{args.split}_metrics.json').write_text(json.dumps(metrics,indent=2))
    print(json.dumps({k:v for k,v in metrics.items() if k not in ['classification_report','confusion_matrix']},indent=2))

if __name__=='__main__':main()
