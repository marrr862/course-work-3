"""Activate validation-selected evaluated artifacts, retaining original baselines."""
import json
import joblib
from .train import ROOT
if __name__=='__main__':
    registry={}
    for name in ['liar','fakenewsnet']:
        path=ROOT/'reports/optimization'/name/'selected.joblib'
        artifact=joblib.load(path)
        if 'metrics' not in artifact:raise RuntimeError('Evaluate selected models before activation.')
        artifact['vocabulary_gate']=joblib.load(ROOT/'models'/f'{name}.joblib')['model'].named_steps['tfidf']
        joblib.dump(artifact,path)
        registry[name]=str(path.relative_to(ROOT))
    destination=ROOT/'models/active_models.json';temporary=destination.with_suffix('.tmp')
    temporary.write_text(json.dumps(registry,indent=2));temporary.replace(destination)
    print('Activated validation-selected models. Restart the local server to load them.')
