"""Validation-only decision thresholds and transparent coverage evaluation."""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
from .train import ROOT,write_json
OUT=ROOT/'reports/reliability'


def binary_metrics(y,p,threshold):
    predicted=p>=threshold
    return {'accuracy':accuracy_score(y,predicted),'macro_f1':f1_score(y,predicted,average='macro',zero_division=0),'fake_precision':precision_score(y,predicted,zero_division=0),'fake_recall':recall_score(y,predicted,zero_division=0),'false_negatives':int(((y==1)&(~predicted)).sum()),'false_positives':int(((y==0)&predicted).sum()),'threshold':float(threshold)}


def main():
    import joblib
    OUT.mkdir(exist_ok=True)
    folder=ROOT/'reports/optimization/fakenewsnet'
    classes=joblib.load(folder/'selected.joblib')['model'].classes_
    fake=list(classes).index('fake')
    frame=pd.read_csv(ROOT/'reports/fakenewsnet/split_manifest.csv');valid=frame[frame.split=='valid']
    p=np.load(folder/'selected_validation_probabilities.npy')[:,fake];y=(valid.label=='fake').to_numpy()
    rows=[binary_metrics(y,p,t) for t in np.linspace(.05,.95,181)]
    balanced=max(rows,key=lambda r:(r['macro_f1'],r['accuracy'],-abs(r['threshold']-.5)))
    eligible=[r for r in rows if r['fake_recall']>=.80]
    recall=max(eligible,key=lambda r:(r['fake_precision'],r['macro_f1']))
    config={'balanced':balanced,'high_recall':recall,'selection':'Validation only; balanced maximizes macro-F1; high_recall achieves validation fake recall >=0.80 then maximizes fake precision.','classes':classes.tolist()}
    write_json(OUT/'decision_modes.json',config)
    pd.DataFrame(rows).to_csv(OUT/'validation_thresholds.csv',index=False)
    # Selection frozen before reading test probabilities/labels.
    test=frame[frame.split=='test'];y=(test.label=='fake').to_numpy();p=np.load(folder/'test_probabilities.npy')[:,fake]
    results={mode:binary_metrics(y,p,row['threshold']) for mode,row in [('previous_argmax',{'threshold':.5}),('balanced',balanced),('high_recall',recall)]}
    write_json(OUT/'decision_modes_test.json',results)
    print(json.dumps(results,indent=2),flush=True)
    # Confidence-coverage curves never replace whole-test accuracy.
    curves={}
    for name in ['liar','fakenewsnet']:
        frame=pd.read_csv(ROOT/'reports'/name/'split_manifest.csv');test=frame[frame.split=='test']
        artifact=joblib.load(ROOT/'reports/optimization'/name/'selected.joblib');labels=artifact['model'].classes_
        probabilities=np.load(ROOT/'reports/optimization'/name/'test_probabilities.npy');pred=labels[probabilities.argmax(1)];conf=probabilities.max(1)
        curves[name]=[{'minimum_raw_model_score':threshold,'answered':int((conf>=threshold).sum()),'total':len(test),'coverage':float((conf>=threshold).mean()),'accuracy_on_answered':float((pred[conf>=threshold]==test.label.to_numpy()[conf>=threshold]).mean()) if (conf>=threshold).any() else None} for threshold in [.0,.5,.6,.7,.8,.9]]
    write_json(OUT/'coverage_curves.json',{'caveat':'Raw model scores are not calibrated truth probabilities. These are descriptive test curves, not test-selected deployment thresholds.','results':curves})

if __name__=='__main__':main()
