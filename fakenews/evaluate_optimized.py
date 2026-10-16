"""One follow-up test evaluation after validation selection has been frozen."""
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score,classification_report,confusion_matrix
from .train import ROOT,metrics,write_json


def main():
    for name in ['liar','fakenewsnet']:
        folder=ROOT/'reports/optimization'/name
        artifact=joblib.load(folder/'selected.joblib')
        df=pd.read_csv(ROOT/'reports'/name/'split_manifest.csv');test=df[df.split=='test']
        model=artifact['model'];probs=model.predict_proba(test.text);pred=model.classes_[probs.argmax(1)]
        baseline=joblib.load(ROOT/'models'/f'{name}.joblib')['model'];old=baseline.predict(test.text)
        y=test.label.to_numpy();rng=np.random.default_rng(2026);accuracy_delta=[];f1_delta=[]
        for _ in range(1000):
            i=rng.integers(0,len(y),len(y))
            accuracy_delta.append((y[i]==pred[i]).mean()-(y[i]==old[i]).mean())
            f1_delta.append(f1_score(y[i],pred[i],average='macro',zero_division=0)-f1_score(y[i],old[i],average='macro',zero_division=0))
        result={'dataset':name,'selected_model':artifact['selection']['name'],'selection':'validation macro-F1 only; train-only fitting','test_rows':len(y),**metrics(y,pred),'baseline':metrics(y,old),'paired_accuracy_delta_95pct_ci':np.quantile(accuracy_delta,[.025,.975]).tolist(),'paired_macro_f1_delta_95pct_ci':np.quantile(f1_delta,[.025,.975]).tolist(),'evaluation_note':'Previously reported test split; not a new blind benchmark. Pretraining overlap is not ruled out.','selection_sha256':hashlib.sha256((folder/'selection.json').read_bytes()).hexdigest()}
        write_json(folder/'test_metrics.json',result)
        write_json(folder/'classification_report.json',classification_report(y,pred,output_dict=True,zero_division=0))
        write_json(folder/'source_metrics.json',{source:metrics(test.label[test.source==source],pred[(test.source==source).to_numpy()]) for source in test.source.unique()})
        output=test[['id','source','text','label']].copy();output['prediction']=pred;output['baseline_prediction']=old
        output.to_csv(folder/'test_predictions.csv',index=False);output[output.label!=output.prediction].to_csv(folder/'errors.csv',index=False)
        np.save(folder/'test_probabilities.npy',probs)
        artifact['metrics']=result;joblib.dump(artifact,folder/'selected.joblib')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay
        fig,ax=plt.subplots(figsize=(8,7));ConfusionMatrixDisplay(confusion_matrix(y,pred,labels=model.classes_),display_labels=model.classes_).plot(ax=ax,cmap='Blues',colorbar=False,xticks_rotation=35)
        ax.set_title(f'{name}: validation-selected optimized model');fig.tight_layout();fig.savefig(folder/'confusion_matrix.png',dpi=150);plt.close(fig)
        print(json.dumps(result),flush=True)

if __name__=='__main__':main()
