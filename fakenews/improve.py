"""Validation-only feature/model experiment; preserve original deployed baselines."""
import json
import time
import hashlib
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from .data import clean_text
from .train import ROOT, metrics, write_json


def features(kind):
    word = TfidfVectorizer(preprocessor=clean_text, ngram_range=(1,2), min_df=2, max_df=.98, max_features=80000, sublinear_tf=True)
    if kind == 'word':
        return word
    char = TfidfVectorizer(preprocessor=clean_text, analyzer='char_wb', ngram_range=(3,5), min_df=3, max_features=60000, sublinear_tf=True)
    return FeatureUnion([('word',word),('char',char)], transformer_weights={'word':1.,'char':.6})


def main():
    for dataset in ['liar','fakenewsnet']:
        out=ROOT/'reports'/'improvements'/dataset
        out.mkdir(parents=True,exist_ok=True)
        path=ROOT/'reports'/dataset/'split_manifest.csv'
        df=pd.read_csv(path)
        train=df[df.split=='train'];valid=df[df.split=='valid']
        baseline=joblib.load(ROOT/'models'/f'{dataset}.joblib')['model']
        rows=[{'name':'original_baseline',**metrics(valid.label,baseline.predict(valid.text))}]
        best=baseline;best_name='original_baseline';best_score=rows[0]['macro_f1']
        for kind in ['word','word_char']:
            transform=features(kind)
            x=transform.fit_transform(train.text);v=transform.transform(valid.text)
            for family,c,weight in [('lr',1,None),('lr',4,'balanced'),('svm',.25,'balanced'),('svm',1,None)]:
                name=f'{kind}_{family}_{c}_{weight}'
                start=time.monotonic()
                clf=(LogisticRegression(C=c,class_weight=weight,max_iter=1500,random_state=42) if family=='lr' else LinearSVC(C=c,class_weight=weight,max_iter=5000,random_state=42))
                clf.fit(x,train.label)
                row={'name':name,**metrics(valid.label,clf.predict(v)),'fit_seconds':time.monotonic()-start}
                rows.append(row);print(dataset,name,round(row['accuracy'],4),round(row['macro_f1'],4),flush=True)
                if row['macro_f1']>best_score:
                    best_score=row['macro_f1'];best_name=name
                    best=Pipeline([('features',transform),('classifier',clf)])
        pd.DataFrame(rows).sort_values('macro_f1',ascending=False).to_csv(out/'validation_comparison.csv',index=False)
        write_json(out/'selection.json',{'selected':best_name,'criterion':'validation macro-F1 only','split_manifest_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'candidates':len(rows),'warning':'Original test was already reported; follow-up test comparison is not a new blind benchmark.'})
        joblib.dump({'model':best,'dataset':dataset,'training_keys':set(train.key)},out/'candidate.joblib')
        # All settings are fixed before this follow-up test comparison.
        test=df[df.split=='test'];pred=best.predict(test.text)
        result={'selected':best_name,'baseline':metrics(test.label,baseline.predict(test.text)),'candidate':metrics(test.label,pred),'test_rows':len(test),'evaluation':'Previously reported held-out split; selection used validation only.'}
        write_json(out/'test_comparison.json',result)
        test=test.copy();test['prediction']=pred;test.to_csv(out/'test_predictions.csv',index=False)
        print(json.dumps(result),flush=True)

if __name__=='__main__':main()
