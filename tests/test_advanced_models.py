"""Regression checks for class alignment and portable sentence representations."""
import joblib
import numpy as np
from fakenews.advanced_models import ProbabilityEnsemble,embed

class FixedProbabilities:
    def __init__(self,classes,probabilities):
        self.classes_=np.asarray(classes);self.probabilities=np.asarray(probabilities)
    def predict_proba(self,texts):return np.tile(self.probabilities,(len(texts),1))


def test_ensemble_aligns_reversed_class_orders(tmp_path):
    a=FixedProbabilities(['fake','real'],[.8,.2]);b=FixedProbabilities(['real','fake'],[.1,.9])
    model=ProbabilityEnsemble([a,b],[.5,.5])
    path=tmp_path/'model.joblib';joblib.dump(model,path);loaded=joblib.load(path)
    assert loaded.predict(['A claim'])[0]=='fake'
    np.testing.assert_allclose(loaded.predict_proba(['A claim']),[[.85,.15]])


def test_embeddings_invariant_to_padding_and_batch_neighbors():
    sentence='A short factual statement.'
    single=embed([sentence])
    mixed=embed([sentence,'This is a substantially longer sentence about an unrelated research project and its evaluation.'])
    np.testing.assert_allclose(single[0],mixed[0],atol=2e-5)
    np.testing.assert_allclose(np.linalg.norm(mixed,axis=1),[1,1],atol=1e-5)
    assert np.isfinite(mixed).all()


def test_active_model_metrics_match_saved_predictions():
    import json
    import pandas as pd
    from fakenews.app import load_model
    from fakenews.advanced_models import ROOT
    for dataset in ['liar','fakenewsnet']:
        artifact=load_model(dataset)
        expected=json.loads((ROOT/'reports/optimization'/dataset/'test_metrics.json').read_text())
        predictions=pd.read_csv(ROOT/'reports/optimization'/dataset/'test_predictions.csv')
        assert artifact['metrics']['accuracy']==expected['accuracy']
        np.testing.assert_allclose((predictions.label==predictions.prediction).mean(),expected['accuracy'])
        np.testing.assert_array_equal(artifact['model'].predict(predictions.text.iloc[:3]),predictions.prediction.iloc[:3].to_numpy())
