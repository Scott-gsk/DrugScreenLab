import json
import numpy as np
import pandas as pd
import pytest
from drug_screen.transfer.metrics_v2 import evaluate_predictions, ndcg_at_k


def table(n=10):
    return pd.DataFrame([dict(patient_id=f'p{i}',model_id=f'o{i}',drug_id=f'd{j}',
                             observed=j+i/10,predicted=j+i/10,baseline=n-j-i/10)
                         for i in range(8) for j in range(n)])


def test_ndcg_ties_and_scale_invariance():
    y,p=[0,1,2,3,4],[1,1,0,0,0]
    order=[3,1,4,0,2]
    assert ndcg_at_k(y,p,k=3)==pytest.approx(ndcg_at_k([y[i] for i in order],[p[i] for i in order],k=3))
    assert ndcg_at_k(y,p,k=3)==pytest.approx(ndcg_at_k(np.exp(y),p,k=3))


def test_json_null_and_k_not_pooled():
    report=evaluate_predictions(table(),resamples=3)
    assert report['patient_average']['ndcg_at_5']==pytest.approx(1)
    assert report['patient_average']['ndcg_at_3'] is None
    short=evaluate_predictions(table(7),resamples=3)
    assert short['bootstrap']['primary_ndcg_difference'] is None
    assert short['patient_average']['ndcg_at_3']==pytest.approx(1)
    tied=evaluate_predictions(table().assign(observed=1),resamples=3)
    assert tied['counts']['pdos_constant_observed']==8
    json.dumps(tied,allow_nan=False)


def test_multiple_pdos_do_not_reweight_patient_or_inflate_panel():
    t=table()
    extra=t[t.patient_id=='p0'].copy().assign(model_id='other',predicted=lambda x:-x.observed)
    report=evaluate_predictions(pd.concat([t,extra]),resamples=3)
    assert report['n_patients']==8 and report['n_models']==9
    p0=next(p for p in report['patients'] if p['patient_id']=='p0')
    assert p0['spearman']==pytest.approx(0)
    assert report['patient_average']['spearman']==pytest.approx(7/8)
    short=table(3)
    extra=short[short.patient_id=='p0'].assign(model_id='other')
    assert evaluate_predictions(pd.concat([short,extra]),resamples=3)['patient_average']['ndcg_at_3'] is None


def test_bootstrap_deterministic_and_per_drug_recomputed():
    a=evaluate_predictions(table(),seed=17,resamples=20)
    b=evaluate_predictions(table(),seed=17,resamples=20)
    assert a['bootstrap']==b['bootstrap']
    assert a['bootstrap']['per_drug_spearman']['point_estimate']==pytest.approx(1)
    assert a['bootstrap']['per_drug_spearman_difference']['point_estimate']==pytest.approx(2)


@pytest.mark.parametrize('mutate,match',[
    (lambda t:t.drop(columns='patient_id'),'missing required'),
    (lambda t:t.assign(patient_id=None),'patient_id'),
    (lambda t:pd.concat([t,t.iloc[:1]]),'duplicate'),
    (lambda t:t.assign(predicted=np.inf),'finite'),
    (lambda t:t.assign(model_id='all'),'multiple patients'),
    (lambda t:t.assign(fold=np.arange(len(t))%2),'spans folds'),
])
def test_validation(mutate,match):
    with pytest.raises(ValueError,match=match):evaluate_predictions(mutate(table()),resamples=2)
