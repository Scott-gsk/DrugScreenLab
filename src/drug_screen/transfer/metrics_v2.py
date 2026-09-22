"""Patient-block evaluation; model_id is a biological PDO, never an algorithm."""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from scipy.stats import rankdata


def ndcg_at_k(observed, predicted, *, k):
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise ValueError('k must be a positive integer')
    y, p = np.asarray(observed, float), np.asarray(predicted, float)
    if y.ndim != 1 or y.shape != p.shape or not len(y) or k < 1:
        raise ValueError('aligned non-empty vectors and positive k required')
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('finite values required')
    if len(y) < k or np.ptp(y) == 0:
        return float('nan')
    gains = (rankdata(y, method='average') - 1) / (len(y) - 1)
    discounts = 1 / np.log2(np.arange(2, k + 2))
    ideal = np.sort(gains)[::-1][:k] @ discounts
    total, offset = 0., 0
    for score in np.unique(p)[::-1]:
        mask = p == score
        count = int(mask.sum())
        total += gains[mask].mean() * discounts[offset:min(offset+count, k)].sum()
        offset += count
        if offset >= k: break
    return float(total / ideal)


def _mean(values):
    values = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return float(np.mean(values)) if values else None


def _rho_with_reason(y, p):
    if len(y) < 3:
        return None, 'fewer_than_three_pairs'
    if np.ptp(y) == 0:
        return None, 'constant_observed'
    if np.ptp(p) == 0:
        return None, 'constant_predicted'
    return float(np.corrcoef(rankdata(y), rankdata(p))[0, 1]), None


def _rho(y, p):
    """Compatibility helper returning only the historical numeric value."""
    value, _ = _rho_with_reason(y, p)
    return value


def _paired(y, p, b):
    a, candidate_reason = _rho_with_reason(y, p)
    base, baseline_reason = _rho_with_reason(y, b)
    return {
        'spearman': a,
        'baseline_spearman': base,
        'spearman_difference': _difference(a, base),
        'candidate_spearman_undefined_reason': candidate_reason,
        'baseline_spearman_undefined_reason': baseline_reason,
    }


def _difference(a, b):
    return None if a is None or b is None else a-b


def _pdo_metrics(table):
    rows = []
    for (patient, model), group in table.groupby(['patient_id', 'model_id'], sort=True):
        y, p, b = [group[key].to_numpy() for key in ('observed', 'predicted', 'baseline')]
        row = {'patient_id': patient, 'model_id': model, 'n_drugs': len(y), **_paired(y,p,b)}
        # Preserve the existing field while exposing a separate reason for
        # candidate and baseline correlations below.
        row['undefined_reason'] = row['candidate_spearman_undefined_reason']
        for k, eligible in [(5, len(y) >= 10), (3, 5 <= len(y) < 10)]:
            a = _mean([ndcg_at_k(y,p,k=k)]) if eligible else None
            base = _mean([ndcg_at_k(y,b,k=k)]) if eligible else None
            row.update({f'ndcg_at_{k}': a, f'baseline_ndcg_at_{k}': base,
                        f'ndcg_at_{k}_difference': _difference(a,base)})
        rows.append(row)
    return rows


_METRICS = ['spearman','baseline_spearman','spearman_difference',
            'ndcg_at_5','baseline_ndcg_at_5','ndcg_at_5_difference',
            'ndcg_at_3','baseline_ndcg_at_3','ndcg_at_3_difference']


def _drug_metrics(matrices, drugs):
    rows = []
    for j, drug in enumerate(drugs):
        valid = np.logical_and.reduce([np.isfinite(m[:,j]) for m in matrices])
        y,p,b = [m[valid,j] for m in matrices]
        rows.append({'drug_id': drug, 'n_patients': int(valid.sum()), **_paired(y,p,b)})
    return rows


def _interval(point, values, seed, resamples):
    finite = [v for v in values if v is not None]
    if point is None or len(finite) < 2: return None
    low, high = np.quantile(finite,[.025,.975])
    return {'point_estimate':point,'low':float(low),'high':float(high),'seed':seed,
            'resamples':resamples,'valid_resamples':len(finite),'undefined_resamples':resamples-len(finite)}


def evaluate_predictions(predictions, *, seed=0, resamples=2000):
    """One algorithm's predictions; PDO average then patient average, separate K.

    Input columns: patient_id, model_id (biological), drug_id, observed,
    predicted, baseline; optional fold must be unique per patient. Undefined
    values are JSON null. Repeated bootstrap patients remain separate draws.
    """
    required = {'patient_id','model_id','drug_id','observed','predicted','baseline'}
    if not isinstance(predictions,pd.DataFrame) or not required <= set(predictions.columns):
        raise ValueError('missing required prediction columns')
    if isinstance(resamples, bool) or not isinstance(resamples, (int, np.integer)) or resamples < 2:
        raise ValueError('resamples must be an integer >=2')
    if predictions.empty: raise ValueError('non-empty table required')
    t = predictions.copy()
    for col in ['patient_id','model_id','drug_id']:
        if not t[col].map(lambda v:isinstance(v,str) and bool(v.strip())).all():
            raise ValueError(f'{col} missing identity')
        t[col] = t[col].str.strip()
    if t.duplicated(['patient_id','model_id','drug_id']).any(): raise ValueError('duplicate biological model-drug rows')
    if t.groupby('model_id').patient_id.nunique().gt(1).any(): raise ValueError('model_id assigned to multiple patients')
    if 'fold' in t and (t.fold.isna().any() or t.groupby('patient_id').fold.nunique().gt(1).any()):
        raise ValueError('patient spans folds')
    for col in ['observed','predicted','baseline']:
        t[col] = pd.to_numeric(t[col],errors='raise').astype(float)
        if not np.isfinite(t[col]).all(): raise ValueError(f'{col} must be finite')
    pdos = _pdo_metrics(t)
    patients = []
    ids = sorted(t.patient_id.unique())
    for patient in ids:
        records = [r for r in pdos if r['patient_id']==patient]
        patients.append({'patient_id':patient,'n_models':len(records),
                         **{key:_mean(r[key] for r in records) for key in _METRICS}})
    average = {key:_mean(r[key] for r in patients) for key in _METRICS}
    grouped = t.groupby(['patient_id','drug_id'])[['observed','predicted','baseline']].mean()
    drugs = sorted(t.drug_id.unique())
    matrices = [grouped[key].unstack('drug_id').reindex(index=ids,columns=drugs).to_numpy()
                for key in ['observed','predicted','baseline']]
    drug_rows = _drug_metrics(matrices,drugs)
    drug_avg = {key:_mean(r[key] for r in drug_rows) for key in ['spearman','baseline_spearman','spearman_difference']}
    fields = ['ndcg_at_5_difference','ndcg_at_3_difference','spearman_difference']
    values = {key:[] for key in fields+['per_drug_spearman','per_drug_spearman_difference']}
    rng = np.random.default_rng(seed)
    if len(ids)>=2:
        for _ in range(resamples):
            draw = rng.integers(0,len(ids),len(ids))
            for key in fields: values[key].append(_mean(patients[i][key] for i in draw))
            rows = _drug_metrics([m[draw] for m in matrices],drugs)
            values['per_drug_spearman'].append(_mean(r['spearman'] for r in rows))
            values['per_drug_spearman_difference'].append(_mean(r['spearman_difference'] for r in rows))
    points = average | {'per_drug_spearman':drug_avg['spearman'], 'per_drug_spearman_difference':drug_avg['spearman_difference']}
    intervals = {key:_interval(points[key],v,seed,resamples) for key,v in values.items()}
    intervals['primary_ndcg_difference'] = intervals['ndcg_at_5_difference']
    return {'protocol_version':'pdo-transfer-v2','n_rows':len(t),'n_patients':len(ids),
            'n_models':len(pdos),'n_drugs':len(drugs),'patient_average':average,
            'per_drug_average':drug_avg,'bootstrap':intervals,'patients':patients,'pdos':pdos,'drugs':drug_rows,
            'counts':{'patients_ndcg_at_5_eligible':sum(r['ndcg_at_5'] is not None for r in patients),
                      'patients_ndcg_at_3_eligible':sum(r['ndcg_at_3'] is not None for r in patients),
                      'pdos_constant_observed':sum(r['undefined_reason']=='constant_observed' for r in pdos),
                      'pdos_spearman_undefined':sum(r['spearman'] is None for r in pdos),
                      'per_drug_spearman_undefined':sum(r['spearman'] is None for r in drug_rows)}}
