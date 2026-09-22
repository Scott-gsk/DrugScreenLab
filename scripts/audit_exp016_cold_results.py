"""Audit frozen cold-cell predictions using the existing rank-gain contract."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from drug_screen.transfer.metrics_v2 import ndcg_at_k


def main():
    root = Path('data/artifacts/EXP-016/cold_cell_v1')
    ids = {part: set(pd.read_csv(root/f'{part}_sample_ids.csv').iloc[:,0].astype(str))
           for part in ['train','validation','test']}
    assert not ids['train'] & ids['validation']
    assert not ids['train'] & ids['test']
    assert not ids['validation'] & ids['test']
    p = pd.read_parquet(root/'oof_predictions.parquet')
    assert set(p.sample_id) == ids['test']
    report = {'split_counts':{k:len(v) for k,v in ids.items()},'split_disjoint':True,
              'models':{},'paired_bootstrap':{},'scope':'Same-study unseen-cell evaluation; not unseen-drug or external-study validation.'}
    tables = {}
    for model, frame in p.groupby('model'):
        assert not frame.duplicated(['sample_id','drug_id']).any()
        rows=[]
        for sid,g in frame.groupby('sample_id'):
            y,z=g.observed.to_numpy(),g.predicted.to_numpy()
            rows.append({'sample_id':sid,'mae':np.abs(y-z).mean(),
                'spearman':float(spearmanr(y,z).statistic) if np.ptp(y)>0 and np.ptp(z)>0 else np.nan,
                'ndcg5':ndcg_at_k(y,z,k=5) if len(g)>=10 else np.nan})
        t=pd.DataFrame(rows).set_index('sample_id').sort_index();tables[model]=t
        per_drug=[]
        for _,g in frame.groupby('drug_id'):
            if len(g)>=3 and g.observed.nunique()>1 and g.predicted.nunique()>1:
                per_drug.append(float(spearmanr(g.observed,g.predicted).statistic))
        report['models'][model]={'cell_macro':t.mean().to_dict(),
            'drug_macro_spearman':float(np.mean(per_drug)) if per_drug else None,
            'evaluable_drug_correlations':len(per_drug),'known_drugs':frame.loc[frame.source_drug_status.eq('known'),'drug_id'].nunique(),
            'unknown_rows':int(frame.source_drug_status.eq('unknown').sum())}
    for ref in ['B0','Ridge']:
        diff=tables['DualEncoder']-tables[ref].reindex(tables['DualEncoder'].index)
        rng=np.random.default_rng(20260913);draws=rng.integers(0,len(diff),(2000,len(diff)))
        report['paired_bootstrap'][f'DualEncoder_minus_{ref}']={}
        for col in diff:
            values=diff[col].to_numpy();boot=np.nanmean(values[draws],axis=1)
            report['paired_bootstrap'][f'DualEncoder_minus_{ref}'][col]={
                'difference':float(np.nanmean(values)), 'ci95':np.quantile(boot,[.025,.975]).tolist()}
    (root/'contract_metrics_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
