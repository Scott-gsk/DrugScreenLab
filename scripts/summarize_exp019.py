"""Finalize compact EXP-019 evidence from a completed frozen run."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np, pandas as pd

ROOT=Path(__file__).resolve().parents[1]
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',default='data/artifacts/EXP-019/prism_frozen_v2');a=p.parse_args();run=ROOT/a.run
    metrics=json.loads((run/'metrics.json').read_text());contrasts=json.loads((run/'contrasts.json').read_text());contract=json.loads((run/'contract.json').read_text())
    original={name:metrics[f'{name}|original|0']['cell_macro'] for name in ('B0','DrugOnly','DualEncoder')}
    permutation={}
    for condition in ('within_cancer_row_permutation','pan_cancer_row_permutation'):
        rows=[]
        for i in range(contract['permutation_repeats']):
            value=metrics[f'DualEncoder|{condition}|{i}']['cell_macro']
            rows.append({k:original['DualEncoder'][k]-value[k] for k in ('spearman','ndcg5','top5_overlap')})
        permutation[condition]={k:{'mean_original_minus_permuted':float(np.mean([r[k] for r in rows])),'q025':float(np.quantile([r[k] for r in rows],.025)),'q975':float(np.quantile([r[k] for r in rows],.975)),'repeats':len(rows)} for k in rows[0]}
    model=pd.read_csv(ROOT/'data/raw/depmap/24q2/Model.csv',usecols=['ModelID','SangerModelID']).dropna()
    gdsc=pd.read_parquet(ROOT/'data/processed/exp014/gdsc_v3/samples.parquet'); prism=pd.read_parquet(ROOT/'data/processed/exp012/prism_extended_primary_samples.parquet')
    source=set(gdsc.sample_id.astype(str));mapped=set(model.loc[model.SangerModelID.astype(str).isin(source),'ModelID'].astype(str)); prism_ids=set(prism.sample_id.astype(str)); overlap=sorted(mapped&prism_ids); unseen=sorted(prism_ids-mapped)
    pd.DataFrame({'cell_line_id':overlap}).to_csv(run/'excluded_gdsc_overlap_cell_lines.csv',index=False)
    pd.DataFrame({'cell_line_id':unseen}).to_csv(run/'strict_unseen_cell_lines.csv',index=False)
    drugs=pd.read_parquet(ROOT/'data/processed/exp012/prism_extended_primary_drugs.parquet');drugs[['drug_id','inchikey','canonical_smiles']].to_csv(run/'drug_intersection.csv',index=False)
    verdict='DOES_NOT_SUPPORT_CROSS_STUDY_CELL_SPECIFIC_GENERALIZATION' if contrasts['DualEncoder_minus_DrugOnly']['ndcg5']['difference']<=0 or contrasts['DualEncoder_minus_B0']['ndcg5']['difference']<0 else 'SUPPORTS_CROSS_STUDY_CELL_SPECIFIC_GENERALIZATION'
    summary={'experiment':'EXP-019','status':'COMPLETE_WITH_IDENTITY_COUNT_CORRECTION','verdict':verdict,'scope':'PRISM cross-study unseen cell-line evaluation; not PDO, unseen-drug, clinical, or independent-expression-platform validation','identity_audit':{'prism_response_cell_lines':len(prism_ids),'excluded_gdsc_overlap_cell_lines':len(overlap),'strict_unseen_cell_lines':len(unseen),'registered_count_was':354,'current_recomputed_count':len(unseen),'count_difference_requires_registry_correction':len(unseen)!=354},'test_pairs':metrics['DualEncoder|original|0']['n_pairs'],'drugs':contract['prism_drugs'],'models':original,'paired_contrasts':contrasts,'expression_counterfactuals':permutation}
    (run/'compact_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    files=[run/'predictions.parquet',run/'metrics.json',run/'contrasts.json',run/'contract.json',run/'compact_summary.json',run/'excluded_gdsc_overlap_cell_lines.csv',run/'strict_unseen_cell_lines.csv',run/'drug_intersection.csv',ROOT/'scripts/run_exp019_prism.py',ROOT/'scripts/summarize_exp019.py']
    hashes={str(x.relative_to(ROOT)):sha(x) for x in files};(run/'run_hashes.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False))
if __name__=='__main__':main()
