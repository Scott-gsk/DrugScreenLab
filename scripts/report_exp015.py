"""Summarize frozen real runs and render a compact scientific report figure."""
from pathlib import Path
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from drug_screen.transfer.metrics_v2 import ndcg_at_k


def main():
    parser = argparse.ArgumentParser();parser.add_argument('--run',default='zhu_transfer_v1');parser.add_argument('--known-only',action='store_true')
    args=parser.parse_args()
    base = Path('data/artifacts/EXP-015')
    p = pd.read_parquet(base/args.run/'outer_oof_predictions.parquet')
    known_count = p.loc[p.source_drug_status.eq('known'),'drug_id'].nunique()
    if args.known_only:p=p[p.source_drug_status.eq('known')]
    cutoff = 3 if args.known_only and 5<=known_count<10 else 5
    minimum = 5 if cutoff==3 else 10
    records = []
    for (algorithm, patient, model), group in p.groupby(['algorithm','patient_id','model_id']):
        if len(group) >= minimum:
            records.append({'algorithm':algorithm,'patient_id':patient,
                'ndcg':ndcg_at_k(group.observed.to_numpy(),group.predicted.to_numpy(),k=cutoff)})
    scores = pd.DataFrame(records)
    if scores.empty:
        raise ValueError('no NDCG5-eligible patient panels; report NDCG3 separately')
    seed_scores = scores.groupby(['algorithm','patient_id']).ndcg.mean().groupby('algorithm').mean()
    compact = []
    for key, value in seed_scores.items():
        name, seed = key.rsplit('_seed_',1)
        compact.append({'algorithm':name,'seed':int(seed),'patient_macro_ndcg':float(value)})
    table = pd.DataFrame(compact)
    comparisons = []
    patient_scores = scores.groupby(['algorithm','patient_id']).ndcg.mean()
    for seed in [20260912,20260913,20260914]:
        candidate = patient_scores.loc[f'M3_seed_{seed}']
        for reference in ['B0','M1','M1_source_scaled']:
            other = patient_scores.loc[f'{reference}_seed_{seed}'].reindex(candidate.index)
            delta = (candidate-other).to_numpy(float)
            draws = np.random.default_rng(seed).choice(delta,size=(2000,len(delta)),replace=True).mean(axis=1)
            comparisons.append({'seed':seed,'candidate':'M3','reference':reference,
                'mean_difference':float(delta.mean()),'patient_bootstrap_95ci':np.quantile(draws,[.025,.975]).tolist()})
    f = json.loads((base/'nature_functional_f0/report.json').read_text())
    summary = {'target_known_drugs':known_count,'target_panel_drugs':p.drug_id.nunique(),
               'ranking_scope':'KNOWN_ONLY_EXPLORATORY' if args.known_only else 'ALL_TARGET_PANEL_EXPLORATORY_INCLUDES_SOURCE_UNSEEN',
               'ndcg_cutoff':cutoff,'legacy_score_field_note':'patient_macro_ndcg field uses ndcg_cutoff; see explicit cutoff',
               'target_patients':p.patient_id.nunique(),
               'target_seed_scores':compact,'exploratory_all_panel_paired_comparisons':comparisons,
               'functional':{key:{'mae':value['mean_gene_mae'],
                  'spearman':value['mean_gene_spearman']} for key,value in f['test'].items()},
               'functional_models':f['n_samples'], 'functional_targets':f['n_targets']}
    reportbase=base/args.run if args.known_only else base
    (reportbase/'weekly_result_summary.json').write_text(json.dumps(summary,indent=2))
    fig, axes = plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    order=['B0','M1','M1_source_scaled','M2','M3']
    means=table.groupby('algorithm').patient_macro_ndcg.mean()
    axes[0].bar(range(5),[means[k] for k in order],color=['#87919c','#487ca6','#689aaa','#c99548','#548b62'])
    for i,name in enumerate(order):
        vals=table[table.algorithm.eq(name)].patient_macro_ndcg
        axes[0].scatter(np.full(len(vals),i),vals,color='black',s=16,zorder=3)
    axes[0].set_xticks(range(5),['Drug mean','PDO only','PDO only\nsource scaling','Source only','Source +\nfine-tuning'])
    axes[0].set_ylabel(f'Patient-macro NDCG@{cutoff}');axes[0].set_ylim(0,1)
    scope='Known source drugs only' if args.known_only else 'Includes unseen drugs'
    axes[0].set_title(f'PDO exploration: {summary["target_patients"]} patients, {summary["target_panel_drugs"]} drugs\n{scope}; dots: 3 target seeds',fontsize=10)
    names=['mean','ridge','mlp','mlp_expression_permuted']
    axes[1].bar(range(4),[summary['functional'][k]['mae'] for k in names],color=['#87919c','#487ca6','#548b62','#c99548'])
    axes[1].set_xticks(range(4),['Gene mean','Ridge','MLP','MLP +\nshuffled expression'])
    axes[1].set_ylabel('Test mean gene MAE (lower is better)')
    axes[1].set_title('Nature dependency prediction: 59 models\nSingle-minLib colorectal subset\n128 training-selected genes',fontsize=10)
    fig.savefig(reportbase/'weekly_results.png',dpi=180)
    fig.savefig(reportbase/'weekly_results.pdf')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
