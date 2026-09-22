# Route-1 stage-1 Kryeziu baselines DRAFT

- seed=`20260918`; inner val carved 4:1 from train_patient_ids only
- drugs=48 DRAFT; split=`data/splits/kryeziu_patient_outer5fold_DRAFT.json`
- official metrics table=`/mnt/d/Code/DrugScreenLab/docs/reports/stage1_baseline_metrics_DRAFT.csv`
- by-fold=`data/artifacts/route1_stage1_DRAFT/kryeziu_stage1_baseline_metrics_by_fold_DRAFT.csv`
- aggregation=`docs/reports/stage1_metrics_aggregation_DRAFT.md`

## Macro test summary (patient-equal Task B)

| fold   | model                  |   n_test_patients |   ndcg@5 |   spearman_B |   top5_overlap |   spearman_A |   n_drugs_A |
|:-------|:-----------------------|------------------:|---------:|-------------:|---------------:|-------------:|------------:|
| macro  | B0_drug_mean           |              10.2 | 0.908877 |     0.807839 |       0.604364 |    0.0872807 |         2.8 |
| macro  | B1_per_drug_elasticnet |              10.2 | 0.905792 |     0.809652 |       0.617091 |    0.12301   |        43   |
| macro  | B1_per_drug_ridge      |              10.2 | 0.900874 |     0.813492 |       0.601455 |    0.149265  |        45.4 |

## Notes

- B0=drug mean; B1=per-drug Ridge / ElasticNet on train-only HVGs
- Hyperparams chosen on inner val by patient-equal NDCG@5
- Outer test never used for selection; untested drugs dropped (no zero-fill)
- DRAFT: not final model-selection evidence
