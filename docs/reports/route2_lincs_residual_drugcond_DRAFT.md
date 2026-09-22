# 路线二：残差药条件（Δ−ctrl_only）DRAFT

- 时间：2026-09-18T17:47:46+08:00
- GPU：1
- metrics：`artifacts/experiments/route2_lincs_residual_drugcond_DRAFT/metrics.json`

| name | mean cosine | mean pearson |
|---|---:|---:|
| null_zero | 0.0000 | 0.0000 |
| global_mean_delta | 0.0776 | 0.0783 |
| ctrl_only | 0.6722 | 0.6726 |
| ctrl_plus_resid_morgan | 0.6181 | 0.6197 |
| ctrl_plus_resid_morgan_zero_at_eval | 0.6508 | 0.6524 |
| ctrl_plus_resid_trained_on_zero_feat | 0.6771 | 0.6774 |
| ctrl_plus_resid_shuffled_morgan | 0.5922 | 0.5939 |
| resid_morgan_vs_true_resid | 0.1239 | 0.1238 |
| zero_vs_true_resid | 0.0000 | 0.0000 |

耗时 50.13 s。
