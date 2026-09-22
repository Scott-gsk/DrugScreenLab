# 路线二：删除对照 + Morgan 结构指纹条件头 DRAFT

- 时间：2026-09-18T17:27:13+08:00
- GPU：0 / NVIDIA RTX A5000
- 产物：`artifacts/experiments/route2_lincs_drugcond_ablation_fp_DRAFT/metrics.json`

## pair_random（药可见，有 Morgan 覆盖）

| name | mean cosine | mean pearson |
|---|---:|---:|
| null_zero | 0.0000 | 0.0000 |
| global_mean_delta | 0.0756 | 0.0766 |
| ctrl_only_linear | 0.6717 | 0.6720 |
| drug_id_emb | 0.6607 | 0.6613 |
| drug_id_emb_shuffle_train_labels | 0.6498 | 0.6507 |
| drug_id_emb_zero_at_eval | 0.6639 | 0.6646 |
| morgan_fp_mlp | 0.6199 | 0.6210 |
| morgan_fp_zero_at_eval | 0.6457 | 0.6482 |

## compound_holdout + Morgan（冷药仍用结构指纹）

| name | mean cosine | mean pearson |
|---|---:|---:|
| null_zero | 0.0000 | 0.0000 |
| global_mean_delta | 0.0766 | 0.0774 |
| ctrl_only_linear | 0.6747 | 0.6751 |
| morgan_fp_mlp | 0.5550 | 0.5574 |

耗时 136.95 s。
