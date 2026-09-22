# 路线一 · 阶段1 基线（本机 mcrc_v3，DRAFT）

- 数据：`data/processed/transfer/mcrc_v3`（本机权威 processed，48 药量级）
- 划分：`data/splits/kryeziu_patient_outer5fold_DRAFT.json`（核实患者；未冻结）
- 规模：患者 51；药 48；pair 3144；表达 [92, 19421]

## 汇总（折间 mean±std）

| 模型 | NDCG@5 | Spearman |
|---|---|---|
| `drug_train_mean` | 0.9050 ± 0.0077 | 0.7857 ± 0.0097 |
| `per_drug_elasticnet_expression` | 0.8943 ± 0.0092 | 0.7679 ± 0.0181 |
| `per_drug_ridge_expression` | 0.8948 ± 0.0106 | 0.7813 ± 0.0172 |

## 方案位置

对应实施方案 **路线一 · 阶段1** 本机完整外 5 折实表。
公开 Mendeley 24 药镜像五折对照见：`docs/reports/route1_stage1_baselines_mendeley_DRAFT.md`。

JSON：`docs/reports/route1_stage1_baselines_mcrc_v3_DRAFT.metrics.json`
