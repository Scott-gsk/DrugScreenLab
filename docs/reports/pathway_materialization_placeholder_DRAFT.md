# 通路活性物化 — 路径与版本占位（DRAFT）

- 写入：2026-09-18 17:23 CST
- 合同接口：`src/drug_screen/data/expression_contracts.py` → `PathwayActivityContract`
- 状态：**占位 / 未物化**；不得把探索性 `dual_mlp_pathway_pca` 结果当作本冻结合同

## 目标产物（拟定路径）

| 产物 | 拟定路径 | 说明 |
|---|---|---|
| 通路矩阵（样本×Hallmark） | `data/processed/identity/pathway/kryeziu_hallmark_ssgsea_DRAFT.parquet` | 行= sample_id，列= pathway_id |
| 特征顺序合同 | `data/processed/identity/pathway/kryeziu_hallmark_feature_order_DRAFT.json` | 有序 pathway_id 列表 + checksum |
| 基因集合版本钉 | `data/registry` 条目（待建）`msigdb_hallmark_2023_2_hs` | 指向 GMT |
| 覆盖/缺失报告 | `docs/reports/pathway_materialization_DRAFT.md` | 物化后填写基因命中率 |

## 版本占位（冻结前填实）

```json
{
  "contract_name": "pathway_activity",
  "contract_version": "draft-0.1",
  "status": "PLACEHOLDER_NOT_MATERIALIZED",
  "gene_set_collection": "MSigDB_Hallmark_v2023.2_Hs",
  "gene_set_gmt_path": "data/raw/exp013/msigdb/h.all.v2023.2.Hs.symbols.gmt",
  "scoring_method": "ssgsea_rank",
  "fit_rule": "any rank/scale fit uses outer-train patient sample_ids only; never test",
  "expression_source": "data/processed/transfer/mcrc_v3/indexed.npz",
  "cohort": "Kryeziu verified subset aligned to kryeziu_patient_outer5fold_DRAFT",
  "feature_order_contract_id": "unset",
  "output_matrix_path": "data/processed/identity/pathway/kryeziu_hallmark_ssgsea_DRAFT.parquet",
  "not_allowed": [
    "treat dual_mlp_pathway_pca exploratory run as frozen contract",
    "fit pathway ranks on full cohort including test patients",
    "silently switch to Reactome as default"
  ]
}
```

占位 JSON 副本：`data/processed/identity/pathway/pathway_contract_PLACEHOLDER_DRAFT.json`

## 下一步（物化时）

1. 确认 GMT 路径与 checksum，登记 registry
2. 用核实样本表达打分，**按外层 train 拟合**（或仅确定性 rank 且无跨折泄漏）
3. 写出 parquet + feature_order JSON，更新 `expression_contracts_gap_list_DRAFT.md` 勾选

## 阻塞

- **无阻塞**当前药敏/扰动并行实验。
- 阻塞「通路合同冻结」：矩阵尚未生成；仅有本占位。
