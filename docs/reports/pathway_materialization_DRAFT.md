# Kryeziu Hallmark 通路物化报告（DRAFT）

- 时间：2026-09-18 17:26 CST
- 状态：`DRAFT_MATERIALIZED`（未冻结）
- 矩阵：`data/processed/identity/pathway/kryeziu_hallmark_ssgsea_DRAFT.parquet` shape=(92, 50)
- 合同：`data/processed/identity/pathway/pathway_contract_DRAFT.json`
- 特征顺序：`data/processed/identity/pathway/kryeziu_hallmark_feature_order_DRAFT.json`
- GMT sha256：`8ff1f036f0988c99…`
- 表达源：`data/processed/transfer/mcrc_v3/indexed.npz`
- 平均基因命中率：0.9990（min=0.9900）
- 评分：gseapy ssGSEA `sample_norm_method=rank`，seed=20260918
- 说明：单样本秩评分无跨样本可训练参数；建模仍须按外层划分使用核实患者子集，且不得用 test 拟合后续缩放。

## 切换特征验收

药敏/评价切换 `feat=pathway_hallmark_ssgsea` 时核对：
1. 读取 parquet，`sample_id` 与 mcrc_v3 samples 对齐
2. 列顺序 == feature_order_contract_id 列表
3. 仅用核实子集样本（与 `kryeziu_patient_outer5fold_DRAFT` 一致）
