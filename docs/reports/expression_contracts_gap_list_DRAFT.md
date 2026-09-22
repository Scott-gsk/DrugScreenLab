# 三套表达输入合同 — 缺口列表（DRAFT）

- 本文：2026-09-18 17:19 CST
- 接口骨架已有：`src/drug_screen/data/expression_contracts.py`、`identity_contracts.py`
- 原则：raw 只读；预处理只在训练集拟合；未测响应不补零；不强制所有任务压成 978

## 1. 基因表达合同（GeneExpression）

| 项 | 现状 | 缺口 |
|---|---|---|
| Protocol/Stub | 已有 `GeneExpressionContract` / `StubGeneExpressionLoader` | 缺 Kryeziu / DepMap / GDSC 实装 loader |
| Kryeziu 表达 | `data/processed/transfer/mcrc_v3/indexed.npz` + manifest（n_genes 在 manifest） | 缺「合同版本 ID + gene_order_contract_id + checksum」注册到 `data/registry` |
| 基因空间 | 全基因组 log2(CPM+1) 与训练常用 top-2000 并存 | **缺口：冻结时必须声明实际 feature_space**（genome_wide vs topk），topk 筛选只能 fit 在 train |
| DepMap/源域 | `processed/depmap/24q2_rnaseq_exact978` 存在 | 与 Kryeziu 基因对齐策略未写成合同；exact978 不得默认套到 PDO 药敏主任务 |
| 标准化 | 合同声明 train-only | 缺可调用的 `fit/transform` 实现与单测 |

## 2. 通路活性合同（PathwayActivity）

| 项 | 现状 | 缺口 |
|---|---|---|
| Protocol/Stub | 已有 `PathwayActivityContract`（默认 Hallmark + ssGSEA-rank） | **无物化通路矩阵**、无 Hallmark GMT→样本打分流水线落盘 |
| MSigDB | raw 有 hallmark gmt（EXP-013 路径） | 未登记为本轮合同输入版本 |
| 阶段1使用 | `dual_mlp_pathway_pca` 等实验已跑 | 属探索；**不能当作冻结通路合同**；需独立 processed 产物 + feature_order_contract_id |
| fit 规则 | docstring 要求 fit_sample_ids | 缺实现与泄漏测试 |

## 3. 扰动基因合同（PerturbationGene）

| 项 | 现状 | 缺口 |
|---|---|---|
| Protocol/Stub | `PerturbationGeneContract` + `measured_perturbation_loader_DRAFT` | Δ 向量全量物化未做；gctx 全矩阵未载入（有意） |
| LINCS | exact978 cache / L3–L5 资产在盘；线性基线 DRAFT 已出 | **L3-978 vs L5 主合同未选定并冻结**；Level 混用禁令需进 registry |
| ChemPert | 元数据可读；raw 在 acquisition 目录 | 常用矩阵物化不完整；与 LINCS ∩ **正式防泄漏键表**未冻结 |
| 对照匹配 | 板内对照 + `assert_matched_controls` 冒烟通过 | 缺跨源 duplicate family → split atomic group 正式表 |
| 与药敏合流 | 方案要求先关系分析再预测 | 不阻塞路线一；缺口单列，勿用路线二 DRAFT 指标选模 |

## 4. 跨合同共享缺口

- [ ] `data/schemas/` 仍空：缺少 JSON Schema / 示例 fixture
- [ ] registry 未登记本轮「表达合同版本」条目（仅有历史 datasets.json）
- [ ] 覆盖分析报告（样本×药×基因×通路交集）仍缺正式草稿
- [ ] 统一 identity master（cell/PDO/patient/drug）仍分散在 audit + 48药表 + samples.parquet

## 建议下一动作（不阻塞当前 GPU 实验）

1. 冻结评审包：补齐内层 val 约定 + 表达 feature_space 钉死（见划分冻结清单）
2. 实装 `GeneExpression` Kryeziu loader（读 indexed.npz，导出 contract_id）
3. 通路：Hallmark 打分 → `data/processed/.../pathway_hallmark_DRAFT.parquet`
4. 扰动：选定 L3-978 或 L5，写出 ∩ 防泄漏键表 DRAFT（评价侧可挂 `check_split_leakage`）

## 阻塞

- **无阻塞**路线一继续用 DRAFT 划分与 mcrc_v3 表达。
- **阻塞「合同冻结」声明**：通路物化、扰动主级别选定、registry 合同版本、内层 val 物化/算法冻结。

## Changelog
- 2026-09-18 17:23 CST: 内层 val 算法钉死写入 `split_freeze_checklist_DRAFT.md`；通路物化占位 → `docs/reports/pathway_materialization_placeholder_DRAFT.md` + `data/processed/identity/pathway/pathway_contract_PLACEHOLDER_DRAFT.json`。
- 2026-09-18 17:26 CST: Hallmark ssGSEA 矩阵已物化 → `data/processed/identity/pathway/kryeziu_hallmark_ssgsea_DRAFT.parquet`；合同 `data/processed/identity/pathway/pathway_contract_DRAFT.json`（替换 PLACEHOLDER）。
