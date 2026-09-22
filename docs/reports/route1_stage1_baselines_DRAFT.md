# 路线一 · 阶段1 基线（DRAFT）

- 范围：Kryeziu 核实患者外 5 折 **fold 0**（真实 processed 数据）
- 数据：`data/processed/transfer/mcrc_v3`
- 状态：**已实跑**；划分仍为 DRAFT，未冻结

## Fold 0

| 模型 | Spearman | MAE | NDCG@5（患者均） |
|---|---:|---:|---:|
| 药物训练均值 | 0.7774 | 3.9081 | 0.8956 |
| 每药 Ridge（表达） | 0.7610 | 4.1003 | — |

- train/test 行：2498 / 646
- train/test 患者：40 / 11
- Ridge 拟合药数：48

## 方案位置

对应实施方案 **路线一 · 阶段1** 的第一张真实结果表。下一步跑折 1–4 并汇总。

JSON：`docs/reports/route1_stage1_fold0_baselines_DRAFT.json`
