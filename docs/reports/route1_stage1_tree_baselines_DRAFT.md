# 路线一 · 阶段1 树模型基线（本机 CPU，DRAFT）

- 环境：WSL `drugscreening-gpu`（CPU，与双 A5000 并行）
- 数据：`/mnt/d/Code/DrugScreenLab/data/processed/transfer/mcrc_v3`
- 划分：核实患者外 5 折 DRAFT
- 训练：每折 train 患者拟合；test 患者仅评估

## 汇总
| 模型 | NDCG@5 | Spearman |
|---|---|---|
| `hist_gradient_boosting` | 0.9036 ± 0.0103 | 0.7837 ± 0.0095 |
| `random_forest` | 0.9047 ± 0.0085 | 0.7085 ± 0.0272 |

JSON：`docs/reports/route1_stage1_tree_baselines_DRAFT.metrics.json`
