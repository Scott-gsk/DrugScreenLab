# 路线一 · Stage2 第二波对照（本机权威，DRAFT）

- **环境**：本机 WSL `drugscreening-gpu`（只认这里）
- **划分**：患者外 5 折 DRAFT；`mcrc_v3`；核实患者；48 药
- **训练定义**：每折训练患者上拟合，测试患者上算患者宏平均 NDCG@5 / Spearman

## 结果表

| 模型 | NDCG@5 | Spearman | 备注 |
|---|---|---|---|
| 药物训练均值 | 0.9050 ± 0.0077 | 0.7857 ± 0.0097 | 阶段1 强基线 |
| HistGBM | 0.9036 ± 0.0103 | 0.7837 ± 0.0095 | CPU 树 |
| DualMLP（浅，40ep） | 0.9042 ± 0.0106 | 0.7871 ± 0.0112 | 双 A5000 五折 |
| DualMLP-deeper（60ep） | 0.8807 ± 0.0140 | 0.7569 ± 0.0122 | **更差，过拟合迹象** |
| Hallmark pathway Ridge | 0.9002 ± 0.0135 | 0.7696 ± 0.0070 | 通路均值+药 one-hot；*std 见 metrics |

## 读数

1. 加深 DualMLP **没有**超过浅网络，也没超过药物均值；NDCG/Spearman 都掉了一截。
2. Hallmark 通路 Ridge 接近基线但仍略低。
3. 下一刀更有价值的方向：正则更强 / 更小网络 / 真正的预训练或 PDO 迁移，而不是继续盲目加宽加深。

## 产物

- `docs/reports/route1_stage2_dual_mlp_deeper_DRAFT.md`
- `docs/reports/route1_stage2_pathway_ridge_DRAFT.metrics.json`
- `artifacts/route1_stage2_dual_mlp_deeper_DRAFT/`
- `artifacts/route1_stage2_pathway_ridge_DRAFT/`
