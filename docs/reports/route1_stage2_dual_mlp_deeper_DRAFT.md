# 路线一 · 阶段2 DualMLP-deeper（本机双 A5000，DRAFT）

- **环境**：本机 WSL `drugscreening-gpu`
- **并行**：GPU0 fold 0/1/2；GPU1 fold 3/4；epochs=60；hidden 512 更深塔
- **数据/划分**：mcrc_v3；患者外 5 折 DRAFT；48 药

## 汇总

| 模型 | NDCG@5 | Spearman |
|---|---|---|
| DualMLP-deeper | 0.8807 ± 0.0140 | 0.7569 ± 0.0122 |

## 对照（同划分，本机）

| 模型 | NDCG@5 | Spearman |
|---|---|---|
| 药物训练均值 | 0.9050 ± 0.0077 | 0.7857 ± 0.0097 |
| DualMLP（浅） | 0.9042 ± 0.0106 | 0.7871 ± 0.0112 |
| DualMLP-deeper | 0.8807 ± 0.0140 | 0.7569 ± 0.0122 |
