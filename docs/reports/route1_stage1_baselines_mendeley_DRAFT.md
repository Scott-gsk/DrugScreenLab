# 路线一·阶段1 基线实跑（Kryeziu 外5折 DRAFT）

- 时间：2026-09-18T15:21:46+08:00（Asia/Shanghai）
- 划分：`data/splits/kryeziu_patient_outer5fold_DRAFT.json`（身份核实患者；32 prefix-only 已排除）
- 标签：DSS（越高越敏感）；未测响应不补 0
- 预处理：log1p 计数 → 训练集 top-2000 方差基因 → 每药 StandardScaler（仅 train fit）
- 约束：不读旧 EXP 选型；raw 只读；本轮非大 GPU 深度模型

## 1. 数据路径确认

| 项 | 路径 |
|---|---|
| 表达 | `data/interim/kryeziu_mendeley_v3/Data_S2.xlsx` |
| 标签 DSS | `data/interim/kryeziu_mendeley_v3/Data_S5.xlsx` |
| 药物表 | `data/registry/kryeziu_structure_confirmed_drugs_DRAFT.json` |
| processed transfer/mcrc_v3 | `None` |
| processed mcrc_pdo_2026 | `None` |
| GDSC2 | `None` |
| CTRPv2 | `None` |

来源说明：Mendeley Data S2/S5 mirrored under data/interim/kryeziu_mendeley_v3 (DOI 10.17632/hr94h42xdc.3). processed transfer/mcrc_v3 absent on this box.

## 2. 数据规模（核实子集 ∩ 可比较 DSS）

- 样本（核实）：60
- 患者：51
- 药物（harmonized DSS 列）：24
- 有效 pair 行：1440
- 表达矩阵形状：[92, 19421]

## 3. 已跑 / 跳过模型

- ✅ `drug_train_mean`
- ⏭️ `drug_only_fingerprint`（无指纹/SMILES 表）
- ✅ `per_drug_ridge_expression`
- ✅ `per_drug_elasticnet_expression`
- ⏭️ 细胞系同协议基线（GDSC2/CTRPv2 未就绪）

## 4. 指标汇总（患者内排序；折间 mean±std）

| 模型 | mean NDCG@5 ± std | mean Spearman ± std |
|---|---|---|
| `drug_train_mean` | 0.9049 ± 0.0158 | 0.7212 ± 0.0261 |
| `per_drug_ridge_expression` | 0.9193 ± 0.0126 | 0.7449 ± 0.0289 |
| `per_drug_elasticnet_expression` | 0.9077 ± 0.0099 | 0.7188 ± 0.0291 |

### 分折明细

| fold | model | NDCG@5 | Spearman | n_patients | n_train_rows | n_test_rows |
|---:|---|---:|---:|---:|---:|---:|
| 0 | `drug_train_mean` | 0.8835 | 0.7421 | 11 | 1128 | 312 |
| 0 | `per_drug_ridge_expression` | 0.9127 | 0.7847 | 11 | 1128 | 312 |
| 0 | `per_drug_elasticnet_expression` | 0.9042 | 0.7527 | 11 | 1128 | 312 |
| 1 | `drug_train_mean` | 0.8975 | 0.7035 | 10 | 1152 | 288 |
| 1 | `per_drug_ridge_expression` | 0.9026 | 0.7390 | 10 | 1152 | 288 |
| 1 | `per_drug_elasticnet_expression` | 0.9132 | 0.7163 | 10 | 1152 | 288 |
| 2 | `drug_train_mean` | 0.9201 | 0.6939 | 10 | 1176 | 264 |
| 2 | `per_drug_ridge_expression` | 0.9407 | 0.6957 | 10 | 1176 | 264 |
| 2 | `per_drug_elasticnet_expression` | 0.9210 | 0.6655 | 10 | 1176 | 264 |
| 3 | `drug_train_mean` | 0.9261 | 0.7618 | 10 | 1176 | 264 |
| 3 | `per_drug_ridge_expression` | 0.9223 | 0.7503 | 10 | 1176 | 264 |
| 3 | `per_drug_elasticnet_expression` | 0.8912 | 0.7294 | 10 | 1176 | 264 |
| 4 | `drug_train_mean` | 0.8972 | 0.7046 | 10 | 1128 | 312 |
| 4 | `per_drug_ridge_expression` | 0.9180 | 0.7550 | 10 | 1128 | 312 |
| 4 | `per_drug_elasticnet_expression` | 0.9088 | 0.7301 | 10 | 1128 | 312 |

## 5. 阻塞 / 跳过

- **PROCESSED_MCRC_V3_ABSENT_ON_HOST**: transfer/mcrc_v3 not on box; using Mendeley interim S2/S5
- **CELL_LINE_SOURCE_NOT_READY**: GDSC2/CTRPv2 not found; skipped source-domain baselines
- **DRUG_FINGERPRINTS_UNAVAILABLE**: drug-only fingerprint model skipped

## 6. 产物

- `docs/reports/route1_stage1_baselines_DRAFT.md`
- `docs/reports/route1_stage1_baselines_DRAFT.metrics.json`

## 7. 复跑

```bash
cd /mnt/d/Code/DrugScreenLab
PYTHONPATH=src python scripts/modeling/run_route1_stage1_baselines_DRAFT.py
# 或 conda: conda run -n drugscreening-gpu python scripts/modeling/run_route1_stage1_baselines_DRAFT.py
```
