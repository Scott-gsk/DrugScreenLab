# 阶段1 指标汇总口径（DRAFT）— NDCG@5 / Spearman

- 对齐：`docs/研究实施方案.md` 第三、八章；代码 `src/drug_screen/evaluation/`
- 适用：Kryeziu 核实子集外 5 折 · 药物均值 / Ridge / ElasticNet 等阶段1基线
- 划分：`data/splits/kryeziu_patient_outer5fold_DRAFT.json`（泄漏干跑见 `kryeziu_leakage_dryrun_DRAFT.md`）
- 日期：2026-09-18（Asia/Shanghai）· DSL-评价审计

## 1. 评哪一列、在哪一集

| 项 | 口径 |
|---|---|
| 响应方向 | `higher_is_sensitive`（与 pairs.response 一致） |
| 未测药 | **剔除**，不计错、不补零 |
| 评价折 | 仅外层 **test** 患者/样本；内层 val 只用于选λ/早停，**不进**正式表 |
| 多样本患者 | 先患者内汇总（同药多样本：响应取均值；表达按合同），再 **患者等权** |
| 候选集 | 同一研究、可比较筛选条件；阶段1默认 48 药集合（Oxaliplatin 不在表） |

## 2. 任务与主指标

| 任务 | 单位 | 主指标 | 补充 | API |
|---|---|---|---|---|
| **B**（同患者跨药排序） | 每位 test 患者一行 | **NDCG@5** | Spearman、Top-5 overlap | `evaluate_task_b` → `summarize_task_b` |
| **A**（同药跨患者） | 每药一行 | Spearman | MAE/RMSE 可选 | `evaluate_task_a` → `summarize_task_a` |

阶段1对外主表以 **任务 B 的 mean NDCG@5** 为排序依据；任务 A 与 Spearman 同表并列，不单独改冠军规则。

## 3. 折内 → 折间汇总（必须统一）

1. **折内**：对该折 `test_patient_ids` 逐人算 `evaluate_task_b`；`summarize_task_b` = 患者等权均值（`mean_ndcg_at_5` / `mean_spearman` / `mean_top5_overlap`）。
2. **折间**：5 折的折内均值再做 **折等权宏平均**（macro-mean over folds）。勿把所有患者拼成一袋再平均（折大小略不均时会偏权）。
3. **任务 A**：每折在该折 test 患者上逐药 `evaluate_task_a`（药须 ≥2 可评患者才计 Spearman）；折内 `mean_spearman`（药等权）→ 再折等权宏平均。
4. 正式增益判断（相对药物均值等基线）：同一折、同一患者集合上的配对差分 → `paired_resample_ci`（n=2000, 95%）+ `judge_model_gain`；阶段1冒烟可先出点估计，CI 可并行后补。

## 4. 指标表建议列

`fold, model, n_test_patients, ndcg@5, spearman_B, top5_overlap, spearman_A, n_drugs_A`  
另附 `macro` 行（5 折宏平均）。文件建议：`docs/reports/stage1_baseline_metrics_DRAFT.csv`（由药敏侧写出）。

## 5. 禁止

- 用内层 val 或全数据指标代替外层 test
- 未测药用 0 / 均值填充后算 NDCG
- 样本级等权冒充患者等权（有双样本患者时）
- 与旧 EXP cold-cell 指标混表比较（集合不同）

## 6. 验收点（评价侧）

- [x] 外层患者泄漏干跑：`docs/reports/kryeziu_leakage_dryrun_DRAFT.md`
- [x] 本汇总口径文档
- [ ] 药敏侧指标表按上列写出后，抽 1 折手工核对 `ndcg_at_k` / `spearman_rho` 与表内数值一致
