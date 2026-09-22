# Kryeziu 患者外 5 折 — 泄漏干跑（DRAFT）

- 划分：`data/splits/kryeziu_patient_outer5fold_DRAFT.json`
- 状态：DRAFT（未冻结）
- 检查：`check_split_leakage`（主任务 `allow_drug_overlap=True`）+ 样本 ID 重叠 + sample→patient 一致性
- 时间：2026-09-18（Asia/Shanghai）
- 执行：DSL-评价审计

## 结果摘要

| fold | train/val/test 患者 | train/val/test 样本 | train∩test 患者 | 样本交叉 | ok |
|---:|---|---|---:|---|---|
| 0 | 40 / 0 / 11 | 47 / 0 / 13 | 0 | 无 | True |
| 1 | 41 / 0 / 10 | 48 / 0 / 12 | 0 | 无 | True |
| 2 | 41 / 0 / 10 | 49 / 0 / 11 | 0 | 无 | True |
| 3 | 41 / 0 / 10 | 49 / 0 / 11 | 0 | 无 | True |
| 4 | 41 / 0 / 10 | 47 / 0 / 13 | 0 | 无 | True |

- 51 核实患者均恰好出现在 1 个外层 test 折；每折 `train∪test` 患者数 = 51。
- 全折 `val_patient_ids` / `val_sample_ids` 为空（见下节约定）。
- 本干跑未检药物集合（划分文件 `drug_set.drug_ids` 仍空）；48 药身份另表。
- **结论**：外层患者泄漏检查 **通过**；仍标 DRAFT，不进正式比较表。

## 内层 val 约定（评价口径）

1. 外层 `test_*` 仅用于该折最终报告；**禁止**参与调参 / early-stop / 模型选择。
2. 当前 DRAFT 未预填内层 val：训练方必须在 **`train_patient_ids` 内部** 再划 val（患者级），不得从 test 取。
3. 推荐默认：外层 train 患者上再做 **患者级 4:1**（或嵌套 5 折）得到 val；样本随患者走，同一患者多样本不得跨 train/val。
4. 运行记录须写下本次 fit 的内层 val 患者列表与 seed；冻结后可将推荐内层固化进 JSON（不得改外层 test）。
5. 指标汇总患者等权；配对 CI / `judge_model_gain` 只对外层 test 结论生效。

## 未做

- ChemPert∩LINCS 扰动 ID 交叉（本划分非扰动任务）
- 预处理仅训练集拟合的审计钩子
