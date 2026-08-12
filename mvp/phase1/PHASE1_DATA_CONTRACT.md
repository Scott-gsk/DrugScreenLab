# Phase-1 DATA CONTRACT

状态：`FROZEN_FOR_FAST_LOOP`
冻结日期：2026-08-13
研究主线：Q1 — unseen drug / unseen context 的 exact-978 Delta978 预测

## 目标与主假设

主假设只有一个：在相同 canonical dose/time 下，加入未经处理的 biological context 表达特征后，`Chemical + Context` 能在 group-atomic 的 unseen drug 和 unseen context 测试集上稳定优于 `Chemical-only`，并输出非塌缩、gene-specific 的 `Delta978`。

## 输入

| 字段 | 定义 |
| --- | --- |
| gene universe | GSE92742 的 978 个直接 landmark genes，顺序由 registered exact978 cache 固定 |
| target | matched treatment expression − matched vehicle/control expression，形状 `[978]` |
| context | matched untreated/control expression，形状 `[978]`；不是 context ID lookup |
| chemical | canonical SMILES 的 Morgan fingerprint，radius=2、128 bits、float32 |
| dose/time | canonical `10.0 µM, 6 h`，模型输入为 `log1p` 后的数值特征 |
| identity | `pert_id + cell_id + dose + time + matched control`；每个记录绑定 exact978 cache row |

## Canonical condition 与覆盖

- 仅纳入 `pert_type=trt_cp`、`10.0 µM`、`6 h`，并按 `(rna_plate, cell_id, pert_time, pert_time_unit)` 匹配 `ctl_vehicle`。
- 当前 manifest：219,901 records、55,470 treatment groups、6,382 个有效结构药物、67 个 context、917 个 matched controls。
- `restricted` 或其他无法由 RDKit 解析的结构被排除，不以 drug ID 代替结构表征。
- 训练、validation、test 按完整 treatment group 原子切分；支持 `random_group`、`cold_drug`、`cold_context`。
- 同一 untreated control 可为多个 treatment 记录提供 pre-treatment feature；这由显式 `control_policy=pre_treatment_context_feature` 声明，不把该控制表达当作 treatment label。

## 训练与评估边界

- normalization（context、dose/time、target）只在 train rows 拟合。
- `Chemical-only` 将 context 置零；`global_mean` 为无输入基线；两者均不读取 test labels。
- Small FAST 使用单 seed、最多 2,048 条代表性记录和短训练；结果只能标注为 `PROMISING`、`NO_SIGNAL` 或 `BROKEN`，不能视为正式多 seed 科学结论。
- 主报告使用 group-macro Pearson/Spearman、RMSE、MAE、direction accuracy，并记录 prediction variance 以检测 collapse。

## 禁止使用

不得使用 test/external phenotype labels 调参；不得改变四个核心问题或 train/calibration/test data role；不得把 PRISM、CTRP、GDSC 标签回灌 Phase-1 训练；不得修改原始 GSE92742 数据；不得依赖 MCPIRE_PDO 或 TriPerturb。

## 绑定资产与复现

- exact978 cache：`data/processed/lincs/GSE92742/exact978_cache_v1/exact978_cache.npy`，SHA256 `04b8bb746a61ba4992e49566315327023783ec1c0448da2a9e263e0881281733`。
- random-group manifest：`artifacts/phase1/canonical_10um_6h/random_group/manifest.json`，SHA256 `d07721affc3f07b29864e7d42f6514b1e7038bbaba9939cfedb3a86869b11d47`。
- candidate checkpoint：`artifacts/phase1/canonical_10um_6h/random_group/candidate.pt`，SHA256 `e3130417f00d6cb4cfe235b42dbd2434112afe6c183b9c67b1993a1a92aff388`。
- runtime：WSL2，Conda environment `drugscreening-gpu`；命令见 `mvp/DRUGSCREENLAB_PROGRAM_FEASIBILITY_REPORT.md`。
