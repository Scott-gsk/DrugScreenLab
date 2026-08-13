# Genetic→Chemical Low-Data Transfer Contract

状态：`FAST_CONTRACT_READY`。该 contract 只定义下一步可证伪的扩展，不改变已冻结的 Phase-1 train/validation/test 角色、gene universe、control policy 或 Delta978 endpoint。

## 一个主要假设

在同一个 `Context + Perturbagen representation → Delta978` 接口下，使用遗传扰动数据进行预训练或联合训练，能够在化学监督逐步减少时保持或提高化学扰动预测质量，并最终改善疾病表达谱逆转后的下游 drug ranking。

## 统一接口

| 输入/输出 | 定义 |
| --- | --- |
| Context | 978 landmark context vector；未来允许 CCLE/DepMap basal、LINCS DMSO、tumor/PDO RNA-seq 经相同 harmonization 进入 |
| Perturbagen | chemical：KPGT/UniPert；genetic：预先登记的 genetic perturbation representation；二者共享 interaction/response head |
| Dose/time | 原 Phase-1 canonical dose/time；遗传数据若不存在同一字段，必须在 dataset-specific adapter 中显式记录，不得伪造 chemical dose |
| Target | matched treatment − matched control，严格为 978 维 Delta978 |
| External endpoint | 只在预测与 ranking digest 冻结后读取 PRISM/GDSC phenotype |

## FAST regimes

化学监督保留比例：`100%`, `50%`, `20%`, `10%`, `5%`（若某比例无法保留 train/test group support，则记录 `BROKEN`，不回填标签）。

每个 regime 必须比较：

1. chemical-only training；
2. genetic pretraining → chemical fine-tuning；
3. genetic + chemical joint training（若数据 contract 允许）。

首轮使用单 seed、小/中等规模和 validation-driven debugging；结果只能为 `PROMISING`、`NO_SIGNAL` 或 `BROKEN`，不构成正式多 seed 结论。

## 预注册 endpoint 与防泄漏

- Primary FAST endpoint：固定 cold-drug 与 cold-context test group 上的 group-macro Spearman，辅以 RMSE、MAE、direction accuracy 和 prediction variance。
- Secondary utility endpoint：预测 Delta978 → disease reversal → ranking，在已冻结的 PRISM/GDSC cohort 上报告 line-level ranking 指标。
- 禁止使用 PRISM/GDSC response、external labels 或任何 test label 调整 feature、loss、threshold、regime 选择或模型 checkpoint。
- genetic dataset identity、gene universe、control、split、mapping、normalization 和 output 必须先写入 audit；任何会改变 response 定义的变化都暂停下游。

## 停止规则

- 若 genetic adapter 不能提供可审计的 978 维 target/control 关系：`BROKEN`，不进入 transfer。
- 若低数据各 regime 均无增益但接口和 split 合法：`NO_SIGNAL`，保留 genetic→chemical 共享空间作为机制假设，不把一次 negative transfer 写成整体否定。
- 只有在低数据和至少一个下游 utility endpoint 同时有方向一致信号后，才进入 Medium/MVP scaling。
