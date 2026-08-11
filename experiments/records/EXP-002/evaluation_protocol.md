# EXP-002 Evaluation And Statistics Protocol

## Scope And Freeze Point

本协议仅评估 EXP-002 已批准的假设：在 GSE92742 Level-3 `pr_is_lm=1` exact-978、同板
vehicle matched-control 和 frozen split contract 下，context-conditioned `Delta978` baseline
是否优于无条件 global baseline。目标是每个已配对 chemical treatment 的 978 维 `Delta978`；
不得将 inferred non-landmark genes、Level-4/5 值或 extended-gene inference 作为 target。

本协议在 Data Steward 冻结 `DATA CONTRACT` 与 split manifest 后生效。它不改变 context
的组成：`context_id` 必须逐字采用 contract 的 canonical identity，并保留其来源字段
（至少 `cell_id`、dose/unit、time/unit；配对另以 `rna_plate` 验证）。任何改变这些字段或
target 定义的变更都必须重新冻结并记录为 `DEVIATION_FROM_PLAN`。

## Required Assertions

每一个 manifest 在计算 metric 前必须通过以下断言。

| Assertion | Required rule |
| --- | --- |
| Landmark target | 每条 target 恰为同一顺序的 978 个 `pr_is_lm=1` gene IDs；不得混入 inferred row。 |
| Treatment-control | 每个 treatment 指向一个 `ctl_vehicle`；treatment 与 control 同 `rna_plate`、`cell_id`、`pert_time`、`pert_time_unit`，并属于同一 split。247 个无候选 chemical instances 必须不在 manifest。 |
| Compound identity | 使用 Data Contract 中的 canonical compound ID；不得按原始名称、dose 或 replicate 重新命名来规避 cold-drug。 |
| Context identity | 使用 frozen canonical `context_id`；不得在 test 后依据结果拆分或合并 context。 |
| Replicate family | 同一 `replicate_family_id` 只能在一个 split；replicate collapse 只能在 split 内完成。 |
| Cold drug | canonical compound 只能属于 train/validation/test 之一，test compound 不得出现在任何 development split。 |
| Cold context | canonical context 只能属于 train/validation/test 之一，test context 不得出现在任何 development split。 |

`cold_drug` 只隔离 compound axis，`cold_context` 只隔离 context axis；二者均必须隔离
treatment-control pair 与 replicate family。一个 stratum 不能伪装为另一个 stratum，也不能
以 random 或 in-domain split 取代。

## Evaluation Units And Metrics

主要 unit 是 canonical `(compound_id, context_id)` 下的 held-out paired-treatment group，
而不是 978 个 gene 或单一 technical replicate。先在 group 内按预先声明的方法聚合，再对
group-level metrics macro average；同时保留每个 treatment、gene、compound、context 的明细。

对每个 exact-978 vector 计算 Pearson、Spearman、RMSE、MAE 和 direction accuracy。Pearson
与 Spearman 的 macro mean 是主要 association report；RMSE、MAE 与 direction accuracy 是
共同主要 error/direction report，不能仅挑选一项。direction 的 zero epsilon、gene 顺序、
replicate 聚合规则必须写入 run config 并在读取 test labels 前冻结。常数 vector 产生的
undefined correlation 必须明确报告计数，不得悄悄替换为零或剔除。

所有指标均按下列 strata 报告：`cold_drug`、`cold_context`、compound、context、gene，及
预注册的 failure strata（例如 low-signal、low-replicate 或 unseen-context fallback）。不得
使用 pooled gene-level 数字掩盖少数 compound/context 的失败。

## Fair Baseline Rules

global baseline、context-conditioned baseline 和任何后续比较方法必须使用同一份 exact-978
manifest、control pairing、train/validation/test assignments、gene ordering 与 scoring code。

- 拟合统计量只能从 train 得到；validation 仅用于已经声明的选择，test 从不参与 fit、tune、
  calibration、normalization parameter estimation 或 fallback 选择。
- global baseline 只能预测 training `Delta978` summary。
- context-conditioned baseline 仅可使用 training 中该 context 的信息。对 cold-context 的未见
  context，必须使用在 test 前声明、只依赖 training 的 fallback；不得从 held-out context 标签
 估计 context mean。
- 报告每个方法的 eligible-count、fallback-count 与 dropped-count；不得因某方法失败而改变
 test cohort。

## Uncertainty, Seeds And Decision Evidence

对每个 seed 独立训练/生成预测，固定并报告 seed list、software/source revision、manifest
digest、config digest。对 group-level values 以 group 为 resampling unit，使用固定 seed 的
nonparametric bootstrap（默认 2,000 resamples，95% CI）；绝不 bootstrap 978 genes 当作独立
biological samples。除各方法 CI 外，必须 bootstrap paired baseline difference。

在首次 test-label 访问前，run config 必须声明：primary metric、最小实际效应阈值、所需 seed
一致性规则、最低有效 group 数、direction epsilon、replicate aggregation 和所有 fallback。
这些值缺失时，评估只能标记 `INCOMPLETE_PRE_REGISTRATION`，不能作 Go/No-Go。

Gate B 通过的证据是：在 **两个** OOD strata 分别满足预注册的 improvement、CI 与 multi-seed
规则，且没有通过改变 cohort、metric、threshold 或 split 获得结果。任一 stratum 未满足时，
结论为 `VALID_NEGATIVE_OR_INCONCLUSIVE`，具体取决于数据/执行完整性；这不是修改 test split
或事后替换指标的理由。

## Required Evidence Package

评估交接必须包含：EXP-ID、approved hypothesis、source revision/commit 或 diff、DATA CONTRACT
与 manifest identifiers/digests、exact-978 gene-list digest、control-pair/exclusion count、每个
stratum 的 split assertions、run config、seed list、per-seed metrics、macro metrics、paired
bootstrap differences、CI、failure/fallback/dropped counts、artifact paths 及已知 deviations。
Reviewer 应从原始 manifest 与 run artifacts 复核上述字段，不能只接受汇总表。
