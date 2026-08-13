# DRUGSCREENLAB PROGRAM FEASIBILITY REPORT

> **Current checkpoint (2026-08-13):** Foundation status is `XPERT_FOUNDATION_READY`. The first
> novel XPert extension, EXP-005, completed its fixed FAST comparison with
> `NO_MATERIAL_FAST_INCREMENT`; official XPert remains the retained baseline and no MEDIUM loop was
> started. The quantitative A/B/C audit is `mvp/foundation/xpert/EXP005_FAST_COMPARISON.json`.

## Current program-level decision

Overall status remains `PARTIAL` / `CONTINUE_WITH_EXTENSIONS`.

| Question | Latest evidence | Status |
| --- | --- | --- |
| Q1 Δ978 prediction | Official XPert foundation is reproducible; six bounded EXP-005 runs are finite and non-collapsed, but KPGT/UniPert additive tokens did not meet the two-split gain rule | `YELLOW` |
| Q2 reversal screening | Broad identity/context integration covers 1,836 drugs and 10 exact CRC contexts; predicted ranking is a diagnostic, not yet a positive cell-line gate | `YELLOW` |
| Q3 mechanism transfer | EXP-004 genetic→chemical FAST result remains unchanged; EXP-006 data preparation is `DATA_PARTIAL` only | `YELLOW` |
| Q4 PDO ranking | No PDO training or conclusion; context adapter track remains preparation-only | `DEFERRED` |

The largest current bottleneck is matched-support downstream validation: the foundation can infer a
large exact-context cohort, but cross-study evidence and additive extension value are not yet sufficient
for a green cell-line gate. The next work should inspect XPert error modes and prepare the response-blind
context/transfer contracts; it should not spend the main budget re-optimizing the historical simple model
or launch an unqualified MEDIUM token loop.

日期：2026-08-13
代码 checkpoint：`9d50521`
依据：Master Research Plan 与用户提供的 Program-Level Autonomous R&D Authorization
本轮级别：`SMALL_FAST_SINGLE_SEED`，不是正式多 seed 研究结论

## 总结

当前整体决策：`PARTIAL`。

DrugScreenLab 的核心 idea 仍值得继续投入，但当前证据只足以支持“Q1 Phase-1 已有可复现的方向性信号，cell-line downstream 有初步可行性，尚未完成外部交叉研究或 organoid gate”。不能把本轮结果写成已经证明可泛化的药物筛选系统。

| Question | Evidence | Status |
| --- | --- | --- |
| Q1 exact-978 prediction | Context+Chemical 在 random-group、cold-drug、cold-context 的 group-macro 测试集均优于 Chemical-only，且预测方差未塌缩 | `YELLOW`（PROMISING FAST signal） |
| Q2 disease reversal / cell-line screening | 4-drug frozen cohort 的预测 reversal 在 PRISM 上有弱到中等可行性信号；exact LINCS context 子集优于 fallback 子集；尚无 CTRP/GDSC 外部 check | `YELLOW`（PARTIAL） |
| Q3 mechanism transfer | UniPert 已完成一次固定 backbone FAST 表示替换；Target/Pathway provenance 已完成但学习增量因 cohort 覆盖不足而未运行 | `YELLOW` |
| Q4 PDO ranking | PDO branch 保持 `DEFERRED_PDO_LEG`，未使用 PDO 结论替代 cell-line evidence | `DEFERRED` |

## Phase-1

### Backbone 与 DATA CONTRACT

Phase-1 固定为：978 landmark untreated/control expression context + 128-bit Morgan chemical fingerprint + numeric dose/time，经 concat + MLP 显式学习 interaction，输出 gene-specific `Delta978`。canonical condition 为 `10.0 µM, 6 h`。

冻结覆盖为 219,901 records、55,470 groups、6,382 个有效结构药物、67 个 contexts、917 个 matched controls。完整契约见 `mvp/phase1/PHASE1_DATA_CONTRACT.md`；manifest、checksum 与运行命令见 `mvp/phase1/PHASE1_FAST_EVIDENCE.json`。

### Chemical-only vs Chemical+Context

以下为 group-macro test point estimates；Chemical-only 将 context 置零，global mean 是额外的无输入基线。

| Split | Model | Pearson | Spearman | RMSE | MAE | Direction | Prediction variance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| random-group | Chemical-only | 0.071 | 0.062 | 1.220 | 0.943 | 0.522 | 0.124 |
| random-group | Chemical+Context | 0.344 | 0.337 | 1.125 | 0.865 | 0.628 | 0.830 |
| cold-drug | Chemical-only | 0.008 | 0.008 | 1.240 | 0.956 | 0.495 | 0.131 |
| cold-drug | Chemical+Context | 0.332 | 0.324 | 1.056 | 0.806 | 0.630 | 1.029 |
| cold-context | Chemical-only | 0.109 | 0.101 | 1.111 | 0.848 | 0.532 | 0.122 |
| cold-context | Chemical+Context | 0.288 | 0.283 | 1.067 | 0.815 | 0.600 | 0.647 |

### Context increment

Context+Chemical 相对 Chemical-only 的 group-macro Pearson 增量分别为 `+0.273`、`+0.324`、`+0.179`（random-group、cold-drug、cold-context）。这满足 FAST loop 的“context 产生增量”判据，并且 cold-drug 与 cold-context 都保留正相关和方向一致性；不过训练仍是单 seed、bounded 2,048-record subset，因此 Q1 暂定 `YELLOW`，下一步需要 full registered cache、多 seed 与正式不确定性估计。

### 最大 prediction bottleneck

最大的瓶颈不是 978 输出本身，而是下游验证的 support mismatch：当前候选只有 4 个药物，PRISM 35 条 CRC lines 中只有 11 条能取得 exact LINCS context，另外 24 条只能使用 reference-context fallback；同时外部 CTRP/GDSC 原始资产本轮未能从官方入口取得。这个瓶颈会把“模型预测误差”和“context/compound identity 不匹配”混在一起。

## Cell-line screening

### Oracle

历史 MVP-001 observed-oracle 保持不变：4-drug cohort、PRISM 135 rows / 35 lines，eligible lines 的 macro mean Spearman 约 `0.479`，median `0.8`，positive fraction `0.788`，mean Top-2 overlap 约 `0.803`。这是上限/可行性诊断，不代表 learned predictor 已达到该水平。

### Predicted

本轮先冻结 4-drug candidate identity，再生成 Phase-1 Delta978、疾病 reversal 与 ranking，最后读取 PRISM phenotype。所有 35 条 lines 中 33 条满足至少 3 个候选药物；预测 reversal 的 all-line macro mean Spearman 为 `0.139`，median `0.2`，positive fraction `0.545`，mean Top-2 overlap `0.606`。

只看 11 条 exact LINCS context lines 时，macro mean Spearman 为 `0.418`，median `0.6`，positive fraction `0.636`，mean Top-2 overlap `0.818`。这说明 exact-context subset 保留了 downstream feasibility signal，但不能把它外推到 24 条 fallback lines。

### Oracle → Predicted degradation 与 PRISM coverage

Predicted-vs-observed global drug score Spearman 为 `0.2`，Top-2 overlap 为 `1/2`。该比较只作 diagnostic：预测分数按 PRISM lines 聚合，而历史 observed score 来自既有 observed-oracle 聚合，支持并不完全匹配。PRISM 当前没有冻结的 binary sensitive label，因此本轮不报告 AUROC、AUPRC 或 label-based Recall@K。

详细 ranking、每行指标和 provenance 位于：

`artifacts/phase1/canonical_10um_6h/random_group/prism_evaluation/phase1_prism_evaluation_summary.json`

## Mechanism

- UniPert increment：`NO_SIGNAL`。同一 random-group、2048-record、single-seed Small 子集上，Chemical+Context 的 group-macro Spearman 为 `0.3373`，替换为官方 UniPert 256 维 chemical encoder 后为 `0.1349`；因此不扩大 UniPert，也不把该结果解释为 UniPert 本身的生物学失败。
- Target/Pathway increment：`NOT_RUN_COVERAGE_INSUFFICIENT`。四个冻结 PRISM 候选的 ChEMBL 结构/机制身份与 Reactome mapping 入口已形成 compact provenance，但 cohort 太小，不能拟合 additive weight 或把 PRISM 标签用于调参。见 `mvp/phase3/TARGET_PATHWAY_FAST_EVIDENCE.json`。
- 本轮没有让 mechanism feature 改变 Phase-1 的标签、split、loss 或 endpoint；UniPert 仅作为 representation FAST 对照，Target/Pathway 保持 forward preparation。

## External cross-study

目标是一次 frozen CTRP 或 GDSC check。官方 [CTRP portal](https://portals.broadinstitute.org/ctrp.v2.2/) 记录了公开的 481 compounds × 860 cancer cell lines 资源；本轮已下载并审计官方 NCI INS 2.2.0 mapping，但 legacy file route 重定向至 `studycatalog.cancer.gov`，公开 GraphQL file service 返回 HTTP 500，未获得可审计的 CTRP sensitivity archive。见 `mvp/phase1/CTRP_EXTERNAL_ASSET_AUDIT.json`。冻结 evaluator 已实现于 `src/drug_screen/evaluation/cross_study.py`，但本轮状态仍为 `NOT_RUN_EXTERNAL_ASSET_UNAVAILABLE`，不对 Q2 作绿色结论。

## Overall Decision

`PARTIAL`。

当前整体 DrugScreenLab idea 仍值得继续投入，理由是：

1. Phase-1 的 context-conditioned signal 在 unseen drug 与 unseen context diagnostics 上方向一致。
2. learned predicted reversal 在 exact-context PRISM 子集上保留了部分 oracle ranking 行为。
3. identity、matched control、exact978 cache、group-atomic split 和 train-only normalization 已形成可复现工程骨架。

但当前不满足 `GREEN` 的 cell-line gate：候选 cohort 太小、fallback context 占比高、external cross-study 尚未完成，且没有 Independent Reviewer 对本轮 FAST evidence 作正式 verdict。

## Largest Bottleneck

`Matched-support downstream validation`：需要扩大 verified LINCS↔PRISM compound/context overlap，并在不调参的前提下完成一次官方可审计 CTRP/GDSC frozen check。

## Next Highest-Value Work

1. 在冻结 DATA CONTRACT 上进入 RIGOROUS Q1：registered exact978 full asset、cold-drug/cold-context、多 seed、bootstrap/CI，并保留 Chemical-only 对照。
2. 修复 downstream support：扩大 exact LINCS context 与 canonical compound overlap，重新运行 predicted-vs-observed、Top-K 和 line-level metrics；随后完成一次官方 CTRP 或 GDSC frozen check。
3. 只有在 cross-study 与 mechanism coverage 形成可用 support 后，才重新评估 Target/Pathway additive prior 与 UniPert 的扩展；当前不扩大已出现 `NO_SIGNAL` 的 UniPert 分支，也不提前进入 PDO。

## Reproducibility and checkpoint

运行环境固定为 WSL2 + Conda `drugscreening-gpu`；Python、测试和数据处理均不得使用 Windows Python。主要命令形态：

```text
PYTHONPATH=src conda run --no-capture-output -n drugscreening-gpu python scripts/data/build_phase1_manifest.py ...
PYTHONPATH=src conda run --no-capture-output -n drugscreening-gpu python scripts/modeling/run_phase1_tiny.py ...
PYTHONPATH=src conda run --no-capture-output -n drugscreening-gpu python scripts/evaluation/run_phase1_prism.py ...
```

Large manifests、cache、checkpoint、prediction tables remain local and ignored by Git. Tracked code, tests, data contract, compact metrics and this report are the audit checkpoint; local large assets are bound by path and checksum in `PHASE1_DATA_CONTRACT.md` and `PHASE1_FAST_EVIDENCE.json`.

验证结果：指定环境下完整测试将覆盖原有 60 项及本轮新增测试；数据注册表命令返回 `PASS`；`git diff --check` 通过。
