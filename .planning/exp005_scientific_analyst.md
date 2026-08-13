# EXP-005 Scientific/Evaluation Analyst evidence

## Scope and endpoint interpretation

本分析只针对 EXP-005。主 endpoint 是固定 XPert A/B/C、`split_cold_cell_1` 与
`split_cold_drug_1` 上的 held-out Delta978 row Spearman 增量。下游 reversal 定义为
`-Spearman(CRC signed disease signature, predicted Delta978)`，Broad PRISM 的
`sensitivity_score` 仅在 identity/context freeze 后读取，因此它是独立 translation
diagnostic，不是训练标签、模型选择依据或临床 efficacy ground truth。

## 已支持主张

1. **FAST architecture decision：NO_MATERIAL_FAST_INCREMENT。**
   `mvp/foundation/xpert/EXP005_FAST_COMPARISON.json` 中：

   | split | A Spearman | B 增量 | C 增量 |
   |---|---:|---:|---:|
   | cold-cell | 0.086005 | -0.004984 | -0.029930 |
   | cold-drug | 0.052936 | +0.032065 | +0.002903 |

   预注册规则要求同一 additive variant 在两个 split 均达到 `+0.02`；B 仅通过
   cold-drug，C 两者均未通过，故不能升级 medium architecture budget。

2. **Downstream 一致性不足以支持 additive 增益。** Broad PRISM predicted
   macro-Spearman：A/B/C 分别为 cold-cell `0.01361/0.00764/-0.00918`，
   cold-drug `-0.01554/-0.02389/-0.03085`；NDCG@10 也无跨 split 一致改善（B
   cold-drug 为 0.4771，低于 A 的 0.5084）。因此 Delta978 的单 split 正增益没有
   转译成 efficacy-ranking 增益。

3. **Oracle 与 predicted 的差距只能作条件性诊断。** 每个 EXP005 FAST Broad
   evaluation 的 observed LINCS oracle 仅覆盖 2 条可配对 lines（macro Spearman
   0.1539，NDCG@10 0.6594），而 predicted PRISM 使用 10 条 lines、最多 1836
   drugs（A cold-cell macro Spearman 0.0136）。support set 不同，不能把 0.1539→0.0136
   报告成严格 oracle-to-predicted degradation。历史 `ADAPTER_DOWNSTREAM_RESULT.json`
   的四药/40-row 诊断（global predicted-vs-observed oracle Spearman 0.8，PRISM
   macro Spearman 0.46）仅为 MVP feasibility 上限参考，不能外推至 1836-drug cohort。

## 未支持或不可声称

- 不能声称 KPGT 或 UniPert 已改善 cold-cell/cold-drug 泛化；B 的 cold-drug 单点增益
  不满足双 split 规则，C 无材料增益。
- 不能声称 reversal ranking 等价于药效、临床获益或机制验证；当前 PRISM 仅为冻结后
  的 sensitivity ranking diagnostic，且没有预注册 binary sensitive label，故不报告
  AUROC/AUPRC。
- 不能声称 Oracle→predicted gap 已被精确量化；缺少同一 line、同一 drug support
  的 paired oracle/predicted digest、置信区间和多 seed 方差。
- 未发现 baseline-equivalence（A≈B/C gate=0）独立 artifact；实现说明该测试应在训练前
  执行，但当前证据包未提供数值结果。因此该子项只能记为未验证，不应回溯改变 FAST 判定。

## Gate 判定

- **Gate A（数据/身份/endpoint/split 完整性）：PASS-with-caveat。** 固定 split、样本
  digest、finite Delta978、prediction std>0、train-test cold context/drug overlap
  断言均有记录，且 PRISM response 在 freeze 后读取。FAST runner 在无 valid partition
  时复用 test 作为 valid，但不用于 checkpoint 选择；这降低了严格独立 validation 的
  解释强度，后续正式 EXP 应显式区分 valid/test。
- **Gate B（两个 OOD strata 的预注册增益及下游一致性）：FAIL。** 没有 additive
  variant 同时在 cold-cell 与 cold-drug 达到 +0.02，且 Broad ranking 没有一致增益。
- **Gate C（cold-cell 与 cold-drug 分层并独立报告）：PASS。** 两个 strata 均单独列出，
  未用一个 split 的结果覆盖另一个；但不能据此推断跨 split 稳定性（单 seed、4096-row
  FAST budget）。

## EXP-006 readiness

`mvp/foundation/xpert/EXP006_GENETIC_CHEMICAL_PREPARATION.md` 明确状态为
`PREPARATION_ONLY / DATA_PARTIAL`，不是正式 EXP。当前缺少新数据的 dataset/version/
accession/checksum、Delta978 gene mapping、matched-control 证据、group-atomic split
manifest 及 provenance bundle；GDSC 官方 release asset 的 local freeze/checksum 也
尚未完成。故 EXP-006 **NOT READY / 不得启动训练或读取 response**。即使 EXP-005 继续，
也不能用其预测、PRISM response 或日志替代 EXP-006 的训练标签或数据身份。

## Bottleneck and stop/go rules

当前主要瓶颈是 learned Delta978 的信息量及其 `Delta978 → reversal → efficacy`
translation（而非 identity join plumbing）。建议：

- **STOP：** 不追加 EXP-005 architecture/medium/full budget，不把 B cold-drug 单点
  增益升级为科学结论；保留官方 XPert foundation baseline。
- **GO（有条件）：** 仅在补充同-support observed-oracle/predicted paired digest、
  显式独立 test、baseline-equivalence 数值审计后，进行预注册 multi-seed 小规模复核。
  复核仍须两 OOD split 同时达到 +0.02，且 PRISM macro-Spearman/NDCG/top-k 至少一个
  预定义指标方向一致并给出 CI，才可考虑更高预算。
- **EXP-006 GO 条件：** Data Steward 将上述 DATA CONTRACT 全部冻结并标记 VALID，
  GDSC 官方资产 checksum 可追溯、四药 identity 与 context mapping 无歧义、evaluator
  返回 COMPLETE 且 `labels_used_for_tuning=false` 后，Manager 才能另行创建并审批
  EXP-006。

## 缺失指标清单

尚未提供或不宜从现有文件推断：

1. A≈B(gate=0)≈C(gate=0) 的数值差异/容差；
2. 多 seed 均值、方差、bootstrap CI 或显著性/实际效应大小；
3. 同一 line×drug support 下的 oracle/predicted reversal Spearman、Kendall、NDCG@10/20
   和 top-k overlap（当前 oracle 2-line 与 predicted 10-line 不同分母）；
4. PRISM ranking 与 Delta978 row metric 的预注册关联检验及 failure-mode 分解；
5. 完整 EXP-006 GDSC release checksum、label schema 与 context crosswalk。
