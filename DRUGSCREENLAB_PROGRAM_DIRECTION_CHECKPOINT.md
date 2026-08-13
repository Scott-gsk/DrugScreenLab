# DRUGSCREENLAB PROGRAM DIRECTION CHECKPOINT

更新日期：2026-08-13
Research Manager：`/root`（persistent orchestrator）

## 当前 Program map

```text
XPert Foundation
  → strong-foundation extensions (EXP-005 ATTEMPT-2)
  → observed/predicted Δ978 disease reversal
  → Broad PRISM Top-K vs null
  → frozen GDSC external check
  → EXP-006 Genetic → Chemical transfer
  → context adapter / organoid genetic adaptation
  → low-chemical-data PDO ranking
```

本轮只把能推进这条链的工作列为主线；数据下载、mapping、registry、adapter 和测试属于 Infrastructure，不单独构成 Research Milestone。

## Track 状态

| Track | 当前状态 | 下一决定性证据 |
|---|---|---|
| A Strong Perturbation Model | `XPERT_FOUNDATION_READY`；XPert 为 foundation champion | Strong XPert 上 KPGT/UniPert 的 gated additive ATTEMPT-2 |
| B Screening Translation | `ORACLE_AND_NULL_NORMALIZED_TOPK_IN_PROGRESS` | Observed LINCS Oracle cohort、Top-K lift/ΔNDCG、predicted→oracle gap |
| C Genetic → Chemical | `PROMISING_FAST_GENETIC_TO_CHEMICAL_TRANSFER`（EXP-004，单 seed FAST） | 20%/10% unique-compound regime 的复核与 downstream translation |
| D Context Transfer | `METADATA_READY_ADAPTATION_PENDING` | L1000/CCLE/RNA-seq → XPert-compatible 978 context adapter |
| E Mechanism | `FORWARD_PREPARATION_COMPLETE` | 在 shared response backbone 稳定后验证 cold-drug/OOD/transfer |

## Strong Foundation

已接受的 XPert foundation evidence 保留：cold-cell 与 cold-drug 存在明确 Δ978 predictive signal；prediction finite 且无 collapse。当前 champion 为 `official_xpert_foundation_exp005_fast_baseline_retained`。任何 extension 若不能直接加载该 checkpoint，或恢复到预定义 foundation-compatible range，必须标记 `INCONCLUSIVE_BASELINE_NOT_RECOVERED`，不得用于 extension Go/No-Go。

## EXP-005 corrected result

ATTEMPT-1 现标记为 `INCONCLUSIVE_PROTOCOL_MISMATCH`，并保留为 `NEGATIVE_ENGINEERING_ATTEMPT`。原因是 A/B/C 重新随机初始化、仅 4,096 rows、2 epochs，A 明显低于已验证 foundation；因此它回答的是“弱化且欠训练的 XPert 上随机 token overlay 是否快速增益”，没有回答“Strong XPert + KPGT/UniPert 是否有增量”。

ATTEMPT-2 已在同一 EXP-ID 下启动：A 直接继承 strong XPert；B/C 从 A 完整继承并以 projection + gate（初始 0）注入；首阶段冻结 foundation，只训练 extension。训练前必须通过 baseline-equivalence test，之后才允许 bounded FAST 训练；只有出现 signal 才 progressive unfreeze。

ATTEMPT-1 compact evidence（仅作 protocol diagnosis）：A cold-cell/cold-drug row Spearman=`0.086005/0.052936`；B gain=`-0.004984/+0.032065`；C gain=`-0.029930/+0.002903`。Broad predicted macro-Spearman 也无一致改善（A/B/C cold-cell=`0.01361/0.00764/-0.00918`，cold-drug=`-0.01554/-0.02389/-0.03085`），因此不支持把单个 cold-drug B 增益升级为科学结论。

## Oracle Gate 与 Broad Top-K

Oracle 问题固定为：Observed LINCS Δ978 → CRC disease reversal → drug ranking → PRISM。Oracle candidate universe 允许与 XPert global universe 不同，以最大化 exact CRC context 与 canonical compound coverage；不得使用 PRISM response 做模型调参或 cohort 选择。

Broad primary metrics 改为 Top-K enrichment、Top-K lift over deterministic seeded random、HitRate@K、Recall@K、NDCG excess over null；Spearman/Kendall 仅 secondary。当前旧报告的 10-line/小 cohort 数字只能作为 feasibility diagnostic，不能外推到冻结 Broad cohort。Oracle 与 null-normalized metrics 正在 Specialist 轨道中实现/审计，尚无新的正式 Gate B 结论。

现已在 `xpert_broad.rank_metrics` 中加入 deterministic seeded random null（overlap/NDCG/Spearman distributions、Top-K lift、ΔNDCG），并加入 response-blind `audit_oracle_coverage`。这些是 evaluation capability，不是新的 PRISM scientific result；尚未运行全 Broad cohort。

决策规则：若 Oracle Top-K 明显高于 random，状态为 `REVERSAL_TO_EFFICACY_SUPPORTED`，后续瓶颈是 predicted→oracle gap；若 Oracle≈random，停止继续堆 perturbation architecture，优先检查 disease signature、reversal scoring、dose/time、context matching 与 PRISM endpoint。

## Cell-line Gate A/B/C

- Gate A：基本满足（Strong XPert cold-drug/cold-context signal）。
- Gate B：未完成；等待 Observed/Predicted reversal 的 null-normalized Broad Top-K evidence。
- Gate C：未完成；GDSC frozen external readiness 仍为 local freeze pending。

达到 Gate A + Gate B 后即可并行启动 Organoid FAST feasibility；Formal organoid claim 仍等待 Gate C。

## EXP-006 readiness/result

EXP-006 继续使用同一个 XPert response backbone。Genetic 与 chemical perturbagen 共享 Context Encoder、gene×perturbagen cross-attention 和 response head；chemical supervision 按 unique compounds 评估 100%/20%/10%。EXP-004 的 FAST 结果（10% chemical supervision gain `+0.00777`）仅为 `PROMISING_FAST`，尚未构成 formal transfer 结论。当前 EXP-006 为 `DATA_PARTIAL/PREPARATION_ONLY`：dataset/version/checksum、gene mapping、matched-control、group-atomic split/provenance bundle 与 GDSC freeze 尚未完整，故不得启动正式训练或读取 response；仅可做 readiness preparation。

## Organoid readiness

Metadata/adapter preparation 可并行推进但不阻塞 EXP-005/Oracle。迁移原则为“share mechanism, condition response”：冻结 UniMol/KPGT/UniPert 与大部分 response backbone，主要学习 organoid context adapter、context×perturbagen interaction 和 small residual response layer。预注册 0/very-low/low organoid chemical supervision 的 A/B/C 对照。

## 当前唯一最大瓶颈

`SCREENING_TRANSLATION_EVIDENCE`：目前缺少覆盖足够 exact CRC context/canonical compound 的 Observed Oracle 与 null-normalized Broad PRISM Top-K 结果。因此暂不能判断 bottleneck 在 perturbation prediction 还是 reversal→efficacy translation，也不能正式推进 Gate C 结论。

## Novel Knowledge Gained

本轮已确认的新增知识是 protocol-level：EXP-005 ATTEMPT-1 不能作为 additive architecture 的科学否定，必须实行 Foundation Fidelity Gate 与 gated additive initialization；同时项目的主评价 endpoint 从孤立全排序相关性转为 Oracle/Predicted 的 Top-K utility over null。KPGT/UniPert chemical 是否有增量、以及 Genetic→Chemical 是否在低化学数据下保留收益，仍是开放问题。

## Agent execution manifest

实际已启动 fresh-context child agents：

- `exp_005_model_engineer`（Model Engineer）：ATTEMPT-2 strong-foundation implementation；`in_progress`。
- `exp_005_oracle_evaluator`（Data/Evaluation Specialist）：Oracle coverage 与 null-normalized Top-K；`in_progress`。
- `exp_005_scientific_analyst`（Scientific/Evaluation Analyst）：endpoint/gate/瓶颈解释；`in_progress`。

当前 runtime 支持真实 child-agent spawning，但未提供隔离 worktree；三项任务通过不重叠 allowed paths 和只读限制协调。主要 evidence merge 后已新建 fresh Independent Reviewer；在其 verdict 前不宣称 formal reviewed result。

Independent Reviewer verdict=`INCONCLUSIVE`：ATTEMPT-2 尚未实际训练；现有 compact JSON 仍是 ATTEMPT-1 旧产物，尚无 checkpoint inheritance、gate=0 数值审计、null baseline 或 expanded Oracle 结果。因此本 checkpoint 只接受 protocol/code readiness，不接受 additive scientific claim。Reviewer 还要求未来运行显式分离 valid/test，并生成 A/B/C 同一 support 的 paired Oracle-vs-predicted digest。代码审计确认 runner 的 `--checkpoint` 会在 A/B/C model 构造后统一调用 `load_xpert_checkpoint`；但 A 的 metadata 仍需在 ATTEMPT-2 实跑时补齐，不能用当前旧 JSON 代替。

## 下一步自动推进路径

1. 完成 ATTEMPT-2 baseline-equivalence 与 frozen-foundation FAST；若 `SUPPORTED`，选择 chemical backbone champion；若 `NO_ADDITIVE_CHEMICAL_VALUE`，保留 XPert 并仍进入 Genetic→Chemical；若 protocol 再次失败，停止 architecture 解释并修复 protocol。
2. 完成 Observed Oracle cohort 与 Broad null-normalized Top-K；据此决定 Gate B 和 predicted→oracle gap。
3. Gate A+B 后并行启动 Organoid FAST；GDSC ready 后执行一次 frozen external check（Gate C）。
4. 每个 sprint 结束必须记录 `NOVEL KNOWLEDGE GAINED`，不得把测试通过、资产下载或 adapter ready 单独称为 Research Milestone。
