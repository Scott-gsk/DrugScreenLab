# DRUGSCREENLAB PROGRAM DIRECTION CHECKPOINT V2

更新：2026-08-13

## Program decision

当前最大瓶颈已从 representation architecture 转为 `predicted Δ978 → disease reversal → efficacy Top-K` translation。XPert chemical fusion 关闭；XPert Foundation 保留为 Chemical Backbone Champion。EXP-006 Genetic→Chemical 进入主研究方向，Context/Organoid 继续前向准备。

## EXP-005 final strong-XPert comparison

ATTEMPT-2 已改为对 XPert 既有 HG/global compound representation 做 gated residual fusion，不再增加 token 或序列长度。Strong checkpoint `l1000_sdst_warm_split.pth` 加载 `182/182` official keys；A 直接推理、不训练；B/C 从同一 checkpoint 出发，仅训练新增 projection/gate。

| run | split/budget | row Spearman | 解释 |
|---|---|---:|---|
| A | cold-cell, 256-row, no-train | 0.63587 | strong-checkpoint dry-run |
| B | cold-cell, 64-row, 1 epoch | 0.6328307 | bounded protocol validation |
| C | cold-cell, 64-row, 1 epoch | 0.6328390 | bounded protocol validation |

这些预算不对称，因此不是正式 OOD effect size；没有显示值得继续 chemical architecture search 的增量。B 的独立 gate metadata run 显示 `pre_training_gate_init=[0.0]`、`baseline_equivalence.exact=true`、`max_abs_delta=0.0`、`official_parameters_frozen=true`。Chemical Backbone Champion=`XPert Foundation`。EXP-005 chemical fusion 到此关闭；不再继续 KPGT/UniPert chemical fusion search。

## Screening Gate: Observed vs Predicted

使用现有 Broad PRISM response、XPert predicted profile 和 observed LINCS exact978 Oracle，计算 Top-K/null-normalized metrics，并在相同 line×drug support 上配对比较。

| evaluation | support | Spearman | Top10 overlap / random | NDCG@10 / ΔNDCG |
|---|---:|---:|---:|---:|
| Predicted reversal | 18,179 rows / 10 lines / 1,836 drugs | -0.02736（null -0.00099） | 0.0800 / 0.00539 = 16.07× | 0.72320 / +0.22240 |
| Observed Oracle | 839 pairs / 2 lines | +0.15389（null -0.00647） | 0.3000 / 0.22012 = 4.58× | 0.65940 / +0.09792 |

Oracle 的 Top-K signal 是正的，但 coverage 只有 `839/62,248=1.35%` PRISM pair identities、`2/10` lines，不能外推成 Broad Gate B 完成。Predicted reversal 的 pooled Top-K/NDCG 被 7/10 lines 的负 Spearman 掩盖，不能解释为已支持 reversal-to-efficacy。

同一 839 个 line×drug pairs 的 predicted−oracle gap：mean `+0.07649`、MAE `0.07659`、RMSE `0.08319`、Pearson `0.11382`、Spearman `0.05287`。单次 CMap-style weighted-KS fallback macro Spearman=`-0.01169`，没有挽救 predicted screening signal。结论：`ORACLE_POSITIVE_PREDICTED_TRANSLATION_NEGATIVE`；下一步应缩小 predicted→oracle gap并扩大 Oracle exact support，而不是堆新的 reversal 网络。

## EXP-006 FAST

现有 FAST contract 已满足进入探索：exact978 cache `1,319,138×978`；genetic `2,131 groups/54,861 records`，256 UniPert genetic features，2,809 mapped IDs，70,035 candidate rows/16 cells；chemical train `4,000 groups/15,879 records`；frozen test manifest SHA256=`27b69fb3c3cc9f7cd57e62edb060c4537256c310dc242860805d83b2b7221a90`。

existing `E2_GENETIC_CHEMICAL_TRANSFER_FAST_RESULT.json`：genetic-pretrain − chemical-only chemical-test Spearman gain 为 100% `+0.0053867`、20% `+0.0023999`、10% `+0.0077657`。状态=`PROMISING_FAST`，单 seed/bounded，不是 formal claim。经理 fresh rerun 超出 wall-clock/I/O budget，未替换既有结果；下一步直接把 20%/10% transfer 接入 Top-K utility。

## Context / Organoid readiness

CCLE/DepMap RNA-seq 本地/registry asset 数量为 0；现有 55 LINCS contexts 中仅 10 个有 exact `context_id→depmap_id` bridge，45 个无 mapping。尚无 RNA→978 crosswalk、platform normalization 参数、lineage/split manifest 或 checksum bundle。Organoid/PDO 现有资产仅 metadata/readiness，不能直接塞入 XPert，也不阻塞 EXP-005/006 FAST。

## Gates

- Gate A Strong perturbation model：`PASS`（XPert Foundation checkpoint、finite prediction、cold split evidence）。
- Gate B Reversal→PRISM Top-K：`INCONCLUSIVE`；Observed Oracle positive but sparse, predicted translation negative.
- Gate C Frozen external GDSC：readiness pending。

## Novel Knowledge Gained

1. 在 strong XPert 上，gated residual fusion 可以保持 gate=0 的 exact baseline，并在 protocol dry-run 中不显示 chemical additive signal；因此 chemical backbone champion 仍是 XPert。
2. Observed LINCS Oracle 在有限 exact support 上有 Top-K enrichment，但 predicted reversal 没有同方向的 broad utility；paired predicted→oracle rank concordance 仅 `0.05287`，明确了当前 translation bottleneck。
3. Genetic pretraining 在 10% chemical supervision 的既有 FAST contract 上显示 `+0.0077657` Spearman gain，成为比 chemical fusion 更有决策价值的主线。

## Automatic next path

1. EXP-006 继续 20%/10% low-chemical FAST，并直接连接 reversal→PRISM Top-K。
2. 扩大 Observed Oracle 到更多 exact CRC line×drug support；保持 deterministic null metrics 和 paired gap。
3. 获取/登记 CCLE/DepMap RNA-seq release、gene crosswalk、train-only normalization、identity/split/checksum 后，运行 context adapter feasibility。
4. Gate A+B 后并行 Organoid FAST；GDSC ready 后执行一次 frozen external check。
