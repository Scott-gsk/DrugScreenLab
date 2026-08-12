# MVP-001 数据契约：GSE117548 / Betge 2022 外部形态活性 guardrail

状态：`DEFERRED_PDO_LEG`（identity 与 auroc direction 已冻结；本分支未进入 MVP-001 Cell-Line Core 的最终 label comparison）。

本契约仅冻结 MVP-001 的 `GSE117548` 外部比较数据；它不是 Formal EXP 的临床/疗效数据声明。

## 身份与溯源

| 项目 | 冻结值 |
| --- | --- |
| 研究身份 | Betge et al., *Nature Communications* 13, 3135 (2022), DOI `10.1038/s41467-022-30722-9` |
| 表达 accession | GEO `GSE117548`，GPL570；GEO 记录为 16 个 human CRC PDO lines、25 array samples |
| 外部表 | `data/processed/betge_2022/external_drug_activity.parquet`，66 rows，SHA-256 `c61d7d27838fcda5c923203b351c7f1e84c04be28be02fb98e982ce1333d9c5d` |
| 作者的 compound annotation | `data/raw/betge_2022/Supp_BetgeRindtorff_2021/references/layouts/Compound_Annotation_Libraries_New.xlsx`，SHA-256 `8856909d82eda387e94f7f93527204118d6f83de82bf2a1d77c110db9965b584`；只读审计，不得修改 |
| GEO series matrix 本地副本 | `data/raw/geo/GSE117548/GSE117548_series_matrix.txt.gz`，SHA-256 `59783d2591af0b2613feddd4e934e127a910d7cc225b5dc4bd118b45d49e07c1`；只读，不作为本任务的计算输入 |
| data role | `EXTERNAL_TEST` only；禁止 training、tuning、feature selection、threshold fitting、model selection 或 label 重定义 |

主要一手来源：

- 论文 Methods 将 activity 定义为逐 `organoid line x drug`、药物处理相对于 untreated/DMSO 对照的逻辑回归分类器 AUROC；其范围为 0.5--1，`>0.85` 被作者定义为 active。[Betge et al.](https://www.nature.com/articles/s41467-022-30722-9)
- GEO 指明 GSE117548 是人 CRC PDO 的 array expression study，整体为 16 PDO lines / 25 samples，并列出 PDO line identity。[GEO GSE117548](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE117548)
- 作者官方可复现仓库提供 compound annotation 和分析材料。[Supp_BetgeRindtorff_2021](https://github.com/boutroslab/Supp_BetgeRindtorff_2021)

## 标签定义与不可翻转方向

`auroc` 是**形态表型扰动可分性**：较高值表示该药物处理后的 organoid morphology 更容易被分类器同 DMSO/negative-control morphology 区分；较低值表示较不可分。冻结比较方向为：

```text
higher auroc = stronger phenotype activity / separability
lower auroc  = weaker phenotype activity / separability
```

它**不是** CellTiter-Glo viability AUC、DSS、IC50、临床获益，也不是可标为 "sensitivity" 或 "resistance" 的 ground truth。任何结果不得事后对 `auroc` 取负、改为 `1-auroc`，或根据 prediction 的相关符号更换高低方向。若 downstream endpoint 需要 efficacy/sensitivity/resistance，状态立即为 `DATA_BLOCKED`，须改用有匹配 viability endpoint 的已批准数据契约。

## 冻结 schema

外部 label 的唯一键是 `(line, drug, factor)`；`factor` 是作者 MOFA factor selection context，**不得**作为训练特征、模型目标或筛选条件。可用于最终 join 的字段如下：

| 字段 | 用途 | 约束 |
| --- | --- | --- |
| `id` | PDO donor-like base ID，例如 `D004` | 必须等于预处理后的 GSE sample line base ID；不以 sample date 推断 |
| `line` | assay line，例如 `D004T01` | 仅允许下述 T01 whitelist；不得以 `D020T02` 替换/合并 |
| `drug` | 作者 display compound name | 与作者 compound annotation 字符串精确核对 |
| `drug_name_normalized` | 追溯性 normalized alias | 仅辅助查找；不能单独作为 identity proof |
| `auroc` | 冻结的 external phenotypic-activity label | label blind until ranking digest is committed/frozen |
| `factor` | source context (`factor1` / `factor2`) | join key 的组成部分；分别报告，不池化为独立 observations |

`factor1`--`factor3`、`mcrc_compound_name`、`mcrc_dss_mean`、`mcrc_n_patients` 均为 **FORBIDDEN**：不得进入 MVP-001 model、ranking、外部 comparison、mapping 决策或结果解释。后 3 字段来自不属于本 contract 的外部 cohort，使用会造成 external-test contamination；且本表 44/66 行为空，不能作为填补依据。

## 可用药物身份与剂量/时间边界

作者 compound annotation 证明以下五种活性表的 display-name、target/pathway、KiStem 库及单剂量 `7.5 uM`。本地 compact activity table 不含处理时长；不得臆测 dose/time，也不得同 LINCS dose/time 等同。

| source display name | normalized alias | source target | source pathway | source library | concentration |
| --- | --- | --- | --- | --- | --- |
| `BMS-536924` | `bms536924` | IGF-1R | Protein Tyrosine Kinase | KiStem | 7.5 uM |
| `OSI-420` | `osi420` | EGFR | Protein Tyrosine Kinase | KiStem | 7.5 uM |
| `PRI-724` | `pri724` | Wnt/beta-catenin | Stem Cells & Wnt | KiStem | 7.5 uM |
| `Trametinib (GSK1120212)` | `trametinib` | MEK | MAPK | KiStem | 7.5 uM |
| `Ulixertinib (BVD-523, VRT752271)` | `ulixertinib` | ERK | MAPK | KiStem | 7.5 uM |

在 final comparison 前，每种实际 retained drug 必须由一个新建、tracked mapping table 同时提供：本表 source display name、`drug_name_normalized`、LINCS `pert_id`/canonical compound name、稳定结构标识（InChIKey 或等价的可审计 ID）、mapping source/version 和 mapping decision。没有这一组字段的药物 `INELIGIBLE`；不能使用 target/pathway、名称相似度或结果相关性补足身份。

## PDO identity、matched control 与 split

eligible external cohort 固定为：`D004T01`, `D007T01`, `D010T01`, `D013T01`, `D018T01`, `D019T01`, `D020T01`, `D022T01`, `D027T01`, `D030T01`, `D046T01`。每条 line 具有五个 source drug 和两个 `factor` labels；不得将同一 line 的两个 factor 视为独立 PDO。

- source primary-paper evidence：13 lines 被 imaging，`D015T`、`D021T` 因 out-of-focus exclusion；`D020T` 有 `T01` 和 `T02` 两 batch，若未另行指明使用 `T01`。本 MVP contract 因而只用存在完整 external activity 行的上述 11 条 `T01` lines。
- expression identity：GEO sample name 的 base PDO ID 与 `id` 作 one-to-one join；25 GEO samples 是 technical/timepoint profile observations，必须先按预先固定的 expression aggregation rule collapse 到 PDO base ID，不能根据 activity label 选择 sample、日期或 replicate。
- matched control：外部 `auroc` 的 DMSO/negative-control matching 属于作者已计算 endpoint；MVP 不重新估计、校准或更换其 controls。PDO state-deviation proxy 的 reference cohort 必须 leave-one-PDO-out，且不得把该 PDO 的 external label、factor 或 drug activity 输入 reference construction。
- split/leakage：GSE117548 的任何 expression、activity、MOFA factor、source target/pathway 和 mcrc 字段均不得出现在 training、validation、early stopping、candidate selection 或 hyperparameter decision。最终 comparison 前必须写入 immutable ranking digest；join 后只能计算预声明指标。

## 放行与失败 gate

`DATA_PARTIAL` 的原因是 source identity 和 `auroc` direction 已证实，但五药均尚未提供至 LINCS 的稳定 compound identity mapping，且 compact table 的原始生成命令/provenance 不足以从作者 source data 独立重建。最终 label comparison 仅在以下全部满足后放行：

1. mapping table 对每个 retained drug 通过 display-name + canonical LINCS ID + stable structure ID 的审计；
2. 至少 3 种 retained drugs，且每条 eligible PDO 均有这些药物的完整标签；
3. 预先冻结 expression aggregation、leave-one-PDO-out state reference、candidate set、ranking digest 和 comparison metric；
4. activity labels 在上述 ranking digest 冻结前未被读取给 model/evaluation selection。

任一条件失败的唯一处理是 `DATA_BLOCKED`（对 final external comparison）；不得靠 alias 猜测、放宽 identity，或把 mcrc / morphology / factor 字段带入以维持候选数。
