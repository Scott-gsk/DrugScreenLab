# DRUGSCREENLAB NOVEL EXTENSION PROGRESS REPORT

> **Latest update (2026-08-13):** Since this report's earlier E2 FAST checkpoint, the official XPert
> foundation has been integrated and EXP-005 A/B/C fixed FAST has completed. The latest decision is
> `NO_MATERIAL_FAST_INCREMENT`; see `DRUGSCREENLAB_XPERT_EXTENSION_REPORT.md` and
> `mvp/foundation/xpert/EXP005_FAST_COMPARISON.json`. The E2/EXP-004 numbers below are preserved and
> are not overwritten or reinterpreted.

日期：2026-08-13
Checkpoint：`c032a8b` 之后的 autonomous FAST extension 进度
运行环境：WSL2 + Conda `drugscreening-gpu`；CUDA 通过，3 张 GPU
EXP scope：`EXP-004`，单一主要假设：遗传扰动监督能否降低化学监督对 exact-978 Delta978 预测的需求。
证据级别：`FAST / PROMISING`，不是 formal multi-seed 结论。

## Executive decision

本 sprint 已产生新的定量科学结果，程序继续：`CONTINUE_WITH_EXTENSIONS`。

E2 在固定 chemical test manifest 上显示，遗传预训练再化学微调相对 chemical-only 的 chemical-test Spearman 增益为：100% 化学监督 `+0.00539`、20% `+0.00240`、10% `+0.00777`。10% 的方向准确率为 `0.5222`，chemical-only 为 `0.5186`。这支持 `GENETIC_TO_CHEMICAL_TRANSFER_FAST_PROMISING`，但由于单 seed、bounded train pool、没有 downstream ranking 翻译，不能升级为正式泛化结论。

## 1. FOUNDATION：已建立的基础，不在本 sprint 重新证明

| 项目 | 当前状态 | 解释 |
| --- | --- | --- |
| exact-978 GSE92742、matched control、split、Delta978 endpoint | `FROZEN` | 未改变定义、train/validation/test 角色或外部标签 firewall |
| Context + chemical backbone | `SUPPORTED_FOUNDATION` | 既有 Phase-1 结果继续作为基础 |
| UniPert chemical replacement | `UNIPERT_REPLACEMENT_NO_SIGNAL` | 不能外推为 UniPert 整体失败 |
| KPGT+UniPert 历史结果 | `VERIFIED_PRIOR_INTERNAL_EVIDENCE` | 指定 DPR_release checkpoint 与 branch 历史记录可核验，支持 additive foundation claim |
| 当前 KPGT 可复用资产 | `WAITING_FOR_ASSET_RECOVERY` | 两个本地资产已复制并校验，但指定 Git tree 不含 pickle，sidecar 标注 KPGT source/license 为 unverified/review_required |

KPGT 本地 recovery 资产只复制、不修改原工作区：

- `data/external/kpgt_recovered/KPGT_emb2304.pickle`，77,663,320 bytes，SHA256 `61b2a1337c8313ecbcb95e9f797ea12caeb5d94fefc2697af810c5955ae962f4`。
- `data/external/kpgt_recovered/lincs_kpgt_unipert_emb2560.pickle`，87,558,085 bytes，SHA256 `0cfcede41b96d0906a99b2233b5144fba51b22f638134451da159eb7e52e230c`。

这两个事实必须分开：历史研究结果已验证；当前资产是否满足可复用 provenance contract 仍待审查。KPGT 是 E1 的 soft blocker，不阻断 E2、PRISM identity freeze、GDSC access audit 或 organoid preparation。

## 2. NOVEL EXTENSION：E2 Genetic→Chemical low-data transfer

### Data contract

遗传 cohort 使用 GSE92742 `trt_sh@96h`，选择至少 100 条记录、至少 2 个细胞背景、同时存在本地 UniPert gene reference 的 256 个基因；共 70,035 个 response-blind 候选记录、16 个细胞背景。最终 manifest 保留 67,951 条、2,647 个 group，其中训练池为 2,131 groups / 54,861 records。

控制优先同 `rna_plate + cell_id + time` 的 `ctl_vector`，本轮实际控制计数为 `ctl_vector=67,951`，未使用外部表型。遗传方向显式记录为 `knockdown`，没有把 genetic perturbagen 折叠成匿名 chemical embedding。

UniPert genetic feature：256×256，256 个基因全部编码成功、0 invalid gene、2,809 个 genetic perturbagen IDs 映射，模型 SHA256 `06b065c2768b440077bdb1488f654b310f67a04c01fceb9f4791043558bfd322`。

### Model and endpoint

实现了统一低容量接口：

`Context + Perturbagen + Perturbation Type/Direction + Dose/Time → Shared Response Model → Delta978`

比较：

1. Chemical-only；
2. Genetic pretraining → Chemical fine-tuning。

chemical test 完全冻结为 `artifacts/phase2/fast_unipert/random_group/manifest.json`，SHA256 `27b69fb3c3cc9f7cd57e62edb060c4537256c310dc242860805d83b2b7221a90`。归一化只使用各化学训练子集；没有用 test label、PRISM/GDSC response 或外部标签选模型。

### Quantitative result

| 化学监督比例 | Chemical-only Spearman | Genetic→Chemical Spearman | 增益 | Chemical-only direction accuracy | Genetic→Chemical direction accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100% | 0.06616 | 0.07154 | **+0.00539** | 0.5191 | 0.5206 |
| 20% | 0.06952 | 0.07192 | **+0.00240** | 0.5203 | 0.5217 |
| 10% | 0.06624 | 0.07401 | **+0.00777** | 0.5186 | 0.5222 |

判定：`PROMISING_FAST`。增益在 10% 比 20% 更清晰，但不是随监督比例单调变化；这是需要多 seed 和 downstream translation 的信号，而不是已完成的正式 claim。

结果 artifact：`mvp/extension/E2_GENETIC_CHEMICAL_TRANSFER_FAST_RESULT.json`。实现与测试：`src/drug_screen/modeling/genetic_transfer.py`、`scripts/modeling/run_e2_genetic_transfer.py`、`tests/test_genetic_transfer.py`；新增单元测试 4/4 通过。

### What is genuinely new

相对于 XPert/MultiDCP/UniPert 已有 foundation，本轮新信息不是“又换了一个 chemical encoder”，而是：在同一个 response head 中保留 genetic modality/direction，并在固定 chemical test 上观察到低化学监督 regime 的迁移增益。该结果把 `genetic→chemical transfer` 从接口假设推进到可量化 FAST evidence；尚未证明它优于所有现有 model，也尚未证明 downstream screening utility 已改善。

## 3. NOVEL EXTENSION：Broad PRISM cohort freeze

本轮只冻结 identity/context 层，没有读取 PRISM response values。

| 层 | 数量 | 解释 |
| --- | ---: | --- |
| PRISM unique Broad treatment identities | 4,686 | metadata inventory |
| formal exact-identity Broad rows | 1,931 | exact `pert_id` / InChIKey identity |
| formal eligible Broad base IDs | **1,851** | `BROAD_PRISM_COHORT_V1` |
| STR-pass cancer lines with tissue metadata | **568** | unique `depmap_id` |
| STR-pass colorectal lines | **35** | `CRC_V1` |

排除 alias-only、ambiguous alias 和 unmatched identity。冻结审计：`mvp/extension/BROAD_PRISM_COHORT_V1.json`；桥接审计：`mvp/extension/LINCS_PRISM_IDENTITY_BRIDGE_AUDIT.json`。

回答“predicted reversal 是否仍保留 screening signal”：本 sprint 尚未在扩展 cohort 读取 response 或运行 predicted-Δ978→reversal→ranking；既有四药 feasibility signal 保留为 `PROMISING_FAST`，不能外推至 1,851-drug cohort。下一步必须在 mapping digest 冻结后，独立读取 observed LINCS oracle 与 PRISM response，报告 per-line Spearman/Kendall、NDCG@10/20、候选分母和 oracle→predicted degradation；没有预注册 binary label 时不报告 AUROC/AUPRC。

## 4. INFRASTRUCTURE BLOCKER：GDSC、mechanism、organoid

### GDSC

官方 Cell Model Passports / Sanger 入口当前可达，GDSC1/2 raw、fitted 与 compound annotation 文件均可通过官方端点获得。因此当前状态不是 `TEMPORARY_EXTERNAL_ASSET_ACCESS_FAILURE`，也不是科学 negative result；正确状态是：`OFFICIAL_ASSET_REACHABLE_LOCAL_FREEZE_PENDING`。

在完成实际下载前不读取 response、不冻结 release/checksum、不运行 GDSC benchmark。官方字段中的 AUC、LN_IC50、Z_SCORE 与 nominal Target/Target Pathway 不能未经 contract 选择就当作 ground truth；官方说明见 [Sanger drug-sensitivity documentation](https://depmap.sanger.ac.uk/documentation/datasets/drug-sensitivity/) 与 [Cell Model Passports changelog](https://cellmodelpassports.sanger.ac.uk/changes)。

### Target / Pathway

状态：`FORWARD_PREPARATION_COMPLETE / NOT_RUN_COVERAGE_INSUFFICIENT`。

当前四药 forward coverage 为约 3/4：BMS-777607→MET，PD-0325901/Trametinib→MEK1/2；BMS-299897 尚无机制行。该四药覆盖不足以验证 target/pathway prior。下一步应扩展 LINCS chemical universe，冻结 target/action/affinity/confidence→STRING→Reactome/GO prior，再只在 cold-drug/low-data/transfer split 比较增量。

### Organoid

状态：`METADATA_READY_ADAPTATION_PENDING`。

- `GSE280506`：primary human gastric organoid、CRISPRi/CRISPRa、DMSO/cisplatin、single-cell CROP-seq；适合作为 genetic/context adaptation reference。
- `GSE145308`：intestinal organoid、WT/APC/ARID1A/SMARCA4、0h/24h、三重复；必须显式处理 GPL20301/GPL24676 platform/batch。

两者都不是 chemical sensitivity ground-truth benchmark。尚需 accession-specific gene universe、gene harmonization、single-cell→context aggregation、control/response contract 与 held-out donor；不能把 organoid transcriptomic response 直接写成 drug ranking evidence。

## 5. 状态分层与下一决策

### FOUNDATION

- exact-978、matched-control、split、endpoint 和 leakage firewall：`SUPPORTED / FROZEN`。
- KPGT+UniPert 历史增益：`VERIFIED_PRIOR_INTERNAL_EVIDENCE`。
- KPGT 当前复用：`WAITING_FOR_ASSET_RECOVERY`（资产字节已恢复，provenance review 未完成）。

### NOVEL EXTENSION

- E2 genetic→chemical low-data：`PROMISING_FAST`，10% chemical supervision Spearman gain `+0.00777`。
- Broad PRISM：`FROZEN_IDENTITY_CONTEXT_COHORT`，1,851 base IDs / 568 cancer lines / 35 CRC lines。
- predicted reversal on expanded cohort：`NOT_RUN_AFTER_FREEZE`，不得把旧四药结果冒充扩展结果。

### INFRASTRUCTURE BLOCKER

- GDSC：官方可达，local version/checksum freeze pending；不是 science block。
- KPGT：recovered bytes available，source/license/extraction provenance review pending；是 E1 soft blocker，不阻断 E2/PRISM/GDSC access/organoid preparation。
- Organoid full benchmark：metadata ready，adapter pending；不是当前 E2 gate。

Expected Decision Value：E2 FAST 若低数据增益稳定，可决定是否投入多 seed、cold-gene/cold-context 和 downstream ranking；PRISM freeze 可把下一轮从四药 feasibility 提升到 identity-defined screening cohort。Expected Cost：当前结果约 12 分钟 wall-clock；下一轮只在信号复核后增加 seeds 和 downstream response I/O，避免无理由 full scan。

## 6. Governance / reproducibility

本轮实际 spawned 的 sidecar：

- `EXP-004 Data/Bioinformatics Steward`：KPGT asset/repository audit，完成；只读。
- `EXP-004 Evaluation & Statistics Analyst`：PRISM response-blind cohort audit，完成；只读。
- `EXP-004 Scientific Analyst`：GDSC official access、mechanism、organoid audit，完成；只读。

当前 runtime 支持独立 child agent，但未提供可验证的 isolated worktree 参数；因此 sidecar 均限制为只读，代码和正式 metadata 由 Manager execution 完成。Independent Reviewer 未启动：这是 FAST probe，按 coarse-to-fine policy 仅在下一轮 signal-to-downstream milestone 后启动。

所有实验与测试均使用 WSL2 `drugscreening-gpu`；原始数据未修改；未使用 MCPIRE_PDO/TriPerturb；未使用 PRISM/GDSC response 调参；大型 feature/cache/prediction 仍保持 local ignored，tracked metadata 记录路径、版本与 checksum。
