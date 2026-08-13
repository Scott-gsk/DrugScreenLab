# EXP-006 Data Steward：最小 FAST DATA CONTRACT（fresh context）

日期：2026-08-13；角色：Data & Bioinformatics Steward；唯一实验：EXP-006。

## 当前数据状态

**DATA_PARTIAL**。现有 LINCS/XPert 与 response-blind genetic 资产可支持下游接口开发；CCLE/DepMap RNA-seq→978 的实际输入、精确 cell-line crosswalk，以及 organoid 基因 harmonization/split 尚未登记，因此不得宣称 DATA_READY，也不得读取外部 efficacy label。

## 已冻结 contract（不改变结果定义）

- **Gene universe/output**：GSE92742 Level-3 `pr_is_lm=1` 的 ordered exact-978；gene-list SHA256 `b4e2fca877c5cfdcc1c712ad0fd67e97a88b6f7566b013e4bab065f699ebb623`。目标仅为 `Delta978 = matched_treatment - matched_control`，float32、shape `(*,978)`；不得补 inferred genes。
- **Identity/schema**：每条记录必须保留 `dataset_id, dataset_version/accession, source_record_id, sample_id, treatment_group_id, modality, perturbagen_id, gene_symbol, perturbation_direction, context_id, context_source, depmap_id (若有), dose_value/unit, time_value/unit, treatment_record_id, control_record_id, control_match_key, gene_universe_digest, split, split_entity_id, normalization_digest, provenance/checksum`。
- **Matched control**：优先同一 `rna_plate || cell_id || pert_time || pert_time_unit` 的 `ctl_vector`，否则 `ctl_untrt`；treatment/control 必须同 context、plate、time、platform。无法证明即 `DATA_BLOCKED`。
- **Genetic identity**：`pert_id + standardized gene_symbol + perturbation_direction`；当前 GSE92742 `trt_sh`、96h 条件仅作 genetic pretraining，不可伪装为 chemical dose。UniPert features response-blind，256 genes、256d。
- **Chemical identity/split**：化合物用登记 `pert_id`/Broad exact bridge；沿用已冻结 compound-level manifests（EXP-005 `split_cold_drug_1_*`，及 random_group/cold_context 仅在其声明角色内）。group/identity atomic，treatment 与 matched control 同 split；禁止 row-random split、跨 split identity、external-test contamination。
- **Normalization/context adapter**：输入 CCLE/DepMap 或 organoid RNA-seq 时，先记录原始单位与 gene-ID namespace；按显式 crosswalk 映射到 exact-978，重复 gene 聚合规则、训练子集拟合的参数及 digest 必须登记。仅允许 `context_id` 的 exact crosswalk（DepMap ID 为主键，CCLE 名称仅辅助）。XPert context 仅消费其 registry 中 exact-control 语义；外部 basal RNA 不能直接当 treatment/control。
- **Forbidden**：PRISM/GDSC response、held-out Delta978/response、EXP-004/005 predictions/checkpoints/logs 不得用于 gene mapping、normalization、feature、split 或 selection；raw data 不可修改。

## 现有可核查资产

1. `lincs_gse92742_raw_level3_v1`（GSE92742，raw immutable；registry source SHA256 `43bbea...9edf`，raw file SHA512 已登记）。
2. `lincs_gse92742_exact978_cache_v1`：`data/processed/lincs/GSE92742/exact978_cache_v1/exact978_cache.npy`，shape `[1319138,978]`，SHA256 `04b8bb746a61ba4992e49566315327023783ec1c0448da2a9e263e0881281733`；EXP-002 contract digest `900592f...e9c`。
3. `unipert_genetic_features_256_v1`：GSE92742 `trt_sh` 96h，256 genes/256d，`mapped_genetic_perturbagen_ids=2809`，feature SHA256 `3a9586...e8045`，mapping SHA256 `da56c7...5037`；候选 70,035 rows、16 cells、response-blind。
4. E2 FAST genetic pool：2,131 groups/54,861 records；chemical train cap 4,000 groups/15,879 records；frozen chemical test 5,667 groups（manifest SHA256 `27b69f...221a90`）。
5. XPert registries：55 exact LINCS contexts、10 Broad exact contexts；8,418 official perturbagens（1,836 Broad eligible）；global adapter 18,360 records，均为 metadata/response-blind 输入。
6. Organoid metadata：GSE280506（4 samples，gastric CRISPRi/a，scRNA-seq）及 GSE145308（24 samples，intestinal WT/APC/ARID1A/SMARCA4，0/24h，3 replicates）为 `METADATA_READY`；当前审计明确 gene/platform contract 与 genetic→chemical adapter 尚待完成，不能作 chemical ground truth。

## CCLE/DepMap 与 organoid 接入门槛

- 仓库未发现已登记 CCLE/DepMap RNA-seq 矩阵或 DepMap-ID crosswalk；因此 context adapter 目前 **PENDING**，不得下游训练。
- 需新增 non-raw manifest（dataset/version/accession、源文件 checksum、generator revision、gene mapping 表、DepMap crosswalk、normalization 参数 digest、split manifest）后再升级状态。
- organoid 仅作 context/mechanism adaptation reference；必须先完成平台分层、样本/供体分组、gene-ID→exact978 映射与 matched-control 证明。缺任何一项则 `DATA_BLOCKED`。

## 可复现验证命令（WSL2）

```bash
PYTHONPATH=src conda run -n drugscreening-gpu python -m drug_screen.data.registry --root data
PYTHONPATH=src conda run -n drugscreening-gpu python -m pytest --capture=no tests/data
```

本会话 Windows PowerShell/WSL 未提供可用 `conda`（`conda: command not found`），故未执行命令；不以 Windows Python 替代。上述命令须在合规 WSL2 `drugscreening-gpu` 环境重跑并把 stdout/checksum 写入 EXP-006 manifest。

