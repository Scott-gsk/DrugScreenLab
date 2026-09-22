# 数据合同草稿（工作包 B）— B1 字段逐数据集

- 状态：DRAFT（2026-09-21）
- 范围：本文件按方案 B1 字段（身份键 / 来源版本与 checksum / 登记状态 / 标签语义（方向·单位）/ 准入集合 / 训练·验证·测试·审计角色 / 禁止用途）逐数据集登记。
- 数字来源：一律引用 `data/registry/datasets.json` 与 `data/registry/datasets_transfer.json`（以及其引用的 `transfer_cohorts.json`）中现成条目；找不到的字段写「未登记」，不臆造。
- 说明：`datasets.json` 内 `path.relative` 相对 `data/`；`transfer_cohorts.json` 内资产路径带 `data/` 前缀，两者口径不同，引用时以原条目为准。

---

## 1. GDSC

- 身份键：细胞系 `sample_id` / `cell_line_id`（`transfer_cohorts.json#gdsc2_cell_line`）；药物 `drug_id`（InChIKey）＋ `canonical_smiles`（`transfer_cohorts.json` assets.drugs / `gdsc_kryeziu_drug_overlap_DRAFT.json`）。
- 来源版本与 checksum：GDSC2；`gdsc2_cell_line` 资产 sha256 —— samples `1074b62300d62a65f9ea2aa8a8ab0d28225c2acda47aeb21567546d07b6d61d6`、drugs `3cf9aa70f990b617578edfe61e9f1fd955dcef4ca01a6c84566520d340102148`、pairs `8b499f9b667be3a22aaf509c4b70d7af73a729a8380163384bee6cd7843c3081`、expression `6aa19a689104b77009b9ed35c1a607eebfe169b5e003844e8ff853ff3acd3083`；source_index sha256 `1d643999d40b9ad74ad4de26d361c2f1b04e4009a7827283a285524ea73a2b03`。
- 登记状态：`transfer_cohorts.json` 已登记（UNFROZEN，`status` 未标注正式冻结）；`reason`：legacy 开发数据，hash 校验不代表独立验证。
- 标签语义：表达 `log2(CMP TPM+1)`（`expression_unit`）；响应 `1_minus_auc`，`higher_is_sensitive=true`（高＝敏感）。
- 准入集合：`gdsc2_cell_line`（963 细胞系，`gdsc_cellline_splits_v1.json` 按 `cell_line_id` 身份 GroupKFold 留出，见工作包 B 任务 3 产物）；`original_source_audit: required`、`external_test_eligible: false`。
- 训练·验证·测试·审计角色：SOURCE_TRAIN_AND_COLD_CELL_EVALUATION（源域训练 + 冷细胞评价）；未登记独立外部测试资格。
- 禁止用途：不得作为外部独立验证集；不得视为新近独立外部测试。

## 2. PRISM

- 身份键：细胞系 `depmap_id`（`prism_repurposing_primary_19q4_compact_crc_v1` schema）；化合物 `pert_id` / `canonical_smiles` / `InChIKey`（`lincs_prism_identity_bridge_v1`）。
- 来源版本与 checksum：PRISM_REPURPOSING_PRIMARY_19Q4；compact 产物 sha256 `5114e789299a794792bc44c818af98e679e8f00d6f6f3fb246781e6df2e960f1`；`xpert_broad_prism_crc_response_v1` output_sha256 `e72020f081a60006d4c97e17fbb9fc91bfb544ca4cc145d4460ec25727e9448c`。
- 登记状态：`datasets.json` 已登记（MVP 外部测试用途）。
- 标签语义：响应「official log2 fold-change relative to DMSO」，`sensitivity_score = -response_raw`（负 fold-change 越负越敏感；方向未按 `higher_is_sensitive` 布尔字段登记，沿用上述公式）。
- 准入集合：colorectal_lines 35（`schema.colorectal_lines`）；`rows 135`、`candidate_count 4`。
- 训练·验证·测试·审计角色：MVP-001_EXTERNAL_TEST（外部测试）；`lincs_prism_identity_bridge_v1` 为 IDENTITY_ONLY_PRE_EVALUATION；`xpert_broad_prism_crc_response_v1` 为外部评价响应 compact，`response_blind=true`。
- 禁止用途：不得充作训练集；身份桥接为「精确身份 only」，别名/歧义行不得进入正式队列。

## 3. Kryzei（mCRC PDO）

- 身份键：样本 `sample_id`（`PtNN_tK`）、患者 `patient_id`（`PtNN`）、药物 `drug_id`（InChIKey，48 药）；`kryeziu_drug_identity_48.json` 为冻结身份表（`frozen-1.0`，含 `canonical_isomeric_smiles`）。
- 来源版本与 checksum：Mendeley `hr94h42xdc.3`（CC BY 4.0）；processed 资产 sha256（manifest `output_sha256`）—— drugs `04612cb03ba3d1cbb78a219a53193ab8810af9797c326d8f5593759c847e7515`、pairs `5254c5a21718abea0b4454bf87b597cbafd7b598963390f80a0407584e293831`、samples `6d2a2fb26c0716dbf0236157967677e01473ab3f1dacb5bad866b5a4b37d53e7`、indexed `ddbab601e461169a7385ac0658a433a7735581e5e9b99bdfd062e9bec5f4c727`；身份审计 sha256 `244c159fad55eea3738fa6e7e28be220e084cc00151a2118be31142e58dc08d2`。
- 登记状态：`datasets.json#mcrc_pdo_2026`（v3）+ `kryeziu_patient_outer5fold_frozen_v1` + `kryeziu_patient_identity_frozen_v1` + `kryeziu_drug_identity_48_frozen_v1` + `kryeziu_mcrc_target_admission_v1`（DEVELOPMENT_ADMITTED，CONDITIONAL）。
- 标签语义：响应 `DSS`，`higher_is_sensitive=true`（高＝敏感）；表达 `log2(CPM+1)`（19421 基因）。
- 准入集合：S1 验证 51 患者 / 60 样本（`S1_PDO_patient_verified`）；32 prefix-only 样本排除（`pending_G0_identity_verification`）；Oxaliplatin 隔离。详见 `kryzei_admission_audit_20260921.json`。
- 训练·验证·测试·审计角色：WITHIN_STUDY_PATIENT_GROUPED_PDO_DRUG_RANKING；外层 5 折（patient 整组、inner val 已物化，seed 20260918）；审计 `kryeziu_identity_assay_audit.json`。
- 禁止用途：禁止全 52 组形式建模/评价；禁止外部测试或临床疗效结论；作者 DSS 保留，不得以原始曲线重拟合替代标签。

## 4. Zhu（肝胆 PDO）

- 身份键：样本/细胞 `sample_id`、药物 `drug_id`（`transfer_cohorts.json#zhu_hepatobiliary`）。
- 来源版本与 checksum：肝胆 PDO（出版源，`original_source_audit: required`）；资产 sha256 —— samples `c6c209b22ed57557245de53eb6b7e9581a209f96b074e153468646b9f7e491a9`、drugs `ffbe4ef2598f0de9c0cb1fa6af4c6ed7460ef10e2b7fc9bdf844f27a30fb5863`、pairs `0153a52f5f149046db677065bb6a437ab6ad90a166fd7b38a3502f49f62a4916`、expression `d3aaec0f18bf61aedb33067c79fd0755ed0085ce30421b36288b6a05225c7ba3`。
- 登记状态：`transfer_cohorts.json` 已登记（UNFROZEN；`external_test_eligible: false`）。
- 标签语义：表达 `log2(author RNA-seq abundance + 1)`（出版单位未标注，按非负丰度处理）；响应 `1_minus_auc`，`higher_is_sensitive=true`。
- 准入集合：未登记独立准入白名单（`original_source_audit: required`）。
- 训练·验证·测试·审计角色：PDO_DEVELOPMENT（发育/迁移用）；非外部测试。
- 禁止用途：不得作为外部验证；出版单位未标注，使用须在报告中声明单位假设。

## 5. LINCS（GSE92742）

- 身份键：`pert_id`、`cell_id`（`lincs_gse92742_raw_level3_v1` files）；身份桥 `pert_id / canonical_smiles / InChIKey / aliases`（`lincs_prism_identity_bridge_v1`）。
- 来源版本与 checksum：Broad LINCS GSE92742；Level3 gctx sha512 `c049fd66717defce00a0073ce19339f8d507991d2a1d7dd9b5fd175cf9680295174f019ad34a666ff58d939a74de522536a3900bda719855b7ea414673a226a4`；exact978_cache_v1 sha256 `04b8bb746a61ba4992e49566315327023783ec1c0448da2a9e263e0881281733`（矩阵 1319138×978，gene 顺序 sha256 `b4e2fca877c5cfdcc1c712ad0fd67e97a88b6f7566b013e4bab065f699ebb623`）。
- 登记状态：`datasets.json` 已登记（raw + interim + processed）。
- 标签语义：Level-3 exact pr_is_lm=1 order-978 提取（`exact978`）；不推断基因；Level-3 delta 与 Level-5 签名不得混用（`expression_contracts`）。
- 准入集合：exact978 全覆盖 cache（1319138 行）；`gene_order_contract_id` = 978 顺序冻结。
- 训练·验证·测试·审计角色：TRAIN（EXP-002 严格划分合同）；`xpert_*` 为特征/上下文桥接（response-blind）。
- 禁止用途：不得跨 Level 混杂；不得把 Level-3 delta 直接当 Level-5 签名标签。

## 6. ChemPert

- 身份键：`Response_ID / Chemical_ID / Data_Accession / Expression_ID`（`chempert_primary_keys_frozen_v1`）；泄漏原子按「小写 Chemical_Compound == 小写 LINCS pert_iname」匹配（`chempert_leakage_keys_frozen_v1`）。
- 来源版本与 checksum：public download snapshot 2026-09-16；primary keys FROZEN sha256 `70ff2aab349d9412932a5cf3cd9ad8cc7e139abcb76b3f2824a9ed9986b1c433`（82270 行 / 15 列）；leakage_keys FROZEN sha256 `337bbb2940687fa0cde7e414adff635343000e5b37547065c5a899a42fd6c6db`。
- 登记状态：`datasets.json` 已登记（raw + `chempert_primary_keys_frozen_v1`、`chempert_leakage_keys_frozen_v1`、`chempert_landmark_contract_v1`、`chempert_contract_v2_supplement`）。
- 标签语义：转录响应（`transcriptional_response_rows 82270`），非药敏标签；delta 监督 refer to `delta_supervision_v2`（true iff `expression_profile_kind in {control_labeled, shared_multi_chemical_baseline}`）。
- 准入集合：未登记为药敏训练真值；`data_license` 未声明（`not stated on download page`），使用前须复核许可。
- 训练·验证·测试·审计角色：PERTURBATION_TRANSCRIPTOMICS_REFERENCE_NOT_DRUG_SENSITIVITY_LABELS；`delta_supervision_v1/v2` 供扰动通路使用。
- 禁止用途：不得作为药物敏感度标签；许可与模态合同待定前不得用于训练/再分发。

## 7. DepMap 24Q2

- 身份键：`depmap_id`（Model.csv / 矩阵首列）；细胞系按 `PatientID` 划分（`ccle_24q2_context_identity_split_v1`）。
- 来源版本与 checksum：DepMap Public 24Q2（doi 10.25452/figshare.plus.25880521.v1）；RNA TPM CSV sha256 `39eff342bbbbe0a40208545e93a69cc85a945899592f9d683f87e3b2bb121670`；CRISPRGeneEffect sha256 `d155149181308f8e16b6a4677cde3e5af1ddef68ada1947486ae8c0bcb231452`；exact978 npy sha256 `36ab8a0f8f65df2cdb2c660f2f4ae7c92e26fc7a06ad18ff32bdd0ef11fff9bd`（1517×978）。
- 登记状态：`datasets.json` 已登记（RNA raw + CRISPR raw + exact978 processed + context identity split）。
- 标签语义：RNA `log2(TPM+1)`（non-LINCS X_ctl）；CRISPR `chronos-corrected gene effect`（负值＝必需）。
- 准入集合：context split `split_unit=PatientID`；lines_by_role：locked_eval 10 / train 1338 / val 169（`ccle_24q2_context_identity_split_v1`）。
- 训练·验证·测试·审计角色：CONTEXT_ADAPTER_BASAL_RNASEQ_SOURCE / CELL_LINE_CRISPR_DEPENDENCY_SOURCE_DOMAIN；`cannot_replace_matched_control`、`cannot_use_as_x_ctl`、`cannot_compute_delta978`。
- 禁止用途：CRISPR 依赖 ≠ 药物响应；不得把依赖标签写成药敏结论；不得用 24Q2 表达替换 LINCS 匹配对照。

## 8. 类器官依赖图（organoid dependency map）

- 身份键：`individual_ID`（整组隔离）、model；`dual_library_rule`：双文库模型只保留 minLib（`transfer_organoid_biobank_paired_dependency_v1`）。
- 来源版本与 checksum：2026 Nature Figshare 处理后数据；`checksum` 为 per-file sha256，见 `artifacts/EXP-026/organoid_depmap_admission.json`。
- 登记状态：`datasets_transfer.json#transfer_organoid_biobank_paired_dependency_v1`，ACQUIRED_EXPLORATORY（EXP-026）。
- 标签语义：表达 `author log2TPM`（勿再取 log）；CRISPR 依赖分（非药敏标签）。
- 准入集合：paired_models 162 / paired_individuals 157；癌种 结直肠 85、食管 59、卵巢 11、胰腺 4、胃 3；CRISPR 依赖 2948036→2683044 行（去双文库）；RNA 与项目主基因空间交集 14900。
- 训练·验证·测试·审计角色：AUXILIARY_DEPENDENCY_TRANSFER_AND_DOMAIN_ADAPTATION（迁移检验 / 域适应 / 辅助预训练）。
- 禁止用途：CRISPR 依赖不是药物响应，不得作药敏标签或目标域评价。

## 9. CTRPv2

- 仅准备完成：`ctrp_v2_pharmacoset_zenodo_3905470_v1` 与 `ctrp_v2_materialized_tables_v1` 已 EXPLORATORY_ADMITTED（源域用途，终点 `aac_published`，higher=more sensitive，887 细胞系 / 544 药 / 395263 响应行）；独立划分合同尚未建立（`not_authorized`: 训练或评价须先有单独 split contract）。B1 字段（准入集合 / 训练·验证·测试·审计角色 / 标签单位细分）未登记。
