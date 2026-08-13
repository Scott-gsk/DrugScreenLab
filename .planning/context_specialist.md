# EXP-006 Context Specialist：CCLE/DepMap RNA-seq readiness audit

审计范围：仅检查本地登记、非原始元数据和接口代码；不下载/解压大型 RNA 资产，不读取
PRISM response，不修改 `data/raw`。结论状态：`DATA_PARTIAL`（future adapter 预备信息存在，
但尚无可验收的 CCLE/DepMap RNA-seq 输入）。该状态不阻断 EXP-005 或 EXP-006 的现有 LINCS/XPert
路径。

## 证据与计数

| 检查项 | 当前证据 | 计数/状态 |
|---|---|---|
| CCLE/DepMap RNA-seq 矩阵 | `data/` 文件清单及 `data/registry/datasets.json` 无 CCLE/DepMap RNA-seq asset；未发现 release、assay、gene namespace 或 checksum | **0 个本地资产；未注册** |
| CCLE/DepMap source release | `datasets.json` 没有 CCLE/DepMap expression 条目；现有 DepMap 仅为 PRISM response/identity 元数据 | **0 个 release/version 可核验** |
| LINCS→DepMap context identity | `mvp/foundation/xpert/CONTEXT_REGISTRY.json`（注册 `xpert_context_registry_v1`，SHA256 `18D5D67344D09978A553AB34A9C4C197FCD5A1848EEE0D76D0F752206E8D0477`） | **55** LINCS contexts；其中 **10** 个 `broad_exact_context=true` 且各有 1 个 `depmap_id`；其余 **45** 无精确 DepMap 映射 |
| 已有 exact context IDs | `CL34→ACH-000895`, `HCT116→ACH-000971`, `HT29→ACH-000552`, `LOVO→ACH-000950`, `RKO→ACH-000943`, `SNUC4→ACH-000959`, `SNUC5→ACH-000970`, `SW480→ACH-000842`, `SW620→ACH-000651`, `SW948→ACH-000680` | **10/55 exact**；仅作 identity bridge，不代表有 RNA 输入 |
| XPert 978 gene universe | `data/external/xpert_source/processed_data/l1000_gene_info_978.csv`，SHA256 `85806FD6D6EB645925C45C04B0AB26E0769BE7DBA252E545CBC878BDCCFA9A60`；registry gene-order digest `b4e2fca877c5cfdcc1c712ad0fd67e97a88b6f7566b013e4bab065f699ebb623` | **978** ordered landmarks；可作为目标顺序，尚无 CCLE/DepMap→978 crosswalk |
| XPert matched controls | CONTEXT_REGISTRY 的 exact-LINCS policy；55 contexts 的登记 control record 总数 **248** | 可用于 LINCS `X_ctl`；**不能**替代 basal RNA context，也没有 RNA-seq matched control 记录 |
| Adapter output/normalization | 仅有 `src/drug_screen/foundation/xpert_adapter.py` 的 h5ad contract validator（`X` 与 `X_ctl` 均为 `[n,978]`）；无 CCLE/DepMap RNA adapter、normalization parameter digest 或 split manifest | **未实现/未验收** |

## 唯一当前 blocker（不升级为 EXP block）

要把 CCLE/DepMap basal RNA 接入 XPert-compatible context，至少必须先冻结并登记：

1. 官方 source release/accession、assay（RNA-seq 定量单位）和下载 URI；
2. 原始 gene namespace→上述 ordered 978 的版本化、可审计 crosswalk（duplicate/missing/one-to-many 规则）；
3. `depmap_id` 为主 identity 的 source sample/library crosswalk，并证明同一 cell line 的技术重复归属；
4. 仅用 TRAIN 拟合的 normalization 参数（不得用 test/external response 统计量）；
5. group-atomic split manifest（`depmap_id`、source sample/library/batch 及派生 profile 不跨 role）；
6. response-blind 断言及 checksum。缺少任一项时，adapter 只能保持 `DATA_PARTIAL`；若下游声称
   已接入或用名称猜测映射，则触发 `DATA_BLOCKED`（provenance/identity 无法确认）。

## 最小 response-blind fixture 计划（仅接口测试，不是训练数据）

使用确定性小 fixture 验证 RNA-seq→978→平台归一化→XPert contract，建议 `tests/data/context_adapter_fixture/`：

- `rna_counts.tsv`：3 个 source profiles（`depmap_id` 取上述 exact bridge 中 3 个，另加 1 个未映射
  context 作为负例），每行含 `source_context_id, depmap_id, sample_id, library_id, batch_id,
  gene_id_namespace, gene_id, raw_count`；基因列可用 978 个目标基因加 2 个非目标基因，值为固定整数。
- `gene_crosswalk.tsv`：仅允许一对一映射到 frozen 978 order；显式列出 2 个 missing、1 个 duplicate
  和 1 个 unmapped gene，测试拒绝/审计计数，不进行隐式补全。
- `normalization_spec.json`：`normalization_id`、assay/unit、duplicate rule、TRAIN-only fit
  population、parameter digest、ordered-gene digest；fixture 参数固定为可复现的 log1p + train-fitted
  per-gene z-score（仅测试接口，不宣称适用于真实 release）。
- `adapter_manifest.json`：`adapter_id, source_dataset_id, release, source_asset_sha256,
  generator_revision, source_context_id, depmap_id, split_entity_id, split, response_blind=true,
  forbidden_inputs=[PRISM response, efficacy labels, external-test aggregates]`；输出
  `context_vector_978` 形状必须为 `[n_context,978]` 且全为 finite。

验收断言：

1. gene order 与 frozen digest 完全一致，shape 非 978 或非 finite 即 `INVALID`；
2. `depmap_id`/sample/library/batch 的 train-test 交集为 0；未映射 context 保留 source namespace，
   不得按显示名猜测 join；
3. fixture adapter 只输出 basal `context_vector_978` 和 provenance，不生成 `X_ctl`、`Delta978`、
   treatment 或 response label；XPert 的真实 `X_ctl` 仍来自 LINCS matched control contract；
4. 任何 response 读取、外部统计量进入 normalization、或 ambiguous crosswalk 均应被测试阻断。

