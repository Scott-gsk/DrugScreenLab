# EXP-006 遗传→化学数据准备契约

> 状态：`PREPARATION_ONLY` · `DATA_PARTIAL`
>
> 本文件只是 Data & Bioinformatics Steward 的准备性数据契约，不创建、登记、批准或执行 `EXP-006`。它不代表新的实验记录、训练运行、预测、评估或科学结论，也不占用 `EXP-005`。

## 范围与受保护边界

本契约仅定义将来可能使用的遗传→化学数据接入条件，以及向下游交接时必须可审计的元数据。它以当前仓库的以下**既有**记录为依据，不重算、不覆盖其中的结果：

- `mvp/extension/GENETIC_CHEMICAL_LOW_DATA_CONTRACT.md`：已冻结的统一 `Context + Perturbagen → Delta978` 接口和化学端角色边界。
- `mvp/extension/GENETIC_UNIPERT_FAST_AUDIT.json`、`scripts/data/build_e2_genetic_manifest.py`：当前遗传侧的 response-blind cohort、对照和 split 实现证据。
- `mvp/foundation/xpert/DRUG_REGISTRY.json`、`mvp/foundation/xpert/CONTEXT_REGISTRY.json`：化合物与 context 的当前登记字段。
- `mvp/extension/BROAD_PRISM_COHORT_V1.json`：在读取 response 前冻结的 Broad identity/context cohort。
- `mvp/foundation/xpert/ASSET_MANIFEST.json`、`mvp/foundation/xpert/XPERT_FOUNDATION_RESULT.json`：XPert foundation 的来源、checksum 和冻结边界。

以下边界不可由本准备文档改变：

- `experiments/records/EXP-004.md` 保持原样；不得迁移、覆盖、重算或重新解释其中的遗传→化学结果。
- 不修改 `PROJECT_STATE.yaml`、`experiments/registry.yaml`、任何 `EXP-005` 文件、XPert backbone、官方资产、raw data、外部 ignored assets、checkpoint、prediction profile 或 PRISM response 值。
- 本文不是“恢复 Context+Chemical 主干”的提案；它只规定日后若另获批准时，数据接口必须达到的完整性条件。

`DATA_PARTIAL` 的含义是：已找到可引用的 response-blind 元数据和既有契约，但尚未为新的数据接入冻结 dataset identity/version、gene mapping、context source、split manifest 或 provenance bundle。因此不能宣布 `DATA_READY`。

## Response-blind 数据契约

所有将来进入遗传→化学接口的非原始派生记录，必须先通过下表的 gate。缺失任何结果定义所需字段，不得以默认值、别名或模型推断补齐。

| 契约项 | 必须冻结的内容 | 禁止的替代方式 |
| --- | --- | --- |
| 数据集身份 | `dataset_id`、发布/版本、accession 或正式来源、源文件相对定位、原始资产 SHA256、派生生成器 revision 与配置 digest | 用模型名、文件昵称、日期或上一次运行目录充当 dataset identity |
| 记录 schema | `source_record_id`、`sample_id`、`treatment_group_id`、`modality`、`perturbagen_id`、`perturbation_direction`、`context_id`、dose/time 原始语义、treatment/control 记录引用、gene-universe digest、`split` 与 `split_entity_id` | 只保存向量行号、无法追溯 treatment/control 或把遗传记录伪装成 chemical |
| Gene universe | 单一、顺序固定的官方 `Delta978` gene list；记录 gene ID namespace、ordered-list SHA256 和每条向量的顺序一致性 | 按字母排序、按样本可用基因重排、补零/插补后不声明，或把不同 gene universe 直接拼接 |
| Target | `Delta978 = matched treatment vector − matched control vector`；两向量须共享同一已冻结 gene order 和同一测量/处理语义 | treatment-only label、全局平均 control、跨 context control 或将 basal RNA 当成 treatment control |
| Context | 明确的 source context identity、其到 `context_id`/DepMap identity 的 exact crosswalk 状态、平台与 normalization ID | 以 tissue、癌种、显示名称或 embedding 相近性作 identity join |
| Perturbagen | chemical 与 genetic 都保留 modality 和方向；化合物使用登记的精确标识，遗传扰动保留 `pert_id`、标准 gene symbol 与方向 | 用 `cmap_name`、自由文本、基因别名或相似结构替代精确 identity |
| Dose/time | 数值、单位、适用性和源字段；chemical 遵守既有 canonical dose/time，遗传扰动保留其实际实验时间和 dose 是否适用 | 把占位数值解释为物理 chemical dose，或将缺失 dose/time 静默填成 chemical 条件 |
| Normalization | 输入单位、gene mapping、重复基因处理、拟合样本范围、参数 digest 与输出向量 finite 检查 | 用 test/external 样本统计量、response 值或未来 cohort 拟合 normalization |
| Split | 先冻结 split entity 与 split seed/algorithm，再产生训练特征；完整 group、源记录、treatment/control pair 和生物学重复不得跨 split | 行级随机切分、在 test 结果后改 split，或把同一 identity 的别名分到不同 role |
| Provenance | 每个输入和派生产物的 checksum、来源、生成命令/代码 revision、创建者角色及验证结果 | 只记录本机绝对路径、无 checksum 的 cache，或从不明 provenance 的产物继续处理 |

### 既有遗传侧对照证据的限定

当前 E2 FAST 资料所述的遗传条件为 `GSE92742 trt_sh`、`96h`，并采用 `rna_plate || cell_id || pert_time || pert_time_unit` 作为 match key；对照优先级为同一 key 的 `ctl_vector`，其次为 `ctl_untrt`。这是可引用的**既有数据契约证据**，不是对任何新数据集的自动授权。

新候选数据若复用这一逻辑，必须显式保存 `control_match_key`、`control_type`、`control_record_id` 与 `treatment_record_id`。若不能证明 matched control，或 treatment/control 的基因顺序、context、时间或平台不一致，则该候选数据为 `DATA_BLOCKED`，不得进入 transfer。遗传 perturbation 没有物理浓度时，必须在 dataset-specific adapter 中标为 `not_applicable` 并冻结其编码语义；不得把现有接口中的数值占位误作化学浓度。

## Identity、split 与泄漏防护

### 精确 identity keys

| 对象 | 主键与允许的桥接 | 明确不允许 |
| --- | --- | --- |
| Chemical perturbagen | `pert_id`；Broad 对齐仅接受登记的 `broad_base_id`、精确 `broad_id`/`broad_id_base`、或已审计的 InChIKey bridge，并要求 `broad_inference_eligible` | `cmap_name`、药名别名、模糊字符串、结构相似性或模型 embedding 最近邻 |
| Genetic perturbagen | `pert_id` + 标准化 `gene_symbol` + `perturbation_direction`；`treatment_group_id` 必须由已登记的源字段确定性生成 | 仅用基因名称、把 knockdown/overexpression 混为同一扰动，或跨来源复用无来源前缀的 group ID |
| LINCS/XPert context | `context_id`，并保留 XPert registry 中的 exact-control 语义 | 用组织名或代表性 dose/time 推断 context identity |
| Broad/CCLE context | `depmap_id` 是跨表 identity；`ccle_name` 仅为辅助字段；映射到现有 `context_id` 时须有登记的 exact crosswalk | 细胞系同名、tissue 相同、人工别名或 response 相似性推断 |
| Organoid/PDO context | 来源内稳定的样本/模型 ID，加上来源 dataset/version 和可审计的 specimen/donor grouping ID（若来源提供） | 以癌种、病人描述或与 cell line 的表达相似性建立等同性 |

每个 split manifest 必须报告并验证下列跨 role 交集为空：`source_record_id`、`sample_id`、`treatment_group_id`、精确 perturbagen identity、`treatment_record_id`、`control_record_id` 与 source-specific biological grouping ID。若设计允许共享 control，必须先有独立的、可审计的泄漏论证；在没有该论证时，非空交集即为 identity leakage，状态为 `DATA_BLOCKED`。

化学 test 的既有冻结 manifest 只能作为上游引用，不能被本文重新划分。遗传预训练记录只能来自其冻结的 `train` role；外部 test 的真实 `Delta978`、response 或任何 downstream label 均不可用于挑选 gene、identity mapping、normalization、feature、阈值、regime 或 checkpoint。

## 禁止数据与标签边界

以下数据在本准备契约和其后数据准备阶段均为 forbidden labels / forbidden inputs，除非未来经正式 EXP 审批并由独立的 evaluation handoff 解封：

- PRISM/GDSC 等药敏/efficacy response 值，以及 `mvp/foundation/xpert/BROAD_PRISM_CRC_V1.parquet` 中的 `response_raw`、`sensitivity_score`、`response_direction`；本准备工作不读取这些值。
- `mvp/foundation/xpert/BROAD_XPERT_EVALUATION_V1.json`、下游 ranking 或疾病逆转结果中可导出的 response 监督信息。
- 化学或遗传 test role 的真实 treatment、control、`Delta978` target，及任何由它们计算的统计量。
- `EXP-004`、`EXP-005` 的模型输出、指标、预测、报告或日志，作为新数据的训练标签、特征、选择依据或数据集身份。
- `data/external/xpert_source/`、`data/external/xpert_assets/` 及其他 ignored/raw 资产中的内容；本文件仅引用 `ASSET_MANIFEST.json` 已登记的来源和 checksum，不访问或修改这些资产。

`mvp/extension/BROAD_PRISM_COHORT_V1.json` 可在其已声明的 response-blind 范围内作为 identity/context freeze 的元数据证据；它不能解除上列 response 值禁令。任何 external-test contamination、来源无法确认或标签边界被破坏，都必须产生 `DATA_BLOCKED`，不能通过缩小 cohort、删除审计字段或放宽规则绕过。

## 角色边界与交接包

| 角色 | 允许职责 | 不允许职责 |
| --- | --- | --- |
| Data & Bioinformatics Steward | 冻结 contract、gene universe、identity crosswalk、matched-control proof、response-blind split、normalization specification、checksum registry 和验证报告 | 改 raw data、读取/使用 forbidden label、改变 XPert foundation、以模型表现放宽数据规则 |
| Model Engineer | 在正式审批后只消费已验收的 non-raw contract output；报告 schema/feature 不兼容 | 选择数据集、改 identity/split/control、拟合 test 统计量、把 PRISM response 作为训练信息 |
| Evaluation & Statistics Analyst | 在预测/ranking digest 冻结后接收 held-out identity 和按审批范围解封的 evaluation package | 反向把评估 response、test label 或选择结果回流到数据准备 |
| Research Manager | 决定是否创建并批准将来的正式 EXP，协调影响结果定义的变更 | 将本准备文档视为 EXP 启动、审批或 `DATA_READY` 证明 |

未来的 handoff 是一个**逻辑** non-raw bundle；本文件不创建目录或 artifact。正式 EXP 获批后，bundle 至少应含：

1. 版本化 data contract 与 source manifest（dataset/version/source/checksum/许可证或访问说明）；
2. ordered `Delta978` gene-universe 文件及其 SHA256；
3. exact identity crosswalk 与拒绝/歧义记录；
4. matched-control manifest（不含 raw vector）及 control-policy proof；
5. group-atomic split manifest、split algorithm/seed、重叠检查与 digest；
6. normalization specification、仅训练角色拟合的参数 digest 与 finite/shape validation；
7. forbidden-label attestation、validator revision、配置 digest、验证结果和已知偏差。

每一个大资产只以相对路径、版本、checksum、来源和生成信息交接；raw matrix、checkpoint、prediction profile 和 response value 不进入 Git 或本契约。只有前述 bundle 全部 `VALID` 后，Data Steward 才可在未来正式记录中重新判定状态；本准备文档的唯一当前数据状态仍为 `DATA_PARTIAL`。

## 变更控制

改变 `Delta978` 定义、978 gene order、matched-control rule、identity key/crosswalk、split entity/role、normalization 拟合范围或 forbidden-label 边界，会改变结果定义，必须暂停受影响下游并重新冻结 contract。新增验证代码、补充 provenance 字段或不改变上述语义的文档澄清不要求暂停。
