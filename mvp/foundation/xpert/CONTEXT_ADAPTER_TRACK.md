# Context Adapter Track：CCLE / DepMap / organoid / PDO

> 状态：`PREPARATION_ONLY` · `DATA_PARTIAL`
>
> 本文件只定义 future context adapter 的 normalization 与 interface 边界。不创建、登记、批准或执行 `EXP-006`；不训练 adapter 或 XPert；不修改 XPert foundation，也不改变 `EXP-004`。

## 目的、现有锚点与非目标

`mvp/foundation/xpert/CONTEXT_REGISTRY.json` 已将当前 XPert/LINCS context 以 `context_id` 登记，并列出 `CCLE`、`DepMap`、`organoid`、`PDO` 为 future adapters。该登记不等于这些来源已被选定、下载、标准化或验收。本 track 的作用是使未来接入在读取数据前就有一致的接口和 leakage gate。

相关的既有边界为：

- `mvp/foundation/xpert/XPERT_FOUNDATION_RESULT.json` 说明 foundation 的官方 contract 采用处理后表达与 matched `X_ctl`，并使用固定的 ordered 978 genes。
- `mvp/foundation/xpert/CONTEXT_REGISTRY.json` 的 primary identity 是 `context_id`，并记录 Broad/DepMap 对齐状态与 exact LINCS control policy。
- `mvp/extension/BROAD_PRISM_COHORT_V1.json` 是 response-blind identity/context freeze；它不能提供 response supervision。
- `mvp/foundation/xpert/ASSET_MANIFEST.json` 是官方 source/asset checksum 的既有登记，而非可由本 track 修改的资产。

本 track 明确不做以下工作：

- 不触碰 `data/external/xpert_source/`、官方 checkpoint、official code、XPert backbone、loss、gene order、训练配置或 foundation split。
- 不运行训练、fine-tuning、adapter weight 学习、预测或 benchmark；也不把 basal context vector 宣称为已验证的 foundation 等价物。
- 不读取 PRISM/GDSC response 值、疾病/疗效标签、external-test target 或 `EXP-004`/`EXP-005` 的输出。
- 不用 context 适配为理由修改或重新解释 `experiments/records/EXP-004.md`。

## Context adapter 的数据接口

一个未来 adapter 只能输出一个有来源的 basal-context 表示；它不能生成、替代或估计处理响应。每一条 non-raw manifest 记录至少需要下列字段：

| 字段组 | 必填字段 | 约束 |
| --- | --- | --- |
| 来源 | `adapter_id`、`source_dataset_id`、release/version、source URI/accession、source asset SHA256、generator revision/config digest | source 文件和版本不可只靠本机路径标识；无 provenance 不接入 |
| 样本 identity | `source_context_id`、`context_kind`、source-specific biological grouping ID、可用时的 `depmap_id`、显示名 | 显示名不是 join key；organoid/PDO 不得因 tissue/疾病描述而等同于 cell line |
| XPert 对齐 | `context_id`（仅确切映射时）、crosswalk evidence、`broad_exact_context`/映射状态 | 无精确 crosswalk 时保持 source namespace，并在后续评估中按 unseen context 处理 |
| RNA 输入 | 原始 gene namespace、原始单位/assay、gene mapping revision、duplicate-gene rule、输入资产 digest | 不隐去 gene missingness、重复或 assay 差异 |
| 标准化输出 | `normalization_id`、fit population、参数 digest、ordered gene-universe SHA256、`context_vector_978` 的 shape/finite validation | 参数只能用预先指定的 train role 拟合；不得用 test/external response 统计量 |
| Split | `split`、`split_entity_id`、生物学/技术重复归属、split manifest digest | 同一来源样本、克隆、donor/specimen 或其派生 profile 不可跨 role |
| 标签边界 | `response_blind=true` attestation、forbidden-input check、验证时间/角色 | 不得存入或派生药敏、疗效、疾病反转或 external-test 标签 |

`context_vector_978` 的含义仅为“可供未来模型接口消费的、已归一化 basal RNA context”。它不是 LINCS `X_ctl`，不能用于计算 `Delta978`，也不能替代 treatment 的 matched control。现有 foundation 中的 `adata.obsm[X_ctl]` 保持原有数据语义和内容不变。

## Normalization 规范

每一个 CCLE、DepMap、organoid 或 PDO adapter 都必须在 source-specific specification 中冻结以下步骤，且在实施前由 Data Steward 验收：

1. 明确 release、assay、表达单位、gene ID namespace、源资产 checksum 和许可/访问边界。
2. 用版本化、确定性的 gene crosswalk 映射到同一 ordered `Delta978` universe；缺失、重复、多对一映射和不确定映射必须显式报告。不得用基于 label、相关性或模型输出的补全方式消除缺失。
3. 预先声明 duplicate-gene 的确定性处理规则、转换和缩放方法。其拟合统计量只能来自 frozen training role，不得将 foundation、validation、test 或未来外部 cohort 混入拟合。
4. 产出固定顺序的 978 维 finite vector，并记录 gene-order SHA256、normalization parameter digest、样本数和拒绝原因；shape 不为 978 或含非有限值即为 `INVALID`。
5. 保留 source-level batch/assay 信息供审计。禁止在测试 context 与训练 context 上联合 batch correction，或为提高模型指标而回写 normalization。

本规范不预设 CCLE/DepMap 与 organoid/PDO 在生物学上可直接互换。它只要求 interface 一致；跨来源可比性必须在将来经独立、response-blind 验证证明，不能由同一表达空间、组织标签或模型表现推定。

## Identity、split 与 leakage gate

### CCLE / DepMap

- 跨 Broad 资源的主 identity 为精确 `depmap_id`。`ccle_name` 仅作可读性辅助，不能单独合并。
- 只有当已登记的 crosswalk 明确支持时，外部 context 才可引用现有 `context_id`；`CONTEXT_REGISTRY.json` 中没有 exact 对齐证据的条目不得用名称猜配。
- 若 adapter context 与训练 context 共享 `context_id`、`depmap_id`、source sample 或其技术/生物重复，则不能在 cold-context test 中宣称未见泛化。

### Organoid / PDO

- 使用 source dataset/version + source-stable model/sample ID 作为基础 identity；若来源提供 donor、specimen、clone 或 passage grouping，则这些字段必须进入 `split_entity_id`。
- 同一 donor/specimen/clone 的所有重复和派生 profile 必须留在同一 split role。无这类 lineage 信息时，不得声称 donor-cold 或 specimen-cold 泛化。
- organoid/PDO 与 cell line 之间只能在有可审计的、预注册的 exact crosswalk 时建立关联；不得以组织、癌种、病人描述、相似表达或药敏/疗效结论进行合并。

所有 adapter 均须在生成可用向量前完成 group-atomic split。验证程序必须报告 train/validation/test 间的 `source_context_id`、`context_id`、`depmap_id`、source sample、donor/specimen/clone 以及 RNA library/profile 的交集；未解释的交集是 identity leakage，状态为 `DATA_BLOCKED`。不得通过删除交集报告、重命名样本或把重复样本降为缺失来放宽这一 gate。

## Response-blind 角色边界与交接

| 阶段 | Data Steward 交付 | 下游可做的事 | 不可做的事 |
| --- | --- | --- | --- |
| 准备 | adapter specification、source/identity/checksum register、normalization plan、split plan、forbidden-label attestation | 审阅接口是否完整 | 下载/训练/预测，或读取 response 值以决定 adapter |
| 验收后 | non-raw context manifest、978 order digest、crosswalk、train-only normalization parameters、leakage validation | 在正式 EXP 批准后按冻结接口消费 | 改变 foundation `X_ctl`、重拟合 test 统计量、把 context source 变成 target/control |
| 评估交接 | 先交付 frozen prediction/ranking digest 与 held-out identity；response package 仅由获授权 evaluator 解封 | 独立评价 | 将 response 或排名回流到 mapping、normalization、split 或训练选择 |

将来 handoff 的 artifact manifest 必须列出每个 non-raw 文件的相对路径、SHA256、source/release、生成器 revision、配置 digest、schema 和 known deviations。raw RNA、外部 ignored assets、XPert checkpoint/prediction profile、PRISM response value 不进入此 handoff，也不作为 Git 跟踪内容。

## 当前 readiness 与升级条件

本 track 的唯一当前数据状态为 `DATA_PARTIAL`：现有 context registry 已表达 future-adapter 的接口意图，但尚未为 CCLE、DepMap、organoid 或 PDO 冻结具体 source release、checksum、gene crosswalk、normalization parameters、identity crosswalk 或 split manifest。因此它不是 `DATA_READY`，也没有启动任何数据接入。

未来任一候选 adapter 若出现下列情形，应立即是 `DATA_BLOCKED`：provenance/checksum 无法确认；978 gene order 无法证明；context identity 或 lineage 不能审计；matched-control 语义被错误替代为 basal context；normalization 使用了 held-out/external 数据；或任何 PRISM/GDSC/疗效/test label 污染了 adapter 选择或参数。只有全部 response-blind gate `VALID` 且正式 EXP 获批后，才可由 Data Steward 重新评定 readiness。
