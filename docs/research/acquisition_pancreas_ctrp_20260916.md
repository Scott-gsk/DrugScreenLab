# 胰腺 PDO 与 CTRPv2 公开资源获取复核（2026-09-16）

## 结论

本轮实际补齐了 Tiriac 2018 的 **完整公开 FPKM-UQ 表达文件集合**：55 个文件、合计
21,265,093 bytes。与历史样本逐文件 SHA-256 去重后，新增保存 53 个，2 个文件已在
`data/raw/exp013/tiriac_2018/gdc_open/` 中存在相同内容，未保留重复副本。

这只补齐了一个一致的公开表达表示，尚未建立表达—药敏任务配对：公开桶只给出 UUID
对象路径，未给 PDO 身份交叉表；Tiriac 的药敏表也没有作为公开可取得平面文件获得。

2025 胰腺癌 PDO 研究已重新从出版方下载其 Supplementary Data 10 AUC 工作簿，确认为
63 个药物的 AUC 表、含 DAC_* 类器官列。HRA010277 的实际测序文件和可将其对应到
DAC_* 的身份表未公开下载；本轮没有申请受控访问或联系数据管理员。

CTRPv2 的已下载 Zenodo PharmacoSet RDS 再次尝试解析仍未完成，官方 NCI 平面表 URL
仍返回 studycatalog HTML 而不是数据文件。因此 CTRPv2 仍是“已下载、未物化”，不能
称为已可训练或已完成外部验证。

## 逐数据源的文件完整性与任务配对完整性

| 数据源 | 公开文件完整性（本轮证据） | 任务配对完整性 | 当前可得结论 |
| --- | --- | --- | --- |
| Tiriac 2018 / phs001611 | 公开桶共有 276 个对象。本轮完整取得其中 55/55 个 `.FPKM-UQ.txt.gz` 文件；未取同一批样本的 55 个 FPKM、55 个 HTSeq、55 个 STAR 及 55 个 augmented-STAR 的重复表达表示，也未下载 FASTQ。 | 缺公开 PDO 身份映射和药敏数值表，0 个可验证表达—药敏对。 | 有完整、同口径的公开表达文件集合；不可作为配对药敏训练或验证数据。 |
| 2025 胰腺 PDO / DOI 10.1038/s41467-025-57520-3 | Supplementary Data 10 已取得，63 药的 AUC 工作簿；包含 DAC_* 类器官标签。论文将功能验证队列类器官 RNA/ATAC 指向 HRA010277。 | AUC 有 DAC_* 身份；没有公开取得的基线表达矩阵和文件—DAC 映射，0 个已验证表达—AUC 对。 | 公开药敏及其工作簿内身份存在；仍是响应单侧资源。 |
| HRA010277 | 官方页面是 controlled access / Request Data；未取原始测序文件。页面描述与论文数据可用性文字对 RNA 范围并不完全一致，不能由元数据推断出可训练矩阵。 | 不适用；未取得文件或可公开身份交叉表。 | “未公开取得”不等于“永久不可获取”；当前准确状态是需受控访问、未申请。 |
| CTRPv2 | RDS 已在历史路径存在且 SHA-256 已复核。现有环境有 `pyreadr`、`rdata`，没有 `R/Rscript` 或 `rpy2`；有界 25 秒复跑未 materialize。官方 NCI flat-file URL 返回 HTML 跳转页。 | 无完整导出的 response、drug、cell、expression 表，故无可验收配对。 | `NOT_INGESTED`；不能把 S4 槽位字符串当作解析后的 AUC 或身份信息。 |

## 可复核产物

- Tiriac 逐文件清单（55 条，含 dataset_id、相对路径、源 URL、SHA-256、字节数、格式、角色、取回时间）：`data/artifacts/resource_acquisition_20260916/pancreas_ctrp/tiriac_2018_public_fpkm_uq_manifest.json`。
- 2025 胰腺 PDO AUC 文件清单与工作簿结构：`data/artifacts/resource_acquisition_20260916/pancreas_ctrp/paad_2025_public_files_manifest.json`。
- 实际未取到/未物化的 URL、HTTP 观察、工具限制和影响：`data/artifacts/resource_acquisition_20260916/pancreas_ctrp/failed_or_unresolved_attempts.json`。
- Tiriac 新获取的 53 个非重复文件：`data/raw/resource_acquisition_20260916/pancreas_ctrp/tiriac_2018_fpkm_uq/`。
- 2025 AUC 原始工作簿复用既有 canonical 文件：`data/raw/exp013/paad_ncomms_2025/41467_2025_57520_MOESM13_ESM.xlsx`；本轮相同哈希副本未保留。

## 去重记录

| 本轮候选路径 | SHA-256 | 保留的 canonical 路径 | 动作 |
| --- | --- | --- | --- |
| `raw/resource_acquisition_20260916/pancreas_ctrp/tiriac_2018_fpkm_uq/a6ddcc84-8e4d-4c68-885c-2d51168eec97.FPKM-UQ.txt.gz` | `9a986bc5c8c0738fe49ce3a2f5127a6f0adea05a7846cbaf38eb1d6f4f52049d` | `raw/exp013/tiriac_2018/gdc_open/a6ddcc84-8e4d-4c68-885c-2d51168eec97.FPKM-UQ.txt.gz` | 删除本轮副本；清单改指向既有文件。 |
| `raw/resource_acquisition_20260916/pancreas_ctrp/tiriac_2018_fpkm_uq/30520ecd-c6bd-4039-9b1a-d3f999235598.FPKM-UQ.txt.gz` | `33759249553a264749c80f89f07aecb6a03d5f3af0781a198332a3013058bd44` | `raw/exp013/tiriac_2018/gdc_open/30520ecd-c6bd-4039-9b1a-d3f999235598.FPKM-UQ.txt.gz` | 删除本轮副本；清单改指向既有文件。 |
| `raw/resource_acquisition_20260916/pancreas_ctrp/paad_2025/41467_2025_57520_MOESM13_ESM.xlsx` | `0e009cd56d2f6c25b0b355590becc4abfee3e1080633173bc33606d1bf325340` | `raw/exp013/paad_ncomms_2025/41467_2025_57520_MOESM13_ESM.xlsx` | 删除本轮副本；清单改指向既有文件。 |

历史原始文件未删除或改写。

## 来源与边界

- Tiriac 公开表达：NCI GDC open S3 bucket `gdc-organoid-pancreatic-phs001611-2-open`；受控研究目录：<https://ncpi-data.org/studies/phs001611>。
- 2025 胰腺 PDO 论文与 Data availability：<https://www.nature.com/articles/s41467-025-57520-3>；AUC 附件 URL 见对应 JSON 清单。
- CTRPv2 官方平面表尝试：<https://ctd2-data.nci.nih.gov/Public/Broad/CTRPv2.0_2015_ctd2_ExpandedDataset/v20.data.curves_post_qc.txt>；RDS 来源：<https://zenodo.org/records/3905470>。

本报告没有新增实验、训练、处理矩阵或共享数据登记。若将任何资源接入正式分析，仍需先取得
同一 PDO 的身份、表达与响应交集，冻结数据合同并登记版本与校验和。
