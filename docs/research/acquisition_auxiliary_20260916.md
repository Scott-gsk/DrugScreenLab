# Kuo 2026 与 Bhatia 2022 辅助资源实际获取记录（2026-09-16）

## 本轮结论

| 资源 | 本轮实际状态 | 已取得的公开文件 | 能说明什么 | 仍缺什么 |
|---|---|---|---|---|
| Kuo et al. 2026，PMC13308604 | **公开处理后数据已下载，尚未转成项目矩阵** | 113 个 PDO 的全量 cBioPortal RNA FPKM、WES 派生 mutation call 和 CNA API 导出，以及研究、样本和 profile 元数据 | 可作为表达/表征辅助候选；公开处理后表已取得 | 未取得 raw WES；没有药物响应字段，因而不构成表达—药敏配对数据集 |
| Bhatia et al. 2022，PMC9135475 | **公开附件未找到（技术拦截）＋原始组学受控访问** | 无 | 论文补充表、代码及元数据线索可公开浏览 | PMC 附件实际下载被网页验证拦截；原始数据声明位于 dbGaP，需获授权；没有药物响应面板 |

本文件仅记录获取与可用性，不代表任何训练、评估或实验已执行。

## Kuo 2026：旧入口失效，但 RNA 实际可取得

论文为 Kuo et al., *Patient-derived organoids across cancers reveal conserved tumor heterogeneity and actionable therapeutic vulnerabilities*，DOI `10.1126/sciadv.adz3351`。论文的 Data, code, and materials availability 段称 PDO 与匹配肿瘤的 WES 和 RNA-seq 在 cBioPortal 公开，文章所给旧链接为 `https://www.cbioportal.org/pdo_wcm_2026`。

旧 study ID `pdo_wcm_2026` 于本轮 API 请求返回 HTTP 404 `Study not found`。这不是资源不可获取：官方 API 的当前 study ID 为 `pan_pdo_wcm_2026`，研究名称为 *Pan-Cancer Patient-Derived Organoids (WCM, 2026)*。其 study 元数据声明 113 个 PDO、99 位患者、10 个癌种，且含 RNA-seq 和 mutation profile。cBioPortal 的 `treatmentCount` 为 0；该公开研究中没有药物响应数据。

### RNA 获取与完整性

下载根目录：`data/raw/resource_acquisition_20260916/auxiliary/kuo_2026_cbioportal/`。逐文件来源、大小、SHA256、格式和获取时间见 `data/artifacts/resource_acquisition_20260916/auxiliary/download_manifest.json`。

RNA 文件 `rna_fpkm_pdo_api.ndjson.gz` 从官方 profile `pan_pdo_wcm_2026_rna_seq_mrna` 的 POST endpoint 获取。筛选规则透明地固定为样本 ID 以 `_O` 结尾；研究样本元数据中这 113 个 `_O` 样本与研究声明的 113 个 PDO 一致。API 不提供单独的 organoid sampleType 枚举：这些样本的 `sampleType` 都标为 `Primary Solid Tumor`，所以本轮没有把该字段误作 PDO 身份证据。

查询了 cBioPortal 返回的全部 44,896 个 Entrez gene ID，按 45 个不重叠的基因分块（44 组 1,000 个、最后 896 个）请求。导出共 1,697,360 条非空表达记录；重读 gzip 行数与下载 manifest 的 `records_written` 一致。部分基因块返回零或少于 113 条/基因，是 profile 中该基因无值，不是漏页；45 个请求块均已写入 `retrieval_manifest.json`。这证明**全量 API 可返回的 PDO RNA FPKM**已到本地，尚未作矩阵化、标准化或同药敏配对。

### WES 派生 mutation call 与 CNA 状态

cBioPortal profile 元数据显示 `pan_pdo_wcm_2026_mutations` 为“Mutation data from whole exome sequencing”。本轮对全体 113 个 `_O` PDO sample ID 以官方 POST endpoint `/molecular-profiles/pan_pdo_wcm_2026_mutations/mutations/fetch?projection=DETAILED&pageSize=10000000` 发出单个批量请求，HTTP 200，取得 41,790 条公开、处理后的 mutation calls。压缩原始响应为 `wes_mutations_pdo_api_detailed.json.gz`；SHA256、字节数及请求体均在 `kuo_wes_mutation_retrieval_audit.json` 和 download manifest 中记录。

突变 calls 实际覆盖 111/113 个 PDO；`WCM1277_2_O` 与 `WCM715_O` 未返回 mutation 行。此处缺失仅表示公开 mutation call response 中未出现记录，不能据此判断未测序或无变异。没有任何非 PDO 样本混入。

同一研究的 discrete CNA profile 描述为“obtained from CNVkit”。这属于 WES 衍生的拷贝数结果这一合理推断，但原始 profile 元数据没有单独给出 CNVkit 的输入测序类型；所以本记录明确称其为**CNVkit CNA profile**，不把该推断夸大为 raw WES。对同一 113 个 PDO sample ID 以 `discreteCopyNumberEventType=ALL` 发起 API 请求，HTTP 200；全量压缩响应和覆盖审计见 `cnvkit_discrete_cna_pdo_api_detailed.json.gz` 与 `kuo_cna_retrieval_audit.json`。

本轮取得的是公开的、处理后的全基因 mutation callset 和 CNVkit CNA profile，**没有下载 raw WES reads、BAM 或 FASTQ**。所以“WES 已获取”在本记录中仅指 WES 派生处理结果，不指原始测序资产。

## Bhatia 2022：附件与受控原始数据必须分开报告

论文为 Bhatia et al., *Patient-derived triple-negative breast cancer organoids provide robust model systems that recapitulate tumor intrinsic characteristics*，DOI `10.1158/0008-5472.CAN-21-2807`。PMC XML 明确列出三份补充附件：

- `NIHMS1799657-supplement-Supplementary_Tables.xlsx`（页面标示 937.2 KB）
- `NIHMS1799657-supplement-Supplementary_Materials.pdf`（8.4 MB）
- `NIHMS1799657-supplement-Revised_fig_S7.pdf`（716.4 KB）

文章 Data Availability Statement 还声明原始数据在 dbGaP `phs002722.v1.p1` 和 `phs001486`，这是受控访问路径，不能在无获批权限下下载。文章仅把 copy-number segment（Table S2）、转录组 signature scores（Table S4）和 PDO metadata（Table S1）列为补充表内容；它没有报告本项目所需的多药逐 PDO 响应矩阵。

本轮对三个 PMC `articles/instance/9135475/bin/` 附件 URL 的请求均收到 HTTP 200、但 content type 为 `text/html`、大小仅 1,817 bytes 的网页验证页面；替代 `articles/PMC9135475/bin/` URL 均为 HTTP 404。Europe PMC supplementaryFiles endpoint 返回 XML 错误 `Article ... is not open access one`。这些都不是附件字节，未写入 raw 也未进入下载 manifest。为避免把 HTML 伪装成 xlsx/pdf，错误页和本轮空目录均已删除。

按要求完整读取 computer-use 操作规范后，尝试以普通浏览器进行无登录访问；本次桌面环境没有可用或已连接的浏览器 surface（computer-use inventory 返回 `browsers: []`），因此不能在 UI 中点击 PMC 附件链接，也没有发生 CAPTCHA 求解、登录或 dbGaP 访问。此为本轮浏览器路径的具体技术阻断；并不改变命令行已观察到的附件存在性与验证页结果。

因此，未保存附件**不是已下载后去重**，也不存在可引用的 canonical 附件路径。本轮没有删除任何真实 Bhatia 原始附件；没有可下载的权威原包，故也不存在“zip 与解压副本”的取舍。后续若网页验证解除，应只保留官方原始 xlsx/pdf 附件，并以 SHA256 与本仓库做重名和内容哈希去重后再登记。

## 去重与失败尝试

在开始 Kuo 下载前，以 `kuo`、`pan_pdo`、`wcm` 搜索 `data/raw/` 和 `data/external/`，未发现既有同研究文件；本次 10 个 Kuo 文件均为新增且 SHA256 写入 manifest。没有发现可复用 canonical 文件，因而没有删除 Kuo 内容。全量 mutation callset 到手后，早期 TP53 单基因探针文件已作为冗余中间文件删除；空测试文件和 1,817-byte 验证页也未保留。

删除清单仅包括 Bhatia 三个 1,817-byte HTML 验证页、三个 48,716-byte HTML 404 页及一个 164-byte Europe PMC XML 错误响应；它们均为本轮新建、非数据文件、无 canonical 数据副本。未删除既有 raw/ 或历史资产，也未保留容器和同内容解压副本。

## 复现入口

- Kuo RNA 获取脚本：`data/artifacts/resource_acquisition_20260916/auxiliary/retrieve_kuo_cbioportal.py`（需 WSL2 Conda `drugscreening-gpu`）
- Kuo 文件 manifest：`data/artifacts/resource_acquisition_20260916/auxiliary/download_manifest.json`
- Kuo API study：`https://www.cbioportal.org/api/studies/pan_pdo_wcm_2026`
- Kuo RNA profile：`https://www.cbioportal.org/api/studies/pan_pdo_wcm_2026/molecular-profiles`
- Bhatia PMC：`https://pmc.ncbi.nlm.nih.gov/articles/PMC9135475/`
- Bhatia 代码仓库：`https://github.com/bhatia-sonam/manuscript-v1`
