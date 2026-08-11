# 附录 A：核心参考文献

本附录按照本项目实际技术模块组织，而不是单纯按照发表年份排列。建议正式论文/标书中优先引用 A1–A6 的核心文献，其余根据具体章节选用。

---

## A1. LINCS / Connectivity Map 与疾病表达谱逆转

### [R1] Subramanian A, et al.  
**A Next Generation Connectivity Map: L1000 Platform and the First 1,000,000 Profiles.**  
*Cell*. 2017;171(6):1437–1452.e17.  
DOI: `10.1016/j.cell.2017.10.049`

**本项目用途：**

- LINCS/L1000 数据体系的基础文献；
- 978 landmark genes 的实验依据；
- chemical/genetic perturbational signatures；
- Connectivity Map 疾病—药物表达关联理论基础。

该工作建立了超过百万 perturbational profiles 的新一代 CMap，是本项目 `LINCS → perturbation → reversal` 主线最核心的数据和理论来源。

---

### [R2] Chen B, et al.  
**Reversal of cancer gene expression correlates with drug efficacy and reveals therapeutic targets.**  
*Nature Communications*. 2017;8:16022.  
DOI: `10.1038/ncomms16022`

**本项目用途：**

- Disease Signature 构建；
- tumor-vs-normal differential expression；
- disease-expression reversal；
- reversal score 与真实药物 efficacy 的联系；
- 细胞系 phenotype validation 的理论依据。

该研究直接证明 cancer-expression reversal 与 breast、liver、colon cancer 的 preclinical drug efficacy 存在系统关联，是本项目“逆转是否真的意味着药效”最重要的前人依据之一。

---

### [R3] Zeng B, et al.  
**OCTAD: an open workspace for virtually screening therapeutics targeting precise cancer patient groups using gene expression features.**  
*Nature Protocols*. 2021;16:728–753.  
DOI: `10.1038/s41596-020-00430-z`

**本项目用途：**

- cancer-specific Disease Signature；
- disease/reference 数据整理；
- transcriptomics-based virtual screening；
- 可复现疾病逆转流程参考。

OCTAD特别适合作为 Disease Signature Engine 和公开 cancer transcriptome 筛药工作流的成熟参考。

---

### [R4] Wang Z, et al.  
**Extraction and analysis of signatures from the Gene Expression Omnibus by the crowd.**  
*Nature Communications*. 2016;7:12846.  
DOI: `10.1038/ncomms12846`

**本项目用途：**

- CREEDS disease/drug/genetic signatures；
- Disease Signature 外部 benchmark；
- 已整理疾病上调/下调基因的辅助验证。

CREEDS包含人工整理和质量检查的 disease-vs-normal、drug 和 genetic perturbation signatures，可作为我们自行构建 Disease Signature 后的 sanity check，而不是取代原始差异分析。

---

# A2. Context-specific Chemical Perturbation Prediction

### [R5] Guo Y, et al.  
**Modelling drug-induced cellular perturbation responses with a biologically informed dual-branch transformer.**  
*Nature Machine Intelligence*. 2026;8:96–112.  
DOI: `10.1038/s42256-025-01165-w`

**模型：XPert**

**本项目用途：**

- 主要 perturbation backbone 参考；
- untreated/pre-perturbation context；
- drug × context interaction；
- dose/time modelling；
- Δexpression prediction；
- cold-drug / cold-cell evaluation。

XPert直接建模 pre-perturbation state 和 drug-induced change，并同时预测 post-treatment expression 与 Δexpression，是当前方案最直接的架构依据。

---

### [R6] Qi X, et al.  
**Predicting transcriptional responses to novel chemical perturbations using deep generative model for drug discovery.**  
*Nature Communications*. 2024;15:9256.  
DOI: `10.1038/s41467-024-53457-1`

**模型：PRnet**

**本项目用途：**

- unseen chemical perturbation baseline；
- LINCS→sci-Plex chemical response modelling；
- context-conditioned perturbation；
- disease-signature virtual screening baseline；
- pretrained implementation参考。

PRnet同时在 bulk L1000 与 single-cell sci-Plex 上研究 novel chemical response prediction，并公开数据处理与模型实现。

**注意：**其 L1000 978→更大基因空间的传统 inference 不作为本项目 reliable gene expansion 的核心方案。

---

### [R7] Zheng MY, et al.  
**Deep representation learning of chemical-induced transcriptional profile for phenotype-based drug discovery.**  
*Nature Communications*. 2024;15:5378.  
DOI: `10.1038/s41467-024-49620-3`

**模型：TranSiGen**

**本项目用途：**

- perturbation denoising；
- self-supervised perturbation representation；
- phenotype-based drug discovery baseline；
- replicate/MODZ processing经验；
- Disease Reversal baseline。

TranSiGen说明 chemical perturbation 数据噪声本身是模型必须处理的问题，因此本项目不会假设所有 LINCS profiles 具有相同可靠度。

---

### [R8] Huang K, et al.  
**Deep learning prediction of chemical-induced dose-dependent and context-specific multiplex phenotype responses and its application to personalized Alzheimer’s disease drug repurposing.**  
*PLOS Computational Biology*. 2022;18:e1010367.  
DOI: `10.1371/journal.pcbi.1010367`

**模型：MultiDCP**

**本项目用途：**

- Drug + Gene + Basal Context + Dose 的 factorized modelling；
- unseen-cell/context prediction；
- gene representation；
- patient-specific disease reversal；
- PENDA 类个体化 Disease Signature参考。

MultiDCP是本项目 gene-aware perturbation modelling 最重要的直接前人证据之一。

---

# A3. Chemical / Genetic Unified Perturbagen Space

### [R9] Li Y, et al.  
**UniPert-G2CP bridges genetic and chemical screens from molecular representation to phenotype modeling.**  
*Cell*. 2026.  
DOI: `10.1016/j.cell.2026.06.005`

**本项目用途：**

- chemical/genetic perturbagen统一表示；
- genetic→chemical transfer；
- 类器官 genetic perturbation → chemical mechanism transfer；
- 缓解 organoid chemical perturbation transcriptomics 数据不足。

这是本项目类器官迁移设计最重要的基础文献之一。UniPert的核心价值不是普通 drug embedding，而是把 genetic 与 chemical perturbagens放到共享机制空间。

---

### [R10] Dixit A, et al.  
**Perturb-Seq: Dissecting Molecular Circuits with Scalable Single-Cell RNA Profiling of Pooled Genetic Screens.**  
*Cell*. 2016;167:1853–1866.e17.

**本项目用途：**

- Perturb-seq 方法学；
- genetic perturbation→whole-transcriptome response；
- cell-line genetic mechanism supervision；
- UniPert chemical/genetic联合训练的数据基础之一。

对应公开数据 **GSE90063 / SRP093670**。

---

### [R11] Norman TM, et al.  
**Exploring genetic interaction manifolds constructed from rich single-cell phenotypes.**

**本项目用途：**

- 单基因和组合 genetic perturbation；
- perturbation latent/mechanism learning；
- downstream genetic-response validation。

对应公开 Perturb-seq 数据 **GSE133344 / SRP212114**，GEO同时提供 expression matrix 和 perturbation identities。

---

# A4. Target Context 与 Knowledge Prior

### [R12]  
**Predicting and interpreting cell-type-specific drug responses in the small-data regime using inductive priors.**  
*Nature Machine Intelligence*. 2026.  
DOI: `10.1038/s42256-026-01202-2`

**模型：PrePR-CT**

**本项目用途：**

- untreated target context；
- cell-type-specific co-expression network；
- unseen-cell-type perturbation prediction；
- 类器官 target-context prior。

其最有价值的思想是：目标 biological context 不需要拥有大量 drug labels，也可以利用 untreated expression建立 context-specific inductive prior。

---

### [R13] Wenkel F, et al.  
**TxPert: using multiple knowledge graphs for prediction of transcriptomic perturbation effects.**  
*Nature Biotechnology*. 2026.  
DOI: `10.1038/s41587-026-03113-4`

**本项目用途：**

- biological knowledge graph prior；
- basal context + perturbation representation；
- unseen genetic perturbation；
- cross-cell generalization；
- STRING/GO 等公开知识的模型化参考。

TxPert支持“structured biological knowledge 可以成为 OOD perturbation prediction 的归纳偏置”这一核心判断。

---

# A5. Reliable Gene Expansion

### [R14] Xing J, et al.  
**Deep-learning-based de novo discovery and design of therapeutics that reverse disease-associated transcriptional phenotypes.**  
*Cell*. 2026;189:2556–2572.e19.  
DOI: `10.1016/j.cell.2026.02.016`

**模型：GPS**

**本项目用途仅限：**

- gene-feature conditioned prediction；
- non-landmark candidate gene expansion；
- independent whole-transcriptome per-gene calibration；
- transcriptomic reversal screening的方法学参考。

GPS论文报告先判断 landmark gene predictability，再使用 gene biological features 扩展 prediction universe，并通过独立 WTS 数据筛选可靠 genes。

**项目特别说明：**

GPS在本项目中属于：

**方法学参考，而不是可信 pretrained dependency。**

不直接使用：

- GPS checkpoint；
- GPS最终 gene list；
- GPS论文报告的固定 predictable-gene 数量。

最终 Reliable Extended Gene Set 必须通过本项目自己的独立 WTS calibration产生。

---

# A6. Drug–Target–Pathway 与 Cell line→Organoid 机制迁移

### [R15] Kong J, et al.  
**Network-based machine learning in colorectal and bladder organoid models predicts anti-cancer drug efficacy in patients.**  
*Nature Communications*. 2020;11:5485.  
DOI: `10.1038/s41467-020-19313-8`

**本项目用途：**

- Drug→Target→PPI→Pathway；
- target-proximal pathway；
- organoid pharmacogenomics；
- mechanism-based cross-system biomarker；
- Cell line/Organoid/Patient 之间机制迁移的依据。

该研究从 known drug targets 出发，在 STRING PPI 网络中筛选 target-proximal pathways，再利用 organoid pharmacogenomics进行药效建模，并进行了患者与独立 cell-line验证。

这是本项目把：

**Drug–Target–Pathway**

提升为 Cell line→Organoid **机制桥梁而非事后解释模块**的核心依据。

---

# A7. Cell-line Drug-response 与真实筛药验证

### [R16] Corsello SM, et al.  
**Discovering the anticancer potential of non-oncology drugs by systematic viability profiling.**  
*Nature Cancer*. 2020;1:235–248.  
DOI: `10.1038/s43018-019-0018-6`

**资源：PRISM**

**本项目用途：**

- 大规模细胞系 drug-screening efficacy benchmark；
- Top-K drug-ranking validation；
- candidate library phenotype calibration。

PRISM公开资源包含4,518个 drugs × 578 human cancer cell lines 的 primary viability screen，并另有多剂量 secondary screen。

---

### [R17] Seashore-Ludlow B, et al.  
**Harnessing Connectivity in a Large-Scale Small-Molecule Sensitivity Dataset.**  
*Cancer Discovery*. 2015;5:1210–1223.  
DOI: `10.1158/2159-8290.CD-15-0235`

**资源：CTRPv2**

**本项目用途：**

- independent cell-line drug-response benchmark；
- cross-study ranking validation；
- dose-response phenotype。

该研究测量860个 cancer cell lines 对481种 compounds 的多浓度 response，并建立 CTRPv2。

---

### [R18] Garnett MJ, et al.  
**Systematic identification of genomic markers of drug sensitivity in cancer cells.**  
*Nature*. 2012;483:570–575.  
DOI: `10.1038/nature11005`

**资源：GDSC**

**本项目用途：**

- 独立 pharmacogenomic drug-response validation；
- cross-study testing；
- cell-line efficacy benchmark。



---

### [R19] Barretina J, et al.  
**The Cancer Cell Line Encyclopedia enables predictive modelling of anticancer drug sensitivity.**  
*Nature*. 2012;483:603–607.  
DOI: `10.1038/nature11003`

**资源：CCLE**

**本项目用途：**

- cell-line basal transcriptome；
- lineage/context metadata；
- pharmacogenomic matching。



---

### [R20] Bernett J, et al.  
**Critical evaluation of drug response prediction models with DrEval.**  
*Nature Communications*. 2026;17:4238.  
DOI: `10.1038/s41467-026-72903-w`

**本项目用途：**

- strict unseen-drug evaluation；
- biologically meaningful splits；
- cross-study evaluation；
- CCLE/CTRP/GDSC统一 preprocessing；
- 防止 drug/cell mean shortcut 与 leakage。

DrEval还公开了标准化后的 CCLE、CTRP、GDSC 等 benchmark 数据及代码，因此非常适合作为本项目 cell-line efficacy evaluation framework 的直接基础。

---

# A8. Whole-transcriptome Chemical Perturbation

### [R21] Srivatsan SR, et al.  
**Massively multiplex chemical transcriptomics at single-cell resolution.**  
*Science*. 2020;367:45–51.  
DOI: `10.1126/science.aax6234`

**资源：sci-Plex**

**本项目用途：**

- real whole-transcriptome chemical perturbation；
- non-landmark gene supervision；
- dose/context modelling；
- pathway-response supervision。



---

# 附录 B：公开参考数据集与知识库清单

## B1. 数据使用等级

所有数据固定划分为四类：

**TRAIN**  
允许用于模型参数学习。

**CALIBRATION**  
允许用于阈值、gene reliability 或 rank calibration，但不能参与对应模块的训练。

**EXTERNAL TEST**  
整个相关训练过程完全不可见。

**KNOWLEDGE PRIOR**  
用于 drug-target/pathway/gene annotation，不作为 expression ground truth。

同一个数据集在不同实验中可以有不同角色，但必须在实验开始前冻结，不能看到结果以后重新划分。

---

## B2. Chemical Perturbation / Transcriptome 数据

| 数据集 | 公开编号 | 主要内容 | 本项目角色 | 建议等级 |
|---|---|---|---|---|
| **LINCS Phase I** | **GSE92742** | 1,319,138 L1000 profiles；chemical + genetic perturbations；Level-2含直接测量978 genes | Δ978核心训练；chemical/genetic perturbation；dose/time/context | **TRAIN / P0** |
| **LINCS Phase II** | **GSE70138** | 354,123 L1000 profiles；chemical/genetic perturbations | 独立 LINCS perturbation validation；或后期扩充训练 | **EXTERNAL TEST优先 / P0** |
| **sci-Plex** | **GSE139944；重点 GSM4150378** | single-cell chemical perturbation；A549/MCF7/K562 等；whole-transcriptome | non-landmark直接监督；WTS/pathway training | **TRAIN / P0** |
| **PANACEA** | **GSE186341** | 32 kinase inhibitors × 11 cancer lines；1,728 RNA-seq；matched vehicle controls；2 replicates | independent WTS gene/pathway calibration | **CALIBRATION / P0** |
| **Lung cancer drug multi-omics** | **DRP006006 / PRJDB6952** | 23 lung cancer lines、95 compounds；3,240 RNA-seq；另有多剂量/多时间子设计 | 第二WTS external calibration；dose/time和cross-context validation | **EXTERNAL TEST / P1** |

GSE92742官方同时提供直接测量978基因的 Level-2 文件和包含 inferred genes 的更高层文件，因此本项目明确把978测量空间与 inferred-gene space区分处理。 GSE70138同样直接提供978-gene Level-2 数据。

sci-Plex原始数据和 processed matrices均公开，且 GSE139944 中明确包含 sciPlex3 A549/MCF7/K562 screen。 PANACEA GSE186341则具有明确 drug identity、DMSO/untreated controls 和重复实验，非常适合独立 WTS calibration。

DRP006006包含23个 lung cancer cell lines、95 compounds以及3,240个 RNA-seq datasets，其中还包括多浓度和24/48/72 h时间设计。

---

# B3. Genetic Perturbation 数据

| 数据集 | 公开编号 | 数据类型 | 本项目用途 | 等级 |
|---|---|---|---|---|
| **LINCS genetic perturbations** | GSE92742 / GSE70138 | shRNA/OE等 L1000 perturbation | chemical/genetic shared perturbagen initial training | TRAIN / P0 |
| **Perturb-seq** | **GSE90063 / SRP093670** | pooled CRISPR + scRNA；K562等 | genetic mechanism→transcriptome supervision | TRAIN / P1 |
| **Norman Perturb-seq** | **GSE133344 / SRP212114** | 单/组合基因过表达 + scRNA | perturbagen mechanism和genetic-response latent | TRAIN / P1 |
| **CRISPRai Perturb-seq** | **GSE220974** | CRISPRa + CRISPRi pairwise perturbation，K562 | activation/inhibition方向性机制辅助 | OPTIONAL / P2 |

GSE90063公开约20万细胞的 pooled CRISPR Perturb-seq，并同时提供raw和processed expression matrices。 GSE133344则公开单基因及组合 perturbation 的 expression matrices与cell identities。

---

# B4. Cell-line Basal Context

| 资源 | 内容 | 用途 | 等级 |
|---|---|---|---|
| **CCLE / DepMap** | cancer cell-line RNA-seq、mutation、CNV、lineage metadata | untreated biological context；LINCS/PRISM/CTRP/GDSC cell mapping | TRAIN/context / P0 |
| **DepMap Model metadata** |统一model identifier、lineage、sample metadata | 跨数据集 cell-line harmonization | infrastructure / P0 |

当前 DepMap公开 release提供 cell-line expression、mutation、copy number、CRISPR和drug-screen等下载数据；CCLE数据也由DepMap持续托管。

**实施要求：**

模型中固定保存具体使用的 DepMap release 版本，不能随着官网更新而悄悄替换数据。

---

# B5. Cell-line Phenotype / Drug Efficacy

| 数据集 | 规模/内容 | 本项目用途 | 推荐角色 |
|---|---|---|---|
| **PRISM Repurposing Primary** | 4,518 compounds × 578 cancer lines | 首要大药库 screening benchmark | **TEST / P0** |
| **PRISM Secondary** | 1,448 compounds × ~500 lines，多剂量 | AUC/dose-response ranking | **TEST / P0** |
| **CTRPv2** | 481 compounds × 860 cell lines，16浓度 | cross-study validation | **TEST / P0** |
| **GDSC1/2** | >1,000 genetically characterized cancer lines与多种anticancer agents | 最关键 independent cross-study test之一 | **TEST / P0** |
| **CCLE pharmacology** | historical cell-line sensitivity data | supplementary benchmark | P1 |
| **DrEval standardized package** | CCLE、CTRP1/2、GDSC1/2等统一处理 | evaluation infrastructure、cross-study reproduction | **P0** |

PRISM primary/secondary文件可以直接从DepMap获得。 CTRPv2原始论文报告481 compounds、860 genetically characterized cancer cell lines和16-point concentration response。 GDSC数据对学术研究公开，并覆盖超过1,000个genetically characterized cancer cell lines。

**推荐固定的验证方式：**

```text
Development:
PRISM

External Test 1:
GDSC

External Test 2:
CTRP
```

再进行反向 cross-study 验证。

不把不同 study 的 absolute IC50/AUC直接拼成统一标签。

---

# B6. Disease Signature 数据

| 数据资源 | 类型 | 用途 | 等级 |
|---|---|---|---|
| **TCGA / GDC** | tumor + adjacent-normal RNA-seq | cohort-level cancer Disease Signature | P0 |
| **GEO disease cohorts** | disease vs normal RNA-seq/microarray | 非癌疾病或癌症外部signature | P0/P1 |
| **CREEDS** | curated disease/drug/gene signatures | Disease Signature benchmark/sanity check | P1 |
| **OCTAD workspace/resources** | cancer-specific transcriptomic virtual screening | disease signature pipeline参考 | P1 |
| **Matched normal organoids** | organoid tumor/normal pairs | patient/organoid-specific disease state | P0 when available |

Disease Signature应保存：

```text
gene
log2FC
p-value
FDR
direction
```

同时保留：

```text
Disease-up genes
Disease-down genes
Full ranked signed expression profile
```

具体显著性阈值由 development disease cohorts 做 sensitivity analysis 后冻结，而不是预设固定基因数。Chen等的癌症逆转工作和OCTAD均支持从 tumor-vs-normal differential expression构建疾病签名后进行药物逆转。

---

# B7. Drug–Target–Pathway Mechanism Knowledge

| 数据库 | 内容 | 在模型中的用途 | 类型 |
|---|---|---|---|
| **ChEMBL** | compound、target、bioactivity、assay | Drug→Target、affinity/evidence | KNOWLEDGE PRIOR / P0 |
| **Open Targets** | target–disease、drug–target与证据 | mechanism support、target-disease relevance | KNOWLEDGE PRIOR / P1 |
| **STRING** | protein–protein association network | Target→network neighborhood | KNOWLEDGE PRIOR / P0 |
| **Reactome** | curated biological pathways/reactions | target-proximal pathways；pathway rescue | KNOWLEDGE PRIOR / P0 |
| **Gene Ontology** | gene BP/MF/CC annotation | gene representation；GPS-style expansion；解释 | KNOWLEDGE PRIOR / P0 |

ChEMBL提供人工整理的compound–target–bioactivity资源。 Reactome明确提供公开下载、API、pathway hierarchy、gene mapping和标准格式数据。

这些资源共同形成：

```text
Drug
↓
Target
↓
STRING/PPI neighborhood
↓
Target-proximal Reactome/GO
↓
Structured Mechanism Prior
```

---

# B8. Organoid Genetic Perturbation 数据

| 数据集 | 数据内容 | 本项目角色 | 等级 |
|---|---|---|---|
| **GSE280506** | primary human 3D gastric organoid；CRISPR KO/i/a；single-cell CRISPR；cisplatin | **Organoid mechanism/context adaptation 主数据** | TRAIN / P0 |
| **GSE145308** | human intestinal organoid；CRISPR；ARID1A/SMARCA4；Wnt/TGFβ backgrounds；24 RNA-seq samples | intestinal/CRC context adaptation | TRAIN / P0 |
| **GSE167285** | 5 donors；SATB2 KO + control；10 bulk RNA-seq | unseen-donor genetic-response test | EXTERNAL TEST / P1 |
| **GSE241659** | PTEN KO human intestinal organoid RNA-seq | pathway/mechanism external validation | EXTERNAL TEST / P1 |

GSE280506明确包含CRISPR KO、CRISPRi、CRISPRa和single-cell CRISPR，并进行了针对1,952个DNA-binding proteins的cisplatin sensitization screen。

GSE145308包含WT、APC-mutant、APC-TP53-mutant intestinal organoids以及ARID1A/SMARCA4/TGFβ相关RNA-seq，适合研究相同扰动在不同initial contexts中的响应差异。

GSE167285具有5个独立donors的control/SATB2-KO配对结构，因此尤其适合作为donor-level generalization，而不应该浪费在主训练中。

GSE241659则提供PTEN-KO intestinal organoid及对应RNA-seq，并观察到包括mTORC1 activation在内的转录效应。

---

# B9. PDO Disease State / Drug Phenotype 数据

| 数据集 | 主要内容 | 本项目用途 | 等级 |
|---|---|---|---|
| **GSE117548** | 16 CRC PDO transcriptomes；>500 substances；近600万confocal images | **最终PDO drug-ranking主benchmark** | EXTERNAL TEST / P0 |
| **GSE64392** | 20 CRC patients living organoid biobank；多数有adjacent-normal organoid；expression + drug-screen capability | tumor/normal organoid Disease Signature；外部biology | P0 |
| **GSE244082** | 32 CRC PDO biobank；5-FU/cisplatin sensitivity/resistance；RNA-seq | resistance/response auxiliary validation | P1 |
| **GSE239386** | CRC PDO/CAF；>2,500 treatment experiments；signaling/DNA damage/cell cycle/apoptosis + limited scRNA | microenvironment/phenotype secondary test | P1 |

GSE117548公开16个人源CRC PDO的expression profiling，并对PDO使用超过500种substances进行高通量筛选，因此最适合最终药物排序验证，而不是训练 drug→transcriptome model。

GSE64392建立于20位CRC患者的living organoid biobank，多数患者同时拥有adjacent-normal organoid，而且主要CRC molecular subtypes均有代表，是构建organoid tumor-vs-normal disease state非常珍贵的数据。

GSE244082则包含CRC PDO chemosensitivity和transcriptomic resistance信息，适合作为secondary validation。 GSE239386重点提供PDO/CAF和治疗后的signaling、DNA damage、cell cycle及apoptosis phenotype，非常适合后续微环境扩展，但不应该成为当前主模型依赖。

---

# B10. 推荐的最终数据分工

为了防止后期“什么数据都拿来训练”，建议在项目启动时直接冻结下面这个版本。

### Core Training

```text
GSE92742
+
CCLE/DepMap basal context
+
UniPert pretrained representation
+
Drug–Target–STRING–Reactome/GO
```

负责：

**chemical/genetic mechanism + Δ978。**

---

### Whole-transcriptome Expansion Training

```text
sci-Plex / GSE139944
```

负责：

**真实 non-landmark chemical response supervision。**

---

### Extended-gene Calibration

```text
PANACEA / GSE186341
```

负责：

**per-gene predictability selection。**

---

### WTS External Confirmation

```text
DRP006006
```

负责：

**第二独立 whole-transcriptome generalization。**

---

### Cell-line Screening Validation

```text
PRISM
+
CTRPv2
+
GDSC1/2
```

负责：

**Disease Reversal 是否真正把 sensitive compounds 排到前面。**

这一阶段是进入类器官以前的硬 Gate。

---

### Organoid Adaptation

```text
GSE280506
+
GSE145308
```

负责：

**Mechanism × Organoid Context → Organoid transcriptional response。**

其中 UniPert 提供：

```text
Genetic ↔ Chemical
```

共享扰动机制空间；

Drug–Target–Pathway提供：

```text
Drug → Target → Pathway
```

结构化迁移桥梁。

---

### Organoid Mechanism External Test

```text
GSE167285
+
GSE241659
```

禁止进入对应 adaptation training。

---

### Final PDO Screening Test

```text
GSE117548
```

作为主要：

**Drug Ranking → Actual PDO Phenotype**

闭环数据。

GSE64392用于：

**tumor/normal organoid disease state**

以及独立 biological validation。

---

# B11. 数据冻结原则

最终实施时，每个数据集都必须在 registry 中固定至少以下字段：

```text
dataset_id
accession
download_source
download_date
version
checksum

sample_id
donor_id
cell_line / organoid_id
tissue
disease

perturbagen_id
canonical_compound_id
SMILES
target
dose
time

control_id
replicate_id
batch
plate

expression_platform
gene_id_version

task_role:
TRAIN
CALIBRATION
EXTERNAL_TEST
KNOWLEDGE_PRIOR
```

最重要的是：

> **同一个 donor、同一 compound structure、同一 biological replicate family 在 split 以前必须先完成 identity harmonization。**

否则模型的所谓 unseen-drug、unseen-context 和 unseen-organoid performance 都可能被数据泄漏污染。

---

# B12. 当前数据优先级

如果按真正实施顺序排序：

### P0：没有这些就不开工

**GSE92742**  
**CCLE/DepMap**  
**UniPert pretrained resources**  
**ChEMBL / STRING / Reactome / GO**  
**sci-Plex GSE139944**  
**PANACEA GSE186341**  
**PRISM / CTRPv2 / GDSC**  
**GSE280506**  
**GSE145308**  
**GSE117548**

### P1：重要独立验证

**GSE70138**  
**DRP006006**  
**GSE90063**  
**GSE133344**  
**GSE167285**  
**GSE241659**  
**GSE64392**  
**GSE244082**

### P2：后续扩展

**GSE220974**  
**GSE239386**  
其它 disease-specific GEO cohorts、microenvironment和patient-level datasets。

这样可以保证项目不是因为“公开数据很多”而无限扩张，而是围绕：

**978 perturbation → reliable gene expansion → disease reversal → cell-line screening → mechanism transfer → PDO screening**

这条闭环逐层增加数据。