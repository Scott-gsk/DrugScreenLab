# EXP-050 冻结20药的类器官药敏与转录响应取数设计

编制日期：2026-10-01。范围：方案3.6的可执行取数设计。此文件供后续实验团队核定；不代表已采购、已取样或已做湿实验。本设计保留已冻结候选的身份与次序，不读取Chen类器官药敏表中的数值，不发起对外联系。

## 1. 研究问题与当前可回答范围

核心问题是药物处理引起的基因表达变化能否从乳腺癌细胞系迁移到患者来源乳腺癌类器官，以及这些转录变化与实测药物反应是否相关。直接药敏仅作检验筛药价值的参照；同一通路变化不保证同一药效。

当前20药是按泛来源细胞系的训练集药敏高、低两端冻结的后续覆盖框架，不是乳腺类器官预测排序。来源端点为CTRPv2已发表AAC（药物剂量反应曲线下面积；依本项目登记，数值越高代表来源细胞系越敏感）。候选的来源对象是DepMap `PatientID` 来源单元，不是PDO患者。该候选框架为探索性，因为来源药敏曾在前期探索中暴露；它不能证明乳腺类器官药效，也不能用作独立验证结果。

目前20/20药物都有精确全InChIKey的LINCS扰动剂匹配及至少一个通过质控的乳腺来源暴露；其中16/20只覆盖MCF7这个独立`base_cell_id`，另4/20覆盖5个来源细胞。目标转录响应方面，20药候选与现有已登记、可核验全结构键的PDO处理响应药物交集为0/20。GSE216540含Fulvestrant处理表达，但其处理药物的精确结构身份仍未解析，不能用于精确键连接。故表内每个候选的PDO预测与配对药效、处理后RNA值均保持空缺。

## 2. 已冻结20药逐药证据与缺口

表中“AAC中位数 (n)”按每药训练侧独立来源对象汇总；`n`为不同DepMap `PatientID` 来源单元数。来源选择用药物级AAC中位数，先要求每药至少10个训练侧来源单元，再取最高和最低各10个；相同中位数按完整InChIKey（按化学结构区分药物的固定键）字典序打破平局。已冻结的高、低组次序即候选次序。表中LINCS `pert_id`是LINCS的药物扰动标识；`base_cell_id`代表独立细胞背景。LINCS剂量和时间是来源覆盖记录，不是PDO建议剂量；条件数是不同质控条件记录数，不是生物学重复数。官方乳腺细胞注释只有宽泛的carcinoma/adenocarcinoma字段，没有在本表推断PAM50等乳腺内在分子亚型。

| 组内序号 | 冻结候选 | 完整InChIKey / LINCS pert_id | 来源AAC中位数 (n) | 乳腺LINCS当前支持（独立来源细胞；质控暴露条件） | 现有PDO处理响应全键覆盖 | 仍缺少的同一PDO配对测量 |
|---:|---|---|---:|---|---|---|
| 高1 | UNC0321 | `AULLUGALUBVBDD-UHFFFAOYSA-N` / `BRD-K74236984` | 15.111 (24) | MCF7；2条：10 μM×6、24 h | 无（此候选全InChIKey未匹配） | PDO药敏剂量反应、匹配溶剂及同条件处理后RNA，均缺 |
| 高2 | darinaparsin | `JGDXFQORBMPJGR-YUMQZZPRSA-N` / `BRD-K35723520` | 15.000 (16) | MCF7；2条：10 μM×6、24 h | 无（此候选全InChIKey未匹配） | PDO药敏剂量反应、匹配溶剂及同条件处理后RNA，均缺 |
| 高3 | CI-976 | `WAFNZAURAWBNDZ-UHFFFAOYSA-N` / `BRD-K88544581` | 14.913 (21) | MCF7；2条：80 μM×6、24 h | 无（此候选全InChIKey未匹配） | PDO药敏剂量反应、匹配溶剂及同条件处理后RNA，均缺 |
| 高4 | temozolomide | `BPEGJWRSRHCHSN-UHFFFAOYSA-N` / `BRD-K32107296` | 14.787 (23) | MCF7；18条：9个记录剂量，0.0015–10 μM，6、24 h | 无（此候选全InChIKey未匹配） | PDO药敏剂量反应、匹配溶剂及同条件处理后RNA，均缺 |
| 高5 | PLX-4032 | `GPXBXXGIAQBQNI-UHFFFAOYSA-N` / `BRD-K56343971` | 14.766 (24) | BT20、HS578T、MCF7、MDAMB231、SKBR3；40条：0.08、0.4、2、10 μM，6、24 h；每来源细胞8条 | 无（此候选全InChIKey未匹配） | PDO药敏剂量反应、匹配溶剂及同条件处理后RNA，均缺 |
| 高6 | veliparib | `JNAHVYVRKWKWKQ-CYBMUJFWSA-N` / `BRD-K87142802` | 14.717 (23) | MCF7；18条：9个记录剂量，约0.0015–10 μM，6、24 h | 无（此候选全InChIKey未匹配） | PDO药敏剂量反应、匹配溶剂及同条件处理后RNA，均缺 |
| 高7 | AC55649 | `HXBKPYIEQLLNBK-UHFFFAOYSA-N` / `BRD-K93176058` | 14.66525 (22) | MCF7；2条：10 μM×6、24 h | 无（此候选全InChIKey未匹配） | PDO药敏剂量反应、匹配溶剂及同条件处理后RNA，均缺 |
| 高8 | purmorphamine | `FYBHCRQFSFYWPY-UHFFFAOYSA-N` / `BRD-K73397362` | 14.662 (18) | MCF7；4条：10、44.4 μM×6、24 h | 无（此候选全InChIKey未匹配） | PDO药敏剂量反应、匹配溶剂及同条件处理后RNA，均缺 |
| 高9 | SKI-II | `ZFGXZJKLOFCECI-UHFFFAOYSA-N` / `BRD-K20755323` | 14.617 (21) | MCF7；26条：记录剂量范围0.0015–40 μM，6、24 h | 无（此候选全InChIKey未匹配） | PDO药敏剂量反应、匹配溶剂及同条件处理后RNA，均缺 |
| 高10 | ML050 | `YOLTZIVRJAPVPH-UHFFFAOYSA-N` / `BRD-A83255679` | 14.5885 (22) | MCF7；4条：10、20 μM×6、24 h | 无（此候选全InChIKey未匹配） | PDO药敏剂量反应、匹配溶剂及同条件处理后RNA，均缺 |
| 低1 | BI-2536 | `XQVVPGYIWAGRNI-JOCHJYFZSA-N` / `BRD-K64890080` | 6.84995 (26) | BT20、HS578T、MCF7、MDAMB231、SKBR3；46条：0.08、0.3704、0.4、1.1111、2、3.3333、10 μM，6、24 h；MCF7 14条，其余每来源细胞8条 | 无（此候选全InChIKey未匹配） | PDO药敏剂量反应、匹配溶剂及同条件处理后RNA，均缺 |
| 低2 | narciclasine | `LZAZURSABQIKGB-AEKGRLRDSA-N` / `BRD-K06792661` | 7.2289 (15) | MCF7；18条：9个记录剂量，约0.0015–10 μM，6、24 h | 无（此候选全InChIKey未匹配） | PDO药敏剂量反应、匹配溶剂及同条件处理后RNA，均缺 |
| 低3 | AZD8055 | `KVLFRAWTRWDEDF-IRXDYDNUSA-N` / `BRD-K69932463` | 8.3700 (27) | BT20、HS578T、MCF7、MDAMB231、SKBR3；40条：0.08、0.4、2、10 μM，6、24 h；每来源细胞8条 | 无（此候选全InChIKey未匹配） | PDO药敏剂量反应、匹配溶剂及同条件处理后RNA，均缺 |
| 低4 | MLN2238 | `MXAYKZJJDUDWDS-LBPRGKRZSA-N` / `BRD-K78659596` | 8.8145 (23) | MCF7；8条：0.3704、1.1111、3.3333、10 μM×6、24 h | 无（此候选全InChIKey未匹配） | PDO药敏剂量反应、匹配溶剂及同条件处理后RNA，均缺 |
| 低5 | CAY10618 | `SJOLTIOPWDLDEB-UHFFFAOYSA-N` / `BRD-K83289131` | 9.2654 (23) | MCF7；2条：10 μM×6、24 h | 无（此候选全InChIKey未匹配） | PDO药敏剂量反应、匹配溶剂及同条件处理后RNA，均缺 |
| 低6 | ML162 | `UNVKYJSNMVDZJE-UHFFFAOYSA-N` / `BRD-A36275421` | 9.56335 (17) | MCF7；13条：记录剂量约0.04–10 μM，6、24 h | 无（此候选全InChIKey未匹配） | PDO药敏剂量反应、匹配溶剂及同条件处理后RNA，均缺 |
| 低7 | parbendazole | `YRWLZFXJFBZBEY-UHFFFAOYSA-N` / `BRD-K02407574` | 9.7205 (23) | MCF7；10条：0.31、0.3704、1.1111、3.3333、10 μM×6、24 h | 无（此候选全InChIKey未匹配） | PDO药敏剂量反应、匹配溶剂及同条件处理后RNA，均缺 |
| 低8 | NVP-BEZ235 | `JOGKUKXHTYWRGZ-UHFFFAOYSA-N` / `BRD-K12184916` | 10.3405 (20) | MCF7；20条：记录剂量约0.0015–10 μM，6、24 h | 无（此候选全InChIKey未匹配） | PDO药敏剂量反应、匹配溶剂及同条件处理后RNA，均缺 |
| 低9 | tanespimycin | `AYUNIORJHRXIBJ-TXHRRWQRSA-N` / `BRD-K81473043` | 10.6415 (21) | MCF7；2条：0.35 μM×6、24 h | 无（此候选全InChIKey未匹配） | PDO药敏剂量反应、匹配溶剂及同条件处理后RNA，均缺 |
| 低10 | selumetinib | `CYOHGALHFOKKQC-UHFFFAOYSA-N` / `BRD-K57080016` | 10.819 (23) | BT20、HS578T、MCF7、MDAMB231、SKBR3；57条：记录剂量约0.0015–80 μM，6、24 h；MCF7 25条，其余每来源细胞8条 | 无（此候选全InChIKey未匹配） | PDO药敏剂量反应、匹配溶剂及同条件处理后RNA，均缺 |

当前不能为任何一行填写“预测PDO药敏”“预测PDO处理转录变化”或“已验证机制通路”。这类值是缺失/未计算，绝不能写作0。LINCS剂量、时长和来源线只是药物有来源侧扰动资料的证据，不是PDO剂量建议，也不等于乳腺患者反应。

乳腺LINCS覆盖回执仅证明这20药在来源侧有可读的药物处理条件，并不提供本表可直接引用的逐药PDO机制结论。现有最接近的目标/来源转录响应比较是另外两种机制参照DOX和BTZ：冻结的MCF7比较中，Hallmark集合内排序在BTZ上较接近（目标23/47小时的中位Spearman相关为0.602/0.639），在DOX上较弱（0.190/0.087）；目标剂量未知，LINCS为6/24小时，且方向反例仍存在。两者均不在这20药面板中，所以不替代20药逐药机制证据，也不改变候选顺序。GSE288798的JQ1/MS177同药比较因精确LINCS键和RNA尺度门未通过而不提供跨平台机制证据。

### 现有处理响应键的边界

- GSE287429：多柔比星（DOX，`AOJJSUZBOXZQNB-TZSSRYMLSA-N`）及硼替佐米（BTZ，`GXJABQQUPOEUTA-RDJZCZTQSA-N`）有冻结结构键和处理RNA；20候选均无精确键交集。
- GSE288798：JQ1（`DNVXATUJJDPFDM-KRWDZBQOSA-N`）和MS177（`LRMNSTSNSCJHGZ-UHFFFAOYSA-N`）由样本标签解析到PubChem结构键，但两者在现行LINCS扰动剂登记均无精确键和乳腺条件，RNA.data转换来源也未确认；该单一PDO对象不构成独立患者重复。20候选均无精确键交集。
- GSE216540：有Fulvestrant对DMSO的处理表达，但当前登记的结构身份尚未解决；不作名称近似匹配或结构键推断。此项不改变20药交集计数。
- GSE303509只有基线表达档案；已发表文章中少数选定案例仅为同研究药敏定性参照，不是药物处理后RNA，也不是候选药物的配对定量PDO药敏测量。

故“0/20”表示当前登记的、可核验精确化学键的PDO处理RNA记录中，没有一个药物能与冻结20药面板精确对应；它不表示这些药物在PDO中无效，也不表示已有阴性转录响应。跨研究类器官编号没有经过对象交叉表核实，不按相似编号连接。

## 3. 后续真实测量的可执行字段链

同一药物、剂量、时间、独立PDO来源、培养批次和处理/对照必须用显式键连接。不得按文件行序、药物名称近似匹配、相似PDO编号或孔号单独连接。

| 数据层 | 最低字段（保留原始记录及版本） | 用途与约束 |
|---|---|---|
| 冻结药物面板 | `panel_id`、候选组/序号、显示名、全InChIKey、LINCS `pert_id`、isomeric SMILES及标准化规则版本 | 固定20药身份及已冻结顺序；同义名仅作显示，连接用完整键 |
| 实际药品与批次 | 供应方、产品/目录号、批号、质量分析证书（CoA）、纯度、对应盐/溶剂化物/立体异构、原始化学结构或SDF、储存与稳定性、stock溶剂、stock浓度、制备/冻融记录 | 到货后核对实物与面板全结构键；本任务不查询供应商、不采购、不联系 |
| 生物对象 | 去标识患者/供体 `patient_id`、独立来源 `independent_origin_id`、PDO `pdo_id`、同患者衍生系/克隆关系、病理/分子亚型及来源、取材部位、建系和传代、培养基/基质及批次、身份与污染质控 | `independent_origin_id` 是生物学统计单位；同患者多株、传代、冻融、孔和技术重复不得拆成独立患者 |
| 实验分组 | `study_id`、预先冻结的患者级队列角色、实验/板/孔/培养批次ID、随机化顺序、盲态码、处理状态、候选全InChIKey、剂量及单位、起止时间/暴露时长、vehicle种类及终浓度 | 每个药物处理须有同患者来源、同培养条件、同时间、同批次的匹配溶剂对照；记录板位、操作和批次效应 |
| 直接药效 | 原始读数、背景孔、活性/存活测定方法与时间、完整浓度序列、每孔值、拟合模型版本、曲线质量标记、预登记主终点（例如曲线汇总端点）、未定义/拟合失败原因 | 药敏端点为参照结果；不得用单剂量值冒充剂量反应，不能把转录扰动或基因依赖当药敏标签 |
| 处理前RNA | 同PDO来源、同传代/培养批次的处理前基线样本及对应样本ID、取样时间、测序批次、原始计数、参考基因组/注释/定量流程版本、文库与测序质控 | 输入侧只使用处理前信息；拆分数据时按患者/独立来源整体分组 |
| 处理后RNA及配对 | 同一 `patient_id`/`pdo_id`/`passage`/`culture_batch_id` 下的药物样本和匹配vehicle样本ID、剂量、时间、药物批次、RNA提取/建库/测序ID、原始计数和所有测序QC字段 | 通过 `(patient_id, pdo_id, full_InChIKey, dose, time, culture_batch_id, biological_replicate_id)` 构造处理响应键；记录处理减匹配vehicle的基因变化及其不确定性 |
| 预测及分析输出 | 冻结模型/代码/基因轴/尺度变换版本、预测生成时间、预测输入样本ID、`predicted_delta_expression`、预测药效值（若有已验证映射）、实测变化、实测药敏、通路/基因集版本及有效基因数 | 预测与实测按完整键连接；当前预测栏必须留空，不能从LINCS来源结果或AAC中位数填补 |

**剂量、时间、规模与重复数尚待实验团队确定并书面冻结**：每药剂量曲线的点数与范围；药敏终点和读数时点；用于转录响应的预设早期采样时间（及是否需多个时点）；PDO培养格式、每孔/每样本所需组织量和20药面板可运行容量；每个校正队列和独立验证队列的独立患者数；每个处理的生物学重复和技术复孔数；车辆比例、批间平衡与测序深度/质控阈值。不能照搬LINCS的6或24小时和来源剂量。技术复孔仅用于实验误差估计，不能替代独立PDO患者数。

在开始前应按实验团队确认的容量，先以不进入正式验证的材料确定可测剂量窗、药敏测量稳定性和早期转录采样时间；随后冻结剂量、时间、随机化、排除/QC标准和配对规则。若某药实物身份、溶解/稳定性、培养基兼容或足量来源无法满足，不可凭LINCS有条件就推定可执行；记录具体缺口并由授权负责人决定如何处理，不能私自换药或改候选顺序。

## 4. 校正与独立验证拆分及一致/不一致样本选择

1. **先按患者/独立来源拆分。** 校正/开发队列与独立验证队列在读取药敏或处理RNA结果前确定；同患者衍生PDO、克隆、培养批次和重复孔只能属于同一队列。当前历史研究之间没有可靠的一对一PDO交叉表，因此不做编号匹配；未来只按来源/招募/样本谱系证明独立性，不能仅凭标签不同推断独立。
2. **先冻结预测，再揭示药敏。** 任何“预测PDO转录响应”必须只以处理前表达、候选结构和通过准入的固定模型产生，并保存输入轴、尺度变换、时间/剂量合同与运行版本。任何“预测药敏”还需一项仅在校正数据或独立来源训练并冻结的转录响应到药效映射。单独预测基因变化不能自动解释为药效预测。
3. **分开两个一致性问题。** 转录一致性比较预测基因/功能变化与匹配处理后实测RNA变化；药效一致性比较冻结的药效预测与独立实测剂量反应。两者不是同一终点。通路方向或基因签名一致，不能单独证明敏感性一致。
4. **仅在校正队列做富集取样。** 在查看该队列处理后RNA之前，固定预测、药效终点、反应阈值和两类一致性规则。待药敏曲线通过预登记QC后，按预定分类列出“预测与实测同向/同类”和“预测与实测相反”的PDO-药物组合；若校正数据不足以固定可靠的效应映射/界值，或候选预测值尚未通过准入，则一致/不一致字段保持空白，不进行预测驱动挑样。若测序容量有限，在每个预定一致性层内按固定随机种子抽样，兼顾AAC高、低两组及不同PDO来源；保存全体符合条件的清单、入选概率、随机种子和未测原因。富集的校正集仅用于机制探索，不能报告为无偏总体迁移或药效准确率。
5. **独立验证不按结果挑样。** 在独立患者中，应优先对预先列定的组合全部做配对RNA；若容量限制，必须在药敏结果揭盲前按候选组和基础可测性分层随机抽样。不得因预测吻合或不吻合、药敏强或弱、通路结果突出而增减验证样本。药敏曲线可在全体预设组合测量；若RNA采样只能覆盖其子集，应保存抽样概率，并把富集样本与独立验证估计分开，必要时按已登记抽样概率加权。
6. **按独立PDO来源计不确定性。** 预定主分析报告每个患者的药物反应、匹配实测/预测基因变化、功能方向与幅度；患者等权汇总，配对重采样以患者为单位。孔、基因、单细胞数或多个条件不能增加独立生物学分母。药物间差异、RNA测量和药效测量的失败值/缺失分母单列，不静默填零。

现阶段本节只定义未来规则，不产生任何PDO预测或一致性标签。由于目标预测当前未通过尺度、剂量/时间和覆盖门槛，20行的预测字段应保持空值。候选之间的来源AAC高低也不能代替患者内实测药效分类。

## 5. 复核入口、可复算统计与证据文件

冻结候选与乳腺来源覆盖应由以下权威文件复算；v1候选文件定义20药顺序，v3文件只追加覆盖信息，未变更候选键或排序。

- 方案与边界：`docs/exec-plans/active/乳腺癌转录响应迁移与筛药实施方案_20260930.md`第3.6节；`experiments/records/EXP-050.md`；`configs/experiments/EXP-050.yaml`中的`source_sensitivity_candidate_frame_v1`与`candidate_breast_source_coverage_v3`。
- 来源AAC逐药数据：`artifacts/experiments/EXP-050/20261001_source_sensitivity_candidate_frame_v1/candidate_panel_source_reference.tsv`；统计定义、训练侧来源分母、禁读项和输入输出哈希：同目录`candidate_coverage_summary.json`与`run_manifest.json`。来源配对表为`artifacts/experiments/EXP-050/20261001_lincs_ctrp_transfer_association_v3/patient_level_pairs.parquet`，固定身份划分为`artifacts/experiments/EXP-007/CCLE_CONTEXT_IDENTITY_SPLIT.json`。
- 乳腺LINCS键、细胞及条件：`artifacts/experiments/EXP-050/20261001_candidate_breast_source_coverage_v3/candidate_panel_breast_source_coverage.tsv`、`candidate_breast_source_exposure_conditions.tsv`、`summary.json`、`run_manifest.json`。审计只按完整InChIKey连至`data/raw/lincs/GSE92742/GSE92742_Broad_LINCS_pert_info.txt.gz`，按官方`base_cell_id`计独立来源，并从通过质控的`data/processed/lincs/GSE92742/continuous_conditions_v2/conditions.parquet`取全部乳腺条件。
- 已登记处理响应药物全键：DOX/BTZ见`artifacts/experiments/EXP-050/20260930_keyed_response_v7/EXP050_drug_identity_crosswalk.tsv`；JQ1/MS177及其LINCS与尺度门见`artifacts/experiments/EXP-050/20261001_gse288798_compound_eligibility_v2/gse288798_compound_identity_scale_eligibility.tsv`与`summary.json`。GSE216540药物身份未解决的状态见`configs/experiments/EXP-050.yaml`中`GSE216540_FULV`说明及`experiments/records/EXP-050.md`。
- 关键复算：按完整全InChIKey核对候选表与覆盖表，确认20个候选键唯一、v3顺序与v1一致、`same_drug_pdo_transcript_key_match=False`为20/20；AAC只报冻结训练侧来源对象的中位数及分母，不对来源值做PDO外推；LINCS暴露条件数不当作患者数。上述检查只复核已登记只读产物，不访问PDO IC50数值，不使用外层/锁定评测结果。

本方案的验收数据仍待真实配对测量：同一个独立PDO来源上的处理前表达、剂量反应药敏、多剂量/预设时间处理的转录变化及匹配溶剂对照。目标PDO预测、药物机制迁移和药效关系在取得这些数据并满足预注册分析条件前均为未知。

