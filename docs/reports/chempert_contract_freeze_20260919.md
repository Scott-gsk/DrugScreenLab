# ChemPert 合同冻结报告（2026-09-19）

- 状态：**FROZEN**；本合同支撑 **EXP-023 正式实验**（扰动转录建模侧的数据边界、主键、泄漏隔离与对照可用性以本报告产物为准）
- 执行：DSL-数据管家（TRAE general-purpose 子 Agent）
- 环境：WSL2 Conda `drugscreening-gpu`；`data/raw/` 全程只读；未训练、未评测

## 结论

ChemPert 合同四件产物已冻结到 `data/processed/chempert/` 并登记 `data/registry/datasets.json`：

| 产物 | 路径 | 规模 | sha256（前 12 位） |
|---|---|---|---|
| 主键表 | `chempert_primary_keys_FROZEN.parquet` | 82,270 行 × 15 列 | `70ff2aab349d` |
| 主键 manifest | `chempert_primary_keys_FROZEN_manifest.json` | 含全部源 checksum | `c0694514dc04` |
| 泄漏键表 | `chempert_leakage_keys_FROZEN.json` | 1,923 命中化合物全列 | `337bbb294068` |
| landmark 合同 | `chempert_landmark_contract_v1/` | 143 行元数据 + 795×69 表达切片 | manifest 内逐文件 |

主键口径核对：responses CSV 82,270 行（唯一 Response_ID），与 responses RDS 82,270 行一致；RDS 侧 14 个版本化 ID 变体（`RIDxxxxx.1`）已由 schema audit 记录，**以 CSV 为权威身份源**。

## 分组与泄漏统计（4.2）

- accession 分组：GSE92742 34,702 / GSE70138 42,978 / other 4,590（共 343 个 accession）。
- LINCS 训练化合物集合（不重新下载）：`GSE92742_Broad_LINCS_inst_info.txt.gz` 全部 1,319,138 行（与 exact978_cache_v1 一一对齐）中 `pert_type=trt_cp` 的 **19,812** 个 pert_iname（pert_id 20,415；pert_info trt_cp 登记 20,413 行佐证）。
- 交集通道：ChemPert `Chemical_ID` 为 ChemPert 内部顺序号（如 CID01776=Adapin，非 PubChem CID 3158），无法做 ID 直连；采用**小写化合物名精确匹配**（`Chemical_Compound` ↔ `pert_iname`）。
- 命中：ChemPert 2,566 个化合物中 **1,923** 个与 LINCS 训练集重合；响应级 **69,498/82,270** 条重合（GSE92742 组 34,702 条**全量命中**——该组本就是 LINCS Phase I 整理子集；GSE70138 组 32,317 条；other 组 2,479 条）。
- 复合原子：chemical×cell 命中原子 8,559 对；细胞源交集 12 个（ASC, CD34, FIBRNPC, HA1E, HEK293T, MCF10A, NEU, NKDBA, NPC, PHH, SKB, SKL）。
- **隔离规则（冻结）**：凡 `lincs_overlap_chemical=true` 的化合物，其全部响应必须与 LINCS 训练用途同侧整组隔离，不得跨集；GSE92742/GSE70138 组建议整 accession 隔离。

## 对照可用性 / delta_supervision（4.3）

数据模型事实（探查结论，写入 manifest）：

- 每条响应恰好指向一个 `Expression_ID`（表达矩阵列）；矩阵为 dgCMatrix **12,266 基因 × 5,028 样本**，行名=gene symbol，列名=Matrix_Colname（CSV `Matrix_Index` 与列位 1-based 对齐，0 mismatch）。矩阵有 11 列未被 CSV 登记（penciclovir/valacyclovir 毒理处理样本，无响应引用）。
- 表达谱三类：明确对照标注（control/DMSO/vehicle 等，78 个谱）、多化合物共享基线谱（77,676 条响应指向，如 LINCS 板级谱 LJP005/AML001）、单化合物一对一谱（4,451 条）。

**delta_supervision 规则（冻结）**：响应的 Expression_ID 解析到明确对照标注谱 → true。结果：**true 143 / false 82,127**（Mouse 109 / Human 28 / Rat 6）。字段已并入主键表，同时给出 `expression_profile_kind` 区分共享基线 vs 一对一谱。

## landmark 交集小合同（4.4）

- 基因空间：ChemPert 矩阵 12,266 symbol ∩ LINCS landmark 978 symbol = **795** 个基因（同 symbol 口径直接交集；跨物种同名噪声已知并记录）。
- 物化：143 条 delta_supervision=true 记录的元数据合同（17 列 parquet）+ 69 个对照谱在 795 landmark 基因上的 dense 表达切片（795×69 parquet，非零率 86.8%）。
- **阻塞项（如实记录）**：ChemPert v2 下载中**不存在任何一个 accession 同时提供对照谱与处理谱**（已逐一验证：11 个含对照谱的研究全部无处理谱，其余研究无对照谱）。因此「对照与处理表达」的配对 delta 在包内不可得；delta 读出以 responses RDS 的 up/down 基因列表（symbol 口径，82,270×5 已解码）为准。本合同物化了对照臂切片 + 全部 true 记录元数据，未伪造处理臂。

## 逐项关闭 scout 缺口（route2_chempert_contract_scout_DRAFT.md）

1. **无冻结扰动基因合同** → 关闭：主键表 + delta_supervision + landmark 合同冻结；化合物分离由泄漏键表钉死。
2. **expression/targets RDS 未转换** → 部分关闭：expression RDS 已解码（dgCMatrix，slot 级 i/p/x/Dimnames），并按 landmark 合同切片；targets RDS 全表转换仍未做（本合同不需要，保持 parser-tree 可读状态）。
3. **泄漏键表未冻结** → 关闭：`chempert_leakage_keys_FROZEN.json`（DRAFT 保留只读，FROZEN 为权威）。
4. **无 processed/chempert 物化树** → 关闭：已建立并登记 datasets.json 三条目（primary_keys / leakage_keys / landmark_contract）。
5. **CSA 药敏基准混淆风险** → 关闭：全部产物 manifest 声明 data_boundary（仅扰动转录参考，无 viability/AUC/IC50），registry 条目 intended_role 与 IMPROVE CSA 条目互斥；ChemPert 扰动标签与 CSA 药敏基准严格分离。

## 测试与校验

- `PYTHONPATH=src conda run -n drugscreening-gpu python -m drug_screen.data.registry --root data` → **PASS**
- `pytest tests/data/test_registry.py tests/data/test_depmap_24q2_registry.py` → **4 passed**

## 遗留与建议

- 名称匹配通道存在同义词漏配风险（如 Tretinoin 在 LINCS 侧用别名）；对泄漏防护是「少隔离」方向的风险，EXP-023 建议对 GSE92742/GSE70138 整组隔离以规避。
- ChemPert 数据许可仍未由官方明示（audit 遗留）；EXP-023 启动训练前需完成许可复核。
- 143 条 true 记录中 Mouse 109 条：人 landmark symbol 与鼠同名基因的直接交集是已知噪声源，建模侧按物种分层时注意。
- 生成器：`scripts/data/freeze_chempert_contract.py`（可重跑复现）；结构探针：`scripts/data/probe_chempert_expression_rds.py`。
