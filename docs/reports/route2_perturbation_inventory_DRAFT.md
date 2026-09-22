# 路线二（转录扰动）摸底清单 DRAFT

- 状态：**DRAFT**（摸底文档；未启动训练 / 未跑 GPU）
- 生成时间：2026-09-18 14:05 CST（本机回填版）
- 依据主方案：`docs/研究实施方案.md`（第六、七章为主；第八章验收交叉引用）
- 执行计划：`docs/exec_plan_active.md` / `docs/exec-plans/`（P1 可并行摸底；本周不做完整 GPU 训练）
- 本机路径：`D:\Code\DrugScreenLab`（machineId `96b95440-6d33-4ea8-a3f3-6284d6770786`，DESKTOP-LEP36QM）
- 本文件落盘：`D:\Code\DrugScreenLab\docs\reports\route2_perturbation_inventory_DRAFT.md`
- 运行环境线索：WSL2 + conda `drugscreening-gpu`（见 `PROJECT_STATE.yaml`）；本轮禁止大训练

---

## 0. 可达性与盘点范围

| 项 | 结果 |
|---|---|
| 项目根 | **已确认** `D:\Code\DrugScreenLab` |
| git | **是**；`origin https://github.com/Scott-gsk/DrugScreenLab.git`；分支 `master`；工作区很脏（大量 D 删除 + ?? untracked） |
| 顶层 | `.git` `artifacts` `configs` `data` `docs` `experiments` `mvp` `scripts` `src` `tests` `tmp` `PROJECT_STATE.yaml` `environment.yml` … |
| 盘点方式 | 本机 PowerShell 只读 `dir` / `Select-String` / `Get-Content`；**未**启动训练 |

---

## 1. 依据（第六、七章要点摘录）

### 1.1 第六章链路

```text
基线表达 + 药物信息 → 转录变化 → 通路变化 / 疾病逆转评分 → 候选药物排序
```

1. **实测扰动基准**：LINCS（GSE92742）与 ChemPert；化/遗分开；L3 差值与 L5 签名分开；ChemPert∩LINCS 防跨集合泄漏；无可靠对照不做 Δ 监督。
2. **变化与药敏关系**：扰动强度 / 通路 / 疾病逆转三类解释对照。
3. **模型阶梯**：无变化与均值参照 → Ridge → 小型条件化网络 → TranSiGen → XPert；评价重点是 **Δexpression**。
4. **固定逆转公式排序**：基因负余弦、排名富集、通路反向一致；**实测与预测评分分开报告**。

### 1.2 「实测与预测评分分开报告」

两套独立结论，而非并列表：

1. 实测扰动 → 固定评分 → 与药敏关联（检验规则本身）
2. 预测扰动 → 同一公式 → 再关联（落差归因预测误差 / PDO 迁移）

### 1.3 第七章接口

联合前先独立比较直接药敏、扰动/逆转、简单联合；联合第二层用内层 CV 防训练分数泄漏。

### 1.4 验收交叉（第八章）

必须通过 LINCS/ChemPert 无重复跨集合泄漏；扰动交付含实测基准、表达变化预测、逆转验证；预测改善但排序未改善 → 只支持转录预测能力。

---

## 2. 数据登记（LINCS / ChemPert）— 本机实测

### 2.1 Registry 条目（`data/registry/datasets.json`）

| id | 相对路径 | 备注 |
|---|---|---|
| `lincs_gse92742_raw_level3_v1` | `raw/lincs/GSE92742` | GEO/Broad；含 Level3 gctx 等 |
| `lincs_gse92742_level3_level4_level5` | `interim/lincs/GSE92742` | 解压/中间层 |
| `lincs_gse92742_exact978_cache_v1` | `processed/lincs/GSE92742/exact978_cache_v1` | 978 landmark 缓存 |
| `lincs_prism_identity_bridge_v1` | mvp/artifacts 桥 | 响应盲身份桥 |
| `chempert_public_download_20260916_v1` | `raw/resource_acquisition_20260916/improve_chempert` | 2026-09-16 获取 |

`DATA_CATALOG.md` 与 registry 一致：LINCS 与 ChemPert 均标为**主要资产**。

### 2.2 磁盘落地（已 ls）

**LINCS raw** `data/raw/lincs/GSE92742/`（约 127 GB 量级，与目录说明一致）：

- Level3 / Level4 / Level5 三个 `.gctx.gz` **均在盘**（L3≈52.4GB，L4≈53.3GB，L5≈21.3GB 压缩）
- `gene_info` / `pert_info` / `cell_info` / `inst_info` / `sig_info` / `sig_metrics` / `SHA512SUMS` / `SHA256SUMS.local`
- 目录 `interim/lincs/GSE92742/`、`processed/lincs/GSE92742/` **存在**

**ChemPert raw** `data/raw/resource_acquisition_20260916/improve_chempert/`：

- `chempert_gene_expression_profiles.rds`、`chempert_transcriptional_responses.rds`、targets RDS
- 3 个 information CSV；`improve_csa_data_zenodo_*.zip`；metadata JSON
- `data/processed/.../chempert`：**目录检索时未见已物化处理树**（或为空）— 与目录「RDS 未完全转为常用格式」一致
- 审计产物：`data/artifacts/resource_acquisition_20260916/improve_chempert/`（含 schema audit 线索）

**目录说明要点（DATA_CATALOG）**：

- ChemPert 82,270 响应中约 34,702 可溯至 GSE92742，约 42,978 溯至 **GSE70138**（未重复拉取）；官方只声明关系、不重写原始数据 → **跨集合防泄漏表仍需显式工程化**
- Registry 对 raw Level3 条目 files 校验可能不完整（L4/L5 等在盘但登记缺口）— 目录已写明，属数据管家跟进项

### 2.3 XPert 资产（路线二阶段 3 相关）

| 路径 | 状态 |
|---|---|
| `data/external/xpert_source/` | 上游源码树在盘（`train_xpert.py` 等） |
| `data/external/xpert_assets/` | `all_drugs_unimol_arr.zip` / `processed_data.zip` / `saved_model.zip` 大包在盘 |
| `mvp/foundation/xpert/` | **结果/登记保留**（EXP005_FAST、ADAPTER/REGISTRY JSON、`CONTEXT_ADAPTER_TRACK.md`） |

`PROJECT_STATE.yaml`：`extension_status.foundation: IMPLEMENTATION_REMOVED_RESULTS_RETAINED`

---

## 3. 代码盘点 — 本机实测

### 3.1 仍在盘、与 ChemPert/LINCS 相关

| 路径 | 角色 | 完整度判断 |
|---|---|---|
| `scripts/data/materialize_chempert_metadata.py` | ChemPert 元数据物化 | 在盘（untracked/新） |
| `scripts/data/audit_improve_chempert_acquisition.py` | ChemPert/IMPROVE 获取审计 | 在盘 |
| `scripts/data/audit_improve_contract.py` | 合同审计（含 ChemPert 引用） | 在盘 |
| `src/drug_screen/data/p0.py` | 数据层命中 LINCS/相关 | 在盘 |
| `mvp/foundation/xpert/*` JSON/md | XPert 基金会试结果与登记 | **结果保留** |
| `mvp/core_eval/observed_oracle/` | 实测 oracle 汇总 JSON | **结果保留** |
| `mvp/core_eval/predicted_reversal/` | 预测逆转汇总 JSON | **结果保留** |

### 3.2 工作区已删除、历史存在（git status `D`）— 实现层缺口

以下对路线二关键，**脚本/模块当前 MISSING**，但部分结果目录仍在：

| 缺失路径（示例） | 原角色 |
|---|---|
| `mvp/core_eval/compute_observed_oracle.py` | 实测 oracle |
| `mvp/core_eval/compute_predicted_reversal.py` | 预测逆转 |
| `scripts/audit_exp001_level3.py` | Level3 审计 |
| `scripts/data/build_lincs_prism_identity_bridge.py` | LINCS–PRISM 桥 |
| `scripts/foundation/build_xpert_*.py` / `evaluate_xpert_*.py` | XPert 适配与评价 |
| `src/drug_screen/foundation/xpert_*.py` | XPert 库代码 |
| `src/drug_screen/modeling/exp002.py` 等 | 早期扰动/建模 |

**解读**：路线二曾有 MVP-001 / EXP-00x / XPert 延伸实现；当前仓库处于「实现移除、结果与资产保留 + 迁移/药敏主线脚本激增」状态。Ridge→XPert/TranSiGen **可复现训练入口需从 git 历史恢复或按新方案重写**，不能假定 `src/` 现树可直接训。

### 3.3 TranSiGen / Ridge

- 本轮 `Select-String` 在 `src/scripts/mvp` 对 `TranSiGen` **无实质命中**（至少当前工作树）。
- Ridge 扰动基线：未见独立、现成的「路线二 Ridge Δexpression」入口文件名；需对照已删 `exp002`/`phase1`/`mvp001` 历史或新建。

### 3.4 泄漏检查

- 方案硬约束明确；ChemPert↔LINCS 重叠在目录中有**数量级描述**，未见单独的「冻结防泄漏划分表」文件在本轮被确认落盘为正式 splits 产物。
- `data/splits/` 目录存在 — 是否含 ChemPert∩LINCS 专用划分：**待数据管家/下一轮只读抽查**（本 DRAFT 未展开逐文件）。

---

## 4. 实测 vs 预测评分是否已分开

| 层 | 结论 |
|---|---|
| 方案 | 必须分开 |
| MVP-001 产物 | **已分开目录**：`mvp/core_eval/observed_oracle/` vs `mvp/core_eval/predicted_reversal/` |
| 计算脚本 | **已删除**（results retained） |
| 路线二重跑能力 | 需恢复脚本或重实现后才能再生产；现有 JSON 可作历史对照，**不宜**直接当本周新实验结论 |

---

## 5. 项目状态与路线二定位

摘自 `PROJECT_STATE.yaml`（2026-09-18 附近）：

- `active_experiment`: EXP-016 验证 / EXP-019 完成语境；下一 `EXP-020`
- `mvp_status`: 受控验证下探索模型；PDO 迁移未建立
- `foundation: IMPLEMENTATION_REMOVED_RESULTS_RETAINED`
- 本周执行口径（协调）：路线二 = **P1 摸底**，不做完整 GPU 大训练

---

## 6. Git / 远程

| 项 | 值 |
|---|---|
| remote | `https://github.com/Scott-gsk/DrugScreenLab.git` |
| branch | `master`（跟踪 origin/master） |
| 注意 | 工作区大量删除与未跟踪文件；任何恢复实现应先与协调方确认，避免与药敏迁移主线冲突 |

---

## 7. 风险与阻塞

1. **实现层空洞**：XPert/逆转/Level3 审计等脚本已从工作树删除，仅余资产与历史 JSON。
2. **ChemPert 处理态**：raw RDS/CSV 在盘，常用矩阵物化不完整；与 LINCS 重叠防泄漏表未冻结为正式实验合同。
3. **Registry 校验缺口**：L4/L5 等在盘 vs raw Level3 登记 files 不完整（目录已记录）。
4. **TranSiGen 缺失**：当前树未见可用入口。
5. **混用风险仍在**：L3/L5、化/遗、跨源泄漏 — 数据虽大而全，合同未冻结前不可开训。
6. **算力诱惑**：XPert zip/权重与 LINCS 百 GB 级数据已在盘，易误触发大训练 — 执行计划明确本周不做。

---

## 8. 建议的下一步（仍勿训练；单步）

**立即下一步**：在不恢复大训练的前提下，产出一页「路线二最小可复现合同 DRAFT」：

1. 冻结 LINCS 使用层（先 L3 landmark 978 cache **或** L5，二选一写死）与 ChemPert 子集键；
2. 写出 ChemPert∩LINCS 防泄漏键表生成脚本规格（可只写规格/伪代码，交数据管家实现）；
3. 列出需从 git 历史恢复的最小文件集合（`compute_observed_oracle` / `compute_predicted_reversal` / Level3 audit），或声明按新方案重写；
4. 明确实测 oracle vs 预测逆转继续分目录报告。

在合同与防泄漏表落地前：**不要**启动 Ridge 全量或 XPert/TranSiGen GPU 作业。

---

## 9. 附录：阶段验收速查

| 阶段 | 验收关注 | 本机现状（一句话） |
|---|---|---|
| 实测基准 | 对照可靠；L3/5 分离；化/遗分离；跨源无泄漏 | 原始数据齐；合同/泄漏表未冻结；处理脚本部分缺失 |
| 关系分析 | 强度/通路/逆转可并列 | 有历史 MVP 产物，现入口弱 |
| 预测模型 | Ridge→…→XPert；评 Δ | XPert 资产在；实现 REMOVED |
| 排序 | 三公式固定；实测 vs 预测分报 | 历史 JSON 已分目录；脚本待恢复 |

