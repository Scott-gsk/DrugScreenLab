# 路线二：ChemPert 正式药条件扰动实验报告（EXP-023）

- 日期：2026-09-19
- 状态：**EXPLORATORY**（数据许可待复核，见限制一节）
- 执行：DSL-扰动建模 Agent（TRAE general-purpose 子 Agent）
- 脚本：`scripts/run_route2_chempert_formal_EXP023.py`（seed 23，GPU 0 / NVIDIA RTX A5000，耗时 361 s）
- 产物：`artifacts/route2_chempert_EXP023_DRAFT/`（metrics.json、per_response_scores.parquet）
- 实验记录：`experiments/records/EXP-023.md`（预登记 + 合同 v2 训练前登记 + 结果）

## 一、结论（先行）

1. **主终点达成**：两个协议（P1 test=GSE92742、P2 test=GSE70138）下，三个药条件臂相对 ctrl_only 的配对 Δ mean cosine 95% CI **全部为正、不跨零**（Δ 点估计 +0.039 ~ +0.049）。预登记的失败条件（CI 跨零或为负）**未触发**。
2. **判定表述**：在 ChemPert 设定下观察到药条件头相对 ctrl_only 的显著增益，如实报告为**跨库（LINCS 之外）药特异扰动信号（EXPLORATORY）**。
3. **与 LINCS（EXP-022）结论不矛盾**：两库任务结构根本不同。LINCS 为一对一配对对照（ctrl_only 主导，0.67）；ChemPert v2 的对照谱被多个化合物共享（中位 36.5 个化合物/谱），ctrl_only 对同一谱的全部响应输出同一预测、结构性不可区分（cosine ≈ 0.01），**药信息是唯一区分信号**。两库结论是任务结构差异的两种结果，合并表述时必须区分（见第六节）。
4. **绝对预测力有限**：P1 所有模型臂未超过 global_mean_signature 参考基线（0.0712）；P2 仅 ctrl_morgan（0.0629）超过（0.0517）。签名到 795 landmark 空间的覆盖率仅 13–16%，目标稀疏是主要瓶颈。

## 二、数据合同 v2 增补（训练前登记）

训练前可用性复核（spec Task 4.5）发现书面证据表明 v1 合同对「对照谱」解析过于保守（v1 仅 143 条 true）。依据以下三条证据创建 **v2 增补**（严格可加，v1 FROZEN 文件不动；先登记于 EXP-023.md 再启动训练，测试集未事后改动）：

1. **官方文档语义**：ChemPert 官方文档（chempert.uni.lu/information，2026-09-19 检索）将表达谱资源定义为 "gene expression profiles of initial cellular states"（初始细胞状态谱，即处理前对照）。
2. **结构必然性**：496 个 shared_multi_chemical_baseline 谱各被 2–265 个不同 Chemical_ID 引用（中位 36.5），处理后谱不可能同时是多个化合物的处理结果。
3. **LINCS 板级交叉验证**：GSE92742 组 272/272 谱名与 GSE92742 inst_info 的 rna_plate 精确匹配（板含 ctl_vehicle/DMSO 孔）；GSE70138 组 225 个谱带显式孔位（如 LJP005_HA1E_24H_X1_B19）且每个被 ≥16 个化合物共享。

**v2 规则**：`delta_supervision_v2 = true ⇔ expression_profile_kind ∈ {control_labeled, shared_multi_chemical_baseline}` → **77,819 条 true**（GSE92742 34,698 / GSE70138 42,978 / other 143）。single_chemical_profile（4,451 条，含化合物名+剂量的一对一谱）语义存疑，保守保持 false。

**产物**（`data/processed/chempert/chempert_v2_manifest.json`，含 sha256；生成器 `scripts/data/chempert_contract_v2_supplement.py`；已登记 `data/registry/datasets.json` 条目 `chempert_contract_v2_supplement`）：

| 产物 | 规模 |
|---|---|
| v2 标记表 | 82,270 行（77,819 true） |
| 对照谱切片（795 landmark） | 795 基因 × 565 谱（非零率 98.4%） |
| 响应表（含 up/down 基因列表） | 77,819 × 17 |
| SMILES 桥（LINCS pert_info canonical_smiles，名称匹配） | 2,508 Chemical_ID，1,921 有 SMILES |

SMILES 桥的名称匹配通道与泄漏键一致（小写化合物名 ↔ pert_iname），覆盖 66,600/77,819 条 v2 响应；结构不可编码的 18 个化合物仅在 ctrl_morgan 臂排除并登记，其余臂用全量。

## 三、实验设计与口径（与 EXP-022 消融同口径）

- **输入**：对照谱（795 landmark 基因，train 统计 z-score），按响应的 Expression_ID 取列。
- **目标**：responses RDS 的 up/down 签名映射到 795 空间（up=+1 / down=-1 / 其他=0）。映射覆盖率：up 平均 7.97/61.58 个基因（13.0%），down 8.75/56.06（15.6%）；543 条全零签名行从训练与评测两侧剔除（计数保留在 coverage）。
- **臂**：
  - `ctrl_only`：Linear(795→795)，仅对照谱；
  - `ctrl_drug_id`：Emb(64) + MLP(512)，对照谱 + 药 ID 嵌入；
  - `ctrl_morgan`：Morgan 1024-bit（radius=2，与 EXP-022 ablation_fp 同口径）+ MLP(512)，仅在 SMILES 可编码子集训练评测；
  - `drug_only`：Emb(64)→MLP(512)，对照删除消融；
  - 参考基线：`null_zero`（全零预测）、`global_mean_signature`（训练集平均签名）。
- **超参**：emb 64 / hidden 512 / batch 512 / lr 1e-3 / wd 1e-4 / 40 epochs（MLP 容量与 EXP-022 同量级）。
- **主指标**：逐响应 cosine 后取均值（pearson 并列报告，结论一致，表略）。
- **统计**：配对 Δ(臂 − ctrl_only) 的化合物级分组 bootstrap 95% CI（2,000 次，percentile 法）。

## 四、划分与泄漏隔离

按冻结泄漏键合同的**accession_group 整组隔离**（GSE92742 组与 LINCS 训练化合物 100% 重合，必须整组隔离；GSE92742/GSE70138 为 ChemPert 内两个 LINCS 整理子集，互不重叠）：

| 协议 | test 组 | train / test | 药词表 | test 未见药行（UNK） | test 化合物（训练可见） |
|---|---|---|---|---|---|
| P1 | GSE92742 | 42,728 / 34,548 | 1,034 | 19,461（56.3%） | 1,844（640） |
| P2 | GSE70138 | 34,664 / 42,612 | 1,854 | 12,116（28.4%） | 1,023（639） |

## 五、结果

### 逐臂 mean cosine

| 协议 | null_zero | global_mean | ctrl_only | ctrl_drug_id | drug_only | ctrl_morgan* |
|---|---:|---:|---:|---:|---:|---:|
| P1（test=GSE92742） | 0.0000 | 0.0712 | 0.0102 | 0.0544 | 0.0567 | 0.0493 |
| P2（test=GSE70138） | 0.0000 | 0.0517 | 0.0121 | 0.0530 | 0.0550 | 0.0629 |

### 配对 Δ(臂 − ctrl_only) 与分组 bootstrap 95% CI（2,000 次）

| 协议 | Δ ctrl_drug_id | Δ drug_only | Δ ctrl_morgan* |
|---|---|---|---|
| P1 | +0.0442 [0.0385, 0.0525] | +0.0465 [0.0378, 0.0573] | +0.0391 [0.0320, 0.0482] |
| P2 | +0.0409 [0.0318, 0.0524] | +0.0429 [0.0316, 0.0565] | +0.0489 [0.0362, 0.0633] |

*ctrl_morgan 仅在 SMILES 可编码子集上评测（P1 test 34,250 行 / P2 test 31,917 行；可编码响应合计 66,582/77,819），其 Δ 与 ctrl_only 在同一子集行上配对。

**主终点判定**：六个 Δ CI 全部为正、不跨零 → 失败条件未触发。

## 六、与 LINCS（EXP-022）结论的合并表述

| | LINCS（EXP-022） | ChemPert（EXP-023） |
|---|---|---|
| 对照结构 | 一对一配对对照（每实例自己的 ctrl） | 共享基线谱（中位 36.5 化合物/谱） |
| ctrl_only 表现 | 0.67，主导 | ≈0.01，结构性不可区分 |
| 药条件增益 | 无（Δ 跨零或为负，「对照主导」） | 有（Δ CI 全正） |
| 结论 | 对照信息主导 | 药信息是唯一区分信号 |

两库结论不矛盾：当对照携带实例级区分信息时对照主导（LINCS）；当对照被多药共享、不含药特异信息时，药信息成为可学习的唯一信号（ChemPert）。**不得**把其中任一结论外推为「药条件编码无价值/有价值」的普适结论。

值得注意的附加证据：P1 测试 56.3% 行为训练未见药（UNK 嵌入），ctrl_morgan 对未见药纯靠结构泛化仍 +0.0391，为结构编码的泛化信号提供了独立支持；P2 中 ctrl_morgan（0.0629）是唯一超过 global_mean 基线（0.0517）的模型臂。

## 七、限制

1. **数据许可待复核**（EXPLORATORY 的根因）：ChemPert 下载页未明示许可，audit 遗留未关闭；本结果限研究用途、不重分发。
2. **签名覆盖率低**：up/down 列表到 795 空间仅 13.0%/15.6%，目标稀疏，压低所有臂的绝对 cosine；这不是模型间比较的偏差源，但限制绝对预测力的解读。
3. **同义词漏配风险**：化学品名称匹配通道（泄漏键 caveat）同样适用于 SMILES 桥，方向为「少匹配」，对 ctrl_morgan 是保守偏倚。
4. **global_mean 对比**：P1 所有模型臂未超 global_mean_signature（0.0712），说明该协议下模型尚未学到超过「平均响应」的信息；增益结论仅相对 ctrl_only 成立。
5. **v2 增补为合同变更**：虽经三方书面证据支持且训练前登记，仍是 v1 冻结口径之上的增补，复现时须同时引用 v1 FROZEN 与 v2 manifest。

## 八、下一步建议

1. 关闭 ChemPert 许可复核后，再考虑将本结论从 EXPLORATORY 升级（前置条件）。
2. 若追求绝对预测力，瓶颈在签名覆盖率：可评估登记一个「更大目标基因空间」的合同增补（需重新冻结，另行登记）。
3. 多种子复跑与未见药分层报告可作为后续登记项（本次单 seed 23，主终点 CI 已不跨零）。
4. 与 LINCS 结论合并进项目级综述时，按第六节的双行表述，不混用来源。

## 九、产物与复现

- 指标：`artifacts/route2_chempert_EXP023_DRAFT/metrics.json`（含模型配置、覆盖率、逐协议逐臂数字与 CI）
- 逐响应分数：`artifacts/route2_chempert_EXP023_DRAFT/per_response_scores.parquet`
- 合同 v2：`data/processed/chempert/chempert_v2_manifest.json` 及四个 parquet（datasets.json 条目 `chempert_contract_v2_supplement`）
- 复现命令：WSL2 Conda `drugscreening-gpu`，`python scripts/run_route2_chempert_formal_EXP023.py --gpu 0 --seed 23`
