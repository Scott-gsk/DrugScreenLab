# Kryeziu 患者外层 5 折 — 冻结检查清单（DRAFT）

- 状态：未冻结（`data/splits/kryeziu_patient_outer5fold_DRAFT.json` 仍标 `DRAFT`）
- 生成时间戳（本文）：2026-09-18 17:19 CST
- 依据：`docs/研究实施方案.md` §三.2；`docs/exec-plans/active/pathway_transfer_pdo_20260918.md`；评价干跑 `docs/reports/kryeziu_leakage_dryrun_DRAFT.md`

## 已满足（可进入冻结评审）

- [x] 划分单位为 `patient_id`；同患者全部 `sample_id` 同折
- [x] 外层 5 折；核实患者 test 覆盖 51 人、无跨折重复 test 患者
- [x] 折内 train∩test 患者/样本为空（评价干跑 5/5 通过）
- [x] 默认队列：S1 核实子集（51 患者 / 60 样本）；32 prefix-only 继续排除
- [x] 药物集合对齐 48 可编码单药（`kryeziu_drug_identity_48_DRAFT`）；Oxaliplatin 隔离不进结构模型
- [x] 未测响应不补零（协议层已声明；训练脚本须剔除未测药）
- [x] loader 可读：`load_processed_mcrc_v3` → processed-only（92/48/3183）

## 冻结前必须补齐

- [x] **内层 val 算法钉死说明**（已写入本节；JSON 物化仍待冻结时执行）
- [ ] **内层 val 列表物化进划分文件**（可选路径 A）：当前各折 `val_patient_ids` 为空；训练侧约定「仅从 `train_patient_ids` 再划患者级 val（推荐 4:1 / 嵌套 CV，外层 test 禁止参与选模）」。冻结时应二选一：
  1. 把固定 seed 的内层 val 患者列表物化进 JSON；或
  2. 在 splits meta 中冻结算法+seed（`random_seed` 已有）并禁止实现漂移
- [ ] **身份表正式版**：将核实/排除名单从 DRAFT 提升为 `data/processed/identity/` 下带 checksum 的冻结表（含 provenance）
- [ ] **药物表正式版**：`kryeziu_drug_identity_48` 去掉 `_DRAFT`，登记 registry checksum；确认 exclusions 列表（组合药/生物药/Oxaliplatin）一并冻结
- [ ] **表达矩阵版本钉死**：`indexed.npz` / gene 顺序 / `n_genes`（现训练常用 top-2000 或全基因组？）写入 split 或 companion manifest，禁止静默换基因集
- [ ] **泄漏复检签字**：冻结当刻重跑 `check_split_leakage`，结果附在 `docs/reports/` 并改状态为 FROZEN
- [ ] **重命名/状态位**：文件名与 JSON `status` 从 `DRAFT` → `FROZEN`；旧 DRAFT 保留只读副本
- [ ] **协调者验收**：scott 在频道确认冻结，建模/评价不得再改 folds

## 内层 val 算法钉死说明（2026-09-18 17:23 CST）

训练可继续用 DRAFT folds；下列算法视为**冻结前约定**，实现不得漂移。正式冻结时二选一物化进 JSON 或保留本算法+seed 引用。

### 算法（患者级，嵌套在外层 train 内）

- 外层：沿用 `kryeziu_patient_outer5fold_DRAFT.json` 的 `train_patient_ids` / `test_patient_ids`（`val_patient_ids` 当前为空属预期）。
- 内层 val：**只从该折 `train_patient_ids` 再划**；**禁止**使用任何 `test_patient_ids` 做选模、早停、特征筛选或阈值。
- 比例：患者级 **4:1**（train_inner : val），按 `patient_id` 分组，不按 sample 行随机。
- 种子：优先使用划分文件内 `random_seed`；若缺失则固定 **`20260918`**，并在训练回执中原样记录。
- 排序：先对 `train_patient_ids` **字典序排序**，再用 `numpy.random.RandomState(seed).permutation` 打乱后取最后 ⌈n/5⌉ 名患者为 val（保证可复现；n<5 时至少 1 名 val，n=1 时跳过内层 val 并在回执标注 `inner_val_skipped`）。
- 样本映射：val/train_inner 的 `sample_id` = 该患者在该折 `train_sample_ids` 中的全部样本。
- 与评价约定对齐：`docs/reports/kryeziu_leakage_dryrun_DRAFT.md`（外层 test 禁入选模）。

### 验收点

- [ ] 任一折：`val_patient_ids ⊂ train_patient_ids_outer` 且与 `test_patient_ids` 不相交
- [ ] 回执含 `random_seed`、每折 `n_train_inner_patients` / `n_val_patients`
- [ ] 冻结时：要么把每折 `val_patient_ids` 写回 splits JSON，要么在 meta 中引用本节算法文本的 checksum

## 明确不作为冻结条件（本轮）

- 不要求 32 prefix-only 患者进入主队列
- 不要求合并不同响应终点
- 不把前序 EXP 指标作为冻结依据

## 阻塞

- 无硬阻塞训练继续用 DRAFT；**若要宣称「固定划分」结果可复现验收，内层 val 物化/算法冻结与表达版本钉死仍是缺口。**
