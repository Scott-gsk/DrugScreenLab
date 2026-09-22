# Kryzei 数据合同冻结报告（2026-09-19）

- 状态：**FROZEN**（冻结条件除"协调者验收"外全部关闭）
- 冻结时间：2026-09-19 12:22:19（UTC+8）
- 执行：DSL-数据管家（TRAE 子 Agent）
- 执行脚本：`scripts/data/materialize_kryeziu_inner_val.py`（内层 val 物化）、`scripts/data/freeze_kryeziu_data_contract.py`（冻结与泄漏复检）
- 依据：`docs/reports/split_freeze_checklist_DRAFT.md`（冻结前缺口清单与内层 val 钉死算法）、`docs/reports/kryeziu_leakage_dryrun_DRAFT.md`（评价干跑）
- 机器可读回执：`data/splits/kryeziu_data_contract_freeze_receipt_20260919.json`

## 一、结论

1. Kryzei（mcrc_v3）患者外层 5 折划分已由 DRAFT 提升为 **FROZEN**；内层 val 患者列表按钉死算法（患者级 4:1、随机种子 20260918）物化进划分 JSON，可复现、不再依赖训练侧自行实现。
2. 身份表、药物表（48 结构确认单药）、admission（51 患者 / 60 样本口径）、表达矩阵版本合同（全基因组 19421 基因 / GDSC∩Kryzei 交集 19091 基因）全部去 `_DRAFT` 正式落盘，checksum 登记于本报告、冻结回执与 `data/registry/datasets.json`。
3. 冻结当刻泄漏复检：**5/5 折通过**（`check_split_leakage` 双向 + 样本级补充检查，患者/细胞/药物重叠均为空）。
4. 建模与评价自本报告起一律使用 FROZEN 划分；正式 loader 已改为优先读 FROZEN、回退 DRAFT。DRAFT 副本保留只读，未删除。
5. 唯一待办：**协调者验收**（第九节，留空待确认）。

## 二、冻结资产与 checksum

| 资产 | 路径 | sha256 |
|---|---|---|
| 冻结划分（外层 5 折 + 物化内层 val） | `data/splits/kryzeiuz_patient_outer5fold_FROZEN.json` | `b8a05e3d788c41a9601b4a3794504eeb58ee4d478d88aac0b060e4fa29f66494` |
| 患者身份表（92 per-sample） | `data/processed/identity/kryeziu_patient_identity_FROZEN.json` | `a78b5dc75cb0eda75004b10d5a455e03f1684a98e40016ebc6dffe126f430984` |
| 药物身份表（JSON） | `data/processed/identity/kryeziu_drug_identity_48.json` | `163691da26a354f12cce2cda3e9c6ae9ef3c47a986714ecfc3f9e5a12166d232` |
| 药物身份表（Parquet） | `data/processed/identity/kryeziu_drug_identity_48.parquet` | `692d59c654cb1aa1cecd55cca31743141ce9c4200c84c84b8d15604752a8b23b` |
| admission（DEVELOPMENT_ADMITTED） | `data/processed/identity/transfer/kryeziu_mcrc_target_admission.json` | `f35d436360b70c5d50889473854520c561d890d4208e7889724d8b4c531866b9` |
| 表达矩阵版本合同 | `data/splits/kryzeiu_expression_manifest_FROZEN.json` | `bc5577e360c5c92eedae54a2aa9a80d6726a767e25110012aaa72f5e67fa4950` |

**hash 依赖方向**（避免循环引用）：admission 嵌入 FROZEN 划分的 sha256；FROZEN 划分只按路径引用 admission（其 sha 登记于本表与 `datasets.json`）；表达 manifest 与划分文件同层（`data/splits/`）并互相引用。

**保留的 DRAFT 只读副本**（不删除、不再更新）：

| 副本 | 路径 | sha256 |
|---|---|---|
| 物化后 DRAFT 划分（folds 与 FROZEN 一致） | `data/splits/kryeziu_patient_outer5fold_DRAFT.json` | `16c11a6b9f724f220c5036db6c96d0cbb8be68bf73ecb37b766d7ea2175f854b` |
| 药物表 DRAFT（JSON） | `data/processed/identity/kryeziu_drug_identity_48_DRAFT.json` | `73098baf352d0f91643aa2ca520560df765343521dfaaff18dd4ce0cd134bc23` |
| 药物表 DRAFT（Parquet，与正式版字节一致） | `data/processed/identity/kryeziu_drug_identity_48_DRAFT.parquet` | `692d59c654cb1aa1cecd55cca31743141ce9c4200c84c84b8d15604752a8b23b` |
| admission DRAFT | `data/processed/identity/transfer/kryzeiu_mcrc_target_admission_DRAFT.json` | （保留原样） |
| 内层 val 物化回执 | `data/splits/kryeziu_inner_val_materialization_receipt.json` | `9a8982824d010176a53c2768c44ca6967dd3200d9a39776587f8c3c26f80b52e` |

## 三、内层 val 物化（路径 A：列表直接写入划分 JSON）

**算法**（与 `split_freeze_checklist_DRAFT.md` 钉死文本一致，实现不得漂移）：

- 只从该折 `train_patient_ids` 再划；外层 `test_patient_ids` 禁止参与选模、早停、特征筛选或阈值。
- 患者级 4:1（train_inner : val）；先按 `patient_id` 字典序排序，再 `numpy.random.RandomState(seed).permutation` 置换，取末 ⌈n/5⌉ 名患者为 val。
- 种子来源：划分 JSON 的 `random_seed`；缺失则固定 20260918。本次实际：JSON `random_seed=null` → **fallback 20260918**（`seed_source=fallback_fixed_20260918`），该 seed 已写回 FROZEN 文件。
- 样本映射：val / train_inner 的 `sample_id` = 该患者在该折 `train_sample_ids` 中的全部样本。

**每折统计**（患者数 / 样本数；核实队列合计 51 患者 / 60 样本）：

| 折 | 外层 train | 内层 train_inner | 内层 val | 外层 test |
|---|---|---|---|---|
| 0 | 40 / 47 | 32 / 39 | 8 / 8 | 11 / 13 |
| 1 | 41 / 48 | 32 / 39 | 9 / 9 | 10 / 12 |
| 2 | 41 / 49 | 32 / 40 | 9 / 9 | 10 / 11 |
| 3 | 41 / 49 | 32 / 39 | 9 / 10 | 10 / 11 |
| 4 | 41 / 47 | 32 / 37 | 9 / 10 | 10 / 13 |

内层 train = 外层 train − val（训练侧按 `val_patient_ids` 排除即可）；`inner_val_skipped` 全部为 false。

## 四、泄漏复检（冻结当刻）

每折执行三组检查：

1. `check_split_leakage`（外层 test vs 内层 train，`allow_drug_overlap=True`）
2. `check_split_leakage`（内层 val vs 内层 train，`allow_drug_overlap=True`）
3. 样本级补充检查：val 样本 ⊆ 外层 train 样本；val 样本 ∩ test 样本 = ∅；样本→患者映射一致

| 折 | 外层 test vs 内层 train | 内层 val vs 内层 train | 样本级补充检查 |
|---|---|---|---|
| 0 | PASS | PASS | PASS |
| 1 | PASS | PASS | PASS |
| 2 | PASS | PASS | PASS |
| 3 | PASS | PASS | PASS |
| 4 | PASS | PASS | PASS |

5/5 折通过；患者、细胞、药物重叠均为空，LINCS/CMap-ChemPert 跨划分重复为 0。

## 五、冻结前缺口逐项关闭

对照 `split_freeze_checklist_DRAFT.md`「冻结前必须补齐」：

| # | 缺口 | 状态 | 关闭方式 |
|---|---|---|---|
| 1 | 内层 val 算法钉死说明 | ✅ 已关闭 | checklist 已含钉死文本；正式物化脚本严格按该算法实现（字典序排序 + `RandomState(20260918).permutation` 取末 ⌈n/5⌉） |
| 2 | 内层 val 列表物化（路径 A） | ✅ 已关闭 | `materialize_kryeziu_inner_val.py` 原子写入各折 `val_patient_ids` / `val_sample_ids`（只填 val 字段，未动其他字段）；DRAFT 与 FROZEN 均含物化结果 |
| 3 | 身份表正式版 | ✅ 已关闭 | `kryeziu_patient_identity_FROZEN.json`：92 条 per-sample 记录（60 核实 + 32 prefix-only pending 排除名单）、患者级聚合、provenance checksum 完整 |
| 4 | 药物表正式版 | ✅ 已关闭 | `kryeziu_drug_identity_48.json` / `.parquet` 去 `_DRAFT`；exclusions 一并冻结（8 项组合药/生物药 + 2 项仅元数据列 + Oxaliplatin 隔离）；checksum 登记于 registry 与本报告 |
| 5 | 表达矩阵版本钉死 | ✅ 已关闭 | companion manifest（`data/splits/kryeziu_expression_manifest_FROZEN.json`，与划分文件同层互相引用）：`indexed.npz` sha256、全基因组 19421、交集 19091、`gene_order_sha256=0fd5fdd7…` 双侧核验一致，禁止静默换基因集 |
| 6 | 泄漏复检签字 | ✅ 已关闭 | 冻结当刻重跑 5/5 通过（第四节），结果附于本报告与冻结回执，状态改 FROZEN |
| 7 | 重命名/状态位 | ✅ 已关闭 | 新建 FROZEN 文件且 JSON `status=FROZEN`；旧 DRAFT 保留只读副本 |
| 8 | 协调者验收 | ⏳ 待办 | 第九节，留空待确认 |

钉死算法验收点核对（checklist「验收点」三问）：

- 任一折 `val_patient_ids ⊂ train_patient_ids` 且与 `test_patient_ids` 不相交：**5/5 通过**
- 回执含 `random_seed` 与每折 `n_train_inner_patients` / `n_val_patients`：**通过**（回执 `inner_val` 段 + 本报告第三节）
- val 列表写回 splits JSON（路径 A）：**通过**

## 六、冻结时输入校验（关键 checksum）

| 输入 | sha256 | 核验结果 |
|---|---|---|
| mcrc_v3 `indexed.npz` | `ddbab601e461169a7385ac0658a433a7735581e5e9b99bdfd062e9bec5f4c727` | 92×19421，与 mcrc_v3 manifest 一致 |
| GDSC v3 `indexed.npz` | `2b27c47b72f53d01d1acc609f13f3b6dac96f92542acdc128fb9d22f3a6b68c7` | 与基因空间合同一致 |
| 交集基因有序表 | `0fd5fdd7ab9d53d569bef5c2a91b508c62d6cb9d1000bc3aeac6f5d424424b39` | 19091 基因，双侧矩阵均包含（`gdsc_kryeziu_intersection_v1`） |
| 身份审计 | `244c159fad55eea3738fa6e7e28be220e084cc00151a2118be31142e58dc08d2` | 60 核实样本 / 32 pending 样本；51 核实患者 |
| 物化后 DRAFT 划分 | `16c11a6b9f724f220c5036db6c96d0cbb8be68bf73ecb37b766d7ea2175f854b` | `n_folds=5`，各折 val 非空 |
| 内层 val 物化回执 | `9a8982824d010176a53c2768c44ca6967dd3200d9a39776587f8c3c26f80b52e` | `all_folds_ok=true` |

## 七、使用说明

- **loader**：`src/drug_screen/data/loaders_kryeziu_DRAFT.py` 中 `KryzeiuzPaths.from_root` 优先解析 FROZEN 划分、缺失时回退 DRAFT；`load_split_draft` 报错文案与 IdentityLoader provenance 同步记录实际读取路径与 `split_status`。（冻结脚本已验证：解析到 FROZEN、status=FROZEN、fold0 val 8 患者/8 样本。）
- 一次性 `_DRAFT` 物化/冻结辅助脚本保持历史原样，不再修改。
- **测试**：`tests/test_data_contracts_draft.py` 中 `test_kryzeiuz_split_frozen_contract` 覆盖冻结合同（status / seed / 每折 val 包含关系与 test 不相交 / `check_split_leakage` / DRAFT 副本一致性），该文件 8/8 通过；冻结后完整测试套件 **162 passed**（WSL `drugscreening-gpu`，CPU，237 秒），数据注册表验证 `python -m drug_screen.data.registry --root data` **PASS**。
- **registry**：`data/registry/datasets.json` 已登记 5 个冻结条目（`kryzeiuz_patient_outer5fold_frozen_v1`、`kryzeiuz_patient_identity_frozen_v1`、`kryzeiuz_drug_identity_48_frozen_v1`、`kryzeiuz_mcrc_target_admission_v1`、`kryzeiuz_expression_manifest_frozen_v1`）。
- `data/raw/` 全程未读未写；全部产物位于 processed / splits / registry 层。

## 八、明确不作为冻结条件（沿用 checklist）

- 32 个 prefix-only pending 样本（26 患者）不进入主队列，持续排除
- 不合并不同响应终点
- 不把前序 EXP 指标作为冻结依据

## 九、协调者验收（留空待确认）

> 待协调者确认冻结后，建模/评价不得再改 folds。

- 验收人：＿＿＿＿＿＿
- 验收时间：＿＿＿＿＿＿
- 验收意见：＿＿＿＿＿＿
