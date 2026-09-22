# DrugScreenLab 项目结构清理收口报告（2026-09-22）

> 性质：工程治理，不创建 EXP（AGENTS.md 规定）。执行依据：用户批准的《项目结构清理执行方案》（11 批次）。
> 恢复点：项目外归档 `D:\Code\DrugScreenLab_archive\cleanup_20260922\`（sha256 manifest）+ git 历史（已跟踪文件）。

## 一、结论

11 个批次全部完成并通过验收：**git 未提交项 122 → 0；根目录杂项 28 个文件 → 0；tmp/ 52 项 → 0；未引用 DRAFT 全部处置**。全量 pytest 214 passed、数据注册表 PASS、工作树 clean。大文件治理清单已产出（12 项、约 330 GB 候选），**未经用户逐项批复，data/ 与 artifacts/ 数据字节未动**。

## 二、清理前后对比

| 指标 | 清理前 | 清理后 |
|---|---|---|
| git 未提交变更 | 122 项（含 1 修改、4 已删未提交） | **0 项** |
| 根目录文件 | 38 个（16 tmp 脚本 + 2 空文件 + 5 make 脚本 + 6 PPTX + 9 治理文件） | **9 个治理/配置文件** |
| tmp/ | 52 项（探针 14、修复类 14、其他 24） | **0 项** |
| scripts/ 会话残留 | 60 项（`_` 前缀 22 + DRAFT 37 + .bak 1） | 8 项（被引用保留，见 §四） |
| docs/reports DRAFT | 54 项 | 1 项（被导航引用保留） |
| src/ DRAFT 模块 | 4 个 + foundation 空包 | 1 个（被冻结报告引用保留） |
| logs/ | 127 项 | 0 项（94.5 KB zip 归档） |
| .codex-build/ + .pytest_cache/ | 177 + 存在 | 不存在（清单归档） |
| __pycache__ 目录 | 23 个 | 0 个 |
| registry EXP-012~015 状态 | 过期中间态（approved_running 等） | 按记录与 PROJECT_STATE 回填终态 |

## 三、批次执行与 commit 索引

| 批次 | 内容 | 结果 |
|---|---|---|
| 0 安全网 | 项目外归档目录 + manifest 骨架 + WSL2 环境确认 | 就绪 |
| 1 根目录清零 | 16 临时脚本归档删除；make_* 5 个迁 `scripts/reporting/`；6 个 PPTX MD5 去重（工作汇报版==最终版，重复归档）后 5 个迁 `docs/reports/presentations/`；.gitattributes 跟踪 | **bfe509c** |
| 2 tmp 清空 | 52 项归档删除（修复类 14 项逐一核查：效果均已被正式脚本/冻结产物覆盖） | 完成 |
| 3 scripts 治理 | 52 项未引用 `_` 前缀/DRAFT 脚本归档删除；8 项被引用保留；23 个 __pycache__ 删除 | 完成 |
| 4 docs 治理 | 53 个已收口 DRAFT 报告/中间数据 git rm；2 个执行方案迁 `exec-plans/completed/`；README 导航同步 | **f960897** |
| 5 registry 治理 | 7 个无引用/已覆盖注册文件归档删除（.bak、exp015 平行、DSS_DRAFT、日期快照）；5 个数据合同跟踪提交；注册表验证 PASS | **4a87d5d** |
| 6 src 治理 | 3 个未引用 DRAFT 模块 + foundation 空包删除；`__init__.py` 同步清理；loaders_kryeziu_DRAFT 保留 | **2db9d16** |
| 7 工具目录 | logs 127 项压缩归档后清空；.codex-build 177 项清单记录后删除；.pytest_cache 删除 | 完成 |
| 8 git 一致化 | AGENTS.md 模型绑定措辞移除提交+残留对齐；治理测试同步；4 个已删文件提交；EXP-012~015 状态回填；保留项 42 文件跟踪；.trae/.workbuddy 豁免 | **34fc9cf** |
| 9 大文件清单 | 12 项盘点含批复栏（`large_file_disposal_list_20260922.md`），零数据改动 | **7c4dcae** |
| 10 防再犯 | QUALITY.md 收口三项硬性验收（tmp 清空/DRAFT 处置/工作树 clean） | **7c4dcae** |
| 11 终验 | pytest 214 passed（283s）+ 注册表 PASS + 工作树 clean + 本报告 | 本 commit |

## 四、引用保留清单（删除前逐一 grep 核查）

| 保留项 | 引用来源 |
|---|---|
| `scripts/modeling/train_stage2_dual_mlp_variants_DRAFT.py` 等 4 个 gain_wave1/joint 脚本 | experiments/records/EXP-024.md、EXP-025.md、gpu_relaunch_receipt、route1_gain_design_wave1 |
| `scripts/modeling/_launch_official_xfer_b2000_seed20260919_DRAFT.sh` 等 3 个启动脚本 | gpu_relaunch_receipt_20260919.md（nohup 启动命令） |
| `scripts/modeling/route1_gain_wave1_ci_DRAFT.py` | EXP-024 记录 + 设计文档 |
| `src/drug_screen/data/loaders_kryeziu_DRAFT.py` | split_freeze_report_20260919.md（FROZEN 划分解析组件） |
| `configs/experiments/EXP-020-*DRAFT.json`（5 个） | 保留的启动脚本 CFG 引用（EXP-021 官方迁移 provenance） |
| `docs/reports/data_contracts_B_20260921_DRAFT.md` | docs/README.md 导航（B 包数据合同入口） |
| `docs/reports/route1_gain_wave1_ci_vs_B0.csv` | EXP-024 记录 + 设计文档 |
| `data/registry/` 5 个数据合同（exp015_source、exp015_target_v3、transfer_cohorts、datasets_transfer、README） | src/drug_screen 生产代码与 EXP-028~035 脚本 |

## 五、归档与恢复

- 归档根：`D:\Code\DrugScreenLab_archive\cleanup_20260922\`
- manifest：`manifest_batch1.json`（根目录 16 项）、`manifest_batch1_pptx_dedup.json`（PPTX 哈希映射）、`manifest_batch2_tmp.json`（tmp 52 项分类）、`manifest_batch3_scripts.json`（52+8）、`manifest_batch5_data_registry.json`（7 项）、`manifest_batch6_src.json`（4 模块）、`manifest_batch7_codex-build_filelist.txt`（177 项清单）
- 已跟踪文件（53 DRAFT 报告、4 个根目录删除文件、src 模块等）：git 历史可恢复（`git log --follow` / 上述 commit 的父提交）
- 例外说明：批次 3 执行中 7 个根级 DRAFT 脚本副本一度写入 D:\ 盘根，已全部移入归档正确位置并复核（D:\ 根无残留）

## 六、大文件治理（待批复）

12 项候选共约 330 GB，详见 `docs/reports/large_file_disposal_list_20260922.md`（每项含体积、产生实验、注册表引用、建议处置与批复栏）。前三项：`data/interim/lincs` 143 GB（可从 raw 重建）、`data/artifacts/EXP-012` 104 GB（已收口）、`data/raw/lincs` 118 GB（只读，默认保留）。

## 七、遗留豁免项

| 项 | 理由 |
|---|---|
| `.trae/`、`.workbuddy/` | 本地会话工具目录，已入 .gitignore 豁免 |
| `.codex/`（6 个 agent toml，git 跟踪） | 历史治理文件；test_agent_governance 校验其完整性；移除需用户单独批准 |
| 8 个被引用 DRAFT 脚本/配置 | 见 §四；若需转正命名，属后续工程项 |
| `mvp/` | legacy（MCPIRE_PDO retired）但 git 已跟踪、体积小；PROJECT_STATE 登记其结论 |

## 八、验证记录

- 全量 pytest：`214 passed, 6 warnings in 283.13s`（WSL2 drugscreening-gpu，2026-09-22）
- 数据注册表：`PASS`（`PYTHONPATH=src conda run -n drugscreening-gpu python -m drug_screen.data.registry --root data`）
- `git status --porcelain`：空（仅豁免项不可见）
- 根目录/tmp/logs/exec-plans 终态清点：见 §二
- registry 状态回填依据：experiments/records/EXP-012~015.md 与 PROJECT_STATE.yaml extension_status，逐条注明回填注释
