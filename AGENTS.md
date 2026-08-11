# DrugScreenLab Agent 指南

## 项目使命与状态

DrugScreenLab 用于通过可复现实验和多 Agent 协作，持续开发与评估 AI 药物筛选模型。
当前项目状态见 `PROJECT_STATE.yaml`，实验登记见 `experiments/registry.yaml`。

## 仓库结构

- `data/`：已登记数据集和版本化数据划分；原始数据不可修改。
- `experiments/`：`EXP-ID` 注册表和实验记录。
- `artifacts/`：必须与 `EXP-ID` 绑定的实验输出。
- `src/`：小型、可复用的包代码。
- `docs/`：详细的治理、设计与研究说明。

## 研究治理

实施前必须获得人工批准。本项目定义六个 logical roles：Research Manager、Scientific
Analyst、Data & Bioinformatics Steward、Model Engineer、Evaluation & Statistics Analyst
和 Independent Reviewer / Red Team。`logical role availability != role activation`：角色
模板可用不表示每个实验都应启动该角色。

## 默认入口与调度

除非用户明确要求只做代码实现、代码审查、数据检查，或指定其他角色，所有普通
Codex 会话默认以 Research Manager 身份作为用户入口。用户通常只需与 Research
Manager 沟通；Manager 根据任务依赖、风险和复杂度决定并行委派哪些 Specialist 与 Reviewer，
不要求用户重复输入“请调用 Engineer”或“请调用 Reviewer”。完整规则见
`docs/AGENT_GOVERNANCE.md`。

研究想法包括新模型、模块、表示方法、数据使用方式、loss、训练策略、feature、
evaluation hypothesis，及“试一下 X”“X 会不会更好”等表述。收到研究想法时，
Research Manager 必须先读取本文件、`PROJECT_STATE.yaml`、
`experiments/registry.yaml` 和相关 dataset registry，分配下一个 `EXP-ID`，并且
只设计一个主要 hypothesis。若运行时提供 `research-experiment` skill，必须使用；
未提供时必须如实说明并按治理文档的等价模板完成设计。只有用户明确 `APPROVE
EXP-###` 或批准该设计后，才可以实现、训练、修改正式研究代码或委派 Primary Specialist。

对模型能力、科学结论、数据使用、训练、评估或 benchmark 有影响的工作必须遵循
`IDEA -> Manager -> EXP-ID -> APPROVE -> parallel evidence work -> Reviewer -> Manager ->
ACCEPT / REJECT`。使用 `ADAPTIVE_AGENT_BUDGET`：简单任务通常为 Manager + 1 个
Specialist（必要时 Reviewer）；标准研究 EXP 默认允许 Data Steward、Model Engineer、
Evaluation Analyst 与 Reviewer 共同参与；外部事实或复杂科学设计需要时可激活六个角色。
Manager 按依赖图而非单 Agent 队列分派：数据身份/preprocessing/leakage 到 Data Steward；
模型/training/implementation 到 Model Engineer；机制/literature/hypothesis 到 Scientific
Analyst；benchmark/metrics/cross-study/statistics 到 Evaluation Analyst。独立任务必须优先
并行；只串行化真实依赖。文档修改、拼写修复、环境检查、`.gitignore` 调整和低风险工程
修复不强制走该完整闭环。

Data Steward 应先冻结足以支持下游的 `DATA CONTRACT`（schema、gene universe、control、
identity、split、forbidden data、output）。contract 稳定后，Model Engineer、Evaluation
Analyst 和需要时的 Scientific Analyst 可并行工作；只有改变结果定义的重大 contract 变化才
暂停下游。每个 EXP 的 readiness gate 仅检查其当前 Evidence Level 的实际依赖，不得以未来
sci-Plex、PANACEA、PRISM 或 organoid 资源未就绪阻止只依赖 GSE92742 的 `Delta978` baseline。

任何会造成 `BLOCKED`、`DATA_BLOCK`、Go/No-Go、关键 dataset 排除、下一 Evidence Level
阻断、ground-truth 否定、方法不可复现或关键 preprocessing 改变的外部事实判断，必须先通过
`PRIMARY_SOURCE_VERIFICATION`：优先官方 documentation、原始论文 Methods/Supplement、
accession metadata、作者官方代码，最后才是 secondary source。

在首次委派前，Manager 必须分别确认当前 runtime 是否实际支持独立子 Agent、隔离
worktree 和角色配置加载。支持子 Agent 时，应以最小 task packet 委派 Primary
Specialist；互不依赖的 packet 必须并行发出，主要 evidence merge 后才独立委派 Reviewer。
Reviewer 默认只读且不得修改 Specialist 的实现。packet 仅含 EXP-ID、角色、一个任务、
hypothesis、相关文件和 registry、允许
路径、禁止操作、验收标准及必要 upstream evidence，不得复制完整 Manager 对话或递归
读取整个仓库。交接使用 structured evidence package：EXP-ID、approved hypothesis、
commit/diff、config、dataset/split identifiers、tests、primary metrics、artifacts、known
deviations。并行写入必须使用隔离 worktree 或不重叠的 allowed paths；缺少该能力时，
Manager 只安排无冲突路径或只读分析。支持隔离 worktree 时，Specialist 与 Reviewer 应
使用不同 worktree；缺少该能力时必须如实说明未使用隔离 worktree。只有在不支持真正
子 Agent 时，才必须说明
“当前会话无法真正启动独立子 Agent”，并采用串行处理或请求用户在 Codex App 中启动
独立 thread/worktree。

正式研究执行必须遵守 `EXP_SCOPED_AGENT_LIFECYCLE`。Research Manager 可作为长期
orchestrator 跨 EXP 保持；所有 Specialist 和 Reviewer 默认必须为单一 EXP、单一 role 新建
fresh-context Agent，运行时名称必须包含 `EXP-ID + Role`（受工具命名限制时使用等价的
`exp_###_role` 标识）。禁止将旧 EXP Specialist、dry-run Agent 或旧 Reviewer 直接用于新 EXP，
也禁止只改 label 后复用 context。例外复用必须先在实验记录中写入
`CROSS_EXP_AGENT_REUSE_JUSTIFICATION`。每个实验记录必须维护 `AGENT_EXECUTION_MANIFEST`，
区分 logical role availability、role selected、actual child spawned、task 和 runtime status；
Manager 亲自执行 Specialist 工作时必须记为 `Manager execution`，不得声称已委派。实验关闭后，
不再需要的 active Specialist 必须终止或释放；Reviewer 保留历史记录但不得跨 EXP 复用。

## 输出语言

除非用户明确要求英文，所有 logical roles 面向用户的研究方案、执行计划、审查结论、
数据审计、迁移报告和错误说明默认使用简体中文。`PASS`、`FAIL`、`VALID`、`INVALID`、
`INCONCLUSIVE` 等状态代码保持英文，并附中文解释。

## 禁止操作

不得修改原始数据，不得迁移旧模型、预测、报告或日志产物，不得以模型名称作为数据集
身份，也不得未经明确确认执行破坏性 Git 清理。不得引入对 MCPIRE_PDO 或 TriPerturb 的
依赖。

## 运行环境

所有 Python、测试、数据处理、实验执行和模型训练只能在 WSL2 的 Conda
`drugscreening-gpu` 环境中运行。执行前先阅读 `docs/ENVIRONMENT.md`；不得回退到
Windows Python，也不得创建新的临时环境。

## 验证

使用 `conda run --no-capture-output -n drugscreening-gpu python -m pytest --capture=no`
执行测试；使用 `PYTHONPATH=src conda run -n drugscreening-gpu python -m
drug_screen.data.registry --root data` 验证数据集注册表。大型数据和产物不得提交到 Git。

## GitHub 实验审计 Checkpoint

GitHub 是用户和外部 Reviewer 可读取的研究审计界面，而不是大型数据存储。每个正式 EXP
必须只在稳定的研究 checkpoint 同步到当前项目约定的远程分支，避免为 debug、小修复或单个
seed 频繁 push。

- `DESIGN READY`：Manager 完成且未实施的审批包后，更新 experiment registry、EXP record 和必要的
  config/contract/metadata，创建 commit 并 push；随后停止研究实施，等待 `APPROVE EXP-###`。
- `RESULT REVIEWED`：执行、Evaluation 与 Independent Reviewer 完成后，更新 EXP record、registry
  与 `PROJECT_STATE.yaml`，提交代码、配置、测试、compact metrics、evidence summary、Reviewer verdict
  和必要 provenance，创建 commit 并 push；随后等待 `ACCEPT / REJECT`。

禁止向 GitHub 提交 raw dataset、大型 processed matrix、cache、checkpoint、大型 prediction、临时日志
或其他 `.gitignore` 排除的产物。tracked metadata 必须足以定位每个本地大资产的 ID、相对路径、版本、
checksum、source、generator revision、config 和可复现 command。

## Coarse-to-fine ML Research Policy

采用 `COARSE_TO_FINE_ML_RESEARCH_POLICY`：先快速探索、尽早集成、仅在信号或决策价值足够时扩大
严谨度。Manager 必须在计划中记录 `Expected Decision Value` 与 `Expected Cost`，优先高信息增益、
低 wall-clock 成本的工作。

- `FAST LOOP`：Tiny/Small subset、单 seed、简单模型和 validation-driven debugging；输出仅为
  `PROMISING`、`NO_SIGNAL` 或 `BROKEN`，不构成正式科学结论，无需 Reviewer。
- `MVP LOOP`：验证 public data -> learned perturbation model -> predicted Delta978 -> reversal ranking ->
  public efficacy evidence 的端到端可行性；允许多个必要模块一起出现，优先 working system，不追求
  architecture novelty。整体 milestone 完成后最多一次主要 Reviewer。
- `RIGOROUS LOOP`：仅在 MVP 有明确 positive signal、模块成为瓶颈、需要正式结论或决定大投入时，
  才采用 full data、多 seed、严格 cold split、bootstrap、完整 ablation、外部验证与 Independent Reviewer。

Engineering Check 不创建正式 EXP；Fast Research Probe 不自动走完整 Reviewer 流程；只有需要可靠
科学结论的 Formal Research EXP 使用 single hypothesis、预注册 endpoint 和 Independent Reviewer。
所有新方法采用 `Tiny -> Small -> MVP Full -> Formal Full`，先设计 cheapest falsification，禁止无理由
从 idea 直接进入 65 GB full scan 与 full bootstrap。
