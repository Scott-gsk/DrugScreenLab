# DrugScreenLab

DrugScreenLab 是一个面向 AI 药物筛选的可复现研究仓库。项目通过明确的
数据身份、实验编号和独立审查流程，持续开发、评估和比较模型。

## 项目目标

在不混淆数据集、模型和实验身份的前提下，建立可审计、可复现的药物筛选
研究流程。当前仓库不预设模型、基线、冠军模型或进行中的实验；这些内容只能
在获得批准的 Experiment 中引入。

## 当前状态

项目状态见 `PROJECT_STATE.yaml`，实验编号与登记记录见
`experiments/registry.yaml`。数据资产通过数据集注册表（Dataset Registry）
独立管理，不以模型名称作为数据集身份。

## 项目结构

- `data/`：已登记的数据资产、版本化数据划分，以及不可修改的原始数据。
- `experiments/`：`EXP-ID` 注册表和实验记录。
- `artifacts/`：与对应 `EXP-ID` 绑定的实验输出，不应提交大文件。
- `src/`：可复用的 Python 包代码。
- `docs/`：环境、质量、复现和数据治理说明。

## 数据集

数据根目录默认是仓库内的 `data/`，可通过 `DRUGSCREEN_DATA_ROOT` 覆盖。
所有用于实验的数据集都必须在 `data/registry/` 中保留身份、版本、来源、
相对路径、校验和与溯源信息。详细约束见 `data/README.md`。

## 运行环境

唯一受支持的执行环境是 WSL2 Linux 中的 Conda `drugscreening-gpu`。
Windows Python、Windows Conda、系统 Python、`venv` 和临时环境均不能用于
项目的 Python、测试、数据处理或训练任务。完整说明见 `docs/ENVIRONMENT.md`。

## 多 Agent 工作流

Research Manager 定义研究问题和 `EXP-ID`；在获得人工批准后，Engineer 才能
实现；Reviewer 独立给出 `VALID`、`INVALID` 或 `INCONCLUSIVE`。详细协作规则
见 `AGENTS.md`。

## 实验流程

1. 定义研究问题、数据集版本与评价标准，并登记 `EXP-ID`。
2. 获得人工批准后实施数据处理、训练或评估。
3. 记录配置、随机种子、环境、源码版本与产物校验和。
4. 由独立 Reviewer 审查后再接受结果或发布。

## 常用命令

```bash
make env-check
make test
make data-check
make gpu-check
```

若当前 WSL shell 未初始化 Conda，可显式指定其二进制：

```bash
make CONDA=/home/dell/miniconda3/bin/conda test
```

## 项目文档

- `AGENTS.md`：协作、研究治理与操作边界。
- `ARCHITECTURE.md`：仓库架构和职责划分。
- `docs/ENVIRONMENT.md`：正式运行环境与验证方式。
- `docs/QUALITY.md`：质量审查要求。
- `docs/REPRODUCIBILITY.md`：实验复现记录要求。
