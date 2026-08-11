# DrugScreenLab 运行环境

DrugScreenLab 只支持一种执行契约：**WSL2 Linux** 加 Conda 环境
**`drugscreening-gpu`**。该要求适用于 Python、测试、数据处理、实验、训练、
评估、基准测试和复现检查。Windows Python、Windows Conda、系统 Python、
virtualenv 以及临时创建的环境均不受支持。

仓库在 Windows 中的路径为 `D:\\Code\\DrugScreenLab`，在 WSL 中的路径为
`/mnt/d/Code/DrugScreenLab`。Linux 源码必须使用 Linux 路径。可用
`DRUGSCREEN_DATA_ROOT` 覆盖数据位置；未设置时使用仓库 `data/`，当前解析为
`/mnt/d/Code/DrugScreenLab/data`。Python 代码不得硬编码 Windows 路径。

## 执行命令

自动化任务必须显式调用 Conda，确保未激活的 shell 不会误选解释器：

```bash
conda run -n drugscreening-gpu python scripts/check_environment.py
conda run --no-capture-output -n drugscreening-gpu python -m pytest --capture=no
PYTHONPATH=src conda run -n drugscreening-gpu python -m drug_screen.data.registry --root data
```

交互式开发可先激活同一环境：

```bash
conda activate drugscreening-gpu
```

仓库提供统一入口，它们都会显式调用指定环境：

```bash
make env-check
make test
make data-check
make gpu-check
```

如果 `conda` 不在 `PATH` 中，请初始化 WSL shell 的 Conda，或显式指定二进制，
例如 `make CONDA=/home/dell/miniconda3/bin/conda test`。不得改用 Windows
可执行文件，也不得另建环境。

## 环境检查

`scripts/check_environment.py` 会快速失败，除非它确认 Linux、WSL、
`CONDA_DEFAULT_ENV=drugscreening-gpu`、匹配的 Conda prefix，以及位于该 prefix
内的 Python 可执行文件。`--require-torch` 会验证并报告 PyTorch；
`--require-cuda` 还会要求 CUDA 可见，应仅通过 `make gpu-check` 用于训练前预检。
CPU-only 单元测试和注册表测试不强制要求 CUDA。

当前受支持的 WSL 终端需要 Conda 直连输出并配合 pytest 的 `--capture=no`，以避开
Conda 临时输出文件与 pytest 捕获机制的冲突。

## 依赖管理

`environment.yml` 是受版本控制的环境规范。它以注释区分运行时科学计算栈、获批
训练所需的 PyTorch/CUDA 栈和开发/测试工具；它不是机器快照，绝不能包含绝对路径、
凭据、token 或本地 secret。

获批工作如需新依赖，应先更新 `environment.yml` 并在实验计划中记录原因，再修改
`drugscreening-gpu`。不得静默执行 `pip install`。Python、PyTorch、CUDA 相关包、
NumPy、pandas、RDKit 或其他主要科学库的变更必须经 Research Manager 批准，因为
该环境由研究工作共享。

## 故障排查

先执行 `make env-check`。失败通常表示命令不在 WSL 中、Conda shell 未初始化、
环境名称错误，或选中了错误的 Python 可执行文件。应修正 shell 或环境后重新执行，
不得绕过检查。训练前执行 `make gpu-check`；若 CUDA 不可见，应在 WSL/NVIDIA 层
解决，而非降低预检要求。
