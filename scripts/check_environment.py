"""对 DrugScreenLab 唯一受支持的运行环境进行快速失败验证。"""

from __future__ import annotations

import argparse
import os
import platform
import sys
from pathlib import Path
from typing import Mapping


EXPECTED_CONDA_ENV = "drugscreening-gpu"


def is_wsl2(release: str, proc_version: str, environ: Mapping[str, str]) -> bool:
    """判断给定 Linux 运行时信号是否表明当前处于 WSL2。"""
    marker = f"{release} {proc_version}".lower()
    return "wsl2" in marker


def environment_errors(
    *,
    environ: Mapping[str, str],
    system: str,
    release: str,
    executable: str,
    proc_version: str,
) -> list[str]:
    """列出与仓库受支持运行环境契约不一致的项目。"""
    errors: list[str] = []
    if system != "Linux":
        errors.append(f"expected Linux, found {system!r}")
    if not is_wsl2(release, proc_version, environ):
        errors.append("expected WSL2; Linux runtime does not identify as WSL2")

    conda_env = environ.get("CONDA_DEFAULT_ENV")
    if conda_env != EXPECTED_CONDA_ENV:
        errors.append(
            f"expected CONDA_DEFAULT_ENV={EXPECTED_CONDA_ENV!r}, found {conda_env!r}"
        )

    conda_prefix = environ.get("CONDA_PREFIX")
    if not conda_prefix:
        errors.append("CONDA_PREFIX is not set")
    else:
        prefix = Path(conda_prefix).resolve()
        python_path = Path(executable).resolve()
        if prefix.name != EXPECTED_CONDA_ENV:
            errors.append(f"expected Conda prefix named {EXPECTED_CONDA_ENV!r}, found {prefix}")
        if prefix not in python_path.parents:
            errors.append(f"Python executable is outside CONDA_PREFIX: {python_path}")
    return errors


def torch_report(require_cuda: bool) -> list[str]:
    """打印 PyTorch/CUDA 预检报告，并返回相应的验证错误。"""
    try:
        import torch
    except ImportError as exc:
        return [f"PyTorch import failed: {exc}"]

    available = torch.cuda.is_available()
    print(f"torch_version={torch.__version__}")
    print(f"cuda_available={available}")
    print(f"cuda_version={torch.version.cuda}")
    print(f"gpu_count={torch.cuda.device_count()}")
    for index in range(torch.cuda.device_count()):
        print(f"gpu_{index}={torch.cuda.get_device_name(index)}")
    return ["CUDA is required but unavailable"] if require_cuda and not available else []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-torch", action="store_true", help="require and report PyTorch")
    parser.add_argument("--require-cuda", action="store_true", help="require visible CUDA via PyTorch")
    args = parser.parse_args()

    try:
        proc_version = Path("/proc/version").read_text(encoding="utf-8")
    except OSError:
        proc_version = ""
    errors = environment_errors(
        environ=os.environ,
        system=platform.system(),
        release=platform.release(),
        executable=sys.executable,
        proc_version=proc_version,
    )
    if errors:
        print("FAIL: unsupported DrugScreenLab runtime")
        print("\n".join(f"- {error}" for error in errors))
        return 1

    print(f"platform={platform.system()} {platform.release()}")
    print(f"python_executable={Path(sys.executable).resolve()}")
    print(f"python_version={platform.python_version()}")
    print(f"conda_environment={os.environ['CONDA_DEFAULT_ENV']}")
    if args.require_torch or args.require_cuda:
        errors = torch_report(args.require_cuda)
        if errors:
            print("FAIL: GPU preflight")
            print("\n".join(f"- {error}" for error in errors))
            return 1
    print("PASS: supported DrugScreenLab runtime")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
