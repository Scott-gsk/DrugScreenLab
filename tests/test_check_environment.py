import importlib.util
from pathlib import Path


SCRIPT_PATH = Path("scripts/check_environment.py")
SPEC = importlib.util.spec_from_file_location("check_environment", SCRIPT_PATH)
assert SPEC and SPEC.loader
check_environment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check_environment)


def test_environment_errors_accepts_the_supported_contract(tmp_path):
    prefix = tmp_path / "drugscreening-gpu"
    executable = prefix / "bin" / "python"
    errors = check_environment.environment_errors(
        environ={
            "CONDA_DEFAULT_ENV": "drugscreening-gpu",
            "CONDA_PREFIX": str(prefix),
        },
        system="Linux",
        release="6.6.114.1-microsoft-standard-WSL2",
        executable=str(executable),
        proc_version="Linux version 6.6.114.1-microsoft-standard-WSL2",
    )
    assert errors == []


def test_environment_errors_rejects_non_conda_non_wsl_runtime():
    errors = check_environment.environment_errors(
        environ={},
        system="Windows",
        release="10.0.0",
        executable="C:/Python/python.exe",
        proc_version="",
    )
    assert any("expected Linux" in error for error in errors)
    assert any("expected WSL2" in error for error in errors)
    assert any("CONDA_DEFAULT_ENV" in error for error in errors)
    assert any("CONDA_PREFIX" in error for error in errors)


def test_environment_errors_rejects_wsl1_even_with_the_expected_conda_environment(tmp_path):
    prefix = tmp_path / "drugscreening-gpu"
    errors = check_environment.environment_errors(
        environ={
            "CONDA_DEFAULT_ENV": "drugscreening-gpu",
            "CONDA_PREFIX": str(prefix),
            "WSL_INTEROP": "/run/WSL/1_interop",
        },
        system="Linux",
        release="4.4.0-19041-Microsoft",
        executable=str(prefix / "bin" / "python"),
        proc_version="Linux version 4.4.0-19041-Microsoft",
    )
    assert any("expected WSL2" in error for error in errors)
