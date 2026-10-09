from __future__ import annotations

import ast
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest
import numpy as np
import pandas as pd


RUNNER = Path(__file__).resolve().parents[2] / "scripts/modeling/exp051_contract_fit_smoke.py"
SPEC = importlib.util.spec_from_file_location("exp051_fit_runner_contract", RUNNER)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class FakeTensor:
    def to(self, device: object) -> "FakeTensor":
        return self


class FakeTranSiGen:
    def forward(self, x1: FakeTensor, features: FakeTensor) -> tuple[object, ...]:
        return x1, None, None, x1, None, None, None

    def encode_x2(self, x2: FakeTensor) -> tuple[object, ...]:
        return x2, None, None

    def decode_x2(self, z2: FakeTensor) -> FakeTensor:
        return z2

    def loss(self, *args: object) -> tuple[int, ...]:
        return tuple(range(7))


def _write_ledger(path: Path, terminal_event: str, charge_field: str, charge: float) -> None:
    path.parent.mkdir(parents=True)
    entries = [
        {"event": "GPU_SMOKE_STARTED", "at_utc": "2026-01-01T00:00:00Z", "reserved_gpu_hours": 1.0},
        {"event": terminal_event, "at_utc": "2026-01-01T00:01:00Z", charge_field: charge},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in entries), encoding="utf-8")


def test_transigen_batch_has_one_four_item_contract() -> None:
    result = runner._transigen_forward_batch(
        None, FakeTranSiGen(), (FakeTensor(), FakeTensor(), FakeTensor()), "cpu")
    assert len(result) == 4
    assert result[3] == tuple(range(7))

    import ast

    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    call_assignments = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "_transigen_forward_batch"
    ]
    assert call_assignments
    assert all(isinstance(node.targets[0], ast.Tuple) and len(node.targets[0].elts) == 4
               for node in call_assignments)


def test_norm_diagnostics_keep_finite_large_float32_gradients_finite() -> None:
    import torch

    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.left = torch.nn.Parameter(torch.zeros(2, dtype=torch.float32))
            self.right = torch.nn.Parameter(torch.zeros(2, dtype=torch.float32))

    model = Tiny()
    model.left.grad = torch.tensor([1e20, 1e20], dtype=torch.float32)
    model.right.grad = torch.tensor([1e20, 1e20], dtype=torch.float32)
    elements_finite, norm, bad_name, norm_finite = runner._lean_monitor_gradient_summary(torch, model)

    assert elements_finite is True
    assert norm_finite is True
    assert bad_name is None
    assert norm == pytest.approx(2e20, rel=5e-8)
    assert runner._grad_global_norm(torch, model) == pytest.approx(2e20, rel=5e-8)


def test_norm_diagnostics_separate_zero_nan_and_inf_gradients() -> None:
    import torch

    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(2, dtype=torch.float32))

    empty = Tiny()
    elements_finite, norm, bad_name, norm_finite = runner._lean_monitor_gradient_summary(torch, empty)
    assert (elements_finite, norm, bad_name, norm_finite) == (True, 0.0, None, True)

    for invalid in (float("nan"), float("inf")):
        model = Tiny()
        model.weight.grad = torch.tensor([1.0, invalid], dtype=torch.float32)
        elements_finite, norm, bad_name, norm_finite = runner._lean_monitor_gradient_summary(torch, model)
        assert elements_finite is False
        assert norm_finite is False
        assert bad_name == "weight"
        assert not np.isfinite(norm)


def test_parameter_norm_and_difference_cast_before_squaring_and_subtraction() -> None:
    import torch

    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(-3e38)
    before = [torch.full_like(model.weight, 3e38)]

    assert runner._param_global_norm(torch, model) == pytest.approx(3e38, rel=1e-6)
    assert runner._norm_difference(torch, before, model) == pytest.approx(6e38, rel=1e-6)


def test_resource_balance_comes_from_closed_append_only_ledgers(tmp_path: Path) -> None:
    exp_root = tmp_path / "EXP-051"
    _write_ledger(exp_root / "old-run/fit_attempt_001/resource_ledger.jsonl",
                  "GPU_SMOKE_FAILED_CLOSED", "charged_gpu_hours_conservative_upper_bound", 0.25)
    _write_ledger(exp_root / "comparison-run/fit_attempt_001/resource_ledger.jsonl",
                  "GPU_ATTEMPT_COMPLETED", "actual_gpu_hours", 0.5)
    pre_gpu = exp_root / "comparison-run/fit_attempt_002/pre_gpu_failure_receipt.json"
    pre_gpu.parent.mkdir(parents=True)
    pre_gpu.write_text(json.dumps({"run_id": "comparison-run", "attempt": "fit_attempt_002",
                                   "status": "CPU_PREFLIGHT_FAILED_CLOSED_NO_GPU",
                                   "gpu_smoke_started": False,
                                   "gpu_model_initialization_started": False,
                                   "charged_gpu_hours": 0.0, "reserved_gpu_hours": 0.0}),
                       encoding="utf-8")

    account = runner.resource_accounting(
        exp_root, "comparison-run", package_cap_gpu_hours=10.0,
        total_exp_cap_gpu_hours=48.0,
        exp_settled_before_artifact_ledgers_gpu_hours=0.584830054434155)
    assert account["package_settled_gpu_hours"] == pytest.approx(0.5)
    assert account["exp_artifact_ledger_settled_gpu_hours"] == pytest.approx(0.75)
    assert account["exp_settled_gpu_hours"] == pytest.approx(1.334830054434155)
    assert account["package_remaining_gpu_hours"] == pytest.approx(9.5)
    assert account["remaining_authorized_gpu_hours"] == pytest.approx(9.5)
    assert len(account["closed_attempts"]) == 3
    assert account["closed_attempts"][1]["close_event"] == "CPU_PREFLIGHT_FAILED_CLOSED_NO_GPU"
    assert account["closed_attempts"][1]["charged_gpu_hours"] == 0.0


def test_resource_accounting_rejects_nonzero_pre_gpu_receipt(tmp_path: Path) -> None:
    exp_root = tmp_path / "EXP-051"
    receipt = exp_root / "run/fit_attempt_001/pre_gpu_failure_receipt.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text(json.dumps({"run_id": "run", "attempt": "fit_attempt_001",
                                   "status": "CPU_PREFLIGHT_FAILED_CLOSED_NO_GPU",
                                   "gpu_smoke_started": False, "charged_gpu_hours": 0.001}),
                       encoding="utf-8")
    with pytest.raises(RuntimeError, match="invalid or nonzero pre-GPU failure closure"):
        runner.resource_accounting(exp_root, "run", 10.0, 48.0, 0.584830054434155)


def test_resource_limits_use_new_package_and_smoke_authorizations() -> None:
    limits = runner.resource_limits({
        "authorization": {"package_gpu_cap": 10.0, "exp_gpu_cap": 48.0,
                          "exp_settled_gpu_hours_before_run": 0.6},
        "resource": {"paired_smoke_gpu_cap_hours": 0.25,
                     "first_package_cap_gpu_hours": 5.0},
    })
    assert limits["package_cap_gpu_hours"] == 10.0
    assert limits["paired_smoke_cap_gpu_hours"] == 0.25
    assert limits["total_exp_cap_gpu_hours"] == 48.0


def test_baselines_read_frozen_values_from_source_run_not_output_run(tmp_path: Path,
                                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    source_run = tmp_path / "source-run"
    source_run.mkdir()
    monkeypatch.setattr(runner, "SOURCE_RUN_DIR", source_run)
    monkeypatch.setattr(runner, "RUN_DIR", tmp_path / "new-output-run")
    genes = np.asarray([f"G{i:03d}" for i in range(978)], dtype=str)
    np.savez_compressed(source_run / "frozen_train_inner_values.npz", genes=genes)
    keys = pd.DataFrame({
        "source_object_id": ["cell-a", "cell-b", "cell-c"],
        "full_inchikey": ["AAAAAAAAAAAAAA-BBBBBBBBBB-C", "CCCCCCCCCCCCCC-DDDDDDDDDD-E",
                          "AAAAAAAAAAAAAA-BBBBBBBBBB-C"],
        "full_condition_key": ["k1", "k2", "k3"],
    })
    delta = np.zeros((3, 978), dtype=np.float32)
    delta[1, 0] = 1.0
    delta[2, 0] = 2.0
    train = np.asarray([True, True, False])
    dev = ~train
    gene_variance = np.var(delta[train].astype(np.float64), axis=0)
    output = tmp_path / "fit_attempt_001"
    output.mkdir()
    result = runner.data_baselines(keys, delta, gene_variance, train, dev, output)
    assert result["metrics"]["zero_change"]["inner_dev"]["n_conditions"] == 1
    with np.load(output / "baseline_predictions.npz", allow_pickle=False) as saved:
        assert np.array_equal(saved["genes"], genes)
        assert saved["inner_dev_condition_key"].tolist() == ["k3"]


def test_attempt_resolver_never_reopens_existing_attempt(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    existing = run_dir / "fit_attempt_003"
    existing.mkdir(parents=True)
    sentinel = existing / "failure_close_receipt.json"
    sentinel.write_text("immutable", encoding="utf-8")

    with pytest.raises(ValueError, match="next unused attempt is fit_attempt_004"):
        runner.new_attempt_path(run_dir, "fit_attempt_003")
    assert sentinel.read_text(encoding="utf-8") == "immutable"


def test_run_id_is_parameterized_and_fit_cli_requires_attempt(monkeypatch: pytest.MonkeyPatch,
                                                              tmp_path: Path) -> None:
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    runner.configure_run("smoke-run-b")
    runner.configure_source_run("frozen-run-a")
    assert runner.RUN_ID == "smoke-run-b"
    assert runner.RUN_DIR == tmp_path / "artifacts/experiments/EXP-051/smoke-run-b"
    assert runner.SOURCE_RUN_ID == "frozen-run-a"
    assert runner.SOURCE_RUN_DIR == tmp_path / "artifacts/experiments/EXP-051/frozen-run-a"
    with pytest.raises(ValueError, match="invalid EXP-051 run_id"):
        runner.configure_run("../outside")

    monkeypatch.setattr(sys, "argv", [str(RUNNER), "--run-id", "smoke-run-b",
                                      "--source-run-id", "frozen-run-a", "--phase", "fit"])
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
    monkeypatch.setattr(sys, "argv", [str(RUNNER), "--phase", "fit"])
    with pytest.raises(SystemExit) as missing_run_id:
        runner.main()
    assert missing_run_id.value.code == 2


def test_close_handler_ignores_preexisting_runner_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    run_id = "test-run"
    runner.configure_run(run_id)
    attempt = runner.RUN_DIR / "fit_attempt_003"
    attempt.mkdir(parents=True)
    manifest = {"run_id": run_id, "fit_attempt": "fit_attempt_003",
                "wsl_pid": os.getpid(), "current_runner_sha256": "previous-runner-sha"}
    (attempt / "fit_smoke_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    ledger = attempt / "resource_ledger.jsonl"
    ledger_bytes = (json.dumps({"event": "GPU_SMOKE_STARTED", "at_utc": "2026-01-01T00:00:00Z"}) + "\n").encode()
    ledger.write_bytes(ledger_bytes)

    runner._close_unhandled_attempt(run_id, "fit_attempt_003", RuntimeError("test"))

    assert ledger.read_bytes() == ledger_bytes
    assert not (attempt / "failure_close_receipt.json").exists()


def test_close_handler_closes_only_current_cpu_preflight_without_gpu(tmp_path: Path,
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    run_id = "test-pre-gpu-run"
    runner.configure_run(run_id)
    runner.configure_source_run("source-run")
    attempt = runner.RUN_DIR / "fit_attempt_001"
    attempt.mkdir(parents=True)
    (attempt / "pre_gpu_preflight_started.json").write_text(json.dumps({
        "run_id": run_id, "attempt": "fit_attempt_001", "status": "CPU_PREFLIGHT_STARTED",
        "started_at_utc": "2026-01-01T00:00:00Z", "wsl_pid": os.getpid(),
        "runner_sha256": runner.sha256_file(RUNNER), "config_sha256": "cfg-sha",
        "source_run_id": "source-run"}), encoding="utf-8")
    runner._close_unhandled_attempt(run_id, "fit_attempt_001", FileNotFoundError("source values"))
    receipt = json.loads((attempt / "pre_gpu_failure_receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "CPU_PREFLIGHT_FAILED_CLOSED_NO_GPU"
    assert receipt["charged_gpu_hours"] == 0.0
    assert receipt["gpu_smoke_started"] is False
    assert not (attempt / "resource_ledger.jsonl").exists()


def test_exp055_fit_manifest_binds_cpu_verified_xpert_schedule_receipt() -> None:
    import ast

    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    fit_phase = next(node for node in tree.body
                     if isinstance(node, ast.FunctionDef) and node.name == "fit_phase")
    schedule_binding = next(
        node for node in ast.walk(fit_phase)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "xpert_schedule_receipt"
                for target in node.targets)
    )
    assert isinstance(schedule_binding.value, ast.IfExp)
    assert isinstance(schedule_binding.value.body, ast.Call)
    assert isinstance(schedule_binding.value.body.func, ast.Name)
    assert schedule_binding.value.body.func.id == "_verify_exp055_xpert_cpu_schedule"

    manifest_binding = next(
        node for node in ast.walk(fit_phase)
        if isinstance(node, ast.Dict)
        and any(isinstance(key, ast.Constant)
                and key.value == "XPert_official_schedule_cpu_receipt" for key in node.keys)
    )
    receipt_index = next(index for index, key in enumerate(manifest_binding.keys)
                         if isinstance(key, ast.Constant)
                         and key.value == "XPert_official_schedule_cpu_receipt")
    assert isinstance(manifest_binding.values[receipt_index], ast.Name)
    assert manifest_binding.values[receipt_index].id == "xpert_schedule_receipt"


def test_exp055_allows_only_bounded_zero_gpu_preflight_retries(tmp_path: Path,
                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "exp055-run"
    monkeypatch.setattr(runner, "RUN_ID", "exp055-run")
    source_run_id = "frozen-source-run"
    for index in (1, 2):
        attempt_id = f"fit_attempt_{index:03d}"
        attempt = run_dir / attempt_id
        attempt.mkdir(parents=True)
        common = {"run_id": "exp055-run", "attempt": attempt_id,
                  "source_run_id": source_run_id}
        (attempt / "pre_gpu_preflight_started.json").write_text(json.dumps({
            **common, "status": "CPU_PREFLIGHT_STARTED"}), encoding="utf-8")
        (attempt / "pre_gpu_failure_receipt.json").write_text(json.dumps({
            **common, "status": "CPU_PREFLIGHT_FAILED_CLOSED_NO_GPU",
            "gpu_smoke_started": False, "model_initialization_started": False,
            "charged_gpu_hours": 0.0, "reserved_gpu_hours": 0.0}), encoding="utf-8")

    runner._validate_exp055_smoke_attempt(run_dir, "fit_attempt_003", source_run_id)
    (run_dir / "fit_attempt_002/resource_ledger.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(RuntimeError, match="has a GPU ledger"):
        runner._validate_exp055_smoke_attempt(run_dir, "fit_attempt_003", source_run_id)

    (run_dir / "fit_attempt_002/resource_ledger.jsonl").unlink()
    (run_dir / "fit_attempt_003").mkdir()
    with pytest.raises(ValueError, match="at most two smoke retries"):
        runner._validate_exp055_smoke_attempt(run_dir, "fit_attempt_004", source_run_id)


def test_exp055_xpert_only_route_uses_official_schedule_and_cost_gate() -> None:
    import ast

    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "_train_xpert_only")
    source = ast.get_source_segment(RUNNER.read_text(encoding="utf-8"), function)
    assert source is not None
    assert 'int(config_train["num_epochs"]) if exp055 else 100' in source
    assert 'int(config_train["patience"]) if exp055 else None' in source
    assert "_xpert_official_selection_loss" in source
    assert "stopped_by_patience = True" in source
    assert "XPert_final_stop_epoch_" in source
    assert "_verify_exp055_prediction_and_checkpoint_readback" in source

    fit_only = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "fit_phase_xpert_only")
    fit_source = ast.get_source_segment(RUNNER.read_text(encoding="utf-8"), fit_only)
    assert fit_source is not None
    assert 'planned_epochs = int(xpert_config["train"]["num_epochs"]) if EXPERIMENT_ID == "EXP-055" else 100' in fit_source
    assert "projected_package_total" in fit_source
    assert "projected_global_total" in fit_source
    assert "one_time_package_reallocation_receipt.json" in fit_source
    assert "Official patience=50 is not used to reduce the pre-run projection." in fit_source

    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    main_source = ast.get_source_segment(RUNNER.read_text(encoding="utf-8"), main)
    assert main_source is not None
    assert 'args.experiment_id == "EXP-055" and args.models == "xpert"' in main_source
    assert 'fit_phase_xpert_only(cfg, config_sha, args.attempt_id)' in main_source


def test_trainpath_trace_lifecycle_is_bound_append_only_and_closed(tmp_path: Path,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    runner.configure_experiment("EXP-055")
    run_id = "20261007_exp055_transigen_trainpath_recovery_v6"
    runner.configure_run(run_id)
    attempt_id = "fit_attempt_001"
    attempt = runner.RUN_DIR / attempt_id
    attempt.mkdir(parents=True)
    assert list(attempt.iterdir()) == []

    runner._open_trainpath_trace_owner(attempt, run_id, attempt_id, os.getpid(), "runner-sha")
    initial_event = {"scope": "initial", "batch_index": None, "module": "encoder",
                     "io_role": "output", "tensor_name": "z", "shape": [1],
                     "numel": 1, "finite_n": 1, "_max_abs_tensor": torch.tensor(1.0)}
    runner._trainpath_flush_layer_events(torch, attempt, [initial_event])
    epoch_event = {"scope": "epoch1", "batch_index": 1, "module": "decoder",
                   "io_role": "output", "tensor_name": "delta", "shape": [1],
                   "numel": 1, "finite_n": 1, "_max_abs_tensor": torch.tensor(2.0)}
    runner._trainpath_flush_layer_events(torch, attempt, [epoch_event])
    final_path = attempt / "final_predictions.npz"
    runner.atomic_npz(final_path, condition_key=np.asarray(["k1"], dtype=str),
                      prediction=np.asarray([[1.0, 2.0]], dtype=np.float32))
    with np.load(final_path, allow_pickle=False) as saved:
        assert saved["condition_key"].tolist() == ["k1"]
        assert np.array_equal(saved["prediction"], np.asarray([[1.0, 2.0]], dtype=np.float32))
    runner._close_trainpath_trace_owner(attempt, run_id, attempt_id, "COMPLETED_TEST_LIFECYCLE")

    trace = [json.loads(line) for line in (attempt / "TranSiGen_trainpath_layer_trace.jsonl").read_text().splitlines()]
    assert [row["scope"] for row in trace] == ["initial", "epoch1"]
    assert all(row["run_id"] == run_id and row["attempt_id"] == attempt_id for row in trace)
    with pytest.raises(RuntimeError, match="attempt is closed"):
        runner._append_trainpath_trace_row(attempt, run_id, attempt_id, {"scope": "late"})
    with pytest.raises(FileExistsError, match="must be empty"):
        runner._open_trainpath_trace_owner(attempt, run_id, attempt_id, os.getpid(), "runner-sha")
    with pytest.raises(ValueError, match="next unused attempt is fit_attempt_002"):
        runner.new_attempt_path(runner.RUN_DIR, attempt_id)

    stale_attempt = runner.RUN_DIR / "fit_attempt_002"
    stale_attempt.mkdir()
    (stale_attempt / "TranSiGen_trainpath_layer_trace.jsonl").write_text("old\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="without a run/attempt owner"):
        runner._append_trainpath_trace_row(stale_attempt, run_id, stale_attempt.name, {"scope": "late"})


def test_transigen_eval_auxiliary_loss_nulls_do_not_hide_main_delta_failure(
        monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    cuda_seed_calls: list[int] = []
    monkeypatch.setattr(torch.cuda, "manual_seed_all", cuda_seed_calls.append)

    class EvalModel:
        nonfinite_delta = False

        def eval(self) -> "EvalModel":
            return self

        def forward(self, x1: torch.Tensor, _features: torch.Tensor) -> tuple[object, ...]:
            delta = torch.full_like(x1, float("nan")) if self.nonfinite_delta else torch.ones_like(x1)
            return x1, None, None, x1 + delta, None, None, None

        def encode_x2(self, x2: torch.Tensor) -> tuple[torch.Tensor, None, None]:
            return x2, None, None

        def decode_x2(self, z2: torch.Tensor) -> torch.Tensor:
            return z2

        def loss(self, x1: torch.Tensor, *_args: object) -> tuple[torch.Tensor, ...]:
            auxiliary = torch.tensor(float("nan")) if len(x1) == 1 else torch.tensor(2.0)
            return (torch.tensor(1.0), auxiliary, torch.tensor(3.0), torch.tensor(4.0),
                    torch.tensor(5.0), torch.tensor(6.0), torch.tensor(7.0))

    model = EvalModel()
    x1 = np.zeros((3, 978), dtype=np.float32)
    x = np.ones((3, 978), dtype=np.float32)
    features = np.zeros((3, 4), dtype=np.float32)
    prediction, report = runner._transigen_predict_and_losses(
        torch, model, x1, x, features, 2, torch.device("cpu"), 11,
        seed_cuda=False,
        allow_nonfinite_auxiliary=True)
    assert cuda_seed_calls == []
    assert np.all(prediction == 1.0)
    assert report["mse_x1"] == {"value": None, "reason": "nonfinite_auxiliary_evaluation_loss",
                               "effective_n": 2, "nonfinite_n": 1}
    with pytest.raises(FloatingPointError, match="non-finite TranSiGen evaluation loss"):
        runner._transigen_predict_and_losses(torch, model, x1, x, features, 2,
            torch.device("cpu"), 11, allow_nonfinite_auxiliary=False)
    assert cuda_seed_calls == [11, 11]
    model.nonfinite_delta = True
    diagnostics: list[dict[str, object]] = []
    with pytest.raises(FloatingPointError, match="main-delta prediction"):
        runner._transigen_predict_and_losses(torch, model, x1, x, features, 2,
            torch.device("cpu"), 11, seed_cuda=False, allow_nonfinite_auxiliary=True,
            nonfinite_main_delta_callback=diagnostics.append)
    assert len(diagnostics) == 1
    assert diagnostics[0]["evaluation_split"] == "TRAIN_ONLY"
    assert diagnostics[0]["batch_start"] == 0
    assert diagnostics[0]["batch_stop"] == 2
    assert diagnostics[0]["shape"] == [2, 978]
    assert diagnostics[0]["finite_n"] == 0
    assert diagnostics[0]["nonfinite_n"] == 2 * 978
    assert diagnostics[0]["nonfinite_row_gene_positions_first_32"]


def test_resource_accounting_recognizes_closed_cpu_entry_smoke_without_gpu_charge(
        tmp_path: Path) -> None:
    run_id = runner.EXP055_TRAINPATH_ENTRY_SMOKE_RUN_ID
    run_dir = tmp_path / "EXP-055" / run_id
    attempt = run_dir / "fit_attempt_001"
    attempt.mkdir(parents=True)
    result_path = attempt / "entry_smoke_result.json"
    result_path.write_text(json.dumps({"status": "PASS_REAL_ORIGINAL_TRANSIGEN_CPU_ENTRY_SMOKE",
        "gpu_hours_charged": 0.0, "inner_dev_numeric_rows_loaded": False,
        "pdo_outer_locked_numeric_values_opened": False}), encoding="utf-8")
    owner_path = attempt / "TranSiGen_trainpath_layer_trace_manifest.json"
    owner_path.write_text(json.dumps({"status": "CLOSED",
        "close_status": "PASS_REAL_ORIGINAL_TRANSIGEN_CPU_ENTRY_SMOKE"}), encoding="utf-8")
    runner.atomic_json(attempt / "entry_smoke_close_receipt.json", {
        "status": "PASS_REAL_ORIGINAL_TRANSIGEN_CPU_ENTRY_SMOKE",
        "run_id": run_id, "attempt_id": "fit_attempt_001", "device": "cpu",
        "gpu_hours_charged": 0.0,
        "result_sha256": runner.sha256_file(result_path),
        "owner_manifest_sha256": runner.sha256_file(owner_path),
        "closed_at_utc": "2026-10-07T00:00:00Z"})

    accounting = runner.resource_accounting(tmp_path / "EXP-055", "new-recovery-run",
        package_cap_gpu_hours=1.0, total_exp_cap_gpu_hours=48.0,
        exp_settled_before_artifact_ledgers_gpu_hours=1.0,
        package_settled_before_artifact_ledgers_gpu_hours=0.003001485)
    assert accounting["package_settled_gpu_hours"] == pytest.approx(0.003001485)
    assert accounting["exp_settled_gpu_hours"] == pytest.approx(1.0)
    assert accounting["closed_attempts"] == [{"run_id": run_id,
        "attempt": "fit_attempt_001", "close_event": "CPU_ENTRY_SMOKE_COMPLETED_NO_GPU",
        "charged_gpu_hours": 0.0, "reserved_gpu_hours": 0.0,
        "close_at_utc": "2026-10-07T00:00:00Z"}]


def test_batch_replay_tensor_summary_preserves_nonfinite_counts_without_nan_json() -> None:
    import torch

    summary = runner._batch_replay_tensor_stats(
        torch, torch.tensor([[1.0, float("nan")], [float("inf"), -3.0]]))
    assert summary["shape"] == [2, 2]
    assert summary["numel"] == 4
    assert summary["finite_n"] == 2
    assert summary["nonfinite_n"] == 2
    assert summary["finite_max_abs"] == 3.0
    assert summary["nonfinite_positions_first_16"] == [[0, 1], [1, 0]]
    assert "NaN" not in json.dumps(summary)
    assert "Infinity" not in json.dumps(summary)


def test_latent_replay_vector_encodes_nonfinite_values_as_text() -> None:
    import torch

    values = runner._latent_replay_json_vector(
        torch, torch.tensor([1.25, float("inf"), float("-inf"), float("nan")]))
    assert values == [1.25, "Infinity", "-Infinity", "NaN"]
    assert json.loads(json.dumps(values)) == values


def test_monitored_main_delta_keeps_full_denominator_and_suppresses_partial_metrics() -> None:
    prediction = np.ones((2, 3), dtype=np.float32)
    prediction[1, 2] = np.inf
    summary = runner._monitored_main_prediction_summary(
        prediction, expected_shape=(2, 3),
        first_layer={"module": "decoder_x2", "io_role": "input"},
        batch_failures=[{"batch_start": 0, "nonfinite_n": 1, "numel": 6}])
    assert summary["status"] == "UNDEFINED_NONFINITE_MAIN_DELTA"
    assert summary["object_macro_nmse"] is None
    assert summary["object_macro_amplitude_ratio"] is None
    assert summary["main_delta_nonfinite_n"] == 1
    assert summary["main_delta_denominator"] == 6
    assert summary["first_nonfinite_layer"]["module"] == "decoder_x2"
    assert summary["finite_subset_metrics_computed"] is False


def test_prediction_roundtrip_comparator_preserves_nonfinite_values() -> None:
    left = np.asarray([[1.0, np.nan, np.inf, -np.inf]], dtype=np.float32)
    right = left.copy()
    assert runner._np_arrays_equal_with_nan(left, right)
    right[0, 2] = 2.0
    assert not runner._np_arrays_equal_with_nan(left, right)


def test_resource_accounting_accepts_closed_cpu_batch_replay_only_with_hashes(
        tmp_path: Path) -> None:
    run_id = runner.EXP055_TRAINPATH_BATCH_REPLAY_RUN_ID
    attempt = tmp_path / "EXP-055" / run_id / "fit_attempt_001"
    attempt.mkdir(parents=True)
    trace_path = attempt / "TranSiGen_trainpath_layer_trace.jsonl"
    trace_path.write_text('{"scope":"eval"}\n', encoding="utf-8")
    owner_path = attempt / "TranSiGen_trainpath_layer_trace_manifest.json"
    owner_path.write_text(json.dumps({"run_id": run_id, "attempt_id": "fit_attempt_001",
        "status": "CLOSED"}), encoding="utf-8")
    result_path = attempt / "batch_replay_result.json"
    result_path.write_text(json.dumps({"status": "COMPLETED_READ_ONLY_CPU_BATCH_REPLAY",
        "gpu_hours_charged": 0.0}), encoding="utf-8")
    runner.atomic_json(attempt / "batch_replay_close_receipt.json", {
        "status": "COMPLETED_READ_ONLY_CPU_BATCH_REPLAY", "run_id": run_id,
        "attempt_id": "fit_attempt_001", "device": "cpu", "gpu_hours_charged": 0.0,
        "result_sha256": runner.sha256_file(result_path),
        "layer_trace_sha256": runner.sha256_file(trace_path),
        "owner_manifest_sha256": runner.sha256_file(owner_path),
        "closed_at_utc": "2026-10-07T00:00:00Z"})

    accounting = runner.resource_accounting(tmp_path / "EXP-055", "later-run",
        package_cap_gpu_hours=1.0, total_exp_cap_gpu_hours=48.0,
        exp_settled_before_artifact_ledgers_gpu_hours=1.0,
        package_settled_before_artifact_ledgers_gpu_hours=0.003001485)
    assert accounting["package_settled_gpu_hours"] == pytest.approx(0.003001485)
    assert accounting["exp_settled_gpu_hours"] == pytest.approx(1.0)
    assert accounting["closed_attempts"] == [{"run_id": run_id,
        "attempt": "fit_attempt_001", "close_event": "CPU_BATCH_REPLAY_COMPLETED_NO_GPU",
        "charged_gpu_hours": 0.0, "reserved_gpu_hours": 0.0,
        "close_at_utc": "2026-10-07T00:00:00Z"}]


def test_resource_accounting_accepts_closed_cpu_latent_replay_only_with_hashes(
        tmp_path: Path) -> None:
    run_id = runner.EXP055_TRAINPATH_LATENT_REPLAY_RUN_ID
    attempt = tmp_path / "EXP-055" / run_id / "fit_attempt_001"
    attempt.mkdir(parents=True)
    trace_path = attempt / "latent_replay_trace.jsonl"
    trace_path.write_text('{"event":"latent"}\n', encoding="utf-8")
    owner_path = attempt / "TranSiGen_trainpath_layer_trace_manifest.json"
    owner_path.write_text(json.dumps({"run_id": run_id, "attempt_id": "fit_attempt_001",
        "status": "CLOSED"}), encoding="utf-8")
    result_path = attempt / "latent_replay_result.json"
    result_path.write_text(json.dumps({"status": "COMPLETED_READ_ONLY_CPU_LATENT_REPLAY",
        "gpu_hours_charged": 0.0}), encoding="utf-8")
    runner.atomic_json(attempt / "latent_replay_close_receipt.json", {
        "status": "COMPLETED_READ_ONLY_CPU_LATENT_REPLAY", "run_id": run_id,
        "attempt_id": "fit_attempt_001", "device": "cpu", "gpu_hours_charged": 0.0,
        "result_sha256": runner.sha256_file(result_path),
        "trace_sha256": runner.sha256_file(trace_path),
        "owner_manifest_sha256": runner.sha256_file(owner_path),
        "closed_at_utc": "2026-10-07T00:00:00Z"})

    accounting = runner.resource_accounting(tmp_path / "EXP-055", "later-run",
        package_cap_gpu_hours=1.0, total_exp_cap_gpu_hours=48.0,
        exp_settled_before_artifact_ledgers_gpu_hours=1.0,
        package_settled_before_artifact_ledgers_gpu_hours=0.003001485)
    assert accounting["package_settled_gpu_hours"] == pytest.approx(0.003001485)
    assert accounting["exp_settled_gpu_hours"] == pytest.approx(1.0)
    assert accounting["closed_attempts"] == [{"run_id": run_id,
        "attempt": "fit_attempt_001", "close_event": "CPU_LATENT_REPLAY_COMPLETED_NO_GPU",
        "charged_gpu_hours": 0.0, "reserved_gpu_hours": 0.0,
        "close_at_utc": "2026-10-07T00:00:00Z"}]


def test_resource_accounting_accepts_prelaunch_cpu_implementation_failure_without_gpu(
        tmp_path: Path) -> None:
    run_id = "20261007_exp055_transigen_trainpath_recovery_v7"
    attempt = tmp_path / "EXP-055" / run_id / "fit_attempt_001"
    attempt.mkdir(parents=True)
    runner_sha = "a" * 64
    runner.atomic_json(attempt / "TranSiGen_trainpath_layer_trace_manifest.json", {
        "run_id": run_id, "attempt_id": "fit_attempt_001", "status": "CLOSED",
        "close_status": "FAILED_CLOSED", "trace_sha256": None,
        "runner_sha256": runner_sha, "closed_at_utc": "2026-10-07T00:00:00Z",
        "exception_type": "KeyError", "exception_message": "source evidence field"})
    runner.atomic_json(attempt / "pre_gpu_preflight_started.json", {
        "run_id": run_id, "attempt": "fit_attempt_001", "gpu_started": False,
        "reserved_gpu_hours": 0.0, "runner_sha256": runner_sha})
    runner.atomic_json(attempt / "fit_smoke_manifest.json", {
        "status": "CPU_IDENTITY_AND_BUDGET_GATE_PASS_GPU_NOT_YET_STARTED",
        "current_runner_sha256": runner_sha})

    accounting = runner.resource_accounting(tmp_path / "EXP-055", "later-run",
        package_cap_gpu_hours=1.0, total_exp_cap_gpu_hours=48.0,
        exp_settled_before_artifact_ledgers_gpu_hours=1.0,
        package_settled_before_artifact_ledgers_gpu_hours=0.003001485)
    assert accounting["package_settled_gpu_hours"] == pytest.approx(0.003001485)
    assert accounting["exp_settled_gpu_hours"] == pytest.approx(1.0)
    assert accounting["closed_attempts"] == [{"run_id": run_id,
        "attempt": "fit_attempt_001", "close_event": "PRE_MODEL_CPU_IMPLEMENTATION_FAILURE_NO_GPU",
        "charged_gpu_hours": 0.0, "reserved_gpu_hours": 0.0,
        "close_at_utc": "2026-10-07T00:00:00Z",
        "exception_type": "KeyError", "exception_message": "source evidence field",
        "runner_sha256": runner_sha}]


def test_trainpath_parameter_snapshots_always_receive_attempt_directory() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    module = ast.parse(source)
    recovery_function = next(node for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "fit_phase_transigen_trainpath_recovery")
    calls = [node for node in ast.walk(recovery_function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "_trainpath_parameter_snapshot"]
    assert calls
    assert all(len(call.args) >= 3 or any(keyword.arg == "attempt" for keyword in call.keywords)
        for call in calls)


def test_trainpath_recovery_baseline_wrapper_matches_shared_fit_gate_contract() -> None:
    train_baselines = {
        "zero_change": {"object_macro_nmse": 0.8},
        "per_drug_mean_delta": {"object_macro_nmse": 0.6},
    }
    wrapped = runner._wrap_trainpath_recovery_baselines_for_fit_gate(train_baselines)
    result = runner._fit_gate({"object_macro_nmse": 0.5,
                               "object_macro_amplitude_ratio": 1.0}, wrapped)
    assert result["strongest_train_baseline"] == "per_drug_mean_delta"
    assert result["strongest_train_baseline_nmse"] == pytest.approx(0.6)
    assert result["status"] == "PASS"


def test_exp055_lean_monitor_runs_are_registered_and_calibration_requires_v12_source() -> None:
    assert runner.EXP055_TRANSIGEN_LEAN_COST_RUN_ID == (
        "20261007_exp055_transigen_lean_monitor_cost_cal_v1")
    assert runner.EXP055_TRANSIGEN_FIXED300_LEAN_RUN_ID == (
        "20261007_exp055_transigen_fixed300_lean_monitor_v1")
    assert runner.EXP055_TRANSIGEN_COST_SOURCE_RUN_ID == (
        "20261007_exp055_transigen_trainpath_recovery_v12")
    assert runner.EXP055_TRANSIGEN_COST_SOURCE_CHECKPOINT_SHA256 == (
        "30ec5444badc936779e681a64753cbc6f62966bdfd95d02467e4eb6e550f2c69")
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert "TranSiGen cost calibration requires the registered run and v12 checkpoint source id" in source
    assert "preflight_exp055_transigen_lean_cost_calibration" in source


def test_exp055_fixed300_normal_training_path_uses_compact_trace_without_routine_hooks() -> None:
    module = ast.parse(Path(runner.__file__).read_text(encoding="utf-8"))
    fit_function = next(node for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "fit_phase_transigen_trainpath_recovery")
    called = {node.func.id for node in ast.walk(fit_function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert "_lean_monitor_gradient_summary" in called
    assert "_lean_monitor_finite_named_tensors" in called
    assert "_trainpath_bn_snapshot" in called
    assert "_trainpath_attach_layer_hooks" not in called
    assert "_trainpath_attach_eval_failure_hooks" not in called
    assert "_lean_monitor_failure_context" in called
    assert "_lean_monitor_eval_failure_replay" in called


def test_exp055_lean_cost_gate_projects_full_fixed300_with_margin_before_autochain() -> None:
    function = next(node for node in ast.parse(Path(runner.__file__).read_text(encoding="utf-8")).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "run_exp055_transigen_lean_cost_calibration")
    source = ast.get_source_segment(Path(runner.__file__).read_text(encoding="utf-8"), function)
    assert source is not None
    assert "300.0 * (train_seconds + eval_seconds + checkpoint_io_seconds)" in source
    assert "projected_fixed300_without_margin * 1.20 / 3600.0" in source
    assert "projected_fixed300_with_margin <= formal_remaining_after_calibration" in source
    assert "source_checkpoint_used_for_formal_initialization" in source
    assert "PASS_FIXED300_WITHIN_RECOVERY_CAP" in source
