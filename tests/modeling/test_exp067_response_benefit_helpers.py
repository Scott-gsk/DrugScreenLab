from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import yaml
from threadpoolctl import threadpool_info


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
SPEC = importlib.util.spec_from_file_location(
    "exp067_response_benefit_runner",
    ROOT / "scripts/modeling/run_response_benefit.py",
)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


ARRAY_COLUMNS = (
    "pair_id", "source_object_id", "lincs_cell_id", "full_inchikey",
    "drug_feature_row_index", "log10_dose_um", "log10_time_hours",
    "nested_fold_id", "model_source", "prediction", "shuffled_prediction",
    "shuffle_donor_object_id", "shuffle_donor_lincs_cell_id",
    "shuffle_donor_depmap_model_id",
)


def _two_row_payload() -> dict[str, np.ndarray]:
    return {
        "pair_id": np.asarray(["p0|k0", "p1|k1"]),
        "source_object_id": np.asarray(["p0", "p1"]),
        "lincs_cell_id": np.asarray(["c0", "c1"]),
        "full_inchikey": np.asarray(["k0", "k1"]),
        "drug_feature_row_index": np.asarray([0, 1]),
        "log10_dose_um": np.asarray([1.0, 1.0]),
        "log10_time_hours": np.asarray([1.0, 1.0]),
        "nested_fold_id": np.asarray([0, 1]),
        "model_source": np.asarray(["inner_oof", "inner_oof"]),
        "prediction": np.vstack([np.arange(978), np.arange(978) + 1]),
        "shuffled_prediction": np.vstack([np.arange(978) + 2, np.arange(978) + 3]),
        "shuffle_donor_object_id": np.asarray(["p2", "p3"]),
        "shuffle_donor_lincs_cell_id": np.asarray(["c2", "c3"]),
        "shuffle_donor_depmap_model_id": np.asarray(["m2", "m3"]),
    }


def test_nested_partition_frame_keeps_978_gene_vectors_bound_to_rows() -> None:
    frame = RUNNER._nested_partition_frame({"train": _two_row_payload()}, "train")
    assert tuple(frame.columns) == ARRAY_COLUMNS
    assert len(frame) == 2
    assert np.array_equal(frame.loc[0, "prediction"], np.arange(978))
    assert np.array_equal(frame.loc[1, "prediction"], np.arange(978) + 1)
    assert np.array_equal(frame.loc[0, "shuffled_prediction"], np.arange(978) + 2)


def test_nested_partition_frame_rejects_wrong_gene_axis() -> None:
    payload = _two_row_payload()
    payload["prediction"] = payload["prediction"][:, :977]
    with pytest.raises(ValueError, match=r"shape \[rows,978\]"):
        RUNNER._nested_partition_frame({"train": payload}, "train")


def test_lincs_cell_alias_is_bound_from_frozen_identity() -> None:
    source = pd.DataFrame({"cell_id": pd.Series(["LCL_A", "LCL_B"], dtype="string")})
    bound = RUNNER._bind_lincs_cell_id(source)
    assert bound["lincs_cell_id"].tolist() == ["LCL_A", "LCL_B"]
    assert source.columns.tolist() == ["cell_id"]


def test_lincs_cell_alias_rejects_conflicting_identity() -> None:
    source = pd.DataFrame({"cell_id": ["LCL_A"], "lincs_cell_id": ["LCL_B"]})
    with pytest.raises(ValueError, match="must exactly match"):
        RUNNER._bind_lincs_cell_id(source)


def test_nested_response_config_uses_measured_loader_cost_without_mutating_source() -> None:
    source = {"experiment_id": "EXP-066"}
    bound = RUNNER._bind_measured_response_loader_cpu(source, 12.5)
    assert bound["data_load_cpu_seconds"] == 12.5
    assert "data_load_cpu_seconds" not in source


def test_nested_response_config_rejects_invalid_loader_cost() -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        RUNNER._bind_measured_response_loader_cpu({"experiment_id": "EXP-066"}, float("nan"))


def _valid_provenance() -> tuple[dict[str, object], pd.DataFrame, set[str], set[str], dict[str, str]]:
    fold_by_patient = {f"p{i}": i for i in range(5)}
    expected_train = {"p1", "p2", "p3", "p4"}
    expected_predict = {"p0"}
    pair_frame = pd.DataFrame({
        "source_object_id": list(fold_by_patient),
        "fold_id": list(fold_by_patient.values()),
    })
    inner = []
    for fold in (1, 2, 3, 4):
        predicted = {f"p{fold}"}
        inner.append({
            "inner_validation_fold_id": fold,
            "training_patient_ids": sorted(expected_train - predicted),
            "prediction_patient_ids": sorted(predicted),
            "training_condition_count": 4,
            "prediction_pair_count": 1,
            "cpu_seconds": 0.1,
            "model_source": "inner_oof",
            "models": {},
        })
    provenance: dict[str, object] = {
        "outer_fold_id": 0,
        "outer_train_patient_ids": sorted(expected_train),
        "outer_predict_patient_ids": sorted(expected_predict),
        "target_axis_hash": "frozen-axis",
        "target_gene_ids": [f"g{i}" for i in range(978)],
        "inner_fold_provenance": inner,
        "outer_model_provenance": {
            "training_patient_ids": sorted(expected_train),
            "prediction_patient_ids": sorted(expected_predict),
            "training_condition_count": 4,
            "prediction_pair_count": 1,
            "cpu_seconds": 0.2,
            "models": {},
        },
        "cost": {},
    }
    return provenance, pair_frame, expected_train, expected_predict, {"target_axis_hash": "frozen-axis"}


def test_nested_patient_provenance_requires_exact_inner_partition() -> None:
    provenance, pair_frame, train, heldout, config = _valid_provenance()
    checked = RUNNER._validate_nested_patient_provenance(
        provenance, pair_frame, 0, train, heldout, config,
    )
    assert checked["target_gene_count"] == 978
    assert len(checked["inner_fold_provenance"]) == 4


def test_nested_patient_provenance_rejects_outer_heldout_leakage() -> None:
    provenance, pair_frame, train, heldout, config = _valid_provenance()
    inner = provenance["inner_fold_provenance"]
    assert isinstance(inner, list)
    inner[0]["training_patient_ids"].append("p0")
    with pytest.raises(ValueError, match="incorrect PatientID training/prediction set"):
        RUNNER._validate_nested_patient_provenance(
            provenance, pair_frame, 0, train, heldout, config,
        )


def test_cpu_thread_setting_is_explicit_and_validated() -> None:
    assert RUNNER._configured_cpu_threads({"cpu_threads": 1}) == 1
    assert RUNNER._configured_cpu_threads({}) == 1
    for invalid in (0, -1, True, 1.5, "1"):
        with pytest.raises(ValueError, match="positive integer"):
            RUNNER._configured_cpu_threads({"cpu_threads": invalid})


def test_cpu_projection_uses_fold_history_and_cumulative_remainder() -> None:
    projected, cumulative = RUNNER._project_package_cpu_seconds(1547.5639088, 10.0, [100.0, 120.0], 3, 20.0)
    assert projected == pytest.approx((10.0 + 3 * 110.0 + 20.0) * 1.2)
    assert cumulative == pytest.approx(1547.5639088 + projected)
    with pytest.raises(ValueError, match="fold reference costs"):
        RUNNER._project_cpu_seconds(10.0, [0.0], 3, 20.0)


def test_disabled_cpu_caps_are_informational_and_known_debits_are_preserved() -> None:
    assert RUNNER._optional_cpu_cap(None, "cap") is None
    assert RUNNER._optional_cpu_cap("disabled", "cap") is None
    assert RUNNER._optional_cpu_cap(1800, "cap") == 1800.0
    configured, effective = RUNNER._effective_cpu_debit(
        {"b_package_cpu_spent_before_seconds": None},
        "b_package_cpu_spent_before_seconds", RUNNER.KNOWN_B_PACKAGE_CPU_DEBIT_SECONDS,
    )
    assert configured == 0.0
    assert effective == RUNNER.KNOWN_B_PACKAGE_CPU_DEBIT_SECONDS
    configured, effective = RUNNER._effective_cpu_debit(
        {"a_b_cpu_spent_before_seconds": 3000.0},
        "a_b_cpu_spent_before_seconds", RUNNER.KNOWN_SHARED_AB_CPU_DEBIT_SECONDS,
    )
    assert configured == effective == 3000.0
    failure_cost = RUNNER._failure_cpu_accounting(
        {"b_package_cpu_spent_before_seconds": None, "a_b_cpu_spent_before_seconds": None}, 2.5,
    )
    assert failure_cost["status"] == "INFORMATIONAL_ONLY"
    assert failure_cost["b_package_cumulative_cpu_seconds"] == pytest.approx(
        RUNNER.KNOWN_B_PACKAGE_CPU_DEBIT_SECONDS + 2.5,
    )


def test_configured_single_thread_limits_loaded_numeric_pools() -> None:
    with RUNNER.threadpool_limits(limits=RUNNER._configured_cpu_threads({"cpu_threads": 1})):
        pools = threadpool_info()
        assert pools
        assert all(pool["num_threads"] <= 1 for pool in pools)


def _make_synthetic_fold0_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, pd.DataFrame, dict[str, np.ndarray]]:
    source = tmp_path / RUNNER.FOLD0_REUSE_RUN_ID
    source.mkdir()
    n = 816
    expected = pd.DataFrame({
        "pair_id": [f"pair-{i:04d}" for i in range(n)],
        "source_object_id": [f"obj-{i % 7}" for i in range(n)],
        "cellid": [f"cell-{i % 34}" for i in range(n)],
        "full_inchikey": [f"inchikey-{i % 138}" for i in range(n)],
        "fold_id": np.zeros(n, dtype=np.int64),
        "aac_published": np.linspace(0.0, 1.0, n),
    })
    predictions = {arm: np.linspace(i, i + 1, n) for i, arm in enumerate(RUNNER.ARMS)}
    table = expected.copy()
    for arm, values in predictions.items():
        table[f"prediction__{arm}"] = values
    prediction_path = source / "fold_0_pair_predictions.tsv"
    table.to_csv(prediction_path, sep="\t", index=False, lineterminator="\n")
    object_path = source / "fold_0_object_metrics.tsv"
    drug_path = source / "fold_0_drug_metrics.tsv"
    audit_path = source / "pair_domain_audit.json"
    gate_path = source / "first_outer_fold_cost_gate.json"
    object_path.write_text("synthetic-object-metrics\n", encoding="utf-8")
    drug_path.write_text("synthetic-drug-metrics\n", encoding="utf-8")
    RUNNER._write_json(audit_path, {"synthetic": True})
    gate = {"status": "FAIL", "measured_outer_fold": 0, "measured_fold_cpu_seconds": 1123.9739088}
    RUNNER._write_json(gate_path, gate)
    fold_summary = {
        "fold_id": 0,
        "fold": {"fold_id": 0, "train_patient_count": 27, "test_patient_count": 7,
                 "train_pair_count": 3149, "test_pair_count": 816},
        "metrics": {},
        "prediction_sha256": RUNNER._sha256(prediction_path),
        "object_metrics_sha256": RUNNER._sha256(object_path),
        "drug_metrics_sha256": RUNNER._sha256(drug_path),
    }
    fold_summary_path = source / "fold_0_summary.json"
    RUNNER._write_json(fold_summary_path, fold_summary)
    summary_path = source / "run_summary.json"
    RUNNER._write_json(summary_path, {
        "run_id": RUNNER.FOLD0_REUSE_RUN_ID, "status": "COST_GATE_FAILED",
        "scientific_comparison_valid": False, "outer_folds_completed": 1,
        "metadata_gate_receipt_sha256": RUNNER.METADATA_RECEIPT_SHA256,
        "pair_count": 3965, "object_count": 34, "drug_count": 138, "target_gene_count": 978,
    })
    hashes = {
        "run_summary_sha256": RUNNER._sha256(summary_path),
        "fold0_summary_sha256": RUNNER._sha256(fold_summary_path),
        "fold0_predictions_sha256": RUNNER._sha256(prediction_path),
        "fold0_object_metrics_sha256": RUNNER._sha256(object_path),
        "fold0_drug_metrics_sha256": RUNNER._sha256(drug_path),
        "pair_domain_audit_sha256": RUNNER._sha256(audit_path),
        "cost_gate_sha256": RUNNER._sha256(gate_path),
    }
    receipt_path = source / "root_cost_gate_readback.json"
    RUNNER._write_json(receipt_path, {
        "root_verification_status": "PASS_PARTIAL_FOLD_AND_COST_GATE_ONLY",
        "run_id": RUNNER.FOLD0_REUSE_RUN_ID,
        "metadata_gate_receipt_sha256": RUNNER.METADATA_RECEIPT_SHA256,
        "denominators": {
            "frozen_source_conditions": 2830, "frozen_source_objects": 34,
            "full_chemical_keys": 138, "genes": 978,
            "fold0_heldout_pairs": 816, "fold0_heldout_objects": 7,
        },
        "hashes": hashes,
    })
    monkeypatch.setattr(RUNNER, "FOLD0_REUSE_ROOT_RECEIPT_SHA256", RUNNER._sha256(receipt_path))
    return source, expected, predictions


def test_verified_fold0_resume_preserves_four_arm_predictions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source, expected, expected_predictions = _make_synthetic_fold0_artifacts(tmp_path, monkeypatch)
    before = RUNNER._sha256(source / "fold_0_pair_predictions.tsv")
    actual, detail, historical_cost = RUNNER._load_reusable_fold0(source, expected)
    assert detail["fold_id"] == 0
    assert detail["train_patient_count"] == 27
    assert detail["test_patient_count"] == 7
    assert historical_cost == pytest.approx(1123.9739088)
    for arm in RUNNER.ARMS:
        assert np.allclose(actual[arm], expected_predictions[arm], rtol=0, atol=1e-12)
    assert RUNNER._sha256(source / "fold_0_pair_predictions.tsv") == before


def test_verified_fold0_resume_fails_closed_on_tampered_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source, expected, _ = _make_synthetic_fold0_artifacts(tmp_path, monkeypatch)
    with (source / "fold_0_pair_predictions.tsv").open("a", encoding="utf-8") as stream:
        stream.write("tamper\n")
    with pytest.raises(IOError, match="immutable root receipt"):
        RUNNER._load_reusable_fold0(source, expected)


def _synthetic_full_pair_frame(fold0: pd.DataFrame) -> pd.DataFrame:
    rows = [fold0.copy()]
    fold_objects = {
        1: [f"obj-{i}" for i in range(7, 14)],
        2: [f"obj-{i}" for i in range(14, 21)],
        3: [f"obj-{i}" for i in range(21, 28)],
        4: [f"obj-{i}" for i in range(28, 34)],
    }
    row_counts = {1: 787, 2: 787, 3: 787, 4: 788}
    for fold, count in row_counts.items():
        objects = fold_objects[fold]
        frame = pd.DataFrame({
            "pair_id": [f"synthetic-f{fold}-pair-{i:04d}" for i in range(count)],
            "source_object_id": [objects[i % len(objects)] for i in range(count)],
            "cellid": [f"cell-{objects[i % len(objects)]}" for i in range(count)],
            "full_inchikey": [f"inchikey-{i % 138}" for i in range(count)],
            "fold_id": np.full(count, fold, dtype=np.int64),
            "aac_published": 0.1 + ((np.arange(count) * 17 + fold) % 997) / 997.0,
        })
        rows.append(frame)
    result = pd.concat(rows, ignore_index=True)
    assert len(result) == 3965
    assert result["source_object_id"].nunique() == 34
    return result


def test_resume_reuses_fold0_and_continues_all_folds_past_old_cpu_caps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, fold0_frame, predictions0 = _make_synthetic_fold0_artifacts(tmp_path, monkeypatch)
    source_predictions_hash = RUNNER._sha256(source / "fold_0_pair_predictions.tsv")
    pair_frame = _synthetic_full_pair_frame(fold0_frame)
    built_folds: list[int] = []
    monkeypatch.setattr(RUNNER, "load_frozen_response_domain", lambda *args, **kwargs: {
        "load_cpu_seconds": 0.01,
        "condition_metadata": pd.DataFrame({"condition": range(2830)}),
    })
    monkeypatch.setattr(RUNNER, "_load_pair_frame", lambda *args, **kwargs: (
        pair_frame.copy(), {"included_finite_aac_pairs": 3965},
    ))
    monkeypatch.setattr(RUNNER, "_pair_inputs", lambda domain, frame: (
        np.empty((0, 978), dtype=np.float64), np.empty((0, 2048), dtype=np.float64), frame.copy(),
    ))

    def build_fold(_loader, frame, _rna, _drug, _raw, fold, _response_config):
        built_folds.append(fold)
        test_ix = np.flatnonzero(frame["fold_id"].to_numpy(dtype=np.int64) == fold)
        train_ix = np.flatnonzero(frame["fold_id"].to_numpy(dtype=np.int64) != fold)
        predictions = {
            arm: np.sin(np.arange(len(test_ix), dtype=np.float64) / 19.0 + index + fold / 10.0)
            for index, arm in enumerate(RUNNER.ARMS)
        }
        detail = {
            "test_indices": test_ix,
            "test_frame": frame.iloc[test_ix].reset_index(drop=True),
            "test_y": frame.iloc[test_ix]["aac_published"].to_numpy(dtype=np.float64),
            "train_y": frame.iloc[train_ix]["aac_published"].to_numpy(dtype=np.float64),
            "fold_detail": {
                "fold_id": fold,
                "train_patient_count": int(frame.iloc[train_ix]["source_object_id"].nunique()),
                "test_patient_count": int(frame.iloc[test_ix]["source_object_id"].nunique()),
                "train_pair_count": int(len(train_ix)),
                "test_pair_count": int(len(test_ix)),
            },
        }
        return predictions, [], detail

    monkeypatch.setattr(RUNNER, "_build_fold_predictions", build_fold)
    monkeypatch.setattr(RUNNER, "_paired_bootstrap", lambda object_metrics, arm, reference, **kwargs: {
        "objects": int(len(object_metrics)), "mean_absolute_rmse_reduction": 0.0,
        "ci95": [-0.01, 0.01], "mean_relative_rmse_reduction": 0.0,
    })

    config_path = tmp_path / "synthetic_config.yaml"
    config = yaml.safe_load((ROOT / "configs/experiments/EXP-067-response-benefit.yaml").read_text(encoding="utf-8"))
    config["outputs"]["root_dir"] = str(tmp_path)
    config["resources"]["b_package_cpu_spent_before_seconds"] = None
    config["resources"]["a_b_cpu_spent_before_seconds"] = None
    config["resources"]["cpu_seconds_package_cap"] = None
    config["resources"]["cpu_seconds_shared_AB_absolute_cap"] = None
    config["resources"]["historical_cpu_seconds_package_cap"] = 1
    config["resources"]["historical_cpu_seconds_shared_AB_absolute_cap"] = 1
    config["resources"]["root_readback_cpu_allowance_seconds"] = 0
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    summary = RUNNER.run_exp067(
        config_path, "synthetic_resume_full_fivefold", RUNNER.FOLD0_REUSE_RUN_ID,
    )
    assert summary["status"] in {"COMPLETE_CANDIDATE_PASS", "COMPLETE_CANDIDATE_CLOSED"}
    assert summary["scientific_comparison_valid"] is True
    assert summary["outer_folds_completed"] == 5
    assert built_folds == [1, 2, 3, 4]
    assert summary["folds"][0]["reused_from_run_id"] == RUNNER.FOLD0_REUSE_RUN_ID
    assert summary["cost"]["budget_enforcement"] == "informational_only"
    assert summary["cost_gates"][-1]["historical_package_cap_exceeded_information_only"] is True
    assert summary["cost_gates"][-1]["historical_shared_AB_cap_exceeded_information_only"] is True
    assert summary["cost"]["package_cumulative_cpu_seconds"] >= RUNNER.KNOWN_B_PACKAGE_CPU_DEBIT_SECONDS
    assert RUNNER._sha256(source / "fold_0_pair_predictions.tsv") == source_predictions_hash
    for arm in RUNNER.ARMS:
        assert np.allclose(
            pd.read_csv(tmp_path / "synthetic_resume_full_fivefold" / "fold_0_pair_predictions.tsv", sep="\t")
            [f"prediction__{arm}"].to_numpy(dtype=np.float64), predictions0[arm], rtol=0, atol=1e-12,
        )


def test_failure_receipt_preserves_prior_cpu_debits_and_propagates_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = yaml.safe_load((ROOT / "configs/experiments/EXP-067-response-benefit.yaml").read_text(encoding="utf-8"))
    config["outputs"]["root_dir"] = str(tmp_path)
    config["resources"]["b_package_cpu_spent_before_seconds"] = None
    config["resources"]["a_b_cpu_spent_before_seconds"] = None
    config_path = tmp_path / "failure_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    def fail(*_args, **_kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(RUNNER, "run_exp067", fail)
    monkeypatch.setattr(sys, "argv", [
        "run_response_benefit.py", "--config", str(config_path), "--run-id", "synthetic_failure",
    ])
    with pytest.raises(RuntimeError, match="synthetic failure"):
        RUNNER.main()
    receipt = json.loads((tmp_path / "synthetic_failure" / "failure_close_receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "FAILED_CLOSED"
    assert receipt["scientific_result_available"] is False
    assert receipt["cpu_accounting"]["cpu_budget_enforcement"] is False
    assert receipt["cpu_accounting"]["b_package_cumulative_cpu_seconds"] >= RUNNER.KNOWN_B_PACKAGE_CPU_DEBIT_SECONDS
