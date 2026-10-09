from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.data import audit_exp044_response_blind_key_coverage as coverage


def _inputs(root: Path) -> tuple[Path, Path, Path]:
    input_dir = root / "inputs"
    input_dir.mkdir(parents=True)
    cell_ids = ["CELL_A", "CELL_B", "CELL_C", "CELL_D"]
    (input_dir / "manifest.json").write_text(json.dumps({
        "identity_resolution_status": coverage.IDENTITY_STATUS,
        "scientific_strict_unseen_accepted": True,
        "target_response_values_read": False,
    }), encoding="utf-8")
    (input_dir / "identity_resolution.json").write_text(json.dumps({
        "status": coverage.IDENTITY_STATUS,
        "strict_unseen_accepted": True,
        "target_response_values_read": False,
    }), encoding="utf-8")
    verification_path = root / "input_verification.json"
    verification_path.write_text(json.dumps({
        "status": "PASS_RESPONSE_BLIND_IDENTITY_RESOLVED",
        "feature_verification_status": "PASS",
        "identity_resolution_status": coverage.IDENTITY_STATUS,
        "scientific_strict_unseen_accepted": True,
        "target_response_values_read": False,
        "input_manifest_sha256": coverage.sha256(input_dir / "manifest.json"),
    }), encoding="utf-8")
    (input_dir / "cell_axis.json").write_text(json.dumps(cell_ids), encoding="utf-8")
    pd.DataFrame({"cell_line_id": cell_ids}).to_csv(
        input_dir / "strict_unseen_cell_lines_v2.csv", index=False,
    )
    pd.DataFrame({"drug_id": ["D1", "D2", "D3", "D4"]}).to_parquet(
        input_dir / "drug_axis.parquet", index=False,
    )
    pd.DataFrame({"cell_line_id": []}).to_csv(
        input_dir / "excluded_source_overlap_cell_lines_v2.csv", index=False,
    )
    (input_dir / "cell_expression.npy").write_bytes(b"feature")
    (input_dir / "drug_ordinal128.npy").write_bytes(b"vectors")
    identity = json.loads((input_dir / "identity_resolution.json").read_text(encoding="utf-8"))
    identity["strict_unseen_ids_sha256"] = coverage.sha256(
        input_dir / "strict_unseen_cell_lines_v2.csv")
    identity["source_overlap_ids_sha256"] = coverage.sha256(
        input_dir / "excluded_source_overlap_cell_lines_v2.csv")
    (input_dir / "identity_resolution.json").write_text(
        json.dumps(identity), encoding="utf-8")
    manifest = json.loads((input_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["files_sha256"] = {
        name: coverage.sha256(input_dir / name)
        for name in ("identity_resolution.json", "strict_unseen_cell_lines_v2.csv",
                     "excluded_source_overlap_cell_lines_v2.csv", "cell_axis.json",
                     "cell_expression.npy", "drug_axis.parquet", "drug_ordinal128.npy")
    }
    (input_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    verification["input_manifest_sha256"] = coverage.sha256(input_dir / "manifest.json")
    verification_path.write_text(json.dumps(verification), encoding="utf-8")
    pairs_path = root / "pairs.parquet"
    pd.DataFrame({
        "cell_line_id": ["CELL_A", "CELL_A", "CELL_A", "CELL_A", "CELL_B", "CELL_B", "CELL_C"],
        "drug_id": ["D1", "D1", "D2", "D3", "D1", "D2", "D4"],
        "response": [91.0, -92.0, 93.0, -94.0, 95.0, -96.0, 97.0],
    }).to_parquet(pairs_path, index=False)
    return input_dir, pairs_path, verification_path, root / "coverage.json"


def test_response_blind_key_audit_requests_only_identity_columns(tmp_path, monkeypatch):
    input_dir, pairs_path, verification_path, output_path = _inputs(tmp_path)
    read_parquet = pd.read_parquet
    observed_pair_columns: list[str] = []

    def guarded_read(path, *args, **kwargs):
        if Path(path) == pairs_path:
            observed_pair_columns.extend(kwargs.get("columns") or [])
            assert kwargs.get("columns") == ["cell_line_id", "drug_id"]
        return read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(coverage.pd, "read_parquet", guarded_read)
    result = coverage.audit(pairs_path, input_dir, verification_path, output_path)

    assert result["status"] == "PASS_RESPONSE_BLIND_IDENTITY_KEY_COVERAGE"
    assert result["target_response_values_read"] is False
    assert result["response_value_columns_requested"] == []
    assert observed_pair_columns == ["cell_line_id", "drug_id"]
    assert result["n_revised_strict_unseen_cells"] == 4
    assert result["n_cells_with_at_least_three_panel_drugs"] == 1
    assert result["n_cells_below_three_panel_drugs"] == 3
    assert len(result["response_blind_cohort_key_sha256"]) == 64
    assert result["unique_panel_drugs_per_cell"] == {
        "CELL_A": 3, "CELL_B": 2, "CELL_C": 1, "CELL_D": 0,
    }


def test_key_audit_digest_binds_the_exact_frozen_cohort_keys_without_values(tmp_path):
    input_dir, pairs_path, verification_path, output_path = _inputs(tmp_path)
    first = coverage.audit(pairs_path, input_dir, verification_path, output_path)

    pairs = pd.read_parquet(pairs_path, columns=["cell_line_id", "drug_id", "response"])
    pairs.loc[(pairs["cell_line_id"] == "CELL_A") & (pairs["drug_id"] == "D3"),
              "drug_id"] = "D4"
    pairs.to_parquet(pairs_path, index=False)
    second = coverage.audit(
        pairs_path, input_dir, verification_path, tmp_path / "coverage_v2.json",
    )

    assert first["response_value_columns_requested"] == []
    assert second["response_value_columns_requested"] == []
    assert first["response_blind_cohort_key_sha256"] != second["response_blind_cohort_key_sha256"]


def test_key_audit_binds_duplicate_key_rows_separately_from_unique_cohort_keys(tmp_path):
    input_dir, pairs_path, verification_path, output_path = _inputs(tmp_path)
    first = coverage.audit(pairs_path, input_dir, verification_path, output_path)
    pairs = pd.read_parquet(pairs_path, columns=["cell_line_id", "drug_id", "response"])
    pairs = pd.concat([pairs, pairs.iloc[[0]]], ignore_index=True)
    pairs.to_parquet(pairs_path, index=False)
    second = coverage.audit(
        pairs_path, input_dir, verification_path, tmp_path / "coverage_v2.json",
    )

    assert first["source_pair_table_key_rows_sha256"] != second["source_pair_table_key_rows_sha256"]
    assert first["source_pair_table_unique_keys_sha256"] == second["source_pair_table_unique_keys_sha256"]
    assert first["response_blind_cohort_key_sha256"] == second["response_blind_cohort_key_sha256"]


def test_response_blind_key_audit_supports_sample_id_only_parquet(tmp_path, monkeypatch):
    input_dir, pairs_path, verification_path, output_path = _inputs(tmp_path)
    pd.DataFrame({
        "sample_id": ["CELL_A", "CELL_A", "CELL_A", "CELL_A", "CELL_B", "CELL_B", "CELL_C"],
        "drug_id": ["D1", "D1", "D2", "D3", "D1", "D2", "D4"],
        "response": [91.0, -92.0, 93.0, -94.0, 95.0, -96.0, 97.0],
    }).to_parquet(pairs_path, index=False)
    read_parquet = pd.read_parquet
    observed_pair_columns: list[str] = []

    def guarded_read(path, *args, **kwargs):
        if Path(path) == pairs_path:
            observed_pair_columns.extend(kwargs.get("columns") or [])
            assert kwargs.get("columns") == ["sample_id", "drug_id"]
        return read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(coverage.pd, "read_parquet", guarded_read)
    result = coverage.audit(pairs_path, input_dir, verification_path, output_path)

    assert result["status"] == "PASS_RESPONSE_BLIND_IDENTITY_KEY_COVERAGE"
    assert result["source_pair_table_columns_requested"] == ["sample_id", "drug_id"]
    assert result["source_pair_table_cell_identity_column"] == "sample_id"
    assert result["source_pair_table_cell_identity_is_verified_sample_axis"] is True
    assert result["response_value_columns_requested"] == []
    assert result["target_response_values_read"] is False
    assert observed_pair_columns == ["sample_id", "drug_id"]


def test_key_audit_accepts_matching_sample_and_cell_line_ids(tmp_path, monkeypatch):
    input_dir, pairs_path, verification_path, output_path = _inputs(tmp_path)
    cell_ids = ["CELL_A", "CELL_A", "CELL_A", "CELL_A", "CELL_B", "CELL_B", "CELL_C"]
    pd.DataFrame({
        "sample_id": cell_ids,
        "cell_line_id": cell_ids,
        "drug_id": ["D1", "D1", "D2", "D3", "D1", "D2", "D4"],
        "response": [91.0, -92.0, 93.0, -94.0, 95.0, -96.0, 97.0],
    }).to_parquet(pairs_path, index=False)
    read_parquet = pd.read_parquet
    observed_pair_columns: list[str] = []

    def guarded_read(path, *args, **kwargs):
        if Path(path) == pairs_path:
            observed_pair_columns.extend(kwargs.get("columns") or [])
            assert kwargs.get("columns") == ["sample_id", "cell_line_id", "drug_id"]
        return read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(coverage.pd, "read_parquet", guarded_read)
    result = coverage.audit(pairs_path, input_dir, verification_path, output_path)

    assert result["status"] == "PASS_RESPONSE_BLIND_IDENTITY_KEY_COVERAGE"
    assert result["source_pair_table_columns_requested"] == [
        "sample_id", "cell_line_id", "drug_id",
    ]
    assert result["response_value_columns_requested"] == []
    assert result["target_response_values_read"] is False
    assert observed_pair_columns == ["sample_id", "cell_line_id", "drug_id"]


def test_key_audit_rejects_disagreeing_sample_and_cell_line_ids(tmp_path, monkeypatch):
    input_dir, pairs_path, verification_path, output_path = _inputs(tmp_path)
    cell_line_ids = ["CELL_A", "CELL_A", "CELL_A", "CELL_A", "CELL_B", "CELL_B", "CELL_C"]
    sample_ids = ["CELL_A", "CELL_A", "CELL_A", "CELL_A", "CELL_B", "CELL_B", "OTHER_C"]
    pd.DataFrame({
        "sample_id": sample_ids,
        "cell_line_id": cell_line_ids,
        "drug_id": ["D1", "D1", "D2", "D3", "D1", "D2", "D4"],
        "response": [91.0, -92.0, 93.0, -94.0, 95.0, -96.0, 97.0],
    }).to_parquet(pairs_path, index=False)
    read_parquet = pd.read_parquet
    observed_pair_columns: list[str] = []

    def guarded_read(path, *args, **kwargs):
        if Path(path) == pairs_path:
            observed_pair_columns.extend(kwargs.get("columns") or [])
            assert kwargs.get("columns") == ["sample_id", "cell_line_id", "drug_id"]
        return read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(coverage.pd, "read_parquet", guarded_read)

    with pytest.raises(ValueError, match="sample_id and cell_line_id columns disagree"):
        coverage.audit(pairs_path, input_dir, verification_path, output_path)

    assert observed_pair_columns == ["sample_id", "cell_line_id", "drug_id"]


def test_key_audit_refuses_unresolved_identity_inputs(tmp_path):
    input_dir, pairs_path, verification_path, output_path = _inputs(tmp_path)
    manifest = json.loads((input_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["scientific_strict_unseen_accepted"] = False
    (input_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="accepted response-blind identity revision"):
        coverage.audit(pairs_path, input_dir, verification_path, output_path)
