"""Synthetic, response-free contract tests for EXP-044 input preparation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.data import materialize_exp044_prism_pharmaformer_inputs as materializer
from scripts.data import verify_exp044_prism_pharmaformer_inputs as verifier


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_fixture(root: Path) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    strict = [f"DEP_STRICT_{index:03d}" for index in range(353)]
    strict[0] = "ACH-000499"
    sanger_ids = [f"SIDM_{index:05d}" for index in range(963)]
    sanger_ids[0] = "SIDM00117"
    sanger_ids[700] = "SIDM00427"
    sanger_ids[800] = "SIDM01117"
    model_ids = [f"DEP_{index:03d}" for index in range(963)]
    model_ids[0] = "ACH-000837"
    overlap = model_ids[:524]
    prism_ids = strict + overlap
    genes = ["ENSG_TEST_A", "ENSG_TEST_B", "ENSG_TEST_C"]
    drug_ids = [f"DRUG_{index:02d}" for index in range(13)]
    smiles = ["C", "CC", "CCC", "CO", "CN", "CF", "CCl", "CBr", "C=O", "C#N", "C1CC1", "c1ccccc1", "CCO"]

    samples_path = root / "prism_samples.parquet"
    pd.DataFrame({"sample_id": list(reversed(prism_ids)),
                  "cell_line_id": list(reversed(prism_ids)),
                  "non_identity_note": ["not read"] * len(prism_ids)}).to_parquet(samples_path, index=False)
    drugs_path = root / "prism_drugs.parquet"
    pd.DataFrame({"drug_id": drug_ids, "non_identity_note": ["not read"] * len(drug_ids)}).to_parquet(drugs_path, index=False)
    gdsc_samples_path = root / "gdsc_samples.parquet"
    pd.DataFrame({"cell_line_id": sanger_ids, "non_identity_note": ["not read"] * len(sanger_ids)}).to_parquet(gdsc_samples_path, index=False)
    bridge_path = root / "depmap_model.csv"
    bridge_rows = [
        {"ModelID": model_id, "SangerModelID": sanger_id,
         "CellLineName": model_id, "non_identity_note": "not read"}
        for index, (model_id, sanger_id) in enumerate(zip(model_ids, sanger_ids))
        if index not in {700, 800}
    ]
    bridge_rows.append({"ModelID": "ACH-002172", "SangerModelID": "SIDM00117",
                        "CellLineName": "NCI-H322M",
                        "non_identity_note": "one-to-many candidate"})
    bridge_rows.append({"ModelID": "UNRELATED_MODEL", "SangerModelID": "UNRELATED_SANGER",
                        "CellLineName": "unrelated",
                        "non_identity_note": "must not affect partition"})
    bridge_rows.extend([
        {"ModelID": "ACH-000499", "SangerModelID": "", "CellLineName": "EW8",
         "non_identity_note": "orthogonal identity candidate"},
        {"ModelID": "ACH-003071", "SangerModelID": "", "CellLineName": "NCI-H748",
         "non_identity_note": "orthogonal identity candidate"},
    ])
    pd.DataFrame(bridge_rows).to_csv(bridge_path, index=False)

    ew8_path = root / "CVCL_1658.json"
    h748_path = root / "CVCL_1588.json"
    for path, accession, cell_name, passport_ids, depmap_ids in (
        (ew8_path, "CVCL_1658", "EW-8", ["SIDM00427", "SIDM01500"],
         ["ACH-000499", "ACH-002194"]),
        (h748_path, "CVCL_1588", "NCI-H748", ["SIDM01117"],
         ["ACH-002176", "ACH-003071"]),
    ):
        path.write_text(json.dumps({"Cellosaurus": {"cell-line-list": [{
            "accession-list": [{"type": "primary", "value": accession}],
            "name-list": [{"type": "identifier", "value": cell_name}],
            "entry-version": "1", "last-updated": "2025-01-01",
            "xref-list": ([{"database": "Cell_Model_Passport", "accession": value}
                           for value in passport_ids]
                          + [{"database": "DepMap", "accession": value}
                             for value in depmap_ids]),
        }]}}), encoding="utf-8")

    strict_path = root / "strict_ids.csv"
    overlap_path = root / "overlap_ids.csv"
    pd.DataFrame({"cell_line_id": strict}).to_csv(strict_path, index=False)
    pd.DataFrame({"cell_line_id": overlap}).to_csv(overlap_path, index=False)
    hashes_path = root / "identity_run_hashes.json"
    hashes_path.write_text(json.dumps({
        "data/artifacts/EXP-019/prism_frozen_v2/strict_unseen_cell_lines.csv": _sha256(strict_path),
        "data/artifacts/EXP-019/prism_frozen_v2/excluded_gdsc_overlap_cell_lines.csv": _sha256(overlap_path),
    }), encoding="utf-8")

    # NPZ order is deliberately different from the Parquet order. Values encode
    # identity so a row-order join produces a detectable error.
    npz_ids = prism_ids[137:] + prism_ids[:137]
    expression = np.empty((len(npz_ids), len(genes)), dtype=np.float32)
    for row, identity in enumerate(npz_ids):
        expression[row] = (float(prism_ids.index(identity)), 5.0, 10.0)
    expression[npz_ids.index(strict[1]), 2] = np.nan
    indexed_npz = root / "prism_indexed.npz"
    # This synthetic sentinel is never accessed by either input preparation path.
    np.savez(indexed_npz, sample_id=np.asarray(npz_ids), ordered_genes=np.asarray(genes),
             expression=expression, pair_response=np.full((1,), 987654.0, dtype=np.float32))

    genes_path = root / "genes.json"
    genes_path.write_text(json.dumps(genes), encoding="utf-8")
    panel_path = root / "panel_drugs.parquet"
    pd.DataFrame({"drug_id": drug_ids, "canonical_isomeric_smiles": smiles,
                  "drug_index": list(range(len(drug_ids)))}).to_parquet(panel_path, index=False)
    vectors_path = root / "panel_vectors.npy"
    np.save(vectors_path, np.arange(13 * 128, dtype=np.float32).reshape(13, 128), allow_pickle=False)

    train_ids = sanger_ids[:577]
    assignments = {identity: ("train" if identity in set(train_ids) else "validation")
                   for identity in sanger_ids}
    source_indices_path = root / "source_fold0_indices.json"
    source_indices_path.write_text(json.dumps({
        "stage": "source", "fold_id": 0, "group_assignments": assignments,
    }), encoding="utf-8")
    checkpoint_path = root / "source_fold0.pt"
    torch.save({
        "gene_transform": {
            "fit_sample_ids": train_ids,
            "mean": np.asarray([0.0, 5.0, 10.0]),
            "scale": np.asarray([2.0, 5.0, 2.0]),
            "min": np.asarray([-1.0, 0.0, -2.0]),
            "range": np.asarray([2.0, 5.0, 4.0]),
        },
        "model_config": {"gene_input_size": len(genes)},
        "smiles_encoder": {"allowed_smiles": smiles},
    }, checkpoint_path)
    return {
        "indexed_npz": indexed_npz,
        "samples": samples_path,
        "drugs": drugs_path,
        "strict_ids": strict_path,
        "overlap_ids": overlap_path,
        "identity_run_hashes": hashes_path,
        "gdsc_samples": gdsc_samples_path,
        "model_csv": bridge_path,
        "cellosaurus_ew8": ew8_path,
        "cellosaurus_h748": h748_path,
        "genes": genes_path,
        "panel_drugs": panel_path,
        "panel_vectors": vectors_path,
        "source_indices": source_indices_path,
        "checkpoint": checkpoint_path,
    }


def _materialize(paths: dict[str, Path], output_dir: Path) -> dict:
    return materializer.materialize(
        paths["indexed_npz"], paths["samples"], paths["drugs"], paths["strict_ids"],
        paths["overlap_ids"], paths["identity_run_hashes"], paths["gdsc_samples"],
        paths["model_csv"], paths["genes"], paths["panel_drugs"], paths["panel_vectors"],
        paths["source_indices"], paths["checkpoint"], paths["cellosaurus_ew8"],
        paths["cellosaurus_h748"], output_dir,
    )


def _verify(paths: dict[str, Path], input_dir: Path) -> dict:
    return verifier.verify(
        paths["indexed_npz"], paths["samples"], paths["drugs"], paths["strict_ids"],
        paths["overlap_ids"], paths["identity_run_hashes"], paths["gdsc_samples"],
        paths["model_csv"], paths["genes"], paths["panel_drugs"], paths["panel_vectors"],
        paths["source_indices"], paths["checkpoint"], paths["cellosaurus_ew8"],
        paths["cellosaurus_h748"], input_dir,
    )


def test_materializer_aligns_by_identity_and_verifier_recomputes_without_response_access(tmp_path, monkeypatch):
    paths = _make_fixture(tmp_path / "fixture")
    access_log: list[str] = []
    numpy_load = np.load

    class GuardedArchive:
        def __init__(self, archive):
            self.archive = archive

        def __enter__(self):
            self.archive.__enter__()
            return self

        def __exit__(self, *args):
            return self.archive.__exit__(*args)

        def __getitem__(self, key):
            access_log.append(str(key))
            if str(key).lower() in {"pair_response", "response", "pairs"}:
                raise AssertionError("response sentinel must never be read")
            return self.archive[key]

    def guarded_load(file, *args, **kwargs):
        archive = numpy_load(file, *args, **kwargs)
        if Path(file) == paths["indexed_npz"]:
            return GuardedArchive(archive)
        return archive

    monkeypatch.setattr(materializer.np, "load", guarded_load)
    monkeypatch.setattr(verifier.np, "load", guarded_load)
    output_dir = tmp_path / "materialized"
    manifest = _materialize(paths, output_dir)
    independent = _verify(paths, output_dir)

    assert manifest["n_strict_unseen_prism_cells"] == 352
    assert manifest["identity_resolution_status"] == "RESOLVED_ORTHOGONAL_AUTHORITATIVE_CROSSWALK"
    assert manifest["n_prism_drugs"] == 13
    assert manifest["target_response_loaded"] is False
    assert manifest["scientific_strict_unseen_accepted"] is True
    assert manifest["identity_bridge_audit"]["unmapped_source_sanger_ids"] == []
    assert manifest["identity_bridge_audit"]["revised_partition_from_frozen_exp019"] == {
        "strict_before": 353, "overlap_before": 524,
        "strict_after": 352, "overlap_after": 525,
        "moved_from_strict_to_overlap": ["ACH-000499"],
    }
    assert manifest["identity_bridge_audit"]["one_to_many_sanger_model_candidates"] == {
        "SIDM00427": ["ACH-000499", "ACH-002194"],
        "SIDM00117": ["ACH-000837", "ACH-002172"],
        "SIDM01117": ["ACH-002176", "ACH-003071"],
    }
    assert manifest["identity_bridge_audit"]["reproduces_frozen_exp019_partition"] is True
    assert independent["status"] == "PASS_RESPONSE_BLIND_IDENTITY_RESOLVED"
    assert independent["feature_verification_status"] == "PASS"
    assert independent["scientific_strict_unseen_accepted"] is True
    assert independent["target_response_values_read"] is False
    assert set(access_log) == {"sample_id", "ordered_genes", "expression"}
    cell_axis = json.loads((output_dir / "cell_axis.json").read_text(encoding="utf-8"))
    assert cell_axis == sorted(cell_axis)
    expression_out = numpy_load(output_dir / "cell_expression.npy", allow_pickle=False)
    # The first strict row has a missing third gene, filled by the source mean.
    assert cell_axis[:2] == ["DEP_STRICT_001", "DEP_STRICT_002"]
    assert expression_out[0, 0] == pytest.approx(0.75)
    assert expression_out[1, 0] == pytest.approx(1.0)
    assert expression_out[0, 2] == pytest.approx(0.5)
    assert independent["max_expression_abs_error"] == 0.0


def test_missing_model_id_for_matched_source_fails_and_failure_receipt_is_preserved(tmp_path):
    paths = _make_fixture(tmp_path / "fixture")
    bridge = pd.read_csv(paths["model_csv"])
    bridge = pd.concat([bridge, pd.DataFrame([{
        "ModelID": np.nan, "SangerModelID": "SIDM_00001", "CellLineName": "broken",
        "non_identity_note": "matched source with missing ModelID must fail",
    }])], ignore_index=True)
    bridge.to_csv(paths["model_csv"], index=False)
    output_dir = tmp_path / "failed_materialization"

    with pytest.raises(ValueError, match="Sanger identity has no DepMap ModelID"):
        _materialize(paths, output_dir)

    receipt_path = output_dir / "FAILURE.json"
    receipt_bytes = receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes)
    assert receipt["status"] == "FAIL"
    assert receipt["target_response_loaded"] is False
    assert receipt["response_label_files_opened"] == []
    with pytest.raises(FileExistsError):
        _materialize(paths, output_dir)
    assert receipt_path.read_bytes() == receipt_bytes


def test_conservative_bridge_unions_all_candidates_and_ignores_unrelated_ids():
    bridge = pd.DataFrame([
        {"ModelID": "M1", "SangerModelID": "S1"},
        {"ModelID": "M2", "SangerModelID": "S1"},
        {"ModelID": "M1", "SangerModelID": "S2"},
        {"ModelID": "UNRELATED", "SangerModelID": "S_OUTSIDE"},
    ])
    strict, overlap, audit = materializer.audit_identity_bridge(
        {"M1", "M2", "PRISM_STRICT"}, {"S1", "S2", "S3"}, bridge,
    )
    assert overlap == {"M1", "M2"}
    assert strict == {"PRISM_STRICT"}
    assert audit["candidate_depmap_model_id_count"] == 2
    assert audit["one_to_many_sanger_model_candidates"] == {"S1": ["M1", "M2"]}
    assert audit["unmapped_source_sanger_ids"] == ["S3"]


@pytest.mark.parametrize(
    ("bridge", "source_ids", "message"),
    [
        (pd.DataFrame(columns=["ModelID", "SangerModelID"]), {"S1"}, "maps no GDSC Sanger identities"),
        (pd.DataFrame([{"ModelID": "M1", "SangerModelID": "OTHER"}]),
         {"S1"}, "maps no GDSC Sanger identities"),
        (pd.DataFrame([{"ModelID": np.nan, "SangerModelID": "S1"}]),
         {"S1"}, "Sanger identity has no DepMap ModelID"),
    ],
)
def test_empty_unrelated_only_and_missing_model_bridges_fail(bridge, source_ids, message):
    with pytest.raises(ValueError, match=message):
        materializer.audit_identity_bridge({"M1", "P"}, source_ids, bridge)


def test_independent_verifier_detects_gene_axis_drift(tmp_path):
    paths = _make_fixture(tmp_path / "fixture")
    output_dir = tmp_path / "materialized"
    _materialize(paths, output_dir)

    # Updating the receipt to a changed NPZ checksum cannot make an incorrect
    # ordered gene list pass the independent source-axis recomputation.
    with np.load(paths["indexed_npz"], allow_pickle=False) as original:
        values = {key: original[key] for key in ("sample_id", "ordered_genes", "expression")}
    values["ordered_genes"] = values["ordered_genes"][::-1]
    np.savez(paths["indexed_npz"], **values)
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["inputs_sha256"]["prism_indexed_npz"] = _sha256(paths["indexed_npz"])
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ordered genes differ"):
        _verify(paths, output_dir)


def test_independent_verifier_recomputes_source_transformed_features(tmp_path):
    paths = _make_fixture(tmp_path / "fixture")
    output_dir = tmp_path / "materialized"
    _materialize(paths, output_dir)

    expression_path = output_dir / "cell_expression.npy"
    changed = np.load(expression_path, allow_pickle=False)
    changed[0, 0] += 0.25
    np.save(expression_path, changed, allow_pickle=False)
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files_sha256"]["cell_expression.npy"] = _sha256(expression_path)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="independent source-fold transform"):
        _verify(paths, output_dir)
