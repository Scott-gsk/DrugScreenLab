"""Create a provenance-only audit for IMPROVE CSA and ChemPert.

This script deliberately does not materialize the IMPROVE benchmark archive or
transform any third-party matrix.  It records the immutable references and the
small public assets acquired for a later, separately approved data contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import Counter
from datetime import date
from pathlib import Path

import pandas as pd


ACQUISITION_DATE = "2026-09-16"
RAW_RELATIVE = Path("raw/resource_acquisition_20260916/improve_chempert")
ARTIFACT_RELATIVE = Path("artifacts/resource_acquisition_20260916/improve_chempert")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def asset(path: Path, root: Path) -> dict[str, object]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data"))
    args = parser.parse_args()
    root = args.root.resolve()
    raw = root / RAW_RELATIVE
    artifact_dir = root / ARTIFACT_RELATIVE
    artifact_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "improve_zenodo_metadata": raw / "improve_zenodo_15258883_metadata.json",
        "improve_source": raw / "improve_v0.1.0_source.zip",
        "improve_benchmark": raw / "improve_csa_data_zenodo_15258883.zip",
        "chempert_response_metadata": raw
        / "chempert_information_for_transcriptional_responses.csv",
        "chempert_expression_metadata": raw
        / "chempert_information_for_gene_expression.csv",
        "chempert_perturbagen_metadata": raw
        / "chempert_information_for_perturbagens.csv",
        "chempert_response_rds": raw / "chempert_transcriptional_responses.rds",
        "chempert_target_rds": raw / "chempert_targets_of_perturbagens.rds",
        "chempert_expression_rds": raw / "chempert_gene_expression_profiles.rds",
    }
    missing = [key for key, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing acquired assets: {missing}")

    with paths["improve_zenodo_metadata"].open(encoding="utf-8") as handle:
        zenodo = json.load(handle)
    expected_zenodo_file = zenodo["files"]
    if len(expected_zenodo_file) != 1 or expected_zenodo_file[0]["key"] != "csa_data.zip":
        raise ValueError("Unexpected IMPROVE Zenodo file manifest")
    if paths["improve_benchmark"].stat().st_size != expected_zenodo_file[0]["size"]:
        raise ValueError("IMPROVE benchmark size does not match the Zenodo manifest")
    expected_md5 = expected_zenodo_file[0]["checksum"].removeprefix("md5:")
    if md5(paths["improve_benchmark"]) != expected_md5:
        raise ValueError("IMPROVE benchmark MD5 does not match the Zenodo manifest")
    with zipfile.ZipFile(paths["improve_benchmark"]) as archive:
        benchmark_members = archive.namelist()
        corrupted_member = archive.testzip()
    response_member = "csa_data/raw_data/y_data/response.tsv"
    if corrupted_member is not None or response_member not in benchmark_members:
        raise ValueError(f"Invalid IMPROVE benchmark zip; corrupted member={corrupted_member!r}")

    with zipfile.ZipFile(paths["improve_source"]) as archive:
        members = archive.namelist()
        csa_members = [member for member in members if "/workflows/csa/" in member]
        has_download_script = any(member.endswith("download_csa.sh") for member in members)
        has_mit_license = any(member.endswith("LICENSE") for member in members)
    if not csa_members or not has_download_script or not has_mit_license:
        raise ValueError("IMPROVE v0.1.0 archive lacks expected CSA workflow files")

    responses = pd.read_csv(paths["chempert_response_metadata"])
    expression = pd.read_csv(paths["chempert_expression_metadata"])
    perturbagens = pd.read_csv(paths["chempert_perturbagen_metadata"], sep="@", quotechar='"')
    needed_response_columns = {
        "Response_ID",
        "Chemical_ID",
        "Data_Accession",
        "Cell_Source",
        "Species",
        "Expression_ID",
    }
    if not needed_response_columns.issubset(responses.columns):
        raise ValueError("ChemPert response metadata columns are not as expected")

    accession_counts = Counter(responses["Data_Accession"].astype(str))
    source_overlap = {
        "gse92742_existing_lincs_phase_i": accession_counts.get("GSE92742", 0),
        "gse70138_lincs_phase_ii_not_registered_locally": accession_counts.get("GSE70138", 0),
        "gse139944_sci_plex": accession_counts.get("GSE139944", 0),
    }
    total_responses = len(responses)
    for key, value in list(source_overlap.items()):
        if key != "gse139944_sci_plex":
            source_overlap[f"{key}_fraction"] = round(value / total_responses, 6)

    rds_headers = {
        key: paths[key].read_bytes()[:16].hex()
        for key in ("chempert_response_rds", "chempert_target_rds", "chempert_expression_rds")
    }
    if not all(header.startswith("1f8b08") for header in rds_headers.values()):
        raise ValueError("ChemPert RDS payloads do not have the expected gzip header")

    audit = {
        "audit_version": "1.0",
        "acquisition_date": ACQUISITION_DATE,
        "scope": "provenance_and_candidate_admission_only_no_training_or_formal_evaluation",
        "improve": {
            "resource": "IMPROVE cross-study analysis (CSA) benchmark",
            "fixed_references": {
                "benchmark_zenodo_record": "https://zenodo.org/records/15258883",
                "benchmark_doi": "10.5281/zenodo.15258883",
                "framework_source_tag": "v0.1.0",
                "framework_source_url": "https://github.com/JDACS4C-IMPROVE/IMPROVE/tree/v0.1.0",
            },
            "zenodo_manifest": {
                "title": zenodo["metadata"]["title"],
                "publication_date": zenodo["metadata"]["publication_date"],
                "access_right": zenodo["metadata"]["access_right"],
                "file": expected_zenodo_file[0],
                "license_in_zenodo_metadata": zenodo["metadata"].get("license"),
            },
            "benchmark_zip_validation": {
                "member_count": len(benchmark_members),
                "crc_error_member": corrupted_member,
                "contains_response_table": response_member in benchmark_members,
            },
            "local_assets": {
                key: asset(path, root) for key, path in paths.items() if key.startswith("improve_")
            },
            "validated_source_archive": {
                "zip_member_count": len(members),
                "csa_workflow_member_count": len(csa_members),
                "has_download_csa_script": has_download_script,
                "has_mit_license": has_mit_license,
                "documented_studies": ["CCLE", "gCSI", "GDSCv1", "GDSCv2", "CTRPv2"],
                "documented_fixed_split_layout": "per-study split_0 through split_9 train/validation/test files",
            },
            "local_status": "FULL_FIXED_BENCHMARK_REFERENCE_CODE_AND_MANIFEST_ACQUIRED_PENDING_CONTRACT",
            "duplication_note": "The 288,561,534-byte benchmark is intentionally retained as one immutable independently versioned package. It includes unified copies of existing CCLE/GDSCv2/CTRPv2 assets as well as gCSI/GDSCv1; it must not silently replace project source assets.",
            "potential_use": "reproduce a fixed five-study cross-dataset evaluation after a separate approved adapter maps this project's inputs and endpoint direction to the benchmark contract.",
            "not_permitted_by_this_audit": [
                "treating IMPROVE results as PDO or clinical validation",
                "training on or selecting thresholds with test-study labels",
                "silently replacing the project GDSC/CTRP/CCLE source versions or endpoints",
            ],
        },
        "chempert": {
            "resource": "ChemPert perturbation-transcriptomics database",
            "references": {
                "download_page": "https://chempert.uni.lu/downloads",
                "terms": "https://chempert.uni.lu/pages/tac",
                "paper_doi": "10.1093/nar/gkac862",
                "code_repository": "https://gitlab.lcsb.uni.lu/CBG/chempert",
            },
            "versioning": "No release identifier is exposed by the download page; endpoints were fixed by URL, HTTP Last-Modified values, acquisition date, byte counts, and SHA-256.",
            "access_and_license": {
                "download_access": "public_without_login_at_acquisition",
                "data_license": "not_stated_in_download_page_or_website_terms; record as a candidate and confirm reuse permission before any model training or redistribution",
                "code_license": "GNU AGPLv3 according to the public GitLab project page",
                "clinical_use": "website terms state that the tool is not approved for clinical use",
            },
            "local_assets": {
                key: asset(path, root) for key, path in paths.items() if key.startswith("chempert_")
            },
            "metadata_counts": {
                "transcriptional_response_rows": total_responses,
                "unique_chemical_ids_with_response": int(responses["Chemical_ID"].nunique()),
                "source_accessions": int(responses["Data_Accession"].nunique()),
                "cell_sources": int(responses["Cell_Source"].nunique()),
                "species_response_rows": {
                    str(key): int(value) for key, value in responses["Species"].value_counts(dropna=False).items()
                },
                "initial_expression_metadata_rows": len(expression),
                "target_metadata_rows": len(perturbagens),
                "target_metadata_response_linked_rows": int(perturbagens["Response_ID"].notna().sum()),
            },
            "existing_resource_overlap": {
                "source_accession_response_counts": source_overlap,
                "interpretation": "The GSE92742 entries are a ChemPert-curated non-cancer subset derived from this project's existing LINCS Phase I source, not a new independent perturbation experiment. GSE70138 is not registered locally. No GSE139944 row was found, so the metadata show no direct sci-Plex accession overlap.",
            },
            "payload_validation": {
                "rds_gzip_headers_hex": rds_headers,
                "note": "Byte/header validation only. The RDS object schema was not materialized or used in a model by this audit.",
            },
            "biological_data_status": "Adds curated non-cancer perturbation response/target metadata and payloads; it adds no viability/AUC/IC50 drug-sensitivity labels and no patient/PDO identity-response pairs.",
            "potential_use": "candidate mechanism reference for non-cancer disease-expression reversal or context-conditioned perturbation analysis, after a source-level license and modality-specific contract review.",
            "not_permitted_by_this_audit": [
                "direct drug-sensitivity supervision",
                "cell-line or PDO efficacy external validation",
                "claiming cancer/PDO transferability from non-cancer perturbation signatures",
                "equating ChemPert response transcription factors with full baseline RNA-seq or post-treatment whole-transcriptome matrices",
            ],
        },
    }

    artifact_path = artifact_dir / "improve_chempert_audit.json"
    artifact_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    candidate_registry = {
        "registry_type": "candidate_resources_not_part_of_datasets_json",
        "created": ACQUISITION_DATE,
        "source_audit": artifact_path.relative_to(root).as_posix(),
        "candidates": [
            {
                "id": "improve_csa_zenodo_15258883_candidate",
                "status": "full_fixed_benchmark_acquired_pending_contract",
                "allowed_roles": ["fixed_cross_dataset_evaluation_infrastructure"],
                "forbidden_roles": ["pdo_validation", "clinical_validation", "unreviewed_training_source"],
                "audit_path": artifact_path.relative_to(root).as_posix(),
            },
            {
                "id": "chempert_public_download_202209_candidate",
                "status": "full_public_payloads_acquired_pending_data_license_and_modality_contract",
                "allowed_roles": ["non_cancer_perturbation_mechanism_reference"],
                "forbidden_roles": ["drug_sensitivity_label_source", "pdo_external_validation", "clinical_decision_support"],
                "audit_path": artifact_path.relative_to(root).as_posix(),
            },
        ],
    }
    registry_path = root / "registry/datasets_improve_chempert_candidates_20260916.json"
    registry_path.write_text(
        json.dumps(candidate_registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(artifact_path)
    print(registry_path)


if __name__ == "__main__":
    main()
