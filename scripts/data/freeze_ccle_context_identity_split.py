"""Freeze a response-blind LINCS ∩ CCLE identity table and patient-atomic split.

Does not read PRISM/GDSC/test metrics.  Does not rewrite the 978 matrix.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from drug_screen.data.ccle_context_identity import (
    FORMAT,
    SPLIT_SEED,
    assert_all_crc_exact_present,
    assert_no_patient_leakage,
    assign_roles,
    build_identity_rows,
    data_contract,
    summarize,
)
from drug_screen.data.lincs_landmarks import CRC_EXACT_CONTEXTS, ORDERED_GENE_IDS_SHA256


CONTEXT_REGISTRY = ROOT / "mvp/foundation/xpert/CONTEXT_REGISTRY.json"
MODEL = ROOT / "data/raw/depmap/24q2/Model.csv"
MAPPING = ROOT / "data/processed/depmap/24q2_rnaseq_exact978/ccle_24q2_exact978_mapping.json"
INTAKE = ROOT / "artifacts/experiments/EXP-007/CCLE_RNASEQ_INTAKE.json"
OUT = ROOT / "artifacts/experiments/EXP-007/CCLE_CONTEXT_IDENTITY_SPLIT.json"


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(
    *,
    registry_path: Path = CONTEXT_REGISTRY,
    model_path: Path = MODEL,
    mapping_path: Path = MAPPING,
    output_path: Path = OUT,
) -> dict[str, Any]:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    models = pd.read_csv(model_path, dtype=str).to_dict(orient="records")
    identity_rows = build_identity_rows(
        lincs_contexts=registry.get("contexts", []),
        models=models,
        expression_rows=mapping,
    )
    assignments = assign_roles(
        identity_rows=identity_rows,
        expression_rows=mapping,
        models=models,
        seed=SPLIT_SEED,
    )
    assert_no_patient_leakage(assignments)
    assert_all_crc_exact_present(assignments)
    summary = summarize(identity_rows, assignments)
    if summary["crc_exact_count"] != len(CRC_EXACT_CONTEXTS):
        raise RuntimeError(
            f"CRC exact overlap is {summary['crc_exact_count']}/10, not 10/10: "
            f"{summary['crc_exact_with_rnaseq']}"
        )
    intake = json.loads(INTAKE.read_text(encoding="utf-8")) if INTAKE.exists() else {}
    payload = {
        "format": FORMAT,
        "status": "IDENTITY_SPLIT_FROZEN",
        "data_status": "DATA_PARTIAL",
        "can_answer_question": {
            "identity_of_lincs_ccle_overlap": True,
            "crc_exact_10_in_expression": True,
            "response_blind_patient_split": True,
            "use_978_log2tpm_as_x_ctl": False,
            "compute_delta978_from_ccle": False,
        },
        "why_not_data_ready": (
            "Identity and split are frozen, but 978 log2(TPM+1) is basal RNA-seq only. "
            "It cannot replace LINCS matched control or serve as a trained context adapter."
        ),
        "sources": {
            "depmap_release": "DepMap Public 24Q2",
            "doi": "10.25452/figshare.plus.25880521.v1",
            "model_path": str(model_path.relative_to(ROOT).as_posix()),
            "model_sha256": _sha256(model_path),
            "mapping_path": str(mapping_path.relative_to(ROOT).as_posix()),
            "mapping_sha256": _sha256(mapping_path),
            "context_registry": str(registry_path.relative_to(ROOT).as_posix()),
            "intake_audit": str(INTAKE.relative_to(ROOT).as_posix()) if INTAKE.exists() else None,
            "intake_units": intake.get("adapter_output", {}).get("units"),
        },
        "gene_universe": {
            "ordered_gene_ids_sha256": ORDERED_GENE_IDS_SHA256,
            "units": "official log2(TPM+1); not LINCS X_ctl",
            "cannot_replace_matched_control": True,
            "cannot_use_as_x_ctl": True,
            "cannot_compute_delta978": True,
        },
        "split_contract": {
            "split_unit": "PatientID",
            "sample_unit": "depmap_id",
            "seed": SPLIT_SEED,
            "rule": (
                "CRC exact 10 lines and any other line sharing their PatientID are locked_eval. "
                "Remaining expression models are hashed by PatientID into train (~90%) / val (~10%). "
                "Roles are assigned before any PRISM/test metric is read."
            ),
            "forbidden_inputs": [
                "PRISM response values",
                "GDSC response values",
                "test-set ranking or lift",
                "Oracle Spearman / Top-K / NDCG",
                "predicted-to-oracle gap",
                "disease-reversal scores",
            ],
            "notes": [
                "SW480 and SW620 share PatientID PT-IPboWn and therefore stay together in locked_eval.",
                "H1299 remains unresolved: LINCS cell_id H1299 vs DepMap stripped name NCIH1299 is an alias, not an exact join.",
                "MCF10A and SKLU1 match Model.csv but have no 24Q2 RNA-seq row.",
            ],
        },
        "identity_rows": identity_rows,
        "split_assignments": assignments,
        "summary": summary,
        "data_contract": data_contract(summary=summary, leakage_status="PASS_PATIENT_ATOMIC"),
        "response_blind": True,
        "exp007_primary_metrics_untouched": True,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    payload = build(output_path=args.output)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "crc_exact": payload["summary"]["crc_exact_count"],
                "expression_overlap": payload["summary"]["expression_overlap"],
                "lines_by_role": payload["summary"]["lines_by_role"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
