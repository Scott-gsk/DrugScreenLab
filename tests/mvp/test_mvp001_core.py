from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_mvp001_core_assets_and_checksums_are_frozen():
    signature = ROOT / "mvp/core_data/crc_disease_signature_exact978.tsv"
    prism = ROOT / "mvp/core_data/compact_prism_response.parquet"
    signature_audit = json.loads(
        (ROOT / "mvp/core_data/crc_disease_signature_audit.json").read_text(encoding="utf-8")
    )
    prism_audit = json.loads(
        (ROOT / "mvp/core_data/prism_compact_audit.json").read_text(encoding="utf-8")
    )
    assert signature_audit["status"] == "DATA_READY"
    assert signature_audit["exact978_overlap"]["formal_gate_pass"] is True
    assert signature_audit["exact978_overlap"]["rows"] == 947
    assert _sha256(signature) == signature_audit["output"]["sha256"]
    assert prism_audit["status"] == "DATA_READY"
    assert prism_audit["response"]["finite_rows"] == 135
    assert _sha256(prism) == prism_audit["output"]["sha256"]


def test_mvp001_core_result_uses_one_cohort_and_frozen_direction():
    evidence = json.loads(
        (ROOT / "mvp/core_eval/MVP-001_core_eval_evidence.json").read_text(encoding="utf-8")
    )
    observed = json.loads(
        (ROOT / "mvp/core_eval/observed_oracle/MVP-001_observed_oracle_summary.json").read_text(
            encoding="utf-8"
        )
    )
    predicted = json.loads(
        (ROOT / "mvp/core_eval/predicted_reversal/MVP-001_predicted_reversal_summary.json").read_text(
            encoding="utf-8"
        )
    )
    prism = json.loads(
        (ROOT / "mvp/core_eval/prism_evaluation/MVP-001_prism_evaluation_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert evidence["status"] in {
        "RESULT_READY_FOR_REVIEW",
        "RESULT_REVIEWED_PENDING_ACCEPTANCE",
        "RESULT_REVIEWED",
    }
    assert evidence["candidate_identity"]["count"] == 4
    assert evidence["predicted_vs_observed"]["same_candidate_cohort"] is True
    assert observed["status"] == "READY_FOR_PRISM_JOIN"
    assert observed["prism"]["status"] == "READY_FOR_JOIN"
    assert predicted["predicted_vs_observed"]["same_candidate_cohort"] is True
    assert prism["status"] == "PROMISING"
    assert prism["decision_case"] == "CORE_MVP_FEASIBILITY_PROMISING"
    assert prism["prism_asset"]["response_direction"].startswith("lower official log2fc")
