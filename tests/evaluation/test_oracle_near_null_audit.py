from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "artifacts" / "experiments" / "EXP-007" / "ORACLE_NEAR_NULL_SOURCE_AUDIT.json"
RESULT = ROOT / "artifacts" / "experiments" / "EXP-007" / "FULL_OBSERVED_ORACLE_RESULT.json"
RECORD = ROOT / "experiments" / "records" / "EXP-007.md"


def test_fail_rule_audit_does_not_change_primary_metrics() -> None:
    if not AUDIT.exists() or not RESULT.exists():
        return
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    assert audit["primary_metrics_unchanged"] is True
    assert audit["result_status"] == "ORACLE_NEAR_NULL"
    assert result["status"] == "ORACLE_NEAR_NULL"
    assert audit["decision"]["do_not_stack_models"] is True
    assert "disease_signature" in audit["decision"]["primary_suspects"]
    assert "prism_endpoint" in audit["decision"]["primary_suspects"]
    assert audit["signature"]["rows"] == 947
    assert audit["signature"]["missing_landmark_count"] == 31
    assert audit["dose_time"]["dose_time_not_mixed"] is True
    assert audit["dose_time"]["lines_with_10uM_24h"] == ["HT29"]
    assert audit["prism_endpoint"]["not_matched_to_10uM_6h"] is True
    assert "use_of_839_pair_xpert_h5ad_oracle" in audit["decision"]["ruled_out"]


def test_exp007_record_hypothesis_untouched() -> None:
    text = RECORD.read_text(encoding="utf-8")
    assert "Observed Δ978 → Disease Reversal" in text
    assert "Top-10/20/50 overlap" in text
    assert "ORACLE_NEAR_NULL" in text
