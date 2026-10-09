#!/usr/bin/env python3
"""Independent readback for the fixed GSE320583 single-agent descriptive case."""
from __future__ import annotations
import csv, hashlib, json, os, sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "artifacts/experiments/EXP-050/20261006_gse320583_single_drug_case_evidence_v1"
OLD = ROOT / "artifacts/experiments/EXP-050/20261001_gse320583_birinapant_transfer_v3"
IDENTITY = ROOT / "artifacts/experiments/EXP-050/20261006_pdo_public_evidence_completion_v1/GSE320583_donor_model_crosswalk.tsv"
LEDGER = ROOT / "artifacts/experiments/EXP-050/gpu_ledger.jsonl"

def read_tsv(p):
    with p.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))

def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

cross = read_tsv(IDENTITY)
case = read_tsv(RUN / "gse320583_single_agent_three_donor_case_evidence.tsv")
hall = read_tsv(RUN / "gse320583_single_agent_hallmark_directions_with_donor_identity.tsv")
old_gene = [r for r in read_tsv(OLD / "gse320583_birinapant_cross_system_gene_agreement.tsv") if r["target_dataset"] == "GSE320583" and r["drug_label"].casefold() == "birinapant"]
old_hall = read_tsv(OLD / "gse320583_hallmark_cross_system_directions.tsv")
old_cons = [r for r in read_tsv(OLD / "gse320583_birinapant_hallmark_model_consistency.tsv") if r["condition_code"] == "Bi"]
source = json.loads((RUN / "single_agent_source_evidence.json").read_text(encoding="utf-8-sig"))
assert {r["article_donor_index"]: r["GEO_model_id"] for r in cross} == {"1":"HUB-C2-120","2":"HUB-C2-152","3":"HUB-C2-123"}
assert [(r["article_donor_index"], r["GEO_model_id"]) for r in case] == [("1","HUB-C2-120"),("2","HUB-C2-152"),("3","HUB-C2-123")]
assert len(case) == 3 and len(hall) == 150
assert all(sum(x["GEO_model_id"] == model for x in hall) == 50 for model in ("HUB-C2-120","HUB-C2-152","HUB-C2-123"))
assert len({(x["GEO_model_id"], x["hallmark_set"]) for x in hall}) == 150
assert all(x["prior_common_gene_count"] == "961" for x in case)
assert all(x["prior_hallmark_zero_or_unavailable_n_of_50"] == "0" for x in case)
by_model = {r["model_id"]: r for r in old_gene}
assert len(by_model) == 3
for row in case:
    src = by_model[row["GEO_model_id"]]
    for out, old in [("prior_gene_spearman_rho_vs_MCF7","gene_spearman_rho"),("prior_gene_pearson_r_vs_MCF7","gene_pearson_r"),("prior_nonzero_gene_sign_agreement","nonzero_gene_sign_agreement"),("prior_nonzero_gene_sign_denominator","nonzero_gene_sign_denominator"),("prior_common_gene_count","n_common_landmark_genes")]:
        assert row[out] == src[old], (row["GEO_model_id"], out, row[out], src[old])
old_lookup = {(r["model_id"], r["hallmark_set"]): r for r in old_hall}
assert len(old_lookup) == 150
for row in hall:
    src = old_lookup[(row["GEO_model_id"], row["hallmark_set"])]
    for field in ("direction_relation", "target_direction", "source_direction", "within_set_gene_spearman_rho", "gmt_gene_count", "source_common_gene_count", "source_gene_coverage_of_gmt"):
        assert row[field] == src[field], (row["GEO_model_id"], row["hallmark_set"], field)
counts = {}
for model in ("HUB-C2-120","HUB-C2-152","HUB-C2-123"):
    subset = [r for r in hall if r["GEO_model_id"] == model]
    counts[model] = {"same": sum(r["direction_relation"] == "same_sign" for r in subset), "opposite": sum(r["direction_relation"] == "opposite_sign" for r in subset), "zero_unavailable": sum(r["direction_relation"] in {"zero","unavailable","zero_or_unavailable"} for r in subset)}
    assert sum(counts[model].values()) == 50
assert counts == {"HUB-C2-120":{"same":29,"opposite":21,"zero_unavailable":0},"HUB-C2-152":{"same":30,"opposite":20,"zero_unavailable":0},"HUB-C2-123":{"same":23,"opposite":27,"zero_unavailable":0}}
assert len(old_cons) == 50 and len({r["hallmark_set"] for r in old_cons}) == 50
same_sets = sum(r["cross_model_direction_consistent"].casefold() == "true" for r in old_cons)
mixed_sets = sum(r["cross_model_direction_consistent"].casefold() == "false" for r in old_cons)
assert same_sets == 16 and mixed_sets == 34 and same_sets + mixed_sets == 50
assert source["analysis_boundary"]["numeric_drug_sensitivity_values_collected"] is False
assert source["analysis_boundary"]["figure_image_values_digitized"] is False
assert source["analysis_boundary"]["drug_response_ranking_or_correlation_calculated"] is False
assert source["analysis_boundary"]["full_inchi_key_gate_closed"] is False
manifest = json.loads((RUN / "run_manifest.json").read_text(encoding="utf-8-sig"))
assert manifest["gpu_hours"] == 0 and manifest["all_50_hallmark_sets_retained_per_model"] is True
analysis_script = ROOT / manifest["script_path"]
assert sha(analysis_script) == manifest["script_sha256"]
for entry in manifest["inputs"].values():
    assert sha(ROOT / entry["path"]) == entry["sha256"]
assert sha(RUN / "protocol_snapshot.json") == manifest["protocol_snapshot_sha256"]
assert sha(RUN / "start_receipt.json") == manifest["start_receipt_sha256"]
for name, digest in manifest["outputs_sha256"].items():
    assert sha(RUN / name) == digest, name
start_receipt = json.loads((RUN / "start_receipt.json").read_text(encoding="utf-8-sig"))
started = datetime.fromisoformat(start_receipt["started_at_local"])
verified = datetime.now(ZoneInfo("Asia/Shanghai"))
package_minutes = (verified - started).total_seconds() / 60
assert 0 <= package_minutes <= 60, package_minutes
ledger = [json.loads(line) for line in LEDGER.read_text(encoding="utf-8").splitlines() if line.strip()]
starts = [r for r in ledger if r.get("event") == "attempt_started"]
finishes = [r for r in ledger if r.get("event") == "attempt_finished"]
start_ids = {r.get("attempt_id") for r in starts}
finish_ids = {r.get("attempt_id") for r in finishes}
settled = sum(float(r["gpu_hours_charged"]) for r in finishes)
assert len(ledger) == 28 and len(starts) == 14 and len(finishes) == 14 and start_ids == finish_ids
assert abs(settled - 0.584830054434155) < 1e-12
assert all(r.get("run_id") != "20261006_gse320583_single_drug_case_evidence_v1" for r in ledger)
checked = ["protocol_snapshot.json","start_receipt.json","single_agent_source_evidence.json","execution_mishaps.tsv","gse320583_single_agent_three_donor_case_evidence.tsv","gse320583_single_agent_hallmark_directions_with_donor_identity.tsv","summary.json","run_manifest.json"]
for name in checked:
    assert (RUN / name).is_file(), name
verify_script = Path(__file__).resolve()
receipt = {
  "run_id": "20261006_gse320583_single_drug_case_evidence_v1",
  "status": "ROOT_INDEPENDENT_READBACK_PASS",
  "verified_at_local": verified.isoformat(timespec="seconds"),
  "package_wall_minutes_from_start_receipt": package_minutes,
  "verification_script": "scripts/evaluation/verify_exp050_gse320583_single_drug_case.py",
  "verification_script_sha256": sha(verify_script),
  "checks": {
    "article_donor_model_crosswalk_exact": True,
    "case_rows": len(case),
    "full_hallmark_rows": len(hall),
    "exact_old_gene_metrics_preserved": True,
    "all_150_hallmark_rows_exactly_copied_from_frozen_output": True,
    "hallmark_counts_by_model": counts,
    "preexisting_cross_model_hallmark_summary_same_mixed": {"same": same_sets,"mixed": mixed_sets},
    "all_input_output_and_protocol_hashes_match_run_manifest": True,
    "numeric_drug_sensitivity_or_response_association_used": False,
    "strict_full_inchi_key_gate_closed": False,
    "gpu_hours": 0
  },
  "gpu_ledger_readback": {"rows":len(ledger),"starts":len(starts),"finishes":len(finishes),"open_attempts":0,"settled_gpu_hours":settled,"run_id_charge":0.0},
  "sha256_checked_inputs_and_outputs": {name: sha(RUN / name) for name in checked},
  "limitations": "Root readback verifies joins, counts, copied expression/Hallmark quantities, declared boundaries, run-manifest hashes, and shared GPU ledger. It does not independently authenticate article text beyond the cited source field record, nor estimate drug efficacy or a drug-effect association."
}
receipt_path = RUN / "root_independent_readback_receipt.json"
receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
package_files = [RUN/name for name in checked] + [receipt_path, ROOT/"scripts/evaluation/evaluate_exp050_gse320583_single_drug_case.py", verify_script]
manifest_path = RUN / "sha256_manifest.tsv"
tmp = manifest_path.with_name(manifest_path.name+".tmp")
with tmp.open("w",encoding="utf-8",newline="") as f:
    w=csv.writer(f,delimiter="\t",lineterminator="\n"); w.writerow(["relative_path","sha256","bytes"])
    for p in package_files: w.writerow([str(p.relative_to(ROOT)),sha(p),p.stat().st_size])
os.replace(tmp,manifest_path)
manifest_rows=read_tsv(manifest_path)
assert len(manifest_rows)==len(package_files)
assert all(sha(ROOT/r["relative_path"])==r["sha256"] and (ROOT/r["relative_path"]).stat().st_size==int(r["bytes"]) for r in manifest_rows)
print(json.dumps({"status":receipt["status"],"case_rows":len(case),"hallmark_rows":len(hall),"counts":counts,"all_same_sets":same_sets,"mixed_sets":mixed_sets,"package_wall_minutes":round(package_minutes,3),"gpu_ledger":receipt["gpu_ledger_readback"],"sha_manifest_rows":len(manifest_rows)},ensure_ascii=False))
