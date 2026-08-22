"""Auditable EXP-008 ChEMBL36 UniProt accession to NCBI GeneID crosswalk."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sqlite3
import sys
import time
import anndata as ad
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "artifacts/experiments/EXP-008/identity_mapping_audit.json"
OUT = ROOT / "artifacts/experiments/EXP-008"
CHEMBL = Path("/mnt/d/Code/TranSiGen/data/screening/libraries/chembl_36/chembl_36_sqlite/chembl_36.db")
API = "https://rest.uniprot.org"
SOURCE = "UniProtKB_AC-ID"
TARGET = "GeneID"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def request_with_log(session: requests.Session, method: str, url: str, *, log: list[dict[str, Any]], **kwargs: Any) -> requests.Response:
    started = now()
    try:
        response = session.request(method, url, **kwargs)
        entry = {"attempted_at": started, "method": method, "url": url, "status_code": response.status_code,
                 "headers": {k: v for k, v in response.headers.items() if k.lower() in {"x-uniprot-release", "x-uniprot-release-date", "x-api-deployment-date", "content-type", "retry-after", "location"}},
                 "response_sha256": sha256_bytes(response.content)}
        log.append(entry)
        response.raise_for_status()
        return response
    except requests.RequestException as exc:
        if not log or log[-1].get("attempted_at") != started:
            log.append({"attempted_at": started, "method": method, "url": url, "error": repr(exc)})
        raise


def make_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(total=3, connect=3, read=3, status=3, backoff_factor=1,
                    status_forcelist=(429, 500, 502, 503, 504),
                    allowed_methods=frozenset({"GET", "POST"}), raise_on_status=False)
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers.update({"User-Agent": "EXP-008-uniprot-crosswalk/1.0"})
    return s


def get_accessions() -> tuple[list[str], dict[str, Any]]:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    rule = audit.get("target_filter", {}).get("rule", "")
    if rule != "target_dictionary.organism='Homo sapiens' AND target_type='SINGLE PROTEIN' AND drug_mechanism.direct_interaction=1":
        raise RuntimeError("identity_mapping_audit.json target filter does not match required contract")
    sdst_h5ad = Path("/mnt/d/Code/DrugScreenLab/data/external/xpert_source/processed_data/l1000_sdst_78453.h5ad")
    backed = ad.read_h5ad(sdst_h5ad, backed="r")
    try:
        official = backed.obs[["pert_id", "pert_idx"]].copy()
        official["pert_id"] = official["pert_id"].astype(str)
        official = official.drop_duplicates("pert_idx")
        official_pert = sorted(set(official["pert_id"]))
    finally:
        backed.file.close()
    sdst_csv = Path("/mnt/d/Code/DrugScreenLab/data/external/xpert_source/processed_data/l1000_sdst_drugs_info_8432.csv")
    pairs = []
    with sdst_csv.open(newline="", encoding="utf-8") as f:
        pairs = [(r["pert_id"], r["inchi_key"]) for r in csv.DictReader(f) if r["pert_id"] in set(official_pert) and r["inchi_key"]]
    con = sqlite3.connect(f"file:{CHEMBL}?mode=ro", uri=True)
    try:
        con.execute("CREATE TEMP TABLE sdst (pert_id TEXT, inchi_key TEXT)")
        con.executemany("INSERT INTO sdst VALUES (?,?)", pairs)
        rows = con.execute("""SELECT DISTINCT s.pert_id, m.chembl_id, td.tid, cs.accession
            FROM sdst s JOIN compound_structures c ON c.standard_inchi_key=s.inchi_key
            JOIN molecule_dictionary m ON m.molregno=c.molregno JOIN drug_mechanism dm ON dm.molregno=m.molregno
            JOIN target_dictionary td ON td.tid=dm.tid JOIN target_components tc ON tc.tid=td.tid
            JOIN component_sequences cs ON cs.component_id=tc.component_id
            WHERE td.organism='Homo sapiens' AND td.target_type='SINGLE PROTEIN'
              AND dm.direct_interaction=1 AND cs.accession IS NOT NULL
            ORDER BY s.pert_id, m.chembl_id, td.tid, cs.accession""").fetchall()
    finally:
        con.close()
    target_rows = [(str(r[0]), str(r[1]), int(r[2]), str(r[3]).strip()) for r in rows if str(r[3]).strip()]
    accessions = [r[3] for r in target_rows]
    if len(target_rows) != 866:
        raise RuntimeError(f"strict SDST target row count mismatch: expected 866, got {len(target_rows)}")
    return accessions, {"audit_path": str(AUDIT), "audit_sha256": sha256_file(AUDIT), "chembl_path": str(CHEMBL), "chembl_sha256": sha256_file(CHEMBL), "filter_rule": rule, "input_accession_rows": len(accessions), "input_unique_accessions": len(set(accessions)), "target_rows": target_rows, "official_sdst_pert_ids": len(official_pert)}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    raw_dir = OUT / "uniprot_api_raw"
    raw_dir.mkdir(exist_ok=True)
    accessions, input_meta = get_accessions()
    accessions = sorted(set(accessions))
    input_text = ("\n".join(accessions) + "\n").encode()
    input_sha = sha256_bytes(input_text)
    (raw_dir / "input_accessions.txt").write_bytes(input_text)
    session = make_session()
    calls: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    metadata_url = f"{API}/configure/idmapping/fields"
    metadata = request_with_log(session, "GET", metadata_url, log=calls, timeout=60)
    (raw_dir / "idmapping_fields.json").write_bytes(metadata.content)
    metadata_json = metadata.json()
    choices = {str(item.get("name")): item for group in metadata_json.get("groups", []) for item in group.get("items", [])}
    if SOURCE not in choices or not choices[SOURCE].get("from"):
        raise RuntimeError("official metadata does not expose required source UniProtKB_AC-ID")
    if TARGET not in choices or not choices[TARGET].get("to"):
        raise RuntimeError("official metadata does not expose required target GeneID")
    submit_at = now()
    try:
        submitted = request_with_log(session, "POST", f"{API}/idmapping/run", log=calls, data={"from": SOURCE, "to": TARGET, "ids": ",".join(accessions)}, timeout=120)
    except requests.RequestException as exc:
        errors.append({"stage": "submit", "at": submit_at, "error": repr(exc)})
        raise
    (raw_dir / "run_response.json").write_bytes(submitted.content)
    job_id = submitted.json().get("jobId")
    if not job_id:
        raise RuntimeError(f"UniProt submit response lacked jobId: {submitted.text[:500]}")
    status_history: list[dict[str, Any]] = []
    deadline = time.time() + 900
    while True:
        if time.time() > deadline:
            raise TimeoutError(f"timed out polling UniProt job {job_id}")
        status_resp = request_with_log(session, "GET", f"{API}/idmapping/status/{job_id}", log=calls, timeout=60)
        status_json = status_resp.json()
        status_history.append({"checked_at": now(), "payload": status_json, "response_sha256": sha256_bytes(status_resp.content)})
        if len(status_history) == 1 or status_json.get("jobStatus") not in ("NEW", "RUNNING"):
            (raw_dir / "status_final.json").write_bytes(status_resp.content)
        if "jobStatus" not in status_json:
            break
        if status_json["jobStatus"] not in ("NEW", "RUNNING"):
            raise RuntimeError(f"UniProt job ended with status {status_json['jobStatus']}")
        time.sleep(3)
    # The paginated results endpoint defaults to 25 rows; use the documented stream
    # endpoint so the downloaded raw file is the complete mapping result.
    results_url = f"{API}/idmapping/stream/{job_id}?format=tsv"
    results_resp = request_with_log(session, "GET", results_url, log=calls, timeout=180)
    result_path = raw_dir / "results.tsv"
    result_path.write_bytes(results_resp.content)
    raw_sha = sha256_bytes(results_resp.content)
    failed_url = f"{API}/idmapping/results/{job_id}?format=json&size=500"
    failed_resp = request_with_log(session, "GET", failed_url, log=calls, timeout=180)
    (raw_dir / "results.json").write_bytes(failed_resp.content)
    result_rows: list[tuple[str, str]] = []
    reader = csv.DictReader(results_resp.text.splitlines(), delimiter="\t")
    for row in reader:
        source = str(row.get("From", "")).strip()
        target = str(row.get("To", "")).strip()
        if source and target.isdigit():
            result_rows.append((source, target))
    failed_ids = failed_resp.json().get("failedIds", []) if failed_resp.text.strip() else []
    by_acc: dict[str, set[str]] = {a: set() for a in accessions}
    for source, target in result_rows:
        if source in by_acc:
            by_acc[source].add(target)
    with (OUT / "uniprot_entrez_crosswalk.tsv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["accession", "entrez_id", "mapping_status", "ambiguous"])
        for accession in accessions:
            ids = sorted(by_acc[accession])
            if not ids:
                w.writerow([accession, "", "unmapped", "false"])
            else:
                for entrez in ids:
                    w.writerow([accession, entrez, "mapped", str(len(ids) > 1).lower()])
    mapped = sum(bool(v) for v in by_acc.values())
    unique = sum(len(v) == 1 for v in by_acc.values())
    ambiguous = sum(len(v) > 1 for v in by_acc.values())
    unmapped = sum(not v for v in by_acc.values())
    # exact coverage over 866 audited accession rows; 978-axis coverage is computed from official gene axis.
    gene_info = Path("/mnt/d/Code/DrugScreenLab/data/external/xpert_source/processed_data/l1000_gene_info_978.csv")
    axis_ids: set[str] = set()
    with gene_info.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for key in ("gene_id", "entrez_id", "Gene ID", "entrez"):
                if row.get(key, "").strip().isdigit(): axis_ids.add(row[key].strip()); break
    axis_mapped = {e for ids in by_acc.values() for e in ids if e in axis_ids}
    manifest = {"format": "exp008_uniprot_entrez_crosswalk_v1", "exp_id": "EXP-008", "status": "PASS" if unique == len(accessions) and unmapped == 0 and axis_mapped else "REMAIN_BLOCKED", "endpoint": API, "source_database": {"name": SOURCE, "displayName": choices[SOURCE].get("displayName")}, "target_database": {"name": TARGET, "displayName": choices[TARGET].get("displayName")}, "jobId": job_id, "submitted_at": submit_at, "downloaded_at": now(), "api_release_headers": {k: v for k, v in metadata.headers.items() if k.lower().startswith("x-")}, "input_accession_sha256": input_sha, "input_accession_count": len(accessions), "raw_return_sha256": raw_sha, "raw_results_path": str(result_path), "failedIds": failed_ids, "status_history": status_history, "http_calls": calls, "errors": errors, "statistics": {"exact_accession_coverage": mapped / len(accessions), "mapped_accessions": mapped, "unique_accessions": unique, "ambiguous_accessions": ambiguous, "unmapped_accessions": unmapped, "result_mapping_rows": len(result_rows), "entrez_978_axis_coverage": len(axis_mapped) / len(axis_ids) if axis_ids else None, "entrez_978_axis_mapped": len(axis_mapped), "entrez_978_axis_size": len(axis_ids)}, "input_provenance": input_meta, "mapping_policy": "Only numeric UniProt From→To GeneID mappings accepted; one-to-many retained per row and marked ambiguous; no symbol/Ensembl/RefSeq inference."}
    (OUT / "uniprot_mapping_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = f"""# EXP-008 UniProt accession→Entrez 审计报告\n\n## 结论\n\n**{manifest['status']}**。仅使用 ChEMBL36 人类 `SINGLE PROTEIN` 且 `direct_interaction=1` 产生的 `{len(accessions)}` 个唯一 accession；只接受官方 UniProt ID Mapping 的数字 GeneID。\n\n## 官方服务与输入\n\n- Endpoint：`{API}`；source：`{SOURCE}`（{choices[SOURCE].get('displayName')}）；target：`{TARGET}`（{choices[TARGET].get('displayName')}）。\n- jobId：`{job_id}`；输入 accession SHA256：`{input_sha}`；原始 TSV 返回 SHA256：`{raw_sha}`。\n- UniProt release/deployment：`{metadata.headers.get('X-UniProt-Release')}` / `{metadata.headers.get('X-API-Deployment-Date')}`。\n- 原始响应目录：`{raw_dir}`。完整 HTTP 调用、状态轮询、响应哈希、failedIds 见 manifest。\n\n## 覆盖统计\n\n- accession 命中率：`{mapped}/{len(accessions)} = {mapped/len(accessions):.6f}`。\n- 唯一 Entrez：`{unique}`；一对多 accession：`{ambiguous}`；未命中：`{unmapped}`；failedIds：`{len(failed_ids)}`。\n- 978 Entrez 轴覆盖：`{len(axis_mapped)}/{len(axis_ids) if axis_ids else 978} = {(len(axis_mapped)/len(axis_ids) if axis_ids else 0):.6f}`。\n\n## 决策\n\n交叉表逐行保留一对多结果并标记 `ambiguous=true`，未命中保留为 `unmapped`。禁止 gene symbol、Ensembl、RefSeq 或其它间接映射。只有所有用于特征的 accession 均唯一 Entrez 且至少 30% SDST unique drugs 可获得非零特征时，才可解阻；本报告不替代特征重建，当前状态以 manifest 为准。\n"""
    (OUT / "uniprot_mapper_report.md").write_text(md, encoding="utf-8")
    print(json.dumps({"status": manifest["status"], **manifest["statistics"], "jobId": job_id}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
