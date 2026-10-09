from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "20260925_transigen_authoritative_kpgt_source_audit_v1"
PANEL_REL = Path("data/processed/dual_route/EXP-036/direct_response_contract_v1/drugs.parquet")
PANEL_SHA256 = "204fec3a13f7893c2def7d5890d79a39c64820418d53f658a1f55ce02fc27343"
OFFICIAL_REPO_URL = "https://github.com/myzhengSIMM/TranSiGen.git"
OFFICIAL_REPO_PAGE = "https://github.com/myzhengSIMM/TranSiGen"
OFFICIAL_RELEASE_PAGE = "https://github.com/myzhengSIMM/TranSiGen/releases/tag/v1.0"
ZENODO_API = "https://zenodo.org/api/records/11435859"
ZENODO_DOI = "https://doi.org/10.5281/zenodo.11435860"
PAPER_URL = "https://pmc.ncbi.nlm.nih.gov/articles/PMC11199551/"
OFFICIAL_FEATURE_REL = Path("data/PRISM/KPGT_emb2304.pickle")
OFFICIAL_FEATURE_SHA256 = "c3510bf97c136ff3a3e83118dd152d1923f059d32b3046bf61d4534c072fb86c"
OFFICIAL_FEATURE_SIZE = 15_172_765
RELEASE_TAG = "v1.0"
RELEASE_TAG_SHA1 = "8ec2218e2fe4fbb5f3a2c14271a8640a62c1af3a"
CURRENT_REMOTE_HEAD_SHA1 = "e7582e7794f13d4dfddfb15e95e3ba7cd24bc07a"
FEATURE_BLOB_SHA1 = "e4a3a23a748044eecca0166cfe898a36baa32494"
PREDICTION_BLOB_SHA1 = "4eb7b261f6c60efeb6085f8a3440701ad205e4e1"
FIXED_RTOL = 1e-5
FIXED_ATOL = 1e-6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_git(repo: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def parse_tree_entry(text: str) -> dict[str, str | int]:
    if not text:
        raise RuntimeError("expected a tracked source file, got an empty git tree entry")
    parts = text.split(None, 3)
    if len(parts) != 4 or parts[1] != "blob":
        raise RuntimeError(f"unexpected git tree entry: {text!r}")
    return {
        "mode": parts[0],
        "blob_sha1": parts[2],
        "size_bytes": int(parts[3].split("\t", 1)[0]),
    }


def source_archive_snapshot() -> dict[str, object]:
    try:
        with urlopen(ZENODO_API, timeout=20) as response:
            metadata = json.load(response)
        zenodo = {
            "status": "FETCHED",
            "requested_record": "11435859",
            "record_id": metadata.get("id"),
            "doi": metadata.get("metadata", {}).get("doi"),
            "version": metadata.get("metadata", {}).get("version"),
            "title": metadata.get("metadata", {}).get("title"),
            "files": [
                {
                    "key": item.get("key"),
                    "size_bytes": item.get("size"),
                    "checksum": item.get("checksum"),
                }
                for item in metadata.get("files", [])
            ],
        }
    except Exception as exc:  # network metadata is supplemental to the local tag audit
        zenodo = {
            "status": "UNAVAILABLE",
            "requested_record": "11435859",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    return zenodo


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit only the authoritative TranSiGen KPGT source and frozen 14-drug identity contract."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/mnt/d/Code/TranSiGen/参考代码/TranSiGen"),
        help="Linux path to the upstream TranSiGen checkout",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts/experiments/EXP-037" / RUN_ID,
    )
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing run directory: {output_dir}")
    if sys.platform != "linux" or os.environ.get("CONDA_DEFAULT_ENV") != "drugscreening-gpu":
        raise RuntimeError("run only in WSL2 Conda environment drugscreening-gpu")
    if "microsoft" not in platform.release().lower() and not os.environ.get("WSL_INTEROP"):
        raise RuntimeError("Linux runtime is not verified as WSL2")

    panel_path = ROOT / PANEL_REL
    official_feature_path = source_root / OFFICIAL_FEATURE_REL
    prediction_path = source_root / "src/prediction.py"
    if not panel_path.is_file() or not official_feature_path.is_file() or not prediction_path.is_file():
        raise FileNotFoundError("frozen panel or expected official source file is missing")

    official_remote = run_git(source_root, "remote", "get-url", "origin")
    if official_remote.rstrip("/").removesuffix(".git") != OFFICIAL_REPO_URL.removesuffix(".git"):
        raise RuntimeError(f"unexpected TranSiGen origin: {official_remote}")

    head = run_git(source_root, "rev-parse", "HEAD")
    tag_commit = run_git(source_root, "rev-parse", f"refs/tags/{RELEASE_TAG}^{{}}")
    feature_head_tree = parse_tree_entry(
        run_git(source_root, "ls-tree", "--long", "HEAD", "--", str(OFFICIAL_FEATURE_REL))
    )
    feature_tag_tree = parse_tree_entry(
        run_git(source_root, "ls-tree", "--long", RELEASE_TAG, "--", str(OFFICIAL_FEATURE_REL))
    )
    prediction_tag_tree = parse_tree_entry(
        run_git(source_root, "ls-tree", "--long", RELEASE_TAG, "--", "src/prediction.py")
    )
    relevant_worktree_status = {
        "feature_cache": run_git(source_root, "status", "--porcelain", "--", str(OFFICIAL_FEATURE_REL)),
        "prediction_source": run_git(source_root, "status", "--porcelain", "--", "src/prediction.py"),
    }
    remote_refs_text = subprocess.run(
        ["git", "ls-remote", OFFICIAL_REPO_URL, "HEAD", f"refs/tags/{RELEASE_TAG}"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()
    remote_refs = {}
    for line in remote_refs_text.splitlines():
        sha, ref = line.split("\t", 1)
        remote_refs[ref] = sha

    feature_sha256 = sha256(official_feature_path)
    panel_sha256 = sha256(panel_path)
    prediction_text = prediction_path.read_text(encoding="utf-8")
    direct_smiles_lookup = "smi2emb[smi]" in prediction_text
    prediction_uses_feature_path = "molecule_feature_embed_path" in prediction_text
    if panel_sha256 != PANEL_SHA256:
        raise RuntimeError("frozen 14-drug panel checksum changed")
    if feature_sha256 != OFFICIAL_FEATURE_SHA256 or official_feature_path.stat().st_size != OFFICIAL_FEATURE_SIZE:
        raise RuntimeError("official PRISM feature cache checksum/size differs from the prior restricted audit")
    if feature_head_tree["blob_sha1"] != FEATURE_BLOB_SHA1 or feature_tag_tree != feature_head_tree:
        raise RuntimeError("official PRISM feature cache is not the expected unchanged upstream tag object")
    if prediction_tag_tree["blob_sha1"] != PREDICTION_BLOB_SHA1:
        raise RuntimeError("official prediction adapter differs from the expected v1.0 source object")
    if relevant_worktree_status["feature_cache"] or relevant_worktree_status["prediction_source"]:
        raise RuntimeError("the relevant files are modified in the local official checkout")
    if tag_commit != RELEASE_TAG_SHA1 or remote_refs.get(f"refs/tags/{RELEASE_TAG}") != RELEASE_TAG_SHA1:
        raise RuntimeError("upstream v1.0 tag cannot be bound to the expected commit")
    if remote_refs.get("HEAD") != CURRENT_REMOTE_HEAD_SHA1 or head != CURRENT_REMOTE_HEAD_SHA1:
        raise RuntimeError("upstream HEAD differs from the verified checkout")
    if not direct_smiles_lookup or not prediction_uses_feature_path:
        raise RuntimeError("official v1.0 prediction source no longer confirms canonical-SMILES feature lookup")

    official_crosswalk_path = ROOT / "artifacts/experiments/EXP-037/20260925_transigen_input_chain_audit/official_prism_crosswalk.json"
    local_crosswalk_path = ROOT / "artifacts/experiments/EXP-037/20260925_transigen_input_chain_audit/local_lincs_restricted_crosswalk.json"
    reproduction_path = ROOT / "artifacts/experiments/EXP-037/20260925_kpgt_local_reproduction_probe_v1/attempt_04/receipt.json"
    for receipt_path in (official_crosswalk_path, local_crosswalk_path, reproduction_path):
        if not receipt_path.is_file():
            raise FileNotFoundError(f"required prior receipt is missing: {receipt_path}")
    official_crosswalk = json.loads(official_crosswalk_path.read_text(encoding="utf-8"))
    local_crosswalk = json.loads(local_crosswalk_path.read_text(encoding="utf-8"))
    reproduction = json.loads(reproduction_path.read_text(encoding="utf-8"))

    if official_crosswalk.get("panel_sha256") != PANEL_SHA256 or official_crosswalk.get("feature_sha256") != feature_sha256:
        raise RuntimeError("prior official feature crosswalk is not bound to these exact bytes")
    if (official_crosswalk.get("panel_rows"), official_crosswalk.get("panel_unambiguous"),
            official_crosswalk.get("panel_ambiguous"), official_crosswalk.get("panel_missing")) != (14, 0, 0, 14):
        raise RuntimeError("prior official PRISM crosswalk no longer reports exact 0/14 panel coverage")
    if local_crosswalk.get("panel_sha256") != PANEL_SHA256:
        raise RuntimeError("prior local LINCS crosswalk is not bound to this exact panel")
    if (local_crosswalk.get("panel_unambiguous_count"), local_crosswalk.get("panel_ambiguous_count"),
            local_crosswalk.get("panel_missing_count")) != (13, 0, 1):
        raise RuntimeError("prior local LINCS crosswalk no longer reports 13 usable identities and one missing drug")
    comparisons = reproduction.get("fixed_threshold_comparisons", [])
    if len(comparisons) != 13 or any(item.get("allclose") for item in comparisons):
        raise RuntimeError("the frozen-threshold local cache check is not the expected 0/13 result")
    if any((item.get("rtol"), item.get("atol")) != (FIXED_RTOL, FIXED_ATOL) for item in comparisons):
        raise RuntimeError("prior cache comparison threshold differs from the frozen threshold")
    if reproduction.get("gpu_training_hours") != 0.0 or not reproduction.get("cpu_only"):
        raise RuntimeError("prior KPGT regeneration cost receipt does not establish CPU-only, zero-GPU execution")

    run_seconds = [float(item["wall_seconds"]) for item in reproduction.get("generator_runs", [])]
    if len(run_seconds) != 2:
        raise RuntimeError("expected exactly two prior isolated CPU feature-generation runs")

    output_dir.mkdir(parents=True, exist_ok=False)
    receipt: dict[str, object] = {
        "experiment_id": "EXP-037",
        "run_id": RUN_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "AUTHORITATIVE_SOURCE_VERIFIED_PANEL_INPUT_BLOCKED",
        "scope": "frozen EXP-036 14-drug panel and its 13 structurally assessable local KPGT candidates only",
        "runtime": {
            "platform": platform.platform(),
            "python": sys.executable,
            "python_version": sys.version,
            "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
            "gpu_used_by_this_audit": False,
            "formal_training_or_gpu_hours": 0.0,
        },
        "official_sources": {
            "repository": OFFICIAL_REPO_PAGE,
            "repository_remote": official_remote,
            "release": OFFICIAL_RELEASE_PAGE,
            "tag": RELEASE_TAG,
            "tag_commit_sha1": tag_commit,
            "current_head_sha1": head,
            "remote_refs_checked_read_only": remote_refs,
            "zenodo_version_record": ZENODO_DOI,
            "zenodo_api_snapshot": source_archive_snapshot(),
            "paper": PAPER_URL,
            "paper_reported_kpgt_dimension": 2034,
            "repository_code_and_cache_dimension": 2304,
            "paper_vs_release_dimension_note": "The paper text reports 2034 dimensions while the released inference path and tracked feature cache use 2304. This source-level discrepancy is recorded without changing the EXP-037 2304 input contract.",
        },
        "official_feature_asset": {
            "relative_path": str(OFFICIAL_FEATURE_REL),
            "tracked_at_head_and_v1_0": feature_head_tree["blob_sha1"] == feature_tag_tree["blob_sha1"],
            "git_blob_sha1": feature_tag_tree["blob_sha1"],
            "size_bytes": official_feature_path.stat().st_size,
            "sha256": feature_sha256,
            "worktree_status_for_asset": relevant_worktree_status["feature_cache"],
            "worktree_status_for_prediction_source": relevant_worktree_status["prediction_source"],
            "prediction_source_git_blob_sha1": prediction_tag_tree["blob_sha1"],
            "prediction_source_confirms_feature_path_argument": prediction_uses_feature_path,
            "prediction_source_confirms_smi2emb_key_lookup": direct_smiles_lookup,
            "official_cache_rows_and_target_identity_crosswalk_reused_from": str(official_crosswalk_path.relative_to(ROOT)),
            "official_cache_vector_dimension": 2304,
            "official_cache_exact_inchikey_matches": 0,
            "frozen_panel_rows": 14,
            "frozen_panel_rows_missing_from_official_prism_cache": 14,
            "feature_values_reloaded_in_this_run": False,
        },
        "local_candidate_cache_check_reused": {
            "source_receipt": str(local_crosswalk_path.relative_to(ROOT)),
            "local_cache_sha256": local_crosswalk.get("feature_sha256"),
            "unambiguous_target_identities": 13,
            "missing_target_identities": 1,
            "unambiguous_drug_index_values": [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13],
            "missing_drug_index_values": [6],
            "regenerated_vs_existing_local_cache_allclose": False,
            "matching_vectors_at_fixed_threshold": 0,
            "compared_vectors_at_fixed_threshold": 13,
            "rtol": FIXED_RTOL,
            "atol": FIXED_ATOL,
            "source_receipt": str(reproduction_path.relative_to(ROOT)),
            "reproduction_receipt_sha256": sha256(reproduction_path),
            "comparison_details_recomputed": False,
        },
        "measured_input_preparation_cost": {
            "reused_receipt": str(reproduction_path.relative_to(ROOT)),
            "cpu_generation_runs_seconds": run_seconds,
            "cpu_generation_total_wall_seconds": float(reproduction.get("wall_seconds_total")),
            "cpu_only": True,
            "gpu_training_hours": 0.0,
            "applies_to": "one isolated 14-drug KPGT feature reconstruction diagnostic, not a successful authoritative input adapter or model forward",
            "model_forward_cost_measured": False,
        },
        "pre_registered_input_contract_conclusion": {
            "required": "Version-bound 2304-dimensional KPGT features, full drug structure identity, row/key mapping, feature-source or generation version and checksum; EXP-037 input axes must not be inferred from an unverified cache order.",
            "observed": "The upstream v1.0 TranSiGen PRISM feature dictionary is source-bound and checksum-verified, but its 1,625 feature keys contain no exact full-InChIKey matches to the frozen 14-drug panel. The local LINCS cache has 13 identity candidates and index 6 missing, while its regenerated candidates fail the fixed rtol=1e-5/atol=1e-6 agreement test for all 13.",
            "decision": "No panel-complete, version-bound authoritative feature input is available. Do not use either cache as a validated TranSiGen input, do not drop index 6 or shrink the frozen panel, and do not infer feature rows from local cache order.",
            "scope_limit": "This is a 14-drug input-availability audit; it does not establish KPGT availability across the separate 20,336-drug EXP-037 B1 axis.",
        },
        "scientific_result": False,
        "forward_or_training_executed": False,
        "blocked_action": "A fair TranSiGen input smoke test and any subsequent model comparison are blocked before forward because the frozen panel lacks a complete publisher-bound feature/key map and the 13 local candidates fail the prespecified numeric parity threshold.",
        "minimum_external_evidence": "Provide a publisher or asset-maintainer versioned 14-row KPGT sidecar with release/source commit, generating checkpoint and SHA-256, explicit canonical isomeric SMILES plus drug_index/drug_id/full InChIKey row keys, 2,304 finite feature values and per-row hashes; clarify the paper's 2,034 versus released 2,304 dimensionality. Then rerun CPU-only axis/key/hash/numeric checks before any forward call.",
        "next_action": "Request the above exact sidecar and provenance from the TranSiGen/KPGT asset maintainer. After receipt, run the sidecar contract check in WSL2 Conda drugscreening-gpu, then stop for a fresh real-input cost gate before forward.",
        "prior_receipts_bound": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (panel_path, official_crosswalk_path, local_crosswalk_path, reproduction_path)
        },
    }
    script_path = Path(__file__).resolve()
    receipt["script_path"] = str(script_path.relative_to(ROOT))
    receipt["script_sha256"] = sha256(script_path)
    receipt_path = output_dir / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": receipt["status"],
        "official_tag": tag_commit,
        "official_cache_sha256": feature_sha256,
        "official_panel_match": "0/14",
        "local_cache_parity": "0/13",
        "cpu_generation_seconds": round(sum(run_seconds), 3),
        "gpu_hours": 0.0,
        "scientific_result": False,
        "receipt": str(receipt_path),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
