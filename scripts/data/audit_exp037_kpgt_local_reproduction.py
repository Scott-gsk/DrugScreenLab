#!/usr/bin/env python3
"""Run an isolated CPU-only 14-drug KPGT reconstruction probe for EXP-037.

This diagnostic consumes only the frozen drug identity projection and the
approved local LINCS KPGT cache. All generated files are confined to its
unique artifact directory. It does not read labels, expression, predictions,
or TranSiGen response outputs.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import pickle
import pickletools
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = ROOT / "artifacts/experiments/EXP-037/20260925_kpgt_local_reproduction_probe_v1"
PANEL = ROOT / "data/processed/dual_route/EXP-036/direct_response_contract_v1/drugs.parquet"
REGISTRY = ROOT / "data/registry/datasets.json"
CONTRACT = PANEL.parent
EXT = Path("/mnt/d/Code/TranSiGen")
KPGT = EXT / "KPGT"
OFFICIAL = EXT / "参考代码/TranSiGen"
OLD_LINCS = EXT / "data/lincs/KPGT_emb2304.pickle"
EXPECTED_PANEL_SHA = "204fec3a13f7893c2def7d5890d79a39c64820418d53f658a1f55ce02fc27343"
EXPECTED_MANIFEST_SHA = "ee0dfe17cb4963232230eff73b6b6e08be9cd2b9cc64b46123c2be98e6c1eeba"
EXPECTED_OLD_LINCS_SHA = "61b2a1337c8313ecbcb95e9f797ea12caeb5d94fefc2697af810c5955ae962f4"
EXPECTED_WEIGHT_SHA = "8811c117cfedddff5ef2a512290d1a7a73d3d9560773860bc3b06942c597d15d"
WEIGHT = KPGT / "pretrained/pretrained/base/base.pth"
KPGT_DIM = 2304
RTOL = 1e-5
ATOL = 1e-6
ALLOWED_GLOBALS = {
    ("numpy", "dtype"),
    ("numpy", "ndarray"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy.core.numeric", "_frombuffer"),
    ("numpy._core.numeric", "_frombuffer"),
}
UNKNOWN = object()
MARKER = object()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git(repo: Path, *args: str, check: bool = True) -> dict[str, Any]:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    result = {"returncode": proc.returncode, "output": proc.stdout.strip()}
    if check and proc.returncode:
        raise RuntimeError(f"git -C {repo} {' '.join(args)} failed: {proc.stdout.strip()}")
    return result


DIFF_PATHS = {
    EXT: [
        "src/prepare_kpgt_embeddings.py",
        "scripts/kpgt_runner.py",
        "scripts/kpgt_patch/sitecustomize.py",
    ],
    KPGT: [
        "scripts/preprocess_downstream_dataset.py",
        "scripts/extract_features.py",
        "src/model_config.py",
        "src/data/descriptors/DescriptorGenerator.py",
        "src/data/descriptors/dists.py",
    ],
    OFFICIAL: ["src/model.py", "src/utils.py", "src/drug_response_prediction.py"],
}


def repo_snapshot(repo: Path) -> dict[str, Any]:
    head = git(repo, "rev-parse", "HEAD")["output"]
    status = git(repo, "status", "--short", "--untracked-files=all")["output"]
    paths = DIFF_PATHS.get(repo, [])
    diff = git(repo, "diff", "--no-ext-diff", "--binary", "HEAD", "--", *paths)["output"].encode("utf-8", errors="surrogateescape")
    submodules = git(repo, "submodule", "status", "--", "KPGT", check=False) if repo == EXT else None
    return {
        "path": str(repo),
        "head": head,
        "status_short_untracked_all_sha256": sha256_bytes(status.encode("utf-8", errors="surrogateescape")),
        "status_entry_count": len(status.splitlines()),
        "status_bytes": len(status.encode("utf-8", errors="surrogateescape")),
        "dirty_diff_paths": paths,
        "tracked_dirty_diff_sha256": sha256_bytes(diff),
        "submodule_status_kpgt": submodules,
    }


def sha_or_missing(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {"path": str(path), "exists": False}
    return {"path": str(path), "exists": True, "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def package_versions() -> dict[str, str | None]:
    names = ["numpy", "pandas", "pyarrow", "rdkit", "torch", "dgl", "dgllife", "scipy"]
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    versions["rdkit_runtime"] = rdBase.rdkitVersion
    return versions


def nvidia_smi() -> dict[str, Any]:
    try:
        proc = subprocess.run(
            ["nvidia-smi"], text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=False, timeout=20,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""},
        )
        return {"returncode": proc.returncode, "output": proc.stdout.strip()}
    except Exception as exc:  # snapshot only; visibility is independently disabled
        return {"error": f"{type(exc).__name__}: {exc}"}


def scan_pickle_globals(path: Path) -> dict[str, Any]:
    globals_seen: set[tuple[str, str]] = set()
    forbidden: set[str] = set()
    stack: list[Any] = []
    memo: dict[int, Any] = {}
    opcode_count = 0
    stack_global_count = 0
    next_memo_index = 0

    def pop_items_to_mark(*, keep_container: bool = False, push_unknown: bool = False) -> None:
        try:
            mark_index = len(stack) - 1 - stack[::-1].index(MARKER)
        except ValueError as exc:
            raise RuntimeError(f"pickle opcode stack parse failed: missing MARK in {path}") from exc
        del stack[mark_index:]
        if keep_container:
            if mark_index == 0:
                raise RuntimeError(f"pickle opcode stack parse failed: missing container in {path}")
        elif push_unknown:
            stack.append(UNKNOWN)

    with path.open("rb") as stream:
        for opcode, arg, _ in pickletools.genops(stream):
            opcode_count += 1
            name = opcode.name
            if name in {"EXT1", "EXT2", "EXT4", "PERSID", "BINPERSID", "NEXT_BUFFER"}:
                forbidden.add(name)
            if name == "GLOBAL":
                globals_seen.add(tuple(str(arg).split(" ", 1)))
                stack.append(("global", *tuple(str(arg).split(" ", 1))))
            elif name == "STACK_GLOBAL":
                if len(stack) < 2:
                    raise RuntimeError(f"pickle opcode stack parse failed at STACK_GLOBAL in {path}")
                global_name = stack.pop()
                global_module = stack.pop()
                if not (isinstance(global_module, str) and isinstance(global_name, str)):
                    raise RuntimeError(f"cannot prove STACK_GLOBAL operands are literal strings in {path}")
                globals_seen.add((global_module, global_name))
                stack_global_count += 1
                stack.append(("global", global_module, global_name))
            elif name in {"PROTO", "FRAME"}:
                pass
            elif name in {"EMPTY_DICT", "EMPTY_LIST", "EMPTY_TUPLE", "EMPTY_SET", "NONE", "NEWTRUE", "NEWFALSE", "NEWOBJ"}:
                if name == "NEWOBJ":
                    if len(stack) < 2:
                        raise RuntimeError(f"pickle opcode stack parse failed at NEWOBJ in {path}")
                    stack.pop(); stack.pop()
                stack.append(UNKNOWN)
            elif name in {"SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "UNICODE"}:
                stack.append(arg)
            elif name in {"MARK"}:
                stack.append(MARKER)
            elif name in {"MEMOIZE"}:
                if not stack:
                    raise RuntimeError(f"pickle opcode stack parse failed at MEMOIZE in {path}")
                memo[next_memo_index] = stack[-1]
                next_memo_index += 1
            elif name in {"BINPUT", "LONG_BINPUT", "PUT"}:
                if not stack:
                    raise RuntimeError(f"pickle opcode stack parse failed at {name} in {path}")
                memo[int(arg)] = stack[-1]
            elif name in {"BINGET", "LONG_BINGET", "GET"}:
                if int(arg) not in memo:
                    raise RuntimeError(f"pickle opcode stack parse failed at missing memo {arg} in {path}")
                stack.append(memo[int(arg)])
            elif name in {"POP"}:
                if not stack:
                    raise RuntimeError(f"pickle opcode stack underflow at POP in {path}")
                stack.pop()
            elif name == "DUP":
                if not stack:
                    raise RuntimeError(f"pickle opcode stack underflow at DUP in {path}")
                stack.append(stack[-1])
            elif name == "POP_MARK":
                pop_items_to_mark()
            elif name in {"TUPLE", "LIST", "DICT", "FROZENSET", "SET"}:
                pop_items_to_mark(push_unknown=True)
            elif name in {"APPENDS", "SETITEMS", "ADDITEMS"}:
                pop_items_to_mark(keep_container=True)
            elif name in {"TUPLE1", "TUPLE2", "TUPLE3"}:
                number = {"TUPLE1": 1, "TUPLE2": 2, "TUPLE3": 3}[name]
                if len(stack) < number:
                    raise RuntimeError(f"pickle opcode stack underflow at {name} in {path}")
                del stack[-number:]
                stack.append(UNKNOWN)
            elif name in {"APPEND"}:
                if len(stack) < 2:
                    raise RuntimeError(f"pickle opcode stack underflow at APPEND in {path}")
                stack.pop()
            elif name == "SETITEM":
                if len(stack) < 3:
                    raise RuntimeError(f"pickle opcode stack underflow at SETITEM in {path}")
                stack.pop(); stack.pop()
            elif name in {"REDUCE", "NEWOBJ_EX"}:
                number = 2 if name == "REDUCE" else 3
                if len(stack) < number:
                    raise RuntimeError(f"pickle opcode stack underflow at {name} in {path}")
                del stack[-number:]
                stack.append(UNKNOWN)
            elif name == "BUILD":
                if len(stack) < 2:
                    raise RuntimeError(f"pickle opcode stack underflow at BUILD in {path}")
                stack.pop()
            elif name in {"BYTEARRAY8", "BINBYTES", "BINBYTES8", "SHORT_BINBYTES", "SHORT_BINSTRING", "BINSTRING", "STRING", "BININT", "BININT1", "BININT2", "INT", "LONG", "LONG1", "LONG4", "FLOAT", "BINFLOAT"}:
                stack.append(UNKNOWN)
            elif name in {"PERSID", "BINPERSID", "NEXT_BUFFER"}:
                if name == "PERSID":
                    stack.append(UNKNOWN)
                elif name == "BINPERSID":
                    if stack:
                        stack.pop()
                    stack.append(UNKNOWN)
                else:
                    stack.append(UNKNOWN)
            elif name == "READONLY_BUFFER":
                if not stack:
                    raise RuntimeError(f"pickle opcode stack underflow at READONLY_BUFFER in {path}")
                stack[-1] = UNKNOWN
            elif name in {"EXT1", "EXT2", "EXT4"}:
                stack.append(UNKNOWN)
            elif name == "STOP":
                break
            else:
                raise RuntimeError(f"unhandled pickle opcode {name}; cannot prove whitelist for {path}")

    if not globals_seen or not globals_seen.issubset(ALLOWED_GLOBALS) or forbidden:
        raise RuntimeError(
            f"restricted pickle precheck failed for {path}: globals={sorted(globals_seen)}, "
            f"forbidden={sorted(forbidden)}"
        )
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "opcode_count": opcode_count,
        "stack_global_count": stack_global_count,
        "dynamic_globals": sorted(".".join(item) for item in globals_seen),
        "forbidden_opcodes": sorted(forbidden),
        "stack_resolution": "all STACK_GLOBAL operands resolved to literal module/name strings",
    }


def restricted_load(path: Path) -> Any:
    scan_pickle_globals(path)

    class RestrictedUnpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str) -> Any:
            safe_globals = {
                ("numpy", "dtype"): np.dtype,
                ("numpy", "ndarray"): np.ndarray,
                ("numpy.core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct,
                ("numpy.core.numeric", "_frombuffer"): np.core.numeric._frombuffer,
                ("numpy._core.numeric", "_frombuffer"): np.core.numeric._frombuffer,
            }
            if (module, name) not in safe_globals:
                raise pickle.UnpicklingError(f"blocked global: {module}.{name}")
            return safe_globals[(module, name)]

    with path.open("rb") as stream:
        return RestrictedUnpickler(stream, encoding="ASCII", errors="strict").load()


def vector_mapping(
    path: Path,
    expected_sha: str | None = None,
    required_inchikeys: set[str] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if expected_sha and sha256(path) != expected_sha:
        raise RuntimeError(f"SHA-256 mismatch before restricted load: {path}")
    payload = restricted_load(path)
    if not isinstance(payload, dict):
        raise RuntimeError(f"feature pickle is not a dictionary: {path}")
    by_ik: dict[str, np.ndarray] = {}
    duplicate_counts: dict[str, int] = {}
    ambiguous: set[str] = set()
    for smiles, value in payload.items():
        if not isinstance(smiles, str):
            raise RuntimeError(f"non-text feature key in {path}")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise RuntimeError(f"unparseable feature SMILES in {path}")
        arr = np.asarray(value)
        if arr.shape != (KPGT_DIM,) or arr.dtype.kind not in "biuf" or not np.isfinite(arr).all():
            raise RuntimeError(f"non-finite or non-2304D feature value in {path}")
        ik = Chem.MolToInchiKey(mol)
        duplicate_counts[ik] = duplicate_counts.get(ik, 0) + 1
        existing = by_ik.get(ik)
        if existing is not None and not np.array_equal(existing, arr):
            ambiguous.add(ik)
        by_ik[ik] = arr
    if required_inchikeys and (ambiguous & required_inchikeys):
        raise RuntimeError(
            f"required panel InChIKeys have multiple distinct feature vectors in {path}: "
            f"{sorted(ambiguous & required_inchikeys)}"
        )
    if expected_sha and sha256(path) != expected_sha:
        raise RuntimeError(f"SHA-256 changed during restricted load: {path}")
    return by_ik, {
        "entry_count": len(payload),
        "unique_inchikey_count": len(by_ik),
        "duplicate_inchikey_count": sum(n > 1 for n in duplicate_counts.values()),
        "ambiguous_inchikey_count": len(ambiguous),
        "all_values_2304d_finite": True,
        "sha256": sha256(path),
    }


def metrics(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    delta = np.abs(aa - bb)
    rel = delta / np.maximum(np.abs(bb), 1e-12)
    return {
        "max_abs": float(delta.max(initial=0.0)),
        "max_rel": float(rel.max(initial=0.0)),
        "exact_count": int(np.count_nonzero(aa == bb)),
        "allclose": bool(np.allclose(aa, bb, rtol=RTOL, atol=ATOL)),
        "rtol": RTOL,
        "atol": ATOL,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        print("NO_GO: PYTHONDONTWRITEBYTECODE=1 is required", file=sys.stderr)
        return 2
    if not ARTIFACT_ROOT.exists():
        artifact = ARTIFACT_ROOT
    else:
        attempt = 2
        while (ARTIFACT_ROOT / f"attempt_{attempt:02d}").exists():
            attempt += 1
        artifact = ARTIFACT_ROOT / f"attempt_{attempt:02d}"
    if artifact.exists():
        print(f"NO_GO: artifact path already exists: {artifact}", file=sys.stderr)
        return 2
    artifact.mkdir(parents=True)
    global ARTIFACT
    ARTIFACT = artifact

    report: dict[str, Any] = {
        "task": "EXP-037 14-drug KPGT local CPU reconstruction probe",
        "status": "PREFLIGHT",
        "scope": "frozen EXP-036 panel identity and local LINCS KPGT cache only",
        "forbidden_inputs_used": [],
        "comparison_threshold": {"rtol": RTOL, "atol": ATOL},
        "cpu_only": True,
        "gpu_training_hours": 0.0,
        "started_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "python": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "environment": package_versions(),
        "environment_variables": {
            key: os.environ.get(key)
            for key in ("CONDA_DEFAULT_ENV", "CONDA_PREFIX", "PYTHONDONTWRITEBYTECODE", "CUDA_VISIBLE_DEVICES")
        },
        "nvidia_smi_before": nvidia_smi(),
    }
    wall_start = time.perf_counter()
    cpu_start = resource.getrusage(resource.RUSAGE_SELF)
    try:
        if "drugscreening-gpu" not in str(sys.prefix):
            raise RuntimeError(f"not running in drugscreening-gpu: {sys.prefix}")
        if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", None):
            raise RuntimeError(f"CUDA_VISIBLE_DEVICES must be empty for CPU-only run: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

        repos = {"extension": EXT, "kpgt": KPGT, "official_reference": OFFICIAL}
        pre_repos = {name: repo_snapshot(path) for name, path in repos.items()}
        report["external_repositories_before"] = pre_repos

        registry_sha = sha256(REGISTRY)
        manifest = CONTRACT / "manifest.json"
        manifest_sha = sha256(manifest)
        manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
        registry_data = json.loads(REGISTRY.read_text(encoding="utf-8"))
        record = next((item for item in registry_data if item.get("id") == "dual_route_exp036_direct_response_contract_v1"), None)
        if record is None or record.get("checksum", {}).get("value") != EXPECTED_MANIFEST_SHA:
            raise RuntimeError("EXP-036 contract registry entry does not match expected manifest SHA-256")
        if manifest_sha != EXPECTED_MANIFEST_SHA:
            raise RuntimeError(f"EXP-036 manifest SHA mismatch: {manifest_sha}")
        if manifest_data.get("files_sha256", {}).get("drugs.parquet") != EXPECTED_PANEL_SHA:
            raise RuntimeError("EXP-036 manifest does not register expected drug panel SHA-256")
        if sha256(PANEL) != EXPECTED_PANEL_SHA:
            raise RuntimeError("frozen drug panel SHA-256 mismatch")

        panel = pd.read_parquet(PANEL, columns=["drug_index", "drug_id", "canonical_isomeric_smiles"])
        if len(panel) != 14 or panel["drug_index"].astype(int).tolist() != list(range(14)):
            raise RuntimeError("frozen 14-drug index axis failed")
        if panel["drug_id"].astype(str).duplicated().any() or panel["canonical_isomeric_smiles"].astype(str).duplicated().any():
            raise RuntimeError("frozen drug ids or canonical SMILES are not unique")
        identities = []
        for row in panel.itertuples(index=False):
            smiles = str(row.canonical_isomeric_smiles)
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise RuntimeError(f"invalid panel canonical SMILES at index {row.drug_index}")
            canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
            if canonical != smiles:
                raise RuntimeError(f"panel SMILES is not canonical at index {row.drug_index}")
            identities.append({
                "drug_index": int(row.drug_index),
                "drug_id": str(row.drug_id),
                "canonical_isomeric_smiles": smiles,
                "inchikey": Chem.MolToInchiKey(mol),
            })
        if len({item["inchikey"] for item in identities}) != 14:
            raise RuntimeError("panel InChIKey identity axis is not unique")
        panel.to_csv(ARTIFACT / "panel.csv", columns=["drug_index", "drug_id", "canonical_isomeric_smiles"], index=False)
        write_json(ARTIFACT / "panel_identities.json", {"panel_sha256": EXPECTED_PANEL_SHA, "rows": identities})
        panel_by_index = {item["drug_index"]: item for item in identities}

        inputs = {
            "drug_panel": sha_or_missing(PANEL),
            "panel_manifest": sha_or_missing(manifest),
            "dataset_registry": sha_or_missing(REGISTRY),
            "local_lincs_cache": sha_or_missing(OLD_LINCS),
            "extension_local_prism_cache": sha_or_missing(EXT / "data/PRISM/KPGT_emb2304.pickle"),
            "official_reference_lincs_cache": sha_or_missing(OFFICIAL / "data/LINCS2020/KPGT_emb2304.pickle"),
            "official_reference_prism_cache": sha_or_missing(OFFICIAL / "data/PRISM/KPGT_emb2304.pickle"),
            "pretrained_base_pth": sha_or_missing(WEIGHT),
        }
        if inputs["pretrained_base_pth"].get("sha256") != EXPECTED_WEIGHT_SHA or inputs["pretrained_base_pth"].get("size_bytes") != 447109411:
            raise RuntimeError("KPGT base.pth size/SHA does not match the frozen preflight")
        if inputs["local_lincs_cache"].get("sha256") != EXPECTED_OLD_LINCS_SHA:
            raise RuntimeError("local LINCS cache size/SHA changed from audited cache")

        code_paths = [
            EXT / "src/prepare_kpgt_embeddings.py",
            EXT / "scripts/kpgt_runner.py",
            EXT / "scripts/kpgt_patch/sitecustomize.py",
            KPGT / "scripts/preprocess_downstream_dataset.py",
            KPGT / "scripts/extract_features.py",
            KPGT / "src/model_config.py",
            KPGT / "src/data/descriptors/DescriptorGenerator.py",
            KPGT / "src/data/descriptors/dists.py",
        ]
        report["frozen_inputs"] = {
            "panel_sha256": EXPECTED_PANEL_SHA,
            "manifest_sha256": manifest_sha,
            "dataset_registry_sha256": registry_sha,
            "local_lincs_cache_sha256": EXPECTED_OLD_LINCS_SHA,
            "pretrained_base_pth_sha256": EXPECTED_WEIGHT_SHA,
            "assets": inputs,
            "generator_preprocessor_extractor_hashes": [sha_or_missing(path) for path in code_paths],
        }

        panel_inchikeys = {item["inchikey"] for item in identities}
        old_vectors, old_summary = vector_mapping(
            OLD_LINCS, EXPECTED_OLD_LINCS_SHA, required_inchikeys=panel_inchikeys
        )
        old_crosswalk = {}
        for index, identity in panel_by_index.items():
            if identity["inchikey"] in old_vectors:
                old_crosswalk[index] = old_vectors[identity["inchikey"]]
        if set(old_crosswalk) != set(range(14)) - {6}:
            raise RuntimeError(f"approved LINCS crosswalk changed; found panel indices {sorted(old_crosswalk)}")
        report["old_lincs_restricted_summary"] = old_summary
        report["old_lincs_panel_indices"] = sorted(old_crosswalk)
        report["preflight_help"] = {"command": [sys.executable, str(EXT / "src/prepare_kpgt_embeddings.py"), "--help"], "returncode": 0}

        prior_artifact = ARTIFACT_ROOT / "attempt_03"
        prior_receipt_path = prior_artifact / "receipt.json"
        if not prior_receipt_path.is_file():
            raise RuntimeError("approved completed run_a receipt is missing; refusing to rerun run_a")
        prior_receipt = json.loads(prior_receipt_path.read_text(encoding="utf-8"))
        if prior_receipt.get("failure", "").find("restricted pickle precheck failed") < 0:
            raise RuntimeError("prior run_a did not fail solely in the diagnostic pickle parser")
        if prior_receipt.get("external_repository_dirty_state_unchanged") is not True:
            raise RuntimeError("prior run_a external repository state was not confirmed unchanged")
        if prior_receipt.get("external_repositories_before") != pre_repos or prior_receipt.get("external_repositories_final") != pre_repos:
            raise RuntimeError("prior run_a repository snapshots differ from current preflight; refusing reuse")
        if prior_receipt.get("frozen_inputs") != report["frozen_inputs"]:
            raise RuntimeError("prior run_a frozen panel/cache/weight/code hashes differ; refusing reuse")
        prior_runs = prior_receipt.get("generator_runs", [])
        if len(prior_runs) != 1 or prior_runs[0].get("run") != "run_a" or prior_runs[0].get("returncode") != 0:
            raise RuntimeError("prior run_a generator did not exit successfully exactly once")
        prior_run_a = prior_runs[0]
        prior_command = prior_run_a.get("command", [])
        if not all(flag in prior_command for flag in ("--base_embedding_paths", "--run_kpgt", "--kpgt_force_cpu", "--kpgt_python")):
            raise RuntimeError("prior run_a command does not match the approved cache-free CPU generator protocol")
        if prior_command[prior_command.index("--base_embedding_paths") + 1] != "--run_kpgt":
            raise RuntimeError("prior run_a did not pass an explicitly empty base cache list")
        if prior_run_a.get("environment", {}).get("PYTHONDONTWRITEBYTECODE") != "1" or prior_run_a.get("environment", {}).get("CUDA_VISIBLE_DEVICES") != "":
            raise RuntimeError("prior run_a did not carry the required CPU/bytecode safeguards")
        prior_pickle_record = prior_run_a.get("generated_pickle", {})
        prior_pickle = Path(prior_pickle_record.get("path", ""))
        if not prior_pickle.is_file() or not str(prior_pickle.resolve()).startswith(str(ARTIFACT_ROOT.resolve()) + os.sep):
            raise RuntimeError("prior run_a feature output missing or outside the authorized artifact")
        if sha256(prior_pickle) != prior_pickle_record.get("sha256"):
            raise RuntimeError("prior run_a feature output SHA changed; refusing reuse")
        opcode_scan = scan_pickle_globals(prior_pickle)
        write_json(ARTIFACT / "pickle_opcode_scan_run_a.json", opcode_scan)
        prior_vectors, prior_summary = vector_mapping(prior_pickle, required_inchikeys=panel_inchikeys)
        if set(prior_vectors) != panel_inchikeys or len(prior_vectors) != 14:
            raise RuntimeError("prior run_a restricted output does not contain 14 unique panel identities")
        generated_by_run: dict[str, dict[str, np.ndarray]] = {"run_a": prior_vectors}
        first_call_record = dict(prior_run_a)
        first_call_record["execution_artifact"] = str(prior_artifact)
        first_call_record["generated_summary"] = prior_summary
        first_call_record["panel_indices"] = list(range(14))
        first_call_record["reused_after_static_opcode_audit"] = True
        report["generator_runs"] = [first_call_record]
        report["run_a_reuse_basis"] = {
            "prior_receipt": str(prior_receipt_path),
            "prior_status": prior_receipt.get("status"),
            "prior_failure_was_diagnostic_parser_only": prior_receipt.get("failure"),
            "opcode_scan": opcode_scan,
            "reran_run_a": False,
        }
        (ARTIFACT / "run_b").mkdir()
        write_json(ARTIFACT / "receipt.json", report)

        command_records = report["generator_runs"]
        for run_name in ("run_b",):
            run_dir = ARTIFACT / run_name
            dataset_name = f"exp037_kpgt_probe_{'a' if run_name == 'run_a' else 'b'}"
            output_pickle = run_dir / "KPGT_emb2304_for_screening.pickle"
            output_csv = run_dir / "screening_compound_kpgt_ready.csv"
            missing_csv = run_dir / "missing_smiles_for_kpgt.csv"
            command = [
                sys.executable, str(EXT / "src/prepare_kpgt_embeddings.py"),
                "--input_csv", str(ARTIFACT / "panel.csv"),
                "--smiles_col", "canonical_isomeric_smiles",
                "--id_col", "drug_id", "--name_col", "drug_id",
                "--base_embedding_paths",
                "--run_kpgt", "--kpgt_force_cpu", "true", "--kpgt_python", "python",
                "--kpgt_root", str(KPGT), "--kpgt_model_path", str(WEIGHT),
                "--kpgt_dataset", dataset_name, "--kpgt_n_jobs", "2",
                "--output_dir", str(run_dir),
                "--output_screening_csv", str(output_csv),
                "--output_embedding_pickle", str(output_pickle),
                "--missing_csv", str(missing_csv),
            ]
            child_env = dict(os.environ)
            child_env["PYTHONDONTWRITEBYTECODE"] = "1"
            child_env["CUDA_VISIBLE_DEVICES"] = ""
            child_env["OMP_NUM_THREADS"] = "2"
            child_env["OPENBLAS_NUM_THREADS"] = "1"
            log_path = run_dir / "generator.log"
            started = time.perf_counter()
            with log_path.open("w", encoding="utf-8") as log:
                proc = subprocess.run(
                    command, cwd=str(EXT), env=child_env,
                    stdout=log, stderr=subprocess.STDOUT, check=False,
                )
            record_run = {
                "run": run_name,
                "command": command,
                "environment": {"PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1"},
                "returncode": proc.returncode,
                "wall_seconds": time.perf_counter() - started,
                "log_path": str(log_path),
                "generated_pickle": sha_or_missing(run_dir / "generated_kpgt_emb2304.pickle"),
                "final_pickle": sha_or_missing(output_pickle),
            }
            command_records.append(record_run)
            report["generator_runs"] = command_records
            if proc.returncode != 0:
                raise RuntimeError(f"{run_name} generator exited {proc.returncode}; stopped immediately")

            generated_path = run_dir / "generated_kpgt_emb2304.pickle"
            vectors, summary = vector_mapping(generated_path, required_inchikeys=panel_inchikeys)
            if set(vectors) != {item["inchikey"] for item in identities} or len(vectors) != 14:
                raise RuntimeError(f"{run_name} did not yield 14 unique frozen panel identities")
            generated_by_run[run_name] = vectors
            record_run["generated_summary"] = summary
            record_run["panel_indices"] = sorted(
                index for index, identity in panel_by_index.items() if identity["inchikey"] in vectors
            )
            if record_run["panel_indices"] != list(range(14)):
                raise RuntimeError(f"{run_name} panel output axis is incomplete")

            post_repos = {name: repo_snapshot(path) for name, path in repos.items()}
            if post_repos != pre_repos:
                report["external_repositories_changed_after"] = post_repos
                raise RuntimeError(f"external repository dirty state changed after {run_name}; stop without cleanup")
            report["external_repositories_after_" + run_name] = post_repos
            if sha256(WEIGHT) != EXPECTED_WEIGHT_SHA or sha256(OLD_LINCS) != EXPECTED_OLD_LINCS_SHA:
                raise RuntimeError(f"protected model weight or local LINCS cache changed after {run_name}")

        comparisons = []
        for index in sorted(old_crosswalk):
            identity = panel_by_index[index]
            result = metrics(generated_by_run["run_a"][identity["inchikey"]], old_crosswalk[index])
            comparisons.append({"drug_index": index, "drug_id": identity["drug_id"], "inchikey": identity["inchikey"], **result})
        index6 = panel_by_index[6]
        stability = metrics(
            generated_by_run["run_a"][index6["inchikey"]],
            generated_by_run["run_b"][index6["inchikey"]],
        )
        report["fixed_threshold_comparisons"] = comparisons
        report["index6_two_run_stability"] = {"drug_id": index6["drug_id"], "inchikey": index6["inchikey"], **stability}
        if len(comparisons) != 13 or not all(row["allclose"] for row in comparisons) or not stability["allclose"]:
            raise RuntimeError("fixed-threshold cache consistency or index 6 stability gate failed")

        report["status"] = "LOCAL_RECONSTRUCTION_CANDIDATE"
        report["index6_status"] = "LOCAL_RECONSTRUCTION_CANDIDATE"
        report["scope_limit"] = "14-drug EXP-036 panel diagnostic only; no inference to EXP-037 B1 20,336-drug axis"
        report["scientific_performance_claim"] = False
        report["official_release_reproduction_claim"] = False
    except Exception as exc:
        report["status"] = "NO_GO"
        report["failure"] = f"{type(exc).__name__}: {exc}"
        report["scientific_performance_claim"] = False
        report["official_release_reproduction_claim"] = False
    finally:
        report["nvidia_smi_after"] = nvidia_smi()
        report["wall_seconds_total"] = time.perf_counter() - wall_start
        cpu_end = resource.getrusage(resource.RUSAGE_SELF)
        report["process_cpu_seconds"] = {
            "user": cpu_end.ru_utime - cpu_start.ru_utime,
            "system": cpu_end.ru_stime - cpu_start.ru_stime,
        }
        try:
            final_repos = {"extension": repo_snapshot(EXT), "kpgt": repo_snapshot(KPGT), "official_reference": repo_snapshot(OFFICIAL)}
            report["external_repositories_final"] = final_repos
            report["external_repository_dirty_state_unchanged"] = final_repos == report.get("external_repositories_before")
            if report.get("status") == "LOCAL_RECONSTRUCTION_CANDIDATE" and not report["external_repository_dirty_state_unchanged"]:
                report["status"] = "NO_GO"
                report["failure"] = "external repository dirty-state differs at final snapshot; no cleanup performed"
        except Exception as exc:
            report["final_state_capture_error"] = f"{type(exc).__name__}: {exc}"
            if report.get("status") == "LOCAL_RECONSTRUCTION_CANDIDATE":
                report["status"] = "NO_GO"
                report["failure"] = "could not verify final external repository state"
        report["completed_local"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
        write_json(ARTIFACT / "receipt.json", report)

    print(json.dumps({
        "status": report.get("status"),
        "failure": report.get("failure"),
        "artifact": str(ARTIFACT),
        "wall_seconds_total": report.get("wall_seconds_total"),
        "generator_runs": [
            {key: run.get(key) for key in ("run", "returncode", "wall_seconds", "generated_summary")}
            for run in report.get("generator_runs", [])
        ],
    }, ensure_ascii=False))
    return 0 if report.get("status") == "LOCAL_RECONSTRUCTION_CANDIDATE" else 2


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--scan-pickle":
        summary = scan_pickle_globals(Path(sys.argv[2]))
        if len(sys.argv) == 5 and sys.argv[3] == "--output":
            output_path = Path(sys.argv[4]).resolve()
            if not output_path.is_relative_to(ARTIFACT_ROOT.resolve()):
                raise SystemExit("scan output must stay under the EXP-037 diagnostic artifact")
            output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        elif len(sys.argv) != 3:
            raise SystemExit("usage: audit_exp037_kpgt_local_reproduction.py --scan-pickle INPUT [--output ARTIFACT_JSON]")
        print(json.dumps(summary, ensure_ascii=False))
        raise SystemExit(0)
    raise SystemExit(main())
