"""EXP-049 exploratory cold-start Hallmark comparison on outer-0 training patients only.

This entrypoint has a CPU-only --preflight-only mode. Training requires a
user-created authorization receipt, one explicit CUDA device, and a reservation
in the pre-existing 40-GPU-hour shared ledger. No CPU fallback is implemented.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts" / "modeling")]
from drug_screen.models.exp046_five_arm import (  # noqa: E402
    AlignedPathwayMap,
    expand_mapped_pathway_map,
    build_exp046_arm_model,
)
from drug_screen.transfer.pathways import parse_gmt  # noqa: E402
from exp036_pharmaformer_fold import pairwise_ranking_loss  # noqa: E402
from shared_gpu_ledger import (  # noqa: E402
    allocation_from_events,
    budget_snapshot,
    ledger_transaction,
    package_budget_snapshot,
    read_events,
)

CONFIG = ROOT / "configs/experiments/EXP-049.yaml"
SPLIT = ROOT / "data/splits/dual_route_splits_20260922_v1.json"
INPUT = ROOT / "data/processed/dual_route/EXP-036/pharmaformer_inputs_source_fold0_v1"
DIRECT_CONTRACT_MANIFEST = ROOT / "data/processed/dual_route/EXP-036/direct_response_contract_v1/manifest.json"
MAPS = ROOT / "artifacts/experiments/EXP-046/20260924_pathway_control_maps_v1/maps.npz"
MAP_RECEIPT = ROOT / "artifacts/experiments/EXP-046/20260924_pathway_control_maps_v1/receipt.json"
GENES = ROOT / "data/processed/dual_route/EXP-036/direct_response_contract_v1/genes.json"
GMT = ROOT / "data/raw/exp013/msigdb/h.all.v2023.2.Hs.symbols.gmt"
LEDGER = ROOT / "artifacts/experiments/EXP-037/20260926_conditional_v2_folds3_4_budget_amendment_v2/queue_ledger.jsonl"
LEDGER_LOCK = LEDGER.with_name("queue.lock")
MIRROR = ROOT / "artifacts/experiments/EXP-049/20260927_budget_request/budget_ledger.jsonl"
RUN_CAP = 1.25
EXPERIMENT_CAP = 1.25
SHUTDOWN_RESERVE = 0.05
TRAINING_STOP = RUN_CAP - SHUTDOWN_RESERVE
SHARED_CAP = 40.0
PACKAGE_ID = "SHARED_EMERGENCY"
SUBALLOCATION_ID = "EXP049_PATHWAY_COLDSTART"
GPU_UUID = "GPU-9c366c2e-bfa5-c7b3-b16a-1117c40e6871"
PROPOSAL_ID = "EXP049_SHARED_CONTINGENCY_REALLOCATION_001"
EXPECTED = {
    "input_manifest": "c6f9c3349b0a8598280ba099381b781c68ca5599f6f3b7e749090a0e46460b72",
    "split": "9bfeeb4e0982421cc188a54e2a65f348709043c05529e641c7dd11ffc0635be8",
    "maps": "96cbba6625e1a0f85b961644225ec82f21259891e1d3b4f8b9d235ea38a47f53",
    "genes": "bf023d7b110bd820b3b41c67ab9ce5ea6b2381ab5753c3ec6440ee2a61340610",
    "gmt": "8ff1f036f0988c99d42cf9e5ea24d355f10b8ece5e0d4f6edf26a0458f480780",
}
MAP_KEYS = {
    "P-C": ("pc", None),
    "P-R-20260922": ("pr_20260922", 20260922),
    "P-R-20260923": ("pr_20260923", 20260923),
    "P-R-20260924": ("pr_20260924", 20260924),
}


class BudgetStop(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_static_inputs() -> dict[str, Any]:
    config = CONFIG.read_bytes()
    manifest = json.loads((INPUT / "manifest.json").read_text(encoding="utf-8"))
    split_bytes = SPLIT.read_bytes()
    genes_bytes = GENES.read_bytes()
    gmt_bytes = GMT.read_bytes()
    maps_bytes = MAPS.read_bytes()
    observed = {
        "input_manifest": sha_bytes((INPUT / "manifest.json").read_bytes()),
        "split": sha_bytes(split_bytes),
        "maps": sha_bytes(maps_bytes),
        "genes": sha_bytes(genes_bytes),
        "gmt": sha_bytes(gmt_bytes),
    }
    # The patient input contract has a separately recorded manifest hash.
    direct_contract_sha256 = sha_file(DIRECT_CONTRACT_MANIFEST)
    if manifest.get("contract_manifest_sha256") != direct_contract_sha256:
        raise RuntimeError("EXP-036 direct-response contract manifest changed")
    observed["direct_response_contract_manifest"] = direct_contract_sha256
    for key, expected in EXPECTED.items():
        if observed[key] != expected:
            raise RuntimeError(f"frozen EXP-049 input hash mismatch: {key}")
    if manifest.get("n_patients") != 51 or manifest.get("n_drugs") != 14:
        raise RuntimeError("EXP-036 materialized target input axes changed")
    registered_files = manifest.get("files_sha256", {})
    for filename in ("patient_pairs.parquet", "patient_expression.npy",
                     "patient_axis.json", "drug_axis.parquet", "drug_ordinal128.npy"):
        expected_file_hash = registered_files.get(filename)
        if not expected_file_hash:
            raise RuntimeError(f"input manifest lacks registered checksum for {filename}")
        actual_file_hash = sha_file(INPUT / filename)
        if actual_file_hash != expected_file_hash:
            raise RuntimeError(f"input file checksum mismatch: {filename}")
        observed[f"input_file__{filename}"] = actual_file_hash
    split = json.loads(split_bytes)
    fold = split["target_folds"][0]
    fraction = next(item for item in fold["label_fractions"] if item["fraction"] == 0.5)
    permitted = tuple(fraction["train_ids"])
    outer_test = set(fold["outer_test_ids"])
    if len(permitted) != 20 or set(permitted) & outer_test:
        raise RuntimeError("outer0 training cohort differs or intersects outer test")
    inner = fraction["inner_folds"]
    expected_sizes = [(13, 7), (13, 7), (14, 6)]
    if [(len(item["train_ids"]), len(item["validation_ids"])) for item in inner] != expected_sizes:
        raise RuntimeError("registered three inner folds changed")
    val_ids = [identity for item in inner for identity in item["validation_ids"]]
    if len(val_ids) != 20 or set(val_ids) != set(permitted) or len(set(val_ids)) != 20:
        raise RuntimeError("inner validation patients do not partition the allowed patients")

    # Read only identities/drug keys for the allowed patient rows. Response values
    # are not requested by this preflight; outer-test rows are filtered out.
    keys = pq.read_table(
        INPUT / "patient_pairs.parquet",
        columns=["patient_id", "drug_id"],
        filters=[("patient_id", "in", list(permitted))],
    ).to_pandas()
    if set(keys["patient_id"].astype(str)) != set(permitted):
        raise RuntimeError("the permitted patient rows are incomplete")
    drug_axis = pd.read_parquet(INPUT / "drug_axis.parquet", columns=["drug_index", "drug_id"])
    drug_axis["drug_id"] = drug_axis["drug_id"].astype(str)
    if (len(drug_axis) != 14 or drug_axis["drug_id"].duplicated().any()
            or sorted(drug_axis["drug_index"].astype(int).tolist()) != list(range(14))):
        raise RuntimeError("frozen indexed drug axis is not a unique ordered 14-drug contract")
    if not set(keys["drug_id"].astype(str)) <= set(drug_axis["drug_id"]):
        raise RuntimeError("patient pair keys are outside the frozen drug axis")

    axis = json.loads((INPUT / "patient_axis.json").read_text(encoding="utf-8"))
    if len(axis) != 51 or len(set(axis)) != 51:
        raise RuntimeError("frozen target patient axis changed")
    axis_positions = {str(value): i for i, value in enumerate(axis)}
    expression = np.load(INPUT / "patient_expression.npy", mmap_mode="r", allow_pickle=False)
    if expression.shape != (51, 19091):
        raise RuntimeError("patient expression tensor shape changed")
    # Touch only permitted rows; do not materialize outer-test expression rows.
    allowed_expression = np.asarray(expression[[axis_positions[value] for value in permitted]])
    if allowed_expression.shape != (20, 19091) or not np.isfinite(allowed_expression).all():
        raise RuntimeError("permitted expression inputs are invalid")
    drug_features = np.load(INPUT / "drug_ordinal128.npy", mmap_mode="r", allow_pickle=False)
    if drug_features.shape != (14, 128) or not np.isfinite(drug_features).all():
        raise RuntimeError("frozen ordered drug feature matrix is invalid")
    return {
        "status": "PREFLIGHT_PASS_LABEL_FREE_OUTER_TEST_EXCLUDED",
        "protocol_sha256": sha_bytes(config),
        "input_hashes": observed,
        "code_hashes": {
            "exp049_runner": sha_file(Path(__file__).resolve()),
            "official_pharmaformer_source": sha_file(ROOT / "data/external/exp010/pharmaformer/repo/pretrain.py"),
            "exp046_model_builder": sha_file(ROOT / "src/drug_screen/models/exp046_five_arm.py"),
            "pairwise_loss_source": sha_file(ROOT / "scripts/modeling/exp036_pharmaformer_fold.py"),
        },
        "outer_fold": 0,
        "training_patients": 20,
        "outer_test_patient_count_in_split_only": len(outer_test),
        "inner_sizes": expected_sizes,
        "allowed_patient_pair_key_rows": int(len(keys)),
        "target_labels_loaded": False,
        "outer_test_labels_loaded_by_preflight_process": False,
        "patient_pair_response_file_bytes_hashed_without_parquet_response_decode": True,
        "outer_test_expression_loaded": False,
        "frozen_maps_archive_sha256": observed["maps"],
        "cuda_initialized": torch.cuda.is_initialized(),
        "preflight_wall_seconds": None,
    }


def check_runtime_device() -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible != GPU_UUID:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must bind exactly the approved A5000 UUID")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("CUDA must expose exactly the approved single GPU; CPU fallback is disabled")
    rows = subprocess.check_output(
        ["nvidia-smi", "-i", GPU_UUID,
         "--query-gpu=uuid,name,utilization.gpu,memory.used", "--format=csv,noheader"], text=True
    ).splitlines()
    found = [line.strip() for line in rows if line.strip().startswith(GPU_UUID)]
    if len(found) != 1 or "RTX A5000" not in found[0]:
        raise RuntimeError("approved GPU UUID is absent or not an RTX A5000")
    parts = [part.strip() for part in found[0].split(",")]
    utilization = int(parts[2].split()[0])
    memory_mib = int(parts[3].split()[0])
    if utilization > 5 or memory_mib > 512:
        raise RuntimeError("approved A5000 is occupied; stop before reservation")
    compute_apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
         "--format=csv,noheader"], text=True
    ).splitlines()
    if any(line.strip().startswith(GPU_UUID) for line in compute_apps):
        raise RuntimeError("a compute process is already attached to the approved A5000")
    device_name = torch.cuda.get_device_name(0)
    if "A5000" not in device_name:
        raise RuntimeError(f"CUDA logical device differs from approved A5000: {device_name}")


def reserve_attempt(attempt_id: str, run_id: str, started_monotonic: float) -> dict[str, Any]:
    with ledger_transaction(LEDGER_LOCK) as tx:
        events = tx.read(LEDGER)
        allocation = allocation_from_events(events)
        total = budget_snapshot(events)
        emergency = package_budget_snapshot(events, PACKAGE_ID)
        prior_exp049_starts = [
            row for row in events
            if row.get("event") == "attempt_started"
            and row.get("suballocation_id") == SUBALLOCATION_ID
        ]
        if prior_exp049_starts:
            raise RuntimeError("EXP-049 authorization permits exactly one attempt; no retry or second reservation")
        if total["active_attempts"]:
            raise RuntimeError("another shared-tranche reservation is active; do not overlap")
        if total["settled_gpu_hours"] + total["reserved_gpu_hours"] + RUN_CAP > SHARED_CAP + 1e-9:
            raise RuntimeError("EXP-049 reservation would exceed the original shared 40 GPUh cap")
        if RUN_CAP > EXPERIMENT_CAP + 1e-9:
            raise RuntimeError("EXP-049 single-attempt cap exceeds its package cap")
        if emergency["remaining_gpu_hours"] + 1e-9 < RUN_CAP:
            raise RuntimeError("existing SHARED_EMERGENCY allocation cannot cover the proposed suballocation")
        if abs(float(allocation["allocations_gpu_hours"][PACKAGE_ID]) - 2.2293433805) > 1e-8:
            raise RuntimeError("shared emergency allocation differs from the proposal basis")
        start = tx.append(
            LEDGER, "attempt_started", attempt_id=attempt_id,
            package_id=PACKAGE_ID, suballocation_id=SUBALLOCATION_ID,
            experiment_ids=["EXP-049"], run_id=run_id,
            started_monotonic_seconds=started_monotonic,
            reserved_gpu_hours=RUN_CAP, gpu_count=1, gpu_uuid=GPU_UUID,
            historical_exp037_48h_usage="UNKNOWN_UNRECOVERED",
            shared_tranche_cap_gpu_hours=SHARED_CAP,
            shared_settled_before_start_gpu_hours=total["settled_gpu_hours"],
            shared_reserved_before_start_gpu_hours=total["reserved_gpu_hours"],
            shared_unreserved_before_start_gpu_hours=total["available_gpu_hours"],
            no_retry=True,
        )
        spawned = tx.append(LEDGER, "process_spawned", attempt_id=attempt_id,
                            pid=os.getpid(), experiment_id="EXP-049", run_id=run_id,
                            gpu_uuid=GPU_UUID)
    MIRROR.parent.mkdir(parents=True, exist_ok=True)
    with MIRROR.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(start, sort_keys=True) + "\n")
        stream.write(json.dumps(spawned, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return start


def close_attempt(attempt_id: str, started: float, exit_code: int,
                  outcome: str, ended: float | None = None) -> float:
    elapsed = (time.perf_counter() if ended is None else ended) - started
    charge = elapsed / 3600.0
    if charge > RUN_CAP:
        outcome = "HARD_CAP_OVERRUN_ACTUAL_TIME_RECORDED"
        exit_code = exit_code or 1
    fields = {
        "attempt_id": attempt_id, "package_id": PACKAGE_ID,
        "suballocation_id": SUBALLOCATION_ID, "experiment_ids": ["EXP-049"],
        "ended_utc": utc_now(), "exit_code": exit_code, "outcome": outcome,
        "gpu_hours_charged": charge, "elapsed_seconds": elapsed,
        "gpu_uuid": GPU_UUID, "gpu_count": 1,
        "historical_exp037_48h_usage": "UNKNOWN_UNRECOVERED",
    }
    with ledger_transaction(LEDGER_LOCK) as tx:
        row = tx.append(LEDGER, "attempt_finished", **fields)
    with MIRROR.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return charge


def finalize_unclosed_attempt(attempt_id: str, exit_code: int, outcome: str,
                              stopped_monotonic: float) -> float | None:
    """Settle a killed child from its durable start row; never charge an attempt twice."""
    events = read_events(LEDGER, LEDGER_LOCK)
    start = next((row for row in events if row.get("event") == "attempt_started"
                  and row.get("attempt_id") == attempt_id), None)
    if start is None:
        return None
    closed = any(row.get("event") in {"attempt_finished", "attempt_reconciled"}
                 and row.get("attempt_id") == attempt_id for row in events)
    if closed:
        charge = next(float(row["gpu_hours_charged"]) for row in events
                      if row.get("event") in {"attempt_finished", "attempt_reconciled"}
                      and row.get("attempt_id") == attempt_id)
    else:
        monotonic_start = start.get("started_monotonic_seconds")
        if not isinstance(monotonic_start, (int, float)):
            raise RuntimeError("cannot settle watchdog-stopped attempt without monotonic start evidence")
        charge = close_attempt(attempt_id, float(monotonic_start), exit_code,
                               outcome, ended=stopped_monotonic)
        events = read_events(LEDGER, LEDGER_LOCK)

    # The shared ledger is authoritative. Repair only missing rows in the EXP-049 mirror.
    mirror_rows = []
    if MIRROR.exists():
        mirror_rows = [json.loads(line) for line in MIRROR.read_text(encoding="utf-8").splitlines()
                       if line.strip()]
    mirror_keys = {(row.get("event"), row.get("attempt_id")) for row in mirror_rows}
    missing = [row for row in events if row.get("attempt_id") == attempt_id
               and (row.get("event"), row.get("attempt_id")) not in mirror_keys]
    if missing:
        MIRROR.parent.mkdir(parents=True, exist_ok=True)
        with MIRROR.open("a", encoding="utf-8", newline="\n") as stream:
            for row in missing:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    run_dir = ROOT / "artifacts/experiments/EXP-049" / attempt_id.removesuffix("_attempt001")
    if run_dir.exists():
        close_path = run_dir / "attempt_close.json"
        close_temp = close_path.with_suffix(".json.tmp")
        close_temp.write_text(json.dumps({
            "attempt_id": attempt_id,
            "outcome": outcome,
            "exit_code": exit_code,
            "gpu_hours_charged": charge,
            "gpu_uuid": GPU_UUID,
            "ended_utc": utc_now(),
            "automatic_retry": False,
            "settled_by_supervisor": True,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(close_temp, close_path)
    return charge


def build_maps() -> tuple[dict[str, AlignedPathwayMap], tuple[str, ...], tuple[str, ...]]:
    genes = tuple(str(value).strip().upper() for value in json.loads(GENES.read_text(encoding="utf-8")))
    resource = parse_gmt(GMT.read_text(encoding="utf-8"), resource_id="MSigDB_Hallmark",
                         version="h.all.v2023.2.Hs.symbols")
    resource_gene_set = set().union(*(set(values) for values in resource.pathways.values()))
    mapped_genes = tuple(gene for gene in genes if gene in resource_gene_set)
    if len(mapped_genes) != 4366:
        raise RuntimeError("Hallmark mapped gene axis differs from the frozen 4,366-gene contract")
    positions = {gene: index for index, gene in enumerate(mapped_genes)}
    names = tuple(resource.pathways)
    true_map = np.zeros((len(names), len(mapped_genes)), dtype=np.uint8)
    for row, pathway in enumerate(names):
        for gene in resource.pathways[pathway]:
            if gene in positions:
                true_map[row, positions[gene]] = 1
    raw_maps: dict[str, np.ndarray] = {"P-H": true_map}
    with np.load(MAPS, allow_pickle=False) as archive:
        for arm, (key, _) in MAP_KEYS.items():
            raw_maps[arm] = np.array(archive[key], dtype=np.uint8, copy=True)
    aligned = {
        arm: expand_mapped_pathway_map(matrix, gene_axis=genes,
                                       mapped_gene_ids=mapped_genes,
                                       pathway_names=names)
        for arm, matrix in raw_maps.items()
    }
    return aligned, genes, names


def validate_model_structure(maps: dict[str, AlignedPathwayMap]) -> dict[str, int]:
    official_path = ROOT / "data/external/exp010/pharmaformer/repo/pretrain.py"
    spec = importlib.util.spec_from_file_location("exp049_preflight_pharmaformer", official_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("official PharmaFormer architecture source is unavailable")
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    torch.manual_seed(20260922)
    backbone = official.CombinedModel(
        gene_input_size=19091, gene_hidden_size=8192, drug_hidden_size=256,
        feature_dim=128, nhead=8, num_layers=3, dim_feedforward=2048, dropout=0.1)
    counts = {"P-A": sum(parameter.numel() for parameter in backbone.parameters())}
    configurations = [
        ("P-C", "P-C"), ("P-H", "P-T"),
        ("P-R-20260922", "P-R"), ("P-R-20260923", "P-R"),
        ("P-R-20260924", "P-R"),
    ]
    for name, arm in configurations:
        pathway_map = maps[name]
        model = build_exp046_arm_model(
            arm, backbone,
            ordinary_control_map=pathway_map if arm == "P-C" else None,
            frozen_random_map=pathway_map if arm == "P-R" else None,
            true_pathway_map=pathway_map if arm == "P-T" else None,
            rank=8, branch_seed=20260922, zero_initialize=True)
        counts[name] = sum(parameter.numel() for parameter in model.parameters())
        del model
    branch_counts = {counts[name] for name in counts if name != "P-A"}
    if len(branch_counts) != 1:
        raise RuntimeError("the frozen pathway arms do not have equal trainable capacity")
    return counts


def load_allowed_rows(permitted: tuple[str, ...]) -> pd.DataFrame:
    frame = pq.read_table(
        INPUT / "patient_pairs.parquet",
        columns=["patient_id", "drug_id", "response"],
        filters=[("patient_id", "in", list(permitted))],
    ).to_pandas()
    frame["patient_id"] = frame["patient_id"].astype(str)
    frame["drug_id"] = frame["drug_id"].astype(str)
    if set(frame["patient_id"]) - set(permitted):
        raise RuntimeError("outer-test or otherwise unauthorized patient key returned by row filter")
    if set(frame["patient_id"]) != set(permitted):
        raise RuntimeError("the permitted patient response rows are incomplete")
    if not np.isfinite(frame["response"].to_numpy(dtype=float)).all():
        raise RuntimeError("permitted training/validation responses include non-finite values")
    if frame.duplicated(["patient_id", "drug_id"]).any():
        raise RuntimeError("duplicate patient/drug label keys in allowed input")
    return frame.sort_values(["patient_id", "drug_id"], kind="stable").reset_index(drop=True)


def patient_scores(rows: pd.DataFrame) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    model_cols = ["P-A", "P-C", "P-H", "P-R-20260922", "P-R-20260923",
                  "P-R-20260924", "B0"]
    for patient, group in rows.groupby("patient_id", sort=True):
        common = group.dropna(subset=model_cols)
        n = int(common["drug_id"].nunique())
        if n < 3:
            reason = "FEWER_THAN_THREE_SHARED_OBSERVED_DRUGS"
        elif common["response"].nunique() < 2:
            reason = "CONSTANT_OBSERVED_RESPONSE"
        else:
            reason = None
        for model in model_cols:
            values = common[model].to_numpy(dtype=float)
            observed = common["response"].to_numpy(dtype=float)
            rho = float("nan")
            this_reason = reason
            if this_reason is None and np.ptp(values) == 0:
                this_reason = "CONSTANT_PREDICTION"
            elif this_reason is None:
                rho = float(spearmanr(observed, values).statistic)
                if not math.isfinite(rho):
                    this_reason = "UNDEFINED_CORRELATION"
            output.append({"patient_id": str(patient),
                           "inner_fold": int(group["inner_fold"].iloc[0]), "model": model,
                           "spearman": rho, "n_shared_drugs": n,
                           "reason": this_reason})
    return pd.DataFrame(output)


def paired_statistics(scores: pd.DataFrame) -> dict[str, Any]:
    wide = scores.pivot(index=["patient_id", "inner_fold"], columns="model",
                        values="spearman").reset_index()
    random_columns = ["P-R-20260922", "P-R-20260923", "P-R-20260924"]
    comparisons = {
        "P-H_minus_P-R_equal_map_mean": ("P-H", *random_columns),
        "P-H_minus_P-C": ("P-H", "P-C"),
        "P-H_minus_P-A": ("P-H", "P-A"),
        "P-H_minus_B0": ("P-H", "B0"),
    }
    results: dict[str, Any] = {}
    raw_p: dict[str, float] = {}
    denominator_counts: dict[str, int] = {}
    rng = np.random.default_rng(20260922)
    for name, required_models in comparisons.items():
        valid_mask = wide[list(required_models)].notna().all(axis=1)
        paired = wide.loc[valid_mask, ["patient_id", "inner_fold", *required_models]].copy()
        denominator_counts[name] = int(len(paired))
        if name == "P-H_minus_P-R_equal_map_mean":
            paired["control"] = paired[list(random_columns)].mean(axis=1, skipna=False)
        else:
            paired["control"] = paired[required_models[1]]
        difference = paired["P-H"].to_numpy(float) - paired["control"].to_numpy(float)
        exclusion_reasons: dict[str, dict[str, int]] = {}
        for model in required_models:
            missing_ids = set(wide.loc[wide[model].isna(), "patient_id"].astype(str))
            missing = scores[(scores["model"] == model)
                             & scores["patient_id"].astype(str).isin(missing_ids)
                             & scores["reason"].notna()]
            exclusion_reasons[model] = {
                str(reason): int(count)
                for reason, count in missing["reason"].value_counts().items()
            }
        if len(difference) == 0:
            results[name] = {"n_patients": 0, "estimate": None,
                             "ci95_percentile_patient_bootstrap": None,
                             "p_two_sided_sign_flip_plus_one": None,
                             "positive_patient_fraction": None,
                             "inner_fold_direction": {},
                             "excluded_patient_count": int(wide["patient_id"].nunique()),
                             "required_models": list(required_models),
                             "excluded_score_reasons": exclusion_reasons}
            raw_p[name] = 1.0
            continue
        bootstrap = np.asarray([
            difference[rng.integers(0, len(difference), len(difference))].mean()
            for _ in range(2000)
        ])
        observed = abs(float(difference.mean()))
        sign = np.asarray([
            abs(float(np.mean(difference * rng.choice([-1.0, 1.0], len(difference)))))
            for _ in range(10000)
        ])
        p_value = float((1 + np.count_nonzero(sign >= observed)) / 10001)
        raw_p[name] = p_value
        fold_direction = {}
        for fold, part in paired.assign(difference=difference).groupby("inner_fold"):
            fold_direction[str(int(fold))] = {
                "n_patients": int(len(part)),
                "mean_difference": float(part["difference"].mean()),
                "direction_positive": bool(part["difference"].mean() > 0),
            }
        results[name] = {
            "n_patients": int(len(difference)),
            "excluded_patient_count": int(wide["patient_id"].nunique() - len(difference)),
            "required_models": list(required_models),
            "excluded_score_reasons": exclusion_reasons,
            "estimate": float(difference.mean()),
            "ci95_percentile_patient_bootstrap": [
                float(np.quantile(bootstrap, 0.025)),
                float(np.quantile(bootstrap, 0.975)),
            ],
            "p_two_sided_sign_flip_plus_one": p_value,
            "positive_patient_fraction": float(np.mean(difference > 0)),
            "inner_fold_direction": fold_direction,
        }
    ordered = sorted(raw_p, key=raw_p.get)
    holm: dict[str, float] = {}
    running = 0.0
    m = len(ordered)
    for index, name in enumerate(ordered):
        adjusted = min(1.0, (m - index) * raw_p[name])
        running = max(running, adjusted)
        holm[name] = running
    for name in results:
        results[name]["p_holm_four_contrasts"] = holm.get(name)
    random_map_summary = {}
    for model in random_columns:
        part = wide[["P-H", model]].dropna()
        difference = part["P-H"].to_numpy(float) - part[model].to_numpy(float)
        random_map_summary[model] = {
            "n_patients": int(len(difference)),
            "P-H_minus_map_difference": float(difference.mean()) if len(difference) else None,
            "positive_patient_fraction": float(np.mean(difference > 0)) if len(difference) else None,
        }
    return {
        "patient_count_in_score_table": int(wide["patient_id"].nunique()),
        "paired_patient_denominators": denominator_counts,
        "registered_contrasts": results,
        "random_map_specific_descriptives": random_map_summary,
        "random_maps_are_not_independent_patient_units": True,
        "bootstrap_unit": "unique_patient",
        "bootstrap_replicates": 2000,
        "sign_flip_replicates": 10000,
        "multiplicity": "Holm across the four registered contrasts",
    }


def train_one(official: Any, model_name: str, pathway_map: AlignedPathwayMap | None,
              train_rows: pd.DataFrame, validation_rows: pd.DataFrame,
              expression_by_patient: dict[str, np.ndarray], drug_position: dict[str, int],
              drug_features: np.ndarray, *, device: torch.device, run_dir: Path,
              inner_index: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    seed = 20260922
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    backbone = official.CombinedModel(
        gene_input_size=19091, gene_hidden_size=8192, drug_hidden_size=256,
        feature_dim=128, nhead=8, num_layers=3, dim_feedforward=2048, dropout=0.1)
    parameter_count = sum(parameter.numel() for parameter in backbone.parameters())
    if model_name == "P-A":
        model = backbone
    else:
        arm = "P-C" if model_name == "P-C" else "P-R" if model_name.startswith("P-R-") else "P-T"
        model = build_exp046_arm_model(
            arm, backbone,
            ordinary_control_map=pathway_map if arm == "P-C" else None,
            frozen_random_map=pathway_map if arm == "P-R" else None,
            true_pathway_map=pathway_map if arm == "P-T" else None,
            rank=8, branch_seed=seed, zero_initialize=True,
        )

    def make_batch(frame: pd.DataFrame, *, labeled: bool):
        patients = tuple(frame["patient_id"].astype(str))
        drugs = tuple(frame["drug_id"].astype(str))
        x = np.stack([expression_by_patient[value] for value in patients]).astype(np.float32)
        d = drug_features[np.asarray([drug_position[value] for value in drugs])]
        tensors = (torch.from_numpy(x), torch.from_numpy(d), patients, drugs)
        if labeled:
            return tensors + (torch.from_numpy(frame["response"].to_numpy(dtype=np.float32)),)
        return tensors

    tx, td, tpatients, tdrugs, ty = make_batch(train_rows, labeled=True)
    vx, vd, vpatients, vdrugs, vy = make_batch(validation_rows, labeled=True)
    groups = [np.flatnonzero(np.asarray(tpatients) == patient)
              for patient in sorted(set(tpatients))]
    if any(len(index) > 16 for index in groups):
        raise RuntimeError("one patient's complete drug set exceeds the frozen batch size")
    counts = np.bincount(np.asarray([sorted(set(tpatients)).index(p) for p in tpatients]))
    weights = torch.as_tensor(1.0 / counts[[sorted(set(tpatients)).index(p) for p in tpatients]],
                              dtype=torch.float64)
    labels64 = ty.to(dtype=torch.float64)
    mean = float((weights * labels64).sum() / weights.sum())
    scale = float(torch.sqrt((weights * (labels64 - mean).square()).sum() / weights.sum()))
    if not math.isfinite(scale) or scale <= 1e-8:
        raise RuntimeError("training-side patient-weighted response scale is invalid")
    target = ((ty.to(dtype=torch.float64) - mean) / scale).to(dtype=torch.float32)
    model = model.to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=1e-4)
    best_score = -float("inf")
    best_epoch = 0
    stale = 0
    best_state: dict[str, torch.Tensor] | None = None
    started = time.perf_counter()
    for epoch in range(1, 101):
        if RUN_CAP <= 0:  # keeps the cap check visible at the outer training loop
            raise BudgetStop("invalid run cap")
        model.train()
        order = np.random.default_rng(seed + epoch).permutation(len(groups))
        for group_position in order:
            if time.perf_counter() - RUN_STARTED[0] >= TRAINING_STOP * 3600:
                raise BudgetStop("EXP-049 hard GPU wall-clock cap reached")
            ix = torch.as_tensor(groups[int(group_position)], dtype=torch.long)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                prediction = model(tx[ix].to(device), td[ix].to(device)).float().reshape(-1)
                observed = target[ix].to(device)
                regression = torch.nn.functional.smooth_l1_loss(prediction, observed)
                rank_loss = pairwise_ranking_loss(prediction, observed)
                loss = regression + 0.1 * rank_loss
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
        model.eval()
        predictions = []
        with torch.no_grad():
            for start in range(0, len(vy), 16):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    values = model(vx[start:start + 16].to(device),
                                   vd[start:start + 16].to(device)).float().reshape(-1)
                predictions.extend(values.cpu().numpy().astype(float).tolist())
        scores = []
        val_frame = validation_rows[["patient_id", "drug_id", "response"]].copy()
        val_frame["pred"] = predictions
        for _, group in val_frame.groupby("patient_id", sort=True):
            if group["drug_id"].nunique() >= 3 and group["response"].nunique() > 1 and group["pred"].nunique() > 1:
                scores.append(float(spearmanr(group["response"], group["pred"]).statistic))
        if not scores:
            raise RuntimeError("inner validation has no defined patient-level Spearman score")
        score = float(np.mean(scores))
        if score > best_score:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 10:
                break
    if best_state is None:
        raise RuntimeError("inner validation did not select a model epoch")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    final_prediction = []
    with torch.no_grad():
        for start in range(0, len(vy), 16):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                values = model(vx[start:start + 16].to(device),
                               vd[start:start + 16].to(device)).float().reshape(-1)
            final_prediction.extend(values.cpu().numpy().astype(float).tolist())
    elapsed = time.perf_counter() - started
    result = validation_rows[["patient_id", "drug_id", "response"]].copy()
    result["model"] = model_name
    result["prediction"] = final_prediction
    result["inner_fold"] = inner_index
    receipt = {"model": model_name, "inner_fold": inner_index,
               "train_patients": len(set(tpatients)),
               "validation_patients": len(set(vpatients)),
               "train_rows": len(train_rows), "validation_rows": len(validation_rows),
               "best_epoch": best_epoch, "epochs_run": epoch,
               "best_inner_validation_mean_spearman": best_score,
               "training_side_response_mean": mean,
               "training_side_response_scale": scale,
               "parameter_count": parameter_count,
               "model_initialization_seed": seed,
               "elapsed_seconds_excluding_process_io": elapsed,
               "checkpoint_written": False}
    del model, optimizer, best_state
    torch.cuda.empty_cache()
    return result, receipt


RUN_STARTED = [0.0]


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    auth = json.loads(args.authorization_json.read_text(encoding="utf-8"))
    if (auth.get("experiment_id") != "EXP-049"
            or auth.get("proposal_id") != PROPOSAL_ID
            or auth.get("status") != "USER_APPROVED"
            or auth.get("max_gpu_hours") != RUN_CAP
            or auth.get("shared_tranche_cap_gpu_hours") != SHARED_CAP
            or auth.get("authorized_gpu_uuid") != GPU_UUID
            or auth.get("protocol_sha256") != sha_file(CONFIG)
            or auth.get("runner_sha256") != sha_file(Path(__file__).resolve())
            or auth.get("source") != "direct_user_approval"):
        raise RuntimeError("valid direct user approval for this exact 1.25 GPUh suballocation is required")
    check_runtime_device()
    run_dir = ROOT / "artifacts" / "experiments" / "EXP-049" / args.run_id
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {run_dir}")
    attempt_id = f"{args.run_id}_attempt001"
    attempt_started = time.perf_counter()
    reserve_attempt(attempt_id, args.run_id, attempt_started)
    RUN_STARTED[0] = attempt_started
    device = torch.device("cuda:0")
    outcome = "NONZERO_EXIT"
    exit_code = 1
    alarm_installed = False

    def stop_at_cap(_signum: int, _frame: Any) -> None:
        raise BudgetStop("EXP-049 training stop deadline reached")

    signal.signal(signal.SIGALRM, stop_at_cap)
    remaining_training_seconds = max(
        0.0, TRAINING_STOP * 3600 - (time.perf_counter() - attempt_started))
    signal.setitimer(signal.ITIMER_REAL, remaining_training_seconds)
    alarm_installed = True
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
        preflight_start = time.perf_counter()
        check = verify_static_inputs()
        split = json.loads(SPLIT.read_text(encoding="utf-8"))
        fold = split["target_folds"][0]
        fraction = next(item for item in fold["label_fractions"] if item["fraction"] == 0.5)
        permitted = tuple(fraction["train_ids"])
        outer_test = set(fold["outer_test_ids"])
        if outer_test & set(permitted):
            raise RuntimeError("outer-test patient IDs intersect permitted model inputs")
        rows = load_allowed_rows(permitted)
        axis = json.loads((INPUT / "patient_axis.json").read_text(encoding="utf-8"))
        position = {str(value): i for i, value in enumerate(axis)}
        expression_file = np.load(INPUT / "patient_expression.npy", mmap_mode="r", allow_pickle=False)
        expression_by_patient = {patient: np.asarray(expression_file[position[patient]], dtype=np.float32)
                                 for patient in permitted}
        if set(expression_by_patient) != set(permitted):
            raise RuntimeError("expression inputs do not match allowed patients")
        drug_axis = pd.read_parquet(INPUT / "drug_axis.parquet", columns=["drug_index", "drug_id"])
        drug_axis = drug_axis.sort_values("drug_index", kind="stable")
        if drug_axis["drug_index"].astype(int).tolist() != list(range(14)):
            raise RuntimeError("frozen drug_index axis is not contiguous from 0 to 13")
        drug_ids = tuple(drug_axis["drug_id"].astype(str))
        drug_position = {drug: index for index, drug in enumerate(drug_ids)}
        drug_features = np.load(INPUT / "drug_ordinal128.npy", allow_pickle=False).astype(np.float32)
        raw_maps, genes, pathway_names = build_maps()
        map_by_model = {
            "P-C": raw_maps["P-C"], "P-H": raw_maps["P-H"],
            "P-R-20260922": raw_maps["P-R-20260922"],
            "P-R-20260923": raw_maps["P-R-20260923"],
            "P-R-20260924": raw_maps["P-R-20260924"],
        }
        if len(drug_ids) != 14 or drug_features.shape != (14, 128):
            raise RuntimeError("frozen drug-feature axis differs from EXP-036")
        inner_folds = fraction["inner_folds"]
        all_predictions: list[pd.DataFrame] = []
        fit_receipts: list[dict[str, Any]] = []
        partial_dir = run_dir / "partial_fits"
        partial_dir.mkdir(parents=True, exist_ok=False)
        official_path = ROOT / "data/external/exp010/pharmaformer/repo/pretrain.py"
        spec = importlib.util.spec_from_file_location("exp049_official_pharmaformer", official_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("official PharmaFormer architecture source is unavailable")
        official = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(official)
        for inner_index, inner in enumerate(inner_folds):
            train_ids = set(inner["train_ids"])
            validation_ids = set(inner["validation_ids"])
            if (train_ids | validation_ids) - set(permitted) or train_ids & validation_ids:
                raise RuntimeError("inner patient split violates outer-training boundary")
            train_rows = rows.loc[rows["patient_id"].isin(train_ids)].copy()
            val_rows = rows.loc[rows["patient_id"].isin(validation_ids)].copy()
            train_mean = train_rows.groupby("drug_id", sort=True)["response"].mean()
            val_rows["B0"] = val_rows["drug_id"].map(train_mean)
            model_names = ["P-A", "P-C", "P-H", "P-R-20260922",
                           "P-R-20260923", "P-R-20260924"]
            for model_name in model_names:
                result, receipt = train_one(
                    official, model_name, map_by_model.get(model_name),
                    train_rows, val_rows, expression_by_patient, drug_position,
                    drug_features, device=device, run_dir=run_dir,
                    inner_index=inner_index)
                safe_name = model_name.replace("-", "_")
                partial_path = partial_dir / f"inner_{inner_index}_{safe_name}.parquet"
                partial_tmp = partial_path.with_suffix(".parquet.tmp")
                result.to_parquet(partial_tmp, index=False)
                os.replace(partial_tmp, partial_path)
                receipt["partial_prediction_path"] = partial_path.relative_to(ROOT).as_posix()
                receipt["partial_prediction_sha256"] = sha_file(partial_path)
                receipt["protocol_sha256"] = sha_file(CONFIG)
                receipt["input_manifest_sha256"] = sha_file(INPUT / "manifest.json")
                receipt["split_sha256"] = sha_file(SPLIT)
                receipt["pathway_map_archive_sha256"] = sha_file(MAPS)
                receipt["runner_sha256"] = sha_file(Path(__file__).resolve())
                receipt["official_pharmaformer_source_sha256"] = sha_file(official_path)
                receipt["model_builder_sha256"] = sha_file(ROOT / "src/drug_screen/models/exp046_five_arm.py")
                receipt["pairwise_loss_source_sha256"] = sha_file(ROOT / "scripts/modeling/exp036_pharmaformer_fold.py")
                receipt_path = partial_dir / f"inner_{inner_index}_{safe_name}.receipt.json"
                receipt_tmp = receipt_path.with_suffix(".json.tmp")
                receipt_tmp.write_text(
                    json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")
                os.replace(receipt_tmp, receipt_path)
                all_predictions.append(result)
                fit_receipts.append(receipt)
                print(json.dumps({"event": "arm_inner_complete", "model": model_name,
                                  "inner_fold": inner_index, "fit_seconds": receipt["elapsed_seconds_excluding_process_io"]}),
                      flush=True)
            baseline = val_rows[["patient_id", "drug_id", "response", "B0"]].copy()
            baseline["model"] = "B0"
            baseline["prediction"] = baseline["B0"]
            baseline["inner_fold"] = inner_index
            all_predictions.append(baseline[["patient_id", "drug_id", "response", "model", "prediction", "inner_fold"]])
        predictions = pd.concat(all_predictions, ignore_index=True)
        if not np.isfinite(predictions["prediction"].to_numpy(dtype=float)).all():
            raise RuntimeError("one or more arm predictions are non-finite")
        value_cols = ["patient_id", "drug_id", "response", "inner_fold"]
        wide = predictions.pivot(index=value_cols, columns="model", values="prediction").reset_index()
        expected_models = ["P-A", "P-C", "P-H", "P-R-20260922", "P-R-20260923",
                          "P-R-20260924", "B0"]
        if any(model not in wide.columns for model in expected_models):
            raise RuntimeError("one or more registered arms are missing")
        scores = patient_scores(wide)
        contrasts = paired_statistics(scores)
        if len(fit_receipts) != 18:
            raise RuntimeError("the frozen workload requires exactly 18 completed fits")
        predictions_path = run_dir / "predictions.parquet"
        scores_path = run_dir / "patient_scores.parquet"
        predictions.to_parquet(predictions_path.with_suffix(".parquet.tmp"), index=False)
        os.replace(predictions_path.with_suffix(".parquet.tmp"), predictions_path)
        scores.to_parquet(scores_path.with_suffix(".parquet.tmp"), index=False)
        os.replace(scores_path.with_suffix(".parquet.tmp"), scores_path)
        summary = {
            "experiment_id": "EXP-049", "run_id": args.run_id,
            "status": "EXPLORATORY_INNER_VALIDATION_COMPARISON_COMPLETE_PENDING_INDEPENDENT_RECOMPUTATION",
            "linked_experiment": "EXP-046",
            "protocol_sha256": sha_file(CONFIG), "input_manifest_sha256": sha_file(INPUT / "manifest.json"),
            "split_sha256": sha_file(SPLIT), "map_archive_sha256": sha_file(MAPS),
            "runner_sha256": sha_file(Path(__file__).resolve()),
            "official_pharmaformer_source_sha256": sha_file(official_path),
            "model_builder_sha256": sha_file(ROOT / "src/drug_screen/models/exp046_five_arm.py"),
            "pairwise_loss_source_sha256": sha_file(ROOT / "scripts/modeling/exp036_pharmaformer_fold.py"),
            "outer_fold": 0, "label_fraction": 0.5,
            "allowed_training_patients": len(permitted),
            "outer_test_patients_in_model_process": 0,
            "outer_test_labels_loaded_by_model_process": False,
            "historical_coordinator_test_label_access": {
                "occurred_after_protocol_freeze": True,
                "loaded_complete_patient_pair_response_table": True,
                "response_values_printed_aggregated_scored_or_used_for_decision": False,
                "sol_ruling": "continue_inner_development_comparison_with_explicit_scope_disclosure; no split rerun",
                "ruling_record": "artifacts/experiments/EXP-049/20260927_sol_access_incident/decision.json",
            },
            "target_label_rows_loaded_for_allowed_patients": int(len(rows)),
            "model_initialization_seed": 20260922,
            "fit_count": len(fit_receipts), "fit_receipts": fit_receipts,
            "partial_fit_artifacts_directory": partial_dir.relative_to(ROOT).as_posix(),
            "primary_metric": "patient_equal_within_patient_spearman_on_common_observed_drugs",
            "patient_scores_path": scores_path.relative_to(ROOT).as_posix(),
            "patient_scores_sha256": sha_file(scores_path),
            "paired_statistics": contrasts,
            "predictions_path": predictions_path.relative_to(ROOT).as_posix(),
            "predictions_sha256": sha_file(predictions_path),
            "runtime_device": torch.cuda.get_device_name(0), "runtime_gpu_uuid": GPU_UUID,
            "process_wall_seconds_before_summary": time.perf_counter() - attempt_started,
            "inner_validation_used_for_early_stopping": True,
            "interpretation": "exploratory only; not an outer-test estimate, source-transfer result, or independent generalization evidence",
        }
        summary_path = run_dir / "run_summary.json"
        temp = summary_path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
        os.replace(temp, summary_path)
        check["preflight_wall_seconds"] = time.perf_counter() - preflight_start
        outcome, exit_code = "EXIT_0", 0
        return summary
    finally:
        if alarm_installed:
            signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize(torch.device("cuda:0"))
        except Exception:
            outcome, exit_code = "CUDA_SYNC_FAILURE", 1
        charge = close_attempt(attempt_id, attempt_started, exit_code, outcome)
        if run_dir.exists():
            (run_dir / "attempt_close.json").write_text(json.dumps({
                "attempt_id": attempt_id, "outcome": outcome, "exit_code": exit_code,
                "gpu_hours_charged": charge, "gpu_uuid": GPU_UUID,
                "ended_utc": utc_now(), "automatic_retry": False,
            }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--include-allowed-training-responses", action="store_true")
    parser.add_argument("--authorization-json", type=Path)
    parser.add_argument("--run-id", default="20260927_innerpathway_outer0_v1")
    parser.add_argument("--_training-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.preflight_only:
        started = time.perf_counter()
        result = verify_static_inputs()
        aligned_maps, genes, pathway_names = build_maps()
        map_contract: dict[str, dict[str, Any]] = {}
        for name, item in aligned_maps.items():
            shape = tuple(int(value) for value in item.incidence_matrix.shape)
            edge_count = int(torch.count_nonzero(item.incidence_matrix).item())
            if shape != (50, 19091) or len(item.mapped_gene_ids) != 4366 or edge_count != 7299:
                raise RuntimeError(f"frozen pathway topology differs for {name}")
            map_contract[name] = {"shape": list(shape), "mapped_genes": len(item.mapped_gene_ids),
                                  "pathways": len(item.pathway_names), "edge_count": edge_count}
        model_parameters = validate_model_structure(aligned_maps)
        result["pathway_map_contract"] = map_contract
        result["official_cold_start_model_parameter_counts"] = model_parameters
        result["mapped_gene_axis_count"] = len(genes)
        result["pathway_count"] = len(pathway_names)
        if args.include_allowed_training_responses:
            split = json.loads(SPLIT.read_text(encoding="utf-8"))
            fraction = next(item for item in split["target_folds"][0]["label_fractions"]
                            if item["fraction"] == 0.5)
            permitted = tuple(fraction["train_ids"])
            rows = load_allowed_rows(permitted)
            if set(rows["patient_id"]) != set(permitted):
                raise RuntimeError("allowed response preflight returned an incomplete development cohort")
            result["target_label_values_loaded"] = True
            result["allowed_response_rows"] = int(len(rows))
            result["allowed_response_patient_count"] = int(rows["patient_id"].nunique())
            result["allowed_observed_drug_count_min_max"] = [
                int(rows.groupby("patient_id")["drug_id"].nunique().min()),
                int(rows.groupby("patient_id")["drug_id"].nunique().max()),
            ]
            result["response_values_printed_or_scored"] = False
        else:
            result["target_label_values_loaded"] = False
        result["cuda_initialized_after_model_structure"] = torch.cuda.is_initialized()
        if result["cuda_initialized_after_model_structure"]:
            raise RuntimeError("CPU-only preflight unexpectedly initialized CUDA")
        result["preflight_wall_seconds"] = time.perf_counter() - started
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.authorization_json is None:
        parser.error("formal training requires --authorization-json from direct user approval")
    if not args._training_child:
        supervisor_started = time.perf_counter()
        child_command = [
            sys.executable, str(Path(__file__).resolve()),
            "--authorization-json", str(args.authorization_json.resolve()),
            "--run-id", args.run_id, "--_training-child",
        ]
        child = subprocess.Popen(child_command, cwd=ROOT, env=os.environ.copy(),
                                 start_new_session=True)
        timed_out = False

        def terminate_child_on_signal(_signum: int, _frame: Any) -> None:
            raise KeyboardInterrupt("EXP-049 supervisor interrupted; stopping the training child")

        signal.signal(signal.SIGTERM, terminate_child_on_signal)
        signal.signal(signal.SIGINT, terminate_child_on_signal)
        try:
            remaining = max(0.0, RUN_CAP * 3600 - (time.perf_counter() - supervisor_started))
            exit_code = child.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            timed_out = True
            child.kill()
            child.wait()
            exit_code = 124
        except BaseException:
            try:
                if child.poll() is None:
                    child.kill()
            finally:
                child.wait()
            stopped = time.perf_counter()
            finalize_unclosed_attempt(
                f"{args.run_id}_attempt001", 130,
                "SUPERVISOR_INTERRUPTED_CHILD_STOPPED_NO_RETRY", stopped)
            raise

        stopped = time.perf_counter()
        if timed_out or exit_code != 0:
            outcome = ("SUPERVISOR_HARD_DEADLINE_FORCED_STOP_NO_RETRY" if timed_out
                       else "CHILD_EXIT_WITH_UNCLOSED_RESERVATION_NO_RETRY")
            effective_exit = exit_code if exit_code != 0 else 1
            finalize_unclosed_attempt(f"{args.run_id}_attempt001", effective_exit,
                                      outcome, stopped)
        if timed_out:
            print(json.dumps({
                "status": "WATCHDOG_STOPPED",
                "run_id": args.run_id,
                "exit_code": exit_code,
                "supervisor_elapsed_seconds": stopped - supervisor_started,
                "automatic_retry": False,
            }, ensure_ascii=False), flush=True)
            raise SystemExit(exit_code)
        raise SystemExit(exit_code)

    result = run_training(args)
    print(json.dumps({"status": result["status"], "run_id": result["run_id"],
                      "fit_count": result["fit_count"], "gpu_hours_charged_from_ledger": True},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
