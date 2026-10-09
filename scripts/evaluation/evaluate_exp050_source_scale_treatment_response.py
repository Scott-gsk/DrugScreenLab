"""CPU-only, development-stage adapter-to-TranSiGen response comparison.

The only scored rows are the frozen 537 MDAMB231 inner-development condition
keys. The existing affine adapter was fitted with MDAMB231 as a source-training
origin, and TranSiGen's epoch-13 checkpoint was selected using this same full
inner-development partition. Results are therefore descriptive development
evidence only, not independent validation or generalization evidence.

Run once with ``--phase preflight`` to freeze response-blind keys and contracts,
then with ``--phase evaluate`` to read only permitted train deltas and those
537 selected development deltas and run CPU inference. This entry point never
opens the locked-test metadata or any outer/PDO response asset.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "20261001_source_scale_treatment_response_v1"
OUT = ROOT / "artifacts/experiments/EXP-050" / RUN_ID
PREPARED = ROOT / "artifacts/experiments/EXP-037/20260928_lincs_scale_16_121/prepared"
ADAPTER = ROOT / "artifacts/experiments/EXP-050/20261001_source_scale_mapping_v1"
TRANSIGEN = ROOT / "artifacts/experiments/EXP-037/20260928_lincs_scale_16_121/transigen_121"
DEPMAP_DIR = ROOT / "data/processed/depmap/24q2_rnaseq_exact978"
LINCS_GENES = ROOT / "data/processed/lincs/GSE92742/continuous_contract_v2/genes.json"
DEFAULT_OFFICIAL_SOURCE = Path("/mnt/d/Code/TranSiGen/参考代码/TranSiGen/src")
TARGET_CELL = "MDAMB231"
N_GENES = 978
N_DRUG_FEATURES = 2048
PREDICTION_SEED = 20260922
EXPECTED_CHECKPOINT_SHA256 = "717b6d611d6375feac7fab15440bae552c7a05ca8e7c6a9ca60289b33332aaaa"
EXPECTED_ADAPTER_SHA256 = "e564c3b7f6c725fd7f3d9f5554655641cef7fdb2974a21a81387d24f6b4a11c5"
EXPECTED_PREPARED_MANIFEST_SHA256 = "6c2ba028fd54e7d42cdf62d523a51c02158e2ead919f77dbd552809473f54a04"
EXPECTED_GENE_AXIS_SHA256 = "b4e2fca877c5cfdcc1c712ad0fd67e97a88b6f7566b013e4bab065f699ebb623"
EXPECTED_TRANSiGEN_SOURCE = {
    "model.py": "a96706b11b29c8be3122640e9a05a54a1e2210039f3fd8a30b0f664d62cf7c0f",
    "utils.py": "54affbf62ff7bcdaa7b87696d94577120743be8dd9e264886b475c30cf58c160",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_write(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def require_runtime() -> None:
    if sys.platform != "linux" or os.environ.get("CONDA_DEFAULT_ENV") != "drugscreening-gpu":
        raise RuntimeError("This evaluation must run in WSL2 Conda drugscreening-gpu")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CUDA must be hidden; this run is CPU-only")
    torch.set_num_threads(2)


def load_official_model(source: Path) -> torch.nn.Module:
    if not source.is_dir():
        raise FileNotFoundError(f"official TranSiGen source directory is missing: {source}")
    for name, expected in EXPECTED_TRANSiGEN_SOURCE.items():
        actual = sha256_file(source / name)
        if actual != expected:
            raise ValueError(f"TranSiGen {name} checksum mismatch: {actual}")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location("exp050_official_transigen", source / "model.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import official TranSiGen from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    device = torch.device("cpu")
    torch.manual_seed(20260928)
    model = module.TranSiGen(
        N_GENES, 100, [1200], [800], N_DRUG_FEATURES, [400],
        device=device, init_w=True, beta=0.1, dropout=0.1, random_seed=20260928,
    ).to(device)
    state = torch.load(TRANSIGEN / "transigen_best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def read_response_blind_inputs(official_source: Path) -> dict[str, Any]:
    """Verify contracts and selected keys without opening any response array."""
    for path in [
        PREPARED / "manifest.json", PREPARED / "conditions.parquet",
        PREPARED / "train121.npy", PREPARED / "validation.npy",
        PREPARED / "gene_scale.npy", PREPARED / "source_to_target.npy",
        ADAPTER / "model_parameters.npz", ADAPTER / "source_object_crosswalk.tsv",
        ADAPTER / "summary.json", ADAPTER / "run_manifest.json",
        TRANSIGEN / "transigen_best.pt", TRANSIGEN / "transigen_preprocess.npz",
        TRANSIGEN / "summary.json", TRANSIGEN / "scope.json", LINCS_GENES,
        DEPMAP_DIR / "ccle_24q2_exact978_log2tpm.npy",
    ]:
        if not path.is_file():
            raise FileNotFoundError(f"required registered input is missing: {path}")

    manifest_path = PREPARED / "manifest.json"
    prepared_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (prepared_manifest.get("status") != "PREPARED_TRAIN_VALIDATION_ONLY"
            or prepared_manifest.get("official_test_responses_read") is not False
            or sha256_file(manifest_path) != EXPECTED_PREPARED_MANIFEST_SHA256):
        raise ValueError("prepared manifest is not the frozen train/inner-development asset")
    file_hashes = prepared_manifest["files_sha256"]
    for name, expected in {
        "conditions.parquet": "f9a5ddc02b13ffc282dab3725de90fa1d17a5f99cafba0722bad880871ccc79a",
        "train121.npy": "96920b6993862cc80da2d910aa790a2f5266aa9a3c630bdea3a0331f056b5f8d",
        "validation.npy": "7fc77d0b51b45f9994686349c81c9e52595f746859fc8da8af57f750ea0b00a3",
        "delta.npy": "060404d11d54cd0e0e2d822fac74d78e1ef33d863a9b60806e55ddc7d759b424",
        "ecfp4.npy": "cb3129ffd12677cb63d8e89cf5b3d15a2f05064940aff998b3190aaae8d3539e",
        "gene_scale.npy": "1f8b2640e752dde5ed75f7fa15f74a95273ee1b9540da85ab5c14388371d4e50",
    }.items():
        if file_hashes.get(name) != expected:
            raise ValueError(f"prepared manifest {name} hash does not match the frozen contract")

    columns = ["condition_id", "cell_id", "pert_id", "pert_dose", "pert_time"]
    conditions = pd.read_parquet(PREPARED / "conditions.parquet", columns=columns)
    if conditions["condition_id"].astype(str).duplicated().any():
        raise ValueError("prepared condition_id keys are not unique")
    train_indices = np.load(PREPARED / "train121.npy", allow_pickle=False)
    dev_indices = np.load(PREPARED / "validation.npy", allow_pickle=False)
    if (train_indices.ndim != 1 or dev_indices.ndim != 1
            or train_indices.dtype.kind not in "iu" or dev_indices.dtype.kind not in "iu"
            or len(train_indices) != 32094 or len(dev_indices) != 4954
            or len(np.unique(train_indices)) != len(train_indices)
            or len(np.unique(dev_indices)) != len(dev_indices)
            or np.intersect1d(train_indices, dev_indices).size
            or max(int(train_indices.max()), int(dev_indices.max())) >= len(conditions)):
        raise ValueError("prepared train/inner-development row-index contract failed")
    train_keys = conditions.iloc[train_indices].reset_index(drop=True)
    dev_keys = conditions.iloc[dev_indices].copy()
    dev_keys["prepared_row_index"] = dev_indices
    if set(train_keys["cell_id"].astype(str)) & set(dev_keys["cell_id"].astype(str)):
        raise ValueError("TranSiGen train and inner-development cells overlap")

    adapter_params_path = ADAPTER / "model_parameters.npz"
    adapter_sha = sha256_file(adapter_params_path)
    if adapter_sha != EXPECTED_ADAPTER_SHA256:
        raise ValueError(f"selected affine adapter checksum mismatch: {adapter_sha}")
    with np.load(adapter_params_path, allow_pickle=False) as params:
        gene_ids = params["gene_ids"].astype(str)
        selected_candidate = str(params["selected_candidate"].item())
        intercept = np.asarray(params["candidate_1_intercept"], dtype=np.float64)
        slope = np.asarray(params["candidate_1_slope"], dtype=np.float64)
    if selected_candidate != "candidate_1_affine" or intercept.shape != (N_GENES,) or slope.shape != (N_GENES,):
        raise ValueError("frozen affine adapter parameters have unexpected shape or candidate")
    lincs_gene_ids = np.asarray(json.loads(LINCS_GENES.read_text(encoding="utf-8")), dtype=str)
    axis_hash = hashlib.sha256(("\n".join(lincs_gene_ids.tolist()) + "\n").encode()).hexdigest()
    if (len(lincs_gene_ids) != N_GENES or len(set(lincs_gene_ids)) != N_GENES
            or axis_hash != EXPECTED_GENE_AXIS_SHA256 or not np.array_equal(gene_ids, lincs_gene_ids)):
        raise ValueError("adapter and frozen LINCS gene axes are not identical")

    crosswalk = pd.read_csv(ADAPTER / "source_object_crosswalk.tsv", sep="\t", dtype=str)
    target_origin = crosswalk.loc[
        crosswalk["cell_line_name"].str.upper().eq(TARGET_CELL)
        & crosswalk["split_role"].eq("train")
    ]
    if len(target_origin) != 1:
        raise ValueError("MDAMB231 must resolve to one frozen adapter-training source origin")
    depmap_row = int(target_origin.iloc[0]["exact978_row"])
    depmap_id = str(target_origin.iloc[0]["depmap_id"])
    source_origin_id = str(target_origin.iloc[0]["patient_id_source_origin"])

    dev_keys["cell_id_upper"] = dev_keys["cell_id"].astype(str).str.upper()
    selected = dev_keys.loc[dev_keys["cell_id_upper"].eq(TARGET_CELL)].copy()
    selected.drop(columns=["cell_id_upper"], inplace=True)
    selected["condition_id"] = pd.to_numeric(selected["condition_id"], errors="raise").astype(np.int64)
    selected = selected.sort_values("condition_id", kind="stable").reset_index(drop=True)
    if len(selected) != 537 or selected["condition_id"].duplicated().any():
        raise ValueError(f"expected 537 unique MDAMB231 inner-dev condition keys, got {len(selected)}")
    if not selected["pert_dose"].map(np.isfinite).all() or not selected["pert_time"].map(np.isfinite).all():
        raise ValueError("selected condition keys contain non-finite dose/time metadata")

    train_drugs = set(train_keys["pert_id"].astype(str))
    selected["same_drug_mean_supported"] = selected["pert_id"].astype(str).isin(train_drugs)
    if selected["cell_id"].astype(str).str.upper().nunique() != 1:
        raise ValueError("selected development keys span more than one cell identity")

    adapter_validation_cells = set(crosswalk.loc[crosswalk["split_role"].eq("val"), "cell_line_name"].str.upper())
    strict_dev = dev_keys.loc[dev_keys["cell_id"].astype(str).str.upper().isin(adapter_validation_cells)]
    strict_common = len(strict_dev)

    source_to_target = np.load(PREPARED / "source_to_target.npy", mmap_mode="r", allow_pickle=False)
    if (source_to_target.shape != (N_GENES,) or len(np.unique(source_to_target)) != N_GENES
            or int(source_to_target.min()) != 0 or int(source_to_target.max()) != N_GENES - 1):
        raise ValueError("prepared source-to-target gene mapping is not a 978-gene permutation")
    with np.load(TRANSIGEN / "transigen_preprocess.npz", allow_pickle=False) as preproc:
        model_gene_scale = np.asarray(preproc["gene_scale"], dtype=np.float32)
        state_mean = np.asarray(preproc["state_mean"], dtype=np.float32)
        state_scale = np.asarray(preproc["state_scale"], dtype=np.float32)
    prepared_gene_scale = np.load(PREPARED / "gene_scale.npy", mmap_mode="r", allow_pickle=False)
    if (model_gene_scale.shape != (N_GENES,) or state_mean.shape != (N_GENES,)
            or state_scale.shape != (N_GENES,) or not np.array_equal(model_gene_scale, prepared_gene_scale)):
        raise ValueError("checkpoint preprocessing vectors do not match the prepared 978-gene scale")

    transigen_summary = json.loads((TRANSIGEN / "summary.json").read_text(encoding="utf-8"))
    transigen_scope = json.loads((TRANSIGEN / "scope.json").read_text(encoding="utf-8"))
    if (int(transigen_summary.get("selected_epoch", -1)) != 13
            or transigen_scope.get("official_test_responses_read") is not False):
        raise ValueError("TranSiGen checkpoint is not the frozen epoch-13 train/development model")
    checkpoint_path = TRANSIGEN / "transigen_best.pt"
    checkpoint_sha = sha256_file(checkpoint_path)
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(f"TranSiGen checkpoint checksum mismatch: {checkpoint_sha}")
    adapter_summary = json.loads((ADAPTER / "summary.json").read_text(encoding="utf-8"))
    if adapter_summary.get("selected_candidate_from_training_cv_only") != "candidate_1_affine":
        raise ValueError("adapter candidate was not selected from training CV")

    # Verify exact source/model loading before any response array is accessed.
    model = load_official_model(official_source)
    model_shapes = {key: list(value.shape) for key, value in model.state_dict().items()}
    del model

    selected["adapter_role"] = "train"
    selected["source_origin_id"] = source_origin_id
    selected["depmap_id"] = depmap_id
    key_columns = [
        "prepared_row_index", "condition_id", "cell_id", "pert_id", "pert_dose", "pert_time",
        "adapter_role", "source_origin_id", "depmap_id", "same_drug_mean_supported",
    ]
    selected = selected[key_columns]

    return {
        "selected_keys": selected,
        "snapshot": {
            "experiment_id": "EXP-050",
            "run_id": RUN_ID,
            "protocol_status": "FROZEN_DEVELOPMENT_ONLY",
            "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "question": "On the exact MDAMB231 inner-development conditions, does applying the already-frozen DepMap-to-LINCS affine map to the DepMap basal profile change the frozen TranSiGen treatment-response prediction error?",
            "analysis_unit": "condition_id within one cell_id; 978 genes are repeated response features, not independent observations",
            "selected_cell_id": TARGET_CELL,
            "n_selected_conditions": int(len(selected)),
            "n_unique_cell_ids": 1,
            "n_unique_condition_ids": int(selected["condition_id"].nunique()),
            "n_unique_pert_id": int(selected["pert_id"].nunique()),
            "same_drug_mean_supported_conditions": int(selected["same_drug_mean_supported"].sum()),
            "adapter_source_origin_id": source_origin_id,
            "adapter_role_for_selected_object": "train; MDAMB231 basal profile contributed to fitting the all-13-object affine adapter",
            "adapter": {
                "run_id": "20261001_source_scale_mapping_v1",
                "candidate": selected_candidate,
                "parameters_sha256": adapter_sha,
                "training_cv_selected": True,
                "parameters_fitted_on_13_train_source_objects": True,
            },
            "strict_adapter_validation_overlap": {
                "n_inner_dev_condition_keys": int(strict_common),
                "n_cell_ids": int(strict_dev["cell_id"].nunique()),
                "status": "LOCAL_NO_GO_ZERO_OBJECTS_ZERO_KEYS",
                "reason": "BT20 has no prepared TranSiGen conditions and HEPG2 has no inner-development conditions; the approved comparison is limited to the development-only MDAMB231 overlap.",
            },
            "model": {
                "architecture": "official TranSiGen",
                "arm": "transigen_121",
                "checkpoint": "artifacts/experiments/EXP-037/20260928_lincs_scale_16_121/transigen_121/transigen_best.pt",
                "checkpoint_sha256": checkpoint_sha,
                "selected_epoch": 13,
                "checkpoint_was_selected_using_same_full_inner_dev_partition": True,
                "official_source": str(official_source),
                "official_source_sha256": EXPECTED_TRANSiGEN_SOURCE,
                "strict_state_dict_load": True,
                "state_dict_shapes": model_shapes,
            },
            "gene_axis": {
                "n_genes": N_GENES,
                "ordered_gene_ids_sha256": axis_hash,
                "adapter_equals_lincs_gene_axis": True,
                "model_input_dim": N_GENES,
                "prepared_gene_scale_equals_checkpoint_gene_scale": True,
                "source_to_target_is_permutation": True,
                "note": "Prepared assets are registered on the exact 978-gene model axis; source_to_target is provenance for the upstream reordering and is not applied a second time.",
            },
            "arms": {
                "raw_input": "Frozen MDAMB231 DepMap log2(TPM+1) basal profile, passed unchanged as TranSiGen x1.",
                "mapped_input": "The same profile transformed by the frozen candidate_1_affine intercept+slope, passed as TranSiGen x1.",
                "response_prediction": "TranSiGen predicted treatment expression minus the arm-specific x1; same 2048-bit prepared drug features and fixed prediction seed for both arms.",
                "baselines": [
                    "zero-change delta",
                    "per-pert_id mean delta fit only on the 32,094 frozen train121 condition_id keys",
                ],
            },
            "metrics": {
                "primary": "per-condition mean squared error normalized by the frozen 978-gene training delta scale, then mean equally over condition keys",
                "additional": ["raw delta RMSE", "raw delta MAE", "per-condition gene-wise Pearson and Spearman", "signed cosine", "gene-wise sign agreement", "absolute-response magnitude bias and ratio", "predicted-vs-observed condition-level response-magnitude correlation"],
                "paired_comparison": "mapped minus raw metrics on identical condition_id keys; no p-values or intervals because support is one source cell and dev was used for early stopping",
                "same_drug_common_denominator": "All arms also scored on the subset supported by train-only per-drug means; unsupported drug keys retained for raw-input, mapped-input, and zero-change full-support reporting.",
            },
            "data_access": {
                "allowed": ["32,094 train121 delta rows for the train-only per-drug mean", "537 selected MDAMB231 inner-development delta rows", "the same 537 prepared ECFP4 feature rows", "one registered MDAMB231 DepMap baseline expression row"],
                "not_read": ["outer-fold responses", "locked_eval values or metadata", "PDO treated labels", "prepared control.npy"],
                "gpu_hours": 0,
                "device": "CPU",
            },
            "interpretation_boundary": "Development-only descriptive comparison. MDAMB231 participated in adapter fitting, and the same full inner-development partition informed TranSiGen epoch selection. No independent validation, source generalization, PDO response, drug efficacy, or clinical claim.",
            "preflight_inputs": {
                "prepared_manifest_sha256": EXPECTED_PREPARED_MANIFEST_SHA256,
                "prepared_registered_files_sha256": file_hashes,
                "adapter_parameters_sha256": adapter_sha,
                "transigen_checkpoint_sha256": checkpoint_sha,
                "transigen_preprocess_sha256": sha256_file(TRANSIGEN / "transigen_preprocess.npz"),
                "depmap_expression_registered_sha256": "36ab8a0f8f65df2cdb2c660f2f4ae7c92e26fc7a06ad18ff32bdd0ef11fff9bd",
                "depmap_matrix_row": depmap_row,
                "depmap_model_id": depmap_id,
            },
        },
        "train_indices": train_indices,
        "train_keys": train_keys,
        "dev_indices": dev_indices,
        "dev_conditions": conditions,
        "depmap_row": depmap_row,
        "adapter_intercept": intercept,
        "adapter_slope": slope,
        "gene_scale": model_gene_scale,
        "prepared_manifest": prepared_manifest,
    }


def run_preflight(official_source: Path) -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite run directory: {OUT}")
    audit = read_response_blind_inputs(official_source)
    OUT.mkdir(parents=True, exist_ok=False)
    keys_path = OUT / "keyed_overlap.tsv"
    audit["selected_keys"].to_csv(keys_path, sep="\t", index=False, lineterminator="\n")
    snapshot = audit["snapshot"]
    snapshot["keyed_overlap_sha256"] = sha256_file(keys_path)
    json_write(OUT / "protocol_snapshot.json", snapshot)
    json_write(OUT / "run_manifest.json", {
        "experiment_id": "EXP-050",
        "run_id": RUN_ID,
        "status": "PREFLIGHT_PASSED_RESPONSE_BLIND",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "python_executable": sys.executable,
            "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "torch": torch.__version__,
            "torch_cuda_available": bool(torch.cuda.is_available()),
            "device": "cpu",
            "torch_threads": torch.get_num_threads(),
        },
        "preflight": {
            "response_values_read": False,
            "strict_state_dict_load": True,
            "selected_condition_keys": int(len(audit["selected_keys"])),
            "keyed_overlap_sha256": snapshot["keyed_overlap_sha256"],
            "protocol_snapshot_sha256": sha256_file(OUT / "protocol_snapshot.json"),
        },
        "gpu_hours": 0,
        "inputs": snapshot["preflight_inputs"],
        "outputs": {
            "protocol_snapshot.json": sha256_file(OUT / "protocol_snapshot.json"),
            "keyed_overlap.tsv": snapshot["keyed_overlap_sha256"],
        },
    })
    print(json.dumps({
        "status": "PREFLIGHT_PASSED_RESPONSE_BLIND",
        "run_dir": str(OUT),
        "n_condition_keys": snapshot["n_selected_conditions"],
        "n_cell_ids": snapshot["n_unique_cell_ids"],
        "same_drug_mean_supported": snapshot["same_drug_mean_supported_conditions"],
        "strict_adapter_validation_keys": snapshot["strict_adapter_validation_overlap"]["n_inner_dev_condition_keys"],
        "labels_read": False,
        "gpu_hours": 0,
    }, ensure_ascii=False), flush=True)


def safe_corr(left: np.ndarray, right: np.ndarray, *, rank: bool = False) -> float | None:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    left, right = left[valid], right[valid]
    if left.size < 2 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return None
    if rank:
        left, right = rankdata(left), rankdata(right)
    value = float(np.corrcoef(left, right)[0, 1])
    return value if np.isfinite(value) else None


def per_condition_metrics(truth: np.ndarray, prediction: np.ndarray, scale: np.ndarray) -> list[dict[str, Any]]:
    output = []
    for y, pred in zip(truth, prediction, strict=True):
        err = pred.astype(np.float64) - y.astype(np.float64)
        y64, p64 = y.astype(np.float64), pred.astype(np.float64)
        abs_y = float(np.mean(np.abs(y64)))
        abs_p = float(np.mean(np.abs(p64)))
        denom = float(np.linalg.norm(y64) * np.linalg.norm(p64))
        output.append({
            "nmse": float(np.mean(np.square(err / scale))),
            "rmse": float(np.sqrt(np.mean(np.square(err)))),
            "mae": float(np.mean(np.abs(err))),
            "mean_signed_error": float(np.mean(err)),
            "pearson_gene": safe_corr(y64, p64),
            "spearman_gene": safe_corr(y64, p64, rank=True),
            "signed_cosine": float(np.dot(y64, p64) / denom) if denom > 0 else None,
            "sign_agreement": float(np.mean(np.sign(y64) == np.sign(p64))),
            "observed_mean_abs_delta": abs_y,
            "predicted_mean_abs_delta": abs_p,
            "absolute_magnitude_bias": abs_p - abs_y,
            "absolute_magnitude_ratio": abs_p / abs_y if abs_y > 0 else None,
        })
    return output


def aggregate_metrics(truth: np.ndarray, prediction: np.ndarray, scale: np.ndarray) -> dict[str, Any]:
    row_metrics = per_condition_metrics(truth, prediction, scale)
    fields = [
        "nmse", "rmse", "mae", "mean_signed_error", "pearson_gene", "spearman_gene",
        "signed_cosine", "sign_agreement", "observed_mean_abs_delta",
        "predicted_mean_abs_delta", "absolute_magnitude_bias", "absolute_magnitude_ratio",
    ]
    summary: dict[str, Any] = {"n_condition_keys": int(truth.shape[0]), "n_genes": int(truth.shape[1])}
    for field in fields:
        values = np.asarray([row[field] for row in row_metrics if row[field] is not None], dtype=np.float64)
        summary[field] = float(values.mean()) if values.size else None
        summary[f"{field}_median"] = float(np.median(values)) if values.size else None
        summary[f"{field}_defined_condition_keys"] = int(values.size)
    summary["global_mean_absolute_error"] = float(np.mean(np.abs(prediction.astype(np.float64) - truth.astype(np.float64))))
    summary["global_root_mean_squared_error"] = float(np.sqrt(np.mean(np.square(prediction.astype(np.float64) - truth.astype(np.float64)))))
    obs_mag = np.mean(np.abs(truth.astype(np.float64)), axis=1)
    pred_mag = np.mean(np.abs(prediction.astype(np.float64)), axis=1)
    summary["condition_magnitude_pearson"] = safe_corr(obs_mag, pred_mag)
    summary["condition_magnitude_spearman"] = safe_corr(obs_mag, pred_mag, rank=True)
    summary["condition_magnitude_correlation_defined_keys"] = int(len(obs_mag)) if summary["condition_magnitude_spearman"] is not None else 0
    summary["global_absolute_magnitude_bias"] = float(np.mean(np.abs(prediction)) - np.mean(np.abs(truth)))
    ratio_denominator = float(np.mean(np.abs(truth)))
    summary["global_absolute_magnitude_ratio"] = float(np.mean(np.abs(prediction)) / ratio_denominator) if ratio_denominator > 0 else None
    return summary


def predict_delta(model: torch.nn.Module, x1: np.ndarray, features: np.ndarray) -> np.ndarray:
    if x1.ndim != 2 or x1.shape[1] != N_GENES or features.shape != (x1.shape[0], N_DRUG_FEATURES):
        raise ValueError(f"prediction input shape mismatch: x1={x1.shape}, features={features.shape}")
    chunks = []
    model.eval()
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(PREDICTION_SEED)
        # Match the original evaluation batch-size convention while restricting
        # execution to the frozen MDAMB231 key set.
        n_chunks = max(1, int(np.ceil(len(x1) / 256)))
        for row_idx in np.array_split(np.arange(len(x1)), n_chunks):
            if not len(row_idx):
                continue
            x = torch.as_tensor(x1[row_idx], dtype=torch.float32, device="cpu")
            f = torch.as_tensor(features[row_idx], dtype=torch.float32, device="cpu")
            treatment = model(x, f)[3]
            predicted_delta = treatment - x
            if not torch.isfinite(predicted_delta).all():
                raise ValueError("TranSiGen produced non-finite response prediction")
            chunks.append(predicted_delta.cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)


def run_evaluation(official_source: Path) -> None:
    require_runtime()
    snapshot_path = OUT / "protocol_snapshot.json"
    keys_path = OUT / "keyed_overlap.tsv"
    manifest_path = OUT / "run_manifest.json"
    if not snapshot_path.is_file() or not keys_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("response-blind preflight receipts must exist before evaluation")
    manifest_receipt = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest_receipt.get("status") not in {
            "PREFLIGHT_PASSED_RESPONSE_BLIND", "FAILED_DURING_CPU_EVALUATION"}
            or manifest_receipt.get("preflight", {}).get("response_values_read") is not False):
        raise ValueError("preflight receipt is not a passing response-blind snapshot")
    if sha256_file(keys_path) != manifest_receipt["preflight"]["keyed_overlap_sha256"]:
        raise ValueError("frozen keyed overlap table was changed after preflight")

    audit = read_response_blind_inputs(official_source)
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if (snapshot.get("keyed_overlap_sha256") != sha256_file(keys_path)
            or len(audit["selected_keys"]) != snapshot.get("n_selected_conditions")
            or audit["snapshot"]["preflight_inputs"] != snapshot["preflight_inputs"]):
        raise ValueError("response-blind input/key contract changed after preflight")
    selected_keys = pd.read_csv(keys_path, sep="\t")
    if (len(selected_keys) != 537 or selected_keys["condition_id"].duplicated().any()
            or set(selected_keys["cell_id"].astype(str).str.upper()) != {TARGET_CELL}):
        raise ValueError("frozen key table no longer contains the expected 537 MDAMB231 dev keys")

    start = time.perf_counter()
    run_start_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    failed_attempts = list(manifest_receipt.get("failed_attempts", []))
    if manifest_receipt.get("status") == "FAILED_DURING_CPU_EVALUATION":
        failed_attempts.append({
            "attempt": len(failed_attempts) + 1,
            "status": "FAILED_DURING_CPU_EVALUATION",
            "run_started_at_utc": manifest_receipt.get("run_started_at_utc"),
            "failed_at_utc": manifest_receipt.get("failed_at_utc"),
            "error_type": manifest_receipt.get("error_type"),
            "error": manifest_receipt.get("error"),
            "response_values_read": manifest_receipt.get("response_values_read"),
            "data_access": manifest_receipt.get("data_access"),
            "gpu_hours": 0,
        })
    receipt = dict(manifest_receipt)
    for stale_field in ("error_type", "error", "failed_at_utc"):
        receipt.pop(stale_field, None)
    receipt["failed_attempts"] = failed_attempts
    receipt["attempt"] = len(failed_attempts) + 1
    receipt["execution_code_sha256"] = sha256_file(Path(__file__).resolve())
    receipt.update({
        "status": "RUNNING_CPU_EVALUATION",
        "run_started_at_utc": run_start_utc,
        "response_values_read": False,
        "data_access": {"train_delta_rows": 0, "selected_inner_dev_delta_rows": 0, "outer_or_locked_rows": 0},
    })
    json_write(manifest_path, receipt)
    try:
        train_indices = audit["train_indices"]
        selected_rows = selected_keys["prepared_row_index"].to_numpy(dtype=np.int64)
        selected_ids = selected_keys["condition_id"].to_numpy(dtype=np.int64)
        conditions = audit["dev_conditions"]
        if np.intersect1d(train_indices, selected_rows).size:
            raise ValueError("selected inner-development rows overlap train121 rows")
        if not np.array_equal(
            conditions.iloc[selected_rows]["condition_id"].astype(int).to_numpy(), selected_ids
        ):
            raise ValueError("frozen condition_id keys do not map back to their prepared row positions")

        # Only permitted training deltas and the selected 537 dev deltas are indexed.
        delta_memmap = np.load(PREPARED / "delta.npy", mmap_mode="r", allow_pickle=False)
        if delta_memmap.ndim != 2 or delta_memmap.shape[1] != N_GENES:
            raise ValueError(f"prepared delta gene axis shape mismatch: {delta_memmap.shape}")
        dev_delta = np.asarray(delta_memmap[selected_rows], dtype=np.float32)
        if not np.isfinite(dev_delta).all():
            raise ValueError("selected development responses contain non-finite values")
        receipt["response_values_read"] = True
        receipt["data_access"] = {
            "train_delta_rows": int(len(train_indices)),
            "selected_inner_dev_delta_rows": int(len(selected_rows)),
            "outer_or_locked_rows": 0,
            "pdo_treated_rows": 0,
        }
        json_write(manifest_path, receipt)

        train_keys = audit["train_keys"]
        train_codes, train_drug_ids = pd.factorize(train_keys["pert_id"].astype(str), sort=True)
        sums = np.zeros((len(train_drug_ids), N_GENES), dtype=np.float64)
        counts = np.bincount(train_codes, minlength=len(train_drug_ids)).astype(np.int64)
        for begin in range(0, len(train_indices), 1024):
            end = min(begin + 1024, len(train_indices))
            batch = np.asarray(delta_memmap[train_indices[begin:end]], dtype=np.float64)
            if not np.isfinite(batch).all():
                raise ValueError(f"non-finite train delta in permitted row range {begin}:{end}")
            np.add.at(sums, train_codes[begin:end], batch)
        sums /= counts[:, None]
        drug_mean = {str(drug): sums[i].astype(np.float32) for i, drug in enumerate(train_drug_ids)}
        del delta_memmap

        features_memmap = np.load(PREPARED / "ecfp4.npy", mmap_mode="r", allow_pickle=False)
        if features_memmap.ndim != 2 or features_memmap.shape[1] != N_DRUG_FEATURES:
            raise ValueError(f"prepared ECFP4 feature shape mismatch: {features_memmap.shape}")
        features = np.asarray(features_memmap[selected_rows], dtype=np.float32)
        if not np.isfinite(features).all():
            raise ValueError("selected drug features contain non-finite values")
        del features_memmap

        crosswalk = pd.read_csv(ADAPTER / "source_object_crosswalk.tsv", sep="\t", dtype=str)
        depmap_id = str(selected_keys["depmap_id"].iloc[0])
        depmap_row = int(audit["depmap_row"])
        if crosswalk.loc[crosswalk["cell_line_name"].str.upper().eq(TARGET_CELL), "depmap_id"].astype(str).tolist() != [depmap_id]:
            raise ValueError("frozen selected key table's DepMap identity differs from adapter crosswalk")
        expression = np.load(DEPMAP_DIR / "ccle_24q2_exact978_log2tpm.npy", mmap_mode="r", allow_pickle=False)
        if expression.ndim != 2 or expression.shape[1] != N_GENES or depmap_row >= expression.shape[0]:
            raise ValueError(f"registered DepMap expression axis mismatch: {expression.shape}")
        raw_profile = np.asarray(expression[depmap_row], dtype=np.float32)
        del expression
        if not np.isfinite(raw_profile).all():
            raise ValueError("selected MDAMB231 DepMap baseline profile has non-finite values")
        mapped_profile = (audit["adapter_intercept"] + audit["adapter_slope"] * raw_profile.astype(np.float64)).astype(np.float32)
        if not np.isfinite(mapped_profile).all():
            raise ValueError("frozen affine adapter produced non-finite MDAMB231 profile")

        x_raw = np.broadcast_to(raw_profile, (len(selected_rows), N_GENES)).copy()
        x_mapped = np.broadcast_to(mapped_profile, (len(selected_rows), N_GENES)).copy()
        model = load_official_model(official_source)
        prediction_raw = predict_delta(model, x_raw, features)
        prediction_mapped = predict_delta(model, x_mapped, features)
        del model

        true_ids = selected_keys["pert_id"].astype(str).to_numpy()
        same_drug_available = np.asarray([drug in drug_mean for drug in true_ids], dtype=bool)
        same_drug_prediction = np.zeros_like(dev_delta)
        for i, drug in enumerate(true_ids):
            if drug in drug_mean:
                same_drug_prediction[i] = drug_mean[drug]
        prediction_zero = np.zeros_like(dev_delta)
        arms = {
            "raw_depmap_input": prediction_raw,
            "mapped_depmap_input": prediction_mapped,
            "zero_change": prediction_zero,
        }
        all_indices = np.arange(len(dev_delta))
        supported_indices = np.flatnonzero(same_drug_available)
        metric_sets: dict[str, Any] = {"all_537_condition_keys": {}}
        for name, prediction in arms.items():
            metric_sets["all_537_condition_keys"][name] = aggregate_metrics(
                dev_delta[all_indices], prediction[all_indices], audit["gene_scale"]
            )
        if len(supported_indices):
            metric_sets["same_drug_supported_common_keys"] = {
                "n_condition_keys": int(len(supported_indices)),
                "n_unsupported_keys_excluded": int(len(dev_delta) - len(supported_indices)),
            }
            for name, prediction in arms.items():
                metric_sets["same_drug_supported_common_keys"][name] = aggregate_metrics(
                    dev_delta[supported_indices], prediction[supported_indices], audit["gene_scale"]
                )
            metric_sets["same_drug_supported_common_keys"]["train_only_same_drug_mean"] = aggregate_metrics(
                dev_delta[supported_indices], same_drug_prediction[supported_indices], audit["gene_scale"]
            )
        metric_sets["same_drug_supported_common_keys"].setdefault("train_only_same_drug_mean", None)
        metric_sets["all_537_condition_keys"]["train_only_same_drug_mean"] = {
            "status": "undefined_for_unsupported_pert_id",
            "n_supported_condition_keys": int(same_drug_available.sum()),
            "n_unsupported_condition_keys": int((~same_drug_available).sum()),
        }

        raw_per = per_condition_metrics(dev_delta, prediction_raw, audit["gene_scale"])
        mapped_per = per_condition_metrics(dev_delta, prediction_mapped, audit["gene_scale"])
        zero_per = per_condition_metrics(dev_delta, prediction_zero, audit["gene_scale"])
        mean_per = per_condition_metrics(dev_delta, same_drug_prediction, audit["gene_scale"])
        per_key = selected_keys.copy()
        per_key["observed_mean_abs_delta"] = [row["observed_mean_abs_delta"] for row in raw_per]
        for arm_name, rows in [
            ("raw_depmap_input", raw_per), ("mapped_depmap_input", mapped_per),
            ("zero_change", zero_per), ("train_only_same_drug_mean", mean_per),
        ]:
            for field in [
                "nmse", "rmse", "mae", "mean_signed_error", "pearson_gene", "spearman_gene",
                "signed_cosine", "sign_agreement", "predicted_mean_abs_delta",
                "absolute_magnitude_bias", "absolute_magnitude_ratio",
            ]:
                per_key[f"{arm_name}__{field}"] = [row[field] for row in rows]
        per_key["mapped_minus_raw_nmse"] = per_key["mapped_depmap_input__nmse"] - per_key["raw_depmap_input__nmse"]

        mapped_minus_raw = per_key["mapped_minus_raw_nmse"].to_numpy(dtype=np.float64)
        paired = {
            "n_paired_condition_keys": int(len(mapped_minus_raw)),
            "mean_mapped_minus_raw_nmse": float(mapped_minus_raw.mean()),
            "median_mapped_minus_raw_nmse": float(np.median(mapped_minus_raw)),
            "n_keys_mapped_nmse_lower": int(np.sum(mapped_minus_raw < 0)),
            "n_keys_equal_nmse": int(np.sum(mapped_minus_raw == 0)),
            "n_keys_raw_nmse_lower": int(np.sum(mapped_minus_raw > 0)),
            "relative_nmse_reduction_percent": float(100.0 * (1.0 - metric_sets["all_537_condition_keys"]["mapped_depmap_input"]["nmse"] / metric_sets["all_537_condition_keys"]["raw_depmap_input"]["nmse"])) if metric_sets["all_537_condition_keys"]["raw_depmap_input"]["nmse"] > 0 else None,
        }

        predictions = selected_keys.copy()
        predictions["delta_observed"] = [row.tolist() for row in dev_delta]
        predictions["delta_pred_raw_depmap_input"] = [row.tolist() for row in prediction_raw]
        predictions["delta_pred_mapped_depmap_input"] = [row.tolist() for row in prediction_mapped]
        predictions["delta_pred_zero_change"] = [row.tolist() for row in prediction_zero]
        predictions["delta_pred_train_only_same_drug_mean"] = [row.tolist() for row in same_drug_prediction]
        predictions["same_drug_mean_supported"] = same_drug_available
        predictions.to_parquet(OUT / "keyed_predictions.parquet", index=False, compression="zstd")
        per_key.to_csv(OUT / "per_key_metrics.tsv", sep="\t", index=False, lineterminator="\n")

        summary = {
            "experiment_id": "EXP-050",
            "run_id": RUN_ID,
            "status": "COMPLETED_DEVELOPMENT_ONLY_DESCRIPTIVE",
            "run_started_at_utc": run_start_utc,
            "run_completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "analysis_unit": "one MDAMB231 source cell_id and its 537 exact inner-development condition_id keys; genes are response dimensions, not independent units",
            "coverage": {
                "n_cell_ids": 1,
                "cell_id": TARGET_CELL,
                "n_condition_keys": int(len(dev_delta)),
                "n_unique_pert_id": int(pd.Series(true_ids).nunique()),
                "same_drug_mean_supported_condition_keys": int(same_drug_available.sum()),
                "same_drug_mean_unsupported_condition_keys": int((~same_drug_available).sum()),
            },
            "metric_sets": metric_sets,
            "paired_mapped_vs_raw": paired,
            "adapter_input_profile": {
                "raw_min": float(np.min(raw_profile)), "raw_max": float(np.max(raw_profile)),
                "raw_mean": float(np.mean(raw_profile)),
                "mapped_min": float(np.min(mapped_profile)), "mapped_max": float(np.max(mapped_profile)),
                "mapped_mean": float(np.mean(mapped_profile)),
                "mapped_negative_gene_count": int(np.sum(mapped_profile < 0)),
                "mapped_finite": bool(np.isfinite(mapped_profile).all()),
            },
            "prediction": {
                "model": "frozen official TranSiGen transigen_121 checkpoint",
                "selected_epoch": 13,
                "prediction_seed": PREDICTION_SEED,
                "same_latent_random_stream_for_raw_and_mapped": True,
                "device": "CPU",
                "gpu_hours": 0,
            },
            "interpretation_boundary": snapshot["interpretation_boundary"],
            "strict_independent_validation": {
                "status": "LOCAL_NO_GO_ZERO_OBJECTS_ZERO_KEYS",
                "n_cell_ids": 0,
                "n_condition_keys": 0,
                "detail": snapshot["strict_adapter_validation_overlap"],
            },
            "data_access": {
                "training_delta_rows_used_for_baseline": int(len(train_indices)),
                "selected_inner_dev_delta_rows_used_for_scoring": int(len(dev_delta)),
                "outer_fold_response_rows_read": 0,
                "locked_eval_response_rows_read": 0,
                "locked_test_metadata_read": False,
                "pdo_treated_labels_read": False,
                "prepared_control_values_read": False,
            },
            "uncertainty": "No confidence interval or p-value: the evaluation has one cell source object, uses one frozen stochastic prediction seed, and the full inner-development partition informed epoch selection.",
        }
        json_write(OUT / "summary.json", summary)
        receipt.update({
            "status": "COMPLETED_DEVELOPMENT_ONLY_DESCRIPTIVE",
            "run_completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "elapsed_cpu_wall_seconds": float(time.perf_counter() - start),
            "response_values_read": True,
            "data_access": {
                "train_delta_rows": int(len(train_indices)),
                "selected_inner_dev_delta_rows": int(len(dev_delta)),
                "outer_or_locked_rows": 0,
                "pdo_treated_rows": 0,
                "prepared_control_values_read": False,
            },
            "gpu_hours": 0,
            "summary": summary,
            "outputs": {
                name: sha256_file(OUT / name)
                for name in ["protocol_snapshot.json", "keyed_overlap.tsv", "keyed_predictions.parquet", "per_key_metrics.tsv", "summary.json"]
            },
        })
        json_write(manifest_path, receipt)
    except Exception as exc:
        failed_attempts.append({
            "attempt": receipt["attempt"],
            "status": "FAILED_DURING_CPU_EVALUATION",
            "run_started_at_utc": run_start_utc,
            "failed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "response_values_read": receipt.get("response_values_read", False),
            "data_access": receipt.get("data_access"),
            "execution_code_sha256": receipt["execution_code_sha256"],
            "gpu_hours": 0,
        })
        receipt.update({
            "status": "FAILED_DURING_CPU_EVALUATION",
            "failed_attempts": failed_attempts,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "failed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "gpu_hours": 0,
        })
        json_write(manifest_path, receipt)
        raise
    print(json.dumps({
        "status": "COMPLETED_DEVELOPMENT_ONLY_DESCRIPTIVE",
        "run_dir": str(OUT),
        "n_condition_keys": int(len(selected_rows)),
        "same_drug_supported": int(same_drug_available.sum()),
        "paired_mapped_minus_raw_nmse": paired["mean_mapped_minus_raw_nmse"],
        "gpu_hours": 0,
        "elapsed_seconds": receipt["elapsed_cpu_wall_seconds"],
    }, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["preflight", "evaluate"], required=True)
    parser.add_argument("--official-source", type=Path, default=DEFAULT_OFFICIAL_SOURCE)
    args = parser.parse_args()
    require_runtime()
    if args.phase == "preflight":
        run_preflight(args.official_source)
    else:
        run_evaluation(args.official_source)


if __name__ == "__main__":
    main()
