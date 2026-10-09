"""Fixed CPU-only EXP-061 Hallmark-constrained state-to-response comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any

REQUESTED_CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/modeling"))
import exp051_contract_fit_smoke as fit  # noqa: E402
import run_exp052_transigen_lr_stability as source_reader  # noqa: E402

ARMS = ("functional_hallmark", "random_gene_map", "random_object_state", "train_state_mismatch")
METHODS = ("zero_change", "per_drug_mean", *ARMS)
METRICS = ("nmse", "mae", "gene_correlation", "direction_agreement", "amplitude_ratio")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha(values: list[str] | np.ndarray) -> str:
    return hashlib.sha256(("\n".join(map(str, values)) + "\n").encode("utf-8")).hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(json_safe(value), ensure_ascii=False, sort_keys=True,
                               indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temp.replace(path)


def atomic_tsv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, sep="\t", index=False, na_rep="")
    temp.replace(path)


def parse_gmt_text(text: str) -> list[tuple[str, list[str]]]:
    sets: list[tuple[str, list[str]]] = []
    seen: set[str] = set()
    for line_no, line in enumerate(text.splitlines(), start=1):
        fields = line.rstrip("\r\n").split("\t")
        if len(fields) < 3:
            raise ValueError(f"GMT line {line_no} has fewer than three fields")
        name = fields[0].strip()
        if not name.startswith("HALLMARK_") or name in seen:
            raise ValueError(f"invalid/duplicate Hallmark set name at GMT line {line_no}: {name}")
        seen.add(name)
        members = sorted(set(gene.strip() for gene in fields[2:] if gene.strip()))
        sets.append((name, members))
    if len(sets) != 50:
        raise ValueError(f"registered Hallmark release must contain 50 sets, found {len(sets)}")
    return sorted(sets, key=lambda item: item[0])


def build_basis_from_symbols(gmt_text: str, axis_symbols: list[str], *, minimum_coverage: int = 5,
                             gene_row_permutation: np.ndarray | None = None
                             ) -> tuple[np.ndarray, dict[str, Any]]:
    if len(axis_symbols) != 978 or len(set(axis_symbols)) != 978:
        raise ValueError("frozen gene-symbol axis must contain 978 unique symbols")
    symbol_to_index = {symbol: i for i, symbol in enumerate(axis_symbols)}
    entries: list[dict[str, Any]] = []
    columns: list[np.ndarray] = []
    for name, members in parse_gmt_text(gmt_text):
        mapped = sorted({symbol_to_index[symbol] for symbol in members if symbol in symbol_to_index})
        if len(mapped) < minimum_coverage:
            entries.append({"name": name, "gmt_members": len(members), "axis_coverage": len(mapped),
                            "included": False, "reason": "axis_coverage_below_registered_minimum"})
            continue
        column = np.zeros(978, dtype=np.float64)
        column[mapped] = 1.0 / math.sqrt(len(mapped))
        if abs(float(np.linalg.norm(column)) - 1.0) > 1e-12:
            raise ArithmeticError(f"Hallmark column normalization failed: {name}")
        columns.append(column)
        entries.append({"name": name, "gmt_members": len(members), "axis_coverage": len(mapped),
                        "covered_axis_indices": mapped, "covered_symbols": [axis_symbols[i] for i in mapped],
                        "included": True})
    if not columns:
        raise ValueError("no Hallmark set meets the fixed >=5-gene coverage rule")
    basis = np.column_stack(columns)
    if gene_row_permutation is not None:
        perm = np.asarray(gene_row_permutation, dtype=np.int64)
        if perm.shape != (978,) or sorted(perm.tolist()) != list(range(978)):
            raise ValueError("gene-row permutation must be a full 978-row permutation")
        basis = basis[perm, :]
    receipt = {"set_count_gmt": 50, "minimum_axis_coverage": minimum_coverage,
               "eligible_set_count": len(columns), "excluded_set_count": 50 - len(columns),
               "basis_shape": list(basis.shape), "column_l2_normalized_by_sqrt_covered_size": True,
               "overlap_preserved": True, "entries": entries,
               "covered_gene_count": int(np.count_nonzero(np.any(basis != 0, axis=1))),
               "uncovered_gene_count": int(978 - np.count_nonzero(np.any(basis != 0, axis=1)))}
    return basis, receipt


def load_hallmark_inputs(cfg: dict[str, Any], axis_mapping: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    data = cfg["data"]
    map_path = Path(str(data["gene_map"]))
    gmt_path = ROOT / str(data["hallmark_gmt"])
    for path in (map_path, gmt_path):
        if not path.is_file():
            raise FileNotFoundError(f"registered functional mapping input is missing: {path}")
    gene_map_sha = sha256(map_path)
    gmt_sha = sha256(gmt_path)
    if gene_map_sha != str(data["gene_map_sha256"]):
        raise ValueError(f"gene_map SHA mismatch; expected {data['gene_map_sha256']}, got {gene_map_sha}")
    if gmt_sha != str(data["hallmark_gmt_sha256"]):
        raise ValueError(f"Hallmark GMT SHA mismatch; expected {data['hallmark_gmt_sha256']}, got {gmt_sha}")
    ref = pd.read_csv(map_path, dtype=str, keep_default_na=False)
    required = {"gene_id", "gene_symbol"}
    if not required.issubset(ref.columns) or len(ref) != 978 or not ref.gene_id.is_unique:
        raise ValueError("verified gene_map must contain exactly 978 unique gene_id rows and gene_symbol")
    axis_gene_ids = axis_mapping["h5ad_var_annotation"].astype(str).tolist()
    if axis_gene_ids != ref["gene_id"].astype(str).tolist():
        raise ValueError("gene_map gene_id sequence differs from the frozen ordered 978-gene axis")
    axis_symbols = ref["gene_symbol"].astype(str).tolist()
    if any(not symbol for symbol in axis_symbols):
        raise ValueError("gene_map contains an empty symbol on the frozen axis")
    duplicate_symbols = pd.Series(axis_symbols).value_counts()
    ambiguous = sorted(duplicate_symbols[duplicate_symbols > 1].index.astype(str).tolist())
    gmt_text = gmt_path.read_text(encoding="utf-8")
    gmt_sets = parse_gmt_text(gmt_text)
    ambiguous_used = sorted({symbol for _, members in gmt_sets for symbol in members if symbol in set(ambiguous)})
    if ambiguous_used:
        raise ValueError(f"ambiguous Hallmark symbols on the frozen axis block this gene-map input: {ambiguous_used}")
    basis, basis_receipt = build_basis_from_symbols(gmt_text, axis_symbols,
        minimum_coverage=int(data["hallmark_minimum_axis_coverage"]))
    perm = np.random.default_rng(int(cfg["arms"]["random_gene_map_seed"])).permutation(978)
    basis_random, random_receipt = build_basis_from_symbols(gmt_text, axis_symbols,
        minimum_coverage=int(data["hallmark_minimum_axis_coverage"]), gene_row_permutation=perm)
    set_names = [entry["name"] for entry in basis_receipt["entries"] if entry["included"]]
    # The random basis has exactly the same set columns and norms; names stay in their original order.
    receipt = {
        "status": "PASS_EXACT_GENE_ID_AND_HALLMARK_SYMBOL_MAPPING",
        "gene_map_path": str(map_path), "gene_map_sha256": gene_map_sha,
        "frozen_gene_axis_mapping_sha256": sha256(ROOT / str(data["frozen_gene_axis_mapping"])),
        "gmt_path": str(gmt_path.relative_to(ROOT)), "gmt_registry_id": data["hallmark_registry_id"],
        "gmt_sha256": gmt_sha, "gene_id_order_exact": True,
        "ambiguous_symbols_on_axis": ambiguous, "ambiguous_hallmark_symbols_used": ambiguous_used,
        "gene_symbol_axis_sha256": text_sha(axis_symbols), "gene_id_axis_sha256": text_sha(axis_gene_ids),
        "hallmark_set_names": set_names, "true_basis": basis_receipt,
        "random_gene_row_permutation": perm.tolist(),
        "random_gene_row_permutation_sha256": hashlib.sha256(perm.astype("<i8").tobytes()).hexdigest(),
        "random_basis": random_receipt,
        "random_basis_column_norms_equal_true": bool(np.allclose(np.linalg.norm(basis, axis=0),
                                                                   np.linalg.norm(basis_random, axis=0), atol=1e-12)),
    }
    return basis, basis_random, axis_symbols, receipt


def validate_source_metadata(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, Path, dict[str, Any]]:
    source_id = str(cfg["data"]["source_run_id"])
    fit.configure_source_run(source_id)
    src = fit.SOURCE_RUN_DIR
    required = ("metadata_freeze_receipt.json", "gene_axis_receipt.json", "value_contract_receipt.json",
                "root_values_readback_receipt.json", "response_blind_conditions.tsv", "gene_axis_mapping.tsv",
                "frozen_train_inner_values.npz")
    missing = [name for name in required if not (src / name).is_file()]
    if missing:
        raise FileNotFoundError(f"frozen EXP-051 source contract missing: {missing}")
    freeze = json.loads((src / "metadata_freeze_receipt.json").read_text(encoding="utf-8"))
    axis = json.loads((src / "gene_axis_receipt.json").read_text(encoding="utf-8"))
    value_receipt = json.loads((src / "value_contract_receipt.json").read_text(encoding="utf-8"))
    readback = json.loads((src / "root_values_readback_receipt.json").read_text(encoding="utf-8"))
    archive = src / "frozen_train_inner_values.npz"
    expected_sha = value_receipt.get("files_sha256", {}).get("frozen_train_inner_values_npz")
    if freeze.get("run_id") != source_id or axis.get("status") != "PASS_EXACT_978_AXIS":
        raise ValueError("source run/axis receipt is not the registered PASS asset")
    if readback.get("checks", {}).get("result") != "PASS" or not expected_sha:
        raise ValueError("source values lack a root PASS readback binding")
    if sha256(archive) != expected_sha or readback.get("input_sha256") != expected_sha:
        raise ValueError("source values archive SHA differs from frozen value/root receipts")
    axis_map_path = src / "gene_axis_mapping.tsv"
    axis_mapping = pd.read_csv(axis_map_path, sep="\t", dtype=str, keep_default_na=False)
    if (len(axis_mapping) != 978 or axis_mapping.axis_index.astype(int).tolist() != list(range(978))
            or axis_mapping.h5ad_var_name.nunique() != 978):
        raise ValueError("frozen gene-axis mapping is not unique ordered 978")
    genes = axis_mapping.h5ad_var_name.astype(str).to_numpy(dtype=str)
    annotation_sha = hashlib.sha256(("\n".join(axis_mapping.h5ad_var_annotation.astype(str)) + "\n").encode()).hexdigest()
    if annotation_sha != axis.get("mapping_sha256"):
        raise ValueError("frozen gene annotation sequence differs from the 978-axis receipt")
    key_path = src / "response_blind_conditions.tsv"
    all_keys = pd.read_csv(key_path, sep="\t", dtype=str, keep_default_na=False)
    required_cols = {"split_role", "source_object_id", "source_row", "pert_id", "full_inchikey",
                     "full_condition_key", "pert_dose_num", "pert_dose_unit", "pert_time_num", "pert_time_unit"}
    if not required_cols.issubset(all_keys.columns) or not all_keys.full_condition_key.is_unique:
        raise ValueError("frozen condition table lacks required keys/exposure fields or has duplicate keys")
    roles = all_keys.split_role.astype(str)
    if not roles.isin(["train", "inner_dev"]).all():
        raise ValueError(f"unregistered source roles present: {sorted(set(roles) - {'train', 'inner_dev'})}")
    train_keys = all_keys.loc[roles.eq("train")].reset_index(drop=True)
    dev_keys = all_keys.loc[roles.eq("inner_dev")].reset_index(drop=True)
    train_objects = set(train_keys.source_object_id.astype(str))
    dev_objects = set(dev_keys.source_object_id.astype(str))
    expected = cfg["data"]
    counts = (len(train_keys), len(train_objects), len(dev_keys), len(dev_objects),
              int(all_keys.full_inchikey.nunique()), len(genes))
    expected_counts = tuple(int(expected[key]) for key in
        ("train_conditions", "train_objects", "inner_dev_conditions", "inner_dev_objects", "exact_full_inchikeys", "genes"))
    if counts != expected_counts or (train_objects & dev_objects):
        raise ValueError(f"frozen split identity/count gate failed: actual={counts}, expected={expected_counts}, overlap={len(train_objects & dev_objects)}")
    if set(train_keys.full_inchikey.astype(str)) != set(dev_keys.full_inchikey.astype(str)):
        raise ValueError("TRAIN and inner-dev complete InChIKey domains differ")
    dose = pd.to_numeric(all_keys.pert_dose_num, errors="coerce")
    duration = pd.to_numeric(all_keys.pert_time_num, errors="coerce")
    fixed = (np.isclose(dose.to_numpy(dtype=float), 10.0, atol=1e-9)
             & all_keys.pert_dose_unit.astype(str).str.casefold().eq("um").to_numpy()
             & np.isclose(duration.to_numpy(dtype=float), 24.0, atol=1e-9)
             & all_keys.pert_time_unit.astype(str).str.casefold().eq("h").to_numpy())
    if not fixed.all():
        raise ValueError(f"fixed 10uM x 24h gate failed: {int((~fixed).sum())} / {len(all_keys)} rows")
    h5ad = Path(source_reader.fit.H5AD)
    h5ad_sha = sha256(h5ad)
    expected_h5ad = freeze.get("file_sha256", {}).get("h5ad")
    if expected_h5ad and h5ad_sha != expected_h5ad:
        raise ValueError("source H5AD differs from frozen EXP-051 metadata asset")
    receipt = {"source_run_id": source_id, "source_h5ad_sha256": h5ad_sha,
        "source_metadata_freeze_receipt_sha256": sha256(src / "metadata_freeze_receipt.json"),
        "source_values_root_readback_receipt_sha256": sha256(src / "root_values_readback_receipt.json"),
        "source_key_table_sha256": sha256(key_path), "source_value_archive_sha256": expected_sha,
        "source_gene_axis_mapping_sha256": sha256(axis_map_path),
        "gene_annotation_axis_sha256": annotation_sha, "train_condition_key_sha256": text_sha(train_keys.full_condition_key.astype(str).to_numpy()),
        "inner_dev_condition_key_sha256": text_sha(dev_keys.full_condition_key.astype(str).to_numpy()),
        "train_object_ids_sha256": text_sha(sorted(train_objects)),
        "inner_dev_object_ids_sha256": text_sha(sorted(dev_objects)),
        "full_inchikey_axis": sorted(all_keys.full_inchikey.astype(str).unique().tolist()),
        "genes": 978, "denominators": {"train_conditions": len(train_keys), "train_objects": len(train_objects),
        "inner_dev_conditions": len(dev_keys), "inner_dev_objects": len(dev_objects),
        "train_dev_object_overlap": 0, "full_inchikeys": int(all_keys.full_inchikey.nunique()),
        "fixed_exposure_rows_10uM_24h": int(fixed.sum())},
        "train_values_loaded": False, "inner_dev_numeric_values_loaded": False,
        "pdo_outer_locked_panacea_numeric_values_opened": False}
    return train_keys, dev_keys, genes, src, receipt


def load_train_values(train_keys: pd.DataFrame, genes: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    values = source_reader.read_role_expression(train_keys, genes)
    x_ctl = values["X_ctl"].astype(np.float64)
    delta = values["delta"].astype(np.float64)
    if x_ctl.shape != (419, 978) or delta.shape != (419, 978):
        raise ValueError(f"TRAIN matrix shape changed: X_ctl={x_ctl.shape}, delta={delta.shape}")
    if not np.isfinite(x_ctl).all() or not np.isfinite(delta).all():
        raise FloatingPointError("TRAIN control or response matrix has nonfinite elements")
    decomp = float(values["max_abs_decomposition_error"])
    if decomp > 1e-6:
        raise ValueError(f"TRAIN treatment-control decomposition exceeds tolerance: {decomp}")
    return x_ctl, delta, {"train_role_value_read_receipt": json_safe(values.get("receipt", {})),
                          "max_abs_treated_control_delta_decomposition_error": decomp,
                          "train_x_ctl_sha256": hashlib.sha256(x_ctl.astype("<f8").tobytes()).hexdigest(),
                          "train_delta_sha256": hashlib.sha256(delta.astype("<f8").tobytes()).hexdigest(),
                          "inner_dev_numeric_values_loaded": False}


def object_state_map(keys: pd.DataFrame, x_ctl: np.ndarray) -> dict[str, np.ndarray]:
    if x_ctl.shape != (len(keys), 978):
        raise ValueError("state matrix does not align with the condition table and 978 axis")
    ids = keys.source_object_id.astype(str).to_numpy()
    return {obj: x_ctl[np.flatnonzero(ids == obj)].mean(axis=0, dtype=np.float64)
            for obj in sorted(set(ids))}


def derive_state_scaler(train_state_map: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    ids = sorted(train_state_map)
    matrix = np.stack([train_state_map[obj] for obj in ids]).astype(np.float64)
    mean = matrix.mean(axis=0, dtype=np.float64)
    raw_scale = matrix.std(axis=0, ddof=0)
    bad = ~np.isfinite(raw_scale) | (raw_scale <= 1e-12)
    scale = raw_scale.copy()
    scale[bad] = 1.0
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise FloatingPointError("TRAIN-only state scaler contains nonfinite values")
    standardized = {obj: (train_state_map[obj] - mean) / scale for obj in ids}
    receipt = {"fit_unit": "110 TRAIN source_object_id object-mean X_ctl only",
        "train_object_count": len(ids), "mean_sha256": hashlib.sha256(mean.astype("<f8").tobytes()).hexdigest(),
        "scale_sha256": hashlib.sha256(scale.astype("<f8").tobytes()).hexdigest(),
        "zero_or_invalid_scale_genes_set_to_one": int(bad.sum()),
        "state_axis": "frozen ordered 978-gene axis", "inner_dev_numeric_values_used": False}
    return mean, scale, {"standardized": standardized, "receipt": receipt}


def derangement(ids: list[str], seed: int) -> dict[str, str]:
    ordered = sorted(map(str, ids))
    if len(ordered) < 2:
        raise ValueError("object-state mismatch requires at least two unique TRAIN objects")
    rng = np.random.default_rng(seed)
    perm = np.asarray(ordered, dtype=str)[rng.permutation(len(ordered))]
    offset = int(rng.integers(1, len(ordered)))
    mapping = {str(perm[i]): str(perm[(i + offset) % len(ordered)]) for i in range(len(ordered))}
    if any(obj == mapping[obj] for obj in mapping) or set(mapping) != set(mapping.values()):
        raise RuntimeError("state mismatch map is not a no-fixed-point permutation")
    return mapping


def random_object_vector(source_object_id: str, width: int, seed: int) -> np.ndarray:
    key = f"EXP-061|{seed}|{source_object_id}".encode("utf-8")
    local_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little", signed=False)
    return np.random.default_rng(local_seed).standard_normal(width).astype(np.float64)


def rows_from_object_vectors(keys: pd.DataFrame, object_vectors: dict[str, np.ndarray], basis: np.ndarray | None = None) -> np.ndarray:
    ids = keys.source_object_id.astype(str).to_numpy()
    matrix = np.stack([object_vectors[obj] for obj in ids]).astype(np.float64)
    return matrix if basis is None else matrix @ basis


def per_drug_mean(train_keys: pd.DataFrame, train_y: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    train_drug = train_keys.full_inchikey.astype(str).to_numpy()
    means = {drug: train_y[train_drug == drug].mean(axis=0, dtype=np.float64)
             for drug in sorted(set(train_drug))}
    return np.stack([means[drug] for drug in train_drug]), means


def make_row_weights(keys: pd.DataFrame) -> np.ndarray:
    ids = keys.source_object_id.astype(str).to_numpy()
    unique = sorted(set(ids))
    counts = pd.Series(ids).value_counts().to_dict()
    weights = np.asarray([1.0 / (len(unique) * counts[obj]) for obj in ids], dtype=np.float64)
    if not np.isfinite(weights).all() or abs(float(weights.sum()) - 1.0) > 1e-12:
        raise ArithmeticError("TRAIN row weights do not sum to one across equally weighted objects")
    return weights


def fit_drug_mapping(train_keys: pd.DataFrame, train_y: np.ndarray, train_mean_rows: np.ndarray,
                     train_q: np.ndarray, basis: np.ndarray, gene_variance: np.ndarray,
                     full_inchikey: str, alpha: float) -> tuple[np.ndarray, dict[str, Any]]:
    mask = train_keys.full_inchikey.astype(str).to_numpy() == str(full_inchikey)
    positions = np.flatnonzero(mask)
    if not len(positions):
        raise ValueError(f"drug has no TRAIN rows: {full_inchikey}")
    subkeys = train_keys.iloc[positions].reset_index(drop=True)
    q = train_q[positions]
    residual = train_y[positions] - train_mean_rows[positions]
    weights = make_row_weights(subkeys)
    if q.shape != (len(positions), basis.shape[1]) or residual.shape != (len(positions), 978):
        raise ValueError("per-drug fit inputs do not align with fixed basis dimensions")
    if not (np.isfinite(q).all() and np.isfinite(residual).all() and np.isfinite(basis).all()):
        raise FloatingPointError(f"nonfinite per-drug fit input for {full_inchikey}")
    if not np.isfinite(gene_variance).all() or np.any(gene_variance <= 1e-12):
        raise ValueError("the fixed full-978 TRAIN variance denominator contains an unusable gene")
    weight_gene_basis = basis / gene_variance[:, None]
    gram = (basis.T @ weight_gene_basis) / 978.0
    state_gram = q.T @ (weights[:, None] * q)
    projected_residual = (residual / gene_variance[None, :]) @ basis / 978.0
    rhs_matrix = projected_residual.T @ (weights[:, None] * q)
    h = basis.shape[1]
    system = np.kron(state_gram.T, gram) + alpha * np.eye(h * h, dtype=np.float64)
    if not (np.isfinite(system).all() and np.isfinite(rhs_matrix).all()):
        raise FloatingPointError(f"nonfinite fixed ridge normal equation for {full_inchikey}")
    solution = np.linalg.solve(system, rhs_matrix.reshape(-1, order="F"))
    matrix = solution.reshape((h, h), order="F")
    if not np.isfinite(matrix).all():
        raise FloatingPointError(f"nonfinite fitted A matrix for {full_inchikey}")
    receipt = {"full_inchikey": str(full_inchikey), "train_rows": int(len(positions)),
        "train_objects": int(subkeys.source_object_id.nunique()), "object_weights_sum": float(weights.sum()),
        "eligible_hallmark_sets": int(h), "parameters": int(h * h), "alpha": float(alpha),
        "objective": "sum_i w_i * mean_g[(B A q_i - residual_i)^2 / TRAIN_gene_variance_g] + alpha*||A||_F^2",
        "normal_equation_sha256": hashlib.sha256(system.astype("<f8").tobytes()).hexdigest(),
        "rhs_sha256": hashlib.sha256(rhs_matrix.astype("<f8").tobytes()).hexdigest(),
        "A_sha256": hashlib.sha256(matrix.astype("<f8").tobytes()).hexdigest()}
    return matrix, receipt


def predictions_for_drug(q: np.ndarray, basis: np.ndarray, a: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    result = (q @ a.T) @ basis.T + anchor
    if not np.isfinite(result).all():
        raise FloatingPointError("nonfinite functional constrained prediction")
    return result


def metrics_for_role(keys: pd.DataFrame, truth: np.ndarray, prediction: np.ndarray,
                     gene_variance: np.ndarray) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if truth.shape != (len(keys), 978) or prediction.shape != truth.shape:
        raise ValueError("evaluation matrix shape differs from the full 978-gene denominator")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise FloatingPointError("nonfinite value; no finite-subset metric will be reported")
    if gene_variance.shape != (978,) or not np.isfinite(gene_variance).all() or np.any(gene_variance <= 1e-12):
        raise ValueError("full 978-gene TRAIN variance denominator is invalid")
    error = prediction - truth
    row_nmse = (np.square(error) / gene_variance[None, :]).mean(axis=1)
    row_mae = np.abs(error).mean(axis=1)
    row_direction = (np.sign(prediction) == np.sign(truth)).mean(axis=1)
    row_corr = np.full(len(keys), np.nan, dtype=np.float64)
    for i in range(len(keys)):
        if np.std(truth[i]) > 0 and np.std(prediction[i]) > 0:
            row_corr[i] = float(np.corrcoef(truth[i], prediction[i])[0, 1])
    object_ids = keys.source_object_id.astype(str).to_numpy()
    drugs = keys.full_inchikey.astype(str).to_numpy()

    def group_stats(index: np.ndarray, obj: str, drug: str | None = None) -> dict[str, Any]:
        corr = row_corr[index]
        finite_corr = corr[np.isfinite(corr)]
        target_amp = float(np.mean(np.abs(truth[index])))
        pred_amp = float(np.mean(np.abs(prediction[index])))
        return {"source_object_id": obj, "full_inchikey": drug,
                "nmse": float(row_nmse[index].mean()), "mae": float(row_mae[index].mean()),
                "gene_correlation": float(finite_corr.mean()) if len(finite_corr) else None,
                "direction_agreement": float(row_direction[index].mean()),
                "amplitude_ratio": pred_amp / target_amp if target_amp > 0 else None,
                "conditions": int(len(index)), "genes_per_condition": 978,
                "condition_gene_correlations_defined": int(len(finite_corr))}

    object_rows: list[dict[str, Any]] = []
    for obj in sorted(set(object_ids)):
        idx = np.flatnonzero(object_ids == obj)
        object_rows.append(group_stats(idx, obj))
    object_drug_rows: list[dict[str, Any]] = []
    for obj in sorted(set(object_ids)):
        for drug in sorted(set(drugs[object_ids == obj])):
            idx = np.flatnonzero((object_ids == obj) & (drugs == drug))
            object_drug_rows.append(group_stats(idx, obj, drug))
    drug_rows: list[dict[str, Any]] = []
    for drug in sorted(set(drugs)):
        rows = [row for row in object_drug_rows if row["full_inchikey"] == drug]
        drug_rows.append({"full_inchikey": drug,
            **{f"object_macro_{metric}": float(np.mean([r[metric] for r in rows if r[metric] is not None]))
               if any(r[metric] is not None for r in rows) else None for metric in METRICS},
            "objects": int(len(rows)), "conditions": int(np.sum(drugs == drug)), "genes": 978})
    summary = {f"object_macro_{metric}": float(np.mean([row[metric] for row in object_rows if row[metric] is not None]))
               if any(row[metric] is not None for row in object_rows) else None for metric in METRICS}
    summary.update({"conditions": int(len(keys)), "objects": int(len(object_rows)), "genes": 978,
                    "condition_gene_values": int(len(keys) * 978), "finite_condition_gene_values": int(np.isfinite(prediction).sum()),
                    "condition_gene_correlations_defined": int(np.isfinite(row_corr).sum()),
                    "nonfinite_values": int(np.size(prediction) - np.isfinite(prediction).sum()),
                    "gene_variance_fit_rows": 419, "gene_variance_usable_genes": 978,
                    "object_scores": {r["source_object_id"]: {k: r[k] for k in METRICS} for r in object_rows}})
    return summary, object_rows, object_drug_rows, drug_rows


def paired_bootstrap(object_scores: dict[str, dict[str, Any]], methods: list[str], *,
                     replicates: int, seed: int) -> dict[str, Any]:
    if "per_drug_mean" not in object_scores:
        raise ValueError("paired bootstrap requires method->object score maps with a per-drug mean anchor")
    objects = sorted(object_scores["per_drug_mean"])
    if not objects:
        raise ValueError("paired object bootstrap has no objects")
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(objects), size=(replicates, len(objects)))
    base = np.asarray([object_scores["per_drug_mean"][obj]["nmse"] for obj in objects], dtype=np.float64)
    results: dict[str, Any] = {"unit": "source_object_id", "paired_objects": len(objects),
        "replicates": int(replicates), "seed": int(seed), "interval_interpretation": "descriptive fixed-split uncertainty only"}
    for method in methods:
        if method not in object_scores:
            raise ValueError(f"paired bootstrap score map is missing method {method}")
        candidate = np.asarray([object_scores[method][obj]["nmse"] for obj in objects], dtype=np.float64)
        differences = base - candidate
        samples = differences[draw].mean(axis=1)
        results[method] = {"mean_absolute_nmse_improvement_over_per_drug_mean": float(differences.mean()),
            "relative_nmse_improvement_over_per_drug_mean": float(differences.mean() / base.mean()) if base.mean() != 0 else None,
            "objects_with_lower_nmse_than_per_drug_mean": int((differences > 0).sum()),
            "paired_95pct_percentile_interval_absolute_improvement": [float(x) for x in np.quantile(samples, [0.025, 0.975])],
            "positive_favors_method": True}
    return results


def append_ledger(path: Path, *, event: str, start: float, process_start: float,
                  status: str, details: dict[str, Any] | None = None) -> None:
    row = {"event": event, "status": status, "at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
           "attempt_elapsed_wall_seconds": time.monotonic() - start,
           "attempt_process_cpu_seconds": time.process_time() - process_start,
           "details": json_safe(details or {})}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def build_key_arrays(keys: pd.DataFrame, genes: np.ndarray, *, prefix: str) -> dict[str, np.ndarray]:
    return {f"{prefix}_condition_key": keys.full_condition_key.astype(str).to_numpy(dtype=np.str_),
        f"{prefix}_source_object_id": keys.source_object_id.astype(str).to_numpy(dtype=np.str_),
        f"{prefix}_full_inchikey": keys.full_inchikey.astype(str).to_numpy(dtype=np.str_),
        f"{prefix}_pert_id": keys.pert_id.astype(str).to_numpy(dtype=np.str_),
        f"{prefix}_gene_axis": genes.astype(str).astype(np.str_)}


def _run(args: argparse.Namespace) -> int:
    if REQUESTED_CUDA_VISIBLE_DEVICES.strip() not in ("", "-1"):
        raise RuntimeError("EXP-061 is CPU-only; CUDA_VISIBLE_DEVICES must be empty")
    cfg_path = (ROOT / args.config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if (cfg.get("experiment_id") != "EXP-061" or cfg.get("run_id") != args.run_id
            or cfg.get("attempt_id") != args.attempt_id or args.device != "cpu"):
        raise ValueError("CLI arguments do not match the fixed EXP-061 CPU run contract")
    if (int(cfg["resources"]["cpu_cap_seconds"]) != 1800
            or float(cfg["resources"]["gpu_hours_cap"]) != 0
            or float(cfg["model"]["ridge_alpha"]) != 1.0
            or float(cfg["resources"]["projection_margin_fraction"]) != 0.20
            or int(cfg["selection"]["bootstrap_replicates"]) != 2000):
        raise ValueError("EXP-061 fixed resource/model contract changed")
    out_root = ROOT / cfg["output"]["root"]
    attempt = out_root / args.attempt_id
    if out_root.exists() or attempt.exists():
        raise FileExistsError(f"immutable unique EXP-061 output already exists: {out_root}")
    out_root.mkdir(parents=True, exist_ok=False)
    attempt.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    process_started = time.process_time()
    manifest_path = attempt / "run_manifest.json"
    manifest = {"experiment_id": "EXP-061", "run_id": args.run_id, "attempt_id": args.attempt_id,
        "status": "RUNNING", "config_sha256": sha256(cfg_path), "runner_sha256": sha256(Path(__file__).resolve()),
        "started_at_utc": pd.Timestamp.now(tz="UTC").isoformat(), "device": "cpu",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "cpu_cap_seconds": int(cfg["resources"]["cpu_cap_seconds"]),
        "inherited_exp055_cpu_cap_seconds": int(cfg["resources"]["inherited_exp055_cpu_cap_seconds"]),
        "inherited_exp055_cpu_settled_before_package_seconds": float(cfg["resources"]["exp055_cpu_settled_before_package_seconds"]),
        "pre_run_package_cpu_seconds_charged": float(cfg["resources"].get("pre_run_package_cpu_seconds_charged", 0)),
        "gpu_hours_cap": 0, "forbidden_numeric_inputs_opened": []}
    atomic_json(manifest_path, manifest)
    ledger_path = attempt / "resource_ledger.jsonl"
    ledger_path.write_text("", encoding="utf-8")
    try:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            train_keys, dev_keys, genes, source_dir, input_receipt = validate_source_metadata(cfg)
            basis, random_basis, axis_symbols, basis_receipt = load_hallmark_inputs(
                cfg, pd.read_csv(source_dir / "gene_axis_mapping.tsv", sep="\t", dtype=str, keep_default_na=False))
            atomic_json(attempt / "metadata_and_functional_input_receipt.json",
                        {"source_metadata": input_receipt, "functional_basis": basis_receipt})
            train_ctl, train_y, train_values_receipt = load_train_values(train_keys, genes)
            input_receipt["train_values_readback"] = train_values_receipt
            input_receipt["train_values_loaded"] = True
            atomic_json(attempt / "train_numeric_input_receipt.json", input_receipt)
            train_states = object_state_map(train_keys, train_ctl)
            state_mean, state_scale, scaler_payload = derive_state_scaler(train_states)
            train_z = scaler_payload["standardized"]
            train_objects = sorted(train_states)
            train_map = {obj: obj for obj in train_objects}
            shuffled_map = derangement(train_objects, int(cfg["arms"]["train_state_mismatch_seed"]))
            train_mismatch_vectors = {obj: train_z[shuffled_map[obj]] for obj in train_objects}
            train_random_vectors = {obj: random_object_vector(obj, basis.shape[1],
                int(cfg["arms"]["random_object_state_seed"])) for obj in train_objects}
            train_q = {
                "functional_hallmark": rows_from_object_vectors(train_keys, train_z, basis),
                "random_gene_map": rows_from_object_vectors(train_keys, train_z, random_basis),
                "random_object_state": rows_from_object_vectors(train_keys, train_random_vectors),
                "train_state_mismatch": rows_from_object_vectors(train_keys, train_mismatch_vectors, basis),
            }
            gene_variance = np.var(train_y.astype(np.float64), axis=0, ddof=0)
            if gene_variance.shape != (978,) or not np.isfinite(gene_variance).all() or np.any(gene_variance <= 1e-12):
                raise ValueError("the fixed 978-gene TRAIN variance axis has zero/invalid components")
            train_mean_rows, means = per_drug_mean(train_keys, train_y)
            train_zero = np.zeros_like(train_y, dtype=np.float64)
            drug_axis = sorted(means)
            if len(drug_axis) != 5:
                raise ValueError(f"expected five exact full-key drugs, got {len(drug_axis)}")
            train_predictions = {"zero_change": train_zero, "per_drug_mean": train_mean_rows}
            coefficients: dict[str, dict[str, np.ndarray]] = {arm: {} for arm in ARMS}
            fit_rows: list[dict[str, Any]] = []
            row_drugs = train_keys.full_inchikey.astype(str).to_numpy()
            first_drug = drug_axis[0]
            first_rows = int((row_drugs == first_drug).sum())
            max_rows = max(int((row_drugs == drug).sum()) for drug in drug_axis)
            if first_rows < 1:
                raise ValueError("lexicographically first registered drug has no TRAIN rows")
            gate_drug_started = time.monotonic()
            for arm in ARMS:
                train_predictions[arm] = np.empty_like(train_y, dtype=np.float64)
            for drug in drug_axis:
                drug_receipts: list[dict[str, Any]] = []
                for arm in ARMS:
                    output_basis = random_basis if arm == "random_gene_map" else basis
                    matrix, receipt = fit_drug_mapping(train_keys, train_y, train_mean_rows,
                        train_q[arm], output_basis, gene_variance, drug,
                        alpha=float(cfg["model"]["ridge_alpha"]))
                    coefficients[arm][drug] = matrix
                    pos = np.flatnonzero(row_drugs == drug)
                    train_predictions[arm][pos] = predictions_for_drug(
                        train_q[arm][pos], output_basis, matrix, train_mean_rows[pos])
                    receipt["arm"] = arm
                    drug_receipts.append(receipt)
                fit_rows.extend(drug_receipts)
                if drug == first_drug:
                    first_positions = np.flatnonzero(row_drugs == drug)
                    partial_arrays: dict[str, np.ndarray] = {
                        "full_condition_key": train_keys.iloc[first_positions].full_condition_key.astype(str).to_numpy(dtype=np.str_),
                        "source_object_id": train_keys.iloc[first_positions].source_object_id.astype(str).to_numpy(dtype=np.str_),
                        "full_inchikey": train_keys.iloc[first_positions].full_inchikey.astype(str).to_numpy(dtype=np.str_),
                        "gene_axis": genes.astype(str).astype(np.str_),
                        "target_delta": train_y[first_positions].astype(np.float32),
                        "per_drug_mean": train_mean_rows[first_positions].astype(np.float32),
                        "zero_change": train_zero[first_positions].astype(np.float32),
                    }
                    for arm in ARMS:
                        partial_arrays[f"prediction_{arm}"] = train_predictions[arm][first_positions].astype(np.float32)
                        partial_arrays[f"A_{arm}_{drug_axis.index(drug):03d}"] = coefficients[arm][drug].astype(np.float64)
                    atomic_npz(attempt / "first_drug_formal_partial.npz", **partial_arrays)
                    atomic_json(attempt / "first_drug_fit_receipt.json", {
                        "status": "FOUR_ARM_FIRST_DRUG_FORMAL_FIT_PREDICT_SAVE_COMPLETE",
                        "full_inchikey": drug, "train_rows": int(len(first_positions)),
                        "objects": int(train_keys.iloc[first_positions].source_object_id.nunique()),
                        "genes_per_condition": 978, "four_arms": list(ARMS),
                        "first_drug_partial_sha256": sha256(attempt / "first_drug_formal_partial.npz"),
                        "fit_receipts": drug_receipts,
                        "inner_dev_numeric_values_loaded": False})
                    append_ledger(ledger_path, event="first_drug_four_arm_fit_predict_save",
                        start=started, process_start=process_started, status="COMPLETE",
                        details={"drug": drug, "rows": int(len(first_positions)), "objects": int(train_keys.iloc[first_positions].source_object_id.nunique())})
                    measured = time.monotonic() - gate_drug_started
                    elapsed = time.monotonic() - started
                    row_scale = max(1.0, max_rows / first_rows)
                    per_remaining_drug_upper = max(measured * row_scale, measured)
                    remaining_fit = per_remaining_drug_upper * (len(drug_axis) - 1)
                    final_metric_and_io_allowance = max(120.0, 0.5 * measured)
                    root_readback_allowance = 180.0
                    projected_pre_margin = elapsed + remaining_fit + final_metric_and_io_allowance + root_readback_allowance
                    projected_total = (float(cfg["resources"].get("pre_run_package_cpu_seconds_charged", 0))
                        + projected_pre_margin * (1.0 + float(cfg["resources"]["projection_margin_fraction"])))
                    exp055_projected = float(cfg["resources"]["exp055_cpu_settled_before_package_seconds"]) + projected_total
                    cost_gate = {"status": "PASS" if projected_total <= int(cfg["resources"]["cpu_cap_seconds"])
                        and exp055_projected <= int(cfg["resources"]["inherited_exp055_cpu_cap_seconds"])
                        else "FAIL_CLOSED_COST_GATE",
                        "first_drug": drug, "first_drug_rows": first_rows, "max_rows_any_drug": max_rows,
                        "measured_first_drug_four_arm_fit_predict_save_wall_seconds": measured,
                        "elapsed_before_gate_seconds": elapsed, "remaining_drug_count": len(drug_axis)-1,
                        "row_scale_upper_factor": row_scale, "projected_remaining_fit_seconds": remaining_fit,
                        "projected_metrics_io_seconds": final_metric_and_io_allowance,
                        "root_independent_readback_allowance_seconds": root_readback_allowance,
                        "projection_margin_fraction": float(cfg["resources"]["projection_margin_fraction"]),
                        "pre_run_package_cpu_seconds_charged": float(cfg["resources"].get("pre_run_package_cpu_seconds_charged", 0)),
                        "projected_new_package_seconds": projected_total,
                        "new_package_cap_seconds": int(cfg["resources"]["cpu_cap_seconds"]),
                        "projected_exp055_cumulative_seconds": exp055_projected,
                        "inherited_exp055_cap_seconds": int(cfg["resources"]["inherited_exp055_cpu_cap_seconds"]),
                        "continued_same_attempt": projected_total <= int(cfg["resources"]["cpu_cap_seconds"])
                            and exp055_projected <= int(cfg["resources"]["inherited_exp055_cpu_cap_seconds"]),
                        "all_arms_included": list(ARMS), "no_drug_or_gene_shortening": True,
                        "inner_dev_values_loaded": False}
                    atomic_json(attempt / "first_drug_cost_gate.json", cost_gate)
                    if cost_gate["status"] != "PASS":
                        atomic_json(attempt / "attempt_close_receipt.json", {
                            "status": "FAILED_CLOSED_COST_GATE", "reason": "full fixed comparison with margin does not fit both CPU caps",
                            "first_drug_formal_partial_retained": True, "inner_dev_numeric_values_loaded": False,
                            "cpu_wall_seconds": elapsed, "process_cpu_seconds": time.process_time() - process_started,
                            "gpu_hours": 0, "cost_gate": cost_gate})
                        manifest.update({"status": "FAILED_CLOSED_COST_GATE", "closed_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                            "cpu_wall_seconds": elapsed, "process_cpu_seconds": time.process_time() - process_started})
                        atomic_json(manifest_path, manifest)
                        append_ledger(ledger_path, event="cost_gate_closed", start=started,
                            process_start=process_started, status="FAILED_CLOSED", details=cost_gate)
                        return 0
                    atomic_json(attempt / "cost_gate_continuation_receipt.json", {"status": "PASS_CONTINUE_SAME_ATTEMPT",
                        "cost_gate_sha256": sha256(attempt / "first_drug_cost_gate.json"),
                        "candidate_protocol_frozen_before_inner_dev_numeric_access": True})

            if any(not np.isfinite(pred).all() for pred in train_predictions.values()):
                raise FloatingPointError("a full 419x978 TRAIN prediction contains nonfinite values")
            atomic_json(attempt / "train_model_freeze_receipt.json", {
                "status": "ALL_FOUR_ARM_COEFFICIENTS_FROZEN_BEFORE_INNER_DEV_NUMERIC_READ",
                "drug_axis": drug_axis, "arms": list(ARMS), "ridge_alpha": float(cfg["model"]["ridge_alpha"]),
                "functional_basis_sha256": hashlib.sha256(basis.astype("<f8").tobytes()).hexdigest(),
                "random_basis_sha256": hashlib.sha256(random_basis.astype("<f8").tobytes()).hexdigest(),
                "coefficient_fit_rows": fit_rows, "train_predictions_all_finite": True,
                "train_predictions_sha256": {method: hashlib.sha256(pred.astype("<f8").tobytes()).hexdigest()
                                              for method, pred in train_predictions.items()},
                "inner_dev_numeric_values_loaded": False})
            # DEV numerical values are deliberately opened only after all TRAIN fits and
            # keyed TRAIN predictions have been frozen above.
            dev_values = source_reader.read_role_expression(dev_keys, genes)
            dev_ctl = dev_values["X_ctl"].astype(np.float64)
            dev_y = dev_values["delta"].astype(np.float64)
            if dev_ctl.shape != (93, 978) or dev_y.shape != (93, 978):
                raise ValueError("frozen inner-dev matrix shape changed")
            if not np.isfinite(dev_ctl).all() or not np.isfinite(dev_y).all():
                raise FloatingPointError("inner-dev matrix contains nonfinite values; no partial score")
            dev_decomp = float(dev_values["max_abs_decomposition_error"])
            if dev_decomp > 1e-6:
                raise ValueError(f"inner-dev treatment-control decomposition exceeds tolerance: {dev_decomp}")
            dev_states = object_state_map(dev_keys, dev_ctl)
            dev_z = {obj: (state - state_mean) / state_scale for obj, state in dev_states.items()}
            dev_q = {
                "functional_hallmark": rows_from_object_vectors(dev_keys, dev_z, basis),
                "random_gene_map": rows_from_object_vectors(dev_keys, dev_z, random_basis),
                "random_object_state": rows_from_object_vectors(dev_keys,
                    {obj: random_object_vector(obj, basis.shape[1], int(cfg["arms"]["random_object_state_seed"])) for obj in dev_z}),
                "train_state_mismatch": rows_from_object_vectors(dev_keys, dev_z, basis),
            }
            dev_base = np.stack([means[drug] for drug in dev_keys.full_inchikey.astype(str)]).astype(np.float64)
            dev_predictions = {"zero_change": np.zeros_like(dev_y), "per_drug_mean": dev_base}
            dev_drugs = dev_keys.full_inchikey.astype(str).to_numpy()
            for arm in ARMS:
                pred = np.empty_like(dev_y, dtype=np.float64)
                q = dev_q[arm]
                output_basis = random_basis if arm == "random_gene_map" else basis
                for drug in drug_axis:
                    pos = np.flatnonzero(dev_drugs == drug)
                    if not len(pos):
                        continue
                    pred[pos] = predictions_for_drug(q[pos], output_basis, coefficients[arm][drug], dev_base[pos])
                dev_predictions[arm] = pred
            train_metrics: dict[str, Any] = {}
            dev_metrics: dict[str, Any] = {}
            object_table: list[dict[str, Any]] = []
            object_drug_table: list[dict[str, Any]] = []
            drug_table: list[dict[str, Any]] = []
            object_scores: dict[str, dict[str, dict[str, Any]]] = {"train": {}, "inner_dev": {}}
            for role, keys, truth, predictions in (("train", train_keys, train_y, train_predictions),
                                                   ("inner_dev", dev_keys, dev_y, dev_predictions)):
                target_metrics = train_metrics if role == "train" else dev_metrics
                for method in METHODS:
                    summary, objects, object_drugs, drugs = metrics_for_role(
                        keys, truth, predictions[method], gene_variance)
                    target_metrics[method] = summary
                    object_scores[role][method] = summary["object_scores"]
                    object_table.extend({"role": role, "method": method, **row} for row in objects)
                    object_drug_table.extend({"role": role, "method": method, **row} for row in object_drugs)
                    drug_table.extend({"role": role, "method": method, **row} for row in drugs)
            paired = {role: paired_bootstrap(object_scores[role], list(METHODS),
                replicates=int(cfg["selection"]["bootstrap_replicates"]), seed=int(cfg["selection"]["bootstrap_seed"]))
                for role in ("train", "inner_dev")}
            candidate = dev_metrics["functional_hallmark"]
            mean = dev_metrics["per_drug_mean"]
            improvement = float(mean["object_macro_nmse"] - candidate["object_macro_nmse"])
            relative_improvement = improvement / float(mean["object_macro_nmse"])
            ci = paired["inner_dev"]["functional_hallmark"]["paired_95pct_percentile_interval_absolute_improvement"]
            improved_objects = paired["inner_dev"]["functional_hallmark"]["objects_with_lower_nmse_than_per_drug_mean"]
            amplitude_min, amplitude_max = map(float, cfg["selection"]["amplitude_ratio_range"])
            controls = {arm: float(dev_metrics[arm]["object_macro_nmse"]) for arm in ARMS if arm != "functional_hallmark"}
            gate = {
                "relative_nmse_improvement_over_per_drug_mean": relative_improvement,
                "absolute_nmse_improvement_over_per_drug_mean": improvement,
                "paired_95pct_ci_absolute_improvement": ci,
                "ci_lower_bound_above_zero": bool(ci[0] > 0),
                "objects_with_lower_nmse_than_per_drug_mean": int(improved_objects),
                "minimum_objects_required": int(cfg["selection"]["minimum_inner_dev_objects_with_lower_nmse_than_drug_mean"]),
                "amplitude_ratio": candidate["object_macro_amplitude_ratio"],
                "amplitude_gate": bool(amplitude_min
                    <= candidate["object_macro_amplitude_ratio"] <= amplitude_max),
                "beats_all_three_registered_controls": bool(all(candidate["object_macro_nmse"] < value for value in controls.values())),
                "control_nmse": controls,
                "gate_pass": bool(relative_improvement >= float(cfg["selection"]["minimum_inner_dev_relative_nmse_improvement_over_drug_mean"])
                    and ci[0] > 0
                    and improved_objects >= int(cfg["selection"]["minimum_inner_dev_objects_with_lower_nmse_than_drug_mean"])
                    and amplitude_min <= candidate["object_macro_amplitude_ratio"] <= amplitude_max
                    and all(candidate["object_macro_nmse"] < value for value in controls.values())),
                "candidate_scope": "one fixed source-development split; inner-dev is not independent validation",
            }
            atomic_npz(attempt / "keyed_predictions.npz",
                **build_key_arrays(train_keys, genes, prefix="train"),
                **build_key_arrays(dev_keys, genes, prefix="inner_dev"),
                train_target_delta=train_y.astype(np.float32), inner_dev_target_delta=dev_y.astype(np.float32),
                train_gene_variance=gene_variance.astype(np.float64),
                true_hallmark_basis=basis.astype(np.float64), random_gene_basis=random_basis.astype(np.float64),
                train_state_mean=state_mean.astype(np.float64), train_state_scale=state_scale.astype(np.float64),
                train_state_matrix=np.stack([train_states[o] for o in train_objects]).astype(np.float64),
                train_state_object_axis=np.asarray(train_objects, dtype=np.str_),
                train_state_mismatch_donor=np.asarray([shuffled_map[o] for o in train_objects], dtype=np.str_),
                train_gene_symbols=np.asarray(axis_symbols, dtype=np.str_),
                hallmark_names=np.asarray(basis_receipt["hallmark_set_names"], dtype=np.str_),
                train_q_functional=train_q["functional_hallmark"].astype(np.float64),
                train_q_random_gene_map=train_q["random_gene_map"].astype(np.float64),
                train_q_random_object_state=train_q["random_object_state"].astype(np.float64),
                train_q_state_mismatch=train_q["train_state_mismatch"].astype(np.float64),
                inner_dev_q_functional=dev_q["functional_hallmark"].astype(np.float64),
                inner_dev_q_random_gene_map=dev_q["random_gene_map"].astype(np.float64),
                inner_dev_q_random_object_state=dev_q["random_object_state"].astype(np.float64),
                inner_dev_q_state_mismatch=dev_q["train_state_mismatch"].astype(np.float64),
                **{f"train_prediction_{m}": train_predictions[m].astype(np.float32) for m in METHODS},
                **{f"inner_dev_prediction_{m}": dev_predictions[m].astype(np.float32) for m in METHODS},
                **{f"A_{arm}_{i:03d}": coefficients[arm][drug].astype(np.float64)
                   for arm in ARMS for i, drug in enumerate(drug_axis)})
            atomic_npz(attempt / "model_coefficients.npz",
                **{f"A_{arm}_{i:03d}": coefficients[arm][drug].astype(np.float64)
                   for arm in ARMS for i, drug in enumerate(drug_axis)},
                drug_axis=np.asarray(drug_axis, dtype=np.str_),
                functional_basis=basis.astype(np.float64), random_gene_basis=random_basis.astype(np.float64),
                train_gene_variance=gene_variance.astype(np.float64), state_mean=state_mean.astype(np.float64),
                state_scale=state_scale.astype(np.float64))
            summary = {"status": "COMPLETED_AWAITING_ROOT_INDEPENDENT_READBACK",
                "experiment_id": "EXP-061", "run_id": args.run_id, "attempt_id": args.attempt_id,
                "arms": list(ARMS), "baselines": ["zero_change", "per_drug_mean"],
                "train_metrics": train_metrics, "inner_dev_metrics": dev_metrics,
                "paired_object_bootstrap": paired, "registered_selection_gate": gate,
                "functional_basis": basis_receipt, "input_receipt": input_receipt,
                "train_state_scaler": scaler_payload["receipt"], "train_state_mismatch_mapping": shuffled_map,
                "train_state_mismatch_mapping_sha256": hashlib.sha256(json.dumps(shuffled_map, sort_keys=True,
                    separators=(",", ":")).encode()).hexdigest(),
                "train_fit_rows": fit_rows, "inner_dev_decomposition_error": dev_decomp,
                "inner_dev_numeric_values_loaded_after_train_freeze": True,
                "forbidden_numeric_inputs_opened": [],
                "denominators": {"train_conditions": len(train_keys), "train_objects": int(train_keys.source_object_id.nunique()),
                    "inner_dev_conditions": len(dev_keys), "inner_dev_objects": int(dev_keys.source_object_id.nunique()),
                    "object_overlap": 0, "full_inchikeys": len(drug_axis), "genes": 978,
                    "train_target_gene_values": int(train_y.size), "inner_dev_target_gene_values": int(dev_y.size)},
                "prediction_archive_sha256": sha256(attempt / "keyed_predictions.npz"),
                "coefficient_archive_sha256": sha256(attempt / "model_coefficients.npz")}
            atomic_json(attempt / "metrics.json", summary)
            atomic_json(attempt / "paired_object_bootstrap.json", paired)
            atomic_tsv(attempt / "object_metrics.tsv", pd.DataFrame(object_table))
            atomic_tsv(attempt / "object_drug_metrics.tsv", pd.DataFrame(object_drug_table))
            atomic_tsv(attempt / "drug_metrics.tsv", pd.DataFrame(drug_table))
            atomic_json(attempt / "functional_basis_receipt.json", basis_receipt)
            atomic_json(attempt / "fit_formula_receipt.json", {
                "prediction": "delta_hat = exact-key TRAIN mean + B @ A_drug @ q",
                "biological_q": "q=B.T@z; z=(object_mean_X_ctl-TRAIN_object_state_mean)/TRAIN_object_state_population_sd",
                "objective": "sum_i w_i*mean_g[((B A q_i-r_i)^2/v_g)] + alpha*||A||_F^2",
                "row_weights": "within each drug, w_i=1/(n_unique_TRAIN_objects_for_drug*n_rows_for_this_object_and_drug)",
                "gene_variance": "v_g=population variance over all 419 TRAIN delta rows; all 978 genes retained",
                "alpha": float(cfg["model"]["ridge_alpha"]), "parameters_per_drug_per_arm": int(basis.shape[1] ** 2),
                "output_basis": "eligible Hallmark columns; all uncovered genes receive residual zero and preserve per-drug mean anchor",
                "no_dev_fitting": True, "coefficient_solver": "float64 Kronecker normal equation with alpha*I; no hyperparameter search"})
            atomic_json(attempt / "attempt_close_receipt.json", {
                "status": "COMPLETED_AWAITING_ROOT_INDEPENDENT_READBACK",
                "candidate_gate_pass": gate["gate_pass"], "candidate_result": "PASS_PREDECLARED_GATE" if gate["gate_pass"] else "FAIL_CLOSE_FIXED_CANDIDATE",
                "cpu_wall_seconds": time.monotonic() - started, "process_cpu_seconds": time.process_time() - process_started,
                "pre_run_package_cpu_seconds_charged": float(cfg["resources"].get("pre_run_package_cpu_seconds_charged", 0)),
                "gpu_hours": 0, "inner_dev_numeric_access": "final frozen development scoring and registered paired intervals only",
                "pdo_outer_locked_panacea_numeric_values_opened": False})
            elapsed = time.monotonic() - started
            process_cpu = time.process_time() - process_started
            manifest.update({"status": "COMPLETED_AWAITING_ROOT_INDEPENDENT_READBACK",
                "closed_at_utc": pd.Timestamp.now(tz="UTC").isoformat(), "cpu_wall_seconds": elapsed,
                "process_cpu_seconds": process_cpu, "pre_run_package_cpu_seconds_charged": float(cfg["resources"].get("pre_run_package_cpu_seconds_charged", 0)),
                "full_fit_keyed_outputs": True, "root_readback_pending": True})
            atomic_json(manifest_path, manifest)
            append_ledger(ledger_path, event="runner_close", start=started, process_start=process_started,
                status=manifest["status"], details={"cpu_wall_seconds": elapsed, "process_cpu_seconds": process_cpu})
            return 0
    except Exception as exc:
        elapsed = time.monotonic() - started
        process_cpu = time.process_time() - process_started
        close = {"status": "FAILED_CLOSED", "exception_type": type(exc).__name__, "exception": str(exc),
            "traceback": traceback.format_exc(), "cpu_wall_seconds": elapsed,
            "process_cpu_seconds": process_cpu,
            "pre_run_package_cpu_seconds_charged": float(cfg["resources"].get("pre_run_package_cpu_seconds_charged", 0)),
            "gpu_hours": 0, "inner_dev_numeric_values_loaded": any(
                (attempt / name).exists() for name in ("train_model_freeze_receipt.json", "metrics.json")),
            "forbidden_numeric_inputs_opened": []}
        atomic_json(attempt / "attempt_close_receipt.json", close)
        manifest.update({"status": "FAILED_CLOSED", "closed_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                         "cpu_wall_seconds": elapsed, "process_cpu_seconds": process_cpu,
                         "failure_type": type(exc).__name__, "failure": str(exc)})
        atomic_json(manifest_path, manifest)
        append_ledger(ledger_path, event="failure_close", start=started, process_start=process_started,
            status="FAILED_CLOSED", details={"exception_type": type(exc).__name__, "exception": str(exc)})
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--attempt-id", default="fit_attempt_001")
    parser.add_argument("--device", choices=["cpu"], default="cpu")
    return _run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
