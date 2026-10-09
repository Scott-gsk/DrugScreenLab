"""Root-side independent numerical readback for the fixed EXP-061 comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/modeling"))
import exp051_contract_fit_smoke as contract  # noqa: E402
import run_exp052_transigen_lr_stability as role_loader  # noqa: E402

ARMS = ("functional_hallmark", "random_gene_map", "random_object_state", "train_state_mismatch")
METHODS = ("zero_change", "per_drug_mean", *ARMS)
METRICS = ("nmse", "mae", "gene_correlation", "direction_agreement", "amplitude_ratio")


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def axis_sha(values: list[str] | np.ndarray) -> str:
    return hashlib.sha256(("\n".join(map(str, values)) + "\n").encode()).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def load_contract_inputs(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, Path]:
    source_id = str(cfg["data"]["source_run_id"])
    contract.configure_source_run(source_id)
    source = contract.SOURCE_RUN_DIR
    keys = pd.read_csv(source / "response_blind_conditions.tsv", sep="\t", dtype=str, keep_default_na=False)
    mapping = pd.read_csv(source / "gene_axis_mapping.tsv", sep="\t", dtype=str, keep_default_na=False)
    if len(mapping) != 978 or mapping.axis_index.astype(int).tolist() != list(range(978)):
        raise ValueError("root readback: source gene axis is not ordered 978")
    roles = keys.split_role.astype(str)
    if not roles.isin(["train", "inner_dev"]).all():
        raise ValueError("root readback: unregistered split role in source key table")
    train = keys.loc[roles.eq("train")].reset_index(drop=True)
    dev = keys.loc[roles.eq("inner_dev")].reset_index(drop=True)
    expected = cfg["data"]
    train_objects = set(train.source_object_id.astype(str))
    dev_objects = set(dev.source_object_id.astype(str))
    observed = [len(train), len(train_objects), len(dev), len(dev_objects), keys.full_inchikey.nunique(), len(mapping)]
    wanted = [int(expected[k]) for k in ("train_conditions", "train_objects", "inner_dev_conditions", "inner_dev_objects", "exact_full_inchikeys", "genes")]
    if observed != wanted or train_objects & dev_objects:
        raise ValueError(f"root readback: role/count/object disjointness mismatch: {observed} != {wanted}")
    if set(train.full_inchikey.astype(str)) != set(dev.full_inchikey.astype(str)):
        raise ValueError("root readback: TRAIN and inner-dev complete key axes differ")
    if not train.full_condition_key.is_unique or not dev.full_condition_key.is_unique:
        raise ValueError("root readback: duplicate condition key")
    genes = mapping.h5ad_var_name.astype(str).to_numpy(dtype=str)
    return train, dev, genes, source


def load_basis(cfg: dict[str, Any], mapping: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    data = cfg["data"]
    gene_map = Path(str(data["gene_map"]))
    gmt = ROOT / str(data["hallmark_gmt"])
    if file_sha(gene_map) != data["gene_map_sha256"] or file_sha(gmt) != data["hallmark_gmt_sha256"]:
        raise ValueError("root readback: gene-map or Hallmark GMT checksum mismatch")
    reference = pd.read_csv(gene_map, dtype=str, keep_default_na=False)
    ids = mapping.h5ad_var_annotation.astype(str).tolist()
    if ids != reference.gene_id.astype(str).tolist():
        raise ValueError("root readback: gene_id ordered join differs from frozen EXP-051 axis")
    symbols = reference.gene_symbol.astype(str).tolist()
    if len(set(symbols)) != 978 or any(not s for s in symbols):
        raise ValueError("root readback: ambiguous or empty gene symbols on the frozen axis")
    symbol_index = {symbol: i for i, symbol in enumerate(symbols)}
    sets: list[tuple[str, list[str]]] = []
    seen: set[str] = set()
    for line_no, line in enumerate(gmt.read_text(encoding="utf-8").splitlines(), start=1):
        fields = line.split("\t")
        if len(fields) < 3 or not fields[0].startswith("HALLMARK_") or fields[0] in seen:
            raise ValueError(f"root readback: malformed Hallmark GMT line {line_no}")
        seen.add(fields[0])
        sets.append((fields[0], sorted(set(x for x in fields[2:] if x))))
    if len(sets) != 50:
        raise ValueError(f"root readback: expected 50 Hallmark sets, got {len(sets)}")
    ambiguous = {s for s, count in pd.Series(symbols).value_counts().items() if count > 1}
    used = sorted({s for _, members in sets for s in members if s in ambiguous})
    if used:
        raise ValueError(f"root readback: ambiguous Hallmark symbol on axis: {used}")
    columns = []
    names = []
    coverage = []
    for name, members in sorted(sets):
        idx = sorted({symbol_index[x] for x in members if x in symbol_index})
        if len(idx) >= int(data["hallmark_minimum_axis_coverage"]):
            col = np.zeros(978, dtype=np.float64)
            col[idx] = 1.0 / math.sqrt(len(idx))
            columns.append(col)
            names.append(name)
            coverage.append(len(idx))
    if not columns:
        raise ValueError("root readback: no Hallmark set passes fixed coverage threshold")
    basis = np.column_stack(columns)
    permutation = np.random.default_rng(int(cfg["arms"]["random_gene_map_seed"])).permutation(978)
    random_basis = basis[permutation]
    receipt = {"sets_total": len(sets), "eligible_names": names, "eligible_coverage": coverage,
               "basis_shape": list(basis.shape), "uncovered_gene_count": int((~np.any(basis != 0, axis=1)).sum()),
               "true_basis_sha256": hashlib.sha256(basis.astype("<f8").tobytes()).hexdigest(),
               "random_basis_sha256": hashlib.sha256(random_basis.astype("<f8").tobytes()).hexdigest(),
               "permutation": permutation.tolist()}
    return basis, random_basis, symbols, receipt


def state_map(keys: pd.DataFrame, controls: np.ndarray) -> dict[str, np.ndarray]:
    ids = keys.source_object_id.astype(str).to_numpy()
    return {obj: controls[ids == obj].mean(axis=0, dtype=np.float64) for obj in sorted(set(ids))}


def make_derangement(objects: list[str], seed: int) -> dict[str, str]:
    ordered = np.asarray(sorted(objects), dtype=str)
    rng = np.random.default_rng(seed)
    perm = ordered[rng.permutation(len(ordered))]
    offset = int(rng.integers(1, len(ordered)))
    mapping = {str(perm[i]): str(perm[(i + offset) % len(ordered)]) for i in range(len(ordered))}
    if len(set(mapping.values())) != len(mapping) or any(k == v for k, v in mapping.items()):
        raise ValueError("root readback: mismatch mapping is not a derangement")
    return mapping


def random_vector(object_id: str, width: int, seed: int) -> np.ndarray:
    key = f"EXP-061|{seed}|{object_id}".encode()
    local_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little", signed=False)
    return np.random.default_rng(local_seed).standard_normal(width).astype(np.float64)


def q_by_rows(keys: pd.DataFrame, states: dict[str, np.ndarray], basis: np.ndarray | None = None) -> np.ndarray:
    ids = keys.source_object_id.astype(str).to_numpy()
    values = np.stack([states[obj] for obj in ids]).astype(np.float64)
    return values if basis is None else values @ basis


def equal_object_weights(keys: pd.DataFrame) -> np.ndarray:
    ids = keys.source_object_id.astype(str).to_numpy()
    counts = pd.Series(ids).value_counts().to_dict()
    weights = np.asarray([1.0 / (len(counts) * counts[obj]) for obj in ids], dtype=np.float64)
    if abs(float(weights.sum()) - 1.0) > 1e-12:
        raise ValueError("root readback: object-balanced weights do not sum to one")
    return weights


def solve_mapping(keys: pd.DataFrame, y: np.ndarray, mean_rows: np.ndarray, q: np.ndarray,
                  basis: np.ndarray, variance: np.ndarray, drug: str, alpha: float) -> np.ndarray:
    keep = keys.full_inchikey.astype(str).to_numpy() == drug
    subkeys = keys.loc[keep].reset_index(drop=True)
    residual = y[keep] - mean_rows[keep]
    qdrug = q[keep]
    w = equal_object_weights(subkeys)
    if not (np.isfinite(residual).all() and np.isfinite(qdrug).all() and np.isfinite(variance).all()):
        raise FloatingPointError("root readback: nonfinite solver input")
    gram = (basis.T @ (basis / variance[:, None])) / 978.0
    state_gram = qdrug.T @ (w[:, None] * qdrug)
    rhs = ((residual / variance[None, :]) @ basis / 978.0).T @ (w[:, None] * qdrug)
    width = basis.shape[1]
    return np.linalg.solve(np.kron(state_gram.T, gram) + alpha * np.eye(width * width),
                           rhs.reshape(-1, order="F")).reshape((width, width), order="F")


def predict(q: np.ndarray, basis: np.ndarray, coefficient: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    values = (q @ coefficient.T) @ basis.T + anchor
    if not np.isfinite(values).all():
        raise FloatingPointError("root readback: reconstructed prediction is nonfinite")
    return values


def object_scores(keys: pd.DataFrame, truth: np.ndarray, prediction: np.ndarray,
                  variance: np.ndarray) -> dict[str, dict[str, float | None]]:
    if prediction.shape != (len(keys), 978) or not np.isfinite(prediction).all():
        raise ValueError("root readback: prediction shape or finite-value gate failed")
    if not np.isfinite(truth).all() or truth.shape != prediction.shape:
        raise ValueError("root readback: target shape or finite-value gate failed")
    err = prediction - truth
    nmse = (err * err / variance[None, :]).mean(axis=1)
    mae = np.abs(err).mean(axis=1)
    direction = (np.sign(prediction) == np.sign(truth)).mean(axis=1)
    corr = np.full(len(keys), np.nan)
    for i in range(len(keys)):
        if np.std(truth[i]) > 0 and np.std(prediction[i]) > 0:
            corr[i] = np.corrcoef(truth[i], prediction[i])[0, 1]
    ids = keys.source_object_id.astype(str).to_numpy()
    result: dict[str, dict[str, float | None]] = {}
    for obj in sorted(set(ids)):
        pos = np.flatnonzero(ids == obj)
        finite_corr = corr[pos][np.isfinite(corr[pos])]
        truth_amp = float(np.abs(truth[pos]).mean())
        pred_amp = float(np.abs(prediction[pos]).mean())
        result[obj] = {
            "nmse": float(nmse[pos].mean()), "mae": float(mae[pos].mean()),
            "gene_correlation": float(finite_corr.mean()) if len(finite_corr) else None,
            "direction_agreement": float(direction[pos].mean()),
            "amplitude_ratio": pred_amp / truth_amp if truth_amp > 0 else None,
        }
    return result


def summarize(scores: dict[str, dict[str, float | None]]) -> dict[str, float | None]:
    return {metric: float(np.mean([row[metric] for row in scores.values() if row[metric] is not None]))
            if any(row[metric] is not None for row in scores.values()) else None for metric in METRICS}


def compare_metric(actual: Any, expected: Any, *, atol: float = 2e-8) -> bool:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            return False
        return all(compare_metric(actual[k], expected[k], atol=atol) for k in expected)
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            compare_metric(a, e, atol=atol) for a, e in zip(actual, expected))
    if expected is None:
        return actual is None
    if isinstance(expected, (float, int, np.floating, np.integer)):
        try:
            return actual is not None and math.isfinite(float(actual)) and abs(float(actual) - float(expected)) <= atol * max(1.0, abs(float(expected)))
        except (TypeError, ValueError):
            return False
    return actual == expected


def _run(args: argparse.Namespace) -> int:
    started = time.monotonic()
    process_started = time.process_time()
    cfg_path = (ROOT / args.config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    attempt = ROOT / cfg["output"]["root"] / args.attempt_id
    receipt_path = attempt / "root_independent_readback.json"
    if receipt_path.exists():
        raise FileExistsError(f"immutable root readback already exists: {receipt_path}")
    files = {name: attempt / name for name in ("metrics.json", "keyed_predictions.npz", "model_coefficients.npz",
        "train_model_freeze_receipt.json", "attempt_close_receipt.json", "functional_basis_receipt.json",
        "paired_object_bootstrap.json")}
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"root readback missing runner outputs: {missing}")
    try:
        close = json.loads(files["attempt_close_receipt.json"].read_text(encoding="utf-8"))
        freeze = json.loads(files["train_model_freeze_receipt.json"].read_text(encoding="utf-8"))
        metric_doc = json.loads(files["metrics.json"].read_text(encoding="utf-8"))
        if close.get("status") != "COMPLETED_AWAITING_ROOT_INDEPENDENT_READBACK" or not freeze.get("train_predictions_all_finite"):
            raise ValueError("root readback refuses a failed or unfrozen attempt")
        if freeze.get("inner_dev_numeric_values_loaded") is not False:
            raise ValueError("root readback: TRAIN model freeze was not made before inner-dev load")
        train_keys, dev_keys, genes, source_dir = load_contract_inputs(cfg)
        mapping = pd.read_csv(source_dir / "gene_axis_mapping.tsv", sep="\t", dtype=str, keep_default_na=False)
        basis, random_basis, symbols, basis_root = load_basis(cfg, mapping)
        if (freeze.get("functional_basis_sha256") != basis_root["true_basis_sha256"]
                or freeze.get("random_basis_sha256") != basis_root["random_basis_sha256"]):
            raise ValueError("root readback: frozen basis hashes differ from independently reconstructed bases")
        basis_recorded = json.loads(files["functional_basis_receipt.json"].read_text(encoding="utf-8"))
        recorded_true_sha = basis_recorded.get("true_basis", {}).get("basis_sha256")
        # The runner records the matrix hash in its frozen model receipt; the functional
        # input receipt independently binds the mapping provenance and set coverage.
        if hashlib.sha256(basis.astype("<f8").tobytes()).hexdigest() != basis_root["true_basis_sha256"]:
            raise ValueError("root readback: internally inconsistent reconstructed basis hash")
        if recorded_true_sha is not None and recorded_true_sha != basis_root["true_basis_sha256"]:
            raise ValueError("root readback: functional basis hash mismatch")
        train_values = role_loader.read_role_expression(train_keys, genes)
        train_y = train_values["delta"].astype(np.float64)
        train_ctl = train_values["X_ctl"].astype(np.float64)
        if train_y.shape != (419, 978) or train_ctl.shape != train_y.shape or not np.isfinite(train_y).all() or not np.isfinite(train_ctl).all():
            raise ValueError("root readback: invalid TRAIN values")
        variance = np.var(train_y, axis=0, ddof=0)
        if np.any(variance <= 1e-12) or not np.isfinite(variance).all():
            raise ValueError("root readback: unusable full-axis TRAIN variance")
        drugs = sorted(train_keys.full_inchikey.astype(str).unique())
        train_means = {drug: train_y[train_keys.full_inchikey.astype(str).to_numpy() == drug].mean(axis=0)
                       for drug in drugs}
        train_anchor = np.stack([train_means[d] for d in train_keys.full_inchikey.astype(str)]).astype(np.float64)
        train_states = state_map(train_keys, train_ctl)
        object_axis = sorted(train_states)
        state_matrix = np.stack([train_states[o] for o in object_axis])
        state_mean = state_matrix.mean(axis=0)
        state_scale = state_matrix.std(axis=0, ddof=0)
        state_scale[(~np.isfinite(state_scale)) | (state_scale <= 1e-12)] = 1.0
        z_train = {obj: (state - state_mean) / state_scale for obj, state in train_states.items()}
        mismatch = make_derangement(object_axis, int(cfg["arms"]["train_state_mismatch_seed"]))
        rand_train = {obj: random_vector(obj, basis.shape[1], int(cfg["arms"]["random_object_state_seed"])) for obj in object_axis}
        train_q = {
            "functional_hallmark": q_by_rows(train_keys, z_train, basis),
            "random_gene_map": q_by_rows(train_keys, z_train, random_basis),
            "random_object_state": q_by_rows(train_keys, rand_train),
            "train_state_mismatch": q_by_rows(train_keys, {o: z_train[mismatch[o]] for o in object_axis}, basis),
        }
        coefficients: dict[str, dict[str, np.ndarray]] = {arm: {} for arm in ARMS}
        for arm in ARMS:
            output_basis = random_basis if arm == "random_gene_map" else basis
            for drug in drugs:
                coefficients[arm][drug] = solve_mapping(train_keys, train_y, train_anchor, train_q[arm],
                    output_basis, variance, drug, float(cfg["model"]["ridge_alpha"]))
        freeze_coeffs = np.load(files["model_coefficients.npz"], allow_pickle=False)
        archive = np.load(files["keyed_predictions.npz"], allow_pickle=False)
        axis_saved = freeze_coeffs["drug_axis"].astype(str).tolist()
        if axis_saved != drugs:
            raise ValueError("root readback: coefficient drug axis mismatch")
        for prefix, keys in (("train", train_keys), ("inner_dev", dev_keys)):
            for archive_name, frame_name in (("condition_key", "full_condition_key"),
                                             ("source_object_id", "source_object_id"),
                                             ("full_inchikey", "full_inchikey"), ("pert_id", "pert_id")):
                if not np.array_equal(archive[f"{prefix}_{archive_name}"].astype(str),
                                      keys[frame_name].astype(str).to_numpy()):
                    raise ValueError(f"root readback: {prefix} {archive_name} identity/order mismatch")
            if not np.array_equal(archive[f"{prefix}_gene_axis"].astype(str), genes.astype(str)):
                raise ValueError(f"root readback: {prefix} gene axis mismatch")
        if not np.array_equal(archive["train_gene_symbols"].astype(str), np.asarray(symbols, dtype=str)):
            raise ValueError("root readback: gene symbols mismatch")
        if (not np.allclose(archive["train_target_delta"], train_y, rtol=2e-7, atol=2e-7)
                or not np.allclose(archive["train_gene_variance"], variance, rtol=1e-10, atol=1e-12)):
            raise ValueError("root readback: TRAIN target or full-axis variance archive mismatch")
        train_predictions: dict[str, np.ndarray] = {
            "zero_change": np.zeros_like(train_y), "per_drug_mean": train_anchor}
        for arm in ARMS:
            pred = np.empty_like(train_y)
            row_drugs = train_keys.full_inchikey.astype(str).to_numpy()
            output_basis = random_basis if arm == "random_gene_map" else basis
            for drug in drugs:
                positions = np.flatnonzero(row_drugs == drug)
                pred[positions] = predict(train_q[arm][positions], output_basis, coefficients[arm][drug], train_anchor[positions])
                saved_coefficient = freeze_coeffs[f"A_{arm}_{drugs.index(drug):03d}"]
                if not np.allclose(saved_coefficient, coefficients[arm][drug], rtol=1e-9, atol=1e-10):
                    raise ValueError(f"root readback: coefficient mismatch for {arm}/{drug}")
            train_predictions[arm] = pred
        train_scores = {method: object_scores(train_keys, train_y, pred, variance) for method, pred in train_predictions.items()}

        # The numeric inner-dev role is opened only after the registered TRAIN coefficients
        # and keyed predictions have been independently reconstructed and frozen above.
        dev_values = role_loader.read_role_expression(dev_keys, genes)
        dev_y = dev_values["delta"].astype(np.float64)
        dev_ctl = dev_values["X_ctl"].astype(np.float64)
        if dev_y.shape != (93, 978) or dev_ctl.shape != dev_y.shape or not np.isfinite(dev_y).all() or not np.isfinite(dev_ctl).all():
            raise ValueError("root readback: invalid inner-dev values")
        if not np.allclose(archive["inner_dev_target_delta"], dev_y, rtol=2e-7, atol=2e-7):
            raise ValueError("root readback: inner-dev target archive mismatch")
        dev_states = state_map(dev_keys, dev_ctl)
        z_dev = {obj: (state - state_mean) / state_scale for obj, state in dev_states.items()}
        rand_dev = {obj: random_vector(obj, basis.shape[1], int(cfg["arms"]["random_object_state_seed"])) for obj in dev_states}
        dev_q = {
            "functional_hallmark": q_by_rows(dev_keys, z_dev, basis),
            "random_gene_map": q_by_rows(dev_keys, z_dev, random_basis),
            "random_object_state": q_by_rows(dev_keys, rand_dev),
            "train_state_mismatch": q_by_rows(dev_keys, z_dev, basis),
        }
        dev_drugs = dev_keys.full_inchikey.astype(str).to_numpy()
        dev_anchor = np.stack([train_means[d] for d in dev_drugs])
        dev_predictions: dict[str, np.ndarray] = {"zero_change": np.zeros_like(dev_y), "per_drug_mean": dev_anchor}
        for arm in ARMS:
            pred = np.empty_like(dev_y)
            output_basis = random_basis if arm == "random_gene_map" else basis
            for drug in drugs:
                positions = np.flatnonzero(dev_drugs == drug)
                pred[positions] = predict(dev_q[arm][positions], output_basis, coefficients[arm][drug], dev_anchor[positions])
            dev_predictions[arm] = pred

        score_docs: dict[str, dict[str, dict[str, float | None]]] = {"train": {}, "inner_dev": {}}
        detailed_scores: dict[str, dict[str, dict[str, dict[str, float | None]]]] = {"train": {}, "inner_dev": {}}
        for method in METHODS:
            score_docs["train"][method] = summarize(train_scores[method])
            detailed_scores["train"][method] = train_scores[method]
            detailed_scores["inner_dev"][method] = object_scores(dev_keys, dev_y, dev_predictions[method], variance)
            score_docs["inner_dev"][method] = summarize(detailed_scores["inner_dev"][method])
        for role, keys, predictions in (("train", train_keys, train_predictions), ("inner_dev", dev_keys, dev_predictions)):
            saved_prefix = "train" if role == "train" else "inner_dev"
            saved_keys = archive[f"{saved_prefix}_condition_key"].astype(str)
            if not np.array_equal(saved_keys, keys.full_condition_key.astype(str).to_numpy()):
                raise ValueError(f"root readback: {role} condition keys differ")
            for method in METHODS:
                saved = archive[f"{saved_prefix}_prediction_{method}"].astype(np.float64)
                if not np.allclose(saved, predictions[method], rtol=2e-6, atol=2e-6):
                    raise ValueError(f"root readback: keyed prediction mismatch for {role}/{method}")
        recorded_metrics = {role: metric_doc[f"{role}_metrics"] for role in ("train", "inner_dev")}
        for role in ("train", "inner_dev"):
            for method in METHODS:
                recorded = recorded_metrics[role][method]
                if any(not compare_metric(recorded.get(f"object_macro_{metric}"), score_docs[role][method][metric])
                       for metric in METRICS):
                    raise ValueError(f"root readback: primary metric mismatch for {role}/{method}")
                expected_conditions = 419 if role == "train" else 93
                expected_objects = 110 if role == "train" else 24
                if (recorded.get("conditions") != expected_conditions or recorded.get("objects") != expected_objects
                        or recorded.get("genes") != 978 or recorded.get("condition_gene_values") != expected_conditions * 978
                        or recorded.get("finite_condition_gene_values") != expected_conditions * 978
                        or recorded.get("nonfinite_values") != 0 or recorded.get("gene_variance_fit_rows") != 419
                        or recorded.get("gene_variance_usable_genes") != 978):
                    raise ValueError(f"root readback: denominator mismatch for {role}/{method}")
                recorded_objects = recorded.get("object_scores", {})
                if set(recorded_objects) != set(detailed_scores[role][method]):
                    raise ValueError(f"root readback: object score keys mismatch for {role}/{method}")
                for obj, expected_scores in detailed_scores[role][method].items():
                    if any(not compare_metric(recorded_objects[obj].get(metric), expected_scores[metric])
                           for metric in METRICS):
                        raise ValueError(f"root readback: object score mismatch for {role}/{method}/{obj}")
        paired_saved = json.loads(files["paired_object_bootstrap.json"].read_text(encoding="utf-8"))
        paired_root: dict[str, Any] = {}
        bootstrap_reps = int(cfg["selection"]["bootstrap_replicates"])
        bootstrap_seed = int(cfg["selection"]["bootstrap_seed"])
        for role in ("train", "inner_dev"):
            rng = np.random.default_rng(bootstrap_seed)
            objects = sorted(detailed_scores[role]["per_drug_mean"])
            draws = rng.integers(0, len(objects), size=(bootstrap_reps, len(objects)))
            baseline = np.asarray([detailed_scores[role]["per_drug_mean"][obj]["nmse"] for obj in objects])
            paired_root[role] = {"unit": "source_object_id", "paired_objects": len(objects),
                "replicates": bootstrap_reps, "seed": bootstrap_seed,
                "interval_interpretation": "descriptive fixed-split uncertainty only"}
            for method in METHODS:
                candidate_values = np.asarray([detailed_scores[role][method][obj]["nmse"] for obj in objects])
                differences = baseline - candidate_values
                sampled_mean = differences[draws].mean(axis=1)
                paired_root[role][method] = {
                    "mean_absolute_nmse_improvement_over_per_drug_mean": float(differences.mean()),
                    "relative_nmse_improvement_over_per_drug_mean": float(differences.mean() / baseline.mean()),
                    "objects_with_lower_nmse_than_per_drug_mean": int((differences > 0).sum()),
                    "paired_95pct_percentile_interval_absolute_improvement": [float(v) for v in np.quantile(sampled_mean, [0.025, 0.975])],
                    "positive_favors_method": True}
        if not compare_metric(paired_saved, paired_root, atol=1e-10):
            raise ValueError("root readback: paired object bootstrap mismatch")
        gate = metric_doc["registered_selection_gate"]
        candidate = recorded_metrics["inner_dev"]["functional_hallmark"]
        mean = recorded_metrics["inner_dev"]["per_drug_mean"]
        relative = (mean["object_macro_nmse"] - candidate["object_macro_nmse"]) / mean["object_macro_nmse"]
        if abs(relative - float(gate["relative_nmse_improvement_over_per_drug_mean"])) > 1e-10:
            raise ValueError("root readback: selection-gate relative improvement mismatch")
        boot_candidate = paired_root["inner_dev"]["functional_hallmark"]
        controls = {arm: float(recorded_metrics["inner_dev"][arm]["object_macro_nmse"])
                    for arm in ARMS if arm != "functional_hallmark"}
        amp_min, amp_max = map(float, cfg["selection"]["amplitude_ratio_range"])
        expected_gate_pass = bool(
            relative >= float(cfg["selection"]["minimum_inner_dev_relative_nmse_improvement_over_drug_mean"])
            and boot_candidate["paired_95pct_percentile_interval_absolute_improvement"][0] > 0
            and boot_candidate["objects_with_lower_nmse_than_per_drug_mean"]
                >= int(cfg["selection"]["minimum_inner_dev_objects_with_lower_nmse_than_drug_mean"])
            and amp_min <= candidate["object_macro_amplitude_ratio"] <= amp_max
            and all(candidate["object_macro_nmse"] < value for value in controls.values()))
        if bool(gate.get("gate_pass")) != expected_gate_pass:
            raise ValueError("root readback: full preregistered candidate gate recomputation mismatch")
        expected_receipt = {
            "status": "PASS", "experiment_id": "EXP-061", "run_id": args.run_id,
            "attempt_id": args.attempt_id, "runner_close_status": close.get("status"),
            "denominators": {"train_conditions": 419, "train_objects": 110, "inner_dev_conditions": 93,
                "inner_dev_objects": 24, "object_overlap": 0, "full_inchikeys": 5, "genes": 978},
            "source_train_values_sha256": hashlib.sha256(train_y.astype("<f8").tobytes()).hexdigest(),
            "source_inner_dev_values_sha256": hashlib.sha256(dev_y.astype("<f8").tobytes()).hexdigest(),
            "source_train_condition_key_sha256": axis_sha(train_keys.full_condition_key.astype(str).to_numpy()),
            "source_inner_dev_condition_key_sha256": axis_sha(dev_keys.full_condition_key.astype(str).to_numpy()),
            "gene_axis_sha256": axis_sha(genes), "gene_symbol_axis_sha256": axis_sha(symbols),
            "root_reconstructed_functional_basis": basis_root,
            "coefficient_and_prediction_reconstruction": "PASS_ALL_FOUR_ARMS_AND_TWO_BASELINES",
            "recorded_primary_metrics_match": True, "recorded_object_metrics_match": True,
            "paired_bootstrap_recomputed": True, "selection_gate_metric_recomputed": True,
            "inner_dev_opened_after_train_coefficient_reconstruction": True,
            "forbidden_numeric_inputs_opened": [], "cuda_visible_devices": "",
            "gpu_hours": 0, "cpu_wall_seconds": time.monotonic() - started,
            "process_cpu_seconds": time.process_time() - process_started,
            "checked_files_sha256": {name: file_sha(path) for name, path in files.items()},
            "root_verifier_sha256": file_sha(Path(__file__).resolve()),
            "recorded_candidate_gate_pass": bool(gate["gate_pass"]),
            "inner_dev_primary_metrics": recorded_metrics["inner_dev"],
            "accounting_note": "Root readback CPU is charged to the EXP-061 CPU package cap and inherited EXP-055 CPU cap."}
        out = attempt / "root_independent_readback.json"
        temp = out.with_suffix(".json.tmp")
        temp.write_text(json.dumps(expected_receipt, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temp.replace(out)
        print(json.dumps(expected_receipt, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        failure = {"status": "FAILED_ROOT_READBACK", "experiment_id": "EXP-061", "run_id": args.run_id,
            "attempt_id": args.attempt_id, "exception_type": type(exc).__name__, "exception": str(exc),
            "traceback": traceback.format_exc(), "cpu_wall_seconds": time.monotonic() - started,
            "process_cpu_seconds": time.process_time() - process_started,
            "forbidden_numeric_inputs_opened": [], "gpu_hours": 0}
        existing_failures = sorted(attempt.glob("root_independent_readback_failure_v*.json"))
        failure_version = len(existing_failures) + 1
        temp = attempt / f"root_independent_readback_failure_v{failure_version}.json.tmp"
        temp.write_text(json.dumps(failure, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        temp.replace(attempt / f"root_independent_readback_failure_v{failure_version}.json")
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--attempt-id", default="fit_attempt_001")
    return _run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
