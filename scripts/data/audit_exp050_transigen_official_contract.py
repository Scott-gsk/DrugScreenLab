"""Freeze 512 response-blind EXP-050 TranSiGen train keys and verify their data contract.

The freeze phase reads only the prepared condition table and train121 membership.
The values phase is separately invoked after the key table has been frozen, and
reads only those 512 train rows from the prepared matrices and source h5ad.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "20261001_transigen_official_contract_audit_v1"
OUT = ROOT / "artifacts/experiments/EXP-050" / RUN_ID
PREPARED = ROOT / "artifacts/experiments/EXP-037/20260928_lincs_scale_16_121/prepared"
SOURCE_H5AD = ROOT / "data/external/xpert_source/processed_data/l1000_sdst_78453.h5ad"
TARGET_GENES = ROOT / "data/processed/lincs/GSE92742/continuous_contract_v2/genes.json"
PREFIX = "EXP050-TRANSIGEN-OFFICIAL-CONTRACT-512-v1|"
KEY_FIELDS = ["condition_id", "source_row", "cell_id", "pert_id", "pert_dose", "pert_time"]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar_text(value: Any) -> str:
    if pd.isna(value):
        raise ValueError("condition key contains a null value")
    if isinstance(value, (float, np.floating, int, np.integer)):
        try:
            number = Decimal(str(value))
        except InvalidOperation as exc:
            raise ValueError(f"invalid numeric identity value: {value!r}") from exc
        return format(number.normalize(), "f")
    return str(value)


def canonical_key(row: pd.Series) -> str:
    values = [scalar_text(row[name]) for name in KEY_FIELDS]
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def freeze() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    keys_path = OUT / "selected_train512_condition_keys.tsv"
    if keys_path.exists():
        raise FileExistsError(f"refusing to replace frozen keys: {keys_path}")
    manifest_path = PREPARED / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "PREPARED_TRAIN_VALIDATION_ONLY":
        raise ValueError("prepared training manifest status is not the approved frozen state")
    if manifest.get("official_test_responses_read") is not False:
        raise ValueError("prepared manifest does not confirm the official test response boundary")

    conditions = pd.read_parquet(PREPARED / "conditions.parquet")
    required = set(KEY_FIELDS)
    if not required.issubset(conditions.columns):
        raise ValueError(f"prepared condition table lacks key fields: {sorted(required - set(conditions.columns))}")
    if len(conditions) != 37048:
        raise ValueError(f"prepared condition-table row count changed: {len(conditions)}")
    if not np.array_equal(pd.to_numeric(conditions["source_row"]).to_numpy(dtype=np.int64),
                          pd.to_numeric(conditions["condition_id"]).to_numpy(dtype=np.int64)):
        raise ValueError("condition_id does not match its registered source_row identity")
    if conditions[KEY_FIELDS].isna().any().any():
        raise ValueError("prepared condition identity contains null values")

    train_rows = np.load(PREPARED / "train121.npy", allow_pickle=False)
    if train_rows.ndim != 1 or not np.issubdtype(train_rows.dtype, np.integer):
        raise ValueError("train121 membership must be a one-dimensional integer row-index array")
    train_rows = train_rows.astype(np.int64, copy=False)
    expected_train_n = int(manifest["n_train121"])
    if len(train_rows) != expected_train_n or len(np.unique(train_rows)) != expected_train_n:
        raise ValueError("train121 membership count or uniqueness differs from its manifest")
    if train_rows.min(initial=0) < 0 or train_rows.max(initial=-1) >= len(conditions):
        raise ValueError("train121 membership contains an out-of-range row")

    train = conditions.iloc[train_rows].copy()
    train["canonical_condition_key"] = train.apply(canonical_key, axis=1)
    if train["canonical_condition_key"].duplicated().any():
        duplicates = train.loc[train["canonical_condition_key"].duplicated(False), KEY_FIELDS]
        raise ValueError(f"duplicate unique-condition identity keys in train121: {duplicates.head().to_dict('records')}")
    train["selection_sha256"] = train["canonical_condition_key"].map(
        lambda key: sha256_bytes((PREFIX + key).encode("utf-8"))
    )
    train = train.sort_values(["selection_sha256", "canonical_condition_key"], kind="mergesort")
    selected = train.head(512).copy()
    if len(selected) != 512:
        raise ValueError(f"only {len(selected)} unique train condition keys are available")
    selected["prepared_row_index"] = selected.index.astype(np.int64)
    selected = selected.reset_index(drop=True)
    selected.insert(0, "selection_rank", np.arange(1, len(selected) + 1))
    selected.to_csv(keys_path, sep="\t", index=False, lineterminator="\n")

    selected_condition_ids = selected["condition_id"].astype(str).tolist()
    receipt = {
        "experiment_id": "EXP-050",
        "run_id": RUN_ID,
        "phase": "RESPONSE_BLIND_KEYS_FROZEN",
        "selection_prefix": PREFIX,
        "selection_rule": "Within only train121-admitted condition rows, serialize [condition_id, source_row, cell_id, pert_id, pert_dose, pert_time] as compact UTF-8 JSON string; rank SHA256(prefix + canonical key) ascending, ties by canonical key; keep first 512. No train response/control matrix was opened during key freeze.",
        "condition_grain": "one frozen Xpert SDST source row at the registered 10 uM x 24 h condition; dose/time units come from the frozen EXP-037 source plan, while the prepared condition table itself contains numeric pert_dose and pert_time only",
        "identity_key_fields": KEY_FIELDS,
        "prepared_root": str(PREPARED.relative_to(ROOT)).replace("\\", "/"),
        "prepared_manifest_sha256": sha256_file(manifest_path),
        "condition_table_sha256": sha256_file(PREPARED / "conditions.parquet"),
        "train121_sha256": sha256_file(PREPARED / "train121.npy"),
        "condition_table_rows": int(len(conditions)),
        "admitted_train_unique_condition_keys": int(len(train_rows)),
        "non_train_rows_excluded_without_role_breakdown": int(len(conditions) - len(train_rows)),
        "train_rows_not_selected_by_fixed_hash_rank": int(len(train_rows) - len(selected)),
        "selected_condition_count": int(len(selected)),
        "selected_object_cell_count": int(selected["cell_id"].nunique()),
        "selected_compound_pert_id_count": int(selected["pert_id"].nunique()),
        "train_object_cell_count": int(train["cell_id"].nunique()),
        "train_compound_pert_id_count": int(train["pert_id"].nunique()),
        "selected_key_sha256": sha256_file(keys_path),
        "selected_condition_id_order_sha256": sha256_bytes("\n".join(selected_condition_ids).encode("utf-8")),
        "selected_train_response_or_expression_arrays_opened": False,
        "validation_or_locked_membership_opened": False,
        "selected_counts_by_pert_id": {str(k): int(v) for k, v in selected.groupby("pert_id").size().sort_index().items()},
        "selected_counts_by_cell_id": {str(k): int(v) for k, v in selected.groupby("cell_id").size().sort_index().items()},
    }
    receipt["selected_key_canonical_list_sha256"] = sha256_bytes(
        "\n".join(selected["canonical_condition_key"].astype(str)).encode("utf-8")
    )
    save_json(OUT / "response_blind_key_freeze_receipt.json", receipt)
    (OUT / "selected_condition_keys.sha256").write_text(receipt["selected_key_sha256"] + "  selected_train512_condition_keys.tsv\n", encoding="ascii")
    return receipt


def contiguous_row_read(dataset: h5py.Dataset, rows: np.ndarray, block: int = 128) -> np.ndarray:
    result = np.empty((len(rows), dataset.shape[1]), dtype=np.float32)
    if len(rows) and (np.diff(rows) <= 0).any():
        raise ValueError("h5py source rows must be sorted and unique")
    for start in range(0, len(rows), block):
        result[start:start + block] = dataset[rows[start:start + block]]
    return result


def values() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    receipt_path = OUT / "response_blind_key_freeze_receipt.json"
    keys_path = OUT / "selected_train512_condition_keys.tsv"
    if not receipt_path.is_file() or not keys_path.is_file():
        raise FileNotFoundError("freeze phase must complete before the values phase")
    freeze_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if freeze_receipt.get("selected_train_response_or_expression_arrays_opened") is not False:
        raise ValueError("freeze receipt violates the response-blind boundary")
    if sha256_file(keys_path) != freeze_receipt["selected_key_sha256"]:
        raise ValueError("frozen key table digest changed")
    selected = pd.read_csv(keys_path, sep="\t", dtype={"condition_id": str, "cell_id": str, "pert_id": str})
    if len(selected) != 512 or selected["condition_id"].duplicated().any():
        raise ValueError("frozen condition keys are not exactly 512 unique source rows")
    row_index = selected["prepared_row_index"].to_numpy(dtype=np.int64)
    source_rows = selected["source_row"].to_numpy(dtype=np.int64)
    if not np.array_equal(selected["condition_id"].astype(str).to_numpy(), source_rows.astype(str)):
        raise ValueError("condition_id is not bound to its original source row identity")
    if len(np.unique(row_index)) != len(row_index) or row_index.min(initial=0) < 0 or row_index.max(initial=-1) >= 37048:
        raise ValueError("frozen prepared row indices are not unique/in range")

    control_map = np.load(PREPARED / "control.npy", mmap_mode="r", allow_pickle=False)
    delta_map = np.load(PREPARED / "delta.npy", mmap_mode="r", allow_pickle=False)
    source_to_target = np.load(PREPARED / "source_to_target.npy", allow_pickle=False).astype(np.int64)
    if control_map.shape != (37048, 978) or delta_map.shape != control_map.shape:
        raise ValueError(f"prepared expression dimensions changed: {control_map.shape} / {delta_map.shape}")
    if source_to_target.shape != (978,) or not np.array_equal(np.sort(source_to_target), np.arange(978)):
        raise ValueError("prepared source_to_target gene map is not a permutation of the 978-gene axis")

    control = np.asarray(control_map[row_index], dtype=np.float32)
    delta = np.asarray(delta_map[row_index], dtype=np.float32)
    treated = control + delta
    if not np.isfinite(control).all() or not np.isfinite(delta).all() or not np.isfinite(treated).all():
        raise ValueError("selected training inputs or treated reconstruction contain non-finite values")

    target_gene_ids = [str(x) for x in json.loads(TARGET_GENES.read_text(encoding="utf-8"))]
    if len(target_gene_ids) != 978 or len(set(target_gene_ids)) != 978:
        raise ValueError("registered target gene axis is not 978 unique IDs")
    with h5py.File(SOURCE_H5AD, "r") as h5:
        source_gene_ids = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in h5["var/gene_id"][:]]
        if len(source_gene_ids) != 978 or [target_gene_ids[i] for i in source_to_target] != source_gene_ids:
            raise ValueError("prepared source_to_target map does not reproduce the source h5ad gene-axis identity")
        inverse = np.argsort(source_to_target)
        source_order = np.argsort(source_rows)
        inverse_order = np.argsort(source_order)
        sorted_source_rows = source_rows[source_order]
        source_control_sorted = contiguous_row_read(h5["obsm/X_ctl"], sorted_source_rows)[:, inverse]
        source_treated_sorted = contiguous_row_read(h5["X"], sorted_source_rows)[:, inverse]
        source_control = source_control_sorted[inverse_order]
        source_treated = source_treated_sorted[inverse_order]
    source_delta = source_treated - source_control
    if not np.allclose(control, source_control, rtol=1e-6, atol=1e-6):
        raise ValueError("prepared control rows differ from the corresponding source h5ad controls")
    if not np.allclose(treated, source_treated, rtol=1e-6, atol=1e-6):
        raise ValueError("control + delta differs from the source h5ad treated values")
    if not np.allclose(delta, source_delta, rtol=1e-6, atol=1e-6):
        raise ValueError("prepared delta differs from source h5ad treated-minus-control")

    archive = OUT / "selected_train512_control_delta_treated.npz"
    if archive.exists():
        raise FileExistsError(f"refusing to replace keyed values artifact: {archive}")
    np.savez_compressed(archive, condition_id=selected["condition_id"].to_numpy(dtype="U"),
                        source_row=source_rows, gene_id=np.asarray(target_gene_ids, dtype="U"),
                        control=control, delta=delta, treated=treated)
    with np.load(archive, allow_pickle=False) as restored:
        for name, expected in (("condition_id", selected["condition_id"].to_numpy(dtype="U")),
                               ("source_row", source_rows), ("gene_id", np.asarray(target_gene_ids, dtype="U")),
                               ("control", control), ("delta", delta), ("treated", treated)):
            if not np.array_equal(restored[name], expected):
                raise ValueError(f"keyed matrix read/write round trip failed for {name}")
        reconstruction_error = float(np.max(np.abs(restored["treated"] - (restored["control"] + restored["delta"]))))

    range_rows: list[dict[str, Any]] = []
    for name, values_array in (("control", control), ("delta", delta), ("treated", treated)):
        flat = values_array.astype(np.float64, copy=False).ravel()
        range_rows.append({"matrix": name, "n_values": int(flat.size), "min": float(np.min(flat)),
                           "q01": float(np.quantile(flat, 0.01)), "q25": float(np.quantile(flat, 0.25)),
                           "median": float(np.median(flat)), "q75": float(np.quantile(flat, 0.75)),
                           "q99": float(np.quantile(flat, 0.99)), "max": float(np.max(flat)),
                           "negative_value_count": int(np.count_nonzero(flat < 0)),
                           "nonfinite_value_count": int(np.count_nonzero(~np.isfinite(flat)))})
    pd.DataFrame(range_rows).to_csv(OUT / "selected_train512_value_ranges.tsv", sep="\t", index=False, lineterminator="\n")
    per_key = []
    for i, row in selected.iterrows():
        err = treated[i] - source_treated[i]
        per_key.append({"selection_rank": int(row.selection_rank), "condition_id": str(row.condition_id),
                        "source_row": int(row.source_row), "cell_id": str(row.cell_id), "pert_id": str(row.pert_id),
                        "pert_dose": float(row.pert_dose), "pert_time": float(row.pert_time),
                        "control_sha256": sha256_bytes(control[i].tobytes()),
                        "delta_sha256": sha256_bytes(delta[i].tobytes()),
                        "treated_sha256": sha256_bytes(treated[i].tobytes()),
                        "source_treated_max_abs_diff": float(np.max(np.abs(err))),
                        "condition_delta_min": float(np.min(delta[i])),
                        "condition_delta_max": float(np.max(delta[i]))})
    pd.DataFrame(per_key).to_csv(OUT / "selected_train512_value_identity.tsv", sep="\t", index=False, lineterminator="\n")

    result = {
        "experiment_id": "EXP-050",
        "run_id": RUN_ID,
        "phase": "TRAIN512_VALUE_CONTRACT_VERIFIED",
        "selected_key_sha256": freeze_receipt["selected_key_sha256"],
        "prepared_control_sha256": sha256_file(PREPARED / "control.npy"),
        "prepared_delta_sha256": sha256_file(PREPARED / "delta.npy"),
        "prepared_gene_map_sha256": sha256_file(PREPARED / "source_to_target.npy"),
        "registered_target_gene_ids_sha256": sha256_file(TARGET_GENES),
        "source_h5ad_sha256": sha256_file(SOURCE_H5AD),
        "selected_train_response_rows_read": 512,
        "validation_or_locked_response_arrays_opened": False,
        "prepared_shapes": {"control": list(control_map.shape), "delta": list(delta_map.shape)},
        "selected_control_delta_treated_shape": list(control.shape),
        "selected_source_row_identity_exact": True,
        "source_gene_axis_identity_exact": True,
        "axis_length": 978,
        "source_to_target_is_bijection": True,
        "prepared_values_match_source_h5ad_selected_rows": True,
        "control_delta_treated_finite": True,
        "treated_equals_control_plus_delta_max_abs_error": reconstruction_error,
        "value_round_trip_exact": True,
        "value_archive_sha256": sha256_file(archive),
        "range_table_sha256": sha256_file(OUT / "selected_train512_value_ranges.tsv"),
        "identity_table_sha256": sha256_file(OUT / "selected_train512_value_identity.tsv"),
        "range_summary": range_rows,
        "interpretation_boundary": "These ranges describe the frozen train121 Xpert SDST 10 uM x 24 h source data. They do not establish the scale compatibility of the upstream official processed_data.h5, whose producer and units remain undocumented/unavailable.",
    }
    save_json(OUT / "train512_value_contract_receipt.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["freeze", "values"], required=True)
    args = parser.parse_args()
    result = freeze() if args.phase == "freeze" else values()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
