"""EXP-037 XPert strict-fold adapter and one-batch gradient smoke.

This entrypoint never trains a candidate, selects a model, loads a checkpoint,
scores an outer fold, or reads outer-test response values. Its smoke mode reads
one supported-drug batch from an inner training partition and runs a single
forward/backward pass without an optimizer.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
import torch
import yaml


EXPERIMENT_ID = "EXP-037"
SPLIT_RELATIVE_PATH = Path("data/splits/lincs_cell_splits_20260922_v2.json")
SPLIT_V2_SHA256 = "e474cede3d7474ad87c5685850726e9a8db813eabbd775e552fd09899df753d7"
SPLIT_V1_SHA256 = "d7f00cdd01454002924d34451a3d61fc4aff499c8ca90f799ac6544017fbeaca"
DELTA_V2_SHA256 = "29317d50b44933452126ddd213017709d54b37f6c0dcc2f403cf5ac5ff8112fb"
XPERT_AUDIT_SHA256 = "04d46c22c48e295574ea97a333aecc9d1b862ec9e051cb5ac5b9fc63c56a751a"
XPERT_GENE_MAP_SHA256 = "6e3cfef9f88bbfadafb67e51d7f11c22fd192216d88795a25cefd159aa3d10fb"
GENE_COUNT = 978
EXPECTED_FOLDS = 5
EXPECTED_INNER_FOLDS = 3
EXPECTED_CELLS = 70
EXPECTED_TARGET_DRUGS = 20336
EXPECTED_XPERT_SHARED_DRUGS = 6994
FORMAL_PLAN_REQUIRED_FIELDS = frozenset({
    "experiment_id",
    "split_sha256",
    "condition_delta_sha256",
    "condition_manifest_sha256",
    "input_manifest_sha256",
    "xpert_asset_audit_sha256",
    "shared_drug_ids_sha256",
    "shared_drug_comparison_id",
    "initialization_seeds",
    "batch_size",
    "candidates",
    "per_run_wallclock_hours",
    "experiment_gpu_hours",
})
FORMAL_CANDIDATE_REQUIRED_FIELDS = frozenset({
    "candidate_id",
    "model_config_sha256",
    "optimizer",
    "learning_rate",
    "weight_decay",
    "max_epochs",
    "patience",
    "loss_weights",
})

CONTRACT_RELATIVE_PATH = Path(
    "data/processed/lincs/GSE92742/continuous_contract_v2"
)
CONDITION_RELATIVE_PATH = Path(
    "data/processed/lincs/GSE92742/continuous_conditions_v2"
)
INPUT_RELATIVE_PATH = Path(
    "data/processed/lincs/GSE92742/continuous_inputs_v2"
)
XPERT_RELATIVE_PATH = Path("data/external/xpert_source")
XPERT_AUDIT_RELATIVE_PATH = Path(
    "artifacts/experiments/EXP-037/20260923_xpert_asset_audit"
)


class ContractError(ValueError):
    """A frozen path, identity, split, or matrix-axis contract failed."""


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    split: Path
    contract_dir: Path
    condition_dir: Path
    input_dir: Path
    xpert_dir: Path
    audit_dir: Path

    @classmethod
    def from_root(cls, root: Path) -> "ProjectPaths":
        root = root.resolve(strict=True)
        if not root.is_dir():
            raise ContractError(f"project root is not a directory: {root}")
        return cls(
            root=root,
            split=root / SPLIT_RELATIVE_PATH,
            contract_dir=root / CONTRACT_RELATIVE_PATH,
            condition_dir=root / CONDITION_RELATIVE_PATH,
            input_dir=root / INPUT_RELATIVE_PATH,
            xpert_dir=root / XPERT_RELATIVE_PATH,
            audit_dir=root / XPERT_AUDIT_RELATIVE_PATH,
        )

    def validate_layout(self) -> None:
        expected = {
            "split": SPLIT_RELATIVE_PATH,
            "contract_dir": CONTRACT_RELATIVE_PATH,
            "condition_dir": CONDITION_RELATIVE_PATH,
            "input_dir": INPUT_RELATIVE_PATH,
            "xpert_dir": XPERT_RELATIVE_PATH,
            "audit_dir": XPERT_AUDIT_RELATIVE_PATH,
        }
        for name, relative in expected.items():
            actual = getattr(self, name).resolve(strict=False)
            canonical = (self.root / relative).resolve(strict=False)
            if actual != canonical:
                raise ContractError(f"{name} escaped its frozen project path")
        required = (
            self.split,
            self.contract_dir / "quality_report.json",
            self.contract_dir / "genes.json",
            self.contract_dir / "treatments.parquet",
            self.contract_dir / "controls.parquet",
            self.condition_dir / "manifest.json",
            self.condition_dir / "conditions.parquet",
            self.condition_dir / "condition_delta.npy",
            self.input_dir / "manifest.json",
            self.input_dir / "condition_control.npy",
            self.input_dir / "condition_exposure.parquet",
            self.input_dir / "drugs.parquet",
            self.input_dir / "drug_morgan2048.npy",
            self.xpert_dir / "models/model_XPert.py",
            self.xpert_dir / "models/model_utils.py",
            self.xpert_dir / "utils.py",
            self.xpert_dir / "datasets/MyDataset.py",
            self.xpert_dir / "configs/config_l1000.yaml",
            self.xpert_dir / "processed_data/l1000_sdst_drugs_info_8432.csv",
            self.xpert_dir / "processed_data/all_drugs_unimol_arr.npy",
            self.xpert_dir / "processed_data/PPI_gene_vector_128d.npy",
            self.xpert_dir / "HG_data/saved_embedding/HG_drug_embeddings.npy",
            self.audit_dir / "asset_audit.json",
            self.audit_dir / "source_to_target_gene_indices.json",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise ContractError("required EXP-037/XPert contract file is missing: " + missing[0])


@dataclass(frozen=True)
class InnerFold:
    inner_fold: int
    train_ids: frozenset[str]
    validation_ids: frozenset[str]


@dataclass(frozen=True)
class SplitFold:
    fold: int
    outer_train_ids: frozenset[str]
    outer_test_ids: frozenset[str]
    inner_folds: tuple[InnerFold, ...]


@dataclass(frozen=True)
class FoldRows:
    role: Literal["outer_train", "outer_test", "inner_train", "inner_validation"]
    positions: np.ndarray


@dataclass(frozen=True)
class DrugFeatureIndex:
    target_index_by_pert_id: dict[str, int]
    node_index_by_pert_id: dict[str, int]
    exact_shared_pert_ids: frozenset[str]
    n_structure_mismatches: int


@dataclass(frozen=True)
class XpertBatch:
    model_inputs: tuple[torch.Tensor, ...]
    target_delta: torch.Tensor
    target_cell_ids: tuple[str, ...]
    target_pert_ids: tuple[str, ...]


@dataclass
class FoldDataLoader:
    keys: pd.DataFrame
    conditions: pd.DataFrame
    exposure: pd.DataFrame
    delta: np.ndarray
    control: np.ndarray
    unimol: np.ndarray
    drug_hg: np.ndarray
    source_to_target_gene_indices: np.ndarray
    drug_index: DrugFeatureIndex
    split_fold: SplitFold
    config: dict[str, Any]

    def _rows(self, cells: frozenset[str], allowed_drugs: frozenset[str],
              role: Literal["outer_train", "outer_test", "inner_train", "inner_validation"],
              train_drugs: frozenset[str] | None = None) -> FoldRows:
        mask = self.keys["cell_id"].astype(str).isin(cells).to_numpy()
        mask &= self.keys["pert_id"].astype(str).isin(allowed_drugs).to_numpy()
        if train_drugs is not None:
            mask &= self.keys["pert_id"].astype(str).isin(train_drugs).to_numpy()
        indices = np.flatnonzero(mask).astype(np.int64)
        if not len(indices):
            raise ContractError(f"{role} has no rows after exact drug-identity filtering")
        return FoldRows(role=role, positions=indices)

    def fold_rows(self, *, shared_drug_ids: frozenset[str]) -> dict[str, Any]:
        if not shared_drug_ids:
            raise ContractError("an explicit shared-drug set is required")
        known = set(self.drug_index.exact_shared_pert_ids)
        if not set(shared_drug_ids).issubset(known):
            raise ContractError("shared-drug set includes an unavailable or structure-mismatched XPert drug")
        outer_train_all = self._rows(
            self.split_fold.outer_train_ids, shared_drug_ids, "outer_train")
        train_drugs = frozenset(self.keys.iloc[outer_train_all.positions]["pert_id"].astype(str))
        outer_test = self._rows(
            self.split_fold.outer_test_ids, shared_drug_ids, "outer_test", train_drugs)
        inner_rows = []
        for item in self.split_fold.inner_folds:
            inner_train = self._rows(item.train_ids, shared_drug_ids, "inner_train")
            inner_train_drugs = frozenset(
                self.keys.iloc[inner_train.positions]["pert_id"].astype(str))
            validation = self._rows(
                item.validation_ids, shared_drug_ids, "inner_validation", inner_train_drugs)
            inner_rows.append({"split": item, "train": inner_train, "validation": validation})
        return {"outer_train": outer_train_all, "outer_test": outer_test,
                "inner_folds": tuple(inner_rows)}

    def batch(self, rows: FoldRows, positions: np.ndarray, *, device: torch.device) -> XpertBatch:
        if rows.role == "outer_test":
            raise ContractError("outer-test response values are locked in this adapter")
        if rows.role not in ("outer_train", "inner_train", "inner_validation"):
            raise ContractError(f"unsupported data role: {rows.role}")
        positions = np.asarray(positions, dtype=np.int64)
        if positions.ndim != 1 or not len(positions):
            raise ContractError("batch positions must be a non-empty one-dimensional index")
        if not np.isin(positions, rows.positions).all():
            raise ContractError("batch includes rows outside its declared fold partition")

        # The only delta indexing in this module is guarded by the role check above.
        control_target_order = np.asarray(self.control[positions], dtype=np.float32)
        delta_target_order = np.asarray(self.delta[positions], dtype=np.float32)
        if (not np.isfinite(control_target_order).all()
                or not np.isfinite(delta_target_order).all()):
            raise ContractError("selected training-side batch contains non-finite expression values")
        control_source_order = control_target_order[:, self.source_to_target_gene_indices]
        delta_source_order = delta_target_order[:, self.source_to_target_gene_indices]
        treatment_source_order = control_source_order + delta_source_order

        source_root = self.config["_source_root"]
        digitize = load_upstream_functions(
            source_root / "datasets/MyDataset.py", ("_digitize",))["_digitize"]
        low = float(self.config["dataset"]["min_value"])
        high = float(self.config["dataset"]["max_value"])
        n_bins = int(self.config["dataset"]["n_bins"])
        bins = np.quantile(np.asarray([low, high]), np.linspace(0.0, 1.0, n_bins - 1))
        control_binned = digitize(control_source_order, bins, side="one").astype(np.int64)
        treatment_binned = digitize(treatment_source_order, bins, side="one").astype(np.int64)

        ids = self.keys.iloc[positions]
        pert_ids = ids["pert_id"].astype(str).tolist()
        node_indices = np.asarray(
            [self.drug_index.node_index_by_pert_id[drug] for drug in pert_ids],
            dtype=np.int64,
        )
        if not np.isfinite(self.unimol[node_indices]).all():
            raise ContractError("XPert UniMol feature batch contains non-finite values")
        if not np.isfinite(self.drug_hg[node_indices]).all():
            raise ContractError("XPert drug-graph feature batch contains non-finite values")
        drug_features = np.asarray(self.unimol[node_indices], dtype=np.float32)

        dose_values = pd.to_numeric(ids["pert_dose"], errors="raise").to_numpy(np.float64)
        dose_bin = load_upstream_functions(
            source_root / "datasets/MyDataset.py",
            ("assign_pert_dose_corrected",),
        )["assign_pert_dose_corrected"]
        dose_indices = np.asarray([dose_bin(float(value)) for value in dose_values],
                                  dtype=np.int64)
        time_values = pd.to_numeric(ids["pert_time"], errors="raise").to_numpy(np.float64)
        time_map = {6.0: 1, 24.0: 2}
        try:
            time_indices = np.asarray([time_map[float(value)] for value in time_values],
                                      dtype=np.int64)
        except KeyError as exc:
            raise ContractError(f"unsupported XPert time value: {exc.args[0]}") from exc

        to_tensor = lambda value, dtype=None: torch.as_tensor(
            value, dtype=dtype, device=device)
        # The upstream model still accepts cell/tissue index tensors when its
        # identity head is disabled. Constant zeros carry no identity signal.
        n = len(positions)
        model_inputs = (
            to_tensor(treatment_source_order, torch.float32),
            to_tensor(control_source_order, torch.float32),
            to_tensor(treatment_binned, torch.long),
            to_tensor(control_binned, torch.long),
            to_tensor(drug_features, torch.float32),
            to_tensor(dose_indices, torch.long),
            to_tensor(time_indices, torch.long),
            to_tensor(node_indices, torch.long),
            to_tensor(np.zeros(n), torch.long),
            to_tensor(np.zeros(n), torch.long),
        )
        return XpertBatch(
            model_inputs=model_inputs,
            target_delta=to_tensor(delta_source_order, torch.float32),
            target_cell_ids=tuple(ids["cell_id"].astype(str)),
            target_pert_ids=tuple(pert_ids),
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read JSON contract {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"JSON contract must be an object: {path}")
    return value


def validate_expected_digest(expected: str | None, *, canonical: str,
                             name: str) -> None:
    if expected is None or not expected.strip():
        raise ContractError(f"explicit --expected-{name} is required")
    expected = expected.strip().lower()
    if expected == SPLIT_V1_SHA256 and name == "split-sha256":
        raise ContractError(
            "d7f00cdd... is the legacy v1 split; v2 requires the registered e474cede... digest")
    if expected != canonical:
        raise ContractError(f"unsupported expected {name}: {expected}")


def _ids(raw: Any, label: str) -> frozenset[str]:
    if not isinstance(raw, list):
        raise ContractError(f"{label} must be a list")
    values = [str(value).strip() for value in raw]
    if not values or any(not value for value in values) or len(set(values)) != len(values):
        raise ContractError(f"{label} contains empty or duplicate identities")
    return frozenset(values)


def validate_split_payload(payload: dict[str, Any], fold_index: int) -> SplitFold:
    if payload.get("protocol") != "dual_route_lincs_level3_cell_holdout_20260922_v2_parseable_structure":
        raise ContractError("split protocol is not the frozen parseable-structure v2 protocol")
    if payload.get("split_seed") != 20260922:
        raise ContractError("split seed differs from the frozen EXP-037 seed")
    if (payload.get("n_eligible_instances") != 661762
            or payload.get("n_eligible_cells") != EXPECTED_CELLS
            or payload.get("n_eligible_drugs") != EXPECTED_TARGET_DRUGS):
        raise ContractError("split cohort counts differ from the frozen v2 registry")
    folds = payload.get("folds")
    if not isinstance(folds, list) or len(folds) != EXPECTED_FOLDS:
        raise ContractError("frozen split must contain exactly five outer folds")
    parsed: list[SplitFold] = []
    seen_tests: set[str] = set()
    for position, raw in enumerate(folds):
        if raw.get("fold") != position:
            raise ContractError("outer fold keys must be the ordered integers 0 through 4")
        train_ids = _ids(raw.get("outer_train_ids"), f"fold {position} outer_train_ids")
        test_ids = _ids(raw.get("outer_test_ids"), f"fold {position} outer_test_ids")
        if len(train_ids) != 56 or len(test_ids) != 14 or train_ids & test_ids:
            raise ContractError(f"fold {position} has invalid outer cell identities")
        if seen_tests & test_ids:
            raise ContractError("a cell occurs in more than one outer test fold")
        seen_tests.update(test_ids)
        fractions = raw.get("label_fractions")
        if (not isinstance(fractions, list) or len(fractions) != 1
                or float(fractions[0].get("fraction", -1)) != 1.0):
            raise ContractError("EXP-037 requires the full-training 1.0 label-fraction split")
        inner = fractions[0].get("inner_folds")
        if not isinstance(inner, list) or len(inner) != EXPECTED_INNER_FOLDS:
            raise ContractError(f"fold {position} must contain exactly three inner folds")
        parsed_inner: list[InnerFold] = []
        validation_seen: set[str] = set()
        for inner_position, inner_raw in enumerate(inner):
            if inner_raw.get("inner_fold") != inner_position:
                raise ContractError("inner fold keys must be the ordered integers 0 through 2")
            inner_train = _ids(
                inner_raw.get("train_ids"),
                f"fold {position} inner {inner_position} train_ids",
            )
            validation = _ids(
                inner_raw.get("validation_ids"),
                f"fold {position} inner {inner_position} validation_ids",
            )
            if (inner_train & validation or inner_train | validation != train_ids
                    or validation_seen & validation):
                raise ContractError(
                    f"fold {position} inner identities are not a disjoint partition of outer training")
            validation_seen.update(validation)
            parsed_inner.append(InnerFold(
                inner_fold=inner_position,
                train_ids=inner_train,
                validation_ids=validation,
            ))
        if validation_seen != set(train_ids):
            raise ContractError(f"fold {position} inner validation folds do not cover outer training")
        parsed.append(SplitFold(
            fold=position,
            outer_train_ids=train_ids,
            outer_test_ids=test_ids,
            inner_folds=tuple(parsed_inner),
        ))
    if len(seen_tests) != EXPECTED_CELLS:
        raise ContractError("five outer test folds do not partition all 70 cell identities")
    for item in parsed:
        if item.outer_train_ids | item.outer_test_ids != seen_tests:
            raise ContractError("outer train/test identities differ across the frozen folds")
    if fold_index not in range(EXPECTED_FOLDS):
        raise ContractError("fold index must be one of 0, 1, 2, 3, or 4")
    chosen = parsed[fold_index]
    all_ids = set(chosen.outer_train_ids | chosen.outer_test_ids)
    if len(all_ids) != EXPECTED_CELLS:
        raise ContractError("selected fold does not cover all 70 cell identities")
    return chosen


def validate_split_file(path: Path, *, expected_sha256: str | None,
                        fold_index: int) -> tuple[SplitFold, str]:
    validate_expected_digest(expected_sha256, canonical=SPLIT_V2_SHA256,
                             name="split-sha256")
    if path.name != SPLIT_RELATIVE_PATH.name:
        raise ContractError("split path must be the canonical v2 file; v1 fallback is forbidden")
    actual = sha256_file(path)
    if actual != SPLIT_V2_SHA256 or actual != expected_sha256.lower():
        raise ContractError(
            f"split bytes do not match the registered v2 digest: actual={actual}")
    return validate_split_payload(read_json(path), fold_index), actual


def _valid_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ContractError(f"{label} must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ContractError(f"{label} is not hexadecimal") from exc
    return value.lower()


def _verify_file_hashes(directory: Path, entries: dict[str, Any], *,
                        skip: frozenset[str] = frozenset()) -> None:
    if not isinstance(entries, dict):
        raise ContractError(f"manifest files_sha256 is missing in {directory}")
    for name, declared in entries.items():
        digest = _valid_sha(declared, f"{directory.name}/{name} sha256")
        path = directory / name
        if not path.is_file():
            raise ContractError(f"manifest file is missing: {path}")
        if name not in skip and sha256_file(path) != digest:
            raise ContractError(f"file checksum disagrees with manifest: {path}")


def read_npy_shape(path: Path) -> tuple[int, ...]:
    """Open only the NPY header; never read the matrix body."""
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except Exception as exc:
        raise ContractError(f"cannot read numeric NPY header {path}: {exc}") from exc
    shape = tuple(int(dim) for dim in array.shape)
    del array
    return shape


def validate_gene_map(target_gene_ids: list[str], source_to_target: Any) -> np.ndarray:
    if len(target_gene_ids) != GENE_COUNT or len(set(target_gene_ids)) != GENE_COUNT:
        raise ContractError("EXP-037 target gene axis must contain 978 unique IDs")
    if not all(isinstance(gene, str) and gene.strip() for gene in target_gene_ids):
        raise ContractError("target gene axis contains an empty/non-string gene ID")
    indices = np.asarray(source_to_target)
    if (indices.shape != (GENE_COUNT,) or not np.issubdtype(indices.dtype, np.integer)
            or not np.array_equal(np.sort(indices), np.arange(GENE_COUNT))):
        raise ContractError("XPert-to-target gene map must be a complete 978-position permutation")
    return indices.astype(np.int64, copy=False)


def validate_formal_training_plan(plan: dict[str, Any]) -> None:
    """Reject incomplete or over-budget plans before a later training entrypoint."""
    missing = FORMAL_PLAN_REQUIRED_FIELDS - set(plan)
    if missing:
        raise ContractError(
            "formal training config is missing required fields: " + ", ".join(sorted(missing)))
    if plan.get("experiment_id") != EXPERIMENT_ID:
        raise ContractError("formal training config is not bound to EXP-037")
    for field, expected in (
        ("split_sha256", SPLIT_V2_SHA256),
        ("condition_delta_sha256", DELTA_V2_SHA256),
        ("xpert_asset_audit_sha256", XPERT_AUDIT_SHA256),
    ):
        if _valid_sha(plan.get(field), field) != expected:
            raise ContractError(f"formal training config {field} differs from its frozen asset")
    for field in ("condition_manifest_sha256", "input_manifest_sha256",
                  "shared_drug_ids_sha256"):
        _valid_sha(plan.get(field), field)
    if not isinstance(plan.get("shared_drug_comparison_id"), str) or not plan[
            "shared_drug_comparison_id"].strip():
        raise ContractError("formal training config needs a named shared-drug comparison")
    if plan.get("initialization_seeds") != [20260922, 20260923, 20260924]:
        raise ContractError("formal training config must use the three registered EXP-037 seeds")
    if not isinstance(plan.get("batch_size"), int) or plan["batch_size"] < 1:
        raise ContractError("formal training config needs a positive explicit batch_size")
    if (not isinstance(plan.get("per_run_wallclock_hours"), (int, float))
            or not 0 < plan["per_run_wallclock_hours"] <= 20):
        raise ContractError("formal training run limit must be explicit and at most 20 hours")
    if (not isinstance(plan.get("experiment_gpu_hours"), (int, float))
            or not 0 < plan["experiment_gpu_hours"] <= 48):
        raise ContractError("formal training GPU budget must be explicit and at most 48 hours")
    candidates = plan.get("candidates")
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 20:
        raise ContractError("formal training config needs an explicit table of 1 to 20 candidates")
    candidate_ids: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ContractError("each candidate must be a complete config object")
        missing_candidate = FORMAL_CANDIDATE_REQUIRED_FIELDS - set(candidate)
        if missing_candidate:
            raise ContractError(
                "candidate is missing required fields: " + ", ".join(sorted(missing_candidate)))
        candidate_id = candidate["candidate_id"]
        if (not isinstance(candidate_id, str) or not candidate_id.strip()
                or candidate_id in candidate_ids):
            raise ContractError("candidate IDs must be non-empty and unique")
        candidate_ids.add(candidate_id)
        _valid_sha(candidate["model_config_sha256"], f"{candidate_id}.model_config_sha256")
        if not isinstance(candidate["optimizer"], str) or not candidate["optimizer"].strip():
            raise ContractError(f"{candidate_id} needs an explicit optimizer")
        if not isinstance(candidate["loss_weights"], list) or len(candidate["loss_weights"]) != 4:
            raise ContractError(f"{candidate_id} needs four explicit official XPert loss weights")
        weights = np.asarray(candidate["loss_weights"], dtype=np.float64)
        if not np.isfinite(weights).all():
            raise ContractError(f"{candidate_id} loss weights must be finite")
        if (float(candidate["learning_rate"]) <= 0
                or float(candidate["weight_decay"]) < 0
                or int(candidate["max_epochs"]) < 1
                or int(candidate["patience"]) < 1):
            raise ContractError(f"{candidate_id} has invalid optimization/early-stop parameters")


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ContractError(f"{label} is missing required key column {missing[0]}")


def _valid_inchikey(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    parts = value.split("-")
    return (len(parts) == 3 and len(parts[0]) == 14 and len(parts[1]) == 10
            and len(parts[2]) == 1
            and all(part.isascii() and part.isalpha() and part.isupper() for part in parts))


def validate_drug_feature_index(
    target_drugs: pd.DataFrame,
    source_drugs: pd.DataFrame,
    unimol_shape: tuple[int, ...],
    drug_hg_shape: tuple[int, ...],
    *,
    hidden_size: int,
    expected_shared_count: int,
) -> DrugFeatureIndex:
    _require_columns(target_drugs, ("drug_index", "pert_id", "inchi_key"), "target drugs")
    _require_columns(source_drugs, ("pert_id", "inchi_key", "drug_node_idx"), "XPert drugs")
    if (target_drugs["pert_id"].astype(str).duplicated().any()
            or target_drugs["drug_index"].duplicated().any()):
        raise ContractError("target drug ID/index axis is not unique")
    target = target_drugs.copy()
    source = source_drugs.copy()
    target["pert_id"] = target["pert_id"].astype(str)
    source["pert_id"] = source["pert_id"].astype(str)
    if not target["inchi_key"].map(_valid_inchikey).all():
        raise ContractError("target drug axis contains an incomplete InChIKey")
    if not source["inchi_key"].map(_valid_inchikey).all():
        raise ContractError("XPert drug axis contains an incomplete InChIKey")
    target_indices = pd.to_numeric(target["drug_index"], errors="raise").to_numpy(np.int64)
    if not np.array_equal(np.sort(target_indices), np.arange(len(target_indices))):
        raise ContractError("target drug_index must be a complete contiguous matrix-row axis")
    source["drug_node_idx"] = pd.to_numeric(
        source["drug_node_idx"], errors="raise").astype(np.int64)
    if source["drug_node_idx"].lt(0).any():
        raise ContractError("XPert drug_node_idx contains a negative feature-axis key")
    grouped = source.groupby("pert_id", sort=False)
    if (grouped["inchi_key"].nunique().gt(1).any()
            or grouped["drug_node_idx"].nunique().gt(1).any()):
        raise ContractError("XPert pert_id maps to multiple structures or encoded nodes")
    node_groups = source.groupby("drug_node_idx", sort=False)["inchi_key"].nunique()
    if node_groups.gt(1).any():
        raise ContractError("one XPert encoded node maps to multiple complete InChIKeys")
    source_unique = source.drop_duplicates("pert_id")
    joined = target.merge(
        source_unique[["pert_id", "inchi_key", "drug_node_idx"]],
        on="pert_id", how="left", suffixes=("_target", "_xpert"),
        validate="one_to_one",
    )
    available = joined["drug_node_idx"].notna()
    exact = available & joined["inchi_key_target"].eq(joined["inchi_key_xpert"])
    n_nodes = unimol_shape[0] if unimol_shape else -1
    if (len(unimol_shape) != 3 or unimol_shape[1:] != (122, 514)
            or len(drug_hg_shape) != 2 or drug_hg_shape[1] != hidden_size
            or unimol_shape[0] != drug_hg_shape[0]):
        raise ContractError("XPert UniMol/HG arrays do not share the complete encoded drug axis")
    encoded_nodes = np.sort(source["drug_node_idx"].unique())
    if not np.array_equal(encoded_nodes, np.arange(n_nodes, dtype=np.int64)):
        raise ContractError("XPert identity table does not cover the complete encoded feature axis")
    node_ids = joined.loc[exact, "drug_node_idx"].astype(np.int64)
    if node_ids.empty or node_ids.min() < 0 or node_ids.max() >= n_nodes:
        raise ContractError("an exact-matched XPert drug has no complete encoded feature row")
    if int(exact.sum()) != expected_shared_count:
        raise ContractError(
            "exact ID + InChIKey XPert drug coverage changed: "
            f"expected={expected_shared_count}, actual={int(exact.sum())}")
    target_index_by_id = dict(zip(target["pert_id"], target["drug_index"].astype(int)))
    node_index_by_id = dict(zip(
        joined.loc[exact, "pert_id"].astype(str),
        joined.loc[exact, "drug_node_idx"].astype(int),
    ))
    mismatches = int((available & ~exact).sum())
    return DrugFeatureIndex(
        target_index_by_pert_id=target_index_by_id,
        node_index_by_pert_id=node_index_by_id,
        exact_shared_pert_ids=frozenset(node_index_by_id),
        n_structure_mismatches=mismatches,
    )


def validate_key_axes(
    conditions: pd.DataFrame,
    exposure: pd.DataFrame,
    target_drugs: pd.DataFrame,
    *,
    delta_shape: tuple[int, ...],
    control_shape: tuple[int, ...],
    morgan_shape: tuple[int, ...],
) -> None:
    required = ("condition_id", "cell_id", "pert_id", "pert_dose",
                "pert_dose_unit", "pert_time", "pert_time_unit")
    _require_columns(conditions, required, "condition keys")
    _require_columns(exposure, ("condition_id", "log10_dose_um", "time_hours"),
                     "condition exposure")
    n_conditions = len(conditions)
    if (conditions["condition_id"].duplicated().any()
            or not np.array_equal(
                pd.to_numeric(conditions["condition_id"], errors="raise").to_numpy(np.int64),
                np.arange(n_conditions))):
        raise ContractError("condition_id must be a unique, ordered, zero-based matrix-row key")
    if len(exposure) != n_conditions or not exposure["condition_id"].reset_index(drop=True).equals(
        conditions["condition_id"].reset_index(drop=True)
    ):
        raise ContractError("condition exposure key order differs from the target matrix key axis")
    if (len(delta_shape) != 2 or delta_shape != (n_conditions, GENE_COUNT)
            or control_shape != delta_shape):
        raise ContractError("target delta/control matrix shape differs from condition/gene axes")
    if morgan_shape != (len(target_drugs), 2048):
        raise ContractError("target Morgan matrix shape differs from its complete drug axis")
    if conditions[list(required[1:])].isna().any().any():
        raise ContractError("condition key contains a missing cell/drug/dose/time value")
    if not conditions["pert_id"].astype(str).isin(
            target_drugs["pert_id"].astype(str)).all():
        raise ContractError("condition matrix contains a drug absent from the target drug axis")
    condition_dose = pd.to_numeric(conditions["pert_dose"], errors="raise").to_numpy(np.float64)
    exposure_dose = pd.to_numeric(exposure["log10_dose_um"], errors="raise").to_numpy(np.float64)
    condition_time = pd.to_numeric(conditions["pert_time"], errors="raise").to_numpy(np.float64)
    exposure_time = pd.to_numeric(exposure["time_hours"], errors="raise").to_numpy(np.float64)
    if (not np.isfinite(condition_dose).all() or (condition_dose <= 0).any()
            or not np.isfinite(exposure_dose).all()
            or not np.allclose(np.log10(condition_dose), exposure_dose, rtol=0, atol=1e-6)):
        raise ContractError("condition dose and exposure matrix axes/units disagree")
    if (not np.isfinite(condition_time).all()
            or not np.array_equal(condition_time, exposure_time)
            or not set(np.unique(condition_time)).issubset({6.0, 24.0})):
        raise ContractError("condition time and exposure matrix axes disagree")
    if not conditions["pert_dose_unit"].astype(str).str.lower().eq("um").all():
        raise ContractError("condition dose unit differs from frozen micromolar units")
    if not conditions["pert_time_unit"].astype(str).str.lower().eq("h").all():
        raise ContractError("condition time unit differs from frozen hour units")


def load_upstream_functions(source_file: Path,
                            function_names: tuple[str, ...]) -> dict[str, Callable[..., Any]]:
    """Load only named upstream pure functions, avoiding training-entrypoint imports."""
    try:
        tree = ast.parse(source_file.read_text(encoding="utf-8"))
    except (OSError, SyntaxError) as exc:
        raise ContractError(f"cannot read official XPert source {source_file}: {exc}") from exc
    selected = [node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name in function_names]
    found = {node.name for node in selected}
    if found != set(function_names):
        raise ContractError(f"official XPert source is missing functions: {set(function_names) - found}")
    code = compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])),
                   str(source_file), "exec")
    namespace: dict[str, Any] = {"np": np, "torch": torch}
    exec(code, namespace)
    return {name: namespace[name] for name in function_names}


def build_official_loss(outputs: tuple[Any, ...], loss_weights: tuple[float, ...],
                        mse_loss: Callable[..., torch.Tensor],
                        pcc_loss: Callable[..., torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if len(loss_weights) != 4 or not np.isfinite(loss_weights).all():
        raise ContractError("four finite official XPert loss weights are required")
    trt_output, ctl_output, delta_output, trt_raw, ctl_raw = outputs[:5]
    delta_true = trt_raw - ctl_raw
    parts = {
        "treatment_mse": mse_loss(trt_output, trt_raw),
        "control_mse": mse_loss(ctl_output, ctl_raw),
        "delta_mse": mse_loss(delta_output, delta_true),
        "delta_pearson": pcc_loss(delta_output, delta_true),
    }
    total = sum(weight * parts[name] for weight, name in zip(
        loss_weights, ("treatment_mse", "control_mse", "delta_mse", "delta_pearson")))
    return total, parts


def load_model_config(paths: ProjectPaths) -> dict[str, Any]:
    config_path = paths.xpert_dir / "configs/config_l1000.yaml"
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ContractError(f"cannot read the official XPert L1000 config: {exc}") from exc
    required = ("model", "dataset", "train")
    if not isinstance(config, dict) or any(key not in config for key in required):
        raise ContractError("official XPert config is missing model/dataset/train sections")
    if int(config["dataset"].get("gene_num", -1)) != GENE_COUNT:
        raise ContractError("official XPert config gene count is not 978")
    if int(config["dataset"].get("max_atom_size", -1)) != 122:
        raise ContractError("official XPert config atom axis is not 122")
    config["_source_root"] = str(paths.xpert_dir)
    config["model"]["ATTN"]["ppi_gene_vector_path"] = str(
        paths.xpert_dir / "processed_data/PPI_gene_vector_128d.npy")
    config["model"]["HG"]["drug_hg_pretrained_embed_path"] = str(
        paths.xpert_dir / "HG_data/saved_embedding/HG_drug_embeddings.npy")
    return config


def _verify_asset_audit(paths: ProjectPaths, *, expected_sha256: str,
                        split_sha256: str, input_manifest: dict[str, Any],
                        gene_map: list[int]) -> dict[str, Any]:
    validate_expected_digest(expected_sha256, canonical=XPERT_AUDIT_SHA256,
                             name="xpert-audit-sha256")
    report_path = paths.audit_dir / "asset_audit.json"
    actual = sha256_file(report_path)
    if actual != expected_sha256.lower():
        raise ContractError(f"XPert asset audit digest mismatch: actual={actual}")
    report = read_json(report_path)
    if (report.get("experiment_id") != EXPERIMENT_ID
            or report.get("status") != "XPERT_ASSET_EXPOSURE_AUDITED"
            or report.get("target_split_sha256") != split_sha256
            or report.get("source_gene_order_equals_target") is not False
            or report.get("n_target_drugs") != EXPECTED_TARGET_DRUGS
            or report.get("n_exact_id_and_structure_matched_drugs")
            != EXPECTED_XPERT_SHARED_DRUGS):
        raise ContractError("XPert audit does not match the frozen EXP-037 identity contract")
    for checkpoint in report.get("checkpoints", {}).values():
        if checkpoint.get("strict_unseen_cell_eligible_all_folds") is not False:
            raise ContractError("legacy XPert checkpoints must remain excluded")
    map_path = paths.audit_dir / "source_to_target_gene_indices.json"
    if (sha256_file(map_path) != XPERT_GENE_MAP_SHA256
            or report.get("source_to_target_gene_indices_sha256") != XPERT_GENE_MAP_SHA256
            or list(map(int, json.loads(map_path.read_text(encoding="utf-8")))
                    ) != list(gene_map)):
        raise ContractError("XPert gene-axis map differs from the frozen asset audit")
    target_drug_sha = input_manifest["files_sha256"]["drugs.parquet"]
    if report.get("target_drugs_sha256") != target_drug_sha:
        raise ContractError("XPert asset audit targets a different project drug axis")
    source_drugs = paths.xpert_dir / "processed_data/l1000_sdst_drugs_info_8432.csv"
    if sha256_file(source_drugs) != report.get("source_drugs_sha256"):
        raise ContractError("XPert source drug identity table differs from its asset audit")
    return report


def load_fold_data(paths: ProjectPaths, split_fold: SplitFold,
                   *, expected_delta_sha256: str,
                   expected_xpert_audit_sha256: str) -> tuple[FoldDataLoader, dict[str, Any]]:
    validate_expected_digest(expected_delta_sha256, canonical=DELTA_V2_SHA256,
                             name="delta-sha256")
    paths.validate_layout()
    quality = read_json(paths.contract_dir / "quality_report.json")
    contract_genes_path = paths.contract_dir / "genes.json"
    gene_ids = json.loads(contract_genes_path.read_text(encoding="utf-8"))
    if not isinstance(gene_ids, list):
        raise ContractError("target genes.json must contain an ordered gene list")
    if (quality.get("gene_axis", {}).get("n") != GENE_COUNT
            or quality.get("gene_axis", {}).get("sha256") != sha256_file(contract_genes_path)
            or quality.get("split_sha256") != SPLIT_V2_SHA256):
        raise ContractError("source quality contract does not match v2 split/978-gene axis")
    _verify_file_hashes(paths.contract_dir, quality.get("files_sha256", {}))

    condition_manifest_path = paths.condition_dir / "manifest.json"
    condition_manifest = read_json(condition_manifest_path)
    if (condition_manifest.get("status") != "PASS"
            or condition_manifest.get("n_genes") != GENE_COUNT
            or condition_manifest.get("source_contract_report_sha256")
            != sha256_file(paths.contract_dir / "quality_report.json")):
        raise ContractError("condition target manifest is not linked to the frozen contract")
    condition_file_hashes = condition_manifest.get("files_sha256", {})
    declared_delta_sha = _valid_sha(
        condition_file_hashes.get("condition_delta.npy"), "condition_delta.npy sha256")
    if declared_delta_sha != expected_delta_sha256.lower():
        raise ContractError("declared target matrix digest differs from explicit expected digest")
    # The response matrix digest is deliberately not recomputed here: doing so
    # would read all outer-test response bytes before model/config freeze.
    _verify_file_hashes(
        paths.condition_dir, condition_file_hashes,
        skip=frozenset({"condition_delta.npy"}),
    )

    input_manifest = read_json(paths.input_dir / "manifest.json")
    if (input_manifest.get("status") != "PASS"
            or input_manifest.get("n_conditions") != condition_manifest.get("n_conditions")
            or input_manifest.get("n_drugs") != EXPECTED_TARGET_DRUGS
            or input_manifest.get("source", {}).get("condition_manifest_sha256")
            != sha256_file(condition_manifest_path)
            or input_manifest.get("source", {}).get("continuous_contract_sha256")
            != sha256_file(paths.contract_dir / "quality_report.json")):
        raise ContractError("condition input manifest is not linked to the frozen target contract")
    # The control matrix is also an XPert auxiliary target. Keep its outer-fold
    # values unread before model freeze; verify its explicit manifest digest and
    # NPY header now, then inspect only selected inner-training rows for smoke.
    input_file_hashes = input_manifest.get("files_sha256", {})
    declared_control_sha = _valid_sha(
        input_file_hashes.get("condition_control.npy"), "condition_control.npy sha256")
    _verify_file_hashes(
        paths.input_dir, input_file_hashes,
        skip=frozenset({"condition_control.npy"}),
    )

    source_map_path = paths.audit_dir / "source_to_target_gene_indices.json"
    source_to_target = json.loads(source_map_path.read_text(encoding="utf-8"))
    gene_map = validate_gene_map(gene_ids, source_to_target)
    audit = _verify_asset_audit(
        paths, expected_sha256=expected_xpert_audit_sha256,
        split_sha256=SPLIT_V2_SHA256, input_manifest=input_manifest,
        gene_map=gene_map.tolist(),
    )

    conditions = pd.read_parquet(paths.condition_dir / "conditions.parquet")
    exposure = pd.read_parquet(paths.input_dir / "condition_exposure.parquet")
    target_drugs = pd.read_parquet(paths.input_dir / "drugs.parquet")
    delta_shape = read_npy_shape(paths.condition_dir / "condition_delta.npy")
    control_path = paths.input_dir / "condition_control.npy"
    control_shape = read_npy_shape(control_path)
    morgan_shape = read_npy_shape(paths.input_dir / "drug_morgan2048.npy")
    validate_key_axes(
        conditions, exposure, target_drugs,
        delta_shape=delta_shape, control_shape=control_shape,
        morgan_shape=morgan_shape,
    )
    if (len(conditions) != condition_manifest.get("n_conditions")
            or len(target_drugs) != EXPECTED_TARGET_DRUGS):
        raise ContractError("parquet row counts differ from manifests")

    config = load_model_config(paths)
    source_drugs_path = paths.xpert_dir / "processed_data/l1000_sdst_drugs_info_8432.csv"
    source_drugs = pd.read_csv(source_drugs_path, dtype={"pert_id": str, "inchi_key": str})
    unimol_path = paths.xpert_dir / "processed_data/all_drugs_unimol_arr.npy"
    drug_hg_path = paths.xpert_dir / "HG_data/saved_embedding/HG_drug_embeddings.npy"
    ppi_path = paths.xpert_dir / "processed_data/PPI_gene_vector_128d.npy"
    unimol_shape = read_npy_shape(unimol_path)
    drug_hg_shape = read_npy_shape(drug_hg_path)
    ppi_shape = read_npy_shape(ppi_path)
    hidden_size = int(config["model"]["ATTN"]["hidden_size"])
    if len(ppi_shape) != 2 or ppi_shape[0] != GENE_COUNT:
        raise ContractError("XPert PPI feature axis does not contain 978 ordered genes")
    drug_index = validate_drug_feature_index(
        target_drugs, source_drugs, unimol_shape, drug_hg_shape,
        hidden_size=hidden_size,
        expected_shared_count=EXPECTED_XPERT_SHARED_DRUGS,
    )

    delta = np.load(paths.condition_dir / "condition_delta.npy",
                    mmap_mode="r", allow_pickle=False)
    control = np.load(control_path, mmap_mode="r", allow_pickle=False)
    unimol = np.load(unimol_path, mmap_mode="r", allow_pickle=False)
    drug_hg = np.load(drug_hg_path, mmap_mode="r", allow_pickle=False)
    loader = FoldDataLoader(
        keys=conditions,
        conditions=conditions,
        exposure=exposure,
        delta=delta,
        control=control,
        unimol=unimol,
        drug_hg=drug_hg,
        source_to_target_gene_indices=gene_map,
        drug_index=drug_index,
        split_fold=split_fold,
        config=config,
    )
    evidence = {
        "split_sha256": SPLIT_V2_SHA256,
        "delta_sha256_declared_not_recomputed": declared_delta_sha,
        "condition_control_sha256_declared_not_recomputed": declared_control_sha,
        "n_conditions": int(len(conditions)),
        "n_genes": GENE_COUNT,
        "n_target_drugs": int(len(target_drugs)),
        "n_exact_xpert_drugs": len(drug_index.exact_shared_pert_ids),
        "n_id_matched_structure_mismatches": drug_index.n_structure_mismatches,
        "source_to_target_gene_map_sha256": XPERT_GENE_MAP_SHA256,
        "gene_axis_reordered": True,
        "legacy_checkpoints_loaded": False,
        "outer_test_response_values_read": False,
        "xpert_source_code_sha256": {
            relative.as_posix(): sha256_file(paths.xpert_dir / relative)
            for relative in (
                Path("models/model_XPert.py"), Path("models/model_utils.py"),
                Path("utils.py"), Path("datasets/MyDataset.py"),
            )
        },
        "asset_audit_status": audit["status"],
    }
    return loader, evidence


def load_xpert_model(paths: ProjectPaths, config: dict[str, Any],
                     *, device: torch.device, seed: int) -> tuple[torch.nn.Module, Callable, Callable]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ContractError("gradient smoke requires a CUDA device")
    source_text = str(paths.xpert_dir)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    try:
        from models.model_XPert import XPertNet
    except Exception as exc:
        raise ContractError(f"cannot import the official XPert model: {exc}") from exc
    finally:
        if source_text in sys.path:
            sys.path.remove(source_text)
    args = type("XPertStrictArgs", (), {})()
    args.device = device
    args.dataset = "l1000_mdmt"
    args.drug_feat = "unimol"
    args.pretrained_mode = "global"
    args.include_cell_idx = False
    args.mode = "train"
    args.output_attention = False
    args.use_gene_pos_emed = False
    args.wo_ppi = False
    args.wo_HG = False
    args.wo_atom = False
    args.wo_atom_HG = False
    args.wo_unimol = False
    logger = _SilentLogger()
    model = XPertNet(args, config, device, logger).to(device)
    model.init_weights()

    loss_functions = load_upstream_functions(
        paths.xpert_dir / "utils.py", ("mse_loss_ls_sum", "pcc_loss_sum"))
    return model, loss_functions["mse_loss_ls_sum"], loss_functions["pcc_loss_sum"]


class _SilentLogger:
    def info(self, message: str) -> None:
        del message


def run_gradient_smoke(loader: FoldDataLoader, rows: FoldRows, *,
                       paths: ProjectPaths, device: torch.device, batch_size: int,
                       seed: int, loss_weights: tuple[float, ...]) -> dict[str, Any]:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ContractError("gradient smoke requires a visible CUDA device")
    if batch_size < 1:
        raise ContractError("an explicit positive smoke batch size is required")
    positions = rows.positions[:batch_size]
    if len(positions) != batch_size:
        raise ContractError("selected inner-training partition is smaller than the smoke batch")
    batch = loader.batch(rows, positions, device=device)
    model_config = loader.config
    model, mse_loss, pcc_loss = load_xpert_model(
        paths, model_config, device=device, seed=seed)
    model.train()
    model.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    outputs = model(batch.model_inputs)
    loss, parts = build_official_loss(outputs, loss_weights, mse_loss, pcc_loss)
    if not torch.isfinite(loss):
        raise ContractError("official XPert loss is non-finite on the smoke batch")
    loss.backward()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    gradients = [parameter.grad for parameter in model.parameters()
                 if parameter.grad is not None]
    if not gradients or not all(torch.isfinite(grad).all().item() for grad in gradients):
        raise ContractError("XPert smoke produced missing or non-finite gradients")
    gradient_l1 = float(sum(grad.detach().abs().sum().item() for grad in gradients))
    if gradient_l1 <= 0:
        raise ContractError("XPert smoke gradients are all zero")
    receipt = {
        "status": "ONE_BATCH_FORWARD_BACKWARD_PASS",
        "optimizer_steps": 0,
        "training_epochs": 0,
        "model_selection": False,
        "outer_test_response_values_read": False,
        "fold": loader.split_fold.fold,
        "role": rows.role,
        "n_batch": batch_size,
        "seed": seed,
        "loss_weights": list(loss_weights),
        "loss": float(loss.detach().item()),
        "loss_components": {name: float(value.detach().item())
                            for name, value in parts.items()},
        "gradient_l1": gradient_l1,
        "elapsed_seconds": elapsed,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)),
        "cuda_device": torch.cuda.get_device_name(device),
    }
    model.zero_grad(set_to_none=True)
    return receipt


def require_supported_runtime() -> None:
    if platform.system() != "Linux":
        raise ContractError("Python/model work must run in WSL2 Linux")
    if os.environ.get("CONDA_DEFAULT_ENV") != "drugscreening-gpu":
        raise ContractError("Python/model work must use Conda environment drugscreening-gpu")
    try:
        kernel = Path("/proc/version").read_text(encoding="utf-8").lower()
    except OSError as exc:
        raise ContractError("cannot verify WSL2 runtime") from exc
    if "microsoft" not in kernel:
        raise ContractError("Linux runtime is not WSL2")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(EXPECTED_FOLDS), required=True)
    parser.add_argument("--mode", choices=("preflight", "gradient-smoke"), required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--expected-delta-sha256", required=True)
    parser.add_argument("--expected-xpert-audit-sha256", required=True)
    parser.add_argument("--shared-drug-ids-json", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--loss-weights", type=float, nargs=4)
    args = parser.parse_args(argv)
    require_supported_runtime()
    paths = ProjectPaths.from_root(args.project_root)
    paths.validate_layout()
    if args.expected_xpert_audit_sha256.lower() != XPERT_AUDIT_SHA256:
        raise ContractError("expected XPert asset-audit digest differs from the registered audit")
    split_fold, split_sha = validate_split_file(
        paths.split, expected_sha256=args.expected_split_sha256,
        fold_index=args.fold,
    )
    if args.expected_delta_sha256.lower() != DELTA_V2_SHA256:
        raise ContractError("expected condition-delta digest differs from the v2 manifest")
    loader, evidence = load_fold_data(
        paths, split_fold, expected_delta_sha256=args.expected_delta_sha256,
        expected_xpert_audit_sha256=args.expected_xpert_audit_sha256)
    if (sha256_file(paths.audit_dir / "asset_audit.json")
            != args.expected_xpert_audit_sha256.lower()):
        raise ContractError("XPert asset-audit file bytes differ from the explicit digest")
    if args.shared_drug_ids_json is None:
        shared_drug_ids = loader.drug_index.exact_shared_pert_ids
        evidence["shared_drug_scope"] = "smoke/preflight support set from exact ID + InChIKey"
    else:
        shared_drug_ids = json.loads(args.shared_drug_ids_json.read_text(encoding="utf-8"))
        if not isinstance(shared_drug_ids, list) or not shared_drug_ids:
            raise ContractError("shared-drug ID file must contain an explicit non-empty list")
        evidence["shared_drug_scope"] = "explicit ID list; does not authorize formal comparison"
    rows = loader.fold_rows(shared_drug_ids=frozenset(map(str, shared_drug_ids)))
    evidence["fold"] = split_fold.fold
    evidence["split_path"] = str(paths.split)
    evidence["split_expected_sha256"] = args.expected_split_sha256.lower()
    evidence["outer_train_rows_shared_drugs"] = int(len(rows["outer_train"].positions))
    evidence["outer_test_rows_shared_seen_drugs"] = int(len(rows["outer_test"].positions))
    evidence["inner_train_rows_shared_drugs"] = [
        int(len(item["train"].positions)) for item in rows["inner_folds"]]
    evidence["inner_validation_rows_shared_seen_drugs"] = [
        int(len(item["validation"].positions)) for item in rows["inner_folds"]]
    if args.mode == "preflight":
        print(json.dumps({"status": "FOLD_INPUT_CONTRACT_PASS",
                          "evidence": evidence}, ensure_ascii=False))
        return
    if args.device is None or args.batch_size is None or args.seed is None:
        raise ContractError("gradient smoke requires explicit --device, --batch-size, and --seed")
    if args.loss_weights is None:
        raise ContractError("gradient smoke requires four explicit --loss-weights values")
    device = torch.device(args.device)
    # One training-side batch only; no validation/test response values are read.
    smoke_rows = rows["inner_folds"][0]["train"]
    smoke_receipt = run_gradient_smoke(
        loader, smoke_rows, paths=paths, device=device,
        batch_size=args.batch_size, seed=args.seed,
        loss_weights=tuple(args.loss_weights),
    )
    print(json.dumps({"status": "FOLD_INPUT_CONTRACT_PASS",
                      "evidence": evidence,
                      "gradient_smoke": smoke_receipt}, ensure_ascii=False))


if __name__ == "__main__":
    main()
