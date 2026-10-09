"""Fold-local linear models for EXP-066 transcription-response prediction.

The candidate predicts LINCS treatment-minus-matched-vehicle deltas from
DepMap baseline RNA and drug structure.  This module never loads data and has
no filesystem side effects; the runner supplies frozen axes and fold keys.
Every scaler, PCA projection, and response basis is fitted from the fold's
training rows only.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.linalg import eigh, solve
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


RESPONSE_RANK = 16
RIDGE_ALPHA = 1.0


def _as_index(values: np.ndarray, name: str, upper: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must be a one-dimensional integer index")
    array = array.astype(np.int64, copy=False)
    if array.size and (int(array.min()) < 0 or int(array.max()) >= upper):
        raise ValueError(f"{name} contains an out-of-range row index")
    return array


def _finite_2d(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must be a nonempty two-dimensional matrix")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array.astype(np.float64, copy=False)


def _check_train_rows(rows: np.ndarray, n_conditions: int, name: str) -> np.ndarray:
    rows = _as_index(rows, name, n_conditions)
    if rows.size == 0 or np.unique(rows).size != rows.size:
        raise ValueError(f"{name} must be nonempty and unique")
    return rows


@dataclass(frozen=True)
class FoldFeatureSpec:
    """Frozen feature dimensions for a single fold.

    The component counts are protocol values, not search ranges.  The
    response rank and Ridge penalty are fixed by the approved EXP-066 design.
    """

    rna_components: int = 16
    drug_components: int = 16
    no_state_seed: int = 660066
    fixed_nonlinear_components: int = 256
    fixed_rff_components: int = 978
    response_rank: int = RESPONSE_RANK
    ridge_alpha: float = RIDGE_ALPHA

    def validate(self, n_rna_features: int, n_drug_features: int,
                 n_train_cells: int, n_train_drugs: int) -> None:
        if n_rna_features != 978 or n_drug_features != 2048:
            raise ValueError("EXP-066 axes are frozen to 978 RNA genes and 2048 Morgan bits")
        if self.rna_components != 16 or self.drug_components != 16:
            raise ValueError("EXP-066 state and drug PCA dimensions are frozen at 16")
        if self.rna_components > min(n_rna_features, n_train_cells):
            raise ValueError("RNA compression rank exceeds the fold-training axis")
        if self.drug_components > min(n_drug_features, n_train_drugs):
            raise ValueError("drug compression rank exceeds the fold-training axis")
        if self.response_rank != RESPONSE_RANK:
            raise ValueError(f"EXP-066 response rank is frozen at {RESPONSE_RANK}")
        if not np.isfinite(self.ridge_alpha) or self.ridge_alpha != RIDGE_ALPHA:
            raise ValueError(f"EXP-066 Ridge alpha is frozen at {RIDGE_ALPHA}")
        if self.fixed_nonlinear_components != 256 or self.fixed_rff_components != 978:
            raise ValueError("EXP-066 no-state feature counts are frozen at 256 and 978")
        if self.no_state_seed != 660066:
            raise ValueError("EXP-066 no-state feature seed is frozen at 660066")

    @property
    def n_candidate_features(self) -> int:
        return 978 + 2048 + 2 + self.rna_components * self.drug_components


@dataclass
class FoldFeatureEncoder:
    """Fold-local RNA/drug compression and fixed multiplicative interaction."""

    spec: FoldFeatureSpec
    rna_scaler: StandardScaler | None = None
    rna_pca: PCA | None = None
    drug_scaler: StandardScaler | None = None
    drug_pca: PCA | None = None
    exposure_scaler: StandardScaler | None = None
    nonlinear_weights: np.ndarray | None = None
    nonlinear_bias: np.ndarray | None = None
    rff_weights: np.ndarray | None = None
    rff_bias: np.ndarray | None = None

    def fit(self, rna_by_cell: np.ndarray, drug_by_key: np.ndarray,
            condition_cell_index: np.ndarray, condition_drug_index: np.ndarray,
            exposure_by_condition: np.ndarray,
            training_rows: np.ndarray) -> "FoldFeatureEncoder":
        rna = _finite_2d(rna_by_cell, "DepMap baseline RNA")
        drug = _finite_2d(drug_by_key, "drug structure features")
        exposure = _finite_2d(exposure_by_condition, "log-dose/time exposure features")
        cell_index = _as_index(condition_cell_index, "condition_cell_index", len(rna))
        drug_index = _as_index(condition_drug_index, "condition_drug_index", len(drug))
        if rna.shape[1] != 978 or drug.shape[1] != 2048 or exposure.shape[1] != 2:
            raise ValueError("EXP-066 expects [978 RNA, 2048 Morgan, 2 log-exposure] feature axes")
        if cell_index.size != drug_index.size or cell_index.size != len(exposure):
            raise ValueError("condition cell, drug, and exposure rows differ in length")
        train = _check_train_rows(training_rows, len(cell_index), "training_rows")
        train_cells = np.unique(cell_index[train])
        train_drugs = np.unique(drug_index[train])
        self.spec.validate(rna.shape[1], drug.shape[1], len(train_cells), len(train_drugs))
        self.rna_scaler = StandardScaler().fit(rna[train_cells])
        self.rna_pca = PCA(n_components=self.spec.rna_components, svd_solver="full")
        self.rna_pca.fit(self.rna_scaler.transform(rna[train_cells]))
        self.drug_scaler = StandardScaler().fit(drug[train_drugs])
        self.exposure_scaler = StandardScaler().fit(exposure[train])
        self.drug_pca = PCA(n_components=self.spec.drug_components, svd_solver="full")
        self.drug_pca.fit(self.drug_scaler.transform(drug[train_drugs]))
        rng = np.random.default_rng(self.spec.no_state_seed)
        self.nonlinear_weights = rng.normal(
            loc=0.0, scale=1.0 / np.sqrt(drug.shape[1]),
            size=(drug.shape[1], self.spec.fixed_nonlinear_components),
        )
        self.nonlinear_bias = rng.uniform(-np.pi, np.pi,
                                          size=self.spec.fixed_nonlinear_components)
        self.rff_weights = rng.normal(
            loc=0.0, scale=1.0 / np.sqrt(drug.shape[1]),
            size=(drug.shape[1], self.spec.fixed_rff_components),
        )
        self.rff_bias = rng.uniform(-np.pi, np.pi,
                                        size=self.spec.fixed_rff_components)
        return self

    def _candidate_raw(self, rna: np.ndarray, drug: np.ndarray,
                       cell_index: np.ndarray, drug_index: np.ndarray,
                       exposure: np.ndarray) -> np.ndarray:
        if (self.rna_scaler is None or self.rna_pca is None or self.drug_scaler is None
                or self.drug_pca is None or self.exposure_scaler is None):
            raise RuntimeError("feature encoder has not been fitted")
        state = self.rna_pca.transform(self.rna_scaler.transform(rna[cell_index]))
        chemical = self.drug_pca.transform(self.drug_scaler.transform(drug[drug_index]))
        interaction = np.einsum("ni,nj->nij", state, chemical, optimize=True).reshape(len(state), -1)
        state_direct = self.rna_scaler.transform(rna[cell_index])
        exposure_scaled = self.exposure_scaler.transform(exposure)
        features = np.concatenate((state_direct, drug[drug_index], exposure_scaled, interaction), axis=1)
        if features.shape[1] != self.spec.n_candidate_features:
            raise ValueError("EXP-066 candidate design does not have 3284 features")
        return features

    def _no_state_raw(self, drug: np.ndarray, drug_index: np.ndarray,
                      exposure: np.ndarray) -> np.ndarray:
        if (self.drug_scaler is None or self.drug_pca is None or self.exposure_scaler is None
                or self.nonlinear_weights is None or self.nonlinear_bias is None
                or self.rff_weights is None or self.rff_bias is None):
            raise RuntimeError("feature encoder has not been fitted")
        structure = self.drug_scaler.transform(drug[drug_index])
        nonlinear_phase = np.sqrt(2.0 / self.spec.fixed_nonlinear_components) * (
            structure @ self.nonlinear_weights + self.nonlinear_bias
        )
        rff_phase = np.sqrt(2.0 / self.spec.fixed_rff_components) * (
            structure @ self.rff_weights + self.rff_bias
        )
        nonlinear = np.cos(nonlinear_phase)
        rff = np.cos(rff_phase)
        exposure_scaled = self.exposure_scaler.transform(exposure)
        features = np.concatenate((drug[drug_index], exposure_scaled, nonlinear, rff), axis=1)
        if features.shape[1] != self.spec.n_candidate_features:
            raise ValueError("EXP-066 no-state design does not have 3284 features")
        return features

    def transform_candidate(self, rna_by_cell: np.ndarray, drug_by_key: np.ndarray,
                            condition_cell_index: np.ndarray,
                            condition_drug_index: np.ndarray,
                            exposure_by_condition: np.ndarray) -> np.ndarray:
        rna = _finite_2d(rna_by_cell, "DepMap baseline RNA")
        drug = _finite_2d(drug_by_key, "drug structure features")
        exposure = _finite_2d(exposure_by_condition, "log-dose/time exposure features")
        cell_index = _as_index(condition_cell_index, "condition_cell_index", len(rna))
        drug_index = _as_index(condition_drug_index, "condition_drug_index", len(drug))
        if len(exposure) != len(cell_index) or len(drug_index) != len(cell_index):
            raise ValueError("condition features are not row-aligned")
        return self._candidate_raw(rna, drug, cell_index, drug_index, exposure)

    def transform_no_state(self, drug_by_key: np.ndarray,
                           condition_drug_index: np.ndarray,
                           exposure_by_condition: np.ndarray) -> np.ndarray:
        drug = _finite_2d(drug_by_key, "drug structure features")
        exposure = _finite_2d(exposure_by_condition, "log-dose/time exposure features")
        drug_index = _as_index(condition_drug_index, "condition_drug_index", len(drug))
        if len(exposure) != len(drug_index):
            raise ValueError("condition exposure and drug rows differ in length")
        return self._no_state_raw(drug, drug_index, exposure)


@dataclass
class ReducedRankRidge:
    """Ridge regression with the exact rank-constrained response subspace.

    It first obtains the alpha-penalized full multi-output Ridge solution, then
    projects coefficients onto the leading response directions under the
    training-design Ridge metric.  Both stages use training rows only.
    """

    rank: int = RESPONSE_RANK
    alpha: float = RIDGE_ALPHA
    x_scaler: StandardScaler | None = None
    y_mean: np.ndarray | None = None
    y_scale: np.ndarray | None = None
    coefficient: np.ndarray | None = None
    target_basis: np.ndarray | None = None
    response_eigenvalues: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "ReducedRankRidge":
        x = _finite_2d(x, "model input")
        y = _finite_2d(y, "training response")
        if len(x) != len(y):
            raise ValueError("model input and response row counts differ")
        if self.rank != RESPONSE_RANK or self.alpha != RIDGE_ALPHA:
            raise ValueError("EXP-066 rank and Ridge alpha are frozen at 16 and 1.0")
        if self.rank > min(x.shape[1], y.shape[1]):
            raise ValueError("response rank exceeds model input or target dimension")

        self.x_scaler = StandardScaler().fit(x)
        x_scaled = self.x_scaler.transform(x)
        self.y_mean = y.mean(axis=0)
        self.y_scale = y.std(axis=0)
        self.y_scale = np.where(self.y_scale > np.finfo(np.float64).eps, self.y_scale, 1.0)
        y_scaled = (y - self.y_mean) / self.y_scale

        # Use the smaller of the primal and dual systems.  The frozen design
        # has 3,284 columns, while synthetic tests and some folds can have
        # fewer training conditions; solving in condition space avoids an
        # unnecessary 3,284-square factorization in that case.
        if x_scaled.shape[0] < x_scaled.shape[1]:
            gram = x_scaled @ x_scaled.T
            gram.flat[::gram.shape[0] + 1] += self.alpha
            dual = solve(gram, y_scaled, assume_a="pos", check_finite=False)
            ridge_coefficient = x_scaled.T @ dual
            fitted = x_scaled @ ridge_coefficient
        else:
            gram = x_scaled.T @ x_scaled
            gram.flat[::gram.shape[0] + 1] += self.alpha
            cross = x_scaled.T @ y_scaled
            ridge_coefficient = solve(gram, cross, assume_a="pos", check_finite=False)
            fitted = x_scaled @ ridge_coefficient
        response_gram = y_scaled.T @ fitted
        response_gram = (response_gram + response_gram.T) * 0.5
        eigenvalues, eigenvectors = eigh(
            response_gram,
            subset_by_index=(y.shape[1] - self.rank, y.shape[1] - 1),
            check_finite=False,
            driver="evr",
        )
        basis = eigenvectors[:, ::-1]
        values = eigenvalues[::-1]
        self.target_basis = basis
        self.response_eigenvalues = values
        self.coefficient = ridge_coefficient @ basis @ basis.T
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.x_scaler is None or self.y_mean is None or self.y_scale is None or self.coefficient is None:
            raise RuntimeError("model has not been fitted")
        x = _finite_2d(x, "prediction input")
        if x.shape[1] != self.coefficient.shape[0]:
            raise ValueError("prediction feature axis differs from fitted model")
        scaled = self.x_scaler.transform(x) @ self.coefficient
        return scaled * self.y_scale + self.y_mean


def zero_change(n_rows: int, n_targets: int) -> np.ndarray:
    if n_rows < 0 or n_targets <= 0:
        raise ValueError("invalid zero-change prediction shape")
    return np.zeros((n_rows, n_targets), dtype=np.float64)


def per_drug_training_mean(y_train: np.ndarray, train_drug_index: np.ndarray,
                           predict_drug_index: np.ndarray,
                           allow_missing: bool = False) -> np.ndarray:
    """Predict each drug's training-only mean.

    If ``allow_missing`` is true, a drug absent from the training partition has
    an explicitly undefined (NaN) prediction.  This preserves the comparison
    denominator instead of silently substituting another drug's mean.
    """
    y = _finite_2d(y_train, "training response")
    train_drug_index = _as_index(train_drug_index, "train_drug_index", upper=int(np.max(train_drug_index)) + 1)
    predict_drug_index = np.asarray(predict_drug_index)
    if predict_drug_index.ndim != 1 or not np.issubdtype(predict_drug_index.dtype, np.integer):
        raise ValueError("predict_drug_index must be a one-dimensional integer index")
    if len(y) != len(train_drug_index):
        raise ValueError("training response and drug index row counts differ")
    means = {int(key): y[train_drug_index == key].mean(axis=0)
             for key in np.unique(train_drug_index)}
    missing = sorted(set(map(int, predict_drug_index)) - set(means))
    if missing and not allow_missing:
        raise ValueError(f"per-drug training mean has no training rows for {len(missing)} prediction keys")
    result = np.full((len(predict_drug_index), y.shape[1]), np.nan, dtype=np.float64)
    for row, key in enumerate(predict_drug_index):
        if int(key) in means:
            result[row] = means[int(key)]
    return result


def validate_object_shuffle(condition_object_id: np.ndarray,
                            shuffled_context_object_id: np.ndarray,
                            partition_role: np.ndarray) -> None:
    """Check an externally frozen shuffle keeps each object block intact.

    ``partition_role`` marks every condition as train/validation/test.  The
    data contract supplies the shuffled donor-object identity after applying
    the frozen seed, so this function can verify it without inventing a
    mapping between multiple DepMap models belonging to one PatientID.
    """
    source = np.asarray(condition_object_id).astype(str)
    donor = np.asarray(shuffled_context_object_id).astype(str)
    role = np.asarray(partition_role).astype(str)
    if source.ndim != 1 or donor.shape != source.shape or role.shape != source.shape:
        raise ValueError("object shuffle arrays must be one-dimensional and aligned")
    if pd_is_missing(source) or pd_is_missing(donor) or pd_is_missing(role):
        raise ValueError("object shuffle identity contains missing values")
    mapping: dict[tuple[str, str], str] = {}
    reverse: dict[tuple[str, str], str] = {}
    for src, dst, part in zip(source.tolist(), donor.tolist(), role.tolist()):
        if src == dst:
            raise ValueError("object shuffle left a source object in place")
        key = (part, src)
        if key in mapping and mapping[key] != dst:
            raise ValueError("object shuffle splits one source object across donor objects")
        mapping[key] = dst
        rkey = (part, dst)
        if rkey in reverse and reverse[rkey] != src:
            raise ValueError("object shuffle is not one-to-one within a partition")
        reverse[rkey] = src


def pd_is_missing(values: np.ndarray) -> bool:
    """Avoid a pandas dependency in the numeric model module."""
    return any(value in {"", "<NA>", "nan", "None"} for value in values.tolist())


def build_fold_local_shuffled_cell_ids(condition_cell_id: np.ndarray,
                                       condition_object_id: np.ndarray,
                                       fold_id: np.ndarray,
                                       n_folds: int = 5,
                                       seed: int = 660067) -> np.ndarray:
    """Build deterministic train/held-out object-block derangements.

    Output is ``[n_folds, n_conditions]``; row ``f`` is the shuffled LINCS
    cell ID to use when fold ``f`` is held out.  For each fold, TRAIN objects
    and that fold's held-out objects are deranged separately.  Objects are
    ordered by SHA-256 of the frozen seed, fold, partition, and PatientID,
    then rotated by one.  A
    target object's sorted LINCS cell IDs map by rank modulo the donor's
    sorted cell IDs, as required by the frozen contract.
    """
    cell_ids = np.asarray(condition_cell_id).astype(str)
    object_ids = np.asarray(condition_object_id).astype(str)
    folds = np.asarray(fold_id)
    if (cell_ids.ndim != 1 or object_ids.shape != cell_ids.shape or folds.shape != cell_ids.shape
            or not np.issubdtype(folds.dtype, np.integer)):
        raise ValueError("cell, object, and fold vectors must be aligned one-dimensional arrays")
    if pd_is_missing(cell_ids) or pd_is_missing(object_ids):
        raise ValueError("cell or source-object identity contains missing values")
    if n_folds != 5 or set(map(int, np.unique(folds))) != set(range(n_folds)):
        raise ValueError("EXP-066 requires the frozen five-fold object assignment 0..4")
    object_fold_counts: dict[str, set[int]] = {}
    for object_id, fold in zip(object_ids.tolist(), folds.tolist()):
        object_fold_counts.setdefault(object_id, set()).add(int(fold))
    if any(len(values) != 1 for values in object_fold_counts.values()):
        raise ValueError("one PatientID appears in multiple outer folds")

    result = np.empty((n_folds, len(cell_ids)), dtype=object)
    for heldout_fold in range(n_folds):
        train_rows = np.flatnonzero(folds != heldout_fold)
        heldout_rows = np.flatnonzero(folds == heldout_fold)
        donor_object_for_row = np.empty(len(cell_ids), dtype=object)
        for partition, partition_rows in (("train", train_rows), ("heldout", heldout_rows)):
            objects = sorted(
                set(object_ids[partition_rows].tolist()),
                key=lambda value: (
                    hashlib.sha256(
                        f"{seed}|shuffle|{heldout_fold}|{partition}|{value}".encode("utf-8")
                    ).hexdigest(),
                    value,
                ),
            )
            if len(objects) < 2:
                raise ValueError("each fold partition needs at least two source objects to derange")
            donor_for = {object_id: objects[(index + 1) % len(objects)]
                         for index, object_id in enumerate(objects)}
            cell_lists = {
                object_id: sorted(set(cell_ids[partition_rows[object_ids[partition_rows] == object_id]].tolist()))
                for object_id in objects
            }
            cell_donor_for: dict[tuple[str, str], str] = {}
            for object_id in objects:
                donor = donor_for[object_id]
                targets = cell_lists[object_id]
                donors = cell_lists[donor]
                for rank, cell_id in enumerate(targets):
                    cell_donor_for[(object_id, cell_id)] = donors[rank % len(donors)]
            for row in partition_rows:
                donor_object_for_row[row] = donor_for[object_ids[row]]
                result[heldout_fold, row] = cell_donor_for[(object_ids[row], cell_ids[row])]
        validate_object_shuffle(
            object_ids,
            donor_object_for_row.astype(str),
            np.where(folds == heldout_fold, "heldout", "train"),
        )
    return result.astype(str)


def model_receipt(model: ReducedRankRidge, encoder: FoldFeatureEncoder) -> dict[str, Any]:
    """Return compact implementation facts suitable for a per-fold receipt."""
    if model.coefficient is None or model.response_eigenvalues is None:
        raise RuntimeError("cannot write a receipt for an unfitted model")
    return {
        "response_rank": int(model.rank),
        "ridge_alpha": float(model.alpha),
        "candidate_feature_count": int(encoder.spec.n_candidate_features),
        "no_state_feature_count": int(encoder.spec.n_candidate_features),
        "candidate_trainable_coefficient_count": int(model.coefficient.size),
        # The construction projects onto exactly ``rank`` response directions.
        # Avoid a second SVD solely for a receipt statistic.
        "coefficient_matrix_rank_upper_bound": int(model.rank),
        "leading_response_eigenvalues": model.response_eigenvalues.astype(float).tolist(),
        "rna_compression_components": int(encoder.spec.rna_components),
        "drug_compression_components": int(encoder.spec.drug_components),
        "interaction_component_count": int(encoder.spec.rna_components * encoder.spec.drug_components),
        "fixed_nonlinear_components": int(encoder.spec.fixed_nonlinear_components),
        "fixed_rff_components": int(encoder.spec.fixed_rff_components),
        "no_state_random_seed": int(encoder.spec.no_state_seed),
    }


def generate_nested_oof_response_features(
    domain: Any,
    train_patient_ids: list[str] | tuple[str, ...],
    predict_patient_ids: list[str] | tuple[str, ...],
    outer_fold_id: int,
    config: Mapping[str, Any],
    prediction_metadata: Any,
) -> dict[str, Any]:
    """Public model-package entry point for EXP-067 nested response features.

    The fold/data-contract orchestration stays in the EXP-066 runner; this
    export lets downstream modeling code depend on the stable model API without
    copying the response-training implementation.
    """
    import importlib.util
    import sys
    from pathlib import Path

    module_name = "_drug_screen_exp066_response_runner"
    runner = sys.modules.get(module_name)
    if runner is None:
        repository_root = Path(__file__).resolve().parents[3]
        runner_path = repository_root / "scripts/modeling/run_exp066_response_prediction.py"
        spec = importlib.util.spec_from_file_location(module_name, runner_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"could not load EXP-066 runner at {runner_path}")
        runner = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = runner
        spec.loader.exec_module(runner)
    return runner.generate_nested_oof_response_features(
        domain=domain,
        train_patient_ids=train_patient_ids,
        predict_patient_ids=predict_patient_ids,
        outer_fold_id=outer_fold_id,
        config=config,
        prediction_metadata=prediction_metadata,
    )
