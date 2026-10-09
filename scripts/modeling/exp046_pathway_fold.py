"""Fold-level EXP-046 training adapter with an unlabeled outer-prediction API.

This module deliberately does not load study files, split data, inspect outer
labels, select candidate grids, or write results. A fold driver supplies
pre-sliced tensors and a fold-scoped source-weight receipt. Training entry
points require CUDA explicitly; they never retry on CPU.
"""

from __future__ import annotations

import hashlib
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT, ROOT / "src"):
    if _path.as_posix() not in sys.path:
        sys.path.insert(0, _path.as_posix())

from drug_screen.models.exp046_five_arm import (  # noqa: E402
    AlignedPathwayMap,
    Exp046Arm,
    build_exp046_arm_model,
    capture_cold_start_state,
    restore_cold_start_state,
)
from drug_screen.models.pharmaformer_target import configure_target_variant  # noqa: E402
from scripts.modeling.exp036_pharmaformer_fold import (  # noqa: E402
    pairwise_ranking_loss,
    predict as exp036_predict,
)


@dataclass(frozen=True)
class PairInputBatch:
    """Pairwise model inputs and keys; this type has no response/label field."""

    expression: torch.Tensor
    drug_features: torch.Tensor
    patient_ids: tuple[str, ...]
    drug_ids: tuple[str, ...]

    def validate(self) -> None:
        _validate_pair_inputs(self)


@dataclass(frozen=True)
class LabeledPairBatch(PairInputBatch):
    """Explicitly sliced training or validation pairs, with their own labels."""

    responses: torch.Tensor

    def validate(self) -> None:
        _validate_pair_inputs(self)
        if not isinstance(self.responses, torch.Tensor):
            raise TypeError("responses must be a torch tensor")
        if self.responses.ndim != 1 or len(self.responses) != len(self.patient_ids):
            raise ValueError("responses must be a vector aligned to pair keys")
        if not self.responses.is_floating_point():
            raise ValueError("responses must use a floating point dtype")
        if not bool(torch.isfinite(self.responses).all()):
            raise ValueError("responses must contain only finite values")


@dataclass(frozen=True)
class SourceFoldInitialization:
    """Caller-supplied backbone weights and their fold/exposure audit receipt.

    ``target_fit_patient_ids`` records target identities whose labels entered
    the source-stage weights. The fold API checks this declaration against the
    exact training slice; the upstream source-stage builder must produce the
    declaration and checksum from its own audited inputs.
    """

    outer_fold_key: str
    state_dict: Mapping[str, torch.Tensor]
    source_sha256: str
    state_dict_sha256: str
    target_fit_patient_ids: tuple[str, ...]

    def validate(self) -> None:
        if not isinstance(self.outer_fold_key, str) or not self.outer_fold_key.strip():
            raise ValueError("outer_fold_key must be a non-empty string")
        if (not isinstance(self.source_sha256, str)
                or len(self.source_sha256) != 64
                or any(character not in "0123456789abcdef" for character in self.source_sha256)):
            raise ValueError("source_sha256 must be a lowercase SHA-256 digest")
        if (not isinstance(self.state_dict_sha256, str)
                or len(self.state_dict_sha256) != 64
                or any(character not in "0123456789abcdef" for character in self.state_dict_sha256)):
            raise ValueError("state_dict_sha256 must be a lowercase SHA-256 digest")
        if not isinstance(self.state_dict, Mapping) or not self.state_dict:
            raise ValueError("source state_dict must be a non-empty mapping")
        if any(not isinstance(name, str) or not isinstance(value, torch.Tensor)
               for name, value in self.state_dict.items()):
            raise TypeError("source state_dict must map names to tensors")
        if any(value.device.type != "cpu" for value in self.state_dict.values()):
            raise ValueError("source state_dict tensors must be on CPU")
        if compute_state_dict_sha256(self.state_dict) != self.state_dict_sha256:
            raise ValueError("state_dict_sha256 does not match the supplied source weights")
        _validate_ids(self.target_fit_patient_ids, "target_fit_patient_ids", allow_empty=True)


@dataclass(frozen=True)
class BranchInitialization:
    """Required pathway-branch initialization settings for P-C/P-R/P-T/P-N."""

    rank: int
    branch_seed: int
    zero_initialize: bool

    def validate(self) -> None:
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank < 1:
            raise ValueError("branch rank must be a positive integer")
        if (isinstance(self.branch_seed, bool) or not isinstance(self.branch_seed, int)
                or self.branch_seed < 0 or self.branch_seed > 0x7FFF_FFFF_FFFF_FFFF):
            raise ValueError("branch_seed must be a non-negative 63-bit integer")
        if not isinstance(self.zero_initialize, bool):
            raise TypeError("zero_initialize must be bool")


@dataclass(frozen=True)
class CandidateTraining:
    """One fully specified candidate; no pending search value is defaulted."""

    learning_rate: float
    weight_decay: float
    ranking_weight: float
    max_epochs: int
    patience: int
    batch_size: int
    min_improvement: float

    def validate(self) -> None:
        numeric = (self.learning_rate, self.weight_decay, self.ranking_weight,
                   self.min_improvement)
        if any(not math.isfinite(value) for value in numeric):
            raise ValueError("candidate numeric settings must be finite")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.ranking_weight < 0:
            raise ValueError("learning_rate must be positive; weights cannot be negative")
        if self.min_improvement < 0:
            raise ValueError("min_improvement cannot be negative")
        for name, value in (("max_epochs", self.max_epochs), ("patience", self.patience),
                            ("batch_size", self.batch_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class InnerFoldReceipt:
    """Validation-only selection evidence from one inner fold."""

    best_validation_score: float
    best_epoch: int
    epochs_run: int
    response_mean_train_only: float
    response_scale_train_only: float
    train_patient_count: int
    validation_patient_count: int
    elapsed_seconds: float


@dataclass(frozen=True)
class ValidationSelectionState:
    """Running best epoch and patience state for inner validation."""

    best_score: float
    best_epoch: int
    stale_epochs: int


@dataclass(frozen=True)
class OuterKeyedPredictions:
    """Outer predictions keyed only by patient/drug; contains no observed label."""

    patient_ids: tuple[str, ...]
    drug_ids: tuple[str, ...]
    predicted_responses: np.ndarray

    def validate(self) -> None:
        if len(self.patient_ids) != len(self.drug_ids):
            raise ValueError("outer prediction keys are not aligned")
        if self.predicted_responses.shape != (len(self.patient_ids),):
            raise ValueError("outer predictions are not aligned to keys")
        if not np.isfinite(self.predicted_responses).all():
            raise ValueError("outer predictions must be finite")
        if len(set(zip(self.patient_ids, self.drug_ids))) != len(self.patient_ids):
            raise ValueError("outer predictions contain duplicate patient/drug keys")


@dataclass(frozen=True)
class OuterFoldReceipt:
    """Training-only receipt for the final outer-fold refit."""

    selected_epochs: int
    response_mean_train_only: float
    response_scale_train_only: float
    train_patient_count: int
    train_pair_count: int
    elapsed_seconds: float


class _TensorPairData:
    """Minimal adapter required by EXP-036's prediction helper."""

    def __init__(self, batch: PairInputBatch):
        self.batch = batch

    def tensor_pair(self, index: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        selector = torch.as_tensor(index, dtype=torch.long, device=self.batch.expression.device)
        return self.batch.expression[selector], self.batch.drug_features[selector]


def _validate_ids(values: Sequence[str], name: str, *, allow_empty: bool = False) -> None:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of identity strings")
    if not isinstance(values, (tuple, list)):
        raise TypeError(f"{name} must be a tuple or list")
    if not allow_empty and not values:
        raise ValueError(f"{name} must not be empty")
    if any(not isinstance(value, str) or not value or value != value.strip()
           for value in values):
        raise ValueError(f"{name} must contain non-empty trimmed strings")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must contain unique identities")


def compute_state_dict_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Hash sorted state names, dtypes, shapes, and contiguous CPU tensor bytes."""
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("state_dict must be a non-empty mapping")
    digest = hashlib.sha256()

    def feed(payload: bytes | memoryview) -> None:
        digest.update(len(payload).to_bytes(8, byteorder="big", signed=False))
        digest.update(payload)

    feed(b"exp046-state-dict-v1")
    for name in sorted(state_dict):
        value = state_dict[name]
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise TypeError("state_dict must map string names to tensors")
        if value.layout != torch.strided:
            raise ValueError("state_dict tensors must use the strided layout")
        if value.device.type != "cpu":
            raise ValueError("state_dict tensors must be on CPU before hashing")
        tensor = value.detach().contiguous()
        metadata = (
            name.encode("utf-8") + b"\0"
            + str(tensor.dtype).encode("ascii") + b"\0"
            + ",".join(str(size) for size in tensor.shape).encode("ascii") + b"\0"
            + sys.byteorder.encode("ascii")
        )
        feed(metadata)
        raw_bytes = memoryview(tensor.reshape(-1).view(torch.uint8).numpy())
        feed(raw_bytes)
    return digest.hexdigest()


def _validate_id_values(values: Sequence[str], name: str) -> None:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of identity strings")
    if any(not isinstance(value, str) or not value or value != value.strip()
           for value in values):
        raise ValueError(f"{name} must contain non-empty trimmed strings")


def _validate_pair_inputs(batch: PairInputBatch) -> None:
    if not isinstance(batch, PairInputBatch):
        raise TypeError("batch must be PairInputBatch or LabeledPairBatch")
    if not isinstance(batch.expression, torch.Tensor) or not isinstance(batch.drug_features, torch.Tensor):
        raise TypeError("expression and drug_features must be torch tensors")
    if batch.expression.ndim != 2 or batch.drug_features.ndim != 2:
        raise ValueError("expression and drug_features must be batch-by-feature matrices")
    count = batch.expression.shape[0]
    if count < 1 or batch.drug_features.shape[0] != count:
        raise ValueError("expression and drug feature rows must be non-empty and aligned")
    if len(batch.patient_ids) != count or len(batch.drug_ids) != count:
        raise ValueError("pair keys must be aligned to model input rows")
    _validate_id_values(batch.patient_ids, "patient_ids")
    _validate_id_values(batch.drug_ids, "drug_ids")
    keys = tuple(zip(batch.patient_ids, batch.drug_ids))
    if len(keys) != len(set(keys)):
        raise ValueError("batch contains duplicate patient/drug keys")
    for name, value in (("expression", batch.expression), ("drug_features", batch.drug_features)):
        if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must contain only finite floating point values")
    if batch.expression.device != batch.drug_features.device:
        raise ValueError("expression and drug_features must share a device")


def patient_group_indices(patient_ids: Sequence[str]) -> tuple[np.ndarray, ...]:
    """Return complete within-patient pair panels for patient-equal updates."""
    if not patient_ids:
        raise ValueError("patient_ids must not be empty")
    _validate_id_values(patient_ids, "patient_ids")
    positions: dict[str, list[int]] = {}
    for index, patient in enumerate(patient_ids):
        positions.setdefault(patient, []).append(index)
    return tuple(np.asarray(positions[patient], dtype=np.int64)
                 for patient in sorted(positions))


def patient_training_objective(
    predicted: torch.Tensor,
    observed: torch.Tensor,
    *,
    ranking_weight: float,
) -> torch.Tensor:
    """EXP-036-compatible Huber plus within-patient pairwise ranking loss."""
    if not math.isfinite(ranking_weight) or ranking_weight < 0:
        raise ValueError("ranking_weight must be finite and non-negative")
    if predicted.ndim != 1 or observed.shape != predicted.shape:
        raise ValueError("predictions and labels must be aligned vectors")
    regression = torch.nn.functional.smooth_l1_loss(predicted, observed)
    if ranking_weight == 0:
        return regression
    return regression + ranking_weight * pairwise_ranking_loss(predicted, observed)


def select_refit_epoch(inner_best_epochs: Sequence[int]) -> int:
    """Use the median inner-fold best epoch for the final refit."""
    if not inner_best_epochs:
        raise ValueError("at least one inner best epoch is required")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1
           for value in inner_best_epochs):
        raise ValueError("inner best epochs must be positive integers")
    return max(1, int(round(float(np.median(inner_best_epochs)))) )


def update_validation_selection(
    state: ValidationSelectionState,
    *,
    score: float,
    epoch: int,
    min_improvement: float,
    patience: int,
) -> tuple[ValidationSelectionState, bool]:
    """Update inner best-epoch selection; return whether patience is exhausted."""
    if not isinstance(state, ValidationSelectionState):
        raise TypeError("state must be ValidationSelectionState")
    if not math.isfinite(score):
        raise ValueError("validation score must be finite")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
        raise ValueError("epoch must be a positive integer")
    if not math.isfinite(min_improvement) or min_improvement < 0:
        raise ValueError("min_improvement must be finite and non-negative")
    if isinstance(patience, bool) or not isinstance(patience, int) or patience < 1:
        raise ValueError("patience must be a positive integer")
    if state.best_epoch < 0 or state.stale_epochs < 0:
        raise ValueError("validation state counters cannot be negative")
    if score > state.best_score + min_improvement:
        return ValidationSelectionState(score, epoch, 0), False
    stale = state.stale_epochs + 1
    return ValidationSelectionState(state.best_score, state.best_epoch, stale), stale >= patience


def require_cuda_training_device(device: torch.device) -> None:
    """Reject CPU and unavailable CUDA before entering a training entry point."""
    if not isinstance(device, torch.device):
        raise TypeError("device must be a torch.device")
    if device.type != "cuda":
        raise ValueError("EXP-046 fold training requires CUDA; CPU fallback is disabled")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for EXP-046 fold training but is unavailable")


def _start_cuda_timing(device: torch.device) -> float:
    """Flush earlier CUDA work before timing one public fold API call."""
    torch.cuda.synchronize(device)
    return time.perf_counter()


def _finish_cuda_timing(started: float, device: torch.device) -> float:
    """Include pending model, training, and prediction kernels in elapsed time."""
    torch.cuda.synchronize(device)
    return time.perf_counter() - started


def _validate_split(
    train: LabeledPairBatch,
    validation: LabeledPairBatch | None,
    source: SourceFoldInitialization,
    fold_key: str,
) -> None:
    train.validate()
    source.validate()
    if source.outer_fold_key != fold_key:
        raise ValueError("source initialization is bound to a different outer fold")
    train_patients = set(train.patient_ids)
    if not set(source.target_fit_patient_ids).issubset(train_patients):
        raise ValueError("source target-label exposure must be a subset of this training slice")
    if validation is not None:
        validation.validate()
        if train_patients.intersection(validation.patient_ids):
            raise ValueError("training and validation patients must be disjoint")


def build_fold_arm_model(
    arm: Exp046Arm,
    *,
    backbone_factory: Callable[[], nn.Module],
    source: SourceFoldInitialization,
    expected_outer_fold_key: str,
    model_seed: int,
    branch_initialization: BranchInitialization | None,
    ordinary_control_map: AlignedPathwayMap | None,
    frozen_random_map: AlignedPathwayMap | None,
    true_pathway_map: AlignedPathwayMap | None,
) -> nn.Module:
    """Build one fresh arm from fold weights; P-N restores every cold state tensor.

    ``backbone_factory`` must reproduce the pinned source architecture. The
    source-stage builder remains responsible for creating fold-safe weights
    and the exposure receipt; this adapter rejects receipt/key mismatches.
    """
    if not callable(backbone_factory):
        raise TypeError("backbone_factory must be callable")
    source.validate()
    if source.outer_fold_key != expected_outer_fold_key:
        raise ValueError("source initialization is bound to a different outer fold")
    if isinstance(model_seed, bool) or not isinstance(model_seed, int) or model_seed < 0:
        raise ValueError("model_seed must be a non-negative integer")
    if arm == "P-A":
        if branch_initialization is not None:
            raise ValueError("P-A must not receive branch initialization settings")
    else:
        if branch_initialization is None:
            raise ValueError("pathway arms require explicit branch initialization settings")
        branch_initialization.validate()

    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(model_seed)
        backbone = backbone_factory()
    if not isinstance(backbone, nn.Module):
        raise TypeError("backbone_factory must return a torch module")
    if any(tensor.device.type != "cpu" for tensor in (*backbone.parameters(), *backbone.buffers())):
        raise ValueError("backbone_factory must construct on CPU; training device is selected explicitly")

    if arm == "P-A":
        if any(value is not None for value in (ordinary_control_map, frozen_random_map, true_pathway_map)):
            raise ValueError("P-A must be constructed without pathway maps")
        model = backbone
        cold_state = None
    else:
        model = build_exp046_arm_model(
            arm,
            backbone,
            ordinary_control_map=ordinary_control_map,
            frozen_random_map=frozen_random_map,
            true_pathway_map=true_pathway_map,
            rank=branch_initialization.rank,
            branch_seed=branch_initialization.branch_seed,
            zero_initialize=branch_initialization.zero_initialize,
        )
        cold_state = capture_cold_start_state(model)

    backbone_target = getattr(model, "backbone", model)
    backbone_target.load_state_dict(source.state_dict, strict=True)
    if arm == "P-A":
        # EXP-036 uses source_full with max(1, candidate.adapter_rank).
        model = configure_target_variant(backbone_target, "source_full", adapter_rank=1)
    elif arm == "P-N":
        assert cold_state is not None
        restore_cold_start_state(model, cold_state)
    return model


def _standardize_training_responses(batch: LabeledPairBatch) -> tuple[float, float, torch.Tensor]:
    labels = batch.responses.detach().to(dtype=torch.float64)
    groups = patient_group_indices(batch.patient_ids)
    weights = torch.empty_like(labels)
    for group in groups:
        index = torch.as_tensor(group, dtype=torch.long, device=weights.device)
        weights[index] = 1.0 / len(group)
    mean = float((weights * labels).sum() / weights.sum())
    variance = float((weights * (labels - mean).square()).sum() / weights.sum())
    scale = math.sqrt(variance)
    if not math.isfinite(scale) or scale <= 1e-8:
        raise ValueError("training response labels have zero or invalid variation")
    return mean, scale, ((labels - mean) / scale).to(dtype=torch.float32)


def _patient_macro_spearman(
    batch: LabeledPairBatch,
    predicted: np.ndarray,
) -> float:
    observed = batch.responses.detach().cpu().numpy().astype(np.float64, copy=False)
    if predicted.shape != observed.shape or not np.isfinite(predicted).all():
        raise ValueError("validation predictions must be finite and aligned")
    correlations: list[float] = []
    for group in patient_group_indices(batch.patient_ids):
        if len(group) < 3:
            continue
        y, p = observed[group], predicted[group]
        if np.ptp(y) == 0 or np.ptp(p) == 0:
            continue
        score = float(spearmanr(y, p).statistic)
        if math.isfinite(score):
            correlations.append(score)
    if not correlations:
        raise ValueError("validation has no patient with a defined within-patient Spearman score")
    return float(np.mean(correlations))


def _fit_training_epochs(
    model: nn.Module,
    train: LabeledPairBatch,
    *,
    validation: LabeledPairBatch | None,
    epochs: int,
    settings: CandidateTraining,
    seed: int,
    device: torch.device,
) -> tuple[int, int, float | None, float, float]:
    settings.validate()
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    mean, scale, standardized = _standardize_training_responses(train)
    groups = patient_group_indices(train.patient_ids)
    if any(len(group) > settings.batch_size for group in groups):
        raise ValueError("batch_size must hold every training patient's complete drug panel")

    model.to(device)
    if any(tensor.device != device for tensor in (*model.parameters(), *model.buffers())):
        raise RuntimeError("model did not remain on the explicitly requested CUDA device")
    train_x = train.expression.to(device)
    train_d = train.drug_features.to(device)
    train_y = standardized.to(device)
    if validation is not None:
        validation = LabeledPairBatch(
            validation.expression.to(device), validation.drug_features.to(device),
            validation.patient_ids, validation.drug_ids, validation.responses.to(device))
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("model has no trainable parameters")
    optimizer = torch.optim.AdamW(parameters, lr=settings.learning_rate,
                                  weight_decay=settings.weight_decay)
    selection = ValidationSelectionState(best_score=-float("inf"), best_epoch=0,
                                         stale_epochs=0)
    epochs_run = 0
    with torch.random.fork_rng(devices=[device.index if device.index is not None
                                       else torch.cuda.current_device()]):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        for epoch in range(1, epochs + 1):
            epochs_run = epoch
            model.train()
            order = np.random.default_rng(seed + epoch).permutation(len(groups))
            for group_position in order:
                positions = groups[int(group_position)]
                selector = torch.as_tensor(positions, dtype=torch.long, device=device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    prediction = model(train_x[selector], train_d[selector]).float().reshape(-1)
                    loss = patient_training_objective(
                        prediction, train_y[selector], ranking_weight=settings.ranking_weight)
                if not bool(torch.isfinite(loss)):
                    raise ValueError("training loss became non-finite")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if any(parameter.grad is not None
                       and not bool(torch.isfinite(parameter.grad).all())
                       for parameter in parameters):
                    raise ValueError("training gradient became non-finite")
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
            if validation is None:
                best_epoch = epoch
                continue
            index = np.arange(len(validation.patient_ids), dtype=np.int64)
            validation_prediction = exp036_predict(
                model, _TensorPairData(validation), index, device, settings.batch_size)
            score = _patient_macro_spearman(validation, validation_prediction)
            selection, should_stop = update_validation_selection(
                selection, score=score, epoch=epoch,
                min_improvement=settings.min_improvement,
                patience=settings.patience)
            if should_stop:
                break
        if validation is None:
            best_epoch = epoch
        else:
            best_epoch = selection.best_epoch
    best_score = selection.best_score
    if validation is not None and best_epoch == 0:
        raise ValueError("inner validation did not produce a selectable score")
    torch.cuda.synchronize(device)
    return best_epoch, epochs_run, (best_score if validation is not None else None), mean, scale


def fit_inner_candidate(
    train: LabeledPairBatch,
    validation: LabeledPairBatch,
    *,
    arm: Exp046Arm,
    backbone_factory: Callable[[], nn.Module],
    source: SourceFoldInitialization,
    outer_fold_key: str,
    inner_fold_key: str,
    model_seed: int,
    branch_initialization: BranchInitialization | None,
    ordinary_control_map: AlignedPathwayMap | None,
    frozen_random_map: AlignedPathwayMap | None,
    true_pathway_map: AlignedPathwayMap | None,
    settings: CandidateTraining,
    device: torch.device,
) -> InnerFoldReceipt:
    """Fit one explicit candidate on an inner train slice and select by validation.

    A fold driver calls this once per candidate and inner split, then performs
    candidate aggregation/selection outside this module. No outer data is
    accepted here.
    """
    require_cuda_training_device(device)
    if not isinstance(inner_fold_key, str) or not inner_fold_key.strip():
        raise ValueError("inner_fold_key must be a non-empty string")
    _validate_split(train, validation, source, outer_fold_key)
    settings.validate()
    if set(source.target_fit_patient_ids).difference(train.patient_ids):
        raise ValueError("source target-label exposure includes an inner validation identity")
    started = _start_cuda_timing(device)
    model = build_fold_arm_model(
        arm, backbone_factory=backbone_factory, source=source,
        expected_outer_fold_key=outer_fold_key, model_seed=model_seed,
        branch_initialization=branch_initialization,
        ordinary_control_map=ordinary_control_map,
        frozen_random_map=frozen_random_map, true_pathway_map=true_pathway_map)
    best_epoch, epochs_run, score, mean, scale = _fit_training_epochs(
        model, train, validation=validation, epochs=settings.max_epochs,
        settings=settings, seed=model_seed, device=device)
    assert score is not None
    torch.cuda.empty_cache()
    elapsed = _finish_cuda_timing(started, device)
    return InnerFoldReceipt(
        best_validation_score=score,
        best_epoch=best_epoch,
        epochs_run=epochs_run,
        response_mean_train_only=mean,
        response_scale_train_only=scale,
        train_patient_count=len(set(train.patient_ids)),
        validation_patient_count=len(set(validation.patient_ids)),
        elapsed_seconds=elapsed,
    )


def fit_outer_fold(
    train: LabeledPairBatch,
    outer_inputs: PairInputBatch,
    *,
    arm: Exp046Arm,
    backbone_factory: Callable[[], nn.Module],
    source: SourceFoldInitialization,
    outer_fold_key: str,
    model_seed: int,
    branch_initialization: BranchInitialization | None,
    ordinary_control_map: AlignedPathwayMap | None,
    frozen_random_map: AlignedPathwayMap | None,
    true_pathway_map: AlignedPathwayMap | None,
    settings: CandidateTraining,
    selected_epochs: int,
    device: torch.device,
) -> tuple[OuterKeyedPredictions, OuterFoldReceipt]:
    """Refit on explicit outer-training labels and predict unlabeled outer inputs.

    ``outer_inputs`` has no label member by construction. This function emits
    keyed predictions and a training-only receipt; scoring requires a separate
    evaluator that joins labels after this function has returned.
    """
    if type(outer_inputs) is not PairInputBatch:
        raise TypeError("outer_inputs must be exactly PairInputBatch and contain no labels")
    require_cuda_training_device(device)
    train.validate()
    outer_inputs.validate()
    source.validate()
    if source.outer_fold_key != outer_fold_key:
        raise ValueError("source initialization is bound to a different outer fold")
    train_patients = set(train.patient_ids)
    if not set(source.target_fit_patient_ids).issubset(train_patients):
        raise ValueError("source target-label exposure must be a subset of outer training patients")
    if train_patients.intersection(outer_inputs.patient_ids):
        raise ValueError("outer training and outer prediction patients must be disjoint")
    settings.validate()
    if isinstance(selected_epochs, bool) or not isinstance(selected_epochs, int) or selected_epochs < 1:
        raise ValueError("selected_epochs must be a positive inner-selected epoch count")

    started = _start_cuda_timing(device)
    model = build_fold_arm_model(
        arm, backbone_factory=backbone_factory, source=source,
        expected_outer_fold_key=outer_fold_key, model_seed=model_seed,
        branch_initialization=branch_initialization,
        ordinary_control_map=ordinary_control_map,
        frozen_random_map=frozen_random_map, true_pathway_map=true_pathway_map)
    best_epoch, epochs_run, _, mean, scale = _fit_training_epochs(
        model, train, validation=None, epochs=selected_epochs,
        settings=settings, seed=model_seed, device=device)
    if best_epoch != selected_epochs or epochs_run != selected_epochs:
        raise RuntimeError("outer refit did not complete the caller-selected epoch count")
    outer_on_device = PairInputBatch(
        outer_inputs.expression.to(device), outer_inputs.drug_features.to(device),
        outer_inputs.patient_ids, outer_inputs.drug_ids)
    indices = np.arange(len(outer_inputs.patient_ids), dtype=np.int64)
    standardized = exp036_predict(
        model, _TensorPairData(outer_on_device), indices, device, settings.batch_size)
    predicted = standardized * scale + mean
    keyed = OuterKeyedPredictions(
        patient_ids=outer_inputs.patient_ids,
        drug_ids=outer_inputs.drug_ids,
        predicted_responses=np.asarray(predicted, dtype=np.float64),
    )
    keyed.validate()
    torch.cuda.empty_cache()
    elapsed = _finish_cuda_timing(started, device)
    return keyed, OuterFoldReceipt(
        selected_epochs=selected_epochs,
        response_mean_train_only=mean,
        response_scale_train_only=scale,
        train_patient_count=len(train_patients),
        train_pair_count=len(train.patient_ids),
        elapsed_seconds=elapsed,
    )
