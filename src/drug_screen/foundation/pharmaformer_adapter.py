"""Leakage-safe boundary around the external PharmaFormer implementation.

The upstream checkout is treated as an immutable dependency.  In particular we
do not call its ``data_prepare`` or ``train_and_evaluate_kfold`` helpers because
they fit transforms before pair-level KFold splitting.  This module imports only
``CombinedModel`` and supplies explicit arrays, folds and device selection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import importlib.util
import io
import json
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

UPSTREAM_COMMIT = "f526ddda4091151f082b38367571a94823314aba"
CHECKPOINT_FORMAT = "drugscreenlab_exp010_pharmaformer_v1"


class PharmaFormerIntegrationError(RuntimeError):
    """Raised when the immutable upstream boundary cannot be verified."""


@dataclass(frozen=True)
class PharmaFormerConfig:
    gene_input_size: int
    gene_hidden_size: int = 8192
    drug_hidden_size: int = 256
    feature_dim: int = 128
    nhead: int = 8
    num_layers: int = 3
    dim_feedforward: int = 2048
    dropout: float = 0.1

    def validate(self) -> None:
        total = self.gene_hidden_size + self.drug_hidden_size
        if total % self.feature_dim:
            raise ValueError("gene_hidden_size + drug_hidden_size must divide into feature_dim tokens")
        if self.feature_dim % self.nhead:
            raise ValueError("feature_dim must be divisible by nhead")


class TrainOnlyGeneTransform:
    """Upstream-compatible StandardScaler then MinMaxScaler, fit on train only."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.min_: np.ndarray | None = None
        self.range_: np.ndarray | None = None
        self.fit_sample_ids: tuple[str, ...] = ()

    def fit(self, x: np.ndarray, sample_ids: Sequence[str]) -> "TrainOnlyGeneTransform":
        values = np.asarray(x, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] != len(sample_ids) or np.isinf(values).any():
            raise ValueError("2D train expression without infinities and aligned sample_ids are required")
        self.mean_ = np.nanmean(values, axis=0)
        if np.isnan(self.mean_).any():
            raise ValueError("train expression contains an all-missing feature")
        values = np.where(np.isnan(values), self.mean_, values)
        std = values.std(axis=0)
        self.scale_ = np.where(std == 0, 1.0, std)
        standardized = (values - self.mean_) / self.scale_
        self.min_ = standardized.min(axis=0)
        span = standardized.max(axis=0) - self.min_
        self.range_ = np.where(span == 0, 1.0, span)
        self.fit_sample_ids = tuple(str(x) for x in sample_ids)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if any(v is None for v in (self.mean_, self.scale_, self.min_, self.range_)):
            raise RuntimeError("gene transform is not fit")
        values = np.asarray(x, dtype=np.float64)
        assert self.mean_ is not None and self.scale_ is not None
        assert self.min_ is not None and self.range_ is not None
        if values.ndim != 2 or values.shape[1] != self.mean_.shape[0]:
            raise ValueError("expression feature count differs from fitted transform")
        if np.isinf(values).any():
            raise ValueError("expression contains infinities")
        values = np.where(np.isnan(values), self.mean_, values)
        return (((values - self.mean_) / self.scale_ - self.min_) / self.range_).astype(np.float32)

    def state_dict(self) -> dict[str, Any]:
        if self.mean_ is None:
            raise RuntimeError("gene transform is not fit")
        return {"mean": self.mean_, "scale": self.scale_, "min": self.min_, "range": self.range_,
                "fit_sample_ids": self.fit_sample_ids, "fit_scope": "train_only"}

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "TrainOnlyGeneTransform":
        obj = cls()
        obj.mean_ = np.asarray(state["mean"])
        obj.scale_ = np.asarray(state["scale"])
        obj.min_ = np.asarray(state["min"])
        obj.range_ = np.asarray(state["range"])
        obj.fit_sample_ids = tuple(state["fit_sample_ids"])
        return obj


class TrainOnlySmilesEncoder:
    """Exact upstream BPE->ordinal->pad(128) representation.

    BPE codes are an explicit source-training asset.  They are never learned
    from a target fold, and their full contents and digest travel in checkpoints.
    """

    width = 128

    def __init__(self) -> None:
        self.allowed_: frozenset[str] | None = None
        self.fit_drug_ids: tuple[str, ...] = ()
        self.bpe_codes: str | None = None
        self.bpe_sha256: str | None = None

    def fit(self, smiles: Sequence[str], drug_ids: Sequence[str], *, bpe_codes_path: str | Path) -> "TrainOnlySmilesEncoder":
        if len(smiles) != len(drug_ids) or len(smiles) == 0:
            raise ValueError("aligned non-empty training SMILES and drug IDs are required")
        path = Path(bpe_codes_path)
        if not path.is_file():
            raise ValueError(f"frozen upstream BPE codes are missing: {path}")
        self.bpe_codes = path.read_text(encoding="utf-8")
        if not self.bpe_codes.strip():
            raise ValueError("frozen upstream BPE codes are empty")
        self.bpe_sha256 = sha256(self.bpe_codes.encode()).hexdigest()
        self.allowed_ = frozenset(str(s) for s in smiles)
        self.fit_drug_ids = tuple(str(x) for x in drug_ids)
        return self

    def transform(self, smiles: Sequence[str], *, allow_external_vocabulary: bool = False) -> np.ndarray:
        if self.allowed_ is None:
            raise RuntimeError("SMILES encoder is not fit")
        unknown = sorted({str(s) for s in smiles if str(s) not in self.allowed_})
        if unknown and not allow_external_vocabulary:
            raise ValueError(f"SMILES absent from training vocabulary: {unknown[:3]}")
        out = np.zeros((len(smiles), self.width), dtype=np.float32)
        if self.bpe_codes is None:
            raise RuntimeError("SMILES encoder has no frozen BPE codes")
        try:
            from subword_nmt.apply_bpe import BPE
        except ImportError as error:
            raise PharmaFormerIntegrationError("subword-nmt is required for upstream SMILES encoding") from error
        bpe = BPE(io.StringIO(self.bpe_codes))
        cache: dict[str, list[int]] = {}
        for i, value in enumerate(smiles):
            text_value = str(value)
            encoded = cache.get(text_value)
            if encoded is None:
                encoded = [ord(char) for char in bpe.process_line(text_value)[: self.width]]
                cache[text_value] = encoded
            out[i, : len(encoded)] = encoded
        return out

    def state_dict(self) -> dict[str, Any]:
        if self.allowed_ is None:
            raise RuntimeError("SMILES encoder is not fit")
        return {"allowed_smiles": sorted(self.allowed_), "fit_drug_ids": self.fit_drug_ids,
                "fit_scope": "source_train_only", "algorithm": "upstream_subword_nmt_BPE_then_ordinal_width_128",
                "bpe_codes": self.bpe_codes, "bpe_sha256": self.bpe_sha256}

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "TrainOnlySmilesEncoder":
        obj = cls()
        obj.allowed_ = frozenset(state["allowed_smiles"])
        obj.fit_drug_ids = tuple(state["fit_drug_ids"])
        obj.bpe_codes = str(state["bpe_codes"])
        obj.bpe_sha256 = str(state["bpe_sha256"])
        if sha256(obj.bpe_codes.encode()).hexdigest() != obj.bpe_sha256:
            raise PharmaFormerIntegrationError("checkpoint BPE codes digest mismatch")
        return obj


def validate_patient_fold(rows: Sequence[Mapping[str, Any]]) -> dict[str, frozenset[str]]:
    """Require every patient and all of their PDO/region rows in one partition."""
    required = {"patient_id", "pdo_id", "partition"}
    if not rows:
        raise ValueError("fold rows are empty")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"fold rows miss fields: {sorted(missing)}")
    by_patient: dict[str, set[str]] = {}
    by_pdo: dict[str, set[str]] = {}
    for row in rows:
        patient, pdo, partition = (str(row[k]) for k in ("patient_id", "pdo_id", "partition"))
        if partition not in {"train", "validation", "test"}:
            raise ValueError(f"invalid partition {partition}")
        by_patient.setdefault(patient, set()).add(partition)
        by_pdo.setdefault(pdo, set()).add(partition)
    leaking = {k: v for k, v in by_patient.items() if len(v) != 1}
    leaking_pdo = {k: v for k, v in by_pdo.items() if len(v) != 1}
    if leaking or leaking_pdo:
        raise ValueError(f"patient/PDO leakage: patients={leaking}, pdos={leaking_pdo}")
    return {p: frozenset(k for k, v in by_patient.items() if next(iter(v)) == p)
            for p in ("train", "validation", "test")}


def load_upstream_module(source_root: str | Path) -> ModuleType:
    root = Path(source_root).resolve()
    source = root / "pretrain.py"
    if not source.is_file():
        raise PharmaFormerIntegrationError(f"upstream pretrain.py is missing: {source}")
    import subprocess
    try:
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
                                capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise PharmaFormerIntegrationError("upstream source must be a verifiable Git checkout") from error
    if commit != UPSTREAM_COMMIT:
        raise PharmaFormerIntegrationError(f"expected upstream {UPSTREAM_COMMIT}, found {commit}")
    spec = importlib.util.spec_from_file_location("drugscreenlab_external_pharmaformer", source)
    if spec is None or spec.loader is None:
        raise PharmaFormerIntegrationError("cannot create upstream import specification")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "CombinedModel"):
        raise PharmaFormerIntegrationError("upstream module has no CombinedModel")
    return module


def build_model(upstream: ModuleType, config: PharmaFormerConfig) -> Any:
    config.validate()
    return upstream.CombinedModel(**asdict(config))


def save_checkpoint(path: str | Path, model: Any, config: PharmaFormerConfig,
                    gene_transform: TrainOnlyGeneTransform, smiles_encoder: TrainOnlySmilesEncoder,
                    *, variant: str, fold_id: int, provenance: Mapping[str, Any]) -> Path:
    import torch
    if variant not in {"M2", "M3", "M4"}:
        raise ValueError("variant must be M2, M3 or M4")
    payload = {"format": CHECKPOINT_FORMAT, "experiment_id": "EXP-010", "variant": variant,
               "fold_id": int(fold_id), "upstream_commit": UPSTREAM_COMMIT,
               "model_config": asdict(config), "model_state_dict": model.state_dict(),
               "gene_transform": gene_transform.state_dict(), "smiles_encoder": smiles_encoder.state_dict(),
               "provenance": dict(provenance)}
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)
    return target


def load_checkpoint(path: str | Path, upstream: ModuleType, *, device: str = "cpu") -> tuple[Any, dict[str, Any]]:
    import torch
    try:
        payload = torch.load(Path(path), map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(Path(path), map_location=device)
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise PharmaFormerIntegrationError("not an EXP-010 PharmaFormer checkpoint")
    if payload.get("upstream_commit") != UPSTREAM_COMMIT:
        raise PharmaFormerIntegrationError("checkpoint upstream commit differs from frozen contract")
    config = PharmaFormerConfig(**payload["model_config"])
    model = build_model(upstream, config).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    payload["gene_transform_object"] = TrainOnlyGeneTransform.from_state_dict(payload["gene_transform"])
    payload["smiles_encoder_object"] = TrainOnlySmilesEncoder.from_state_dict(payload["smiles_encoder"])
    return model, payload


def contract_digest(config: PharmaFormerConfig, ordered_genes: Iterable[str]) -> str:
    value = {"config": asdict(config), "ordered_genes": list(ordered_genes), "upstream_commit": UPSTREAM_COMMIT}
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
