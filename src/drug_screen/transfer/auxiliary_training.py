"""N0/N1/N3 auxiliary training without changing the inference contract.

The learned drug model remains a ``DualEncoderPairModel``: its public
``predict`` method takes only preprocessed baseline expression and a Morgan
fingerprint.  Reconstruction and dependency heads exist only during training.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from .cohorts import load_registered_cohort
from .cohorts import sha256
from .models import DualEncoderPairModel
from .preprocessing import ExpressionPreprocessor
from .transfer_experiment import _move_to_device

_VARIANTS = frozenset({"N0", "N1", "N3"})


def select_dependency_targets(values: Any, train_rows: Any, genes: Any, *, max_targets: int = 512,
                              min_coverage: float = .9) -> tuple[np.ndarray, np.ndarray]:
    """Pick dependency columns from auxiliary-training rows only."""
    matrix = np.asarray(values, float); train = np.asarray(train_rows, bool)
    if matrix.ndim != 2 or len(train) != len(matrix) or not train.any(): raise ValueError("invalid dependency training mask")
    if not 0 < min_coverage <= 1 or max_targets < 1: raise ValueError("invalid dependency selection settings")
    coverage = np.isfinite(matrix[train]).mean(axis=0); allowed = np.flatnonzero(coverage >= min_coverage)
    if not len(allowed): raise ValueError("no dependency target has required training coverage")
    variance = np.nanvar(matrix[train][:, allowed], axis=0)
    selected = allowed[np.argsort(-variance, kind="mergesort")[:min(max_targets, len(allowed))]]
    return matrix[:, selected], np.asarray(genes, str)[selected]


def permute_dependency_rows(values: Any, cancer: Any, library: Any, *, seed: int) -> np.ndarray:
    """Permute whole dependency rows within cancer/library strata.

    A whole-row permutation preserves each row's missingness and gene-gene
    relation, preventing an overly easy independent-cell permutation control.
    """
    answer = np.asarray(values, float).copy(); c, lib = np.asarray(cancer, str), np.asarray(library, str)
    if len(answer) != len(c) or len(c) != len(lib): raise ValueError("dependency and stratum lengths differ")
    rng = np.random.default_rng(seed)
    for key in sorted(set(zip(c, lib, strict=True))):
        rows = np.flatnonzero((c == key[0]) & (lib == key[1]))
        if len(rows) > 1: answer[rows] = answer[rng.permutation(rows)]
    return answer


@dataclass
class AuxiliaryInputs:
    expression: np.ndarray
    dependency: np.ndarray
    cancer: np.ndarray
    library: np.ndarray
    sample_ids: np.ndarray
    dependency_genes: np.ndarray


class AuxiliaryTransferModel:
    """Source drug model plus training-only expression/dependency heads."""
    def __init__(self, drug_model: DualEncoderPairModel, *, dependency_dim: int, mask_rate: float = .15):
        if not 0 < mask_rate < 1: raise ValueError("mask_rate must be in (0, 1)")
        if drug_model.network is None or drug_model.sample_dim is None: raise ValueError("drug model must be fitted")
        self.drug_model, self.mask_rate = drug_model, float(mask_rate)
        device, hidden = drug_model.device, drug_model.config["hidden_dim"]
        self.reconstruction = nn.Linear(hidden, drug_model.sample_dim).to(device)
        self.dependency = nn.Linear(hidden, int(dependency_dim)).to(device)
        self.history: list[dict[str, float]] = []

    def predict(self, expression: Any, fingerprints: Any) -> np.ndarray:
        return self.drug_model.predict(expression, fingerprints)

    def _sample_embedding(self, x: torch.Tensor) -> torch.Tensor:
        assert self.drug_model.network is not None
        return self.drug_model.network.sample(x)

    def fit(self, drug_x: Any, drug_d: Any, drug_y: Any, drug_groups: Any, auxiliary: AuxiliaryInputs,
            *, variant: str, auxiliary_weight: float, steps: int, seed: int, dependency_enabled: bool = True,
            drug_batch_groups: int = 8, auxiliary_batch_size: int = 32) -> "AuxiliaryTransferModel":
        if variant not in _VARIANTS or steps < 1 or auxiliary_weight not in {0.0, .01, .1}: raise ValueError("invalid auxiliary protocol")
        x, d, y = (np.asarray(v, float) for v in (drug_x, drug_d, drug_y)); groups = np.asarray(drug_groups, str)
        aux_x, dep = np.asarray(auxiliary.expression, float), np.asarray(auxiliary.dependency, float)
        if x.ndim != 2 or d.ndim != 2 or len(x) != len(d) or len(y) != len(x) or len(groups) != len(x): raise ValueError("invalid drug arrays")
        if aux_x.ndim != 2 or dep.ndim != 2 or len(aux_x) != len(dep) or aux_x.shape[1] != x.shape[1]: raise ValueError("invalid auxiliary arrays")
        if variant == "N3" and dep.shape[1] != self.dependency.out_features: raise ValueError("dependency dimension mismatch")
        device = self.drug_model.device
        # Separate streams ensure N0's absence of an auxiliary update cannot
        # change the following drug replay batch relative to N1/N3.
        drug_rng, aux_rng = np.random.default_rng(seed), np.random.default_rng(seed + 1)
        mean, std = self.drug_model.label_mean, self.drug_model.label_std
        ta = torch.tensor(aux_x, dtype=torch.float32, device=device)
        tdv = torch.tensor(np.nan_to_num(dep, nan=0.0), dtype=torch.float32, device=device)
        valid_dep = torch.tensor(np.isfinite(dep), dtype=torch.float32, device=device)
        params = list(self.drug_model.network.parameters()) + list(self.reconstruction.parameters()) + list(self.dependency.parameters())
        optimizer = torch.optim.Adam(params, lr=self.drug_model.config["learning_rate"])
        self.history = []
        group_order = np.asarray(sorted(set(groups)))
        # Fixed-size biological-model group batches prevent the 179k source
        # pairs from becoming one giant GPU graph.  The seeded sequence is
        # shared across N0/N1/N3 runs with the same seed/configuration.
        if drug_batch_groups < 1 or auxiliary_batch_size < 1: raise ValueError("batch sizes must be positive")
        batch_groups = min(int(drug_batch_groups), len(group_order))
        aux_batch_size = min(int(auxiliary_batch_size), len(aux_x))
        for step in range(steps):
            selected_groups = drug_rng.choice(group_order, size=batch_groups, replace=False)
            rows = np.flatnonzero(np.isin(groups, selected_groups))
            bx, bd = (torch.tensor(v[rows], dtype=torch.float32, device=device) for v in (x, d))
            by = torch.tensor((y[rows] - mean) / std, dtype=torch.float32, device=device)
            local_groups, local_y = groups[rows], y[rows]
            pair_groups = [(indices[left], indices[right]) for indices, left, right in self.drug_model._group_rank_pairs(local_y, local_groups) if len(left)]
            # One source-drug update is always present.  It is replayed for N0
            # to match N1/N3's two optimizer updates per logical step.
            optimizer.zero_grad(); output = self.drug_model.network(bx, bd)
            drug_loss = F.huber_loss(output, by, delta=self.drug_model.config["huber_delta"])
            drug_loss = drug_loss + self.drug_model.config["ranking_weight"] * self.drug_model._ranking_loss(output, pair_groups=pair_groups)
            drug_loss.backward(); optimizer.step()
            if variant == "N0":
                optimizer.zero_grad(); replay = self.drug_model.network(bx, bd)
                replay_loss = F.huber_loss(replay, by, delta=self.drug_model.config["huber_delta"])
                replay_loss = replay_loss + self.drug_model.config["ranking_weight"] * self.drug_model._ranking_loss(replay, pair_groups=pair_groups)
                replay_loss.backward(); optimizer.step(); aux_loss = replay_loss.detach()
            else:
                aux_rows = aux_rng.choice(len(aux_x), size=aux_batch_size, replace=False)
                batch_aux = ta[aux_rows]; batch_dep, batch_valid = tdv[aux_rows], valid_dep[aux_rows]
                mask = torch.tensor(aux_rng.random(batch_aux.shape) < self.mask_rate, dtype=torch.bool, device=device)
                masked = batch_aux.masked_fill(mask, 0.0); embedding = self._sample_embedding(masked)
                reconstruction = self.reconstruction(embedding)
                reconstruction_loss = ((reconstruction - batch_aux).square() * mask).sum() / mask.sum().clamp_min(1)
                dep_loss = reconstruction_loss.new_zeros(())
                if variant == "N3" and dependency_enabled:
                    predicted_dep = self.dependency(embedding)
                    dep_loss = ((predicted_dep - batch_dep).square() * batch_valid).sum() / batch_valid.sum().clamp_min(1)
                aux_loss = reconstruction_loss + dep_loss
                optimizer.zero_grad(); (auxiliary_weight * aux_loss).backward(); optimizer.step()
            self.history.append({"step": step + 1, "drug_loss": float(drug_loss.detach().cpu()), "auxiliary_loss": float(aux_loss.detach().cpu())})
        return self

    def save(self, path: str | Path) -> None:
        if self.drug_model.network is None: raise RuntimeError("model is not fitted")
        torch.save({"drug": {"config": self.drug_model.config, "sample_dim": self.drug_model.sample_dim, "drug_dim": self.drug_model.drug_dim,
                              "label_mean": self.drug_model.label_mean, "label_std": self.drug_model.label_std, "state_dict": self.drug_model.network.state_dict()},
                    "reconstruction": self.reconstruction.state_dict(), "dependency": self.dependency.state_dict(), "dependency_dim": self.dependency.out_features,
                    "mask_rate": self.mask_rate, "history": self.history}, Path(path))

    @classmethod
    def load(cls, path: str | Path, *, device: str = "cpu") -> "AuxiliaryTransferModel":
        state = torch.load(Path(path), map_location="cpu", weights_only=False); drug = state["drug"]
        model = DualEncoderPairModel(**drug["config"], device=device); model._ensure_network(drug["sample_dim"], drug["drug_dim"])
        assert model.network is not None; model.network.load_state_dict(drug["state_dict"]); model.label_mean, model.label_std = drug["label_mean"], drug["label_std"]
        result = cls(model, dependency_dim=int(state["dependency_dim"]), mask_rate=float(state["mask_rate"]))
        result.reconstruction.load_state_dict(state["reconstruction"]); result.dependency.load_state_dict(state["dependency"]); result.history = list(state["history"])
        return result


def load_biobank_auxiliary(data_dir: str | Path, source_preprocessor: ExpressionPreprocessor, *, library: str = "minLib",
                           max_targets: int = 512, min_coverage: float = .9, permutation_seed: int | None = None) -> AuxiliaryInputs:
    """Materialise one labelled auxiliary table, retaining cancer/library strata."""
    root = Path(data_dir); samples = pd.read_parquet(root / "samples.parquet"); expression = pd.read_parquet(root / "expression_author_log2tpm.parquet"); dependency = pd.read_parquet(root / "dependency_by_library.parquet")
    expression.index = expression.index.astype(str); selected = samples.loc[samples.sample_ID.astype(str).isin(expression.index)].copy(); selected.sample_ID = selected.sample_ID.astype(str)
    dep = dependency.loc[dependency.library.astype(str).eq(library)].copy(); available = sorted(set(selected.sample_ID) & set(dep.sample_ID.astype(str)))
    if not available: raise ValueError("no paired samples for requested library")
    selected = selected.set_index("sample_ID").loc[available].reset_index(); genes = np.asarray([str(c).removesuffix("_exp") for c in expression.columns])
    transformed = source_preprocessor.transform(expression.loc[available].to_numpy(float), genes)
    pivot = dep.pivot(index="sample_ID", columns="gene", values="LFC").reindex(available); raw, dep_genes = pivot.to_numpy(float), pivot.columns.to_numpy(str)
    train_rows = np.ones(len(available), dtype=bool); chosen, target_genes = select_dependency_targets(raw, train_rows, dep_genes, max_targets=max_targets, min_coverage=min_coverage)
    cancer = selected.primary_tumour_type.astype(str).to_numpy(); libraries = np.full(len(selected), library, dtype=str)
    if permutation_seed is not None: chosen = permute_dependency_rows(chosen, cancer, libraries, seed=permutation_seed)
    return AuxiliaryInputs(transformed, chosen, cancer, libraries, np.asarray(available), target_genes)


def _hash(path: Path) -> str:
    return sha256(path)


def run_nature_auxiliary(config: Mapping[str, Any], *, source_output: str | Path, data_root: str | Path, source_registry: str | Path,
                         nature_data: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Run a bounded source-side N0/N1/N3 trial and save deployable drug weights."""
    cfg = dict(config); variant = str(cfg.get("variant", "")); output = Path(output_dir)
    if variant not in _VARIANTS or output.exists() and any(output.iterdir()): raise ValueError("new output and N0/N1/N3 variant required")
    source_prep = ExpressionPreprocessor.load(Path(source_output) / "expression_transform.npz")
    source_model = _move_to_device(DualEncoderPairModel.load(Path(source_output) / "source_model.pt"), str(cfg.get("device", "cpu")))
    cohort = load_registered_cohort(str(cfg.get("source_cohort_id", "gdsc2_v2")), root=Path(data_root), registry=Path(source_registry))
    positions = {str(k): i for i, k in enumerate(cohort.sample_ids)}; features = source_prep.transform(cohort.expression, cohort.gene_ids)
    drug_col = next((c for c in ("canonical_smiles", "canonical_isomeric_smiles", "smiles") if c in cohort.drugs), None)
    if drug_col is None: raise ValueError("source drug SMILES absent")
    from .chemistry import morgan_fingerprint
    chemical = cohort.drugs.set_index("drug_id")[drug_col].to_dict(); pairs = cohort.pairs.copy()
    # The smoke contract admits a deterministic biological-model replay pool;
    # this is not a held-out source evaluation and cannot support a source
    # generalization statement.  N0/N1/N3 receive the same pool and seed.
    all_models = np.asarray(sorted(pairs.sample_id.astype(str).unique()))
    replay_models = min(int(cfg.get("source_replay_models", len(all_models))), len(all_models))
    if replay_models < int(cfg.get("drug_batch_groups", 8)): raise ValueError("source replay pool must cover one drug batch")
    chosen_models = np.random.default_rng(int(cfg.get("seed", 0))).choice(all_models, size=replay_models, replace=False)
    pairs = pairs.loc[pairs.sample_id.astype(str).isin(chosen_models)].copy()
    # Fingerprints are drug entities, never response rows.  Materialising once
    # per unique drug keeps a short smoke run genuinely short and makes the
    # source replay sampling independent of redundant chemistry work.
    by_drug = {str(d): morgan_fingerprint(chemical[str(d)], radius=2, n_bits=2048) for d in pairs.drug_id.astype(str).unique()}
    fingerprints = np.vstack([by_drug[str(d)] for d in pairs.drug_id.astype(str)]).astype(float)
    x = features[[positions[str(v)] for v in pairs.sample_id]]; y, groups = pairs.response.to_numpy(float), pairs.sample_id.astype(str).to_numpy()
    aux = load_biobank_auxiliary(nature_data, source_prep, library=str(cfg.get("library", "minLib")), max_targets=int(cfg.get("max_targets", 512)), permutation_seed=cfg.get("permutation_seed"))
    model = AuxiliaryTransferModel(source_model, dependency_dim=aux.dependency.shape[1], mask_rate=float(cfg.get("mask_rate", .15)))
    model.fit(x, fingerprints, y, groups, aux, variant=variant, auxiliary_weight=float(cfg.get("auxiliary_weight", .1)), steps=int(cfg.get("steps", 10)), seed=int(cfg.get("seed", 0)), dependency_enabled=bool(cfg.get("dependency_enabled", True)), drug_batch_groups=int(cfg.get("drug_batch_groups", 8)), auxiliary_batch_size=int(cfg.get("auxiliary_batch_size", 32)))
    output.mkdir(parents=True); model.save(output / "auxiliary_transfer.pt"); model.drug_model.save(output / "source_model.pt"); source_prep.save(output / "expression_transform.npz")
    parent_frozen = json.loads((Path(source_output) / "frozen_run.json").read_text(encoding="utf-8"))
    input_hashes = {"source_registry": _hash(Path(source_registry)), "parent_source_model": _hash(Path(source_output) / "source_model.pt"),
                    "parent_expression_transform": _hash(Path(source_output) / "expression_transform.npz")}
    for name in ("samples.parquet", "expression_author_log2tpm.parquet", "dependency_by_library.parquet"):
        input_hashes[f"nature_{name}"] = _hash(Path(nature_data) / name)
    frozen = dict(parent_frozen) | {"auxiliary_protocol": {"variant": variant, "config": cfg, "input_hashes": input_hashes,
                                                          "source_replay_models": int(replay_models), "auxiliary_sample_ids": aux.sample_ids.tolist(),
                                                          "dependency_gene_ids": aux.dependency_genes.tolist()}}
    (output / "frozen_run.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    report = {"status": "SMOKE_NOT_DOWNSTREAM_EVIDENCE" if int(cfg.get("steps", 10)) <= 10 else "BOUNDED_DEVELOPMENT_SOURCE_ONLY", "variant": variant, "steps": len(model.history), "dependency_targets": int(aux.dependency.shape[1]), "auxiliary_samples": int(len(aux.sample_ids)), "drug_batch_groups": int(cfg.get("drug_batch_groups", 8)), "source_replay_models": int(replay_models), "auxiliary_batch_size": int(cfg.get("auxiliary_batch_size", 32)), "inference_requires": ["baseline_expression", "SMILES"], "last_history": model.history[-1], "input_hashes": input_hashes}
    (output / "report.json").write_text(json.dumps(report, indent=2)); return report
