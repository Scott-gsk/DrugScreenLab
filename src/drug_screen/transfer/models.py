"""Small, auditable pair models for preprocessed PDO transfer inputs.

These models accept arrays already produced by the frozen preprocessing and
chemical-fingerprint contracts.  They neither read data files nor infer
identities.  Their explanations describe model attribution only; they are not
mechanistic claims.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def _arrays(sample_features: Any, drug_fingerprints: Any, labels: Any | None = None):
    sample = np.asarray(sample_features, dtype=np.float64)
    drug = np.asarray(drug_fingerprints, dtype=np.float64)
    if sample.ndim != 2 or drug.ndim != 2 or sample.shape[0] != drug.shape[0] or sample.shape[0] == 0:
        raise ValueError("sample and drug features must be non-empty aligned 2D arrays")
    if not np.isfinite(sample).all() or not np.isfinite(drug).all():
        raise ValueError("features must be finite")
    if labels is None:
        return sample, drug
    target = np.asarray(labels, dtype=np.float64).reshape(-1)
    if target.shape[0] != sample.shape[0] or not np.isfinite(target).all():
        raise ValueError("labels must be aligned finite values")
    return sample, drug, target


class RidgePairModel:
    """Ridge regression with capped elementwise sample--drug interactions.

    The interaction block uses only the first ``interaction_features`` paired
    dimensions.  This deliberately bounds capacity while preventing the model
    from collapsing into a purely additive drug-main-effect predictor.
    """

    def __init__(self, *, alpha: float = 1.0, interaction_features: int = 16):
        if not np.isfinite(alpha) or alpha < 0 or isinstance(interaction_features, bool) or interaction_features < 1:
            raise ValueError("alpha must be non-negative and interaction_features positive")
        self.alpha = float(alpha)
        self.interaction_features = int(interaction_features)
        self.sample_dim: int | None = None
        self.drug_dim: int | None = None
        self.interaction_dim: int | None = None
        self.coefficients: np.ndarray | None = None
        self.intercept: float | None = None

    def _design(self, sample_features: Any, drug_fingerprints: Any) -> np.ndarray:
        sample, drug = _arrays(sample_features, drug_fingerprints)
        if self.sample_dim is not None and (sample.shape[1], drug.shape[1]) != (self.sample_dim, self.drug_dim):
            raise ValueError("feature dimensions differ from fitted model")
        width = min(sample.shape[1], drug.shape[1], self.interaction_features)
        return np.concatenate((sample, drug, sample[:, :width] * drug[:, :width]), axis=1)

    def fit(self, sample_features: Any, drug_fingerprints: Any, labels: Any, *, sample_weight: Any | None = None):
        sample, drug, target = _arrays(sample_features, drug_fingerprints, labels)
        self.sample_dim, self.drug_dim = sample.shape[1], drug.shape[1]
        self.interaction_dim = min(self.sample_dim, self.drug_dim, self.interaction_features)
        design = self._design(sample, drug)
        if sample_weight is None:
            weight = np.ones(len(target))
        else:
            weight = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
            if len(weight) != len(target) or not np.isfinite(weight).all() or (weight < 0).any() or not weight.any():
                raise ValueError("sample_weight must be aligned, finite, non-negative, and non-zero")
        root = np.sqrt(weight)[:, None]
        centered_x, centered_y = design - np.average(design, axis=0, weights=weight), target - np.average(target, weights=weight)
        penalty = np.eye(design.shape[1]) * self.alpha
        # ``lstsq`` also gives the unregularized (alpha=0) baseline a stable,
        # minimum-norm solution when a synthetic or small cohort is singular.
        self.coefficients = np.linalg.lstsq((centered_x * root).T @ (centered_x * root) + penalty, (centered_x * root).T @ (centered_y * root[:, 0]), rcond=None)[0]
        self.intercept = float(np.average(target, weights=weight) - np.average(design, axis=0, weights=weight) @ self.coefficients)
        return self

    def predict(self, sample_features: Any, drug_fingerprints: Any) -> np.ndarray:
        if self.coefficients is None or self.intercept is None: raise RuntimeError("model is not fitted")
        return self._design(sample_features, drug_fingerprints) @ self.coefficients + self.intercept

    def explain(self, sample_features: Any, drug_fingerprints: Any) -> dict[str, np.ndarray]:
        if self.coefficients is None or self.intercept is None: raise RuntimeError("model is not fitted")
        design = self._design(sample_features, drug_fingerprints)
        s, d, z = self.sample_dim, self.drug_dim, self.interaction_dim
        sample = design[:, :s] * self.coefficients[:s]
        drug = design[:, s:s+d] * self.coefficients[s:s+d]
        interaction = design[:, s+d:s+d+z] * self.coefficients[s+d:]
        return {"reference": np.full(len(design), self.intercept), "sample_contributions": sample, "drug_contributions": drug, "interaction_contributions": interaction, "prediction": self.intercept + sample.sum(1) + drug.sum(1) + interaction.sum(1)}

    def save(self, path: str | Path) -> None:
        with Path(path).open("wb") as handle: pickle.dump(self.__dict__, handle)

    @classmethod
    def load(cls, path: str | Path) -> "RidgePairModel":
        with Path(path).open("rb") as handle: state = pickle.load(handle)
        model = cls(alpha=state["alpha"], interaction_features=state["interaction_features"])
        model.__dict__.update(state)
        return model


class _DualNet(nn.Module):
    def __init__(self, sample_dim: int, drug_dim: int, hidden_dim: int):
        super().__init__()
        self.sample = nn.Sequential(nn.Linear(sample_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.drug = nn.Sequential(nn.Linear(drug_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, sample: torch.Tensor, drug: torch.Tensor) -> torch.Tensor:
        left, right = self.sample(sample), self.drug(drug)
        return self.head(torch.cat((left, right, left * right), dim=1)).squeeze(1)


class DualEncoderPairModel:
    """Small dual encoder with Huber regression plus within-PDO ranking loss.

    Calling ``fit`` on a loaded instance is M3 fine-tuning; a newly created
    instance is M1 random training. ``load`` provides M2 source initialization.
    Validation data is used only for early stopping and is never mixed into the
    label-standardization statistics.
    """

    def __init__(self, *, hidden_dim: int = 16, learning_rate: float = 1e-3,
                 ranking_weight: float = .2, huber_delta: float = 1.0,
                 seed: int = 0, epochs: int = 100, patience: int | None = None,
                 max_rank_pairs: int | None = 1024, device: str = "cpu",
                 batch_groups: int | None = None):
        if (isinstance(hidden_dim, bool) or isinstance(epochs, bool) or hidden_dim < 1 or epochs < 1
                or not np.isfinite(learning_rate) or learning_rate <= 0
                or not np.isfinite(ranking_weight) or ranking_weight < 0
                or not np.isfinite(huber_delta) or huber_delta <= 0
                or (patience is not None and (isinstance(patience, bool) or patience < 1))
                or (max_rank_pairs is not None and (isinstance(max_rank_pairs, bool) or max_rank_pairs < 1))
                or device not in {"cpu", "cuda"}
                or (batch_groups is not None and (isinstance(batch_groups, bool) or batch_groups < 1))):
            raise ValueError("invalid dual encoder configuration")
        self.config = {"hidden_dim": int(hidden_dim), "learning_rate": float(learning_rate), "ranking_weight": float(ranking_weight), "huber_delta": float(huber_delta), "seed": int(seed), "epochs": int(epochs), "patience": None if patience is None else int(patience), "max_rank_pairs": None if max_rank_pairs is None else int(max_rank_pairs)}
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("device='cuda' requires an available CUDA device")
        # Device is runtime state, deliberately outside ``config``.  Saved
        # checkpoints therefore remain portable to CPU-only hosts and older
        # checkpoints lacking these new runtime options still load unchanged.
        self.device = torch.device(device)
        self.batch_groups = None if batch_groups is None else int(batch_groups)
        self.network: _DualNet | None = None
        self.sample_dim: int | None = None; self.drug_dim: int | None = None
        self.label_mean: float | None = None; self.label_std: float | None = None

    @staticmethod
    def ranking_pairs(labels: Any, pdo_ids: Any, *, max_pairs_per_pdo: int | None = None,
                      seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        target = np.asarray(labels, dtype=float).reshape(-1); groups = np.asarray(pdo_ids).reshape(-1)
        if len(target) != len(groups): raise ValueError("pdo_ids must align labels")
        if max_pairs_per_pdo is not None and (isinstance(max_pairs_per_pdo, bool) or max_pairs_per_pdo < 1):
            raise ValueError("max_pairs_per_pdo must be positive or None")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise ValueError("seed must be an integer")
        left, right = [], []
        for group_number, group in enumerate(np.unique(groups)):
            indices = np.flatnonzero(groups == group)
            first, second = np.triu_indices(len(indices), k=1)
            a, b = indices[first], indices[second]
            a_higher, b_higher = target[a] > target[b], target[b] > target[a]
            pair_left = np.concatenate((a[a_higher], b[b_higher]))
            pair_right = np.concatenate((b[a_higher], a[b_higher]))
            # Preserve the former uncapped row-order contract while deriving
            # pairs from each unordered combination only once.
            order = np.lexsort((pair_right, pair_left))
            pair_left, pair_right = pair_left[order], pair_right[order]
            if max_pairs_per_pdo is not None and len(pair_left) > max_pairs_per_pdo:
                # Sampling starts from every valid unordered drug pair.  The
                # per-group seed makes a capped source cohort reproducible
                # without favouring the first rows of its input table.
                rng = np.random.default_rng(np.random.SeedSequence([int(seed), group_number]))
                selected = rng.choice(len(pair_left), size=max_pairs_per_pdo, replace=False)
                pair_left, pair_right = pair_left[selected], pair_right[selected]
            left.extend(pair_left); right.extend(pair_right)
        return np.asarray(left, dtype=int), np.asarray(right, dtype=int)

    def _ranking_loss(self, output: torch.Tensor, labels: np.ndarray | None = None,
                      groups: np.ndarray | None = None,
                      *, pair_groups: list[tuple[np.ndarray, np.ndarray]] | None = None) -> torch.Tensor:
        """Mean the ranking loss within each PDO before averaging PDOs."""
        if pair_groups is None:
            if labels is None or groups is None:
                raise ValueError("labels and groups are required when ranking pairs are not supplied")
            pair_groups = []
            for group in np.unique(groups):
                local = np.flatnonzero(groups == group)
                left, right = self.ranking_pairs(labels[local], groups[local], max_pairs_per_pdo=self.config["max_rank_pairs"])
                if len(left):
                    pair_groups.append((local[left], local[right]))
        group_losses = []
        for left, right in pair_groups:
            if len(left):
                left_index = torch.as_tensor(left, dtype=torch.long, device=output.device)
                right_index = torch.as_tensor(right, dtype=torch.long, device=output.device)
                group_losses.append(F.softplus(-(output[left_index] - output[right_index])).mean())
        return torch.stack(group_losses).mean() if group_losses else output.new_zeros(())

    def _group_rank_pairs(self, labels: np.ndarray, groups: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Build within-PDO rank pairs once per fit, in deterministic group order."""
        pairs = []
        for group_number, group in enumerate(np.unique(groups)):
            indices = np.flatnonzero(groups == group)
            left, right = self.ranking_pairs(
                labels[indices], groups[indices], max_pairs_per_pdo=self.config["max_rank_pairs"],
                seed=self.config["seed"] + group_number,
            )
            pairs.append((indices, left, right))
        return pairs

    def _ensure_network(self, sample_dim: int, drug_dim: int) -> None:
        if self.network is None:
            torch.manual_seed(self.config["seed"])
            self.sample_dim, self.drug_dim = sample_dim, drug_dim
            self.network = _DualNet(sample_dim, drug_dim, self.config["hidden_dim"]).to(self.device)
        elif (sample_dim, drug_dim) != (self.sample_dim, self.drug_dim): raise ValueError("feature dimensions differ from initialized model")

    def fit(self, sample_features: Any, drug_fingerprints: Any, labels: Any, *, pdo_ids: Any,
            validation_data: tuple[Any, Any, Any, Any] | None = None,
            reset_label_scaler: bool = True):
        sample, drug, target = _arrays(sample_features, drug_fingerprints, labels)
        groups = np.asarray(pdo_ids).reshape(-1)
        if len(groups) != len(target): raise ValueError("pdo_ids must align labels")
        self._ensure_network(sample.shape[1], drug.shape[1])
        if not isinstance(reset_label_scaler, bool):
            raise ValueError("reset_label_scaler must be boolean")
        if reset_label_scaler or self.label_mean is None:
            new_mean, new_std = float(target.mean()), float(target.std())
            if new_std == 0:
                new_std = 1.0
            # Preserve raw predictions while changing the normalized target
            # frame.  This makes a loaded source network a well-defined M2
            # initializer before M3 target-endpoint optimization begins.
            if self.label_mean is not None:
                factor = self.label_std / new_std
                shift = (self.label_mean - new_mean) / new_std
                final = self.network.head[-1]
                with torch.no_grad():
                    final.weight.mul_(factor)
                    final.bias.mul_(factor).add_(shift)
            self.label_mean, self.label_std = new_mean, new_std
        normalized = (target - self.label_mean) / self.label_std
        # Retain the complete feature matrices in host memory.  Mini-batches
        # move only complete PDO groups to the selected device.
        x, d, y = (torch.tensor(value, dtype=torch.float32) for value in (sample, drug, normalized))
        group_pairs = self._group_rank_pairs(target, groups)
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.config["learning_rate"])
        best_state, best_loss, stale = None, float("inf"), 0
        self.training_history = []
        for epoch in range(self.config["epochs"]):
            self.network.train()
            epoch_losses = []
            if self.batch_groups is None:
                batches = [(np.arange(len(target)), [(indices[left], indices[right]) for indices, left, right in group_pairs])]
            else:
                order = np.random.default_rng(self.config["seed"] + epoch).permutation(len(group_pairs))
                batches = []
                for start in range(0, len(order), self.batch_groups):
                    selected = [group_pairs[index] for index in order[start:start + self.batch_groups]]
                    indices = np.concatenate([entry[0] for entry in selected])
                    pair_groups, offset = [], 0
                    for group_indices, left, right in selected:
                        if len(left):
                            pair_groups.append((left + offset, right + offset))
                        offset += len(group_indices)
                    batches.append((indices, pair_groups))
            for indices, pair_groups in batches:
                index = torch.as_tensor(indices, dtype=torch.long)
                batch_x, batch_d, batch_y = (value[index].to(self.device) for value in (x, d, y))
                optimizer.zero_grad(); output = self.network(batch_x, batch_d)
                loss = F.huber_loss(output, batch_y, delta=self.config["huber_delta"])
                loss = loss + self.config["ranking_weight"] * self._ranking_loss(output, pair_groups=pair_groups)
                loss.backward(); optimizer.step()
                epoch_losses.append(float(loss.detach().cpu()))
            validation_loss = self._validation_loss(validation_data)
            self.training_history.append({'epoch': epoch + 1,
                'mean_batch_training_loss': float(np.mean(epoch_losses)),
                'validation_loss': validation_loss})
            if validation_loss is None: continue
            if validation_loss < best_loss:
                best_loss, stale = validation_loss, 0
                best_state = {key: value.detach().clone() for key, value in self.network.state_dict().items()}
            else:
                stale += 1
                if self.config["patience"] is not None and stale >= self.config["patience"]: break
        if best_state is not None: self.network.load_state_dict(best_state)
        return self

    def _validation_loss(self, data: tuple[Any, Any, Any, Any] | None) -> float | None:
        if data is None: return None
        if len(data) != 4:
            raise ValueError("validation_data must be (sample_features, drug_fingerprints, labels, pdo_ids)")
        sample, drug, target = _arrays(data[0], data[1], data[2])
        if (sample.shape[1], drug.shape[1]) != (self.sample_dim, self.drug_dim):
            raise ValueError("validation feature dimensions differ from fitted model")
        groups = np.asarray(data[3]).reshape(-1)
        if len(groups) != len(target): raise ValueError("validation pdo_ids must align labels")
        with torch.no_grad():
            pred = self.network(torch.tensor(sample, dtype=torch.float32, device=self.device), torch.tensor(drug, dtype=torch.float32, device=self.device))
            y = torch.tensor((target - self.label_mean) / self.label_std, dtype=torch.float32, device=self.device)
            loss = F.huber_loss(pred, y, delta=self.config["huber_delta"])
            loss = loss + self.config["ranking_weight"] * self._ranking_loss(pred, target, groups)
            return float(loss)

    def predict(self, sample_features: Any, drug_fingerprints: Any) -> np.ndarray:
        if self.network is None: raise RuntimeError("model is not fitted")
        sample, drug = _arrays(sample_features, drug_fingerprints)
        if (sample.shape[1], drug.shape[1]) != (self.sample_dim, self.drug_dim):
            raise ValueError("feature dimensions differ from fitted model")
        self.network.eval()
        with torch.no_grad(): return (self.network(torch.tensor(sample, dtype=torch.float32, device=self.device), torch.tensor(drug, dtype=torch.float32, device=self.device)).cpu().numpy() * self.label_std + self.label_mean)

    def explain(self, sample_features: Any, drug_fingerprints: Any, *, steps: int = 64) -> dict[str, np.ndarray]:
        """Integrated-gradient attributions; attribution is not mechanism evidence."""
        if self.network is None or isinstance(steps, bool) or steps < 2: raise ValueError("fitted model and integer steps >=2 required")
        sample, drug = _arrays(sample_features, drug_fingerprints)
        if (sample.shape[1], drug.shape[1]) != (self.sample_dim, self.drug_dim):
            raise ValueError("feature dimensions differ from fitted model")
        self.network.eval()
        sample_attr, drug_attr, reference, prediction = [], [], [], []
        # The zero-feature baseline makes the decomposition auditable: the
        # reference plus both feature blocks approximates the prediction.
        # Trapezoidal integration includes both endpoints and is materially
        # more accurate than a left- or right-endpoint Riemann sum.
        alphas = torch.linspace(0, 1, steps, device=self.device)
        integration_weights = torch.ones(steps, dtype=torch.float32, device=self.device)
        integration_weights[[0, -1]] = 0.5
        for s, d in zip(sample, drug):
            s_tensor, d_tensor = torch.tensor(s, dtype=torch.float32, device=self.device), torch.tensor(d, dtype=torch.float32, device=self.device)
            grad_s, grad_d = torch.zeros_like(s_tensor), torch.zeros_like(d_tensor)
            for alpha, weight in zip(alphas, integration_weights):
                si, di = (alpha * s_tensor).requires_grad_(True), (alpha * d_tensor).requires_grad_(True)
                self.network.zero_grad(); self.network(si[None], di[None]).backward()
                grad_s += weight * si.grad; grad_d += weight * di.grad
            denominator = steps - 1
            sample_attr.append((s_tensor * grad_s / denominator * self.label_std).detach().cpu().numpy())
            drug_attr.append((d_tensor * grad_d / denominator * self.label_std).detach().cpu().numpy())
            with torch.no_grad():
                reference.append(float((self.network(torch.zeros_like(s_tensor)[None], torch.zeros_like(d_tensor)[None]) * self.label_std + self.label_mean).item()))
                prediction.append(float((self.network(s_tensor[None], d_tensor[None]) * self.label_std + self.label_mean).item()))
        return {"sample_attributions": np.asarray(sample_attr), "drug_attributions": np.asarray(drug_attr), "reference": np.asarray(reference), "prediction": np.asarray(prediction)}

    def save(self, path: str | Path) -> None:
        if self.network is None: raise RuntimeError("model is not fitted")
        torch.save({"config": self.config, "sample_dim": self.sample_dim, "drug_dim": self.drug_dim, "label_mean": self.label_mean, "label_std": self.label_std, "state_dict": self.network.state_dict()}, Path(path))

    @classmethod
    def load(cls, path: str | Path) -> "DualEncoderPairModel":
        state = torch.load(Path(path), map_location="cpu", weights_only=False)
        model = cls(**state["config"]); model._ensure_network(state["sample_dim"], state["drug_dim"])
        model.label_mean, model.label_std = state["label_mean"], state["label_std"]
        model.network.load_state_dict(state["state_dict"])
        return model
