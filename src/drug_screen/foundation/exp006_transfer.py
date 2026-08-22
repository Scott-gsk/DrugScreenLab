"""Large-scale Genetic→Chemical transfer helpers for EXP-006.

Design chain
------------
data structure
    paired exact978 treatments + matched controls, UniPert 256-d genetic
    vectors, official XPert UniMol/HG chemical tokens
scientific question
    does large-scale genetic supervision reduce the chemical supervision
    required for the same XPert Δ978 backbone?
inductive bias
    genetic and chemical perturbagens occupy the same treatment-token
    slot that XPert already attends to; no extra sequence, no KPGT
    fusion, no new transformer
model
    A: official UniMol/HG → shared XPert Context / treatment / Δ978
    B: UniPert → one small projection (+ optional direction embedding)
       replacing the drug token; then chemical fine-tune of the same
       backbone
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


SEED = 20260813
UNIPERT_DIM = 256
HIDDEN_SIZE = 256
GENE_COUNT = 978
GENETIC_TYPES = ("trt_sh", "trt_sh.cgs", "trt_sh.css", "trt_oe", "trt_oe.mut")
KNOCKDOWN_TYPES = {"trt_sh", "trt_sh.cgs", "trt_sh.css"}
OVEREXPRESSION_TYPES = {"trt_oe"}
MUTANT_TYPES = {"trt_oe.mut"}
DIRECTIONS = ("knockdown", "overexpression", "mutant")
FRACTIONS = (1.0, 0.2, 0.1)

try:
    import torch
    from torch import Tensor
    from torch import nn
except ImportError:  # pragma: no cover - GPU environment is the execution contract
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc, assignment]
    nn = None  # type: ignore[assignment]


def file_sha256(path: Path | str) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value: object) -> str:
    import json

    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()


def _eligible_genetic(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"cell_id", "gene_symbol", "has_matched_control", "unipert_mappable"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"genetic coverage frame missing columns: {missing}")
    selected = frame.loc[
        frame["has_matched_control"].astype(bool)
        & frame["unipert_mappable"].astype(bool)
        & frame["gene_symbol"].astype(str).ne("")
    ].copy()
    selected["context_id"] = selected["cell_id"].astype(str)
    selected["gene_symbol"] = selected["gene_symbol"].astype(str).str.strip().str.upper()
    return selected


def _eligible_chemical(frame: pd.DataFrame) -> pd.DataFrame:
    context_col = "cell_iname" if "cell_iname" in frame.columns else "context_id"
    if context_col not in frame.columns or "pert_id" not in frame.columns:
        raise ValueError("chemical coverage frame requires cell_iname/context_id and pert_id")
    selected = frame.copy()
    selected["context_id"] = selected[context_col].astype(str)
    selected["pert_id"] = selected["pert_id"].astype(str)
    return selected


def summarize_context_coverage(
    genetic_frame: pd.DataFrame,
    chemical_frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Response-blind unique-gene / unique-compound counts per context."""

    genetic = _eligible_genetic(genetic_frame)
    chemical = _eligible_chemical(chemical_frame)
    contexts = sorted(set(genetic["context_id"]).union(set(chemical["context_id"])))
    rows: list[dict[str, Any]] = []
    for context_id in contexts:
        g = genetic.loc[genetic["context_id"].eq(context_id)]
        c = chemical.loc[chemical["context_id"].eq(context_id)]
        rows.append(
            {
                "context_id": context_id,
                "unique_genes": int(g["gene_symbol"].nunique()),
                "genetic_records": int(len(g)),
                "unique_genetic_perturbagens": int(g["pert_id"].nunique()) if "pert_id" in g.columns else int(g["gene_symbol"].nunique()),
                "unique_compounds": int(c["pert_id"].nunique()),
                "chemical_records": int(len(c)),
            }
        )
    return rows


def select_dual_coverage_contexts(
    genetic_frame: pd.DataFrame,
    chemical_frame: pd.DataFrame,
    *,
    min_unique_genes: int = 200,
    min_unique_compounds: int = 200,
    max_contexts: int = 5,
    min_contexts: int = 3,
    target_unique_genes: int = 2000,
) -> dict[str, Any]:
    """Select 3–5 dual-high-coverage contexts without reading responses."""

    if max_contexts < 1:
        raise ValueError("max_contexts must be positive")
    coverage = summarize_context_coverage(genetic_frame, chemical_frame)
    ranked = sorted(
        coverage,
        key=lambda row: (
            -int(row["unique_genes"]),
            -int(row["unique_compounds"]),
            str(row["context_id"]),
        ),
    )
    eligible = [
        row
        for row in ranked
        if int(row["unique_genes"]) >= min_unique_genes
        and int(row["unique_compounds"]) >= min_unique_compounds
    ]
    if len(eligible) >= min_contexts:
        selected = eligible[:max_contexts]
        eligibility_note = "enough dual-high-coverage contexts after response-blind gates"
    else:
        selected = ranked[: min(max_contexts, max(len(ranked), 0))]
        eligibility_note = (
            "fewer than 3 contexts met dual-coverage gates; using all available "
            "ranked contexts without tightening thresholds"
        )
    for row in selected:
        row["meets_target_unique_genes"] = int(row["unique_genes"]) >= target_unique_genes
        row["meets_min_unique_genes"] = int(row["unique_genes"]) >= min_unique_genes
        row["meets_min_unique_compounds"] = int(row["unique_compounds"]) >= min_unique_compounds
    return {
        "format": "exp006_context_coverage_v1",
        "selected_contexts": [row["context_id"] for row in selected],
        "per_context": selected,
        "all_contexts_ranked": ranked,
        "eligible_context_count": int(len(eligible)),
        "eligibility_note": eligibility_note,
        "downsample": {"applied": False, "reason": None},
        "selection_policy": {
            "response_values_used": False,
            "test_performance_used": False,
            "prism_used": False,
            "criteria": [
                "unique_mappable_genetic_genes_with_matched_control",
                "unique_chemical_compounds_with_official_xpert_identity",
                "exact978_available",
            ],
            "min_unique_genes": min_unique_genes,
            "min_unique_compounds": min_unique_compounds,
            "target_unique_genes": target_unique_genes,
            "max_contexts": max_contexts,
        },
    }


def assign_compound_level_splits(
    compounds: Sequence[str],
    *,
    seed: int = SEED,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
) -> dict[str, list[str]]:
    """Deterministic compound-level split; no row-level leakage."""

    if not 0 < train_fraction < 1 or not 0 <= validation_fraction < 1:
        raise ValueError("split fractions must leave a positive test remainder")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("split fractions must leave a positive test remainder")
    unique = sorted({str(value) for value in compounds if str(value)})
    if not unique:
        raise ValueError("compound list is empty")
    buckets = {"train": [], "validation": [], "test": []}
    for compound in unique:
        digest = sha256(f"{seed}:compound:{compound}".encode("utf-8")).digest()
        unit = int.from_bytes(digest[:8], "big") / float(2**64)
        if unit < train_fraction:
            buckets["train"].append(compound)
        elif unit < train_fraction + validation_fraction:
            buckets["validation"].append(compound)
        else:
            buckets["test"].append(compound)
    if any(not values for values in buckets.values()):
        raise ValueError("compound-level split left an empty role")
    return buckets


def sample_unique_compounds(
    compounds: Sequence[str],
    *,
    fraction: float,
    seed: int = SEED,
) -> list[str]:
    """Response-blind unique-compound subsample of a frozen training pool."""

    if fraction <= 0 or fraction > 1:
        raise ValueError("fraction must be in (0, 1]")
    unique = sorted({str(value) for value in compounds if str(value)})
    if not unique:
        raise ValueError("compound list is empty")
    if fraction >= 1:
        return unique
    ranked = sorted(
        unique,
        key=lambda compound: sha256(f"{seed}:frac:{fraction}:{compound}".encode("utf-8")).hexdigest(),
    )
    count = max(1, int(round(len(unique) * fraction)))
    return sorted(ranked[:count])


def direction_from_pert_type(pert_type: str) -> str:
    value = str(pert_type)
    if value in KNOCKDOWN_TYPES:
        return "knockdown"
    if value in OVEREXPRESSION_TYPES:
        return "overexpression"
    if value in MUTANT_TYPES:
        return "mutant"
    raise ValueError(f"unsupported genetic pert_type: {pert_type}")


def _corr(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.size == 0 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    return ranks


def delta978_metrics(true_delta: np.ndarray, pred_delta: np.ndarray) -> dict[str, Any]:
    """Frozen perturbation metrics: Pearson / Spearman / MSE / direction."""

    true_delta = np.asarray(true_delta, dtype=float)
    pred_delta = np.asarray(pred_delta, dtype=float)
    if true_delta.shape != pred_delta.shape or true_delta.ndim != 2:
        raise ValueError("true and predicted Δ978 must share a [n, 978] shape")
    row_pearson = [_corr(left, right) for left, right in zip(true_delta, pred_delta, strict=True)]
    row_spearman = [
        _corr(_rank(left), _rank(right)) for left, right in zip(true_delta, pred_delta, strict=True)
    ]
    direction = float(np.mean(np.sign(true_delta) == np.sign(pred_delta)))
    return {
        "rows": int(true_delta.shape[0]),
        "genes": int(true_delta.shape[1]),
        "mse": float(np.mean((true_delta - pred_delta) ** 2)),
        "pearson_row_mean": float(np.nanmean(row_pearson)) if row_pearson else None,
        "spearman_row_mean": float(np.nanmean(row_spearman)) if row_spearman else None,
        "direction_consistency": direction,
        "prediction_std": float(np.std(pred_delta)),
    }


if nn is not None:

    class GeneticPerturbagenAdapter(nn.Module):
        """One projection from UniPert 256-d into the XPert treatment slot."""

        def __init__(
            self,
            *,
            unipert_dim: int = UNIPERT_DIM,
            hidden_size: int = HIDDEN_SIZE,
            dropout: float = 0.1,
            n_directions: int = len(DIRECTIONS),
        ) -> None:
            super().__init__()
            if unipert_dim < 1 or hidden_size < 1:
                raise ValueError("adapter dimensions must be positive")
            self.projection = nn.Sequential(
                nn.Linear(unipert_dim, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.direction_embedding = nn.Embedding(n_directions, hidden_size)

        def forward(self, unipert: Tensor, direction: Tensor | None = None) -> Tensor:
            token = self.projection(unipert)
            if direction is not None:
                token = token + self.direction_embedding(direction)
            return token

else:  # pragma: no cover - import fallback

    class GeneticPerturbagenAdapter:  # type: ignore[no-redef]
        def __init__(self, **_: Any) -> None:
            raise RuntimeError("PyTorch is required for GeneticPerturbagenAdapter")


def build_xpert_genetic_transfer_model(
    official_model: type[Any],
    *,
    unipert_dim: int = UNIPERT_DIM,
    hidden_size: int | None = None,
    dropout: float = 0.1,
) -> type[Any]:
    """Return an XPertNet subclass that can consume a genetic treatment token.

    Chemical mode (10 official fields): unchanged UniMol/HG path.
    Genetic mode (official fields + UniPert + direction): the official drug
    embedding is replaced by a single projected UniPert token so the shared
    Context / treatment-interaction / Δ978 heads stay intact. Sequence
    length is 1, not lengthened. No KPGT, no extra tokens.
    """

    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required for XPert genetic transfer")

    class XPertGeneticTransferNet(official_model):
        extension_name = "genetic_unipert_minimal_adapter"

        def __init__(self, args: Any, config: dict[str, Any], device: Any, logger: Any) -> None:
            super().__init__(args, config, device, logger)
            size = int(hidden_size or config["model"]["ATTN"]["hidden_size"])
            self.genetic_adapter = GeneticPerturbagenAdapter(
                unipert_dim=unipert_dim,
                hidden_size=size,
                dropout=dropout,
            )
            self.official_checkpoint_loaded = False
            self.checkpoint_audit: dict[str, Any] | None = None
            self._active_genetic: tuple[Tensor, Tensor] | None = None
            self._drug_embedding_hook_handle = self.drug_emb.register_forward_hook(
                self._replace_drug_token
            )
            if hasattr(self, "attnEncoder_trt"):
                self._attn_hook_handle = self.attnEncoder_trt.register_forward_pre_hook(
                    self._drop_chemical_atom_mask
                )
            else:
                self._attn_hook_handle = None

        def _replace_drug_token(self, _module: Any, _inputs: Any, output: Tensor) -> Tensor:
            if self._active_genetic is None:
                return output
            unipert, direction = self._active_genetic
            token = self.genetic_adapter(unipert, direction).unsqueeze(1)
            return token

        def _drop_chemical_atom_mask(self, _module: Any, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
            if self._active_genetic is None or len(inputs) < 4:
                return inputs
            cell_embed, drug_embed, cell_mask, _drug_mask, *rest = inputs
            return (cell_embed, drug_embed, cell_mask, None, *rest)

        def forward(self, data: Any, mode: str = "ST") -> Any:
            if len(data) == 10:
                self._active_genetic = None
                return super().forward(data, mode=mode)
            if len(data) == 12:
                unipert = data[10]
                direction = data[11]
                if unipert is None:
                    raise ValueError("genetic mode requires a UniPert tensor")
                self._active_genetic = (
                    unipert.to(self.device),
                    direction.to(self.device) if direction is not None else None,
                )
                try:
                    return super().forward(tuple(data[:10]), mode=mode)
                finally:
                    self._active_genetic = None
            if len(data) == 2:
                # Unit-test stub: (drug_embed, cell_embed[, genetic, direction])
                return super().forward(data, mode=mode)
            if len(data) == 4:
                unipert = data[2]
                direction = data[3]
                self._active_genetic = (unipert, direction)
                try:
                    return super().forward(tuple(data[:2]), mode=mode)
                finally:
                    self._active_genetic = None
            raise ValueError(
                "XPert genetic transfer expects 10 chemical fields or 12 fields "
                f"with UniPert+direction; received {len(data)}"
            )

    XPertGeneticTransferNet.__name__ = "XPertGeneticTransferNet"
    return XPertGeneticTransferNet


def split_digest(mapping: Mapping[str, Sequence[str]]) -> dict[str, Any]:
    return {
        role: {
            "count": int(len(values)),
            "sha256": sha256(("\n".join(sorted(map(str, values))) + "\n").encode("utf-8")).hexdigest(),
        }
        for role, values in mapping.items()
    }


def write_json(path: Path | str, payload: Mapping[str, Any]) -> Path:
    import json

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output
