"""Frozen PharmaFormer boundary and checkpoint contract for EXP-011.

This module deliberately keeps the upstream ``CombinedModel`` intact for M2.
The only structural addition is :class:`PharmaFormerAdapterModel`, which
inserts a zero-initialised residual adapter after ``gene_fc2``.  Training and
data materialisation live outside this boundary so an upstream checkout stays
an auditable immutable dependency.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from drug_screen.foundation.pharmaformer_adapter import (
    PharmaFormerConfig,
    PharmaFormerIntegrationError,
    TrainOnlyGeneTransform,
    TrainOnlySmilesEncoder,
    UPSTREAM_COMMIT,
    build_model,
    load_upstream_module,
)


EXP011_CHECKPOINT_FORMAT = "drugscreenlab_exp011_pharmaformer_v1"
EXP011_EXPERIMENT_ID = "EXP-011"
OFFICIAL_GENE_INPUT_SIZE = 19_091


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA256 digest without loading a large asset at once."""
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Exp011ModelConfig:
    """The fixed official M2 geometry plus explicit EXP-011 adapter settings."""

    gene_input_size: int = OFFICIAL_GENE_INPUT_SIZE
    gene_hidden_size: int = 8192
    drug_hidden_size: int = 256
    feature_dim: int = 128
    nhead: int = 8
    num_layers: int = 3
    dim_feedforward: int = 2048
    dropout: float = 0.1
    adapter_bottleneck: int = 256
    adapter_dropout: float = 0.3
    test_only_geometry: bool = False

    def validate(self) -> None:
        if self.gene_input_size < 1:
            raise ValueError("gene_input_size must be positive")
        if not self.test_only_geometry and (self.gene_hidden_size != 8192 or self.drug_hidden_size != 256):
            raise ValueError("EXP-011 locks official PharmaFormer 8192/256 hidden dimensions")
        if not self.test_only_geometry and (self.feature_dim != 128 or self.nhead != 8 or self.num_layers != 3):
            raise ValueError("EXP-011 locks official PharmaFormer 128-dim, 8-head, 3-layer Transformer")
        if not self.test_only_geometry and self.dim_feedforward != 2048:
            raise ValueError("EXP-011 locks the official 2048-dim feed-forward layer")
        if not self.test_only_geometry and (self.adapter_bottleneck != 256 or self.adapter_dropout != 0.3):
            raise ValueError("EXP-011 locks the M3 adapter to 8192-256 and dropout 0.3")

    def upstream_config(self) -> PharmaFormerConfig:
        self.validate()
        return PharmaFormerConfig(
            gene_input_size=self.gene_input_size,
            gene_hidden_size=self.gene_hidden_size,
            drug_hidden_size=self.drug_hidden_size,
            feature_dim=self.feature_dim,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
        )


def require_official_gene_contract(gene_ids: Sequence[str], config: Exp011ModelConfig) -> str:
    """Validate the ordered full HGNC input contract and return its digest."""
    config.validate()
    values = tuple(str(x) for x in gene_ids)
    if config.test_only_geometry or len(values) != OFFICIAL_GENE_INPUT_SIZE or config.gene_input_size != OFFICIAL_GENE_INPUT_SIZE:
        raise ValueError(
            "formal EXP-011 runs require exactly 19,091 ordered common HGNC genes; "
            "synthetic tests must opt out explicitly"
        )
    if len(set(values)) != len(values) or any(not value for value in values):
        raise ValueError("ordered genes must be non-empty and unique")
    return sha256("\n".join(values).encode("utf-8")).hexdigest()


def build_official_m2(upstream: Any, config: Exp011ModelConfig) -> Any:
    """Build unchanged upstream ``CombinedModel`` for M2 or M4."""
    return build_model(upstream, config.upstream_config())


def verify_upstream_adapter_surface(backbone: Any, config: Exp011ModelConfig) -> None:
    """Fail closed if the pinned upstream module no longer has the audited API."""
    extractor = getattr(backbone, "feature_extractor", None)
    required = ("gene_fc1", "gene_fc2", "smiles_fc")
    if extractor is None or any(not hasattr(extractor, name) for name in required):
        raise PharmaFormerIntegrationError("upstream CombinedModel feature extractor API differs from audited commit")
    if not hasattr(backbone, "transformer") or not hasattr(backbone, "seq_len") or not hasattr(backbone, "feature_dim"):
        raise PharmaFormerIntegrationError("upstream CombinedModel transformer API differs from audited commit")
    if int(backbone.seq_len) != (config.gene_hidden_size + config.drug_hidden_size) // config.feature_dim:
        raise PharmaFormerIntegrationError("upstream CombinedModel sequence shape differs from frozen contract")


class GeneResidualAdapter:  # deferred torch import keeps data-only imports cheap
    """Factory namespace for the actual torch module.

    ``torch`` is intentionally imported only when a model is built; the data
    and manifest validation code remains importable in CPU-only environments.
    """

    @staticmethod
    def build(hidden_size: int = 8192, bottleneck: int = 256, dropout: float = 0.3) -> Any:
        import torch
        from torch import nn

        class _GeneResidualAdapter(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm = nn.LayerNorm(hidden_size)
                self.down = nn.Linear(hidden_size, bottleneck)
                self.activation = nn.GELU()
                self.dropout = nn.Dropout(dropout)
                self.up = nn.Linear(bottleneck, hidden_size)
                # The M3 initial forward pass is exactly M2.  Both bias and
                # weights are reset because Linear's default bias is nonzero.
                nn.init.zeros_(self.up.weight)
                nn.init.zeros_(self.up.bias)

            def forward(self, value: Any) -> Any:
                return value + self.up(self.dropout(self.activation(self.down(self.norm(value)))))

        return _GeneResidualAdapter()


def build_m3_adapter(backbone: Any, config: Exp011ModelConfig) -> Any:
    """Wrap an M2 backbone with the declared post-``gene_fc2`` residual adapter."""
    import torch
    import torch.nn.functional as functional
    from torch import nn

    verify_upstream_adapter_surface(backbone, config)

    class PharmaFormerAdapterModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = backbone
            self.adapter = GeneResidualAdapter.build(
                hidden_size=config.gene_hidden_size,
                bottleneck=config.adapter_bottleneck,
                dropout=config.adapter_dropout,
            )

        def forward(self, gene_expr: torch.Tensor, smiles: torch.Tensor) -> torch.Tensor:
            extractor = self.backbone.feature_extractor
            gene_out = functional.relu(extractor.gene_fc1(gene_expr))
            gene_out = functional.relu(extractor.gene_fc2(gene_out))
            gene_out = self.adapter(gene_out)
            smiles_out = functional.relu(extractor.smiles_fc(smiles))
            features = torch.cat((gene_out, smiles_out), dim=1)
            features = features.view(features.size(0), self.backbone.seq_len, self.backbone.feature_dim)
            return self.backbone.transformer(features)

    return PharmaFormerAdapterModel()


def model_variant(base_model: Any) -> str:
    """Return a compact auditable architecture identifier for a model instance."""
    return "M3-adapter" if hasattr(base_model, "adapter") and hasattr(base_model, "backbone") else "CombinedModel"


def save_exp011_checkpoint(
    path: str | Path,
    model: Any,
    config: Exp011ModelConfig,
    gene_transform: TrainOnlyGeneTransform,
    smiles_encoder: TrainOnlySmilesEncoder,
    *,
    variant: str,
    fold_id: int,
    gene_contract_sha256: str,
    provenance: Mapping[str, Any],
    source_checkpoint_sha256: str | None = None,
    response_transform_state: Mapping[str, Any] | None = None,
) -> Path:
    """Save all transforms, model weights and immutable-source provenance together."""
    import torch

    if variant not in {"M2", "M3-adapter", "M3-full", "M4"}:
        raise ValueError("unsupported EXP-011 model variant")
    if not gene_contract_sha256 or len(gene_contract_sha256) != 64:
        raise ValueError("gene_contract_sha256 is required")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": EXP011_CHECKPOINT_FORMAT,
        "experiment_id": EXP011_EXPERIMENT_ID,
        "variant": variant,
        "fold_id": int(fold_id),
        "upstream_commit": UPSTREAM_COMMIT,
        "model_variant": model_variant(model),
        "model_config": asdict(config),
        "model_state_dict": model.state_dict(),
        "gene_transform": gene_transform.state_dict(),
        "smiles_encoder": smiles_encoder.state_dict(),
        "gene_contract_sha256": gene_contract_sha256,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "response_transform": None if response_transform_state is None else dict(response_transform_state),
        "provenance": dict(provenance),
    }
    torch.save(payload, target)
    return target


def load_exp011_checkpoint(
    path: str | Path,
    upstream: Any,
    *,
    device: str = "cpu",
) -> tuple[Any, dict[str, Any]]:
    """Strictly restore an M2/M3/M4 checkpoint, including the frozen BPE asset."""
    import torch

    try:
        payload = torch.load(Path(path), map_location=device, weights_only=False)
    except TypeError:  # torch < 2.0 compatibility in the approved environment
        payload = torch.load(Path(path), map_location=device)
    if not isinstance(payload, dict) or payload.get("format") != EXP011_CHECKPOINT_FORMAT:
        raise PharmaFormerIntegrationError("not an EXP-011 PharmaFormer checkpoint")
    if payload.get("experiment_id") != EXP011_EXPERIMENT_ID or payload.get("upstream_commit") != UPSTREAM_COMMIT:
        raise PharmaFormerIntegrationError("checkpoint experiment or upstream commit differs from frozen contract")
    config = Exp011ModelConfig(**payload["model_config"])
    config.validate()
    if payload.get("model_variant") == "M3-adapter":
        model = build_m3_adapter(build_official_m2(upstream, config), config)
    elif payload.get("model_variant") == "CombinedModel":
        model = build_official_m2(upstream, config)
    else:
        raise PharmaFormerIntegrationError("unsupported checkpoint model variant")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    payload["gene_transform_object"] = TrainOnlyGeneTransform.from_state_dict(payload["gene_transform"])
    payload["smiles_encoder_object"] = TrainOnlySmilesEncoder.from_state_dict(payload["smiles_encoder"])
    if payload["smiles_encoder_object"].bpe_sha256 != payload["smiles_encoder"].get("bpe_sha256"):
        raise PharmaFormerIntegrationError("checkpoint BPE checksum mismatch")
    return model, payload


def upstream_checkout(upstream_root: str | Path) -> Any:
    """Load the pinned official checkout through the existing commit verifier."""
    return load_upstream_module(upstream_root)


def architecture_digest(config: Exp011ModelConfig, gene_contract_sha256: str) -> str:
    """Digest configuration plus the exact ordered-gene identity, not only its size."""
    document = {
        "config": asdict(config),
        "gene_contract_sha256": gene_contract_sha256,
        "upstream_commit": UPSTREAM_COMMIT,
    }
    return sha256(json.dumps(document, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
