"""Additive perturbagen extensions around the official XPert implementation.

The module deliberately does not reimplement XPert's transformer.  It provides
two small overlays:

* ``PerturbagenEncoder`` projects KPGT and UniPert independently to the
  official hidden size and emits one token per representation;
* ``build_xpert_additive_model`` creates a subclass of the official XPertNet
  and appends those tokens through a hook on the official ``drug_emb`` output.

The hook keeps the official HG + UniMol path and all downstream attention,
loss, and prediction heads unchanged.  The external XPert source is supplied
at runtime, so this project module remains import-safe without that checkout.
"""

from __future__ import annotations

from typing import Any, Callable, Type


try:  # Keep registry/unit-test imports usable in lightweight environments.
    import torch
    from torch import Tensor
    from torch import nn
except ImportError:  # pragma: no cover - GPU environment is the execution contract
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc,assignment]
    nn = None  # type: ignore[assignment]


if nn is not None:

    class PerturbagenEncoder(nn.Module):
        """Project each additive chemical representation to one hidden token."""

        def __init__(
            self,
            *,
            hidden_size: int,
            kpgt_dim: int = 2304,
            unipert_dim: int = 256,
            use_kpgt: bool = False,
            use_unipert: bool = False,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.use_kpgt = bool(use_kpgt)
            self.use_unipert = bool(use_unipert)
            if not self.use_kpgt and not self.use_unipert:
                raise ValueError("at least one additive perturbagen token is required")
            self.kpgt = (
                nn.Sequential(
                    nn.Linear(kpgt_dim, hidden_size),
                    nn.LayerNorm(hidden_size),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                if self.use_kpgt
                else None
            )
            self.unipert = (
                nn.Sequential(
                    nn.Linear(unipert_dim, hidden_size),
                    nn.LayerNorm(hidden_size),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                if self.use_unipert
                else None
            )

        def forward(self, *, kpgt: Tensor | None = None, unipert: Tensor | None = None) -> Tensor:
            tokens: list[Tensor] = []
            if self.use_kpgt:
                if kpgt is None:
                    raise ValueError("KPGT token is enabled but no KPGT features were supplied")
                assert self.kpgt is not None
                tokens.append(self.kpgt(kpgt).unsqueeze(1))
            if self.use_unipert:
                if unipert is None:
                    raise ValueError("UniPert token is enabled but no UniPert features were supplied")
                assert self.unipert is not None
                tokens.append(self.unipert(unipert).unsqueeze(1))
            return torch.cat(tokens, dim=1)

else:

    class PerturbagenEncoder:  # pragma: no cover - only import fallback
        def __init__(self, **_: Any) -> None:
            raise RuntimeError("PyTorch is required for PerturbagenEncoder")


def append_perturbagen_tokens(base_drug_embed: Tensor, extra_tokens: Tensor) -> Tensor:
    """Append separate additive tokens without changing the official sequence."""
    if base_drug_embed.ndim != 3 or extra_tokens.ndim != 3:
        raise ValueError("drug embeddings and additive tokens must be rank-3 tensors")
    if base_drug_embed.shape[0] != extra_tokens.shape[0]:
        raise ValueError("drug embedding and token batch dimensions must match")
    if base_drug_embed.shape[2] != extra_tokens.shape[2]:
        raise ValueError("drug embedding and token hidden dimensions must match")
    return torch.cat([base_drug_embed, extra_tokens], dim=1)


def build_xpert_additive_model(
    official_model: Type[Any],
    get_unimol_drug_feat: Callable[..., Any] | None = None,
    *,
    use_kpgt: bool,
    use_unipert: bool,
    kpgt_dim: int = 2304,
    unipert_dim: int = 256,
) -> Type[Any]:
    """Return an XPertNet subclass that adds frozen-feature tokens.

    ``official_model`` must be the imported upstream ``XPertNet`` class.  The
    returned class consumes the ordinary ten-item XPert tuple plus KPGT and/or
    UniPert tensors appended at the end.  A forward hook intercepts only the
    output of the official drug embedding module; all transformer layers and
    heads continue to execute in upstream code.
    """
    if torch is None:  # pragma: no cover - runtime dependency
        raise RuntimeError("PyTorch is required for XPert additive models")
    if not use_kpgt and not use_unipert:
        raise ValueError("at least one additive feature must be enabled")

    class XPertAdditiveNet(official_model):
        extension_name = "+KPGT+UniPert" if use_unipert else "+KPGT"

        def __init__(self, args: Any, config: dict[str, Any], device: Any, logger: Any) -> None:
            super().__init__(args, config, device, logger)
            hidden_size = int(config["model"]["ATTN"]["hidden_size"])
            self.perturbagen_encoder = PerturbagenEncoder(
                hidden_size=hidden_size,
                kpgt_dim=kpgt_dim,
                unipert_dim=unipert_dim,
                use_kpgt=use_kpgt,
                use_unipert=use_unipert,
            )
            self._active_additive_features: tuple[Tensor | None, Tensor | None] | None = None
            self._drug_embedding_hook_handle = self.drug_emb.register_forward_hook(self._drug_embedding_hook)

        def _drug_embedding_hook(self, _module: Any, _inputs: Any, output: Tensor) -> Tensor:
            if self._active_additive_features is None:
                return output
            kpgt, unipert = self._active_additive_features
            tokens = self.perturbagen_encoder(kpgt=kpgt, unipert=unipert)
            return append_perturbagen_tokens(output, tokens)

        def forward(self, data: Any, mode: str = "ST") -> Any:
            if len(data) < 10:
                raise ValueError("XPert additive model requires the ten official data fields")
            base_data = tuple(data[:10])
            extra = list(data[10:])
            expected = int(use_kpgt) + int(use_unipert)
            if len(extra) != expected:
                raise ValueError(f"expected {expected} additive feature tensors, received {len(extra)}")
            kpgt = extra.pop(0) if use_kpgt else None
            unipert = extra.pop(0) if use_unipert else None
            self._active_additive_features = (
                kpgt.to(self.device) if kpgt is not None else None,
                unipert.to(self.device) if unipert is not None else None,
            )
            try:
                # The official forward handles all sequence construction,
                # attention, heads, and return contracts.
                return super().forward(base_data, mode=mode)
            finally:
                self._active_additive_features = None

    XPertAdditiveNet.__name__ = "XPertAdditiveNet"
    return XPertAdditiveNet


def build_xpert_extension_dataset(base_dataset: Type[Any]) -> Type[Any]:
    """Wrap the official MyDataset and append per-row frozen feature tensors."""
    if torch is None:  # pragma: no cover - runtime dependency
        raise RuntimeError("PyTorch is required for XPert extension datasets")

    class XPertExtensionDataset(base_dataset):
        def __init__(
            self,
            raw_data: Any,
            drug_feat: Any,
            *,
            kpgt_features: dict[Any, Any] | None = None,
            unipert_features: Any | None = None,
            use_kpgt: bool = False,
            use_unipert: bool = False,
            **kwargs: Any,
        ) -> None:
            self._extension_kpgt = kpgt_features
            self._extension_unipert = unipert_features
            self._extension_use_kpgt = bool(use_kpgt)
            self._extension_use_unipert = bool(use_unipert)
            if self._extension_use_kpgt and self._extension_kpgt is None:
                raise ValueError("KPGT extension enabled without feature map")
            if self._extension_use_unipert and self._extension_unipert is None:
                raise ValueError("UniPert extension enabled without feature array")
            super().__init__(raw_data, drug_feat, **kwargs)

        def load_data(self) -> list[Any]:
            base_rows = super().load_data()
            rows: list[Any] = []
            for index, row in enumerate(base_rows):
                pert_idx = int(self.raw_data_items.iloc[index]["pert_idx"])
                extras: list[Any] = []
                if self._extension_use_kpgt:
                    assert self._extension_kpgt is not None
                    feature = self._extension_kpgt.get(pert_idx, self._extension_kpgt.get(str(pert_idx)))
                    if feature is None:
                        raise ValueError(f"missing KPGT feature for official pert_idx={pert_idx}")
                    extras.append(torch.as_tensor(feature, dtype=torch.float32))
                if self._extension_use_unipert:
                    assert self._extension_unipert is not None
                    extras.append(torch.as_tensor(self._extension_unipert[pert_idx], dtype=torch.float32))
                rows.append((*row, *extras))
            return rows

    XPertExtensionDataset.__name__ = "XPertExtensionDataset"
    return XPertExtensionDataset
