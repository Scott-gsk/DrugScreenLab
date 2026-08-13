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

from pathlib import Path
from typing import Any, Callable, Mapping, Type


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
    gate_init: float = 0.0,
    freeze_official: bool = False,
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
            # A zero-initialised residual gate gives strict baseline
            # equivalence at construction time.  The hook bypasses sequence
            # extension while all gates are zero, so the extra positions do
            # not perturb attention/normalisation in the official network.
            self.additive_gate = nn.Parameter(
                torch.full((int(use_kpgt) + int(use_unipert),), float(gate_init))
            )
            self.official_checkpoint_loaded = False
            self.official_parameters_frozen = False
            self.checkpoint_audit: dict[str, Any] | None = None
            self._active_additive_features: tuple[Tensor | None, Tensor | None] | None = None
            self._drug_embedding_hook_handle = self.drug_emb.register_forward_hook(self._drug_embedding_hook)

            if freeze_official:
                self.freeze_official_parameters()

        def freeze_official_parameters(self) -> None:
            """Freeze inherited XPert weights; leave only extension weights trainable."""
            for name, parameter in self.named_parameters():
                if name == "additive_gate" or name.startswith("perturbagen_encoder."):
                    continue
                parameter.requires_grad_(False)
            self.official_parameters_frozen = True

        def _drug_embedding_hook(self, _module: Any, _inputs: Any, output: Tensor) -> Tensor:
            if self._active_additive_features is None:
                return output
            kpgt, unipert = self._active_additive_features
            gates = self.additive_gate
            if torch.count_nonzero(gates).item() == 0:
                # Keep the official sequence *exactly* unchanged while still
                # exposing a straight-through gradient to the gate.  This
                # avoids a dead gate and makes a freshly inherited checkpoint
                # numerically equivalent to the official baseline in both
                # train and eval modes.
                straight_through_zero = gates - gates.detach()
                # Use raw feature means here to avoid invoking dropout (and
                # consuming RNG state) while the extension is exactly off.
                signal = (kpgt if kpgt is not None else unipert).mean(dim=-1, keepdim=True)
                if kpgt is not None and unipert is not None:
                    signal = signal + unipert.mean(dim=-1, keepdim=True)
                residual = signal.unsqueeze(-1) * straight_through_zero.sum().to(signal)
                return output + residual
            tokens = self.perturbagen_encoder(kpgt=kpgt, unipert=unipert)
            tokens = tokens * gates.to(device=tokens.device, dtype=tokens.dtype).view(1, -1, 1)
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


def load_xpert_checkpoint(
    model: Any,
    checkpoint: str | Path | Mapping[str, Any],
    *,
    strict_official: bool = True,
    map_location: Any = "cpu",
) -> dict[str, Any]:
    """Inherit an official XPert checkpoint without silently dropping weights.

    Checkpoints may be a raw state dict or a common ``state_dict``/
    ``model_state_dict`` wrapper.  Extension-only parameters are allowed to be
    missing; every official parameter must be present when ``strict_official``
    is true.  The returned audit is attached to ``model.checkpoint_audit``.
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required for checkpoint loading")
    if isinstance(checkpoint, (str, Path)):
        try:
            payload = torch.load(str(checkpoint), map_location=map_location, weights_only=False)
        except TypeError:  # older torch versions
            payload = torch.load(str(checkpoint), map_location=map_location)
    else:
        payload = checkpoint
    state: Any = payload
    if isinstance(payload, Mapping):
        for key in ("state_dict", "model_state_dict", "model"):
            value = payload.get(key)
            if isinstance(value, Mapping):
                state = value
                break
    if not isinstance(state, Mapping):
        raise ValueError("XPert checkpoint must contain a state-dict mapping")
    target_keys = set(model.state_dict().keys())
    normalized: dict[str, Any] = {}
    unrecognized: list[str] = []
    for key, value in state.items():
        name = str(key)
        for prefix in ("module.", "model."):
            if name.startswith(prefix):
                name = name[len(prefix) :]
        if name in target_keys:
            normalized[name] = value
        else:
            unrecognized.append(name)
    extension_prefixes = ("perturbagen_encoder.", "additive_gate")
    official_keys = {k for k in target_keys if not k.startswith(extension_prefixes)}
    result = model.load_state_dict(normalized, strict=False)
    missing_official = sorted(k for k in result.missing_keys if k in official_keys)
    unexpected = sorted(set(unrecognized) | set(result.unexpected_keys))
    if strict_official and (missing_official or unexpected):
        raise ValueError(
            "official XPert checkpoint inheritance failed: "
            f"missing={missing_official}, unexpected={unexpected}"
        )
    audit = {
        "loaded_keys": len(normalized),
        "official_parameter_count": len(official_keys),
        "missing_official": missing_official,
        "missing_extension": sorted(k for k in result.missing_keys if k not in official_keys),
        "unexpected": unexpected,
        "strict_official": bool(strict_official),
    }
    model.official_checkpoint_loaded = not missing_official
    model.checkpoint_audit = audit
    return audit


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
