"""Model-variant construction and freeze policies for EXP-011."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from drug_screen.foundation.pharmaformer_exp011 import Exp011ModelConfig, build_m3_adapter, build_official_m2


MODEL_IDS = frozenset({"M2", "M3-adapter", "M3-full", "M4"})


@dataclass(frozen=True)
class TargetPhase:
    """The fixed target-domain optimization phase for one epoch."""

    name: str
    train_adapter_head: bool
    train_last_transformer: bool
    train_all: bool


def target_phase(model_id: str, epoch: int) -> TargetPhase:
    """Return the pre-registered training policy; epochs are one-indexed."""
    if model_id not in MODEL_IDS or model_id == "M2":
        raise ValueError("only M3-adapter, M3-full and M4 have a target training phase")
    if epoch < 1:
        raise ValueError("epoch is one-indexed and must be positive")
    if model_id == "M3-adapter":
        return TargetPhase(
            name="adapter_head" if epoch <= 20 else "adapter_head_last_transformer",
            train_adapter_head=True,
            train_last_transformer=epoch > 20,
            train_all=False,
        )
    return TargetPhase(name="all_parameters", train_adapter_head=False, train_last_transformer=False, train_all=True)


def build_variant(model_id: str, upstream: Any, config: Exp011ModelConfig, *, source_backbone: Any | None = None) -> Any:
    """Build the specified model without modifying the immutable upstream class."""
    if model_id not in MODEL_IDS:
        raise ValueError(f"unsupported model id: {model_id}")
    if model_id == "M3-adapter":
        if source_backbone is None:
            raise ValueError("M3-adapter requires a loaded source M2 backbone")
        return build_m3_adapter(source_backbone, config)
    if model_id == "M3-full":
        if source_backbone is None:
            raise ValueError("M3-full requires a loaded source M2 backbone")
        return source_backbone
    if source_backbone is not None:
        raise ValueError(f"{model_id} must not receive a source backbone")
    return build_official_m2(upstream, config)


def _last_transformer_parameter_names(model: Any) -> set[str]:
    """Discover the final transformer block from the pinned upstream module."""
    root = model.backbone if hasattr(model, "backbone") else model
    try:
        layers = root.transformer.transformer_encoder.layers
        last_index = len(layers) - 1
    except (AttributeError, TypeError) as error:
        raise ValueError("cannot discover final upstream Transformer layer") from error
    prefix = "backbone." if hasattr(model, "backbone") else ""
    return {name for name, _ in model.named_parameters() if name.startswith(f"{prefix}transformer.transformer_encoder.layers.{last_index}.")}


def set_target_trainability(model: Any, model_id: str, epoch: int) -> TargetPhase:
    """Freeze/unfreeze exactly the declared target stage and return that stage."""
    phase = target_phase(model_id, epoch)
    for _, parameter in model.named_parameters():
        parameter.requires_grad = False
    if phase.train_all:
        for _, parameter in model.named_parameters():
            parameter.requires_grad = True
        return phase
    # M3-adapter always trains its residual and the official output head.
    for name, parameter in model.named_parameters():
        if name.startswith("adapter.") or name.startswith("backbone.transformer.output."):
            parameter.requires_grad = True
    if phase.train_last_transformer:
        names = _last_transformer_parameter_names(model)
        for name, parameter in model.named_parameters():
            if name in names:
                parameter.requires_grad = True
    return phase


def optimizer_parameter_groups(model: Any, model_id: str, epoch: int) -> list[dict[str, Any]]:
    """Build the prescribed target optimizer groups after applying freeze policy."""
    phase = set_target_trainability(model, model_id, epoch)
    if model_id == "M3-adapter":
        last_names = _last_transformer_parameter_names(model) if phase.train_last_transformer else set()
        adapter_head: list[Any] = []
        last_transformer: list[Any] = []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if name in last_names:
                last_transformer.append(parameter)
            else:
                adapter_head.append(parameter)
        groups: list[dict[str, Any]] = [{"params": adapter_head, "lr": 1e-4, "name": "adapter_head"}]
        if last_transformer:
            groups.append({"params": last_transformer, "lr": 1e-6, "name": "last_transformer"})
        return groups
    if model_id == "M3-full":
        return [{"params": [parameter for parameter in model.parameters() if parameter.requires_grad], "lr": 1e-6, "name": "full"}]
    if model_id == "M4":
        return [{"params": [parameter for parameter in model.parameters() if parameter.requires_grad], "lr": 1e-4, "name": "target_only"}]
    raise ValueError("M2 has no target optimizer")


def trainable_parameter_names(model: Any) -> tuple[str, ...]:
    return tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)


def assert_m3_zero_initial_delta(model: Any, gene: Any, drug: Any, source: Any, *, atol: float = 0.0) -> None:
    """Regression guard: a freshly inserted zero-initialised adapter equals M2."""
    import torch

    model.eval(); source.eval()
    with torch.no_grad():
        actual = model(gene, drug)
        expected = source(gene, drug)
    if not torch.allclose(actual, expected, rtol=0.0, atol=atol):
        raise AssertionError("zero-initialised M3 adapter changed the source prediction")
