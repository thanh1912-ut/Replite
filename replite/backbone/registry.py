"""Replite backbone registry and factory."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .base import BackboneBase, validate_out_indices
from .mobilenet_v3 import MobileNetV3Backbone
from .mobilenet_v4 import MobileNetV4Backbone

_REGISTRY: dict[str, type[BackboneBase]] = {
    "mobilenetv3_small_050": MobileNetV3Backbone,
    "mobilenetv4_conv_small": MobileNetV4Backbone,
}


def list_backbones() -> tuple[str, ...]:
    """Return all canonical backbone names in stable registration order."""
    return tuple(_REGISTRY)


def create_backbone(
    name: str,
    pretrained: bool = False,
    out_indices: Sequence[int] = (0, 1, 2, 3),
    *,
    checkpoint_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    force_download: bool = False,
) -> BackboneBase:
    """Create a feature backbone by canonical name.

    Args:
        name: One of the canonical names returned by :func:`list_backbones`.
        pretrained: When True, download and strict-load the verified
            ImageNet-1K checkpoint pinned in the backbone spec; no network
            access happens when False.
        out_indices: Feature stages to return, as a non-empty, strictly
            increasing sequence of values in ``0..3`` mapping onto C2..C5.
            Defaults to all stages ``(0, 1, 2, 3)``.
        checkpoint_path: Optional offline copy of the pinned official
            safetensors checkpoint. Requires ``pretrained=True``.
        cache_dir: Optional Hugging Face cache directory.
        local_files_only: Require the pinned file to exist in the local cache.
        force_download: Refresh the pinned Hub file before verification.

    Returns:
        The backbone module. Calling it returns a tuple of feature maps
        selected by ``out_indices``.

    Raises:
        ValueError: If ``name`` is not a registered backbone (the error lists
            the valid names) or ``out_indices`` is invalid.
    """
    if not isinstance(name, str):
        raise ValueError(
            f"Backbone name must be a string, got {type(name).__name__}. "
            f"Available backbones: {', '.join(list_backbones())}"
        )
    validated_indices = validate_out_indices(out_indices)
    try:
        backbone_cls = _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown backbone name {name!r}. "
            f"Available backbones: {', '.join(list_backbones())}"
        ) from None
    return backbone_cls(
        pretrained=pretrained,
        out_indices=validated_indices,
        checkpoint_path=checkpoint_path,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        force_download=force_download,
    )
