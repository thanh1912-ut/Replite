"""Shared native-trunk backbone implementation for Replite wrappers."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Sequence

import timm
from torch import Tensor, nn

from .feature_info import FeatureInfo, StageSpec, validate_out_indices
from .weights import PretrainedWeightsSpec, load_verified_state_dict

_STAGE_NAMES = ("C2", "C3", "C4", "C5")


class BackboneBase(nn.Module):
    """Base class for backbones exposing native C2..C5 stages of a timm model.

    The full timm classification model is instantiated with
    ``pretrained=False``, optionally strict-loaded from a verified
    ImageNet-1K checkpoint, and then only its native trunk modules (stem plus
    leading block groups) are registered on this wrapper. The projection block
    group, pooling, head and classifier are discarded and never become child
    modules, so they cannot leak into the state dict or parameter count.

    Subclasses define the timm architecture, the verified weight spec and the
    stage table; see :mod:`replite.backbone.mobilenet_v3` and
    :mod:`replite.backbone.mobilenet_v4`.
    """

    feature_info: FeatureInfo

    def __init__(
        self,
        *,
        timm_architecture: str,
        weights_spec: PretrainedWeightsSpec,
        stages: tuple[StageSpec, ...],
        trunk_block_groups: int,
        out_indices: Sequence[int] = (0, 1, 2, 3),
        pretrained: bool = False,
        checkpoint_path: str | Path | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        force_download: bool = False,
    ) -> None:
        """Initialize the backbone.

        Args:
            timm_architecture: Name of the timm architecture the trunk is
                taken from.
            weights_spec: Verified pretrained checkpoint specification.
            stages: Stage table (C2..C5) of the architecture.
            trunk_block_groups: Number of leading ``blocks`` children kept as
                native trunk; all later children (projection groups) and the
                classification head are removed.
            out_indices: Feature stages to return, as a non-empty, strictly
                increasing sequence of values in ``0..3``.
            pretrained: When True, download and strict-load the verified
                checkpoint; no network access happens when False.
            checkpoint_path: Optional offline copy of the pinned checkpoint.
            cache_dir: Optional Hugging Face cache directory.
            local_files_only: Forbid network access while resolving Hub weights.
            force_download: Refresh the Hub file before verification.
        """
        super().__init__()
        if (
            isinstance(trunk_block_groups, bool)
            or not isinstance(trunk_block_groups, int)
            or trunk_block_groups <= 0
        ):
            raise ValueError("trunk_block_groups must be a positive integer")
        self.feature_info = FeatureInfo(stages, out_indices)
        self.out_indices = self.feature_info.out_indices
        self._timm_architecture = timm_architecture
        self._weights_spec = weights_spec
        self._stages = stages
        self._trunk_block_groups = trunk_block_groups
        self._stage_by_group = {
            stage.blocks_end: stage_index for stage_index, stage in enumerate(stages)
        }
        self._last_group = max(
            stages[stage_index].blocks_end for stage_index in self.out_indices
        )

        if not pretrained and any(
            option is not None and option is not False
            for option in (checkpoint_path, cache_dir, local_files_only, force_download)
        ):
            raise ValueError("weight loading options require pretrained=True")

        full = timm.create_model(timm_architecture, pretrained=False)
        if pretrained:
            state_dict = load_verified_state_dict(
                weights_spec,
                checkpoint_path=checkpoint_path,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                force_download=force_download,
            )
            full.load_state_dict(state_dict, strict=True)
        self.conv_stem = full.conv_stem
        self.bn1 = full.bn1
        native_groups = list(full.blocks.children())[:trunk_block_groups]
        if self._last_group >= len(native_groups):
            raise ValueError(
                f"stage table requires block group {self._last_group}, but "
                f"{timm_architecture!r} exposes only {len(native_groups)} native groups"
            )
        self._legacy_pruned_state_keys = frozenset(
            f"blocks.{group_index}.{key}"
            for group_index in range(self._last_group + 1, trunk_block_groups)
            for key in native_groups[group_index].state_dict()
        )
        self.blocks = nn.Sequential(*native_groups[: self._last_group + 1])
        del full

        self._weights_loaded = bool(pretrained)
        self._weights_source = (
            "checkpoint_path"
            if checkpoint_path is not None
            else "huggingface_hub" if pretrained else "random_init"
        )
        self._checkpoint_path = (
            str(Path(checkpoint_path).expanduser().resolve())
            if checkpoint_path is not None
            else None
        )

    @property
    def timm_architecture(self) -> str:
        """Name of the timm architecture the trunk was taken from."""
        return self._timm_architecture

    @property
    def pretrained_cfg(self) -> dict[str, Any]:
        """Metadata of the pretrained checkpoint associated with this backbone."""
        return self._weights_spec.as_pretrained_cfg()

    @property
    def weights_loaded(self) -> bool:
        """Whether verified pretrained weights were loaded at construction."""

        return self._weights_loaded

    @property
    def weights_source(self) -> str:
        """Initialization source: random, Hugging Face Hub, or local checkpoint."""

        return self._weights_source

    @property
    def weights_provenance(self) -> dict[str, Any]:
        """JSON-serializable weight provenance to save beside checkpoints."""

        provenance = {
            "loaded": self.weights_loaded,
            "source": self.weights_source,
            "dataset": self._weights_spec.dataset,
            "repository": self._weights_spec.repository,
            "revision": self._weights_spec.revision,
            "filename": self._weights_spec.filename,
            "sha256": self._weights_spec.sha256,
        }
        if self._checkpoint_path is not None:
            provenance["checkpoint_path"] = self._checkpoint_path
        return provenance

    @property
    def backbone_config(self) -> dict[str, Any]:
        """JSON-serializable constructor/provenance configuration."""

        return {
            "architecture": self.timm_architecture,
            "out_indices": list(self.out_indices),
            "weights": self.weights_provenance,
        }

    def load_legacy_full_trunk_state_dict(
        self, state_dict: Mapping[str, Tensor]
    ) -> tuple[str, ...]:
        """Load a v0.1 full-trunk state dict into a trimmed shallow backbone.

        Only block groups deeper than this instance's deepest requested stage
        are ignored. Missing keys, shape mismatches, classifier keys, and all
        other unexpected entries still fail strictly.

        Returns:
            Sorted state-dict keys belonging to safely ignored deeper groups.
        """

        expected_keys = set(self.state_dict())
        unexpected_keys = set(state_dict) - expected_keys

        invalid = sorted(unexpected_keys - self._legacy_pruned_state_keys)
        if invalid:
            raise RuntimeError(
                "Legacy checkpoint contains unexpected non-pruned keys: "
                + ", ".join(invalid)
            )
        missing_pruned = sorted(self._legacy_pruned_state_keys - unexpected_keys)
        if missing_pruned:
            raise RuntimeError(
                "Legacy full-trunk checkpoint is missing canonical pruned keys: "
                + ", ".join(missing_pruned)
            )

        filtered = OrderedDict(
            (key, value) for key, value in state_dict.items() if key in expected_keys
        )
        metadata = getattr(state_dict, "_metadata", None)
        if metadata is not None:
            filtered._metadata = metadata
        self.load_state_dict(filtered, strict=True)
        return tuple(sorted(unexpected_keys))

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        """Run the trunk and return the selected feature maps as a tuple.

        Args:
            x: RGB image tensor in NCHW layout, already normalized by the
                caller. The backbone performs no resizing or normalization.

        Returns:
            One feature map per selected stage, ordered like ``out_indices``.
        """
        x = self.bn1(self.conv_stem(x))
        features: dict[int, Tensor] = {}
        for group_index, group in enumerate(self.blocks):
            x = group(x)
            stage_index = self._stage_by_group.get(group_index)
            if stage_index is not None and stage_index in self.out_indices:
                features[stage_index] = x
        return tuple(features[stage_index] for stage_index in self.out_indices)

    def extra_repr(self) -> str:
        return (
            f"architecture={self._timm_architecture}, "
            f"out_indices={self.out_indices}, "
            f"stages={tuple(_STAGE_NAMES[index] for index in self.out_indices)}, "
            f"weights_loaded={self.weights_loaded}"
        )
