"""Replite MobileNetV3-Small x0.5 native-trunk backbone."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .base import BackboneBase, StageSpec
from .weights import PretrainedWeightsSpec

_WEIGHTS_SPEC = PretrainedWeightsSpec(
    architecture="mobilenetv3_small_050",
    repository="timm/mobilenetv3_small_050.lamb_in1k",
    revision="f58e7345afe2832abd6f81cc60f67cd1ddf7ce00",
    sha256="2e3f6937afd4b3704450518a2710168775d6c70ebfb7a0e9aaf06200c6fbe0c4",
)

_STAGES = (
    StageSpec(module="blocks.0", num_chs=8, reduction=4, blocks_end=0),
    StageSpec(module="blocks.1", num_chs=16, reduction=8, blocks_end=1),
    StageSpec(module="blocks.3", num_chs=24, reduction=16, blocks_end=3),
    StageSpec(module="blocks.4", num_chs=48, reduction=32, blocks_end=4),
)

_TRUNK_BLOCK_GROUPS = 5


class MobileNetV3Backbone(BackboneBase):
    """MobileNetV3-Small x0.5 backbone exposing native C2..C5 feature maps.

    Architecture source: timm ``mobilenetv3_small_050``; pretrained weights:
    ``timm/mobilenetv3_small_050.lamb_in1k`` (ImageNet-1K).

    The trunk keeps ``conv_stem``/``bn1`` plus block groups 0-4:

    - C2: ``blocks.0`` (8 channels, stride 4)
    - C3: ``blocks.1`` (16 channels, stride 8)
    - C4: ``blocks.2`` then ``blocks.3`` (24 channels, stride 16)
    - C5: ``blocks.4`` (48 channels, stride 32)

    Block group 5 (the 48->288 channel projection), pooling, head and
    classifier are removed entirely. Inputs must be RGB tensors in NCHW
    layout, already normalized by the caller.
    """

    def __init__(
        self,
        pretrained: bool = False,
        out_indices: Sequence[int] = (0, 1, 2, 3),
        *,
        checkpoint_path: str | Path | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        force_download: bool = False,
    ) -> None:
        """Initialize the backbone.

        Args:
            pretrained: When True, download and strict-load the verified
                ImageNet-1K checkpoint; no network access happens when False.
            out_indices: Feature stages to return, as a non-empty, strictly
                increasing sequence of values in ``0..3``.
            checkpoint_path: Optional offline copy of the pinned checkpoint.
            cache_dir: Optional Hugging Face cache directory.
            local_files_only: Resolve pretrained weights from cache only.
            force_download: Refresh the cached Hub file before verification.
        """
        super().__init__(
            timm_architecture="mobilenetv3_small_050",
            weights_spec=_WEIGHTS_SPEC,
            stages=_STAGES,
            trunk_block_groups=_TRUNK_BLOCK_GROUPS,
            out_indices=out_indices,
            pretrained=pretrained,
            checkpoint_path=checkpoint_path,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            force_download=force_download,
        )
