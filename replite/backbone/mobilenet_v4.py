"""Replite MobileNetV4-Conv-S native-trunk backbone."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .base import BackboneBase, StageSpec
from .weights import PretrainedWeightsSpec

_WEIGHTS_SPEC = PretrainedWeightsSpec(
    architecture="mobilenetv4_conv_small",
    repository="timm/mobilenetv4_conv_small.e2400_r224_in1k",
    revision="7249cacba963f438597f373327119b22f4d3a848",
    sha256="7a7102ec18f62bbfb555b6fe829bbb5af749516b84174926c29ffdfdfc03aec4",
    test_input_size=(3, 256, 256),
    test_crop_pct=0.95,
)

_STAGES = (
    StageSpec(module="blocks.0", num_chs=32, reduction=4, blocks_end=0),
    StageSpec(module="blocks.1", num_chs=64, reduction=8, blocks_end=1),
    StageSpec(module="blocks.2", num_chs=96, reduction=16, blocks_end=2),
    StageSpec(module="blocks.3", num_chs=128, reduction=32, blocks_end=3),
)

_TRUNK_BLOCK_GROUPS = 4


class MobileNetV4Backbone(BackboneBase):
    """MobileNetV4-Conv-S backbone exposing native C2..C5 feature maps.

    Architecture source: timm ``mobilenetv4_conv_small``; pretrained weights:
    ``timm/mobilenetv4_conv_small.e2400_r224_in1k`` (ImageNet-1K).

    The trunk keeps ``conv_stem``/``bn1`` plus block groups 0-3:

    - C2: ``blocks.0`` (32 channels, stride 4)
    - C3: ``blocks.1`` (64 channels, stride 8)
    - C4: ``blocks.2`` (96 channels, stride 16)
    - C5: ``blocks.3`` (128 channels, stride 32)

    Block group 4 (the 128->960 channel projection), pooling, head and
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
            timm_architecture="mobilenetv4_conv_small",
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
