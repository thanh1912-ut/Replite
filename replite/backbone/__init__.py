"""Pretrained native-trunk feature backbones for dense prediction.

Public API::

    from replite.backbone import create_backbone, list_backbones

    model = create_backbone(
        name="mobilenetv4_conv_small",
        pretrained=True,
        out_indices=(0, 1, 2, 3),
    )
    features = model(images)  # tuple of C2..C5 feature maps

The backbones expose the *native* trunk stages C2-C5 (no classification
head, no 288/960-channel projection). Inputs are RGB tensors in NCHW
layout, already normalized by the caller; the backbone performs no
resizing or normalization itself.
"""

from .base import BackboneBase, StageSpec, validate_out_indices
from .feature_info import FeatureInfo
from .mobilenet_v3 import MobileNetV3Backbone
from .mobilenet_v4 import MobileNetV4Backbone
from .registry import create_backbone, list_backbones
from .weights import (
    CheckpointFormatError,
    CheckpointDownloadError,
    ChecksumMismatchError,
    PretrainedWeightsSpec,
    load_verified_state_dict,
)

__all__ = [
    "BackboneBase",
    "CheckpointDownloadError",
    "CheckpointFormatError",
    "ChecksumMismatchError",
    "FeatureInfo",
    "MobileNetV3Backbone",
    "MobileNetV4Backbone",
    "PretrainedWeightsSpec",
    "StageSpec",
    "create_backbone",
    "list_backbones",
    "load_verified_state_dict",
    "validate_out_indices",
]
