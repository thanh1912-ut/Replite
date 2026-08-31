"""Lightweight, portable task heads for the RepLite multi-task model.

The heads in this module deliberately return raw task predictions.  Losses,
anchor/grid construction, box decoding, and post-processing remain caller
responsibilities so the model stays useful across training stacks and export
backends.
"""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import NamedTuple, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .blocks import ConvBNAct, DepthwiseSeparableConv, RepDepthwiseBlock


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _validate_feature(x: Tensor, channels: int, name: str) -> None:
    if not isinstance(x, Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if x.ndim != 4:
        raise ValueError(f"{name} must use NCHW layout, got shape {tuple(x.shape)}")
    if torch.jit.is_tracing():
        return
    if x.shape[1] != channels:
        raise ValueError(f"{name} must have {channels} channels, got {int(x.shape[1])}")


def _output_size(value: Sequence[int]) -> tuple[int, int]:
    try:
        size = tuple(value)
    except TypeError as exc:
        raise ValueError("output_size must be a two-element integer sequence") from exc
    if len(size) != 2:
        raise ValueError("output_size must contain exactly two positive integers")
    checked = []
    integer_tensor_dtypes = {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }
    for dimension in size:
        if isinstance(dimension, bool):
            raise ValueError("output_size must contain exactly two positive integers")
        if isinstance(dimension, Integral):
            if dimension <= 0:
                raise ValueError(
                    "output_size must contain exactly two positive integers"
                )
            checked.append(int(dimension))
            continue
        if isinstance(dimension, torch.SymInt):
            checked.append(dimension)
            continue
        if (
            isinstance(dimension, Tensor)
            and dimension.ndim == 0
            and dimension.dtype in integer_tensor_dtypes
        ):
            # Legacy ONNX tracing represents dynamic shape values as scalar
            # integer tensors. Keep them symbolic instead of calling item().
            checked.append(dimension)
            continue
        raise ValueError("output_size must contain exactly two positive integers")
    return checked[0], checked[1]  # type: ignore[return-value]


def _resize(x: Tensor, output_size: Sequence[int]) -> Tensor:
    size = _output_size(output_size)
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class TaskAdapter(nn.Module):
    """Project a shared dense feature into a task-specific feature space."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 1,
    ) -> None:
        super().__init__()
        self.in_channels = _positive_int(in_channels, "in_channels")
        self.out_channels = _positive_int(out_channels, "out_channels")
        num_blocks = _positive_int(num_blocks, "num_blocks")

        self.projection = ConvBNAct(
            self.in_channels,
            self.out_channels,
            kernel_size=1,
        )
        self.refinement = nn.Sequential(
            *(RepDepthwiseBlock(self.out_channels) for _ in range(num_blocks))
        )

    def forward(self, x: Tensor) -> Tensor:
        _validate_feature(x, self.in_channels, "x")
        return self.refinement(self.projection(x))


class DetectionOutput(NamedTuple):
    """Raw anchor-free predictions ordered as ``(P3, P4, P5)``.

    ``cls_logits`` and ``quality`` are logits.  With ``reg_max == 0``, each
    box tensor contains four positive direct LTRB distances.  Otherwise it
    contains ``4 * (reg_max + 1)`` unnormalized distribution logits.
    """

    cls_logits: tuple[Tensor, Tensor, Tensor]
    box_regression: tuple[Tensor, Tensor, Tensor]
    quality: tuple[Tensor, Tensor, Tensor]


def _pyramid_channels(value: int | Sequence[int]) -> tuple[int, int, int]:
    if isinstance(value, Integral) and not isinstance(value, bool):
        channel = _positive_int(value, "in_channels")
        return channel, channel, channel
    try:
        channels = tuple(value)
    except TypeError as exc:
        raise ValueError(
            "in_channels must be a positive integer or three channel counts"
        ) from exc
    if len(channels) != 3:
        raise ValueError("in_channels must define exactly P3, P4, and P5")
    return tuple(
        _positive_int(channel, f"in_channels[{index}]")
        for index, channel in enumerate(channels)
    )  # type: ignore[return-value]


class DetectionHead(nn.Module):
    """Anchor-free decoupled head with towers shared across P3/P4/P5.

    Per-level 1x1 projections normalize input widths, after which one shared
    classification tower and one shared regression tower process all three
    levels.  Final predictors remain level-specific so each scale can learn
    its own calibration.
    """

    num_levels = 3

    def __init__(
        self,
        in_channels: int | Sequence[int],
        num_classes: int,
        head_channels: int = 64,
        num_convs: int = 2,
        reg_max: int = 0,
    ) -> None:
        super().__init__()
        self.in_channels = _pyramid_channels(in_channels)
        self.num_classes = _positive_int(num_classes, "num_classes")
        self.head_channels = _positive_int(head_channels, "head_channels")
        num_convs = _positive_int(num_convs, "num_convs")
        self.reg_max = _non_negative_int(reg_max, "reg_max")
        self.regression_channels = 4 if self.reg_max == 0 else 4 * (self.reg_max + 1)

        self.input_projections = nn.ModuleList(
            ConvBNAct(channels, self.head_channels, kernel_size=1)
            for channels in self.in_channels
        )
        self.classification_tower = nn.Sequential(
            *(
                DepthwiseSeparableConv(self.head_channels, self.head_channels)
                for _ in range(num_convs)
            )
        )
        self.regression_tower = nn.Sequential(
            *(
                DepthwiseSeparableConv(self.head_channels, self.head_channels)
                for _ in range(num_convs)
            )
        )
        self.class_predictors = nn.ModuleList(
            nn.Conv2d(self.head_channels, self.num_classes, kernel_size=1)
            for _ in range(self.num_levels)
        )
        self.box_predictors = nn.ModuleList(
            nn.Conv2d(self.head_channels, self.regression_channels, kernel_size=1)
            for _ in range(self.num_levels)
        )
        self.quality_predictors = nn.ModuleList(
            nn.Conv2d(self.head_channels, 1, kernel_size=1)
            for _ in range(self.num_levels)
        )
        self._initialize_predictors()

    def _initialize_predictors(self) -> None:
        prior_bias = -math.log((1.0 - 0.01) / 0.01)
        for predictor in self.class_predictors:
            nn.init.normal_(predictor.weight, std=0.01)
            nn.init.constant_(predictor.bias, prior_bias)
        for predictor in self.quality_predictors:
            nn.init.normal_(predictor.weight, std=0.01)
            nn.init.constant_(predictor.bias, prior_bias)
        for predictor in self.box_predictors:
            nn.init.normal_(predictor.weight, std=0.01)
            nn.init.constant_(predictor.bias, 1.0 if self.reg_max == 0 else 0.0)

    def forward(self, features: Sequence[Tensor]) -> DetectionOutput:
        try:
            pyramid = tuple(features)
        except TypeError as exc:
            raise ValueError("features must contain exactly P3, P4, and P5") from exc
        if len(pyramid) != self.num_levels:
            raise ValueError("features must contain exactly P3, P4, and P5")

        batch_size: int | None = None
        class_logits: list[Tensor] = []
        box_regression: list[Tensor] = []
        quality: list[Tensor] = []
        for level, (feature, channels, projection) in enumerate(
            zip(pyramid, self.in_channels, self.input_projections)
        ):
            _validate_feature(feature, channels, f"features[{level}]")
            if batch_size is None:
                batch_size = feature.shape[0]
            elif (
                not torch.compiler.is_compiling()
                and not torch.jit.is_tracing()
                and feature.shape[0] != batch_size
            ):
                raise ValueError("all pyramid levels must have the same batch size")

            shared = projection(feature)
            class_feature = self.classification_tower(shared)
            regression_feature = self.regression_tower(shared)
            class_logits.append(self.class_predictors[level](class_feature))
            box = self.box_predictors[level](regression_feature)
            box_regression.append(F.softplus(box) if self.reg_max == 0 else box)
            quality.append(self.quality_predictors[level](regression_feature))

        return DetectionOutput(
            cls_logits=(class_logits[0], class_logits[1], class_logits[2]),
            box_regression=(
                box_regression[0],
                box_regression[1],
                box_regression[2],
            ),
            quality=(quality[0], quality[1], quality[2]),
        )


class DensePredictionHead(nn.Module):
    """Produce semantic logits and resize them to the requested image size."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_channels: int | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = _positive_int(in_channels, "in_channels")
        self.num_classes = _positive_int(num_classes, "num_classes")
        hidden_channels = (
            self.in_channels
            if hidden_channels is None
            else _positive_int(hidden_channels, "hidden_channels")
        )
        self.refinement = DepthwiseSeparableConv(
            self.in_channels,
            hidden_channels,
        )
        self.predictor = nn.Conv2d(hidden_channels, self.num_classes, kernel_size=1)

    def forward(self, x: Tensor, output_size: tuple[int, int]) -> Tensor:
        _validate_feature(x, self.in_channels, "x")
        logits = self.predictor(self.refinement(x))
        return _resize(logits, output_size)


class DepthHead(nn.Module):
    """Produce a strictly positive one-channel depth map."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int | None = None,
        min_depth: float = 1e-3,
        max_depth: float | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = _positive_int(in_channels, "in_channels")
        hidden_channels = (
            self.in_channels
            if hidden_channels is None
            else _positive_int(hidden_channels, "hidden_channels")
        )
        if (
            isinstance(min_depth, bool)
            or not isinstance(min_depth, Real)
            or not math.isfinite(float(min_depth))
            or min_depth < 0
        ):
            raise ValueError("min_depth must be a finite non-negative number")
        if max_depth is not None and (
            isinstance(max_depth, bool)
            or not isinstance(max_depth, Real)
            or not math.isfinite(float(max_depth))
            or max_depth <= min_depth
        ):
            raise ValueError("max_depth must be finite and greater than min_depth")
        self.min_depth = float(min_depth)
        self.max_depth = None if max_depth is None else float(max_depth)

        self.refinement = DepthwiseSeparableConv(
            self.in_channels,
            hidden_channels,
        )
        self.predictor = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, x: Tensor, output_size: tuple[int, int]) -> Tensor:
        _validate_feature(x, self.in_channels, "x")
        raw_depth = _resize(self.predictor(self.refinement(x)), output_size)
        if self.max_depth is None:
            return F.softplus(raw_depth) + self.min_depth
        return self.min_depth + (self.max_depth - self.min_depth) * torch.sigmoid(
            raw_depth
        )


class ClassificationHead(nn.Module):
    """Global-average-pooling classification head."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_channels = _positive_int(in_channels, "in_channels")
        self.num_classes = _positive_int(num_classes, "num_classes")
        if (
            isinstance(dropout, bool)
            or not isinstance(dropout, Real)
            or not math.isfinite(float(dropout))
            or not 0.0 <= dropout < 1.0
        ):
            raise ValueError("dropout must be finite and in the interval [0, 1)")
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(self.in_channels, self.num_classes)

    def forward(self, x: Tensor) -> Tensor:
        _validate_feature(x, self.in_channels, "x")
        return self.classifier(self.dropout(self.pool(x).flatten(1)))


class _GatedCrossProjection(nn.Module):
    """Create one gated residual update solely from a source feature."""

    def __init__(
        self,
        source_channels: int,
        target_channels: int,
        hidden_channels: int,
    ) -> None:
        super().__init__()
        self.context = nn.Sequential(
            ConvBNAct(source_channels, hidden_channels, kernel_size=1),
            DepthwiseSeparableConv(hidden_channels, hidden_channels),
        )
        self.update = nn.Conv2d(hidden_channels, target_channels, kernel_size=1)
        self.gate = nn.Conv2d(hidden_channels, target_channels, kernel_size=1)

    def forward(self, source: Tensor) -> Tensor:
        context = self.context(source)
        return self.update(context) * torch.sigmoid(self.gate(context))


class ResidualGatedFusion(nn.Module):
    """Simultaneously exchange segmentation and depth information.

    Both cross-task residuals are calculated from the *pre-fusion* features.
    The two learnable residual scales start at zero, making the initialized
    module an exact identity while still allowing each direction to turn on
    independently during optimization.
    """

    def __init__(
        self,
        seg_channels: int,
        depth_channels: int | None = None,
        hidden_channels: int | None = None,
        enabled: bool = True,
    ) -> None:
        super().__init__()
        self.seg_channels = _positive_int(seg_channels, "seg_channels")
        self.depth_channels = (
            self.seg_channels
            if depth_channels is None
            else _positive_int(depth_channels, "depth_channels")
        )
        hidden_channels = (
            min(self.seg_channels, self.depth_channels)
            if hidden_channels is None
            else _positive_int(hidden_channels, "hidden_channels")
        )
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be a boolean")
        self.enabled = enabled

        self.depth_to_seg = _GatedCrossProjection(
            self.depth_channels,
            self.seg_channels,
            hidden_channels,
        )
        self.seg_to_depth = _GatedCrossProjection(
            self.seg_channels,
            self.depth_channels,
            hidden_channels,
        )
        self.depth_to_seg_scale = nn.Parameter(torch.zeros(()))
        self.seg_to_depth_scale = nn.Parameter(torch.zeros(()))

    def _validate_inputs(
        self,
        seg_features: Tensor,
        depth_features: Tensor,
    ) -> None:
        _validate_feature(seg_features, self.seg_channels, "seg_features")
        _validate_feature(depth_features, self.depth_channels, "depth_features")
        if not torch.jit.is_tracing():
            if seg_features.shape[0] != depth_features.shape[0] or tuple(
                seg_features.shape[-2:]
            ) != tuple(depth_features.shape[-2:]):
                raise ValueError(
                    "seg_features and depth_features must share batch and spatial shapes"
                )
            if seg_features.device != depth_features.device:
                raise ValueError("seg_features and depth_features must share a device")
            if seg_features.dtype != depth_features.dtype:
                raise ValueError("seg_features and depth_features must share a dtype")

    def _use_fusion(self, enabled: bool | None) -> bool:
        if enabled is not None and not isinstance(enabled, bool):
            raise ValueError("enabled override must be a boolean or None")
        return self.enabled if enabled is None else enabled

    def forward_segmentation(
        self,
        seg_features: Tensor,
        depth_features: Tensor,
        *,
        enabled: bool | None = None,
    ) -> Tensor:
        """Apply only the depth-to-segmentation residual direction."""

        self._validate_inputs(seg_features, depth_features)
        if not self._use_fusion(enabled):
            return seg_features
        seg_delta = self.depth_to_seg(depth_features)
        return seg_features + self.depth_to_seg_scale * seg_delta

    def forward_depth(
        self,
        seg_features: Tensor,
        depth_features: Tensor,
        *,
        enabled: bool | None = None,
    ) -> Tensor:
        """Apply only the segmentation-to-depth residual direction."""

        self._validate_inputs(seg_features, depth_features)
        if not self._use_fusion(enabled):
            return depth_features
        depth_delta = self.seg_to_depth(seg_features)
        return depth_features + self.seg_to_depth_scale * depth_delta

    def forward(
        self,
        seg_features: Tensor,
        depth_features: Tensor,
        *,
        enabled: bool | None = None,
    ) -> tuple[Tensor, Tensor]:
        self._validate_inputs(seg_features, depth_features)
        if not self._use_fusion(enabled):
            return seg_features, depth_features

        # Compute both deltas before either residual is applied.  Keeping these
        # expressions separate makes accidental circular/sequential fusion hard.
        seg_delta = self.depth_to_seg(depth_features)
        depth_delta = self.seg_to_depth(seg_features)
        fused_seg = seg_features + self.depth_to_seg_scale * seg_delta
        fused_depth = depth_features + self.seg_to_depth_scale * depth_delta
        return fused_seg, fused_depth


__all__ = [
    "ClassificationHead",
    "DensePredictionHead",
    "DepthHead",
    "DetectionHead",
    "DetectionOutput",
    "ResidualGatedFusion",
    "TaskAdapter",
]
