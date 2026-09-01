"""Anchor-free detection assignment, losses, decoding, and post-processing.

The model deliberately exposes raw detection maps.  This module supplies one
small, deterministic FCOS-style training contract for those maps. Coordinates
use half-open ``XYXY`` pixel boxes, while head regressions use LTRB distances
measured in feature cells.  CUDA validation opportunistically uses
``torchvision.ops.batched_nms`` and retains the dependency-free implementation
as the deterministic CPU/fallback path.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from replite.multitask.heads import DetectionOutput

try:  # timm installations normally provide torchvision, but keep portability.
    from torchvision.ops import batched_nms as _torchvision_batched_nms
except (ImportError, RuntimeError):  # pragma: no cover - environment dependent
    _torchvision_batched_nms = None


DEFAULT_STRIDES = (8, 16, 32)
DEFAULT_SIZE_RANGES = ((0.0, 64.0), (64.0, 128.0), (128.0, math.inf))


class DetectionPoints(NamedTuple):
    """Flattened point metadata ordered P3, then P4, then P5."""

    points: Tensor
    strides: Tensor
    levels: Tensor


class FCOSAssignment(NamedTuple):
    """Targets for every flattened point in one image.

    Label ``-1`` means background and ``-2`` means ignored. ``ltrb_cells`` is
    meaningful only at positive locations.
    """

    labels: Tensor
    boxes: Tensor
    ltrb_cells: Tensor
    centerness: Tensor
    positive_mask: Tensor
    valid_mask: Tensor
    matched_gt_indices: Tensor
    num_targets: Tensor
    num_unmatched: Tensor


class DetectionLosses(NamedTuple):
    """Differentiable detection terms plus detached assignment statistics."""

    total: Tensor
    classification: Tensor
    box: Tensor
    quality: Tensor
    dfl: Tensor
    num_positive: Tensor
    num_targets: Tensor
    num_unmatched: Tensor


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _unit_interval(value: object, name: str, *, inclusive_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    lower_ok = result >= 0.0 if inclusive_zero else result > 0.0
    if not lower_ok or result > 1.0:
        raise ValueError(f"{name} must be in {'[0, 1]' if inclusive_zero else '(0, 1]'}")
    return result


def _non_negative_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a non-negative real number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite non-negative real number")
    return result


def _pair(value: Sequence[int] | Tensor, name: str) -> tuple[int, int]:
    if isinstance(value, Tensor):
        if value.ndim != 1 or value.numel() != 2:
            raise ValueError(f"{name} must contain height and width")
        value = value.detach().cpu().tolist()
    try:
        raw = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must contain height and width") from exc
    if len(raw) != 2:
        raise ValueError(f"{name} must contain height and width")
    return (
        _positive_int(raw[0], f"{name}[0]"),
        _positive_int(raw[1], f"{name}[1]"),
    )


def _strides(value: Sequence[int]) -> tuple[int, int, int]:
    try:
        raw = tuple(value)
    except TypeError as exc:
        raise ValueError("strides must contain exactly three positive integers") from exc
    if len(raw) != 3:
        raise ValueError("strides must contain exactly three positive integers")
    return (
        _positive_int(raw[0], "strides[0]"),
        _positive_int(raw[1], "strides[1]"),
        _positive_int(raw[2], "strides[2]"),
    )


def _size_ranges(
    value: Sequence[Sequence[float]],
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    try:
        raw = tuple(tuple(level) for level in value)
    except TypeError as exc:
        raise ValueError("size_ranges must define three (lower, upper) pairs") from exc
    if len(raw) != 3 or any(len(level) != 2 for level in raw):
        raise ValueError("size_ranges must define three (lower, upper) pairs")
    checked: list[tuple[float, float]] = []
    previous_upper = 0.0
    for index, (lower, upper) in enumerate(raw):
        if isinstance(lower, bool) or isinstance(upper, bool):
            raise ValueError("size range bounds must be real numbers")
        lower_float, upper_float = float(lower), float(upper)
        if lower_float < 0 or upper_float <= lower_float:
            raise ValueError("each size range must have 0 <= lower < upper")
        if index and lower_float != previous_upper:
            raise ValueError("size_ranges must be contiguous")
        checked.append((lower_float, upper_float))
        previous_upper = upper_float
    if checked[0][0] != 0.0 or not math.isinf(checked[-1][1]):
        raise ValueError("size_ranges must cover [0, infinity)")
    return checked[0], checked[1], checked[2]


def make_detection_points(
    feature_shapes: Sequence[Sequence[int]],
    *,
    strides: Sequence[int] = DEFAULT_STRIDES,
    device: torch.device | str | None = None,
) -> DetectionPoints:
    """Create float32 cell-center points for P3/P4/P5 feature shapes."""

    strides = _strides(strides)
    try:
        shapes = tuple(_pair(shape, f"feature_shapes[{index}]") for index, shape in enumerate(feature_shapes))
    except TypeError as exc:
        raise ValueError("feature_shapes must contain exactly three shapes") from exc
    if len(shapes) != 3:
        raise ValueError("feature_shapes must contain exactly three shapes")

    point_groups: list[Tensor] = []
    stride_groups: list[Tensor] = []
    level_groups: list[Tensor] = []
    for level, ((height, width), stride) in enumerate(zip(shapes, strides)):
        y = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) * stride
        x = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) * stride
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        points = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)
        point_groups.append(points)
        stride_groups.append(points.new_full((points.shape[0],), float(stride)))
        level_groups.append(
            torch.full(
                (points.shape[0],),
                level,
                dtype=torch.long,
                device=points.device,
            )
        )
    return DetectionPoints(
        points=torch.cat(point_groups, dim=0),
        strides=torch.cat(stride_groups, dim=0),
        levels=torch.cat(level_groups, dim=0),
    )


def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Pairwise IoU for half-open XYXY boxes."""

    if boxes1.ndim != 2 or boxes1.shape[1] != 4:
        raise ValueError("boxes1 must have shape N,4")
    if boxes2.ndim != 2 or boxes2.shape[1] != 4:
        raise ValueError("boxes2 must have shape M,4")
    boxes1 = boxes1.float()
    boxes2 = boxes2.float()
    top_left = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    intersection = (bottom_right - top_left).clamp_min(0).prod(dim=-1)
    area1 = (boxes1[:, 2:] - boxes1[:, :2]).clamp_min(0).prod(dim=-1)
    area2 = (boxes2[:, 2:] - boxes2[:, :2]).clamp_min(0).prod(dim=-1)
    union = area1[:, None] + area2[None, :] - intersection
    return intersection / union.clamp_min(torch.finfo(torch.float32).eps)


def generalized_box_iou_aligned(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Generalized IoU for aligned ``N,4`` XYXY box pairs."""

    if boxes1.ndim != 2 or boxes1.shape[1] != 4:
        raise ValueError("boxes1 must have shape N,4")
    if boxes2.shape != boxes1.shape:
        raise ValueError("boxes2 must have the same shape as boxes1")
    boxes1 = boxes1.float()
    boxes2 = boxes2.float()
    top_left = torch.maximum(boxes1[:, :2], boxes2[:, :2])
    bottom_right = torch.minimum(boxes1[:, 2:], boxes2[:, 2:])
    intersection = (bottom_right - top_left).clamp_min(0).prod(dim=-1)
    area1 = (boxes1[:, 2:] - boxes1[:, :2]).clamp_min(0).prod(dim=-1)
    area2 = (boxes2[:, 2:] - boxes2[:, :2]).clamp_min(0).prod(dim=-1)
    union = area1 + area2 - intersection
    iou = intersection / union.clamp_min(torch.finfo(torch.float32).eps)
    enclosing_top_left = torch.minimum(boxes1[:, :2], boxes2[:, :2])
    enclosing_bottom_right = torch.maximum(boxes1[:, 2:], boxes2[:, 2:])
    enclosing_area = (
        enclosing_bottom_right - enclosing_top_left
    ).clamp_min(0).prod(dim=-1)
    return iou - (enclosing_area - union) / enclosing_area.clamp_min(
        torch.finfo(torch.float32).eps
    )


def decode_box_regression(
    box_regression: Tensor,
    points: Tensor,
    point_strides: Tensor,
    *,
    reg_max: int,
) -> Tensor:
    """Decode direct or distributional cell distances into pixel XYXY boxes."""

    reg_max = _non_negative_int(reg_max, "reg_max")
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape N,2")
    if point_strides.ndim != 1 or point_strides.shape[0] != points.shape[0]:
        raise ValueError("point_strides must have shape N")
    if box_regression.shape[-2 if reg_max > 0 else -1] != 4:
        expected = "...,N,4,K" if reg_max > 0 else "...,N,4"
        raise ValueError(f"box_regression must have shape {expected}")
    if box_regression.shape[-3 if reg_max > 0 else -2] != points.shape[0]:
        raise ValueError("box_regression point count must match points")

    regression = box_regression.float()
    if reg_max > 0:
        if regression.shape[-1] != reg_max + 1:
            raise ValueError("distribution bin count must equal reg_max + 1")
        bins = torch.arange(
            reg_max + 1,
            device=regression.device,
            dtype=torch.float32,
        )
        distances = (regression.softmax(dim=-1) * bins).sum(dim=-1)
    else:
        distances = regression

    leading_dimensions = distances.ndim - 2
    view_prefix = (1,) * leading_dimensions
    points_float = points.to(device=distances.device, dtype=torch.float32).view(
        *view_prefix, points.shape[0], 2
    )
    stride_float = point_strides.to(
        device=distances.device, dtype=torch.float32
    ).view(*view_prefix, points.shape[0], 1)
    distances = distances * stride_float
    return torch.cat(
        (
            points_float - distances[..., :2],
            points_float + distances[..., 2:],
        ),
        dim=-1,
    )


def _validate_assignment_inputs(
    points: Tensor,
    point_strides: Tensor,
    level_ids: Tensor,
    boxes: Tensor,
    labels: Tensor,
    ignore_boxes: Tensor | None,
) -> None:
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape P,2")
    if point_strides.shape != (points.shape[0],):
        raise ValueError("point_strides must have shape P")
    if level_ids.shape != (points.shape[0],) or level_ids.dtype != torch.long:
        raise ValueError("level_ids must be a long tensor with shape P")
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must have shape N,4")
    if labels.shape != (boxes.shape[0],) or labels.dtype != torch.long:
        raise ValueError("labels must be a long tensor with shape N")
    if ignore_boxes is not None and (
        ignore_boxes.ndim != 2 or ignore_boxes.shape[1] != 4
    ):
        raise ValueError("ignore_boxes must have shape M,4")


def assign_fcos_targets(
    points: Tensor,
    point_strides: Tensor,
    level_ids: Tensor,
    boxes: Tensor,
    labels: Tensor,
    valid_size: Sequence[int] | Tensor,
    *,
    ignore_boxes: Tensor | None = None,
    size_ranges: Sequence[Sequence[float]] = DEFAULT_SIZE_RANGES,
    center_radius: float = 1.5,
    reg_max: int = 0,
) -> FCOSAssignment:
    """Assign one image's GT boxes to points with deterministic tie-breaking."""

    size_ranges = _size_ranges(size_ranges)
    center_radius = _non_negative_real(center_radius, "center_radius")
    if center_radius == 0:
        raise ValueError("center_radius must be positive")
    reg_max = _non_negative_int(reg_max, "reg_max")
    valid_height, valid_width = _pair(valid_size, "valid_size")

    device = points.device
    boxes = boxes.to(device=device, dtype=torch.float32)
    labels = labels.to(device=device, dtype=torch.long)
    if ignore_boxes is not None:
        ignore_boxes = ignore_boxes.to(device=device, dtype=torch.float32)
    _validate_assignment_inputs(
        points, point_strides, level_ids, boxes, labels, ignore_boxes
    )
    if not torch.isfinite(points).all() or not torch.isfinite(boxes).all():
        raise ValueError("points and boxes must be finite")
    if ignore_boxes is not None and not torch.isfinite(ignore_boxes).all():
        raise ValueError("ignore_boxes must be finite")
    if level_ids.numel() and (
        bool((level_ids < 0).any()) or bool((level_ids >= 3).any())
    ):
        raise ValueError("level_ids must contain only 0, 1, or 2")

    num_points = points.shape[0]
    num_targets = boxes.shape[0]
    assigned_labels = torch.full(
        (num_points,), -1, dtype=torch.long, device=device
    )
    assigned_boxes = points.new_zeros((num_points, 4), dtype=torch.float32)
    assigned_ltrb = points.new_zeros((num_points, 4), dtype=torch.float32)
    centerness = points.new_zeros((num_points,), dtype=torch.float32)
    matched_indices = torch.full(
        (num_points,), -1, dtype=torch.long, device=device
    )
    valid_points = (
        (points[:, 0] >= 0)
        & (points[:, 0] < valid_width)
        & (points[:, 1] >= 0)
        & (points[:, 1] < valid_height)
    )
    positive = torch.zeros(num_points, dtype=torch.bool, device=device)

    if num_targets:
        px = points[:, 0, None]
        py = points[:, 1, None]
        left = px - boxes[None, :, 0]
        top = py - boxes[None, :, 1]
        right = boxes[None, :, 2] - px
        bottom = boxes[None, :, 3] - py
        ltrb = torch.stack((left, top, right, bottom), dim=-1)
        inside_box = ltrb.amin(dim=-1) > 0

        centers_x = (boxes[:, 0] + boxes[:, 2]) * 0.5
        centers_y = (boxes[:, 1] + boxes[:, 3]) * 0.5
        radius = point_strides[:, None].float() * center_radius
        center_x1 = torch.maximum(boxes[None, :, 0], centers_x[None, :] - radius)
        center_y1 = torch.maximum(boxes[None, :, 1], centers_y[None, :] - radius)
        center_x2 = torch.minimum(boxes[None, :, 2], centers_x[None, :] + radius)
        center_y2 = torch.minimum(boxes[None, :, 3], centers_y[None, :] + radius)
        inside_center = (
            (px > center_x1)
            & (px < center_x2)
            & (py > center_y1)
            & (py < center_y2)
        )

        ranges = torch.tensor(size_ranges, device=device, dtype=torch.float32)
        point_ranges = ranges[level_ids]
        max_distance = ltrb.amax(dim=-1)
        in_level = (
            (max_distance >= point_ranges[:, None, 0])
            & (max_distance < point_ranges[:, None, 1])
        )
        candidate = inside_box & inside_center & in_level & valid_points[:, None]
        if reg_max > 0:
            cell_ltrb = ltrb / point_strides[:, None, None].float()
            candidate &= cell_ltrb.amax(dim=-1) <= float(reg_max)

        areas = (
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        )
        costs = torch.where(
            candidate,
            areas[None, :],
            torch.full_like(max_distance, math.inf),
        )
        minimum_area, best_target = costs.min(dim=1)
        positive = torch.isfinite(minimum_area)
        if positive.any():
            chosen = best_target[positive]
            matched_indices[positive] = chosen
            assigned_labels[positive] = labels[chosen]
            assigned_boxes[positive] = boxes[chosen]
            selected_ltrb = ltrb[positive, chosen]
            assigned_ltrb[positive] = selected_ltrb / point_strides[
                positive, None
            ].float()
            horizontal = selected_ltrb[:, (0, 2)]
            vertical = selected_ltrb[:, (1, 3)]
            horizontal_ratio = horizontal.amin(dim=1) / horizontal.amax(dim=1)
            vertical_ratio = vertical.amin(dim=1) / vertical.amax(dim=1)
            centerness[positive] = (horizontal_ratio * vertical_ratio).clamp_min(0).sqrt()

    if ignore_boxes is not None and ignore_boxes.shape[0]:
        px = points[:, 0, None]
        py = points[:, 1, None]
        inside_ignore = (
            (px > ignore_boxes[None, :, 0])
            & (px < ignore_boxes[None, :, 2])
            & (py > ignore_boxes[None, :, 1])
            & (py < ignore_boxes[None, :, 3])
        ).any(dim=1)
        valid_points = valid_points & ~(inside_ignore & ~positive)
    assigned_labels[~valid_points] = -2

    matched_target_count = (
        torch.unique(matched_indices[positive]).numel() if positive.any() else 0
    )
    return FCOSAssignment(
        labels=assigned_labels,
        boxes=assigned_boxes,
        ltrb_cells=assigned_ltrb,
        centerness=centerness,
        positive_mask=positive,
        valid_mask=valid_points,
        matched_gt_indices=matched_indices,
        num_targets=torch.tensor(num_targets, device=device, dtype=torch.long),
        num_unmatched=torch.tensor(
            num_targets - matched_target_count,
            device=device,
            dtype=torch.long,
        ),
    )


def _flatten_predictions(
    predictions: DetectionOutput,
    *,
    num_classes: int | None,
    reg_max: int,
) -> tuple[Tensor, Tensor, Tensor, tuple[tuple[int, int], ...]]:
    if not isinstance(predictions, DetectionOutput):
        raise TypeError("predictions must be a DetectionOutput")
    groups = (
        predictions.cls_logits,
        predictions.box_regression,
        predictions.quality,
    )
    if any(len(group) != 3 for group in groups):
        raise ValueError("predictions must contain exactly P3, P4, and P5")
    batch_size = predictions.cls_logits[0].shape[0]
    class_count = predictions.cls_logits[0].shape[1]
    if num_classes is not None and class_count != num_classes:
        raise ValueError(
            f"class prediction count must be {num_classes}, got {class_count}"
        )
    box_channels = 4 if reg_max == 0 else 4 * (reg_max + 1)
    feature_shapes: list[tuple[int, int]] = []
    flat_classes: list[Tensor] = []
    flat_boxes: list[Tensor] = []
    flat_quality: list[Tensor] = []
    for level in range(3):
        classes = predictions.cls_logits[level]
        boxes = predictions.box_regression[level]
        quality = predictions.quality[level]
        if classes.ndim != 4 or boxes.ndim != 4 or quality.ndim != 4:
            raise ValueError("all detection maps must use NCHW layout")
        shape = tuple(classes.shape[-2:])
        if (
            classes.shape[0] != batch_size
            or boxes.shape[0] != batch_size
            or quality.shape[0] != batch_size
            or classes.shape[1] != class_count
            or boxes.shape[1] != box_channels
            or quality.shape[1] != 1
            or tuple(boxes.shape[-2:]) != shape
            or tuple(quality.shape[-2:]) != shape
        ):
            raise ValueError(f"inconsistent prediction contract at level {level}")
        height, width = shape
        feature_shapes.append((height, width))
        flat_classes.append(
            classes.permute(0, 2, 3, 1).reshape(batch_size, -1, class_count)
        )
        if reg_max == 0:
            flat_boxes.append(
                boxes.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
            )
        else:
            flat_boxes.append(
                boxes.reshape(batch_size, 4, reg_max + 1, height, width)
                .permute(0, 3, 4, 1, 2)
                .reshape(batch_size, -1, 4, reg_max + 1)
            )
        flat_quality.append(
            quality.permute(0, 2, 3, 1).reshape(batch_size, -1)
        )
    return (
        torch.cat(flat_classes, dim=1),
        torch.cat(flat_boxes, dim=1),
        torch.cat(flat_quality, dim=1),
        tuple(feature_shapes),
    )


def _sigmoid_focal_loss(
    logits: Tensor,
    targets: Tensor,
    *,
    alpha: float,
    gamma: float,
) -> Tensor:
    cross_entropy = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    probabilities = logits.sigmoid()
    probability_target = probabilities * targets + (1 - probabilities) * (1 - targets)
    alpha_target = alpha * targets + (1 - alpha) * (1 - targets)
    return alpha_target * (1 - probability_target).pow(gamma) * cross_entropy


def _validate_target(
    target: Mapping[str, Tensor],
    *,
    num_classes: int,
    fallback_size: tuple[int, int] | None,
    device: torch.device,
) -> tuple[Tensor, Tensor, tuple[int, int], Tensor | None]:
    if not isinstance(target, Mapping):
        raise TypeError("each detection target must be a mapping")
    if "boxes" not in target or "labels" not in target:
        raise ValueError("each detection target requires boxes and labels")
    boxes = torch.as_tensor(target["boxes"], device=device, dtype=torch.float32)
    labels = torch.as_tensor(target["labels"], device=device)
    if labels.dtype != torch.long:
        raise ValueError("target labels must use torch.long")
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("target boxes must have shape N,4")
    if labels.shape != (boxes.shape[0],):
        raise ValueError("target labels must have shape N")
    if "valid_size" in target:
        valid_size = _pair(target["valid_size"], "target valid_size")
    elif fallback_size is not None:
        valid_size = fallback_size
    else:
        raise ValueError("image_size or target valid_size is required")
    ignore_boxes = None
    if "ignore_boxes" in target:
        ignore_boxes = torch.as_tensor(
            target["ignore_boxes"], device=device, dtype=torch.float32
        )
        if ignore_boxes.ndim != 2 or ignore_boxes.shape[1] != 4:
            raise ValueError("target ignore_boxes must have shape M,4")

    if not torch.isfinite(boxes).all():
        raise ValueError("target boxes must be finite")
    if boxes.shape[0]:
        if bool((boxes[:, 2:] <= boxes[:, :2]).any()):
            raise ValueError("target boxes must have positive width and height")
        height, width = valid_size
        if bool((boxes[:, :2] < 0).any()) or bool(
            (boxes[:, 2] > width).any() | (boxes[:, 3] > height).any()
        ):
            raise ValueError("target boxes must lie within valid_size")
        if bool((labels < 0).any()) or bool((labels >= num_classes).any()):
            raise ValueError("target labels are outside the configured class range")
    return boxes, labels, valid_size, ignore_boxes


class DetectionCriterion(nn.Module):
    """FCOS focal/GIoU/centerness criterion with optional DFL regression."""

    def __init__(
        self,
        num_classes: int,
        *,
        reg_max: int = 0,
        strides: Sequence[int] = DEFAULT_STRIDES,
        size_ranges: Sequence[Sequence[float]] = DEFAULT_SIZE_RANGES,
        center_radius: float = 1.5,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        cls_weight: float = 1.0,
        box_weight: float = 2.5,
        quality_weight: float = 1.0,
        dfl_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_classes = _positive_int(num_classes, "num_classes")
        self.reg_max = _non_negative_int(reg_max, "reg_max")
        self.strides = _strides(strides)
        self.size_ranges = _size_ranges(size_ranges)
        self.center_radius = _non_negative_real(center_radius, "center_radius")
        if self.center_radius == 0:
            raise ValueError("center_radius must be positive")
        self.focal_alpha = _unit_interval(
            focal_alpha, "focal_alpha", inclusive_zero=True
        )
        self.focal_gamma = _non_negative_real(focal_gamma, "focal_gamma")
        self.cls_weight = _non_negative_real(cls_weight, "cls_weight")
        self.box_weight = _non_negative_real(box_weight, "box_weight")
        self.quality_weight = _non_negative_real(
            quality_weight, "quality_weight"
        )
        self.dfl_weight = _non_negative_real(dfl_weight, "dfl_weight")

    def forward(
        self,
        predictions: DetectionOutput,
        targets: Sequence[Mapping[str, Tensor]],
        image_size: Sequence[int] | None = None,
    ) -> DetectionLosses:
        fallback_size = None if image_size is None else _pair(image_size, "image_size")
        classes, boxes, quality, feature_shapes = _flatten_predictions(
            predictions,
            num_classes=self.num_classes,
            reg_max=self.reg_max,
        )
        batch_size = classes.shape[0]
        try:
            targets = tuple(targets)
        except TypeError as exc:
            raise ValueError("targets must contain one mapping per image") from exc
        if len(targets) != batch_size:
            raise ValueError("targets length must match prediction batch size")
        if fallback_size is not None:
            height, width = fallback_size
            expected_shapes = tuple(
                (
                    (height + stride - 1) // stride,
                    (width + stride - 1) // stride,
                )
                for stride in self.strides
            )
            if feature_shapes != expected_shapes:
                raise ValueError(
                    f"feature shapes must be {expected_shapes} for image_size "
                    f"{fallback_size}, got {feature_shapes}"
                )

        point_data = make_detection_points(
            feature_shapes,
            strides=self.strides,
            device=classes.device,
        )
        assignments: list[FCOSAssignment] = []
        for target in targets:
            target_boxes, target_labels, valid_size, ignore_boxes = _validate_target(
                target,
                num_classes=self.num_classes,
                fallback_size=fallback_size,
                device=classes.device,
            )
            assignments.append(
                assign_fcos_targets(
                    point_data.points,
                    point_data.strides,
                    point_data.levels,
                    target_boxes,
                    target_labels,
                    valid_size,
                    ignore_boxes=ignore_boxes,
                    size_ranges=self.size_ranges,
                    center_radius=self.center_radius,
                    reg_max=self.reg_max,
                )
            )

        assigned_labels = torch.stack([item.labels for item in assignments])
        assigned_boxes = torch.stack([item.boxes for item in assignments])
        assigned_ltrb = torch.stack([item.ltrb_cells for item in assignments])
        assigned_centerness = torch.stack(
            [item.centerness for item in assignments]
        )
        positive = torch.stack([item.positive_mask for item in assignments])
        valid = torch.stack([item.valid_mask for item in assignments])

        class_logits = classes.float()
        class_targets = torch.zeros_like(class_logits)
        positive_indices = positive.nonzero(as_tuple=True)
        if positive_indices[0].numel():
            class_targets[
                positive_indices[0],
                positive_indices[1],
                assigned_labels[positive],
            ] = 1.0
        local_positive = positive.sum().float()
        positive_normalizer = local_positive.clamp_min(1.0)
        classification_loss = _sigmoid_focal_loss(
            class_logits,
            class_targets,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
        )
        classification_loss = (
            classification_loss * valid[..., None]
        ).sum() / positive_normalizer

        connected_box_zero = boxes.float().sum() * 0.0
        connected_quality_zero = quality.float().sum() * 0.0
        if positive.any():
            positive_boxes = boxes[positive]
            positive_points = point_data.points[None].expand(
                batch_size, -1, -1
            )[positive]
            positive_strides = point_data.strides[None].expand(
                batch_size, -1
            )[positive]
            decoded_boxes = decode_box_regression(
                positive_boxes,
                positive_points,
                positive_strides,
                reg_max=self.reg_max,
            )
            center_weights = assigned_centerness[positive].float()
            center_normalizer = center_weights.sum().clamp_min(1.0)
            box_loss = (
                (1.0 - generalized_box_iou_aligned(
                    decoded_boxes, assigned_boxes[positive]
                ))
                * center_weights
            ).sum() / center_normalizer + connected_box_zero
            quality_loss = F.binary_cross_entropy_with_logits(
                quality.float()[positive],
                center_weights,
                reduction="sum",
            ) / positive_normalizer + connected_quality_zero
            if self.reg_max > 0:
                target_distances = assigned_ltrb[positive].float()
                if bool((target_distances > self.reg_max).any()):
                    raise RuntimeError(
                        "assigned DFL target exceeds representable reg_max"
                    )
                lower = target_distances.floor().long()
                upper = (lower + 1).clamp(max=self.reg_max)
                upper_weight = target_distances - lower.float()
                lower_weight = 1.0 - upper_weight
                logits = positive_boxes.float().reshape(-1, self.reg_max + 1)
                lower_loss = F.cross_entropy(
                    logits, lower.reshape(-1), reduction="none"
                ).reshape_as(target_distances)
                upper_loss = F.cross_entropy(
                    logits, upper.reshape(-1), reduction="none"
                ).reshape_as(target_distances)
                distribution_loss = (
                    (lower_loss * lower_weight + upper_loss * upper_weight).sum(dim=1)
                    * center_weights
                ).sum() / center_normalizer + connected_box_zero
            else:
                distribution_loss = connected_box_zero
        else:
            box_loss = connected_box_zero
            quality_loss = connected_quality_zero
            distribution_loss = connected_box_zero

        total = (
            self.cls_weight * classification_loss
            + self.box_weight * box_loss
            + self.quality_weight * quality_loss
            + self.dfl_weight * distribution_loss
        )
        return DetectionLosses(
            total=total,
            classification=classification_loss,
            box=box_loss,
            quality=quality_loss,
            dfl=distribution_loss,
            num_positive=local_positive.detach(),
            num_targets=torch.stack(
                [item.num_targets for item in assignments]
            ).sum().detach(),
            num_unmatched=torch.stack(
                [item.num_unmatched for item in assignments]
            ).sum().detach(),
        )


def nms(boxes: Tensor, scores: Tensor, iou_threshold: float) -> Tensor:
    """Pure-Torch greedy NMS with stable score-tie behavior."""

    iou_threshold = _unit_interval(
        iou_threshold, "iou_threshold", inclusive_zero=True
    )
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must have shape N,4")
    if scores.shape != (boxes.shape[0],):
        raise ValueError("scores must have shape N")
    if boxes.shape[0] == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    boxes = boxes.float()
    scores = scores.float()
    order = torch.argsort(scores, descending=True, stable=True)
    keep: list[Tensor] = []
    while order.numel():
        index = order[0]
        keep.append(index)
        if order.numel() == 1:
            break
        remainder = order[1:]
        overlaps = box_iou(boxes[index : index + 1], boxes[remainder]).squeeze(0)
        order = remainder[overlaps <= iou_threshold]
    return torch.stack(keep)


def _accelerated_class_aware_nms(
    boxes: Tensor,
    scores: Tensor,
    labels: Tensor,
    iou_threshold: float,
) -> Tensor:
    """Run torchvision NMS with repository-stable score ordering."""

    if _torchvision_batched_nms is None:
        raise RuntimeError("torchvision batched_nms is unavailable")
    stable_order = torch.argsort(scores.float(), descending=True, stable=True)
    rank_scores = torch.empty_like(scores, dtype=torch.float32)
    rank_scores[stable_order] = torch.arange(
        boxes.shape[0],
        0,
        -1,
        device=boxes.device,
        dtype=torch.float32,
    )
    return _torchvision_batched_nms(
        boxes.float(),
        rank_scores,
        labels,
        iou_threshold,
    )


def class_aware_nms(
    boxes: Tensor,
    scores: Tensor,
    labels: Tensor,
    iou_threshold: float,
) -> Tensor:
    """Apply one pure-Torch NMS pass while separating different classes."""

    if labels.shape != (boxes.shape[0],) or labels.dtype != torch.long:
        raise ValueError("labels must be a long tensor with shape N")
    if boxes.shape[0] == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    boxes_float = boxes.float()
    if boxes.is_cuda and _torchvision_batched_nms is not None:
        # torchvision's CUDA kernel removes the Python/kernel-launch loop from
        # validation. Give it unique rank scores so equal input scores preserve
        # the repository's stable, first-occurrence tie contract on every
        # backend. NMS only consumes score order, not score magnitudes.
        return _accelerated_class_aware_nms(
            boxes_float, scores, labels, iou_threshold
        )
    span = (boxes_float.max() - boxes_float.min()).clamp_min(0) + 1.0
    offsets = labels.to(dtype=torch.float32) * span
    return nms(boxes_float + offsets[:, None], scores.float(), iou_threshold)


def decode_detections(
    predictions: DetectionOutput,
    image_sizes: Sequence[Sequence[int] | Tensor],
    *,
    reg_max: int,
    strides: Sequence[int] = DEFAULT_STRIDES,
    score_threshold: float = 0.05,
    nms_iou_threshold: float = 0.6,
    pre_nms_topk: int = 1000,
    max_detections: int = 300,
    min_box_size: float = 0.0,
) -> list[dict[str, Tensor]]:
    """Decode a batch and apply score filtering plus class-aware NMS."""

    reg_max = _non_negative_int(reg_max, "reg_max")
    strides = _strides(strides)
    score_threshold = _unit_interval(
        score_threshold, "score_threshold", inclusive_zero=True
    )
    nms_iou_threshold = _unit_interval(
        nms_iou_threshold, "nms_iou_threshold", inclusive_zero=True
    )
    pre_nms_topk = _positive_int(pre_nms_topk, "pre_nms_topk")
    max_detections = _positive_int(max_detections, "max_detections")
    min_box_size = _non_negative_real(min_box_size, "min_box_size")
    classes, boxes, quality, feature_shapes = _flatten_predictions(
        predictions,
        num_classes=None,
        reg_max=reg_max,
    )
    try:
        image_sizes = tuple(image_sizes)
    except TypeError as exc:
        raise ValueError("image_sizes must contain one size per image") from exc
    if len(image_sizes) != classes.shape[0]:
        raise ValueError("image_sizes length must match prediction batch size")
    checked_sizes = tuple(
        _pair(size, f"image_sizes[{index}]")
        for index, size in enumerate(image_sizes)
    )
    point_data = make_detection_points(
        feature_shapes,
        strides=strides,
        device=classes.device,
    )
    class_probabilities = classes.float().sigmoid()
    quality_probabilities = quality.float().sigmoid()
    scores = (class_probabilities * quality_probabilities[..., None]).clamp_min(0).sqrt()

    results: list[dict[str, Tensor]] = []
    num_classes = classes.shape[-1]
    for batch_index, (height, width) in enumerate(checked_sizes):
        decoded = decode_box_regression(
            boxes[batch_index],
            point_data.points,
            point_data.strides,
            reg_max=reg_max,
        )
        decoded[:, 0::2] = decoded[:, 0::2].clamp(0, width)
        decoded[:, 1::2] = decoded[:, 1::2].clamp(0, height)
        flat_scores = scores[batch_index].reshape(-1)
        candidate = torch.isfinite(flat_scores) & (flat_scores >= score_threshold)
        candidate_indices = candidate.nonzero(as_tuple=False).squeeze(1)
        if candidate_indices.numel() > pre_nms_topk:
            candidate_scores = flat_scores[candidate_indices]
            top_indices = torch.argsort(
                candidate_scores,
                descending=True,
                stable=True,
            )[:pre_nms_topk]
            candidate_indices = candidate_indices[top_indices]
        selected_scores = flat_scores[candidate_indices]
        selected_points = torch.div(
            candidate_indices, num_classes, rounding_mode="floor"
        )
        selected_labels = torch.remainder(candidate_indices, num_classes).long()
        selected_boxes = decoded[selected_points]
        sizes = selected_boxes[:, 2:] - selected_boxes[:, :2]
        keep_size = (sizes[:, 0] > min_box_size) & (sizes[:, 1] > min_box_size)
        selected_boxes = selected_boxes[keep_size]
        selected_scores = selected_scores[keep_size]
        selected_labels = selected_labels[keep_size]
        if selected_boxes.shape[0]:
            keep = class_aware_nms(
                selected_boxes,
                selected_scores,
                selected_labels,
                nms_iou_threshold,
            )[:max_detections]
            selected_boxes = selected_boxes[keep]
            selected_scores = selected_scores[keep]
            selected_labels = selected_labels[keep]
        results.append(
            {
                "boxes": selected_boxes,
                "scores": selected_scores,
                "labels": selected_labels,
            }
        )
    return results


__all__ = [
    "DEFAULT_SIZE_RANGES",
    "DEFAULT_STRIDES",
    "DetectionCriterion",
    "DetectionLosses",
    "DetectionPoints",
    "FCOSAssignment",
    "assign_fcos_targets",
    "box_iou",
    "class_aware_nms",
    "decode_box_regression",
    "decode_detections",
    "generalized_box_iou_aligned",
    "make_detection_points",
    "nms",
]
