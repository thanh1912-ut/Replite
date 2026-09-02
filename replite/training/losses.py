"""Task losses and composition for RepLite training.

The perception model intentionally returns raw predictions and has no target
schema.  This module defines a small, explicit training contract while keeping
partial supervision visible: an active prediction requires a matching target
key, and a value of ``None`` means that the task is deliberately unsupervised
for that batch.  Per-sample or per-pixel validity masks provide finer-grained
partial supervision for dense tasks.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from replite.multitask.heads import DetectionOutput
from replite.multitask.model import RepLiteOutput


_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}
_TASK_NAMES = ("detection", "segmentation", "depth", "classification")


def _differentiable_zero(*tensors: Tensor) -> Tensor:
    if not tensors:
        raise ValueError("at least one tensor is required to construct zero loss")
    zero = tensors[0].sum() * 0.0
    for tensor in tensors[1:]:
        zero = zero + tensor.sum() * 0.0
    return zero


def _validate_class_weights(
    class_weights: Tensor | None,
    *,
    num_classes: int,
    reference: Tensor,
) -> Tensor | None:
    if class_weights is None:
        return None
    if not isinstance(class_weights, Tensor):
        raise TypeError("class_weights must be a torch.Tensor or None")
    if class_weights.ndim != 1 or class_weights.shape[0] != num_classes:
        raise ValueError(f"class_weights must have shape ({num_classes},)")
    if not class_weights.is_floating_point():
        raise TypeError("class_weights must use a floating-point dtype")
    if class_weights.device != reference.device:
        raise ValueError("class_weights and predictions must be on the same device")
    weights = class_weights.to(dtype=reference.dtype)
    if not bool(torch.isfinite(weights).all()):
        raise ValueError("class_weights must contain only finite values")
    if bool((weights < 0).any()) or not bool((weights > 0).any()):
        raise ValueError("class_weights must be non-negative with a positive entry")
    return weights


def inverse_sqrt_class_weights(
    pixel_counts: Tensor | Sequence[int | float],
    *,
    min_weight: float = 0.25,
    max_weight: float = 5.0,
    eps: float = 1e-12,
) -> Tensor:
    """Build bounded inverse-square-root class weights from fit-only counts.

    The returned weights satisfy ``sum(frequency * weight) == 1`` (up to
    floating-point precision) while remaining in ``[min_weight, max_weight]``.
    Zero-count classes receive the upper bound but do not affect the weighted
    mean. This helper is intentionally independent of a dataset split; callers
    should obtain ``pixel_counts`` from fit samples only.
    """

    if isinstance(pixel_counts, (str, bytes)):
        raise TypeError("pixel_counts must be a one-dimensional numeric sequence")
    try:
        counts = torch.as_tensor(pixel_counts, dtype=torch.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "pixel_counts must be a one-dimensional numeric sequence"
        ) from exc
    if counts.ndim != 1 or counts.numel() < 2:
        raise ValueError("pixel_counts must contain at least two classes")
    if not bool(torch.isfinite(counts).all()) or bool((counts < 0).any()):
        raise ValueError("pixel_counts must be finite and non-negative")
    if not bool(counts.sum() > 0):
        raise ValueError("pixel_counts must contain at least one labeled pixel")
    for value, name in (
        (min_weight, "min_weight"),
        (max_weight, "max_weight"),
        (eps, "eps"),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ValueError(f"{name} must be finite and positive")
    if float(min_weight) > 1.0 or float(max_weight) < 1.0:
        raise ValueError("weight bounds must satisfy min_weight <= 1 <= max_weight")
    if float(min_weight) > float(max_weight):
        raise ValueError("min_weight must not exceed max_weight")

    frequencies = counts / counts.sum()
    raw = torch.rsqrt(frequencies + float(eps))

    # Find the scale of the inverse-sqrt curve whose clipped, frequency-
    # weighted mean is one. Bisection makes both normalization and hard bounds
    # exact, unlike a final re-normalization that can violate the clip limits.
    lower = raw.new_zeros(())
    upper = raw.new_tensor(float(max_weight)) / raw.min()
    for _ in range(80):
        middle = (lower + upper) * 0.5
        candidate = (raw * middle).clamp(float(min_weight), float(max_weight))
        if bool((frequencies * candidate).sum() < 1.0):
            lower = middle
        else:
            upper = middle
    result = (raw * upper).clamp(float(min_weight), float(max_weight))
    return result.to(dtype=torch.float32)


def _valid_mask_like(
    valid_mask: Tensor | None,
    target: Tensor,
    *,
    name: str,
) -> Tensor:
    """Broadcast a boolean batch/pixel mask exactly to ``target.shape``."""

    if valid_mask is None:
        return torch.ones_like(target, dtype=torch.bool)
    if not isinstance(valid_mask, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor or None")
    if valid_mask.dtype != torch.bool:
        raise TypeError(f"{name} must use torch.bool dtype")
    if valid_mask.device != target.device:
        raise ValueError(f"{name} and target must be on the same device")

    if tuple(valid_mask.shape) == tuple(target.shape):
        return valid_mask
    if valid_mask.ndim == 1 and valid_mask.shape[0] == target.shape[0]:
        view_shape = (target.shape[0],) + (1,) * (target.ndim - 1)
        return valid_mask.reshape(view_shape).expand_as(target)
    if (
        target.ndim == 4
        and target.shape[1] == 1
        and valid_mask.ndim == 3
        and tuple(valid_mask.shape)
        == (target.shape[0], target.shape[2], target.shape[3])
    ):
        return valid_mask.unsqueeze(1)
    if (
        target.ndim == 3
        and valid_mask.ndim == 4
        and valid_mask.shape[1] == 1
        and tuple(valid_mask.shape[0:1] + valid_mask.shape[2:])
        == tuple(target.shape)
    ):
        return valid_mask[:, 0]
    raise ValueError(
        f"{name} must have target shape {tuple(target.shape)} or batch shape "
        f"({target.shape[0]},)"
    )


def _lovasz_gradient(sorted_foreground: Tensor) -> Tensor:
    """Gradient of the Lovasz extension for one sorted binary foreground."""

    count = sorted_foreground.numel()
    foreground_total = sorted_foreground.sum()
    intersection = foreground_total - sorted_foreground.cumsum(0)
    union = foreground_total + (1.0 - sorted_foreground).cumsum(0)
    gradient = 1.0 - intersection / union.clamp_min(1.0)
    if count > 1:
        gradient = torch.cat((gradient[:1], gradient[1:] - gradient[:-1]))
    return gradient


def lovasz_softmax_loss(
    logits: Tensor,
    target: Tensor,
    *,
    ignore_index: int = 255,
    valid_mask: Tensor | None = None,
) -> Tensor:
    """Batch-flat Lovasz-Softmax loss over classes present in the target.

    The implementation follows the multiclass Lovasz extension and excludes
    ignored/invalid pixels before sorting. Empty supervision returns a
    differentiable zero connected to ``logits``.
    """

    if not isinstance(logits, Tensor) or not isinstance(target, Tensor):
        raise TypeError("logits and target must be torch.Tensor objects")
    if logits.ndim != 4 or not logits.is_floating_point():
        raise ValueError("segmentation logits must be floating B,C,H,W tensors")
    if target.ndim == 4 and target.shape[1] == 1:
        target = target[:, 0]
    expected = (logits.shape[0], logits.shape[2], logits.shape[3])
    if target.ndim != 3 or tuple(target.shape) != expected:
        raise ValueError(f"segmentation target must have shape {expected}")
    if target.dtype not in _INTEGER_DTYPES:
        raise TypeError("segmentation target must use an integer dtype")
    if not isinstance(ignore_index, Integral) or isinstance(ignore_index, bool):
        raise TypeError("ignore_index must be an integer")

    mask = _valid_mask_like(valid_mask, target, name="valid_mask")
    mask = mask & target.ne(int(ignore_index))
    labels = target[mask].long()
    if labels.numel() == 0:
        return _differentiable_zero(logits)
    if bool((labels < 0).any()) or bool((labels >= logits.shape[1]).any()):
        raise ValueError("valid segmentation labels must be in [0, num_classes)")

    probabilities = logits.float().softmax(dim=1).permute(0, 2, 3, 1)[mask]
    losses: list[Tensor] = []
    for class_index in range(logits.shape[1]):
        foreground = labels.eq(class_index).to(dtype=probabilities.dtype)
        if not bool(foreground.any()):
            continue
        errors = (foreground - probabilities[:, class_index]).abs()
        sorted_errors, permutation = torch.sort(errors, descending=True)
        sorted_foreground = foreground[permutation]
        losses.append(
            torch.dot(sorted_errors, _lovasz_gradient(sorted_foreground))
        )
    if not losses:
        return _differentiable_zero(logits)
    return torch.stack(losses).mean()


def segmentation_loss(
    logits: Tensor,
    target: Tensor,
    *,
    ignore_index: int = 255,
    valid_mask: Tensor | None = None,
    class_weights: Tensor | None = None,
    label_smoothing: float = 0.0,
    normalize_weighted_loss: bool = False,
    lovasz_weight: float = 0.0,
) -> Tensor:
    """Masked multiclass cross-entropy for semantic segmentation.

    ``valid_mask`` may have shape ``B``, ``B,H,W``, or ``B,1,H,W``. Pixels
    equal to ``ignore_index`` are excluded independently. If no valid pixels
    remain, the result is a differentiable scalar zero connected to ``logits``.
    """

    if not isinstance(logits, Tensor) or not isinstance(target, Tensor):
        raise TypeError("logits and target must be torch.Tensor objects")
    if logits.ndim != 4:
        raise ValueError("segmentation logits must have shape B,C,H,W")
    if not logits.is_floating_point():
        raise TypeError("segmentation logits must use a floating-point dtype")
    if target.ndim == 4 and target.shape[1] == 1:
        target = target[:, 0]
    if target.ndim != 3:
        raise ValueError("segmentation target must have shape B,H,W or B,1,H,W")
    if target.dtype not in _INTEGER_DTYPES:
        raise TypeError("segmentation target must use an integer dtype")
    expected = (logits.shape[0], logits.shape[2], logits.shape[3])
    if tuple(target.shape) != expected:
        raise ValueError(
            f"segmentation target must have shape {expected}, got {tuple(target.shape)}"
        )
    if not isinstance(ignore_index, Integral) or isinstance(ignore_index, bool):
        raise TypeError("ignore_index must be an integer")
    if (
        isinstance(label_smoothing, bool)
        or not isinstance(label_smoothing, Real)
        or not math.isfinite(float(label_smoothing))
        or not 0.0 <= float(label_smoothing) < 1.0
    ):
        raise ValueError("label_smoothing must be finite and in [0, 1)")
    if not isinstance(normalize_weighted_loss, bool):
        raise TypeError("normalize_weighted_loss must be a boolean")
    if (
        isinstance(lovasz_weight, bool)
        or not isinstance(lovasz_weight, Real)
        or not math.isfinite(float(lovasz_weight))
        or float(lovasz_weight) < 0.0
    ):
        raise ValueError("lovasz_weight must be finite and non-negative")

    mask = _valid_mask_like(valid_mask, target, name="valid_mask")
    mask = mask & target.ne(int(ignore_index))
    selected = target[mask]
    if selected.numel() > 0 and (
        bool((selected < 0).any()) or bool((selected >= logits.shape[1]).any())
    ):
        raise ValueError("valid segmentation labels must be in [0, num_classes)")
    if not bool(mask.any()):
        return _differentiable_zero(logits)

    safe_target = torch.where(mask, target, torch.zeros_like(target)).long()
    weights = _validate_class_weights(
        class_weights,
        num_classes=logits.shape[1],
        reference=logits,
    )
    per_pixel = F.cross_entropy(
        logits,
        safe_target,
        weight=weights,
        reduction="none",
        label_smoothing=float(label_smoothing),
    )
    if weights is not None and normalize_weighted_loss:
        denominator = weights[safe_target][mask].sum()
        if not bool(denominator > 0):
            raise ValueError("selected class weights must have a positive sum")
        cross_entropy = per_pixel[mask].sum() / denominator
    else:
        cross_entropy = per_pixel[mask].mean()
    if float(lovasz_weight) == 0.0:
        return cross_entropy
    lovasz = lovasz_softmax_loss(
        logits,
        target,
        ignore_index=int(ignore_index),
        valid_mask=valid_mask,
    )
    return cross_entropy + float(lovasz_weight) * lovasz


def masked_depth_loss(
    prediction: Tensor,
    target: Tensor,
    *,
    valid_mask: Tensor | None = None,
    min_depth: float = 0.0,
    max_depth: float | None = None,
    loss_type: str = "l1",
    smooth_l1_beta: float = 1.0,
    log_l1_weight: float = 1.0,
    silog_weight: float = 1.0,
    gradient_weight: float = 0.0,
    silog_lambda: float = 0.5,
    eps: float = 1e-6,
) -> Tensor:
    """Compute depth loss over valid finite target pixels only.

    Targets must be strictly greater than ``min_depth`` and, when supplied, no
    greater than ``max_depth``. Supported losses are ``l1``, ``smooth_l1``, and
    ``log_l1``, ``log_l1_silog``, and
    ``per_image_silog_log_l1_gradient``. The final mode averages each component
    per image, then combines SiLog, absolute log error, and valid-neighbour log
    gradient error. Depth arithmetic is always FP32 under autocast. An empty
    valid set returns differentiable zero.
    """

    if not isinstance(prediction, Tensor) or not isinstance(target, Tensor):
        raise TypeError("prediction and target must be torch.Tensor objects")
    if prediction.ndim != 4 or prediction.shape[1] != 1:
        raise ValueError("depth prediction must have shape B,1,H,W")
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise TypeError("depth prediction and target must use floating-point dtypes")
    if target.ndim == 3:
        target = target.unsqueeze(1)
    if target.ndim != 4 or target.shape[1] != 1:
        raise ValueError("depth target must have shape B,H,W or B,1,H,W")
    if tuple(target.shape) != tuple(prediction.shape):
        raise ValueError(
            "depth target and prediction must have identical batch/spatial shapes"
        )
    if target.device != prediction.device:
        raise ValueError("depth target and prediction must be on the same device")
    if (
        isinstance(min_depth, bool)
        or not isinstance(min_depth, Real)
        or not math.isfinite(float(min_depth))
        or float(min_depth) < 0.0
    ):
        raise ValueError("min_depth must be a finite non-negative number")
    if max_depth is not None and (
        isinstance(max_depth, bool)
        or not isinstance(max_depth, Real)
        or not math.isfinite(float(max_depth))
        or float(max_depth) <= float(min_depth)
    ):
        raise ValueError("max_depth must be finite and greater than min_depth")
    if loss_type not in {
        "l1",
        "smooth_l1",
        "log_l1",
        "log_l1_silog",
        "per_image_silog_log_l1_gradient",
    }:
        raise ValueError(
            "loss_type must be 'l1', 'smooth_l1', 'log_l1', or "
            "'log_l1_silog', or 'per_image_silog_log_l1_gradient'"
        )
    if (
        isinstance(smooth_l1_beta, bool)
        or not isinstance(smooth_l1_beta, Real)
        or not math.isfinite(float(smooth_l1_beta))
        or float(smooth_l1_beta) <= 0.0
    ):
        raise ValueError("smooth_l1_beta must be a finite positive number")
    for value, name in (
        (log_l1_weight, "log_l1_weight"),
        (silog_weight, "silog_weight"),
        (gradient_weight, "gradient_weight"),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f"{name} must be finite and non-negative")
    if (
        isinstance(silog_lambda, bool)
        or not isinstance(silog_lambda, Real)
        or not math.isfinite(float(silog_lambda))
        or not 0.0 <= float(silog_lambda) <= 1.0
    ):
        raise ValueError("silog_lambda must be finite and in [0, 1]")
    if (
        isinstance(eps, bool)
        or not isinstance(eps, Real)
        or not math.isfinite(float(eps))
        or float(eps) <= 0.0
    ):
        raise ValueError("eps must be a finite positive number")

    mask = _valid_mask_like(valid_mask, target, name="valid_mask")
    mask = mask & torch.isfinite(target) & target.gt(float(min_depth))
    if max_depth is not None:
        mask = mask & target.le(float(max_depth))
    if not bool(mask.any()):
        return _differentiable_zero(prediction)

    selected_prediction = prediction[mask]
    selected_target = target[mask]
    if not bool(torch.isfinite(selected_prediction).all()):
        raise ValueError("depth prediction contains non-finite values at valid pixels")

    # Depth ratios and logarithms are numerically fragile in FP16/BF16. Cast
    # explicitly instead of depending on a caller's autocast configuration.
    selected_prediction = selected_prediction.float()
    selected_target = selected_target.float()
    if loss_type == "l1":
        return F.l1_loss(selected_prediction, selected_target)
    if loss_type == "smooth_l1":
        return F.smooth_l1_loss(
            selected_prediction,
            selected_target,
            beta=float(smooth_l1_beta),
        )
    if bool((selected_prediction <= 0).any()):
        raise ValueError("log_l1 requires positive predictions at valid pixels")
    if loss_type == "per_image_silog_log_l1_gradient":
        prediction_fp32 = prediction.float()
        target_fp32 = target.float()
        safe_prediction = torch.where(
            mask, prediction_fp32.clamp_min(float(eps)), torch.ones_like(prediction_fp32)
        )
        safe_target = torch.where(
            mask, target_fp32.clamp_min(float(eps)), torch.ones_like(target_fp32)
        )
        log_prediction = safe_prediction.log()
        log_target = safe_target.log()
        difference = log_prediction - log_target
        mask_fp32 = mask.to(dtype=torch.float32)
        counts = mask_fp32.flatten(1).sum(1)
        denominator = counts.clamp_min(1.0)
        mean_difference = (difference * mask_fp32).flatten(1).sum(1) / denominator
        mean_squared = (
            difference.square() * mask_fp32
        ).flatten(1).sum(1) / denominator
        # Subtract sqrt(eps) so an exact prediction has exact zero loss while
        # retaining a finite derivative close to zero variance.
        silog_variance = (
            mean_squared - float(silog_lambda) * mean_difference.square()
        ).clamp_min(0.0)
        silog = (silog_variance + float(eps)).sqrt() - math.sqrt(float(eps))
        log_l1 = (difference.abs() * mask_fp32).flatten(1).sum(1) / denominator

        horizontal_mask = mask[..., :, 1:] & mask[..., :, :-1]
        vertical_mask = mask[..., 1:, :] & mask[..., :-1, :]
        horizontal_error = (
            (log_prediction[..., :, 1:] - log_prediction[..., :, :-1])
            - (log_target[..., :, 1:] - log_target[..., :, :-1])
        ).abs()
        vertical_error = (
            (log_prediction[..., 1:, :] - log_prediction[..., :-1, :])
            - (log_target[..., 1:, :] - log_target[..., :-1, :])
        ).abs()
        pair_count = (
            horizontal_mask.flatten(1).sum(1) + vertical_mask.flatten(1).sum(1)
        )
        gradient = (
            (horizontal_error * horizontal_mask).flatten(1).sum(1)
            + (vertical_error * vertical_mask).flatten(1).sum(1)
        ) / pair_count.clamp_min(1)
        per_image = (
            float(silog_weight) * silog
            + float(log_l1_weight) * log_l1
            + float(gradient_weight) * gradient
        )
        return per_image[counts > 0].mean()
    log_difference = torch.log(selected_prediction.clamp_min(float(eps))) - torch.log(
        selected_target.clamp_min(float(eps))
    )
    log_l1 = log_difference.abs().mean()
    if loss_type == "log_l1":
        return log_l1
    scale_invariant = (
        log_difference.square().mean()
        - 0.5 * log_difference.mean().square()
    ).clamp_min(0.0)
    return (
        float(log_l1_weight) * log_l1
        + float(silog_weight) * scale_invariant
    )


def classification_loss(
    logits: Tensor,
    target: Tensor,
    *,
    ignore_index: int = -100,
    valid_mask: Tensor | None = None,
    class_weights: Tensor | None = None,
    label_smoothing: float = 0.0,
) -> Tensor:
    """Masked multiclass cross-entropy for image classification."""

    if not isinstance(logits, Tensor) or not isinstance(target, Tensor):
        raise TypeError("logits and target must be torch.Tensor objects")
    if logits.ndim != 2:
        raise ValueError("classification logits must have shape B,C")
    if not logits.is_floating_point():
        raise TypeError("classification logits must use a floating-point dtype")
    if target.ndim == 2 and target.shape[1] == 1:
        target = target[:, 0]
    if target.ndim != 1 or target.shape[0] != logits.shape[0]:
        raise ValueError("classification target must have shape B or B,1")
    if target.dtype not in _INTEGER_DTYPES:
        raise TypeError("classification target must use an integer dtype")
    if (
        isinstance(label_smoothing, bool)
        or not isinstance(label_smoothing, Real)
        or not math.isfinite(float(label_smoothing))
        or not 0.0 <= float(label_smoothing) < 1.0
    ):
        raise ValueError("label_smoothing must be finite and in [0, 1)")

    if not isinstance(ignore_index, Integral) or isinstance(ignore_index, bool):
        raise TypeError("ignore_index must be an integer")
    mask = _valid_mask_like(valid_mask, target, name="valid_mask")
    mask = mask & target.ne(int(ignore_index))
    selected = target[mask]
    if selected.numel() > 0 and (
        bool((selected < 0).any()) or bool((selected >= logits.shape[1]).any())
    ):
        raise ValueError("valid classification labels must be in [0, num_classes)")
    if not bool(mask.any()):
        return _differentiable_zero(logits)

    safe_target = torch.where(mask, target, torch.zeros_like(target)).long()
    weights = _validate_class_weights(
        class_weights,
        num_classes=logits.shape[1],
        reference=logits,
    )
    per_sample = F.cross_entropy(
        logits,
        safe_target,
        weight=weights,
        reduction="none",
        label_smoothing=float(label_smoothing),
    )
    return per_sample[mask].mean()


def _detection_tensors(predictions: DetectionOutput) -> tuple[Tensor, ...]:
    return tuple(
        tensor
        for group in (
            predictions.cls_logits,
            predictions.box_regression,
            predictions.quality,
        )
        for tensor in group
    )


def _subset_detection(
    predictions: DetectionOutput,
    indices: Sequence[int],
) -> DetectionOutput:
    reference = predictions.cls_logits[0]
    index = torch.as_tensor(indices, dtype=torch.long, device=reference.device)
    return DetectionOutput(
        cls_logits=tuple(
            tensor.index_select(0, index) for tensor in predictions.cls_logits
        ),
        box_regression=tuple(
            tensor.index_select(0, index) for tensor in predictions.box_regression
        ),
        quality=tuple(tensor.index_select(0, index) for tensor in predictions.quality),
    )


class MultiTaskCriterion(nn.Module):
    """Compose detection, segmentation, depth, and classification losses.

    The target mapping uses task names as keys. Dense/classification targets
    accept companion boolean masks named ``<task>_valid``. Detection targets are
    a batch-length sequence in which ``None`` means missing supervision and an
    empty ``boxes`` tensor means a labeled negative image.

    The returned ordered mapping always starts with ``total`` and then exposes
    active task losses. Detection additionally exposes its component losses and
    detached assignment counts using ``detection_*`` keys.
    """

    def __init__(
        self,
        detection_criterion: nn.Module | None = None,
        *,
        detection_num_classes: int | None = None,
        detection_reg_max: int = 0,
        task_weights: Mapping[str, float] | None = None,
        segmentation_ignore_index: int = 255,
        classification_ignore_index: int = -100,
        depth_loss_type: str = "log_l1_silog",
        depth_min: float = 0.0,
        depth_max: float | None = None,
        depth_log_l1_weight: float = 1.0,
        depth_silog_weight: float = 1.0,
        depth_gradient_weight: float = 0.0,
        depth_silog_lambda: float = 0.5,
        segmentation_class_weights: Tensor | Sequence[float] | None = None,
        segmentation_normalize_weighted_loss: bool = True,
        segmentation_lovasz_weight: float = 0.0,
        segmentation_auxiliary_weight: float = 0.0,
    ) -> None:
        super().__init__()
        if detection_criterion is not None and detection_num_classes is not None:
            raise ValueError(
                "provide detection_criterion or detection_num_classes, not both"
            )
        if detection_criterion is not None and not isinstance(
            detection_criterion, nn.Module
        ):
            raise TypeError("detection_criterion must be an nn.Module or None")
        if detection_num_classes is not None:
            if (
                isinstance(detection_num_classes, bool)
                or not isinstance(detection_num_classes, Integral)
                or detection_num_classes <= 0
            ):
                raise ValueError("detection_num_classes must be a positive integer")
            if (
                isinstance(detection_reg_max, bool)
                or not isinstance(detection_reg_max, Integral)
                or detection_reg_max < 0
            ):
                raise ValueError("detection_reg_max must be a non-negative integer")
            from .detection import DetectionCriterion

            detection_criterion = DetectionCriterion(
                int(detection_num_classes), reg_max=int(detection_reg_max)
            )
        self.detection_criterion = detection_criterion

        if not isinstance(segmentation_ignore_index, Integral) or isinstance(
            segmentation_ignore_index, bool
        ):
            raise TypeError("segmentation_ignore_index must be an integer")
        self.segmentation_ignore_index = int(segmentation_ignore_index)
        if not isinstance(classification_ignore_index, Integral) or isinstance(
            classification_ignore_index, bool
        ):
            raise TypeError("classification_ignore_index must be an integer")
        self.classification_ignore_index = int(classification_ignore_index)
        if depth_loss_type not in {
            "l1",
            "smooth_l1",
            "log_l1",
            "log_l1_silog",
            "per_image_silog_log_l1_gradient",
        }:
            raise ValueError(
                "depth_loss_type must be 'l1', 'smooth_l1', 'log_l1', or "
                "'log_l1_silog', or 'per_image_silog_log_l1_gradient'"
            )
        self.depth_loss_type = depth_loss_type
        self.depth_min = float(depth_min)
        self.depth_max = None if depth_max is None else float(depth_max)
        for value, name in (
            (depth_log_l1_weight, "depth_log_l1_weight"),
            (depth_silog_weight, "depth_silog_weight"),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(f"{name} must be finite and non-negative")
        self.depth_log_l1_weight = float(depth_log_l1_weight)
        self.depth_silog_weight = float(depth_silog_weight)
        for value, name in (
            (depth_gradient_weight, "depth_gradient_weight"),
            (segmentation_lovasz_weight, "segmentation_lovasz_weight"),
            (segmentation_auxiliary_weight, "segmentation_auxiliary_weight"),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(f"{name} must be finite and non-negative")
        if (
            isinstance(depth_silog_lambda, bool)
            or not isinstance(depth_silog_lambda, Real)
            or not math.isfinite(float(depth_silog_lambda))
            or not 0.0 <= float(depth_silog_lambda) <= 1.0
        ):
            raise ValueError("depth_silog_lambda must be finite and in [0,1]")
        if not isinstance(segmentation_normalize_weighted_loss, bool):
            raise TypeError("segmentation_normalize_weighted_loss must be a boolean")
        if segmentation_class_weights is None:
            self.register_buffer("segmentation_class_weights", None)
        else:
            weights = torch.as_tensor(segmentation_class_weights, dtype=torch.float32)
            if weights.ndim != 1 or weights.numel() < 2:
                raise ValueError("segmentation_class_weights must be a 1D class vector")
            if not bool(torch.isfinite(weights).all()) or bool((weights <= 0).any()):
                raise ValueError("segmentation_class_weights must be finite and positive")
            self.register_buffer("segmentation_class_weights", weights)
        self.segmentation_normalize_weighted_loss = segmentation_normalize_weighted_loss
        self.segmentation_lovasz_weight = float(segmentation_lovasz_weight)
        self.segmentation_auxiliary_weight = float(segmentation_auxiliary_weight)
        self.depth_gradient_weight = float(depth_gradient_weight)
        self.depth_silog_lambda = float(depth_silog_lambda)

        raw_weights = {} if task_weights is None else dict(task_weights)
        unknown = set(raw_weights) - set(_TASK_NAMES)
        if unknown:
            raise ValueError("unknown task weights: " + ", ".join(sorted(unknown)))
        checked_weights: dict[str, float] = {}
        for task in _TASK_NAMES:
            value = raw_weights.get(task, 1.0)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(f"task weight {task!r} must be finite and non-negative")
            checked_weights[task] = float(value)
        self.task_weights = checked_weights

    @property
    def resolved_task_weights(self) -> dict[str, float]:
        """Return all task weights, including explicit unit defaults.

        A copy is returned so checkpoint/log metadata cannot accidentally
        mutate the criterion used by an active training run.
        """

        return dict(self.task_weights)

    @property
    def loss_metadata(self) -> dict[str, Any]:
        """JSON-serializable, resolved loss configuration for run manifests."""

        return {
            "schema_version": 1,
            "criterion": "replite_multitask",
            "task_weights": self.resolved_task_weights,
            "segmentation_ignore_index": self.segmentation_ignore_index,
            "classification_ignore_index": self.classification_ignore_index,
            "depth": {
                "loss_type": self.depth_loss_type,
                "min": self.depth_min,
                "max": self.depth_max,
                "log_l1_weight": self.depth_log_l1_weight,
                "silog_weight": self.depth_silog_weight,
                "gradient_weight": self.depth_gradient_weight,
                "silog_lambda": self.depth_silog_lambda,
            },
            "segmentation": {
                "class_weights": None
                if self.segmentation_class_weights is None
                else self.segmentation_class_weights.detach().cpu().tolist(),
                "normalize_weighted_loss": self.segmentation_normalize_weighted_loss,
                "lovasz_weight": self.segmentation_lovasz_weight,
                "auxiliary_weight": self.segmentation_auxiliary_weight,
            },
        }

    @staticmethod
    def _require_target(targets: Mapping[str, Any], task: str) -> Any:
        if task not in targets:
            raise KeyError(f"active task {task!r} requires a target key")
        return targets[task]

    def _detection_loss(
        self,
        predictions: DetectionOutput,
        target_value: Any,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        tensors = _detection_tensors(predictions)
        zero = _differentiable_zero(*tensors)
        zero_count = zero.detach().new_zeros(())
        empty = {
            "detection": zero,
            "detection_classification": zero,
            "detection_box": zero,
            "detection_quality": zero,
            "detection_dfl": zero,
            "detection_num_positive": zero_count,
            "detection_num_targets": zero_count,
            "detection_num_unmatched": zero_count,
        }
        if target_value is None:
            return zero, empty
        if isinstance(target_value, (str, bytes, Tensor)) or not isinstance(
            target_value, Sequence
        ):
            raise TypeError(
                "detection target must be a batch-length sequence of mappings or None"
            )
        batch_size = predictions.cls_logits[0].shape[0]
        if len(target_value) != batch_size:
            raise ValueError(
                f"detection target length must be {batch_size}, got {len(target_value)}"
            )
        supervised_indices: list[int] = []
        supervised_targets: list[Mapping[str, Tensor]] = []
        for index, sample in enumerate(target_value):
            if sample is None:
                continue
            if not isinstance(sample, Mapping):
                raise TypeError("each detection target must be a mapping or None")
            supervised_indices.append(index)
            supervised_targets.append(sample)
        if not supervised_indices:
            return zero, empty
        if self.detection_criterion is None:
            raise RuntimeError(
                "detection predictions require a DetectionCriterion; pass one or "
                "set detection_num_classes"
            )

        selected = _subset_detection(predictions, supervised_indices)
        result = self.detection_criterion(selected, supervised_targets)
        required = (
            "total",
            "classification",
            "box",
            "quality",
            "dfl",
            "num_positive",
            "num_targets",
            "num_unmatched",
        )
        if any(not hasattr(result, name) for name in required):
            raise TypeError("detection criterion returned an incompatible result")
        values = {
            "detection": result.total,
            "detection_classification": result.classification,
            "detection_box": result.box,
            "detection_quality": result.quality,
            "detection_dfl": result.dfl,
            "detection_num_positive": result.num_positive.detach(),
            "detection_num_targets": result.num_targets.detach(),
            "detection_num_unmatched": result.num_unmatched.detach(),
        }
        for name, value in values.items():
            if not isinstance(value, Tensor) or value.ndim != 0:
                raise TypeError(f"detection result {name!r} must be a scalar tensor")
        return result.total, values

    def forward(
        self,
        predictions: RepLiteOutput,
        targets: Mapping[str, Any],
    ) -> dict[str, Tensor]:
        if not isinstance(predictions, RepLiteOutput):
            raise TypeError("predictions must be a RepLiteOutput")
        if not isinstance(targets, Mapping):
            raise TypeError("targets must be a mapping keyed by task name")

        task_losses: dict[str, Tensor] = {}
        details: dict[str, Tensor] = {}

        if predictions.detection is not None:
            target = self._require_target(targets, "detection")
            task_loss, detection_details = self._detection_loss(
                predictions.detection, target
            )
            task_losses["detection"] = task_loss
            details.update(detection_details)

        if predictions.segmentation is not None:
            target = self._require_target(targets, "segmentation")
            if target is None:
                loss = _differentiable_zero(predictions.segmentation)
            else:
                loss = segmentation_loss(
                    predictions.segmentation,
                    target,
                    ignore_index=self.segmentation_ignore_index,
                    valid_mask=targets.get("segmentation_valid"),
                    class_weights=self.segmentation_class_weights,
                    normalize_weighted_loss=self.segmentation_normalize_weighted_loss,
                    lovasz_weight=self.segmentation_lovasz_weight,
                )
            task_losses["segmentation"] = loss
            details["segmentation"] = loss
            if (
                predictions.segmentation_aux is not None
                and self.segmentation_auxiliary_weight > 0.0
                and target is not None
            ):
                aux_loss = segmentation_loss(
                    predictions.segmentation_aux,
                    target,
                    ignore_index=self.segmentation_ignore_index,
                    valid_mask=targets.get("segmentation_valid"),
                    class_weights=self.segmentation_class_weights,
                    normalize_weighted_loss=self.segmentation_normalize_weighted_loss,
                )
                details["segmentation_auxiliary"] = aux_loss
                task_losses["segmentation"] = task_losses["segmentation"] + (
                    self.segmentation_auxiliary_weight * aux_loss
                )

        if predictions.depth is not None:
            target = self._require_target(targets, "depth")
            if target is None:
                loss = _differentiable_zero(predictions.depth)
            else:
                loss = masked_depth_loss(
                    predictions.depth,
                    target,
                    valid_mask=targets.get("depth_valid"),
                    min_depth=self.depth_min,
                    max_depth=self.depth_max,
                    loss_type=self.depth_loss_type,
                    log_l1_weight=self.depth_log_l1_weight,
                    silog_weight=self.depth_silog_weight,
                    gradient_weight=self.depth_gradient_weight,
                    silog_lambda=self.depth_silog_lambda,
                )
            task_losses["depth"] = loss
            details["depth"] = loss

        if predictions.classification is not None:
            target = self._require_target(targets, "classification")
            if target is None:
                loss = _differentiable_zero(predictions.classification)
            else:
                loss = classification_loss(
                    predictions.classification,
                    target,
                    ignore_index=self.classification_ignore_index,
                    valid_mask=targets.get("classification_valid"),
                )
            task_losses["classification"] = loss
            details["classification"] = loss

        if not task_losses:
            raise ValueError("predictions contain no active task outputs")
        weighted = [
            loss * self.task_weights[task] for task, loss in task_losses.items()
        ]
        total = weighted[0]
        for value in weighted[1:]:
            total = total + value
        return {"total": total, **details}


__all__ = [
    "MultiTaskCriterion",
    "classification_loss",
    "masked_depth_loss",
    "segmentation_loss",
]
