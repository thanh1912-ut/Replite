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
    return class_weights.to(dtype=reference.dtype)


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


def segmentation_loss(
    logits: Tensor,
    target: Tensor,
    *,
    ignore_index: int = 255,
    valid_mask: Tensor | None = None,
    class_weights: Tensor | None = None,
    label_smoothing: float = 0.0,
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
    return per_pixel[mask].mean()


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
    eps: float = 1e-6,
) -> Tensor:
    """Compute depth loss over valid finite target pixels only.

    Targets must be strictly greater than ``min_depth`` and, when supplied, no
    greater than ``max_depth``. Supported losses are ``l1``, ``smooth_l1``, and
    ``log_l1``, and ``log_l1_silog``. The latter combines mean absolute log
    error with the non-negative scale-invariant term
    ``mean(d**2) - 0.5 * mean(d)**2``. Depth arithmetic is always FP32 under
    autocast. An empty valid set returns differentiable zero.
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
    if loss_type not in {"l1", "smooth_l1", "log_l1", "log_l1_silog"}:
        raise ValueError(
            "loss_type must be 'l1', 'smooth_l1', 'log_l1', or "
            "'log_l1_silog'"
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
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f"{name} must be finite and non-negative")
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
        }:
            raise ValueError(
                "depth_loss_type must be 'l1', 'smooth_l1', 'log_l1', or "
                "'log_l1_silog'"
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
                )
            task_losses["segmentation"] = loss
            details["segmentation"] = loss

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
