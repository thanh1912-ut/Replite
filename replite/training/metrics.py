"""Dependency-free validation metrics for RepLite perception tasks.

Accumulators keep detached state on CPU so validation does not retain graphs
or consume GPU memory across an epoch. States can be merged across shards.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any

import torch
from torch import Tensor

from replite.multitask.model import RepLiteOutput

from .detection import box_iou, decode_detections


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _broadcast_bool_mask(mask: Tensor | None, target: Tensor) -> Tensor:
    if mask is None:
        return torch.ones_like(target, dtype=torch.bool)
    if not isinstance(mask, Tensor) or mask.dtype != torch.bool:
        raise TypeError("valid_mask must be a boolean tensor or None")
    if mask.device != target.device:
        raise ValueError("valid_mask and target must be on the same device")
    if mask.shape == target.shape:
        return mask
    if mask.ndim == 1 and mask.shape[0] == target.shape[0]:
        shape = (target.shape[0],) + (1,) * (target.ndim - 1)
        return mask.reshape(shape).expand_as(target)
    if target.ndim == 4 and target.shape[1] == 1 and mask.ndim == 3:
        if mask.shape == (target.shape[0], target.shape[2], target.shape[3]):
            return mask.unsqueeze(1)
    if target.ndim == 3 and mask.ndim == 4 and mask.shape[1] == 1:
        if (mask.shape[0], mask.shape[2], mask.shape[3]) == target.shape:
            return mask[:, 0]
    raise ValueError("valid_mask is not broadcastable to the target shape")


class SegmentationMetrics:
    """Global confusion-matrix mIoU and pixel accuracy."""

    def __init__(self, num_classes: int, *, ignore_index: int = 255) -> None:
        self.num_classes = _positive_int(num_classes, "num_classes")
        if isinstance(ignore_index, bool) or not isinstance(ignore_index, Integral):
            raise ValueError("ignore_index must be an integer")
        self.ignore_index = int(ignore_index)
        self.reset()

    def reset(self) -> None:
        self.confusion_matrix = torch.zeros(
            self.num_classes, self.num_classes, dtype=torch.int64
        )

    def update(
        self,
        prediction: Tensor,
        target: Tensor,
        *,
        valid_mask: Tensor | None = None,
    ) -> None:
        if not isinstance(prediction, Tensor) or not isinstance(target, Tensor):
            raise TypeError("prediction and target must be tensors")
        if prediction.ndim == 4:
            if prediction.shape[1] == self.num_classes and prediction.is_floating_point():
                prediction = prediction.argmax(dim=1)
            elif prediction.shape[1] == 1:
                prediction = prediction[:, 0]
            else:
                raise ValueError("segmentation prediction has the wrong class count")
        if target.ndim == 4 and target.shape[1] == 1:
            target = target[:, 0]
        if prediction.ndim != 3 or target.ndim != 3:
            raise ValueError("segmentation labels must have shape B,H,W")
        if prediction.shape != target.shape:
            raise ValueError("prediction and target shapes must match")
        mask = _broadcast_bool_mask(valid_mask, target)
        mask &= target.ne(self.ignore_index)
        if not bool(mask.any()):
            return
        actual = target[mask].long()
        predicted = prediction[mask].long()
        if bool((actual < 0).any()) or bool((actual >= self.num_classes).any()):
            raise ValueError("valid targets are outside the configured class range")
        if bool((predicted < 0).any()) or bool(
            (predicted >= self.num_classes).any()
        ):
            raise ValueError("predictions are outside the configured class range")
        encoded = actual.cpu() * self.num_classes + predicted.cpu()
        self.confusion_matrix += torch.bincount(
            encoded, minlength=self.num_classes**2
        ).reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict[str, Any]:
        matrix = self.confusion_matrix.double()
        true_positive = matrix.diag()
        union = matrix.sum(1) + matrix.sum(0) - true_positive
        present = union > 0
        per_class = torch.zeros(self.num_classes, dtype=torch.float64)
        per_class[present] = true_positive[present] / union[present]
        total = matrix.sum()
        return {
            "miou": float(per_class[present].mean()) if bool(present.any()) else 0.0,
            "pixel_accuracy": float(true_positive.sum() / total) if total > 0 else 0.0,
            "per_class_iou": per_class.tolist(),
            "present_classes": present.tolist(),
            "confusion_matrix": self.confusion_matrix.clone(),
            "num_pixels": int(total),
        }

    def merge_state(self, other: "SegmentationMetrics") -> None:
        if not isinstance(other, SegmentationMetrics):
            raise TypeError("other must be SegmentationMetrics")
        if (self.num_classes, self.ignore_index) != (
            other.num_classes,
            other.ignore_index,
        ):
            raise ValueError("segmentation metric configurations do not match")
        self.confusion_matrix += other.confusion_matrix

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "num_classes": self.num_classes,
            "ignore_index": self.ignore_index,
            "confusion_matrix": self.confusion_matrix.clone(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema_version") != 1:
            raise ValueError("unsupported segmentation metric state")
        if (state.get("num_classes"), state.get("ignore_index")) != (
            self.num_classes,
            self.ignore_index,
        ):
            raise ValueError("segmentation metric configuration mismatch")
        matrix = torch.as_tensor(state.get("confusion_matrix"), dtype=torch.int64)
        if matrix.shape != (self.num_classes, self.num_classes):
            raise ValueError("invalid confusion matrix shape")
        self.confusion_matrix = matrix.clone().cpu()


class DepthMetrics:
    """Globally aggregated metric-depth errors and delta accuracy."""

    _SUM_FIELDS = (
        "sum_abs_rel",
        "sum_sq_rel",
        "sum_squared_error",
        "sum_squared_log_error",
        "delta1_count",
        "delta2_count",
        "delta3_count",
    )

    def __init__(
        self,
        *,
        min_depth: float = 0.0,
        max_depth: float | None = None,
        eps: float = 1e-6,
    ) -> None:
        for value, name in ((min_depth, "min_depth"), (eps, "eps")):
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(f"{name} must be finite and non-negative")
        if float(eps) == 0.0:
            raise ValueError("eps must be positive")
        if max_depth is not None and (
            isinstance(max_depth, bool)
            or not isinstance(max_depth, Real)
            or not math.isfinite(float(max_depth))
            or float(max_depth) <= float(min_depth)
        ):
            raise ValueError("max_depth must be finite and greater than min_depth")
        self.min_depth = float(min_depth)
        self.max_depth = None if max_depth is None else float(max_depth)
        self.eps = float(eps)
        self.reset()

    def reset(self) -> None:
        self.count = 0
        for name in self._SUM_FIELDS:
            setattr(self, name, 0.0)

    def update(
        self,
        prediction: Tensor,
        target: Tensor,
        *,
        valid_mask: Tensor | None = None,
    ) -> None:
        if not isinstance(prediction, Tensor) or not isinstance(target, Tensor):
            raise TypeError("prediction and target must be tensors")
        if prediction.ndim == 3:
            prediction = prediction.unsqueeze(1)
        if target.ndim == 3:
            target = target.unsqueeze(1)
        if prediction.ndim != 4 or prediction.shape[1] != 1:
            raise ValueError("depth prediction must have shape B,1,H,W")
        if prediction.shape != target.shape:
            raise ValueError("depth prediction and target shapes must match")
        if not prediction.is_floating_point() or not target.is_floating_point():
            raise TypeError("depth prediction and target must be floating point")
        mask = _broadcast_bool_mask(valid_mask, target)
        mask &= torch.isfinite(target) & target.gt(self.min_depth)
        if self.max_depth is not None:
            mask &= target.le(self.max_depth)
        if not bool(mask.any()):
            return
        predicted = prediction[mask].detach().float()
        actual = target[mask].detach().float()
        if not bool(torch.isfinite(predicted).all()) or bool((predicted <= 0).any()):
            raise ValueError("valid depth predictions must be finite and positive")
        difference = predicted - actual
        ratio = torch.maximum(
            predicted / actual.clamp_min(self.eps),
            actual / predicted.clamp_min(self.eps),
        )
        log_difference = torch.log(predicted.clamp_min(self.eps)) - torch.log(
            actual.clamp_min(self.eps)
        )
        self.count += int(actual.numel())
        self.sum_abs_rel += float((difference.abs() / actual).double().sum().cpu())
        self.sum_sq_rel += float((difference.square() / actual).double().sum().cpu())
        self.sum_squared_error += float(difference.double().square().sum().cpu())
        self.sum_squared_log_error += float(
            log_difference.double().square().sum().cpu()
        )
        self.delta1_count += float((ratio < 1.25).sum().cpu())
        self.delta2_count += float((ratio < 1.25**2).sum().cpu())
        self.delta3_count += float((ratio < 1.25**3).sum().cpu())

    def compute(self) -> dict[str, float | int]:
        if self.count == 0:
            return {
                "abs_rel": 0.0,
                "sq_rel": 0.0,
                "rmse": 0.0,
                "rmse_log": 0.0,
                "delta1": 0.0,
                "delta2": 0.0,
                "delta3": 0.0,
                "num_pixels": 0,
            }
        count = float(self.count)
        return {
            "abs_rel": self.sum_abs_rel / count,
            "sq_rel": self.sum_sq_rel / count,
            "rmse": math.sqrt(self.sum_squared_error / count),
            "rmse_log": math.sqrt(self.sum_squared_log_error / count),
            "delta1": self.delta1_count / count,
            "delta2": self.delta2_count / count,
            "delta3": self.delta3_count / count,
            "num_pixels": self.count,
        }

    def merge_state(self, other: "DepthMetrics") -> None:
        if not isinstance(other, DepthMetrics):
            raise TypeError("other must be DepthMetrics")
        if (self.min_depth, self.max_depth, self.eps) != (
            other.min_depth,
            other.max_depth,
            other.eps,
        ):
            raise ValueError("depth metric configurations do not match")
        self.count += other.count
        for name in self._SUM_FIELDS:
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "min_depth": self.min_depth,
            "max_depth": self.max_depth,
            "eps": self.eps,
            "count": self.count,
            **{name: getattr(self, name) for name in self._SUM_FIELDS},
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema_version") != 1:
            raise ValueError("unsupported depth metric state")
        if (state.get("min_depth"), state.get("max_depth"), state.get("eps")) != (
            self.min_depth,
            self.max_depth,
            self.eps,
        ):
            raise ValueError("depth metric configuration mismatch")
        count = state.get("count")
        if isinstance(count, bool) or not isinstance(count, Integral) or count < 0:
            raise ValueError("invalid depth metric count")
        self.count = int(count)
        for name in self._SUM_FIELDS:
            value = float(state.get(name, math.nan))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"invalid depth metric field {name}")
            setattr(self, name, value)


class ClassificationMetrics:
    """Top-1 classification accuracy with explicit ignored labels."""

    def __init__(self, num_classes: int, *, ignore_index: int = -100) -> None:
        self.num_classes = _positive_int(num_classes, "num_classes")
        if isinstance(ignore_index, bool) or not isinstance(ignore_index, Integral):
            raise ValueError("ignore_index must be an integer")
        self.ignore_index = int(ignore_index)
        self.reset()

    def reset(self) -> None:
        self.correct = 0
        self.count = 0

    def update(
        self,
        prediction: Tensor,
        target: Tensor,
        *,
        valid_mask: Tensor | None = None,
    ) -> None:
        if not isinstance(prediction, Tensor) or not isinstance(target, Tensor):
            raise TypeError("prediction and target must be tensors")
        if prediction.ndim == 2:
            if prediction.shape[1] != self.num_classes:
                raise ValueError("classification logits have the wrong class count")
            prediction = prediction.argmax(dim=1)
        if target.ndim == 2 and target.shape[1] == 1:
            target = target[:, 0]
        if prediction.ndim != 1 or target.ndim != 1 or prediction.shape != target.shape:
            raise ValueError("classification labels must have shape B")
        mask = _broadcast_bool_mask(valid_mask, target)
        mask &= target.ne(self.ignore_index)
        if not bool(mask.any()):
            return
        actual = target[mask].long()
        predicted = prediction[mask].long()
        if bool((actual < 0).any()) or bool((actual >= self.num_classes).any()):
            raise ValueError("valid targets are outside the configured class range")
        if bool((predicted < 0).any()) or bool(
            (predicted >= self.num_classes).any()
        ):
            raise ValueError("predictions are outside the configured class range")
        self.correct += int((predicted == actual).sum().detach().cpu())
        self.count += int(actual.numel())

    def compute(self) -> dict[str, float | int]:
        return {
            "top1_accuracy": self.correct / self.count if self.count else 0.0,
            "num_samples": self.count,
        }

    def merge_state(self, other: "ClassificationMetrics") -> None:
        if not isinstance(other, ClassificationMetrics):
            raise TypeError("other must be ClassificationMetrics")
        if (self.num_classes, self.ignore_index) != (
            other.num_classes,
            other.ignore_index,
        ):
            raise ValueError("classification metric configurations do not match")
        self.correct += other.correct
        self.count += other.count

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "num_classes": self.num_classes,
            "ignore_index": self.ignore_index,
            "correct": self.correct,
            "count": self.count,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema_version") != 1:
            raise ValueError("unsupported classification metric state")
        if (state.get("num_classes"), state.get("ignore_index")) != (
            self.num_classes,
            self.ignore_index,
        ):
            raise ValueError("classification metric configuration mismatch")
        correct, count = state.get("correct"), state.get("count")
        if any(
            isinstance(value, bool) or not isinstance(value, Integral) or value < 0
            for value in (correct, count)
        ) or correct > count:
            raise ValueError("invalid classification metric counters")
        self.correct, self.count = int(correct), int(count)


def _validate_box_mapping(
    item: Mapping[str, Any], *, prediction: bool, num_classes: int
) -> dict[str, Tensor]:
    required = {"boxes", "labels"}
    if prediction:
        required.add("scores")
    missing = required - set(item)
    if missing:
        raise ValueError("box mapping is missing: " + ", ".join(sorted(missing)))
    boxes = torch.as_tensor(item["boxes"], dtype=torch.float32).detach().cpu()
    labels = torch.as_tensor(item["labels"]).detach().cpu()
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must have shape N,4")
    if labels.dtype != torch.long or labels.shape != (boxes.shape[0],):
        raise ValueError("labels must be a long tensor with shape N")
    if not bool(torch.isfinite(boxes).all()):
        raise ValueError("boxes must be finite")
    if boxes.shape[0] and bool((boxes[:, 2:] <= boxes[:, :2]).any()):
        raise ValueError("boxes must have positive width and height")
    if labels.numel() and (
        bool((labels < 0).any()) or bool((labels >= num_classes).any())
    ):
        raise ValueError("labels are outside the configured class range")
    result = {"boxes": boxes, "labels": labels}
    if prediction:
        scores = torch.as_tensor(item["scores"], dtype=torch.float32).detach().cpu()
        if scores.shape != (boxes.shape[0],) or not bool(torch.isfinite(scores).all()):
            raise ValueError("scores must be a finite tensor with shape N")
        result["scores"] = scores
    return result


class DetectionMAP:
    """Pure-Torch COCO-style 101-point mAP over IoU 0.50:0.05:0.95."""

    def __init__(
        self,
        num_classes: int,
        *,
        iou_thresholds: Sequence[float] | None = None,
        max_detections: int = 300,
    ) -> None:
        self.num_classes = _positive_int(num_classes, "num_classes")
        self.max_detections = _positive_int(max_detections, "max_detections")
        raw = (
            tuple(0.5 + 0.05 * index for index in range(10))
            if iou_thresholds is None
            else tuple(iou_thresholds)
        )
        if not raw:
            raise ValueError("iou_thresholds must not be empty")
        checked: list[float] = []
        for value in raw:
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                or not 0.0 < float(value) <= 1.0
            ):
                raise ValueError("IoU thresholds must be finite values in (0,1]")
            checked.append(float(value))
        if len(set(checked)) != len(checked):
            raise ValueError("IoU thresholds must be unique")
        self.iou_thresholds = tuple(sorted(checked))
        self.reset()

    def reset(self) -> None:
        self.predictions: list[dict[str, Tensor]] = []
        self.targets: list[dict[str, Tensor]] = []

    def update(
        self,
        predictions: Sequence[Mapping[str, Any]],
        targets: Sequence[Mapping[str, Any] | None],
    ) -> None:
        if len(predictions) != len(targets):
            raise ValueError("prediction and target batch lengths must match")
        for prediction, target in zip(predictions, targets):
            if target is None:
                continue
            checked_prediction = _validate_box_mapping(
                prediction, prediction=True, num_classes=self.num_classes
            )
            order = torch.argsort(
                checked_prediction["scores"], descending=True, stable=True
            )[: self.max_detections]
            self.predictions.append(
                {key: value[order] for key, value in checked_prediction.items()}
            )
            self.targets.append(
                _validate_box_mapping(
                    target, prediction=False, num_classes=self.num_classes
                )
            )

    @staticmethod
    def _interpolated_ap(tp: Tensor, fp: Tensor, num_targets: int) -> float:
        if num_targets == 0:
            return 0.0
        cumulative_tp = tp.cumsum(0)
        cumulative_fp = fp.cumsum(0)
        recall = cumulative_tp / float(num_targets)
        precision = cumulative_tp / (cumulative_tp + cumulative_fp).clamp_min(1.0)
        total = 0.0
        for level in torch.linspace(0.0, 1.0, 101, dtype=torch.float64):
            eligible = precision[recall >= level]
            total += float(eligible.max()) if eligible.numel() else 0.0
        return total / 101.0

    def _class_ap(self, class_id: int, threshold: float) -> float | None:
        targets_by_image: list[Tensor] = []
        total_targets = 0
        detections: list[tuple[float, int, int, Tensor]] = []
        insertion = 0
        for image_index, (prediction, target) in enumerate(
            zip(self.predictions, self.targets)
        ):
            boxes = target["boxes"][target["labels"] == class_id]
            targets_by_image.append(boxes)
            total_targets += int(boxes.shape[0])
            mask = prediction["labels"] == class_id
            for score, box in zip(prediction["scores"][mask], prediction["boxes"][mask]):
                detections.append((float(score), insertion, image_index, box))
                insertion += 1
        if total_targets == 0:
            return None
        detections.sort(key=lambda item: (-item[0], item[1]))
        matched = [torch.zeros(len(boxes), dtype=torch.bool) for boxes in targets_by_image]
        true_positive = torch.zeros(len(detections), dtype=torch.float64)
        false_positive = torch.zeros(len(detections), dtype=torch.float64)
        for index, (_, _, image_index, box) in enumerate(detections):
            targets = targets_by_image[image_index]
            if targets.numel() == 0:
                false_positive[index] = 1.0
                continue
            overlaps = box_iou(box.unsqueeze(0), targets).squeeze(0)
            overlaps = overlaps.masked_fill(matched[image_index], -1.0)
            overlap, target_index = overlaps.max(dim=0)
            if overlap >= threshold:
                true_positive[index] = 1.0
                matched[image_index][target_index] = True
            else:
                false_positive[index] = 1.0
        return self._interpolated_ap(true_positive, false_positive, total_targets)

    def compute(self) -> dict[str, Any]:
        threshold_values: dict[float, list[float]] = {}
        per_class: dict[int, list[float]] = {index: [] for index in range(self.num_classes)}
        for threshold in self.iou_thresholds:
            values: list[float] = []
            for class_id in range(self.num_classes):
                value = self._class_ap(class_id, threshold)
                if value is not None:
                    values.append(value)
                    per_class[class_id].append(value)
            threshold_values[threshold] = values
        all_values = [value for values in threshold_values.values() for value in values]

        def mean_at(wanted: float) -> float:
            values = next(
                (values for threshold, values in threshold_values.items() if abs(threshold - wanted) < 1e-9),
                [],
            )
            return sum(values) / len(values) if values else 0.0

        return {
            "map50_95": sum(all_values) / len(all_values) if all_values else 0.0,
            "map50": mean_at(0.50),
            "map75": mean_at(0.75),
            "per_class_map": {
                class_id: sum(values) / len(values)
                for class_id, values in per_class.items()
                if values
            },
            "num_images": len(self.targets),
            "num_targets": sum(int(item["boxes"].shape[0]) for item in self.targets),
        }

    def merge_state(self, other: "DetectionMAP") -> None:
        if not isinstance(other, DetectionMAP):
            raise TypeError("other must be DetectionMAP")
        if (
            self.num_classes,
            self.iou_thresholds,
            self.max_detections,
        ) != (
            other.num_classes,
            other.iou_thresholds,
            other.max_detections,
        ):
            raise ValueError("detection metric configurations do not match")
        self.predictions.extend(copy.deepcopy(other.predictions))
        self.targets.extend(copy.deepcopy(other.targets))

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "num_classes": self.num_classes,
            "iou_thresholds": self.iou_thresholds,
            "max_detections": self.max_detections,
            "predictions": copy.deepcopy(self.predictions),
            "targets": copy.deepcopy(self.targets),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema_version") != 1:
            raise ValueError("unsupported detection metric state")
        if (
            state.get("num_classes"),
            tuple(state.get("iou_thresholds", ())),
            state.get("max_detections"),
        ) != (self.num_classes, self.iou_thresholds, self.max_detections):
            raise ValueError("detection metric configuration mismatch")
        self.reset()
        self.update(state.get("predictions", ()), state.get("targets", ()))


class MultiTaskMetrics:
    """Adapter from ``RepLiteOutput`` to configured per-task accumulators."""

    def __init__(
        self,
        *,
        detection: DetectionMAP | None = None,
        segmentation: SegmentationMetrics | None = None,
        depth: DepthMetrics | None = None,
        classification: ClassificationMetrics | None = None,
        detection_reg_max: int = 0,
        detection_score_threshold: float = 0.001,
        detection_nms_iou_threshold: float = 0.6,
    ) -> None:
        metrics = (detection, segmentation, depth, classification)
        expected = (DetectionMAP, SegmentationMetrics, DepthMetrics, ClassificationMetrics)
        for value, kind in zip(metrics, expected):
            if value is not None and not isinstance(value, kind):
                raise TypeError(f"metric must be {kind.__name__} or None")
        if not any(metric is not None for metric in metrics):
            raise ValueError("at least one task metric is required")
        if (
            isinstance(detection_reg_max, bool)
            or not isinstance(detection_reg_max, Integral)
            or detection_reg_max < 0
        ):
            raise ValueError("detection_reg_max must be non-negative")
        for value, name in (
            (detection_score_threshold, "detection_score_threshold"),
            (detection_nms_iou_threshold, "detection_nms_iou_threshold"),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(f"{name} must be finite and in [0,1]")
        self.detection = detection
        self.segmentation = segmentation
        self.depth = depth
        self.classification = classification
        self.detection_reg_max = int(detection_reg_max)
        self.detection_score_threshold = float(detection_score_threshold)
        self.detection_nms_iou_threshold = float(detection_nms_iou_threshold)

    @property
    def _configured(self) -> dict[str, Any]:
        return {
            "detection": self.detection,
            "segmentation": self.segmentation,
            "depth": self.depth,
            "classification": self.classification,
        }

    def reset(self) -> None:
        for metric in self._configured.values():
            if metric is not None:
                metric.reset()

    @staticmethod
    def _fallback_image_size(targets: Mapping[str, Any]) -> tuple[int, int] | None:
        for name in ("segmentation", "depth"):
            target = targets.get(name)
            if isinstance(target, Tensor) and target.ndim >= 3:
                return int(target.shape[-2]), int(target.shape[-1])
        value = targets.get("image_size")
        if value is not None:
            raw = tuple(value)
            if len(raw) == 2:
                return int(raw[0]), int(raw[1])
        return None

    def update(self, predictions: RepLiteOutput, targets: Mapping[str, Any]) -> None:
        if not isinstance(predictions, RepLiteOutput):
            raise TypeError("predictions must be RepLiteOutput")
        if not isinstance(targets, Mapping):
            raise TypeError("targets must be a task mapping")
        if self.detection is not None:
            if predictions.detection is None:
                raise ValueError("detection metric configured but prediction is absent")
            target_batch = targets.get("detection")
            if target_batch is not None:
                if not isinstance(target_batch, Sequence) or isinstance(
                    target_batch, (str, bytes, Tensor)
                ):
                    raise TypeError("detection targets must be a batch sequence")
                if not any(item is not None for item in target_batch):
                    target_batch = None
            if target_batch is not None:
                fallback = self._fallback_image_size(targets)
                first_explicit = next(
                    (
                        item["valid_size"]
                        for item in target_batch
                        if isinstance(item, Mapping) and "valid_size" in item
                    ),
                    None,
                )
                sizes = []
                for item in target_batch:
                    size = item.get("valid_size") if isinstance(item, Mapping) else None
                    size = size if size is not None else fallback or first_explicit
                    if size is None:
                        raise ValueError(
                            "detection validation requires valid_size or a dense target"
                        )
                    sizes.append(size)
                decoded = decode_detections(
                    predictions.detection,
                    sizes,
                    reg_max=self.detection_reg_max,
                    score_threshold=self.detection_score_threshold,
                    nms_iou_threshold=self.detection_nms_iou_threshold,
                )
                self.detection.update(decoded, target_batch)
        if self.segmentation is not None:
            if predictions.segmentation is None:
                raise ValueError("segmentation metric configured but prediction is absent")
            target = targets.get("segmentation")
            if target is not None:
                self.segmentation.update(
                    predictions.segmentation,
                    target,
                    valid_mask=targets.get("segmentation_valid"),
                )
        if self.depth is not None:
            if predictions.depth is None:
                raise ValueError("depth metric configured but prediction is absent")
            target = targets.get("depth")
            if target is not None:
                self.depth.update(
                    predictions.depth,
                    target,
                    valid_mask=targets.get("depth_valid"),
                )
        if self.classification is not None:
            if predictions.classification is None:
                raise ValueError("classification metric configured but prediction is absent")
            target = targets.get("classification")
            if target is not None:
                self.classification.update(
                    predictions.classification,
                    target,
                    valid_mask=targets.get("classification_valid"),
                )

    def compute(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for task, metric in self._configured.items():
            if metric is not None:
                result.update(
                    {f"{task}/{name}": value for name, value in metric.compute().items()}
                )
        return result

    def merge_state(self, other: "MultiTaskMetrics") -> None:
        if not isinstance(other, MultiTaskMetrics):
            raise TypeError("other must be MultiTaskMetrics")
        if (
            self.detection_reg_max,
            self.detection_score_threshold,
            self.detection_nms_iou_threshold,
        ) != (
            other.detection_reg_max,
            other.detection_score_threshold,
            other.detection_nms_iou_threshold,
        ):
            raise ValueError("multi-task metric configurations do not match")
        for task, metric in self._configured.items():
            other_metric = other._configured[task]
            if (metric is None) != (other_metric is None):
                raise ValueError("multi-task metric task sets do not match")
            if metric is not None:
                metric.merge_state(other_metric)

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "detection_reg_max": self.detection_reg_max,
            "detection_score_threshold": self.detection_score_threshold,
            "detection_nms_iou_threshold": self.detection_nms_iou_threshold,
            "tasks": {
                task: None if metric is None else metric.state_dict()
                for task, metric in self._configured.items()
            },
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema_version") != 1:
            raise ValueError("unsupported multi-task metric state")
        if (
            state.get("detection_reg_max"),
            state.get("detection_score_threshold"),
            state.get("detection_nms_iou_threshold"),
        ) != (
            self.detection_reg_max,
            self.detection_score_threshold,
            self.detection_nms_iou_threshold,
        ):
            raise ValueError("multi-task metric configuration mismatch")
        task_states = state.get("tasks")
        if not isinstance(task_states, Mapping):
            raise ValueError("invalid multi-task metric task states")
        for task, metric in self._configured.items():
            saved = task_states.get(task)
            if (metric is None) != (saved is None):
                raise ValueError("multi-task metric task set mismatch")
            if metric is not None:
                metric.load_state_dict(saved)


__all__ = [
    "ClassificationMetrics",
    "DepthMetrics",
    "DetectionMAP",
    "MultiTaskMetrics",
    "SegmentationMetrics",
]
