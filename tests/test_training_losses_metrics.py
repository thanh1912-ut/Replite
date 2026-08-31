"""Golden tests for task losses and validation accumulators."""

from __future__ import annotations

import math

import pytest
import torch
from torch.nn import functional as F

from replite.multitask.heads import DetectionOutput
from replite.multitask.model import RepLiteOutput
from replite.training.losses import (
    MultiTaskCriterion,
    classification_loss,
    masked_depth_loss,
    segmentation_loss,
)
from replite.training.metrics import (
    ClassificationMetrics,
    DepthMetrics,
    DetectionMAP,
    MultiTaskMetrics,
    SegmentationMetrics,
)


def test_segmentation_loss_matches_cross_entropy_and_all_ignore_is_connected() -> None:
    logits = torch.tensor(
        [[[[2.0, -1.0]], [[-2.0, 1.0]]]], requires_grad=True
    )
    target = torch.tensor([[[0, 1]]])
    actual = segmentation_loss(logits, target)
    expected = F.cross_entropy(logits, target)
    torch.testing.assert_close(actual, expected)

    ignored_logits = torch.randn(2, 3, 2, 3, requires_grad=True)
    ignored = segmentation_loss(
        ignored_logits, torch.full((2, 2, 3), 255, dtype=torch.long)
    )
    assert ignored.item() == 0.0
    ignored.backward()
    assert ignored_logits.grad is not None
    assert torch.count_nonzero(ignored_logits.grad) == 0


def test_depth_losses_filter_invalid_targets_and_use_fp32() -> None:
    prediction = torch.tensor([[[[1.0, 4.0, 8.0]]]], requires_grad=True)
    target = torch.tensor([[[[2.0, float("nan"), 0.0]]]])
    actual = masked_depth_loss(
        prediction,
        target,
        valid_mask=torch.tensor([[[[True, True, True]]]]),
        loss_type="l1",
    )
    torch.testing.assert_close(actual, torch.tensor(1.0))
    actual.backward()
    torch.testing.assert_close(
        prediction.grad, torch.tensor([[[[-1.0, 0.0, 0.0]]]])
    )

    half_prediction = torch.tensor([[[[1.0, 4.0]]]], dtype=torch.float16)
    log_combo = masked_depth_loss(
        half_prediction,
        torch.tensor([[[[2.0, 2.0]]]], dtype=torch.float16),
        loss_type="log_l1_silog",
    )
    assert log_combo.dtype == torch.float32
    assert torch.isfinite(log_combo)


def test_depth_all_invalid_returns_differentiable_zero() -> None:
    prediction = torch.ones(1, 1, 2, 2, requires_grad=True)
    target = torch.zeros_like(prediction)
    loss = masked_depth_loss(prediction, target)
    assert loss.item() == 0.0
    loss.backward()
    assert prediction.grad is not None


def test_classification_ignore_index_and_explicit_valid_mask() -> None:
    logits = torch.tensor(
        [[3.0, -1.0], [-1.0, 3.0], [2.0, -2.0]], requires_grad=True
    )
    targets = torch.tensor([0, -100, 1])
    loss = classification_loss(
        logits,
        targets,
        valid_mask=torch.tensor([True, True, False]),
    )
    torch.testing.assert_close(loss, F.cross_entropy(logits[:1], targets[:1]))
    loss.backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad[1:]) == 0


def test_multitask_criterion_weights_losses_and_requires_active_targets() -> None:
    segmentation = torch.randn(2, 3, 3, 4, requires_grad=True)
    depth = torch.ones(2, 1, 3, 4, requires_grad=True)
    classification = torch.randn(2, 4, requires_grad=True)
    output = RepLiteOutput(None, segmentation, depth, classification)
    targets = {
        "segmentation": torch.zeros(2, 3, 4, dtype=torch.long),
        "depth": torch.full((2, 1, 3, 4), 2.0),
        "depth_valid": torch.ones(2, 1, 3, 4, dtype=torch.bool),
        "classification": torch.tensor([1, -100]),
    }
    criterion = MultiTaskCriterion(
        task_weights={"segmentation": 2.0, "depth": 0.5, "classification": 3.0},
        depth_loss_type="l1",
    )
    losses = criterion(output, targets)
    expected = (
        2.0 * losses["segmentation"]
        + 0.5 * losses["depth"]
        + 3.0 * losses["classification"]
    )
    torch.testing.assert_close(losses["total"], expected)
    losses["total"].backward()
    assert segmentation.grad is not None
    assert depth.grad is not None
    assert classification.grad is not None

    with pytest.raises(KeyError, match="depth"):
        criterion(output, {key: value for key, value in targets.items() if key != "depth"})


def test_segmentation_metrics_golden_confusion_and_merge() -> None:
    prediction = torch.tensor([[[0, 0], [1, 1]]])
    target = torch.tensor([[[0, 1], [1, 255]]])
    metric = SegmentationMetrics(2)
    metric.update(prediction, target)
    result = metric.compute()
    # Rows are ground truth and columns are predictions.
    assert result["confusion_matrix"].tolist() == [[1, 0], [1, 1]]
    assert result["miou"] == pytest.approx(0.5)
    assert result["pixel_accuracy"] == pytest.approx(2.0 / 3.0)

    left = SegmentationMetrics(2)
    right = SegmentationMetrics(2)
    left.update(prediction[:, :, :1], target[:, :, :1])
    right.update(prediction[:, :, 1:], target[:, :, 1:])
    left.merge_state(right)
    assert torch.equal(left.compute()["confusion_matrix"], result["confusion_matrix"])


def test_segmentation_metric_state_round_trip_and_all_ignore() -> None:
    metric = SegmentationMetrics(3)
    metric.update(torch.zeros(1, 2, 2, dtype=torch.long), torch.full((1, 2, 2), 255))
    assert metric.compute()["num_pixels"] == 0
    state = metric.state_dict()
    restored = SegmentationMetrics(3)
    restored.load_state_dict(state)
    assert torch.equal(restored.confusion_matrix, metric.confusion_matrix)


def test_depth_metrics_match_hand_computation_and_merge() -> None:
    prediction = torch.tensor([[[[1.0, 3.0]]]])
    target = torch.tensor([[[[2.0, 2.0]]]])
    metric = DepthMetrics()
    metric.update(prediction, target)
    result = metric.compute()
    assert result["abs_rel"] == pytest.approx(0.5)
    assert result["sq_rel"] == pytest.approx(0.5)
    assert result["rmse"] == pytest.approx(1.0)
    expected_log = math.sqrt((math.log(0.5) ** 2 + math.log(1.5) ** 2) / 2)
    assert result["rmse_log"] == pytest.approx(expected_log)
    assert result["delta1"] == pytest.approx(0.0)
    assert result["delta2"] == pytest.approx(0.5)
    assert result["delta3"] == pytest.approx(0.5)

    left, right = DepthMetrics(), DepthMetrics()
    left.update(prediction[..., :1], target[..., :1])
    right.update(prediction[..., 1:], target[..., 1:])
    left.merge_state(right)
    assert left.compute() == pytest.approx(result)


def test_depth_metric_no_valid_pixels_is_explicit_and_invalid_prediction_rejected() -> None:
    metric = DepthMetrics()
    metric.update(torch.ones(1, 1, 1, 1), torch.zeros(1, 1, 1, 1))
    assert metric.compute()["num_pixels"] == 0
    with pytest.raises(ValueError, match="finite and positive"):
        metric.update(torch.zeros(1, 1, 1, 1), torch.ones(1, 1, 1, 1))


def test_classification_and_multitask_metric_adapter() -> None:
    classification = ClassificationMetrics(2)
    classification.update(
        torch.tensor([[5.0, 0.0], [0.0, 5.0], [5.0, 0.0]]),
        torch.tensor([0, 1, -100]),
    )
    assert classification.compute() == {"top1_accuracy": 1.0, "num_samples": 2}

    adapter = MultiTaskMetrics(
        segmentation=SegmentationMetrics(2),
        depth=DepthMetrics(),
        classification=ClassificationMetrics(2),
    )
    segmentation = torch.tensor(
        [[[[5.0, 0.0]], [[0.0, 5.0]]]]
    )
    output = RepLiteOutput(
        None,
        segmentation,
        torch.tensor([[[[2.0, 3.0]]]]),
        torch.tensor([[0.0, 5.0]]),
    )
    adapter.update(
        output,
        {
            "segmentation": torch.tensor([[[0, 1]]]),
            "depth": torch.tensor([[[[2.0, 3.0]]]]),
            "depth_valid": torch.ones(1, 1, 1, 2, dtype=torch.bool),
            "classification": torch.tensor([1]),
        },
    )
    result = adapter.compute()
    assert result["segmentation/miou"] == pytest.approx(1.0)
    assert result["depth/abs_rel"] == pytest.approx(0.0)
    assert result["classification/top1_accuracy"] == pytest.approx(1.0)


def _detection(boxes, scores, labels):
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "scores": torch.tensor(scores, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def _target(boxes, labels):
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def test_detection_map_is_one_for_perfect_predictions() -> None:
    metric = DetectionMAP(2)
    metric.update(
        [_detection([[0, 0, 10, 10], [20, 20, 30, 30]], [0.9, 0.8], [0, 1])],
        [_target([[0, 0, 10, 10], [20, 20, 30, 30]], [0, 1])],
    )
    result = metric.compute()
    assert result["map50_95"] == pytest.approx(1.0)
    assert result["map50"] == pytest.approx(1.0)
    assert result["map75"] == pytest.approx(1.0)
    assert result["per_class_map"] == pytest.approx({0: 1.0, 1: 1.0})


def test_detection_map_penalizes_higher_scored_false_positive() -> None:
    metric = DetectionMAP(1, iou_thresholds=(0.5,))
    metric.update(
        [_detection([[20, 20, 30, 30], [0, 0, 10, 10]], [0.9, 0.8], [0, 0])],
        [_target([[0, 0, 10, 10]], [0])],
    )
    result = metric.compute()
    assert result["map50"] == pytest.approx(0.5)
    assert result["map50_95"] == pytest.approx(0.5)


def test_detection_map_merge_and_state_round_trip() -> None:
    first, second = DetectionMAP(1), DetectionMAP(1)
    first.update([_detection([[0, 0, 4, 4]], [0.9], [0])], [_target([[0, 0, 4, 4]], [0])])
    second.update([_detection([], [], [])], [_target([], [])])
    first.merge_state(second)
    restored = DetectionMAP(1)
    restored.load_state_dict(first.state_dict())
    assert restored.compute() == first.compute()


def test_multitask_metric_adapter_decodes_raw_detection_output() -> None:
    shapes = ((8, 8), (4, 4), (2, 2))
    classes = [torch.full((1, 1, height, width), -20.0) for height, width in shapes]
    quality = [torch.full((1, 1, height, width), -20.0) for height, width in shapes]
    boxes = [torch.ones(1, 4, height, width) for height, width in shapes]
    classes[0][0, 0, 0, 0] = 20.0
    quality[0][0, 0, 0, 0] = 20.0
    output = RepLiteOutput(
        DetectionOutput(tuple(classes), tuple(boxes), tuple(quality)),
        None,
        None,
        None,
    )
    adapter = MultiTaskMetrics(detection=DetectionMAP(1), detection_reg_max=0)
    adapter.update(
        output,
        {
            "detection": [
                {
                    "boxes": torch.tensor([[0.0, 0.0, 12.0, 12.0]]),
                    "labels": torch.tensor([0], dtype=torch.long),
                    "valid_size": (64, 64),
                }
            ]
        },
    )
    assert adapter.compute()["detection/map50_95"] == pytest.approx(1.0)
