"""Tests for the anchor-free RepLite detection training contract."""

from __future__ import annotations

import math

import pytest
import torch

from replite.multitask.heads import DetectionHead, DetectionOutput
from replite.training.detection import (
    DetectionCriterion,
    assign_fcos_targets,
    class_aware_nms,
    decode_box_regression,
    decode_detections,
    generalized_box_iou_aligned,
    make_detection_points,
)


def _single_point_assignment(
    boxes: torch.Tensor,
    labels: torch.Tensor,
    *,
    point: tuple[float, float] = (20.0, 20.0),
    stride: float = 8.0,
    level: int = 0,
    reg_max: int = 0,
    ignore_boxes: torch.Tensor | None = None,
):
    return assign_fcos_targets(
        torch.tensor([point]),
        torch.tensor([stride]),
        torch.tensor([level], dtype=torch.long),
        boxes,
        labels,
        (256, 256),
        reg_max=reg_max,
        ignore_boxes=ignore_boxes,
    )


def _prediction_maps(
    *,
    batch_size: int = 1,
    num_classes: int = 2,
    reg_max: int = 0,
    shapes: tuple[tuple[int, int], ...] = ((8, 8), (4, 4), (2, 2)),
) -> DetectionOutput:
    box_channels = 4 if reg_max == 0 else 4 * (reg_max + 1)
    classes = tuple(
        torch.full(
            (batch_size, num_classes, height, width),
            -4.0,
            requires_grad=True,
        )
        for height, width in shapes
    )
    boxes = tuple(
        torch.full(
            (batch_size, box_channels, height, width),
            1.0 if reg_max == 0 else 0.0,
            requires_grad=True,
        )
        for height, width in shapes
    )
    quality = tuple(
        torch.full(
            (batch_size, 1, height, width),
            -4.0,
            requires_grad=True,
        )
        for height, width in shapes
    )
    return DetectionOutput(classes, boxes, quality)


def test_points_are_cell_centers_in_p3_p4_p5_order() -> None:
    result = make_detection_points(((2, 3), (1, 2), (1, 1)))

    assert result.points.shape == (9, 2)
    torch.testing.assert_close(result.points[0], torch.tensor([4.0, 4.0]))
    torch.testing.assert_close(result.points[5], torch.tensor([20.0, 12.0]))
    torch.testing.assert_close(result.points[6], torch.tensor([8.0, 8.0]))
    torch.testing.assert_close(result.points[-1], torch.tensor([16.0, 16.0]))
    assert result.strides.tolist() == [8.0] * 6 + [16.0] * 2 + [32.0]
    assert result.levels.tolist() == [0] * 6 + [1] * 2 + [2]


def test_assignment_uses_smallest_area_and_first_target_for_equal_area() -> None:
    nested = _single_point_assignment(
        torch.tensor(
            [
                [0.0, 0.0, 40.0, 40.0],
                [10.0, 10.0, 30.0, 30.0],
            ]
        ),
        torch.tensor([1, 2], dtype=torch.long),
    )
    assert nested.labels.item() == 2
    assert nested.matched_gt_indices.item() == 1

    tied = _single_point_assignment(
        torch.tensor(
            [
                [10.0, 10.0, 30.0, 30.0],
                [10.0, 10.0, 30.0, 30.0],
            ]
        ),
        torch.tensor([7, 9], dtype=torch.long),
    )
    assert tied.labels.item() == 7
    assert tied.matched_gt_indices.item() == 0


def test_assignment_respects_scale_ignore_padding_and_empty_targets() -> None:
    # max(LTRB)==64 belongs to P4's [64,128), not P3's [0,64).
    boxes = torch.tensor([[0.0, 0.0, 128.0, 128.0]])
    labels = torch.tensor([3], dtype=torch.long)
    p3 = _single_point_assignment(
        boxes, labels, point=(64.0, 64.0), stride=8.0, level=0
    )
    p4 = _single_point_assignment(
        boxes, labels, point=(64.0, 64.0), stride=16.0, level=1
    )
    assert not p3.positive_mask.item()
    assert p3.num_unmatched.item() == 1
    assert p4.positive_mask.item()

    points = torch.tensor([[4.0, 4.0], [20.0, 4.0], [36.0, 4.0]])
    ignored = assign_fcos_targets(
        points,
        torch.full((3,), 8.0),
        torch.zeros(3, dtype=torch.long),
        torch.empty(0, 4),
        torch.empty(0, dtype=torch.long),
        (16, 32),
        ignore_boxes=torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
    )
    assert ignored.labels.tolist() == [-2, -1, -2]
    assert ignored.valid_mask.tolist() == [False, True, False]
    assert ignored.num_targets.item() == 0
    assert ignored.num_unmatched.item() == 0


def test_tiny_box_without_a_grid_center_is_reported_unmatched() -> None:
    result = _single_point_assignment(
        torch.tensor([[5.0, 5.0, 7.0, 7.0]]),
        torch.tensor([0], dtype=torch.long),
        point=(4.0, 4.0),
    )

    assert not result.positive_mask.item()
    assert result.labels.item() == -1
    assert result.num_unmatched.item() == 1


def test_dfl_assignment_accepts_exact_reg_max_bin_boundary() -> None:
    result = _single_point_assignment(
        torch.tensor([[0.0, 0.0, 32.0, 32.0]]),
        torch.tensor([0], dtype=torch.long),
        point=(16.0, 16.0),
        stride=8.0,
        reg_max=2,
    )

    assert result.positive_mask.item()
    torch.testing.assert_close(result.ltrb_cells[0], torch.full((4,), 2.0))


def test_direct_and_distributional_box_decoding() -> None:
    point = torch.tensor([[4.0, 4.0]])
    stride = torch.tensor([8.0])
    direct = decode_box_regression(
        torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
        point,
        stride,
        reg_max=0,
    )
    torch.testing.assert_close(
        direct,
        torch.tensor([[-4.0, -12.0, 28.0, 36.0]]),
    )

    logits = torch.full((1, 4, 3), -100.0)
    logits[:, :, 1] = 100.0
    distributional = decode_box_regression(
        logits,
        point,
        stride,
        reg_max=2,
    )
    torch.testing.assert_close(
        distributional,
        torch.tensor([[-4.0, -4.0, 12.0, 12.0]]),
    )


@pytest.mark.parametrize("reg_max", [0, 16])
def test_criterion_is_finite_and_connects_every_head_parameter(reg_max: int) -> None:
    torch.manual_seed(11)
    head = DetectionHead(
        8,
        num_classes=3,
        head_channels=8,
        num_convs=1,
        reg_max=reg_max,
    )
    predictions = head(
        (
            torch.randn(2, 8, 8, 8),
            torch.randn(2, 8, 4, 4),
            torch.randn(2, 8, 2, 2),
        )
    )
    criterion = DetectionCriterion(3, reg_max=reg_max)
    losses = criterion(
        predictions,
        (
            {
                "boxes": torch.tensor([[8.0, 8.0, 40.0, 40.0]]),
                "labels": torch.tensor([1], dtype=torch.long),
            },
            {
                "boxes": torch.empty(0, 4),
                "labels": torch.empty(0, dtype=torch.long),
            },
        ),
        image_size=(64, 64),
    )

    assert losses.num_positive.item() > 0
    assert losses.num_targets.item() == 1
    assert losses.num_unmatched.item() == 0
    assert all(
        torch.isfinite(value)
        for value in (
            losses.total,
            losses.classification,
            losses.box,
            losses.quality,
            losses.dfl,
        )
    )
    if reg_max == 0:
        assert losses.dfl.item() == 0.0
    else:
        assert losses.dfl.item() > 0.0

    losses.total.backward()
    missing = [
        name
        for name, parameter in head.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert not missing, f"parameters disconnected from detection loss: {missing}"


def test_all_empty_targets_keep_regression_and_quality_in_autograd_graph() -> None:
    head = DetectionHead(8, num_classes=2, head_channels=8, num_convs=1)
    predictions = head(
        (
            torch.randn(1, 8, 8, 8),
            torch.randn(1, 8, 4, 4),
            torch.randn(1, 8, 2, 2),
        )
    )
    losses = DetectionCriterion(2)(
        predictions,
        ({"boxes": torch.empty(0, 4), "labels": torch.empty(0, dtype=torch.long)},),
        image_size=(64, 64),
    )

    assert losses.num_positive.item() == 0
    assert torch.isfinite(losses.total)
    losses.total.backward()
    assert all(
        parameter.grad is not None
        for parameter in head.parameters()
        if parameter.requires_grad
    )


def test_giou_is_one_for_identical_boxes_and_below_zero_when_disjoint() -> None:
    actual = generalized_box_iou_aligned(
        torch.tensor([[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 1.0, 1.0]]),
        torch.tensor([[0.0, 0.0, 2.0, 2.0], [2.0, 2.0, 3.0, 3.0]]),
    )

    torch.testing.assert_close(actual[0], torch.tensor(1.0))
    assert actual[1] < 0


def test_class_aware_nms_suppresses_only_same_class_and_is_stable() -> None:
    boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [0.0, 0.0, 10.0, 10.0],
            [0.0, 0.0, 10.0, 10.0],
        ]
    )
    scores = torch.tensor([0.9, 0.9, 0.8])
    labels = torch.tensor([0, 0, 1], dtype=torch.long)

    keep = class_aware_nms(boxes, scores, labels, 0.5)

    assert keep.tolist() == [0, 2]


def test_decode_uses_stable_pre_nms_tie_order_and_keeps_different_classes() -> None:
    classes = (
        torch.tensor([[[[10.0, 10.0]], [[-20.0, -20.0]]]]),
        torch.full((1, 2, 1, 1), -20.0),
        torch.full((1, 2, 1, 1), -20.0),
    )
    boxes = (
        torch.full((1, 4, 1, 2), 0.5),
        torch.full((1, 4, 1, 1), 0.5),
        torch.full((1, 4, 1, 1), 0.5),
    )
    quality = (
        torch.full((1, 1, 1, 2), 10.0),
        torch.full((1, 1, 1, 1), -20.0),
        torch.full((1, 1, 1, 1), -20.0),
    )
    prediction = DetectionOutput(classes, boxes, quality)

    first_only = decode_detections(
        prediction,
        ((16, 16),),
        reg_max=0,
        pre_nms_topk=1,
        score_threshold=0.1,
    )[0]
    torch.testing.assert_close(
        first_only["boxes"], torch.tensor([[0.0, 0.0, 8.0, 8.0]])
    )
    assert first_only["labels"].tolist() == [0]

    two_classes = list(classes)
    two_classes[0] = torch.full((1, 2, 1, 2), -20.0)
    two_classes[0][:, :, 0, 0] = 10.0
    two_class_prediction = DetectionOutput(tuple(two_classes), boxes, quality)
    decoded = decode_detections(
        two_class_prediction,
        ((16, 16),),
        reg_max=0,
        pre_nms_topk=10,
        score_threshold=0.1,
    )[0]
    assert decoded["labels"].tolist() == [0, 1]
    torch.testing.assert_close(decoded["boxes"][0], decoded["boxes"][1])


@pytest.mark.parametrize(
    "factory",
    [
        lambda: DetectionCriterion(2, center_radius=float("nan")),
        lambda: DetectionCriterion(2, focal_gamma=float("inf")),
        lambda: decode_detections(
            _prediction_maps(),
            ((64, 64),),
            reg_max=0,
            score_threshold=float("nan"),
        ),
    ],
)
def test_non_finite_configuration_is_rejected(factory) -> None:
    with pytest.raises(ValueError):
        factory()


def test_invalid_feature_shape_and_target_contract_fail_clearly() -> None:
    criterion = DetectionCriterion(2)
    with pytest.raises(ValueError, match="feature shapes"):
        criterion(
            _prediction_maps(shapes=((7, 8), (4, 4), (2, 2))),
            ({"boxes": torch.empty(0, 4), "labels": torch.empty(0, dtype=torch.long)},),
            image_size=(64, 64),
        )
    with pytest.raises(ValueError, match="torch.long"):
        criterion(
            _prediction_maps(),
            ({"boxes": torch.tensor([[0.0, 0.0, 8.0, 8.0]]), "labels": torch.tensor([0.0])},),
            image_size=(64, 64),
        )

