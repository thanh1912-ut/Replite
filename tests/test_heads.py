"""Unit tests for lightweight modular multi-task heads."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch import Tensor, nn

from replite.multitask.heads import (
    ClassificationHead,
    DensePredictionHead,
    DepthHead,
    DetectionHead,
    DetectionOutput,
    ResidualGatedFusion,
    TaskAdapter,
)


def _assert_all_parameters_receive_grad(module: nn.Module, loss: Tensor) -> None:
    loss.backward()
    missing = [
        name
        for name, parameter in module.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    non_finite = [
        name
        for name, parameter in module.named_parameters()
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
    ]
    assert not missing, f"parameters disconnected from loss: {missing}"
    assert not non_finite, f"parameters have non-finite gradients: {non_finite}"


def test_task_adapter_shape_and_all_parameter_gradients() -> None:
    adapter = TaskAdapter(12, 8, num_blocks=2)
    inputs = torch.randn(2, 12, 11, 13, requires_grad=True)

    outputs = adapter(inputs)

    assert outputs.shape == (2, 8, 11, 13)
    _assert_all_parameters_receive_grad(adapter, outputs.square().mean())
    assert inputs.grad is not None


@pytest.mark.parametrize(
    ("reg_max", "box_channels"),
    [(0, 4), (4, 20)],
)
def test_detection_head_shapes_level_count_and_gradients(
    reg_max: int,
    box_channels: int,
) -> None:
    head = DetectionHead(
        (16, 24, 32),
        num_classes=7,
        head_channels=12,
        num_convs=1,
        reg_max=reg_max,
    )
    features = (
        torch.randn(2, 16, 16, 20),
        torch.randn(2, 24, 8, 10),
        torch.randn(2, 32, 4, 5),
    )

    output = head(features)

    assert isinstance(output, DetectionOutput)
    assert (
        len(output.cls_logits) == len(output.box_regression) == len(output.quality) == 3
    )
    for level, spatial_size in enumerate(((16, 20), (8, 10), (4, 5))):
        assert output.cls_logits[level].shape == (2, 7, *spatial_size)
        assert output.box_regression[level].shape == (
            2,
            box_channels,
            *spatial_size,
        )
        assert output.quality[level].shape == (2, 1, *spatial_size)
    if reg_max == 0:
        assert all((boxes > 0).all() for boxes in output.box_regression)

    loss = sum(
        prediction.mean() for predictions in output for prediction in predictions
    )
    _assert_all_parameters_receive_grad(head, loss)


def test_detection_towers_are_shared_but_predictors_are_per_level() -> None:
    head = DetectionHead(8, num_classes=3, head_channels=8, num_convs=2)

    assert len(head.classification_tower) == 2
    assert len(head.regression_tower) == 2
    assert len(head.class_predictors) == 3
    assert len(head.box_predictors) == 3
    assert len(head.quality_predictors) == 3


def test_dense_heads_resize_and_all_parameters_receive_gradients() -> None:
    inputs = torch.randn(2, 8, 7, 9)
    segmentation = DensePredictionHead(8, num_classes=5, hidden_channels=6)
    depth = DepthHead(8, hidden_channels=6)

    segmentation_logits = segmentation(inputs, output_size=(29, 31))
    depth_map = depth(inputs, output_size=(29, 31))

    assert segmentation_logits.shape == (2, 5, 29, 31)
    assert depth_map.shape == (2, 1, 29, 31)
    assert torch.all(depth_map > 0)
    _assert_all_parameters_receive_grad(
        segmentation,
        segmentation_logits.square().mean(),
    )
    _assert_all_parameters_receive_grad(depth, depth_map.square().mean())


def test_bounded_depth_stays_in_configured_interval() -> None:
    head = DepthHead(4, min_depth=0.2, max_depth=25.0)
    output = head(torch.randn(2, 4, 3, 5), output_size=(8, 10))

    assert output.shape == (2, 1, 8, 10)
    assert torch.all(output > 0.2)
    assert torch.all(output < 25.0)


def test_classification_head_shape_and_all_parameter_gradients() -> None:
    head = ClassificationHead(11, num_classes=4, dropout=0.1)
    inputs = torch.randn(3, 11, 5, 7, requires_grad=True)

    logits = head(inputs)

    assert logits.shape == (3, 4)
    _assert_all_parameters_receive_grad(head, logits.square().mean())
    assert inputs.grad is not None


def test_gated_fusion_is_exact_identity_at_initialization_and_when_bypassed() -> None:
    fusion = ResidualGatedFusion(8)
    segmentation = torch.randn(2, 8, 6, 7)
    depth = torch.randn(2, 8, 6, 7)

    fused_segmentation, fused_depth = fusion(segmentation, depth)

    assert torch.equal(fused_segmentation, segmentation)
    assert torch.equal(fused_depth, depth)
    assert fusion.depth_to_seg_scale.item() == 0.0
    assert fusion.seg_to_depth_scale.item() == 0.0

    with torch.no_grad():
        fusion.depth_to_seg_scale.fill_(1.0)
        fusion.seg_to_depth_scale.fill_(1.0)
    bypassed_segmentation, bypassed_depth = fusion(
        segmentation,
        depth,
        enabled=False,
    )
    assert bypassed_segmentation is segmentation
    assert bypassed_depth is depth


def test_gated_fusion_uses_both_pre_fusion_features_without_circular_updates() -> None:
    torch.manual_seed(7)
    fusion = ResidualGatedFusion(5, depth_channels=7, hidden_channels=6).eval()
    with torch.no_grad():
        fusion.depth_to_seg_scale.fill_(0.75)
        fusion.seg_to_depth_scale.fill_(-0.4)

    segmentation_a = torch.randn(2, 5, 4, 6)
    segmentation_b = torch.randn(2, 5, 4, 6)
    depth_a = torch.randn(2, 7, 4, 6)
    depth_b = torch.randn(2, 7, 4, 6)

    seg_from_a, depth_from_a = fusion(segmentation_a, depth_a)
    seg_direction_only = fusion.forward_segmentation(segmentation_a, depth_a)
    depth_direction_only = fusion.forward_depth(segmentation_a, depth_a)
    seg_from_other_seg, _ = fusion(segmentation_b, depth_a)
    _, depth_from_other_depth = fusion(segmentation_a, depth_b)

    torch.testing.assert_close(seg_direction_only, seg_from_a, rtol=0.0, atol=0.0)
    torch.testing.assert_close(depth_direction_only, depth_from_a, rtol=0.0, atol=0.0)

    # The depth-to-seg delta depends on pre-fusion depth, not updated seg.
    torch.testing.assert_close(
        seg_from_a - segmentation_a,
        seg_from_other_seg - segmentation_b,
    )
    # The seg-to-depth delta depends on pre-fusion seg, not updated depth/seg.
    torch.testing.assert_close(
        depth_from_a - depth_a,
        depth_from_other_depth - depth_b,
    )

    expected_seg = segmentation_a + 0.75 * fusion.depth_to_seg(depth_a)
    expected_depth = depth_a - 0.4 * fusion.seg_to_depth(segmentation_a)
    torch.testing.assert_close(seg_from_a, expected_seg)
    torch.testing.assert_close(depth_from_a, expected_depth)


def test_gated_fusion_all_parameters_receive_gradients() -> None:
    fusion = ResidualGatedFusion(6)
    with torch.no_grad():
        fusion.depth_to_seg_scale.fill_(1.0)
        fusion.seg_to_depth_scale.fill_(1.0)
    segmentation = torch.randn(2, 6, 5, 7)
    depth = torch.randn(2, 6, 5, 7)

    fused_segmentation, fused_depth = fusion(segmentation, depth)

    _assert_all_parameters_receive_grad(
        fusion,
        fused_segmentation.square().mean() + fused_depth.square().mean(),
    )


@pytest.mark.parametrize("direction", ["seg_to_depth", "depth_to_seg"])
def test_directional_gated_fusion_physically_prunes_inactive_branch(
    direction: str,
) -> None:
    fusion = ResidualGatedFusion(6, direction=direction)

    assert fusion.uses_seg_to_depth is (direction == "seg_to_depth")
    assert fusion.uses_depth_to_seg is (direction == "depth_to_seg")
    assert hasattr(fusion, "seg_to_depth_scale") is (direction == "seg_to_depth")
    assert hasattr(fusion, "depth_to_seg_scale") is (direction == "depth_to_seg")
    state_keys = set(fusion.state_dict())
    assert any(key.startswith("seg_to_depth.") for key in state_keys) is (
        direction == "seg_to_depth"
    )
    assert any(key.startswith("depth_to_seg.") for key in state_keys) is (
        direction == "depth_to_seg"
    )
    assert all(
        ("seg_to_depth" in name) == (direction == "seg_to_depth")
        for name, _ in fusion.named_parameters()
        if "_to_" in name
    )


def test_default_gated_fusion_keeps_legacy_bidirectional_state_dict() -> None:
    fusion = ResidualGatedFusion(4)
    state = fusion.state_dict()

    assert fusion.direction == "bidirectional"
    assert fusion.detach_source is False
    assert "depth_to_seg_scale" in state
    assert "seg_to_depth_scale" in state
    assert any(key.startswith("depth_to_seg.") for key in state)
    assert any(key.startswith("seg_to_depth.") for key in state)
    restored = ResidualGatedFusion(4)
    incompatible = restored.load_state_dict(state, strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []


def test_directional_fusion_stop_gradient_protects_source_adapter() -> None:
    fusion = ResidualGatedFusion(
        5,
        direction="seg_to_depth",
        detach_source=True,
    )
    with torch.no_grad():
        fusion.seg_to_depth_scale.fill_(1.0)
    segmentation = torch.randn(2, 5, 4, 6, requires_grad=True)
    depth = torch.randn(2, 5, 4, 6, requires_grad=True)

    fused_segmentation, fused_depth = fusion(segmentation, depth)

    assert fused_segmentation is segmentation
    _assert_all_parameters_receive_grad(fusion, fused_depth.square().mean())
    assert segmentation.grad is None
    assert depth.grad is not None


def test_directional_fusion_without_stop_gradient_updates_source() -> None:
    fusion = ResidualGatedFusion(
        5,
        direction="seg_to_depth",
        detach_source=False,
    )
    with torch.no_grad():
        fusion.seg_to_depth_scale.fill_(1.0)
    segmentation = torch.randn(2, 5, 4, 6, requires_grad=True)
    depth = torch.randn(2, 5, 4, 6, requires_grad=True)

    fusion.forward_depth(segmentation, depth).square().mean().backward()

    assert segmentation.grad is not None
    assert torch.count_nonzero(segmentation.grad) > 0


@pytest.mark.parametrize(
    "factory",
    [
        lambda: TaskAdapter(0, 8),
        lambda: TaskAdapter(8, 8, num_blocks=0),
        lambda: DetectionHead((8, 8), num_classes=2),
        lambda: DetectionHead((8, 8, 8), num_classes=0),
        lambda: DetectionHead(8, num_classes=2, num_convs=0),
        lambda: DetectionHead(8, num_classes=2, reg_max=-1),
        lambda: DensePredictionHead(8, num_classes=0),
        lambda: DepthHead(8, min_depth=-0.1),
        lambda: DepthHead(8, min_depth=1.0, max_depth=1.0),
        lambda: ClassificationHead(8, num_classes=2, dropout=1.0),
        lambda: ResidualGatedFusion(8, enabled="yes"),
        lambda: ResidualGatedFusion(8, direction="sideways"),
        lambda: ResidualGatedFusion(8, detach_source=1),
    ],
)
def test_invalid_head_configuration_raises(factory: Callable[[], nn.Module]) -> None:
    with pytest.raises(ValueError):
        factory()


def test_invalid_runtime_shapes_raise_clear_errors() -> None:
    adapter = TaskAdapter(8, 4)
    with pytest.raises(ValueError, match="NCHW"):
        adapter(torch.randn(2, 8, 5))
    with pytest.raises(ValueError, match="8 channels"):
        adapter(torch.randn(2, 7, 5, 5))

    detector = DetectionHead(8, num_classes=2)
    with pytest.raises(ValueError, match="exactly P3, P4, and P5"):
        detector((torch.randn(2, 8, 8, 8),) * 2)
    with pytest.raises(ValueError, match="same batch size"):
        detector(
            (
                torch.randn(2, 8, 8, 8),
                torch.randn(1, 8, 4, 4),
                torch.randn(2, 8, 2, 2),
            )
        )

    segmentation = DensePredictionHead(8, num_classes=2)
    with pytest.raises(ValueError, match="two positive integers"):
        segmentation(torch.randn(2, 8, 4, 4), output_size=(0, 8))

    fusion = ResidualGatedFusion(8)
    with pytest.raises(ValueError, match="batch and spatial"):
        fusion(torch.randn(2, 8, 4, 4), torch.randn(2, 8, 3, 4))
    with pytest.raises(ValueError, match="boolean or None"):
        fusion(
            torch.randn(2, 8, 4, 4),
            torch.randn(2, 8, 4, 4),
            enabled=1,
        )
