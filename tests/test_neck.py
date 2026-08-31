"""Tests for recurrent multi-task feature refinement and fusion."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
import torch
from torch import Tensor

from replite.multitask.neck import NeckOutput, NeckState, RecurrentMultiTaskNeck


CHANNEL_SETS = (
    pytest.param((8, 16, 24, 48), id="mobilenetv3-small-native"),
    pytest.param((32, 64, 96, 128), id="mobilenetv4-conv-small-native"),
)


def _features(
    channels: Sequence[int],
    *,
    batch: int = 2,
    requires_grad: bool = False,
    seed: int = 123,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    sizes = ((17, 19), (9, 10), (5, 5), (3, 3))
    return tuple(
        torch.randn(
            batch, width, height, spatial_width, generator=generator
        ).requires_grad_(requires_grad)
        for width, (height, spatial_width) in zip(channels, sizes)
    )


@pytest.mark.parametrize("channels", CHANNEL_SETS)
def test_odd_native_pyramid_shapes_and_state(channels) -> None:
    neck = RecurrentMultiTaskNeck(channels, refine_steps=2).eval()
    features = _features(channels)

    with torch.no_grad():
        output, state = neck.refine(features)

    assert isinstance(output, NeckOutput)
    assert isinstance(state, NeckState)
    assert output.d3 is not None and output.d3.shape == (2, 48, 9, 10)
    assert output.d4 is not None and output.d4.shape == (2, 48, 5, 5)
    assert output.d5 is not None and output.d5.shape == (2, 48, 3, 3)
    assert output.f2 is not None and output.f2.shape == (2, 32, 17, 19)
    assert output.r4.shape == (2, 48, 5, 5)
    assert output.r5.shape == (2, 64, 3, 3)
    assert output.detection == (output.d3, output.d4, output.d5)
    for tensor in state.level4:
        assert tensor.shape == output.r4.shape
    for tensor in state.level5:
        assert tensor.shape == output.r5.shape


def _assert_output_close(actual: NeckOutput, expected: NeckOutput) -> None:
    for actual_value, expected_value in zip(actual, expected):
        assert (actual_value is None) == (expected_value is None)
        if actual_value is not None and expected_value is not None:
            torch.testing.assert_close(actual_value, expected_value)


def _assert_state_close(actual: NeckState, expected: NeckState) -> None:
    for actual_level, expected_level in zip(actual, expected):
        for actual_value, expected_value in zip(actual_level, expected_level):
            torch.testing.assert_close(actual_value, expected_value)


def test_static_refinement_matches_explicit_repeated_streaming_steps() -> None:
    channels = (8, 16, 24, 48)
    neck = RecurrentMultiTaskNeck(channels, refine_steps=3).eval()
    features = _features(channels, batch=1)

    with torch.no_grad():
        refined_output, refined_state = neck.refine(features, steps=3)
        stream_state = None
        for _ in range(3):
            stream_output, stream_state = neck.step(features, stream_state)

    _assert_output_close(stream_output, refined_output)
    assert stream_state is not None
    _assert_state_close(stream_state, refined_state)


def test_forward_is_static_refinement() -> None:
    channels = (8, 16, 24, 48)
    neck = RecurrentMultiTaskNeck(channels, refine_steps=2).eval()
    features = _features(channels, batch=1)
    with torch.no_grad():
        direct_output, direct_state = neck.refine(features)
        forward_output, forward_state = neck(features)
    _assert_output_close(forward_output, direct_output)
    _assert_state_close(forward_state, direct_state)


def test_detection_and_dense_losses_reach_every_trainable_parameter() -> None:
    channels = (8, 16, 24, 48)
    neck = RecurrentMultiTaskNeck(
        channels,
        recurrent_channels=(12, 16),
        detection_channels=12,
        dense_channels=8,
        refine_steps=2,
    ).train()
    features = _features(channels, requires_grad=True)

    output, _ = neck(features)
    task_outputs = (output.d3, output.d4, output.d5, output.f2)
    assert all(value is not None for value in task_outputs)
    loss = sum(
        value.float().square().mean() for value in task_outputs if value is not None
    )
    loss.backward()

    for name, parameter in neck.named_parameters():
        assert parameter.grad is not None, f"unused trainable parameter: {name}"
        assert torch.isfinite(parameter.grad).all(), f"non-finite gradient: {name}"
    for feature in features:
        assert feature.grad is not None
        assert torch.isfinite(feature.grad).all()


@pytest.mark.parametrize(
    ("enable_detection", "enable_dense"),
    ((False, True), (True, False), (False, False)),
)
def test_disabled_task_paths_are_physically_absent(
    enable_detection: bool, enable_dense: bool
) -> None:
    channels = (8, 16, 24, 48)
    neck = RecurrentMultiTaskNeck(
        channels,
        recurrent_channels=(8, 8),
        detection_channels=8,
        dense_channels=8,
        refine_steps=1,
        use_sppf=True,
        enable_detection=enable_detection,
        enable_dense=enable_dense,
    ).eval()
    parameter_names = tuple(name for name, _ in neck.named_parameters())

    assert hasattr(neck, "detection_path") is enable_detection
    assert hasattr(neck, "dense_path") is enable_dense
    has_detection_parameters = any(
        name.startswith("detection_path.") for name in parameter_names
    )
    has_dense_parameters = any(
        name.startswith("dense_path.") for name in parameter_names
    )
    assert has_detection_parameters is enable_detection
    assert has_dense_parameters is enable_dense

    with torch.no_grad():
        output, _ = neck.refine(_features(channels, batch=1))
    assert (output.d3 is not None) is enable_detection
    assert (output.d4 is not None) is enable_detection
    assert (output.d5 is not None) is enable_detection
    assert (output.f2 is not None) is enable_dense
    assert (output.detection is not None) is enable_detection
    assert output.r4 is not None and output.r5 is not None


def test_dense_only_neck_prunes_c5_and_level5_state() -> None:
    neck = RecurrentMultiTaskNeck(
        (8, 16, 24),
        recurrent_channels=(12, 16),
        dense_channels=8,
        enable_detection=False,
        enable_dense=True,
        enable_level4=True,
        enable_level5=False,
    ).eval()
    parameter_names = tuple(name for name, _ in neck.named_parameters())
    assert not hasattr(neck, "c5_projection")
    assert not hasattr(neck, "recurrent5")
    assert not any(name.startswith("c5_projection.") for name in parameter_names)
    assert not any(name.startswith("recurrent5.") for name in parameter_names)

    with torch.no_grad():
        output, state = neck.refine(_features((8, 16, 24), batch=1))
    assert output.f2 is not None
    assert output.r4 is not None
    assert output.r5 is None
    assert state.level4 is not None
    assert state.level5 is None


def test_streaming_state_validation_uses_autocast_projection_dtype() -> None:
    neck = RecurrentMultiTaskNeck(
        (8, 16, 24, 48),
        recurrent_channels=(12, 16),
        detection_channels=12,
        dense_channels=8,
    ).eval()
    features = _features((8, 16, 24, 48), batch=1)

    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        _, state = neck.step(features)
        output, continued = neck.step(features, state)

    assert output.d3 is not None
    assert continued.level4 is not None and continued.level5 is not None
    assert continued.level4[0].dtype == torch.bfloat16
    assert continued.level5[0].dtype == torch.bfloat16


def test_optional_sppf_is_only_materialized_when_requested() -> None:
    channels = (8, 16, 24, 48)
    plain = RecurrentMultiTaskNeck(channels, use_sppf=False, refine_steps=1)
    pooled = RecurrentMultiTaskNeck(channels, use_sppf=True, refine_steps=1).eval()

    assert "sppf" not in plain.detection_path
    assert "sppf" in pooled.detection_path
    with torch.no_grad():
        output, _ = pooled.refine(_features(channels, batch=1))
    assert output.d5 is not None and output.d5.shape[-2:] == (3, 3)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"in_channels": (8, 16, 24)},
        {"in_channels": (8, 16, 24, 0)},
        {"in_channels": (8, 16, True, 48)},
        {"in_channels": (8, 16, 24, 48), "recurrent_channels": (48,)},
        {"in_channels": (8, 16, 24, 48), "detection_channels": 0},
        {"in_channels": (8, 16, 24, 48), "dense_channels": False},
        {"in_channels": (8, 16, 24, 48), "refine_steps": 0},
        {"in_channels": (8, 16, 24, 48), "use_sppf": 1},
        {"in_channels": (8, 16, 24, 48), "enable_detection": 1},
        {"in_channels": (8, 16, 24, 48), "enable_dense": None},
    ),
)
def test_invalid_constructor_configuration_is_rejected(kwargs) -> None:
    with pytest.raises(ValueError):
        RecurrentMultiTaskNeck(**kwargs)


def test_invalid_feature_count_channels_rank_and_pyramid_are_rejected() -> None:
    channels = (8, 16, 24, 48)
    neck = RecurrentMultiTaskNeck(channels, refine_steps=1)
    valid = list(_features(channels))

    with pytest.raises(ValueError, match="exactly"):
        neck.refine(valid[:3])

    wrong_channels = valid.copy()
    wrong_channels[1] = torch.randn(2, 15, 9, 10)
    with pytest.raises(ValueError, match="16 channels"):
        neck.refine(wrong_channels)

    wrong_rank = valid.copy()
    wrong_rank[2] = torch.randn(2, 24, 5)
    with pytest.raises(ValueError, match="4D"):
        neck.refine(wrong_rank)

    wrong_pyramid = valid.copy()
    wrong_pyramid[2] = torch.randn(2, 24, 4, 5)
    with pytest.raises(ValueError, match="ceil-half"):
        neck.refine(wrong_pyramid)

    with pytest.raises(ValueError, match="positive integer"):
        neck.refine(valid, steps=0)


def test_invalid_streaming_state_is_rejected_before_recurrence() -> None:
    channels = (8, 16, 24, 48)
    neck = RecurrentMultiTaskNeck(channels, refine_steps=1)
    features = _features(channels)
    bad_level4 = (
        torch.zeros(2, 48, 4, 5),
        torch.zeros(2, 48, 4, 5),
    )
    valid_level5 = (
        torch.zeros(2, 64, 3, 3),
        torch.zeros(2, 64, 3, 3),
    )
    with pytest.raises(ValueError, match="level4.h"):
        neck.step(features, NeckState(bad_level4, valid_level5))
