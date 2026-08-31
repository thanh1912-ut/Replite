"""Registry and factory contract tests."""

from __future__ import annotations

import pytest

from replite.backbone import (
    MobileNetV3Backbone,
    MobileNetV4Backbone,
    create_backbone,
    list_backbones,
)

from .specs import BACKBONE_NAMES


def test_registry_contains_exactly_two_canonical_names() -> None:
    assert list_backbones() == BACKBONE_NAMES


def test_list_backbones_is_stable_across_calls() -> None:
    assert list_backbones() == list_backbones() == tuple(list_backbones())


def test_unknown_name_raises_and_lists_valid_names() -> None:
    with pytest.raises(ValueError) as exc_info:
        create_backbone("resnet50")
    message = str(exc_info.value)
    for name in BACKBONE_NAMES:
        assert name in message


@pytest.mark.parametrize("invalid_name", [None, 123, [], {}])
def test_non_string_name_raises_value_error(invalid_name) -> None:
    with pytest.raises(ValueError, match="must be a string"):
        create_backbone(invalid_name)


@pytest.mark.parametrize("name", BACKBONE_NAMES)
def test_create_backbone_returns_matching_wrapper(name: str) -> None:
    expected_type = {
        "mobilenetv3_small_050": MobileNetV3Backbone,
        "mobilenetv4_conv_small": MobileNetV4Backbone,
    }[name]
    model = create_backbone(name)
    assert isinstance(model, expected_type)
    assert model.out_indices == (0, 1, 2, 3)


@pytest.mark.parametrize(
    "out_indices",
    [
        (),
        (4,),
        (-1,),
        (0, 0),
        (1, 1),
        (2, 1),
        (3, 2, 1),
        (0, 2, 1),
        (0, 1, 2, 3, 0),
        (0, 1, 4),
        (True,),
        (False,),
        (1.0,),
        ("1",),
        ([0],),
        None,
        1,
    ],
)
def test_invalid_out_indices_are_rejected(out_indices) -> None:
    for name in BACKBONE_NAMES:
        with pytest.raises(ValueError):
            create_backbone(name, out_indices=out_indices)


@pytest.mark.parametrize(
    "out_indices",
    [(0,), (1,), (2,), (3,), (0, 1), (0, 2), (1, 3), (0, 1, 2, 3)],
)
def test_valid_out_indices_are_accepted(out_indices) -> None:
    for name in BACKBONE_NAMES:
        model = create_backbone(name, out_indices=out_indices)
        assert model.out_indices == out_indices


def test_list_input_is_normalized_to_tuple() -> None:
    model = create_backbone("mobilenetv3_small_050", out_indices=[0, 1])
    assert model.out_indices == (0, 1)
    assert isinstance(model.out_indices, tuple)


def test_extra_repr_reports_the_selected_stage_names() -> None:
    model = create_backbone("mobilenetv4_conv_small", out_indices=(1, 3))
    representation = model.extra_repr()
    assert "stages=('C3', 'C5')" in representation
    assert "weights_loaded=False" in representation
