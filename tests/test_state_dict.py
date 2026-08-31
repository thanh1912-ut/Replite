"""Parameter count and state dict hygiene tests."""

from __future__ import annotations

import timm

from replite.backbone import create_backbone

from .specs import EXPECTED


def test_native_trunk_parameter_count_is_exact(backbone_name, expected_for) -> None:
    model = create_backbone(backbone_name)
    total = sum(parameter.numel() for parameter in model.parameters())
    assert total == expected_for["param_count"]


def test_shallow_out_indices_trim_unused_parameters(
    backbone_name, expected_for
) -> None:
    full = create_backbone(backbone_name)
    subset = create_backbone(backbone_name, out_indices=(0,))
    total_full = sum(parameter.numel() for parameter in full.parameters())
    total_subset = sum(parameter.numel() for parameter in subset.parameters())
    assert total_full == expected_for["param_count"]
    assert total_subset < total_full
    assert len(subset.blocks) == expected_for["feature_block_ends"][0] + 1


def test_state_dict_matches_native_trunk_of_full_timm_model(
    backbone_name, expected_for
) -> None:
    model = create_backbone(backbone_name)
    trunk_groups = expected_for["trunk_block_groups"]
    trunk_prefixes = ("conv_stem.", "bn1.") + tuple(
        f"blocks.{group}." for group in range(trunk_groups)
    )
    full = timm.create_model(expected_for["full_timm_arch"], pretrained=False)
    trunk_keys = {key for key in full.state_dict() if key.startswith(trunk_prefixes)}

    wrapper_state = model.state_dict()
    assert set(wrapper_state) == trunk_keys
    full_state = full.state_dict()
    for key, value in wrapper_state.items():
        assert value.shape == full_state[key].shape


def test_state_dict_has_no_classifier_head_or_projection(
    backbone_name, expected_for
) -> None:
    model = create_backbone(backbone_name)
    trunk_groups = expected_for["trunk_block_groups"]
    removed_groups = expected_for["removed_block_groups"]
    forbidden_prefixes = (
        "classifier",
        "conv_head",
        "bn2",
        "global_pool",
        "flatten",
        "norm_head",
        "act2",
    ) + tuple(f"blocks.{group}." for group in removed_groups)

    for key in model.state_dict():
        assert not key.startswith(forbidden_prefixes), f"leaked head key: {key}"


def test_outputs_never_have_projection_channel_counts(
    backbone_name, expected_for, make_input
) -> None:
    """No returned feature map may have the projected C5 width (288 / 960)."""
    import torch

    model = create_backbone(backbone_name)
    model.eval()
    with torch.no_grad():
        features = model(make_input(1, 3, 224, 224))
    for stage, feature in enumerate(features):
        assert feature.shape[1] == expected_for["channels"][stage]
        for projection_channels in expected_for["projection_channels"]:
            assert feature.shape[1] != projection_channels
