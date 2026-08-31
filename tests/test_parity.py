"""Parity tests: wrapper forward vs native stages of the full timm model."""

from __future__ import annotations

import pytest
import timm
import torch

from replite.backbone import create_backbone

from .specs import EXPECTED


def _full_model_with_wrapper_weights(
    backbone_name: str, model: torch.nn.Module
) -> torch.nn.Module:
    """Build the full timm model and copy the wrapper's trunk weights into it."""
    spec = EXPECTED[backbone_name]
    full = timm.create_model(spec["full_timm_arch"], pretrained=False)
    incompatible = full.load_state_dict(model.state_dict(), strict=False)
    trunk_prefixes = ("conv_stem.", "bn1.") + tuple(
        f"blocks.{group}." for group in range(spec["trunk_block_groups"])
    )
    assert incompatible.unexpected_keys == []
    full_state = full.state_dict()
    # BatchNorm silently backfills a missing integer ``num_batches_tracked``
    # buffer, so only floating-point entries can ever show up as missing.
    expected_missing = {
        key
        for key, value in full_state.items()
        if not key.startswith(trunk_prefixes) and value.is_floating_point()
    }
    assert set(incompatible.missing_keys) == expected_missing
    return full


def _full_model_stage_features(
    full: torch.nn.Module, x: torch.Tensor, backbone_name: str
):
    """Run the native trunk stages of the full timm model step by step."""
    block_ends = EXPECTED[backbone_name]["feature_block_ends"]
    with torch.no_grad():
        h = full.bn1(full.conv_stem(x))
        features = []
        for group_index, group in enumerate(full.blocks):
            h = group(h)
            if group_index in block_ends:
                features.append(h)
    return tuple(features)


@pytest.mark.parametrize("size", [(224, 224), (160, 256)])
def test_parity_with_full_timm_trunk(backbone_name, size, make_input) -> None:
    model = create_backbone(backbone_name)
    model.eval()
    full = _full_model_with_wrapper_weights(backbone_name, model)
    full.eval()

    height, width = size
    inputs = make_input(2, 3, height, width, seed=777)
    with torch.no_grad():
        wrapper_features = model(inputs)
    full_features = _full_model_stage_features(full, inputs, backbone_name)

    assert len(wrapper_features) == len(full_features) == 4
    for wrapper_feature, full_feature in zip(wrapper_features, full_features):
        torch.testing.assert_close(wrapper_feature, full_feature, rtol=1e-5, atol=1e-6)
