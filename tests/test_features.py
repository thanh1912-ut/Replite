"""Feature map shape, out_indices subset and metadata tests."""

from __future__ import annotations

import pytest
import torch

from replite.backbone import FeatureInfo, StageSpec, create_backbone

from .specs import EXPECTED, REQUIRED_CFG_KEYS


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("size", [(224, 224), (512, 1024)])
def test_feature_shapes(backbone_name, size, batch, make_input) -> None:
    height, width = size
    model = create_backbone(backbone_name)
    model.eval()
    inputs = make_input(batch, 3, height, width)
    with torch.no_grad():
        features = model(inputs)

    assert isinstance(features, tuple)
    assert len(features) == 4
    for stage, feature in enumerate(features):
        channels, stride = (
            EXPECTED[backbone_name]["channels"][stage],
            EXPECTED[backbone_name]["reduction"][stage],
        )
        assert feature.shape == (batch, channels, height // stride, width // stride)


@pytest.mark.parametrize("out_indices", [(0,), (1,), (2,), (3,), (0, 2), (1, 3)])
def test_out_indices_subset_returns_requested_stages(
    backbone_name, out_indices, make_input
) -> None:
    spec = EXPECTED[backbone_name]
    model = create_backbone(backbone_name, out_indices=out_indices)
    model.eval()
    with torch.no_grad():
        features = model(make_input(1, 3, 224, 224))

    assert isinstance(features, tuple)
    assert len(features) == len(out_indices)
    for stage, feature in zip(out_indices, features):
        assert feature.shape[1] == spec["channels"][stage]
        assert feature.shape[-2:] == (
            224 // spec["reduction"][stage],
            224 // spec["reduction"][stage],
        )


def test_subset_outputs_match_full_model_outputs(backbone_name, make_input) -> None:
    full = create_backbone(backbone_name)
    subset = create_backbone(backbone_name, out_indices=(1, 3))
    subset.load_state_dict(full.state_dict())
    full.eval()
    subset.eval()
    inputs = make_input(1, 3, 96, 160)
    with torch.no_grad():
        full_features = full(inputs)
        subset_features = subset(inputs)

    assert len(subset_features) == 2
    torch.testing.assert_close(subset_features[0], full_features[1])
    torch.testing.assert_close(subset_features[1], full_features[3])


def test_feature_info_reflects_out_indices(backbone_name) -> None:
    spec = EXPECTED[backbone_name]
    model = create_backbone(backbone_name)
    info = model.feature_info
    assert info.out_indices == (0, 1, 2, 3)
    assert info.channels() == list(spec["channels"])
    assert info.reduction() == list(spec["reduction"])
    assert info.module_name() == list(spec["module_names"])
    assert len(info) == 4


def test_feature_info_subset_reflects_out_indices(backbone_name) -> None:
    spec = EXPECTED[backbone_name]
    model = create_backbone(backbone_name, out_indices=(1, 3))
    info = model.feature_info
    assert info.out_indices == (1, 3)
    assert info.channels() == [spec["channels"][1], spec["channels"][3]]
    assert info.reduction() == [spec["reduction"][1], spec["reduction"][3]]
    assert info.module_name() == [spec["module_names"][1], spec["module_names"][3]]
    assert len(info) == 4
    assert info.num_outputs == 2


def test_feature_info_matches_timm_accessor_semantics(backbone_name) -> None:
    spec = EXPECTED[backbone_name]
    info = create_backbone(backbone_name, out_indices=(1, 3)).feature_info

    assert info.channels(0) == spec["channels"][0]
    assert info.channels([1, 3]) == [spec["channels"][1], spec["channels"][3]]
    assert info.get("num_chs") == [spec["channels"][1], spec["channels"][3]]
    assert info.get("reduction", 2) == spec["reduction"][2]
    assert info.get_dicts(["num_chs"], [0, 2]) == [
        {"num_chs": spec["channels"][0]},
        {"num_chs": spec["channels"][2]},
    ]
    assert info[2]["module"] == spec["module_names"][2]

    other = info.from_other((0, 2))
    assert other.out_indices == (0, 2)
    assert other.channels() == [spec["channels"][0], spec["channels"][2]]
    other.info[0]["num_chs"] = -1
    assert info.info[0]["num_chs"] == spec["channels"][0]


def test_feature_info_direct_construction_rejects_invalid_indices() -> None:
    stages = (
        StageSpec("blocks.0", 8, 4, 0),
        StageSpec("blocks.1", 16, 8, 1),
    )
    for invalid in ((2,), (-1,), (True,), (1.0,), (0, 0), (1, 0)):
        with pytest.raises(ValueError):
            FeatureInfo(stages, invalid)


def test_pretrained_cfg_metadata(backbone_name) -> None:
    model = create_backbone(backbone_name)
    cfg = model.pretrained_cfg
    assert isinstance(cfg, dict)
    for key in REQUIRED_CFG_KEYS:
        assert key in cfg, f"pretrained_cfg is missing {key!r}"
    assert cfg == EXPECTED[backbone_name]["pretrained_cfg"]
    assert len(cfg["sha256"]) == 64


def test_pretrained_cfg_is_not_mutable_from_outside(backbone_name) -> None:
    model = create_backbone(backbone_name)
    cfg = model.pretrained_cfg
    cfg["sha256"] = "tampered"
    assert model.pretrained_cfg["sha256"] != "tampered"
