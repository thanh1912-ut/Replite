"""Network tests: real pretrained checkpoint download, verification and loading.

These tests download the pinned ``model.safetensors`` from the Hugging Face
Hub. They are excluded from the default pytest run via the ``network`` marker;
run them explicitly with ``pytest -m network``.
"""

from __future__ import annotations

import pytest
import timm
import torch

from replite.backbone import (
    PretrainedWeightsSpec,
    create_backbone,
    load_verified_state_dict,
)
from replite.backbone.weights import ChecksumMismatchError, file_sha256
from replite.multitask import RepLiteConfig, RepLiteMultiTaskModel, TaskConfig
from safetensors.torch import load_file

from .specs import EXPECTED

pytestmark = pytest.mark.network


def _spec_from_cfg(cfg: dict) -> PretrainedWeightsSpec:
    return PretrainedWeightsSpec(
        architecture=cfg["architecture"],
        repository=cfg["repository"],
        revision=cfg["revision"],
        sha256=cfg["sha256"],
        input_size=cfg["input_size"],
        interpolation=cfg["interpolation"],
        mean=cfg["mean"],
        std=cfg["std"],
    )


def test_pretrained_download_verify_and_strict_load(
    backbone_name, expected_for, make_input
) -> None:
    model = create_backbone(backbone_name, pretrained=True)
    cfg = model.pretrained_cfg
    assert cfg == expected_for["pretrained_cfg"]

    # Independent re-download at the pinned revision and SHA-256 verification.
    state = load_verified_state_dict(_spec_from_cfg(cfg))
    full = timm.create_model(expected_for["full_timm_arch"], pretrained=False)
    full.load_state_dict(state, strict=True)

    # The wrapper parameters must match the verified checkpoint exactly.
    wrapper_state = model.state_dict()
    assert set(wrapper_state) <= set(state)
    for key, value in wrapper_state.items():
        torch.testing.assert_close(value, state[key], rtol=0.0, atol=0.0)

    # Native trunk only: parameter count must stay identical to random init.
    total = sum(parameter.numel() for parameter in model.parameters())
    assert total == expected_for["param_count"]

    # BN statistics must come from the checkpoint, not stay at init values.
    first_norm = model.bn1
    assert first_norm.num_batches_tracked.item() > 0


def test_pretrained_parity_with_full_timm_model(
    backbone_name, expected_for, make_input
) -> None:
    model = create_backbone(backbone_name, pretrained=True)
    model.eval()
    full = timm.create_model(expected_for["full_timm_arch"], pretrained=False)
    state = load_verified_state_dict(_spec_from_cfg(model.pretrained_cfg))
    full.load_state_dict(state, strict=True)
    full.eval()

    inputs = make_input(1, 3, 224, 224, seed=4242)
    with torch.no_grad():
        wrapper_features = model(inputs)
        h = full.bn1(full.conv_stem(inputs))
        full_features = []
        for group_index, group in enumerate(full.blocks):
            h = group(h)
            if group_index in expected_for["feature_block_ends"]:
                full_features.append(h)

    for wrapper_feature, full_feature in zip(wrapper_features, full_features):
        torch.testing.assert_close(wrapper_feature, full_feature, rtol=1e-5, atol=1e-6)


def test_pretrained_shapes(backbone_name, expected_for, make_input) -> None:
    model = create_backbone(backbone_name, pretrained=True)
    model.eval()
    with torch.no_grad():
        features = model(make_input(1, 3, 512, 1024, seed=1))
    for stage, feature in enumerate(features):
        channels = expected_for["channels"][stage]
        reduction = expected_for["reduction"][stage]
        assert feature.shape == (1, channels, 512 // reduction, 1024 // reduction)


def test_checksum_mismatch_is_rejected(backbone_name) -> None:
    cfg = create_backbone(backbone_name).pretrained_cfg
    spec = _spec_from_cfg(cfg)
    tampered = PretrainedWeightsSpec(
        architecture=spec.architecture,
        repository=spec.repository,
        revision=spec.revision,
        sha256="0" * 64,
    )
    with pytest.raises(ChecksumMismatchError):
        load_verified_state_dict(tampered, local_files_only=True)


def test_safetensors_file_matches_recorded_sha256(backbone_name) -> None:
    cfg = create_backbone(backbone_name).pretrained_cfg
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=cfg["repository"],
        filename="model.safetensors",
        revision=cfg["revision"],
    )
    assert file_sha256(path) == cfg["sha256"]
    assert load_file(path)


def test_verified_local_checkpoint_path_strict_loads(backbone_name) -> None:
    cfg = create_backbone(backbone_name).pretrained_cfg
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=cfg["repository"],
        filename=cfg["filename"],
        revision=cfg["revision"],
        local_files_only=True,
    )
    model = create_backbone(
        backbone_name,
        pretrained=True,
        checkpoint_path=path,
    )
    assert model.weights_loaded is True
    assert model.weights_source == "checkpoint_path"
    assert model.weights_provenance["checkpoint_path"]


def test_multitask_model_uses_verified_pretrained_backbone(
    backbone_name, make_input
) -> None:
    config = RepLiteConfig(
        tasks=TaskConfig(segmentation_classes=2, depth=True),
        backbone_name=backbone_name,
        pretrained=True,
        recurrent_c4_channels=16,
        recurrent_c5_channels=24,
        dense_channels=12,
        task_adapter_channels=12,
    )
    model = RepLiteMultiTaskModel(config).eval()
    with torch.no_grad():
        output = model(make_input(1, 3, 65, 97))
    assert model.backbone.weights_loaded
    assert output.segmentation is not None
    assert output.depth is not None
    assert output.segmentation.shape == (1, 2, 65, 97)
    export_metadata = model.export_task("segmentation").model_metadata
    assert export_metadata["config"]["pretrained"] is True
    assert export_metadata["source"]["backbone"]["weights"]["loaded"] is True
    assert (
        export_metadata["source"]["backbone"]["weights"]["source"]
        == model.backbone.weights_source
    )
