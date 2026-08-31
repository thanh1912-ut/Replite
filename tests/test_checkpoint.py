"""Checkpoint save/load round-trip and offline-construct tests."""

from __future__ import annotations

import pytest
import torch

from replite.backbone import create_backbone


def test_checkpoint_round_trip_without_pretrained(
    backbone_name, tmp_path, make_input
) -> None:
    model = create_backbone(backbone_name, pretrained=False, out_indices=(0, 1, 2, 3))
    model.eval()

    checkpoint_path = tmp_path / f"{backbone_name}.pt"
    torch.save(model.state_dict(), checkpoint_path)
    reloaded = create_backbone(backbone_name, pretrained=False)
    missing, unexpected = reloaded.load_state_dict(
        torch.load(checkpoint_path, weights_only=True), strict=True
    )
    assert not missing and not unexpected
    reloaded.eval()

    inputs = make_input(1, 3, 64, 96)
    with torch.no_grad():
        original = model(inputs)
        restored = reloaded(inputs)
    for a, b in zip(original, restored):
        torch.testing.assert_close(a, b, rtol=0.0, atol=0.0)


def test_checkpoint_round_trip_with_subset_indices(
    backbone_name, tmp_path, make_input
) -> None:
    model = create_backbone(backbone_name, out_indices=(0, 3))
    checkpoint_path = tmp_path / f"{backbone_name}_subset.pt"
    torch.save(model.state_dict(), checkpoint_path)
    reloaded = create_backbone(backbone_name, out_indices=(1, 3))
    # Selections with the same deepest stage have identical registered trunks.
    reloaded.load_state_dict(
        torch.load(checkpoint_path, weights_only=True), strict=True
    )
    assert reloaded.out_indices == (1, 3)


def test_state_dict_is_portable_between_out_indices(backbone_name) -> None:
    a = create_backbone(backbone_name, out_indices=(0, 3))
    b = create_backbone(backbone_name, out_indices=(2, 3))
    assert set(a.state_dict()) == set(b.state_dict())


def test_shallow_state_dict_is_strict_subset_of_full(backbone_name) -> None:
    shallow = create_backbone(backbone_name, out_indices=(0,))
    full = create_backbone(backbone_name)
    assert set(shallow.state_dict()) < set(full.state_dict())


def test_legacy_full_trunk_checkpoint_migrates_to_shallow_subset(
    backbone_name, make_input
) -> None:
    legacy = create_backbone(backbone_name).eval()
    shallow = create_backbone(backbone_name, out_indices=(0,)).eval()
    ignored = shallow.load_legacy_full_trunk_state_dict(legacy.state_dict())

    assert ignored
    assert all(key.startswith("blocks.") for key in ignored)
    inputs = make_input(1, 3, 64, 96)
    with torch.no_grad():
        torch.testing.assert_close(shallow(inputs)[0], legacy(inputs)[0])


def test_legacy_migration_rejects_unrelated_unexpected_keys(backbone_name) -> None:
    legacy_state = create_backbone(backbone_name).state_dict()
    legacy_state["classifier.injected"] = torch.zeros(1)
    shallow = create_backbone(backbone_name, out_indices=(0,))
    with pytest.raises(RuntimeError, match="non-pruned"):
        shallow.load_legacy_full_trunk_state_dict(legacy_state)


def test_legacy_migration_rejects_fabricated_key_in_pruned_group(
    backbone_name,
) -> None:
    legacy_state = create_backbone(backbone_name).state_dict()
    legacy_state["blocks.1.definitely_not_a_real_tensor"] = torch.zeros(1)
    shallow = create_backbone(backbone_name, out_indices=(0,))
    with pytest.raises(RuntimeError, match="non-pruned"):
        shallow.load_legacy_full_trunk_state_dict(legacy_state)


def test_legacy_migration_rejects_missing_key_in_pruned_group(backbone_name) -> None:
    legacy_state = create_backbone(backbone_name).state_dict()
    shallow = create_backbone(backbone_name, out_indices=(0,))
    pruned_key = next(key for key in legacy_state if key not in shallow.state_dict())
    del legacy_state[pruned_key]

    with pytest.raises(RuntimeError, match="missing canonical pruned"):
        shallow.load_legacy_full_trunk_state_dict(legacy_state)


def test_pretrained_false_constructor_never_touches_network(
    backbone_name, monkeypatch
) -> None:
    """pretrained=False must not download or call any network API."""

    def _forbidden(*args, **kwargs):
        raise AssertionError("network access attempted while pretrained=False")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _forbidden)
    import replite.backbone.weights as weights_module

    monkeypatch.setattr(weights_module, "hf_hub_download", _forbidden)
    monkeypatch.setattr(torch.hub, "download_url_to_file", _forbidden)

    model = create_backbone(backbone_name, pretrained=False)
    assert model.pretrained_cfg["revision"]
    assert sum(p.numel() for p in model.parameters()) > 0


def test_network_load_error_does_not_fall_back_to_random(
    backbone_name, monkeypatch
) -> None:
    """A download failure must raise, never silently return random weights."""
    import replite.backbone.weights as weights_module

    def _boom(*args, **kwargs):
        raise OSError("simulated network failure")

    monkeypatch.setattr(weights_module, "hf_hub_download", _boom)
    with pytest.raises(Exception) as exc_info:
        create_backbone(backbone_name, pretrained=True)
    assert "simulated network failure" in str(
        exc_info.value
    ) or "Failed to download" in str(exc_info.value)
