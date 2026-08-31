"""Verified weight loading, cache recovery, and provenance tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import timm
import torch
from safetensors.torch import save_file

from replite.backbone import create_backbone
from replite.backbone.weights import (
    CheckpointDownloadError,
    CheckpointFormatError,
    ChecksumMismatchError,
    PretrainedWeightsSpec,
    load_verified_state_dict,
)
import replite.backbone.base as base_module
import replite.backbone.weights as weights_module


def _valid_checkpoint(tmp_path: Path) -> tuple[Path, PretrainedWeightsSpec]:
    path = tmp_path / "model.safetensors"
    save_file({"weight": torch.arange(6).reshape(2, 3)}, path)
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, PretrainedWeightsSpec(
        architecture="test_arch",
        repository="test/repository",
        revision="a" * 40,
        sha256=sha256,
    )


def test_explicit_checkpoint_path_is_verified_without_hub(
    tmp_path, monkeypatch
) -> None:
    path, spec = _valid_checkpoint(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("Hub must not be called for checkpoint_path")

    monkeypatch.setattr(weights_module, "hf_hub_download", forbidden)
    state = load_verified_state_dict(spec, checkpoint_path=path)
    torch.testing.assert_close(state["weight"], torch.arange(6).reshape(2, 3))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"checkpoint_path": "x", "cache_dir": "cache"},
        {"checkpoint_path": "x", "local_files_only": True},
        {"checkpoint_path": "x", "force_download": True},
        {"local_files_only": True, "force_download": True},
    ],
)
def test_invalid_weight_source_combinations_are_rejected(kwargs) -> None:
    spec = PretrainedWeightsSpec("a", "r", "v", "0" * 64)
    with pytest.raises(ValueError):
        load_verified_state_dict(spec, **kwargs)


def test_factory_rejects_ignored_load_options_without_pretrained() -> None:
    for kwargs in (
        {"checkpoint_path": "x"},
        {"cache_dir": "cache"},
        {"local_files_only": True},
        {"force_download": True},
    ):
        with pytest.raises(ValueError, match="pretrained=True"):
            create_backbone("mobilenetv3_small_050", pretrained=False, **kwargs)


def test_hub_controls_are_forwarded_exactly(tmp_path, monkeypatch) -> None:
    path, spec = _valid_checkpoint(tmp_path)
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return str(path)

    monkeypatch.setattr(weights_module, "hf_hub_download", fake_download)
    load_verified_state_dict(
        spec,
        cache_dir=tmp_path / "cache",
        local_files_only=True,
    )
    assert calls == [
        {
            "repo_id": spec.repository,
            "filename": spec.filename,
            "revision": spec.revision,
            "cache_dir": str(tmp_path / "cache"),
            "local_files_only": True,
            "force_download": False,
        }
    ]


def test_weight_spec_canonicalizes_mutable_metadata_inputs() -> None:
    input_size = [3, 224, 224]
    mean = [0.1, 0.2, 0.3]
    std = [1.0, 1.0, 1.0]
    spec = PretrainedWeightsSpec(
        "arch",
        "repo",
        "revision",
        "A" * 64,
        input_size=input_size,
        mean=mean,
        std=std,
    )
    input_size[1] = 999
    mean[0] = 999.0
    std[0] = -1.0

    cfg = spec.as_pretrained_cfg()
    assert cfg["input_size"] == (3, 224, 224)
    assert cfg["mean"] == (0.1, 0.2, 0.3)
    assert cfg["std"] == (1.0, 1.0, 1.0)
    assert cfg["sha256"] == "a" * 64


@pytest.mark.parametrize(
    "overrides",
    [
        {"sha256": "not-a-digest"},
        {"input_size": (3, 0, 224)},
        {"input_size": (1, 224, 224)},
        {"test_input_size": (4, 256, 256)},
        {"mean": (0.1, 0.2)},
        {"std": (1.0, 0.0, 1.0)},
        {"crop_pct": 1.1},
        {"test_crop_pct": 0.0},
        {"fixed_input_size": 1},
    ],
)
def test_weight_spec_rejects_invalid_metadata(overrides) -> None:
    kwargs = {
        "architecture": "arch",
        "repository": "repo",
        "revision": "revision",
        "sha256": "0" * 64,
        **overrides,
    }
    with pytest.raises(ValueError):
        PretrainedWeightsSpec(**kwargs)


def test_corrupt_hub_cache_is_refetched_once(tmp_path, monkeypatch) -> None:
    good, spec = _valid_checkpoint(tmp_path)
    bad = tmp_path / "bad.safetensors"
    bad.write_bytes(b"corrupt")
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs["force_download"])
        return str(good if kwargs["force_download"] else bad)

    monkeypatch.setattr(weights_module, "hf_hub_download", fake_download)
    state = load_verified_state_dict(spec)
    assert calls == [False, True]
    assert "weight" in state


def test_offline_corrupt_cache_is_not_refetched(tmp_path, monkeypatch) -> None:
    _, spec = _valid_checkpoint(tmp_path)
    bad = tmp_path / "bad.safetensors"
    bad.write_bytes(b"corrupt")
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs["force_download"])
        return str(bad)

    monkeypatch.setattr(weights_module, "hf_hub_download", fake_download)
    with pytest.raises(ChecksumMismatchError):
        load_verified_state_dict(spec, local_files_only=True)
    assert calls == [False]


def test_second_checksum_mismatch_is_rejected(tmp_path, monkeypatch) -> None:
    _, spec = _valid_checkpoint(tmp_path)
    bad = tmp_path / "bad.safetensors"
    bad.write_bytes(b"corrupt")
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs["force_download"])
        return str(bad)

    monkeypatch.setattr(weights_module, "hf_hub_download", fake_download)
    with pytest.raises(ChecksumMismatchError, match=str(bad.resolve())):
        load_verified_state_dict(spec)
    assert calls == [False, True]


def test_refresh_failure_reports_both_corruption_and_download_error(
    tmp_path, monkeypatch
) -> None:
    _, spec = _valid_checkpoint(tmp_path)
    bad = tmp_path / "bad.safetensors"
    bad.write_bytes(b"corrupt")

    def fake_download(**kwargs):
        if kwargs["force_download"]:
            raise OSError("network unavailable")
        return str(bad)

    monkeypatch.setattr(weights_module, "hf_hub_download", fake_download)
    with pytest.raises(CheckpointDownloadError) as exc_info:
        load_verified_state_dict(spec)
    message = str(exc_info.value)
    assert "failed verification" in message
    assert "forced refresh also failed" in message
    assert "network unavailable" in message


def test_missing_local_checkpoint_reports_resolved_path(tmp_path) -> None:
    missing = tmp_path / "missing.safetensors"
    spec = PretrainedWeightsSpec("a", "r", "v", "0" * 64)
    with pytest.raises(CheckpointDownloadError, match=str(missing.resolve())):
        load_verified_state_dict(spec, checkpoint_path=missing)


def test_invalid_safetensors_is_reported_after_hash_verification(tmp_path) -> None:
    path = tmp_path / "invalid.safetensors"
    path.write_bytes(b"not-a-safetensors-file")
    spec = PretrainedWeightsSpec(
        "a", "r", "v", hashlib.sha256(path.read_bytes()).hexdigest()
    )
    with pytest.raises(CheckpointFormatError, match=str(path.resolve())):
        load_verified_state_dict(spec, checkpoint_path=path)


def test_hash_and_decode_use_the_same_immutable_bytes(tmp_path, monkeypatch) -> None:
    path, spec = _valid_checkpoint(tmp_path)
    original_payload = path.read_bytes()
    real_load = weights_module.load_safetensors
    seen = []

    def mutating_load(payload):
        seen.append(payload)
        path.write_bytes(b"changed-after-read")
        return real_load(payload)

    monkeypatch.setattr(weights_module, "load_safetensors", mutating_load)
    state = load_verified_state_dict(spec, checkpoint_path=path)
    assert seen == [original_payload]
    assert "weight" in state


def test_weight_provenance_is_explicit_and_copy_safe(monkeypatch, tmp_path) -> None:
    random_model = create_backbone("mobilenetv3_small_050")
    assert random_model.weights_loaded is False
    assert random_model.weights_source == "random_init"

    full_state = timm.create_model(
        "mobilenetv3_small_050", pretrained=False
    ).state_dict()
    monkeypatch.setattr(
        base_module, "load_verified_state_dict", lambda *args, **kwargs: full_state
    )
    local = tmp_path / "official.safetensors"
    loaded_model = create_backbone(
        "mobilenetv3_small_050",
        pretrained=True,
        checkpoint_path=local,
    )
    provenance = loaded_model.weights_provenance
    assert provenance["loaded"] is True
    assert provenance["source"] == "checkpoint_path"
    assert provenance["checkpoint_path"] == str(local.resolve())
    json.dumps(loaded_model.backbone_config)

    provenance["sha256"] = "tampered"
    assert loaded_model.weights_provenance["sha256"] != "tampered"
