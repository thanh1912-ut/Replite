"""Tests for the manifest-driven SANPO joint training adapter."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import replite.data.sanpo_joint as sanpo_joint_module
from replite.data import (
    SANPO_SEGMENTATION_CLASS_NAMES,
    SanpoJointDataset,
    load_sanpo_joint_manifest,
    read_sanpo_depth,
    sanpo_joint_collate,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_depth(path: Path, depth: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = np.asarray(depth.shape, dtype="<f2")
    payload = np.concatenate((header, depth.astype("<f2").reshape(-1)))
    with gzip.open(path, "wb") as handle:
        handle.write(payload.tobytes())


def _build_session(tmp_path: Path, *, include_detection: bool = True) -> Path:
    session_id = "session-alpha"
    sensor = "camera_head"
    session_root = tmp_path / "sanpo-real" / session_id
    left = session_root / sensor / "left"
    rgb_dir = left / "video_frames"
    mask_dir = left / "segmentation_masks"
    depth_dir = left / "depth_maps"
    detection_dir = left / "detection_boxes"
    for directory in (rgb_dir, mask_dir, depth_dir, detection_dir):
        directory.mkdir(parents=True, exist_ok=True)

    for frame, red in zip((8, 9, 10), (10, 20, 30)):
        rgb = np.zeros((4, 8, 3), dtype=np.uint8)
        rgb[..., 0] = red
        Image.fromarray(rgb).save(rgb_dir / f"{frame:06d}.png")

    panoptic = np.zeros((4, 8, 3), dtype=np.uint8)
    panoptic[:, :4, 0] = 1
    panoptic[:, 4:, 0] = 21
    panoptic[:, 4:, 2] = 7
    Image.fromarray(panoptic).save(mask_dir / "000010.png")

    depth = np.asarray(
        [[0.0, 0.2, 1.0, 2.0], [3.0, 4.0, np.inf, 100.0]],
        dtype=np.float32,
    )
    _write_depth(depth_dir / "000010.float16.gz", depth)

    detection = {
        "schema_version": 1,
        "dataset": "SANPO-Real-v0-derived-detection",
        "bbox_format": "absolute_half_open_xyxy",
        "valid_size": [4, 8],
        "boxes": [[4.0, 1.0, 8.0, 4.0]],
        "labels": [8],
        "ignore_boxes": [[0.0, 0.0, 2.0, 1.0]],
    }
    _write_json(detection_dir / "000010.json", detection)

    sample = {
        "target_frame": 10,
        "rgb_context_paths": [
            f"{sensor}/left/video_frames/000008.png",
            f"{sensor}/left/video_frames/000009.png",
            f"{sensor}/left/video_frames/000010.png",
        ],
        "panoptic_path": f"{sensor}/left/segmentation_masks/000010.png",
        "depth_path": f"{sensor}/left/depth_maps/000010.float16.gz",
    }
    if include_detection:
        sample["detection_path"] = (
            f"{sensor}/left/detection_boxes/000010.json"
        )
    manifest = {
        "schema_version": 2,
        "dataset": "SANPO-Real-v0-joint",
        "official_split": "train",
        "session_id": session_id,
        "sensor": sensor,
        "joint_frames": 1,
        "samples": [sample],
    }
    path = left / "_sanpo_joint_manifest.json"
    _write_json(path, manifest)
    return path


def _write_detection_threshold_case(manifest_path: Path) -> None:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    session_root = manifest_path.parents[2]
    sample = payload["samples"][0]
    panoptic_path = session_root / sample["panoptic_path"]
    panoptic = np.zeros((20, 25, 3), dtype=np.uint8)
    panoptic[..., 0] = 1
    # Exactly 100 pixels: positive under the locked >=100 policy.
    panoptic[0:10, 0:10, 0] = 21
    panoptic[0:10, 0:10, 2] = 1
    # Exactly 99 pixels: retained as an ignore box.
    panoptic[11:20, 0:11, 0] = 20
    panoptic[11:20, 0:11, 2] = 2
    Image.fromarray(panoptic).save(panoptic_path)
    detection_path = sample.get("detection_path")
    if detection_path is not None:
        _write_json(
            session_root / detection_path,
            {
                "schema_version": 1,
                "dataset": "SANPO-Real-v0-derived-detection",
                "bbox_format": "absolute_half_open_xyxy",
                "valid_size": [20, 25],
                "boxes": [[0.0, 0.0, 10.0, 10.0]],
                "labels": [8],
                "ignore_boxes": [[0.0, 11.0, 11.0, 20.0]],
            },
        )


def test_segmentation_names_exclude_unlabeled_and_keep_source_order() -> None:
    assert len(SANPO_SEGMENTATION_CLASS_NAMES) == 30
    assert SANPO_SEGMENTATION_CLASS_NAMES[:3] == ("road", "curb", "sidewalk")
    assert SANPO_SEGMENTATION_CLASS_NAMES[-1] == "terrain"


def test_read_depth_validates_header_and_decodes_metres(tmp_path: Path) -> None:
    path = tmp_path / "depth.float16.gz"
    expected = np.asarray([[1.0, 2.0], [3.5, 4.0]], dtype=np.float32)
    _write_depth(path, expected)
    actual = read_sanpo_depth(path)
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, expected)

    broken = tmp_path / "broken.float16.gz"
    with gzip.open(broken, "wb") as handle:
        handle.write(np.asarray([2, 2, 1], dtype="<f2").tobytes())
    with pytest.raises(ValueError, match="size mismatch"):
        read_sanpo_depth(broken)


def test_joint_dataset_emits_synchronized_replite_contract(tmp_path: Path) -> None:
    manifest_path = _build_session(tmp_path)
    dataset = SanpoJointDataset(
        manifest_path,
        image_size=(8, 16),
        depth_min=0.1,
        depth_max=80.0,
        normalize=False,
    )

    clip, targets = dataset[0]

    assert clip.shape == (3, 3, 8, 16)
    assert clip.dtype == torch.float32
    assert [round(float(clip[i, 0, 0, 0] * 255)) for i in range(3)] == [10, 20, 30]
    detection = targets["detection"]
    torch.testing.assert_close(
        detection["boxes"], torch.tensor([[8.0, 2.0, 16.0, 8.0]])
    )
    torch.testing.assert_close(
        detection["ignore_boxes"], torch.tensor([[0.0, 0.0, 4.0, 2.0]])
    )
    assert detection["labels"].dtype == torch.int64
    assert detection["valid_size"] == (8, 16)

    segmentation = targets["segmentation"]
    assert segmentation.dtype == torch.int64
    assert set(segmentation.unique().tolist()) == {0, 20}
    assert bool(targets["segmentation_valid"].all())
    assert targets["depth"].shape == (1, 8, 16)
    assert targets["depth"].dtype == torch.float32
    assert targets["depth_valid"].dtype == torch.bool
    assert not bool(targets["depth_valid"][0, 0, 0])
    assert not bool(targets["depth_valid"][0, -1, -1])
    assert torch.all(targets["depth"][~targets["depth_valid"]] == 0)

    provenance = dataset.sample_provenance(0)
    assert provenance["session_id"] == "session-alpha"
    assert provenance["target_frame"] == 10
    assert provenance["detection_source"] == "packaged_json"


def test_legacy_joint_dataset_derives_detection_from_panoptic_per_sample(
    tmp_path: Path,
) -> None:
    manifest_path = _build_session(tmp_path, include_detection=False)
    manifest, _ = load_sanpo_joint_manifest(manifest_path)
    assert "detection_path" not in manifest["samples"][0]

    dataset = SanpoJointDataset(
        manifest_path,
        image_size=(8, 16),
        detection_min_area=1,
        normalize=False,
    )
    _, targets = dataset[0]

    detection = targets["detection"]
    torch.testing.assert_close(
        detection["boxes"], torch.tensor([[8.0, 0.0, 16.0, 8.0]])
    )
    torch.testing.assert_close(detection["labels"], torch.tensor([8]))
    assert detection["ignore_boxes"].shape == (0, 4)
    provenance = dataset.sample_provenance(0)
    assert provenance["detection_path"] is None
    assert provenance["detection_source"] == "panoptic_on_load"


@pytest.mark.parametrize("schema_version", [1, 2])
def test_legacy_manifest_versions_are_supported(
    tmp_path: Path, schema_version: int
) -> None:
    path = _build_session(tmp_path, include_detection=False)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = schema_version
    _write_json(path, payload)

    dataset = SanpoJointDataset(path, detection_min_area=1, normalize=False)

    assert len(dataset) == 1
    assert dataset.sample_provenance(0)["detection_source"] == "panoptic_on_load"


def test_packaged_detection_remains_authoritative_over_panoptic(tmp_path: Path) -> None:
    dataset = SanpoJointDataset(
        _build_session(tmp_path),
        image_size=(8, 16),
        detection_min_area=1,
        normalize=False,
    )

    _, targets = dataset[0]

    # The packaged target deliberately starts at source y=1.  On-load
    # derivation from the mask would start at y=0, so this proves a new archive
    # keeps its immutable packaged labels.
    torch.testing.assert_close(
        targets["detection"]["boxes"],
        torch.tensor([[8.0, 2.0, 16.0, 8.0]]),
    )


def test_legacy_catalog_policy_can_force_panoptic_over_unversioned_json(
    tmp_path: Path,
) -> None:
    dataset = SanpoJointDataset(
        _build_session(tmp_path),
        image_size=(8, 16),
        detection_min_area=1,
        use_packaged_detection=False,
        normalize=False,
    )

    _, targets = dataset[0]

    torch.testing.assert_close(
        targets["detection"]["boxes"],
        torch.tensor([[8.0, 0.0, 16.0, 8.0]]),
    )
    assert dataset.sample_provenance(0)["detection_source"] == "panoptic_on_load"


def test_locked_threshold_and_packaged_on_load_detection_are_equivalent(
    tmp_path: Path,
) -> None:
    legacy_manifest = _build_session(
        tmp_path / "legacy", include_detection=False
    )
    packaged_manifest = _build_session(tmp_path / "packaged")
    _write_detection_threshold_case(legacy_manifest)
    _write_detection_threshold_case(packaged_manifest)

    legacy = SanpoJointDataset(
        legacy_manifest, image_size=(20, 25), normalize=False
    )
    packaged = SanpoJointDataset(
        packaged_manifest, image_size=(20, 25), normalize=False
    )
    forced = SanpoJointDataset(
        packaged_manifest,
        image_size=(20, 25),
        use_packaged_detection=False,
        normalize=False,
    )

    legacy_target = legacy[0][1]["detection"]
    for candidate in (packaged[0][1]["detection"], forced[0][1]["detection"]):
        assert candidate["valid_size"] == legacy_target["valid_size"]
        for field in ("boxes", "labels", "ignore_boxes"):
            torch.testing.assert_close(candidate[field], legacy_target[field])
    torch.testing.assert_close(
        legacy_target["boxes"], torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    )
    torch.testing.assert_close(
        legacy_target["ignore_boxes"],
        torch.tensor([[0.0, 11.0, 11.0, 20.0]]),
    )
    assert legacy.sample_provenance(0)["detection_source"] == "panoptic_on_load"
    assert packaged.sample_provenance(0)["detection_source"] == "packaged_json"
    assert forced.sample_provenance(0)["detection_source"] == "panoptic_on_load"


@pytest.mark.parametrize("value", [True, -1, 1.5])
def test_detection_min_area_is_validated_early(tmp_path: Path, value: object) -> None:
    with pytest.raises((TypeError, ValueError), match="detection_min_area"):
        SanpoJointDataset(
            _build_session(tmp_path),
            detection_min_area=value,  # type: ignore[arg-type]
        )


def test_use_packaged_detection_is_validated_early(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="use_packaged_detection"):
        SanpoJointDataset(
            _build_session(tmp_path),
            use_packaged_detection=1,  # type: ignore[arg-type]
        )


def test_collate_keeps_detection_targets_as_list(tmp_path: Path) -> None:
    dataset = SanpoJointDataset(_build_session(tmp_path), image_size=(8, 16))
    batch = sanpo_joint_collate([dataset[0], dataset[0]])
    inputs, targets = batch
    assert inputs.shape == (2, 3, 3, 8, 16)
    assert isinstance(targets["detection"], list)
    assert len(targets["detection"]) == 2
    assert targets["segmentation"].shape == (2, 8, 16)
    assert targets["depth"].shape == (2, 1, 8, 16)


def _assert_joint_sample_equal(
    first: tuple[torch.Tensor, dict[str, object]],
    second: tuple[torch.Tensor, dict[str, object]],
) -> None:
    torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)
    first_targets, second_targets = first[1], second[1]
    assert set(first_targets) == set(second_targets)
    for name in ("segmentation", "segmentation_valid", "depth", "depth_valid"):
        torch.testing.assert_close(
            first_targets[name],  # type: ignore[arg-type]
            second_targets[name],  # type: ignore[arg-type]
            rtol=0,
            atol=0,
        )
    first_detection = first_targets["detection"]
    second_detection = second_targets["detection"]
    assert isinstance(first_detection, dict)
    assert isinstance(second_detection, dict)
    assert first_detection["valid_size"] == second_detection["valid_size"]
    for name in ("boxes", "labels", "ignore_boxes"):
        torch.testing.assert_close(
            first_detection[name],
            second_detection[name],
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("include_detection", [False, True])
def test_prepared_cache_is_bit_exact_and_avoids_redecoding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    normalize: bool,
    include_detection: bool,
) -> None:
    manifest = _build_session(
        tmp_path / "source",
        include_detection=include_detection,
    )
    kwargs = {
        "image_size": (8, 16),
        "detection_min_area": 1,
        "normalize": normalize,
    }
    expected = SanpoJointDataset(manifest, **kwargs)[0]
    calls = 0
    original = sanpo_joint_module.sanpo_panoptic_to_detection

    def counted(*args: object, **call_kwargs: object):
        nonlocal calls
        calls += 1
        return original(*args, **call_kwargs)

    monkeypatch.setattr(
        sanpo_joint_module,
        "sanpo_panoptic_to_detection",
        counted,
    )
    cache_root = tmp_path / "prepared"
    first_dataset = SanpoJointDataset(
        manifest,
        prepared_cache_dir=cache_root,
        **kwargs,
    )
    first = first_dataset[0]
    first_calls = calls
    assert first_calls == (0 if include_detection else 1)
    _assert_joint_sample_equal(expected, first)

    cache_file = next(first_dataset.prepared_cache_dir.glob("*.pt"))  # type: ignore[union-attr]
    payload = torch.load(cache_file, map_location="cpu", weights_only=True)
    assert payload["tensors"]["rgb_uint8"].dtype == torch.uint8
    assert payload["tensors"]["segmentation_uint8"].dtype == torch.uint8
    assert payload["tensors"]["depth_float16"].dtype == torch.float16

    second_dataset = SanpoJointDataset(
        manifest,
        prepared_cache_dir=cache_root,
        **kwargs,
    )
    monkeypatch.setattr(
        second_dataset,
        "_build_prepared_sample",
        lambda index: (_ for _ in ()).throw(AssertionError(index)),
    )
    second = second_dataset[0]
    assert calls == first_calls
    _assert_joint_sample_equal(expected, second)


def test_corrupt_prepared_cache_is_atomically_rebuilt(tmp_path: Path) -> None:
    manifest = _build_session(tmp_path / "source", include_detection=False)
    dataset = SanpoJointDataset(
        manifest,
        image_size=(8, 16),
        detection_min_area=1,
        normalize=False,
        prepared_cache_dir=tmp_path / "prepared",
    )
    expected = dataset[0]
    cache_file = next(dataset.prepared_cache_dir.glob("*.pt"))  # type: ignore[union-attr]
    cache_file.write_bytes(b"interrupted local cache")

    repaired = dataset[0]

    _assert_joint_sample_equal(expected, repaired)
    payload = torch.load(cache_file, map_location="cpu", weights_only=True)
    assert payload["cache_key"] == dataset._prepared_cache_key


def test_manifest_rejects_path_traversal(tmp_path: Path) -> None:
    path = _build_session(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["samples"][0]["panoptic_path"] = "../../secret.png"
    _write_json(path, payload)
    with pytest.raises(ValueError, match="unsafe"):
        load_sanpo_joint_manifest(path)


@pytest.mark.parametrize("schema_version", [True, 0, 3, "2"])
def test_manifest_rejects_unsupported_schema(
    tmp_path: Path, schema_version: object
) -> None:
    path = _build_session(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = schema_version
    _write_json(path, payload)

    with pytest.raises(ValueError, match="unsupported SANPO joint manifest schema"):
        load_sanpo_joint_manifest(path)
