"""Tests for the manifest-driven SANPO joint training adapter."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

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


def _build_session(tmp_path: Path) -> Path:
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
        "detection_path": f"{sensor}/left/detection_boxes/000010.json",
    }
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


def test_collate_keeps_detection_targets_as_list(tmp_path: Path) -> None:
    dataset = SanpoJointDataset(_build_session(tmp_path), image_size=(8, 16))
    batch = sanpo_joint_collate([dataset[0], dataset[0]])
    inputs, targets = batch
    assert inputs.shape == (2, 3, 3, 8, 16)
    assert isinstance(targets["detection"], list)
    assert len(targets["detection"]) == 2
    assert targets["segmentation"].shape == (2, 8, 16)
    assert targets["depth"].shape == (2, 1, 8, 16)


def test_manifest_rejects_path_traversal(tmp_path: Path) -> None:
    path = _build_session(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["samples"][0]["panoptic_path"] = "../../secret.png"
    _write_json(path, payload)
    with pytest.raises(ValueError, match="unsafe"):
        load_sanpo_joint_manifest(path)

