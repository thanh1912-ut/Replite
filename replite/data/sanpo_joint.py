"""Manifest-driven SANPO-Real joint dataset for RepLite.

The pilot downloader writes sparse, per-session manifests.  This module reads
those manifests directly and never infers frame IDs from directory contents.
Each item contains the three RGB context frames and supervision for the final
frame: derived detection boxes, semantic segmentation, and metric depth.

Geometric preprocessing is deliberately small and deterministic.  RGB is
resized bilinearly, while semantic labels, metric depth, and depth validity are
resized with nearest-neighbour sampling.  Detection boxes remain absolute
half-open XYXY coordinates and are scaled to the configured output size.
"""

from __future__ import annotations

import gzip
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from os import PathLike
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from .sanpo import (
    SANPO_DETECTION_CLASS_NAMES,
    SANPO_LABELMAP,
    decode_sanpo_panoptic,
    sanpo_panoptic_to_detection,
)


SANPO_SEGMENTATION_CLASS_NAMES = tuple(
    name
    for name, source_id in sorted(SANPO_LABELMAP.items(), key=lambda item: item[1])
    if source_id != 0
)
SANPO_SEGMENTATION_IGNORE_INDEX = 255
IMAGENET_RGB_MEAN = (0.485, 0.456, 0.406)
IMAGENET_RGB_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class SanpoJointInfo:
    """Validated identity and sample count from one joint manifest."""

    manifest_path: Path
    session_root: Path
    official_split: str
    session_id: str
    sensor: str
    sample_count: int


def _positive_hw(value: Sequence[int], name: str) -> tuple[int, int]:
    try:
        raw = tuple(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a height,width sequence") from exc
    if len(raw) != 2 or any(
        isinstance(item, bool) or not isinstance(item, (int, np.integer)) or item <= 0
        for item in raw
    ):
        raise ValueError(f"{name} must contain two positive integers")
    return int(raw[0]), int(raw[1])


def _safe_relative_path(session_root: Path, value: object, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty relative path")
    posix = PurePosixPath(value)
    if posix.is_absolute() or any(part in {"", ".", ".."} for part in posix.parts):
        raise ValueError(f"unsafe {name}: {value!r}")
    candidate = session_root.joinpath(*posix.parts)
    try:
        candidate.resolve(strict=False).relative_to(session_root.resolve())
    except ValueError as exc:
        raise ValueError(f"{name} escapes the session root: {value!r}") from exc
    return candidate


def load_sanpo_joint_manifest(
    filename: str | PathLike[str],
) -> tuple[dict[str, Any], SanpoJointInfo]:
    """Load and validate a downloader-produced SANPO joint manifest."""

    path = Path(filename).expanduser().resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read SANPO joint manifest: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("SANPO joint manifest must contain a JSON object")
    schema_version = payload.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version not in {1, 2}
    ):
        raise ValueError("unsupported SANPO joint manifest schema")
    if payload.get("dataset") != "SANPO-Real-v0-joint":
        raise ValueError("manifest is not a SANPO-Real-v0 joint subset")
    split = payload.get("official_split")
    if split not in {"train", "test"}:
        raise ValueError("official_split must be 'train' or 'test'")
    session_id = payload.get("session_id")
    sensor = payload.get("sensor")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("manifest session_id is invalid")
    if sensor not in {"camera_head", "camera_chest"}:
        raise ValueError("manifest sensor is invalid")
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("SANPO joint manifest has no samples")
    declared = payload.get("joint_frames")
    if isinstance(declared, bool) or not isinstance(declared, int):
        raise ValueError("manifest joint_frames is invalid")
    if declared != len(samples):
        raise ValueError("manifest joint_frames does not match samples")

    # .../<session>/<sensor>/left/_sanpo_joint_manifest.json
    session_root = path.parents[2]
    if session_root.name != session_id:
        raise ValueError("manifest path and session_id disagree")
    if path.parent.name != "left" or path.parent.parent.name != sensor:
        raise ValueError("manifest path and sensor/lens disagree")

    required_paths = ("panoptic_path", "depth_path")
    seen_frames: set[int] = set()
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"sample {index} must be a mapping")
        frame = sample.get("target_frame")
        if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
            raise ValueError(f"sample {index} has an invalid target_frame")
        if frame in seen_frames:
            raise ValueError(f"duplicate target_frame in manifest: {frame}")
        seen_frames.add(frame)
        context = sample.get("rgb_context_paths")
        if not isinstance(context, list) or len(context) != 3:
            raise ValueError(f"sample {index} must contain exactly three RGB paths")
        for context_index, relative in enumerate(context):
            _safe_relative_path(
                session_root,
                relative,
                f"samples[{index}].rgb_context_paths[{context_index}]",
            )
        for key in required_paths:
            _safe_relative_path(session_root, sample.get(key), f"samples[{index}].{key}")
        # Archives created before detection targets were packaged contain the
        # same official panoptic supervision but no detection_path.  That is a
        # supported legacy representation; malformed paths that are present
        # remain strict failures rather than silently falling back.
        if "detection_path" in sample:
            _safe_relative_path(
                session_root,
                sample.get("detection_path"),
                f"samples[{index}].detection_path",
            )

    return payload, SanpoJointInfo(
        manifest_path=path,
        session_root=session_root,
        official_split=split,
        session_id=session_id,
        sensor=sensor,
        sample_count=len(samples),
    )


def read_sanpo_depth(filename: str | PathLike[str]) -> np.ndarray:
    """Decode SANPO gzip little-endian float16 depth into float32 metres."""

    path = Path(filename)
    try:
        with gzip.open(path, "rb") as handle:
            payload = handle.read()
    except (OSError, EOFError) as exc:
        raise ValueError(f"cannot read SANPO depth file: {path}") from exc
    if len(payload) < 4 or len(payload) % 2:
        raise ValueError(f"invalid SANPO depth byte length: {path}")
    values = np.frombuffer(payload, dtype="<f2")
    if values.size < 2 or not np.isfinite(values[:2]).all():
        raise ValueError(f"invalid SANPO depth header: {path}")
    height, width = (int(round(float(value))) for value in values[:2])
    if height <= 0 or width <= 0 or values.size != 2 + height * width:
        raise ValueError(f"SANPO depth header/payload size mismatch: {path}")
    return values[2:].reshape(height, width).astype(np.float32, copy=True)


def _read_detection_target(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read derived detection target: {path}") from exc
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ValueError(f"unsupported derived detection target: {path}")
    if payload.get("dataset") != "SANPO-Real-v0-derived-detection":
        raise ValueError(f"unexpected detection dataset: {path}")
    if payload.get("bbox_format") not in {
        "absolute_half_open_xyxy",
        "XYXY half-open, absolute target-RGB pixels",
    }:
        raise ValueError(f"unexpected detection box format: {path}")
    boxes = np.asarray(payload.get("boxes"), dtype=np.float32).reshape(-1, 4)
    labels = np.asarray(payload.get("labels"), dtype=np.int64).reshape(-1)
    ignore_boxes = np.asarray(payload.get("ignore_boxes", ()), dtype=np.float32).reshape(-1, 4)
    if boxes.shape[0] != labels.shape[0]:
        raise ValueError(f"detection boxes/labels length mismatch: {path}")
    if not np.isfinite(boxes).all() or not np.isfinite(ignore_boxes).all():
        raise ValueError(f"detection target contains non-finite boxes: {path}")
    if labels.size and (labels.min() < 0 or labels.max() >= len(SANPO_DETECTION_CLASS_NAMES)):
        raise ValueError(f"detection target label is outside [0,14]: {path}")
    source_hw = _positive_hw(payload.get("valid_size"), "detection valid_size")
    for name, array in (("boxes", boxes), ("ignore_boxes", ignore_boxes)):
        if array.size and (
            np.any(array[:, 0] < 0)
            or np.any(array[:, 1] < 0)
            or np.any(array[:, 2] > source_hw[1])
            or np.any(array[:, 3] > source_hw[0])
            or np.any(array[:, 2] <= array[:, 0])
            or np.any(array[:, 3] <= array[:, 1])
        ):
            raise ValueError(f"detection {name} are invalid for source size: {path}")
    return {
        "boxes": boxes,
        "labels": labels,
        "ignore_boxes": ignore_boxes,
        "valid_size": source_hw,
    }


def _scale_boxes(boxes: np.ndarray, source_hw: tuple[int, int], output_hw: tuple[int, int]) -> Tensor:
    result = torch.as_tensor(boxes, dtype=torch.float32).clone()
    if result.numel():
        result[:, (0, 2)] *= output_hw[1] / source_hw[1]
        result[:, (1, 3)] *= output_hw[0] / source_hw[0]
        result[:, (0, 2)].clamp_(0.0, float(output_hw[1]))
        result[:, (1, 3)].clamp_(0.0, float(output_hw[0]))
    return result.reshape(-1, 4)


class SanpoJointDataset(Dataset[tuple[Tensor, dict[str, Any]]]):
    """Read one sparse SANPO session manifest as synchronized joint samples."""

    def __init__(
        self,
        manifest_path: str | PathLike[str],
        *,
        image_size: Sequence[int] = (288, 512),
        depth_min: float = 0.1,
        depth_max: float = 80.0,
        detection_min_area: int = 100,
        use_packaged_detection: bool = True,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.manifest, self.info = load_sanpo_joint_manifest(manifest_path)
        self.image_size = _positive_hw(image_size, "image_size")
        if (
            isinstance(depth_min, bool)
            or not isinstance(depth_min, (int, float))
            or not math.isfinite(float(depth_min))
            or float(depth_min) < 0.0
        ):
            raise ValueError("depth_min must be finite and non-negative")
        if (
            isinstance(depth_max, bool)
            or not isinstance(depth_max, (int, float))
            or not math.isfinite(float(depth_max))
            or float(depth_max) <= float(depth_min)
        ):
            raise ValueError("depth_max must be finite and greater than depth_min")
        if not isinstance(normalize, bool):
            raise TypeError("normalize must be a boolean")
        if isinstance(detection_min_area, bool) or not isinstance(
            detection_min_area, Integral
        ):
            raise TypeError("detection_min_area must be an integer")
        if detection_min_area < 0:
            raise ValueError("detection_min_area must be non-negative")
        if not isinstance(use_packaged_detection, bool):
            raise TypeError("use_packaged_detection must be a boolean")
        self.depth_min = float(depth_min)
        self.depth_max = float(depth_max)
        self.detection_min_area = int(detection_min_area)
        self.use_packaged_detection = use_packaged_detection
        self.normalize = normalize
        self._mean = torch.tensor(IMAGENET_RGB_MEAN, dtype=torch.float32).reshape(3, 1, 1)
        self._std = torch.tensor(IMAGENET_RGB_STD, dtype=torch.float32).reshape(3, 1, 1)

    def __len__(self) -> int:
        return len(self.manifest["samples"])

    def _path(self, value: object, name: str) -> Path:
        path = _safe_relative_path(self.info.session_root, value, name)
        if not path.is_file():
            raise FileNotFoundError(f"missing SANPO sample file: {path}")
        return path

    def _load_rgb(self, path: Path) -> Tensor:
        with Image.open(path) as image:
            image = image.convert("RGB").resize(
                (self.image_size[1], self.image_size[0]),
                resample=Image.Resampling.BILINEAR,
            )
            array = np.asarray(image, dtype=np.uint8).copy()
        tensor = torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0)
        return (tensor - self._mean) / self._std if self.normalize else tensor

    def __getitem__(self, index: int) -> tuple[Tensor, dict[str, Any]]:
        sample = self.manifest["samples"][index]
        clip = torch.stack(
            [
                self._load_rgb(self._path(relative, "rgb_context_path"))
                for relative in sample["rgb_context_paths"]
            ]
        )

        panoptic_path = self._path(sample["panoptic_path"], "panoptic_path")
        with Image.open(panoptic_path) as image:
            image.load()
            panoptic_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        semantic, _ = decode_sanpo_panoptic(panoptic_rgb)
        semantic_image = Image.fromarray(semantic, mode="L").resize(
            (self.image_size[1], self.image_size[0]),
            resample=Image.Resampling.NEAREST,
        )
        source_labels = np.asarray(semantic_image, dtype=np.uint8)
        segmentation = np.full(source_labels.shape, SANPO_SEGMENTATION_IGNORE_INDEX, dtype=np.int64)
        valid_semantic = (source_labels >= 1) & (source_labels <= 30)
        segmentation[valid_semantic] = source_labels[valid_semantic].astype(np.int64) - 1

        depth_path = self._path(sample["depth_path"], "depth_path")
        source_depth = read_sanpo_depth(depth_path)
        source_valid = (
            np.isfinite(source_depth)
            & (source_depth > self.depth_min)
            & (source_depth <= self.depth_max)
        )
        safe_depth = np.where(source_valid, source_depth, 0.0).astype(np.float32)
        depth = np.asarray(
            Image.fromarray(safe_depth, mode="F").resize(
                (self.image_size[1], self.image_size[0]),
                resample=Image.Resampling.NEAREST,
            ),
            dtype=np.float32,
        ).copy()
        depth_valid = np.asarray(
            Image.fromarray(source_valid.astype(np.uint8), mode="L").resize(
                (self.image_size[1], self.image_size[0]),
                resample=Image.Resampling.NEAREST,
            ),
            dtype=np.uint8,
        ).astype(bool, copy=False)
        depth[~depth_valid] = 0.0

        detection_relative = (
            sample.get("detection_path")
            if self.use_packaged_detection
            else None
        )
        if detection_relative is None:
            detection = sanpo_panoptic_to_detection(
                panoptic_rgb,
                min_area=self.detection_min_area,
            )
        else:
            detection_path = self._path(detection_relative, "detection_path")
            detection = _read_detection_target(detection_path)
        if tuple(semantic.shape) != detection["valid_size"]:
            raise ValueError(
                "panoptic mask and derived detection target have different source sizes"
            )
        detection_target = {
            "boxes": _scale_boxes(detection["boxes"], detection["valid_size"], self.image_size),
            "labels": torch.as_tensor(detection["labels"], dtype=torch.int64),
            "valid_size": self.image_size,
            "ignore_boxes": _scale_boxes(
                detection["ignore_boxes"], detection["valid_size"], self.image_size
            ),
        }

        targets: dict[str, Any] = {
            "detection": detection_target,
            "segmentation": torch.from_numpy(segmentation),
            "segmentation_valid": torch.from_numpy(valid_semantic.copy()),
            "depth": torch.from_numpy(depth).unsqueeze(0),
            "depth_valid": torch.from_numpy(depth_valid.copy()).unsqueeze(0),
        }
        return clip, targets

    def sample_provenance(self, index: int) -> dict[str, Any]:
        """Return JSON-compatible identity and source paths for one sample."""

        sample = self.manifest["samples"][index]
        return {
            "official_split": self.info.official_split,
            "session_id": self.info.session_id,
            "sensor": self.info.sensor,
            "target_frame": int(sample["target_frame"]),
            "rgb_context_paths": list(sample["rgb_context_paths"]),
            "panoptic_path": sample["panoptic_path"],
            "depth_path": sample["depth_path"],
            "detection_path": sample.get("detection_path"),
            "detection_source": self._detection_source(sample),
        }

    def _detection_source(self, sample: Mapping[str, Any]) -> str:
        return (
            "packaged_json"
            if self.use_packaged_detection
            and sample.get("detection_path") is not None
            else "panoptic_on_load"
        )


def sanpo_joint_collate(
    batch: Sequence[tuple[Tensor, Mapping[str, Any]]],
) -> tuple[Tensor, dict[str, Any]]:
    """Collate variable-length detection targets without padding boxes."""

    if not batch:
        raise ValueError("cannot collate an empty SANPO batch")
    clips, targets = zip(*batch)
    if any(not isinstance(clip, Tensor) or clip.ndim != 4 for clip in clips):
        raise ValueError("each SANPO clip must have shape T,C,H,W")
    inputs = torch.stack(clips)
    required = {"detection", "segmentation", "segmentation_valid", "depth", "depth_valid"}
    if any(not isinstance(target, Mapping) or not required.issubset(target) for target in targets):
        raise ValueError("each SANPO target mapping is incomplete")
    return inputs, {
        "detection": [target["detection"] for target in targets],
        "segmentation": torch.stack([target["segmentation"] for target in targets]),
        "segmentation_valid": torch.stack([target["segmentation_valid"] for target in targets]),
        "depth": torch.stack([target["depth"] for target in targets]),
        "depth_valid": torch.stack([target["depth_valid"] for target in targets]),
    }


__all__ = [
    "IMAGENET_RGB_MEAN",
    "IMAGENET_RGB_STD",
    "SANPO_SEGMENTATION_CLASS_NAMES",
    "SANPO_SEGMENTATION_IGNORE_INDEX",
    "SanpoJointDataset",
    "SanpoJointInfo",
    "load_sanpo_joint_manifest",
    "read_sanpo_depth",
    "sanpo_joint_collate",
]
