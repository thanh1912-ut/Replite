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
import hashlib
import json
import math
import os
import pickle
import uuid
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
_PREPARED_CACHE_SCHEMA_VERSION = 1


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
        prepared_cache_dir: str | PathLike[str] | None = None,
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
        self._prepared_cache_key = self._make_prepared_cache_key()
        self._prepared_sample_estimates: dict[int, int] = {}
        self.prepared_cache_dir: Path | None = None
        if prepared_cache_dir is not None:
            root = Path(prepared_cache_dir).expanduser().resolve()
            self.prepared_cache_dir = root / self._prepared_cache_key
            self.prepared_cache_dir.mkdir(parents=True, exist_ok=True)
            self._ensure_prepared_cache_marker()

    def __len__(self) -> int:
        return len(self.manifest["samples"])

    def _path(self, value: object, name: str) -> Path:
        path = _safe_relative_path(self.info.session_root, value, name)
        if not path.is_file():
            raise FileNotFoundError(f"missing SANPO sample file: {path}")
        return path

    def _make_prepared_cache_key(self) -> str:
        """Bind a local prepared cache to source manifest and preprocessing."""

        manifest_sha256 = hashlib.sha256(
            self.info.manifest_path.read_bytes()
        ).hexdigest()
        contract = {
            "schema_version": _PREPARED_CACHE_SCHEMA_VERSION,
            "manifest_sha256": manifest_sha256,
            "image_size": list(self.image_size),
            "depth_min": self.depth_min,
            "depth_max": self.depth_max,
            "detection_min_area": self.detection_min_area,
            "use_packaged_detection": self.use_packaged_detection,
            "rgb_resize": "pillow_bilinear_uint8",
            "dense_resize": "pillow_nearest",
            "depth_storage": "float16_exact_source_values",
            "detection_classes": list(SANPO_DETECTION_CLASS_NAMES),
            "segmentation_classes": list(SANPO_SEGMENTATION_CLASS_NAMES),
        }
        encoded = json.dumps(
            contract,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _ensure_prepared_cache_marker(self) -> None:
        assert self.prepared_cache_dir is not None
        marker = self.prepared_cache_dir / "cache_manifest.json"
        expected = {
            "schema_version": _PREPARED_CACHE_SCHEMA_VERSION,
            "cache_key": self._prepared_cache_key,
            "sample_count": len(self),
        }
        if marker.exists():
            try:
                observed = json.loads(marker.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"cannot read SANPO prepared-cache marker: {marker}") from exc
            if observed != expected:
                raise RuntimeError(f"SANPO prepared-cache marker mismatch: {marker}")
            return
        temporary = marker.with_name(f".{marker.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(
                json.dumps(expected, separators=(",", ":"), sort_keys=True),
                encoding="utf-8",
            )
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            try:
                os.link(temporary, marker)
            except FileExistsError:
                observed = json.loads(marker.read_text(encoding="utf-8"))
                if observed != expected:
                    raise RuntimeError(
                        f"SANPO prepared-cache marker mismatch: {marker}"
                    )
        finally:
            temporary.unlink(missing_ok=True)

    def _load_rgb_uint8(self, path: Path) -> Tensor:
        with Image.open(path) as image:
            image = image.convert("RGB").resize(
                (self.image_size[1], self.image_size[0]),
                resample=Image.Resampling.BILINEAR,
            )
            array = np.asarray(image, dtype=np.uint8).copy()
        return torch.from_numpy(array)

    def _prepared_cache_path(self, index: int) -> Path | None:
        if self.prepared_cache_dir is None:
            return None
        return self.prepared_cache_dir / f"{index:08d}.pt"

    def prepared_cache_status(self) -> dict[str, int | bool | str | None]:
        """Return a cheap local-cache count and conservative disk estimate."""

        estimates = [self._estimated_prepared_sample_bytes(index) for index in range(len(self))]
        estimated_sample_bytes = max(estimates, default=0)
        if self.prepared_cache_dir is None:
            return {
                "enabled": False,
                "cache_key": self._prepared_cache_key,
                "cache_dir": None,
                "sample_count": len(self),
                "ready_samples": 0,
                "pending_samples": len(self),
                "estimated_sample_bytes": estimated_sample_bytes,
                "estimated_pending_bytes": sum(estimates),
                "cached_bytes": 0,
            }
        ready_indices: set[int] = set()
        cached_bytes = 0
        for path in self.prepared_cache_dir.glob("*.pt"):
            if len(path.stem) != 8 or not path.stem.isdigit():
                continue
            index = int(path.stem)
            if 0 <= index < len(self):
                ready_indices.add(index)
                try:
                    cached_bytes += path.stat().st_size
                except OSError:
                    pass
        pending = len(self) - len(ready_indices)
        return {
            "enabled": True,
            "cache_key": self._prepared_cache_key,
            "cache_dir": str(self.prepared_cache_dir),
            "sample_count": len(self),
            "ready_samples": len(ready_indices),
            "pending_samples": pending,
            "estimated_sample_bytes": estimated_sample_bytes,
            "estimated_pending_bytes": sum(
                estimate
                for index, estimate in enumerate(estimates)
                if index not in ready_indices
            ),
            "cached_bytes": cached_bytes,
        }

    def _estimated_prepared_sample_bytes(self, index: int) -> int:
        cached = self._prepared_sample_estimates.get(index)
        if cached is not None:
            return cached
        sample = self.manifest["samples"][index]
        target_pixels = self.image_size[0] * self.image_size[1]
        # Three RGB uint8 frames (9), segmentation uint8 (1), float16 depth
        # (2), and depth validity (1) use exactly 13 bytes per output pixel.
        fixed_tensors = 13 * target_pixels
        detection_relative = (
            sample.get("detection_path") if self.use_packaged_detection else None
        )
        if detection_relative is not None:
            detection = _read_detection_target(
                self._path(detection_relative, "detection_path")
            )
            detection_bytes = (
                int(detection["boxes"].shape[0]) * (4 * 4 + 8)
                + int(detection["ignore_boxes"].shape[0]) * (4 * 4)
            )
        else:
            panoptic_path = self._path(sample["panoptic_path"], "panoptic_path")
            with Image.open(panoptic_path) as image:
                source_width, source_height = image.size
            # With 8-connectivity, isolated components occupy at most every
            # second row and column. A positive component uses four float32 box
            # values plus one int64 label (24 bytes); ignore components use less.
            component_upper_bound = math.ceil(source_height / 2) * math.ceil(
                source_width / 2
            )
            detection_bytes = component_upper_bound * 24
        # The fixed allowance covers tensor metadata, zip records, alignment,
        # filesystem allocation and the small identity payload.
        estimate = fixed_tensors + detection_bytes + 64 * 1024
        self._prepared_sample_estimates[index] = estimate
        return estimate

    def _validate_prepared_sample(
        self,
        payload: object,
        *,
        index: int,
    ) -> dict[str, Tensor]:
        if not isinstance(payload, Mapping):
            raise ValueError("prepared sample must be a mapping")
        if (
            payload.get("schema_version") != _PREPARED_CACHE_SCHEMA_VERSION
            or payload.get("cache_key") != self._prepared_cache_key
            or payload.get("sample_index") != index
        ):
            raise ValueError("prepared sample identity mismatch")
        tensors = payload.get("tensors")
        if not isinstance(tensors, Mapping):
            raise ValueError("prepared sample has no tensor mapping")
        required = {
            "rgb_uint8",
            "segmentation_uint8",
            "depth_float16",
            "depth_valid",
            "boxes",
            "labels",
            "ignore_boxes",
        }
        if set(tensors) != required or any(
            not isinstance(tensors[name], Tensor) for name in required
        ):
            raise ValueError("prepared sample tensor schema mismatch")
        checked = {name: tensors[name] for name in required}
        height, width = self.image_size
        expected = {
            "rgb_uint8": (torch.uint8, (3, height, width, 3)),
            "segmentation_uint8": (torch.uint8, (height, width)),
            "depth_float16": (torch.float16, (1, height, width)),
            "depth_valid": (torch.bool, (1, height, width)),
        }
        for name, (dtype, shape) in expected.items():
            value = checked[name]
            if value.dtype != dtype or tuple(value.shape) != shape:
                raise ValueError(f"prepared sample {name} shape/dtype mismatch")
        boxes = checked["boxes"]
        labels = checked["labels"]
        ignore_boxes = checked["ignore_boxes"]
        if (
            boxes.dtype != torch.float32
            or boxes.ndim != 2
            or boxes.shape[1:] != (4,)
            or labels.dtype != torch.int64
            or labels.ndim != 1
            or labels.shape[0] != boxes.shape[0]
            or ignore_boxes.dtype != torch.float32
            or ignore_boxes.ndim != 2
            or ignore_boxes.shape[1:] != (4,)
            or not bool(torch.isfinite(boxes).all())
            or not bool(torch.isfinite(ignore_boxes).all())
        ):
            raise ValueError("prepared sample detection tensors are invalid")
        return checked

    def _read_prepared_sample(self, path: Path, *, index: int) -> dict[str, Tensor]:
        try:
            payload = torch.load(
                path,
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
            return self._validate_prepared_sample(payload, index=index)
        except (OSError, RuntimeError, ValueError, EOFError, pickle.UnpicklingError) as exc:
            raise ValueError(f"cannot read SANPO prepared sample: {path}") from exc

    def _write_prepared_sample(
        self,
        path: Path,
        *,
        index: int,
        tensors: Mapping[str, Tensor],
    ) -> None:
        payload = {
            "schema_version": _PREPARED_CACHE_SCHEMA_VERSION,
            "cache_key": self._prepared_cache_key,
            "sample_index": index,
            "tensors": {name: value.contiguous() for name, value in tensors.items()},
        }
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            torch.save(payload, temporary)
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def _build_prepared_sample(self, index: int) -> dict[str, Tensor]:
        sample = self.manifest["samples"][index]
        rgb_uint8 = torch.stack(
            [
                self._load_rgb_uint8(self._path(relative, "rgb_context_path"))
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
        segmentation = np.full(
            source_labels.shape,
            SANPO_SEGMENTATION_IGNORE_INDEX,
            dtype=np.uint8,
        )
        valid_semantic = (source_labels >= 1) & (source_labels <= 30)
        segmentation[valid_semantic] = source_labels[valid_semantic] - 1

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
        return {
            "rgb_uint8": rgb_uint8,
            "segmentation_uint8": torch.from_numpy(segmentation),
            # SANPO source values are float16. Nearest-neighbour resize only
            # selects those exact values, so float16 cache storage is lossless.
            "depth_float16": torch.from_numpy(depth).to(torch.float16).unsqueeze(0),
            "depth_valid": torch.from_numpy(depth_valid.copy()).unsqueeze(0),
            "boxes": _scale_boxes(detection["boxes"], detection["valid_size"], self.image_size),
            "labels": torch.as_tensor(detection["labels"], dtype=torch.int64),
            "ignore_boxes": _scale_boxes(
                detection["ignore_boxes"], detection["valid_size"], self.image_size
            ),
        }

    def _prepared_sample(self, index: int) -> dict[str, Tensor]:
        cache_path = self._prepared_cache_path(index)
        if cache_path is not None and cache_path.is_file():
            try:
                return self._read_prepared_sample(cache_path, index=index)
            except ValueError:
                # The cache is derived, local, and replaceable. The immutable
                # archive remains authoritative when a cache file is damaged.
                pass
        tensors = self._build_prepared_sample(index)
        if cache_path is not None:
            self._write_prepared_sample(cache_path, index=index, tensors=tensors)
        return tensors

    def __getitem__(self, index: int) -> tuple[Tensor, dict[str, Any]]:
        tensors = self._prepared_sample(index)
        clip = tensors["rgb_uint8"].permute(0, 3, 1, 2).float().div_(255.0)
        if self.normalize:
            clip = (clip - self._mean) / self._std
        segmentation = tensors["segmentation_uint8"].to(torch.int64)
        segmentation_valid = segmentation != SANPO_SEGMENTATION_IGNORE_INDEX
        detection_target = {
            "boxes": tensors["boxes"],
            "labels": tensors["labels"],
            "valid_size": self.image_size,
            "ignore_boxes": tensors["ignore_boxes"],
        }

        targets: dict[str, Any] = {
            "detection": detection_target,
            "segmentation": segmentation,
            "segmentation_valid": segmentation_valid,
            "depth": tensors["depth_float16"].to(torch.float32),
            "depth_valid": tensors["depth_valid"],
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
