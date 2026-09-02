"""Strict NYUv2 RGB/semantic-segmentation/depth dataset adapter.

The adapter is intentionally explicit about the two dataset conventions that
are most often handled incorrectly: semantic label IDs and depth units.  A
caller must provide a raw-label mapping and a depth unit scale; neither is
guessed from pixel values.  Official train/test membership is read from
``gt_sets`` and validated before any sample is exposed.

Inputs are static RGB tensors (``C,H,W``), not synthetic temporal clips.  The
default output size is 288x384, preserving NYUv2's 4:3 aspect ratio.  Training
augmentation uses one deterministic geometric transform for RGB, semantic
labels and depth, followed by RGB-only colour jitter.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


NYUV2_IMAGE_SIZE = (288, 384)
IMAGENET_RGB_MEAN = (0.485, 0.456, 0.406)
IMAGENET_RGB_STD = (0.229, 0.224, 0.225)

_IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
_TARGET_EXTENSIONS = {".npy", ".png", ".tif", ".tiff"}
_SPLIT_EXTENSIONS = {".json", ".npy", ".txt"}
_MODALITY_PREFIX = re.compile(
    r"^(?:depth|image|images|img|label|labels|seg|segmentation|semantic)[_-]+",
    flags=re.IGNORECASE,
)
_MODALITY_SUFFIX = re.compile(
    r"[_-]+(?:depth|image|images|img|label|labels|rgb|seg|segmentation|semantic)$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class Nyuv2Sample:
    """One fully matched NYUv2 RGB/segmentation/depth sample."""

    key: str
    image_path: Path
    segmentation_path: Path
    depth_path: Path


@dataclass(frozen=True)
class Nyuv2Index:
    """Validated official NYUv2 train/test index."""

    root: Path
    train: tuple[Nyuv2Sample, ...]
    test: tuple[Nyuv2Sample, ...]
    heldout_source_name: Literal["val", "test"]

    @property
    def sample_count(self) -> int:
        return len(self.train) + len(self.test)


@dataclass(frozen=True)
class Nyuv2Augmentation:
    """Light, synchronized augmentation for the official training split."""

    horizontal_flip_probability: float = 0.5
    scale_min: float = 0.75
    scale_max: float = 1.25
    brightness: float = 0.10
    contrast: float = 0.10
    saturation: float = 0.08
    class_aware_crop_probability: float = 0.35
    rare_classes: tuple[int, ...] = ()
    blur_probability: float = 0.08
    blur_kernel_size: int = 3

    def __post_init__(self) -> None:
        values = {
            "horizontal_flip_probability": self.horizontal_flip_probability,
            "scale_min": self.scale_min,
            "scale_max": self.scale_max,
            "brightness": self.brightness,
            "contrast": self.contrast,
            "saturation": self.saturation,
            "class_aware_crop_probability": self.class_aware_crop_probability,
            "blur_probability": self.blur_probability,
        }
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in values.values()
        ):
            raise ValueError("NYUv2 augmentation values must be finite numbers")
        if not 0.0 <= self.horizontal_flip_probability <= 1.0:
            raise ValueError("horizontal_flip_probability must be in [0,1]")
        for name in ("class_aware_crop_probability", "blur_probability"):
            if not 0.0 <= float(getattr(self, name)) <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
        if self.scale_min <= 0.0 or self.scale_max < self.scale_min:
            raise ValueError("augmentation scale range must satisfy 0 < min <= max")
        for name in ("brightness", "contrast", "saturation"):
            if not 0.0 <= float(getattr(self, name)) <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
        if not isinstance(self.rare_classes, (tuple, list)) or any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer))
            or int(value) < 0
            for value in self.rare_classes
        ):
            raise ValueError("rare_classes must contain non-negative integers")
        object.__setattr__(self, "rare_classes", tuple(int(value) for value in self.rare_classes))
        if (
            isinstance(self.blur_kernel_size, bool)
            or not isinstance(self.blur_kernel_size, int)
            or self.blur_kernel_size < 3
            or self.blur_kernel_size % 2 == 0
        ):
            raise ValueError("blur_kernel_size must be an odd integer >= 3")


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


def _resolve_root(filename: str | PathLike[str]) -> Path:
    root = Path(filename).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"NYUv2 root is not a directory: {root}")
    required = ("images", "segmentation", "depth", "gt_sets")
    if all((root / name).is_dir() for name in required):
        return root
    candidates = {
        path.parent
        for path in root.rglob("images")
        if path.is_dir() and all((path.parent / name).is_dir() for name in required)
    }
    if len(candidates) != 1:
        raise ValueError(
            "expected exactly one extracted NYUv2 root containing images, "
            "segmentation, depth and gt_sets"
        )
    return candidates.pop().resolve()


def _canonical_component(value: str) -> str:
    value = _MODALITY_PREFIX.sub("", value).casefold()
    value = _MODALITY_SUFFIX.sub("", value)
    if value.isdigit():
        return str(int(value))
    return value


def _canonical_relative_stem(path: Path, base: Path) -> str:
    relative = path.relative_to(base).with_suffix("")
    parts = list(relative.parts)
    parts[-1] = _canonical_component(parts[-1])
    return PurePosixPath(*parts).as_posix().casefold()


def _collect_modality(
    directory: Path,
    *,
    extensions: set[str],
    modality: str,
) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(directory.rglob("*")):
        if (
            not path.is_file()
            or path.name.startswith(".")
            or path.suffix.casefold() not in extensions
        ):
            continue
        key = _canonical_relative_stem(path, directory)
        previous = result.get(key)
        if previous is not None:
            raise ValueError(
                f"duplicate canonical {modality} sample key {key!r}: "
                f"{previous} and {path}"
            )
        result[key] = path.resolve()
    if not result:
        raise ValueError(f"NYUv2 {modality} directory contains no supported files")
    return result


def _by_unique_basename(
    values: Mapping[str, Path], *, modality: str
) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for key, path in values.items():
        basename = PurePosixPath(key).name
        if basename in result:
            raise ValueError(
                f"cannot align {modality} by basename because {basename!r} "
                "is ambiguous"
            )
        result[basename] = path
    return result


def _split_values(path: Path) -> list[str]:
    try:
        if path.suffix.casefold() == ".txt":
            values: list[object] = []
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                line = line.split("#", 1)[0].strip()
                if line:
                    values.append(re.split(r"[\s,]+", line, maxsplit=1)[0])
        elif path.suffix.casefold() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, Mapping):
                lists = [value for value in payload.values() if isinstance(value, list)]
                if len(lists) != 1:
                    raise ValueError("split JSON object must contain exactly one list")
                values = lists[0]
            elif isinstance(payload, list):
                values = payload
            else:
                raise ValueError("split JSON must contain a list")
        elif path.suffix.casefold() == ".npy":
            payload = np.load(path, allow_pickle=False)
            values = np.asarray(payload).reshape(-1).tolist()
        else:  # pragma: no cover - caller filters extensions
            raise ValueError(f"unsupported split file: {path}")
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"cannot read NYUv2 split file: {path}") from exc
    if not values:
        raise ValueError(f"NYUv2 split file is empty: {path}")
    result: list[str] = []
    for value in values:
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if not isinstance(value, (str, int, np.integer)) or isinstance(value, bool):
            raise ValueError(f"invalid sample identifier in split file: {path}")
        result.append(str(value).strip())
    return result


def _split_kind(path: Path) -> Literal["train", "val", "test"] | None:
    tokens = set(re.split(r"[^a-z0-9]+", path.stem.casefold()))
    kinds = [kind for kind in ("train", "val", "test") if kind in tokens]
    if len(kinds) != 1:
        return None
    return kinds[0]  # type: ignore[return-value]


def _sample_aliases(samples: Mapping[str, Nyuv2Sample]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    ambiguous: set[str] = set()
    for key in samples:
        basename = PurePosixPath(key).name
        candidates = {key, basename}
        if basename.isdigit():
            candidates.add(str(int(basename)))
        for alias in candidates:
            previous = aliases.get(alias)
            if previous is not None and previous != key:
                ambiguous.add(alias)
            else:
                aliases[alias] = key
    for alias in ambiguous:
        aliases.pop(alias, None)
    return aliases


def _normalise_split_identifier(value: str) -> tuple[str, str]:
    posix = PurePosixPath(value.replace("\\", "/"))
    parts = [part for part in posix.parts if part not in {"", "."}]
    if not parts or ".." in parts:
        raise ValueError(f"unsafe NYUv2 split identifier: {value!r}")
    if parts[0].casefold() in {"depth", "images", "segmentation"}:
        parts = parts[1:]
    if not parts:
        raise ValueError(f"invalid NYUv2 split identifier: {value!r}")
    parts[-1] = Path(parts[-1]).stem
    parts[-1] = _canonical_component(parts[-1])
    full = PurePosixPath(*parts).as_posix().casefold()
    return full, PurePosixPath(full).name


def discover_nyuv2(filename: str | PathLike[str]) -> Nyuv2Index:
    """Discover matched samples and validate the complete official split.

    The official test split remains a separate tuple.  No validation subset is
    synthesized and no test sample is ever folded into training.
    """

    root = _resolve_root(filename)
    images = _collect_modality(
        root / "images", extensions=_IMAGE_EXTENSIONS, modality="image"
    )
    segmentations = _collect_modality(
        root / "segmentation",
        extensions=_TARGET_EXTENSIONS,
        modality="segmentation",
    )
    depths = _collect_modality(
        root / "depth", extensions=_TARGET_EXTENSIONS, modality="depth"
    )
    if set(images) != set(segmentations) or set(images) != set(depths):
        images = _by_unique_basename(images, modality="images")
        segmentations = _by_unique_basename(
            segmentations, modality="segmentations"
        )
        depths = _by_unique_basename(depths, modality="depths")
    if set(images) != set(segmentations) or set(images) != set(depths):
        missing = {
            "missing_images": sorted((set(segmentations) | set(depths)) - set(images)),
            "missing_segmentations": sorted(
                (set(images) | set(depths)) - set(segmentations)
            ),
            "missing_depths": sorted((set(images) | set(segmentations)) - set(depths)),
        }
        summary = {name: values[:5] for name, values in missing.items() if values}
        raise ValueError(f"NYUv2 modalities are not one-to-one matched: {summary}")

    samples = {
        key: Nyuv2Sample(
            key=key,
            image_path=images[key],
            segmentation_path=segmentations[key],
            depth_path=depths[key],
        )
        for key in sorted(images)
    }
    source_split_files: dict[str, Path] = {}
    for path in sorted((root / "gt_sets").rglob("*")):
        if not path.is_file() or path.suffix.casefold() not in _SPLIT_EXTENSIONS:
            continue
        kind = _split_kind(path)
        if kind is None:
            continue
        if kind in source_split_files:
            raise ValueError(f"multiple official NYUv2 {kind} split files found")
        source_split_files[kind] = path
    if "train" not in source_split_files:
        raise ValueError("gt_sets must contain one official train split file")
    heldout_names = set(source_split_files) & {"val", "test"}
    if len(heldout_names) != 1:
        raise ValueError(
            "gt_sets must contain exactly one held-out val or test split file"
        )
    heldout_source_name = heldout_names.pop()
    split_files = {
        "train": source_split_files["train"],
        "test": source_split_files[heldout_source_name],
    }

    aliases = _sample_aliases(samples)
    resolved: dict[str, tuple[str, ...]] = {}
    for kind in ("train", "test"):
        keys: list[str] = []
        seen: set[str] = set()
        for value in _split_values(split_files[kind]):
            full, basename = _normalise_split_identifier(value)
            key = aliases.get(full) or aliases.get(basename)
            if key is None:
                raise ValueError(
                    f"official {kind} split references an unknown or ambiguous "
                    f"sample: {value!r}"
                )
            if key in seen:
                raise ValueError(f"duplicate sample in official {kind} split: {key}")
            seen.add(key)
            keys.append(key)
        resolved[kind] = tuple(keys)

    overlap = set(resolved["train"]) & set(resolved["test"])
    if overlap:
        raise ValueError(f"official NYUv2 train/test overlap: {sorted(overlap)[:5]}")
    covered = set(resolved["train"]) | set(resolved["test"])
    if covered != set(samples):
        missing = sorted(set(samples) - covered)
        raise ValueError(
            "official NYUv2 splits do not cover every matched sample: "
            f"{missing[:5]}"
        )
    return Nyuv2Index(
        root=root,
        train=tuple(samples[key] for key in resolved["train"]),
        test=tuple(samples[key] for key in resolved["test"]),
        heldout_source_name=heldout_source_name,  # type: ignore[arg-type]
    )


def _squeeze_single_channel(array: np.ndarray, *, name: str, path: Path) -> np.ndarray:
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    elif array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"NYUv2 {name} must be a single-channel array: {path}")
    return array


def read_nyuv2_segmentation(filename: str | PathLike[str]) -> np.ndarray:
    """Read raw integer semantic IDs without applying an assumed schema."""

    path = Path(filename)
    try:
        if path.suffix.casefold() == ".npy":
            array = np.load(path, allow_pickle=False)
        else:
            with Image.open(path) as image:
                image.load()
                array = np.asarray(image)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read NYUv2 segmentation: {path}") from exc
    array = _squeeze_single_channel(np.asarray(array), name="segmentation", path=path)
    if array.dtype.kind not in {"b", "i", "u"}:
        raise ValueError(f"NYUv2 segmentation IDs must be integers: {path}")
    return array.astype(np.int64, copy=True)


def read_nyuv2_depth(
    filename: str | PathLike[str],
    *,
    unit_scale: float,
) -> np.ndarray:
    """Read PNG/TIFF/NPY depth and explicitly convert source units to metres."""

    if (
        isinstance(unit_scale, bool)
        or not isinstance(unit_scale, (int, float))
        or not math.isfinite(float(unit_scale))
        or float(unit_scale) <= 0.0
    ):
        raise ValueError("unit_scale must be an explicit positive finite number")
    path = Path(filename)
    try:
        if path.suffix.casefold() == ".npy":
            array = np.load(path, allow_pickle=False)
        else:
            with Image.open(path) as image:
                image.load()
                array = np.asarray(image)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read NYUv2 depth: {path}") from exc
    array = _squeeze_single_channel(np.asarray(array), name="depth", path=path)
    if array.dtype.kind not in {"f", "i", "u"}:
        raise ValueError(f"NYUv2 depth must contain numeric values: {path}")
    return array.astype(np.float32, copy=True) * float(unit_scale)


def scan_nyuv2_label_ids(
    samples: Nyuv2Index | Sequence[Nyuv2Sample],
    *,
    include_heldout: bool = False,
) -> tuple[int, ...]:
    """Return raw semantic IDs for an explicit label-map audit.

    Passing an index scans official training data only by default.  Reading
    held-out labels requires an explicit opt-in and must not be used to choose
    the training schema or hyperparameters.
    """

    if not isinstance(include_heldout, bool):
        raise TypeError("include_heldout must be a boolean")
    if isinstance(samples, Nyuv2Index):
        records: Sequence[Nyuv2Sample] = samples.train
        if include_heldout:
            records = (*records, *samples.test)
    else:
        if include_heldout:
            raise ValueError("include_heldout is only valid with a Nyuv2Index")
        records = samples
    values: set[int] = set()
    for sample in records:
        raw = read_nyuv2_segmentation(sample.segmentation_path)
        values.update(int(value) for value in np.unique(raw))
    return tuple(sorted(values))


def _validated_label_contract(
    label_mapping: Mapping[int, int],
    *,
    source_ignore_labels: Sequence[int],
    num_classes: int,
    ignore_index: int,
) -> tuple[dict[int, int], frozenset[int]]:
    if (
        isinstance(num_classes, bool)
        or not isinstance(num_classes, int)
        or num_classes <= 1
    ):
        raise ValueError("num_classes must be an integer greater than one")
    if isinstance(ignore_index, bool) or not isinstance(ignore_index, int):
        raise TypeError("ignore_index must be an integer")
    if 0 <= ignore_index < num_classes:
        raise ValueError("ignore_index must not overlap a valid training class")
    if not isinstance(label_mapping, Mapping) or not label_mapping:
        raise ValueError("label_mapping must explicitly map raw IDs to train IDs")
    mapping: dict[int, int] = {}
    for source, target in label_mapping.items():
        if (
            isinstance(source, bool)
            or not isinstance(source, (int, np.integer))
            or isinstance(target, bool)
            or not isinstance(target, (int, np.integer))
        ):
            raise TypeError("label_mapping keys and values must be integers")
        if not 0 <= int(target) < num_classes:
            raise ValueError("label_mapping target is outside [0,num_classes)")
        mapping[int(source)] = int(target)
    ignored = frozenset(int(value) for value in source_ignore_labels)
    if any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer))
        for value in source_ignore_labels
    ):
        raise TypeError("source_ignore_labels must contain integers")
    overlap = set(mapping) & ignored
    if overlap:
        raise ValueError(
            f"raw IDs cannot be both mapped and ignored: {sorted(overlap)}"
        )
    return mapping, ignored


def _map_segmentation(
    raw: np.ndarray,
    *,
    mapping: Mapping[int, int],
    ignored: frozenset[int],
    ignore_index: int,
    path: Path,
) -> Tensor:
    output = np.full(raw.shape, ignore_index, dtype=np.int64)
    observed = {int(value) for value in np.unique(raw)}
    unknown = observed - set(mapping) - ignored
    if unknown:
        raise ValueError(
            f"NYUv2 segmentation contains unmapped raw IDs {sorted(unknown)}: {path}"
        )
    for source, target in mapping.items():
        output[raw == source] = target
    return torch.from_numpy(output)


def _load_rgb(path: Path) -> Tensor:
    try:
        with Image.open(path) as image:
            image = image.convert("RGB")
            array = np.asarray(image, dtype=np.uint8).copy()
    except OSError as exc:
        raise ValueError(f"cannot read NYUv2 RGB image: {path}") from exc
    return torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0)


def _uniform(generator: torch.Generator, low: float, high: float) -> float:
    if low == high:
        return float(low)
    return float(torch.empty(()).uniform_(low, high, generator=generator))


def _resize_and_crop(
    rgb: Tensor,
    segmentation: Tensor,
    depth: Tensor,
    depth_valid: Tensor,
    *,
    output_hw: tuple[int, int],
    scale_multiplier: float,
    generator: torch.Generator | None,
    focus_coordinate: tuple[int, int] | None = None,
    segmentation_pad_value: int = 255,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    source_h, source_w = segmentation.shape
    output_h, output_w = output_hw
    base_scale = max(output_h / source_h, output_w / source_w)
    scale = base_scale * scale_multiplier
    resized_h = int(math.ceil(source_h * scale))
    resized_w = int(math.ceil(source_w * scale))
    rgb = F.interpolate(
        rgb.unsqueeze(0),
        size=(resized_h, resized_w),
        mode="bilinear",
        align_corners=False,
    )[0]
    segmentation = F.interpolate(
        segmentation[None, None].float(),
        size=(resized_h, resized_w),
        mode="nearest",
    )[0, 0].long()
    valid_coverage = F.interpolate(
        depth_valid[None].float(), size=(resized_h, resized_w), mode="bilinear", align_corners=False
    )[0]
    depth = F.interpolate(
        (depth * depth_valid.float())[None],
        size=(resized_h, resized_w),
        mode="bilinear",
        align_corners=False,
    )[0] / valid_coverage.clamp_min(1e-6)
    depth_valid = valid_coverage > 1e-6
    pad_h = max(output_h - resized_h, 0)
    pad_w = max(output_w - resized_w, 0)
    if pad_h or pad_w:
        if generator is None:
            pad_top = pad_h // 2
            pad_left = pad_w // 2
        else:
            pad_top = int(torch.randint(pad_h + 1, (), generator=generator))
            pad_left = int(torch.randint(pad_w + 1, (), generator=generator))
        pad_bottom = pad_h - pad_top
        pad_right = pad_w - pad_left
        padding = (pad_left, pad_right, pad_top, pad_bottom)
        rgb = F.pad(rgb, padding, mode="replicate")
        segmentation = F.pad(
            segmentation,
            padding,
            value=int(segmentation_pad_value),
        )
        depth = F.pad(depth, padding, value=0.0)
        depth_valid = F.pad(depth_valid, padding, value=False)
        resized_h += pad_h
        resized_w += pad_w
    excess_h = resized_h - output_h
    excess_w = resized_w - output_w
    if focus_coordinate is not None:
        focus_y, focus_x = focus_coordinate
        focus_y = int(round((focus_y + 0.5) * scale - 0.5))
        focus_x = int(round((focus_x + 0.5) * scale - 0.5))
        top = min(max(focus_y - output_h // 2, 0), excess_h)
        left = min(max(focus_x - output_w // 2, 0), excess_w)
    elif generator is None:
        top = excess_h // 2
        left = excess_w // 2
    else:
        top = int(torch.randint(excess_h + 1, (), generator=generator))
        left = int(torch.randint(excess_w + 1, (), generator=generator))
    region = (..., slice(top, top + output_h), slice(left, left + output_w))
    return rgb[region], segmentation[region], depth[region], depth_valid[region]


def _colour_jitter(
    rgb: Tensor,
    augmentation: Nyuv2Augmentation,
    generator: torch.Generator,
) -> Tensor:
    brightness = _uniform(
        generator, 1.0 - augmentation.brightness, 1.0 + augmentation.brightness
    )
    contrast = _uniform(
        generator, 1.0 - augmentation.contrast, 1.0 + augmentation.contrast
    )
    saturation = _uniform(
        generator, 1.0 - augmentation.saturation, 1.0 + augmentation.saturation
    )
    rgb = rgb * brightness
    rgb = (rgb - rgb.mean(dim=(-2, -1), keepdim=True)) * contrast + rgb.mean(
        dim=(-2, -1), keepdim=True
    )
    gray = (
        0.2989 * rgb[0:1] + 0.5870 * rgb[1:2] + 0.1140 * rgb[2:3]
    )
    return (gray + (rgb - gray) * saturation).clamp_(0.0, 1.0)


def _light_blur(rgb: Tensor, kernel_size: int) -> Tensor:
    """Apply a tiny channel-wise average blur without extra dependencies."""

    channels = rgb.shape[0]
    kernel = torch.ones(
        channels, 1, kernel_size, kernel_size, dtype=rgb.dtype, device=rgb.device
    ) / float(kernel_size * kernel_size)
    radius = kernel_size // 2
    padded = F.pad(
        rgb.unsqueeze(0),
        (radius, radius, radius, radius),
        mode="replicate",
    )
    return F.conv2d(padded, kernel, groups=channels)[0]


class Nyuv2Dataset(Dataset[tuple[Tensor, dict[str, Tensor]]]):
    """Official-split NYUv2 adapter for static RGB segmentation+depth training."""

    def __init__(
        self,
        root: str | PathLike[str],
        *,
        split: Literal["train", "test"],
        num_classes: int,
        label_mapping: Mapping[int, int],
        source_ignore_labels: Sequence[int],
        depth_unit_scale: float,
        image_size: Sequence[int] = NYUV2_IMAGE_SIZE,
        ignore_index: int = 255,
        depth_min: float = 0.1,
        depth_max: float = 10.0,
        augmentation: Nyuv2Augmentation | None = None,
        seed: int = 42,
        normalize: bool = True,
        index: Nyuv2Index | None = None,
        samples: Sequence[Nyuv2Sample] | None = None,
    ) -> None:
        super().__init__()
        if split not in {"train", "test"}:
            raise ValueError("NYUv2 split must be 'train' or 'test'")
        if split == "test" and augmentation is not None:
            raise ValueError("official NYUv2 test preprocessing must be deterministic")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError("seed must be an integer")
        if not isinstance(normalize, bool):
            raise TypeError("normalize must be a boolean")
        if (
            isinstance(depth_unit_scale, bool)
            or not isinstance(depth_unit_scale, (int, float))
            or not math.isfinite(float(depth_unit_scale))
            or float(depth_unit_scale) <= 0.0
        ):
            raise ValueError("depth_unit_scale must be explicitly positive and finite")
        if (
            isinstance(depth_min, bool)
            or not isinstance(depth_min, (int, float))
            or not math.isfinite(float(depth_min))
            or float(depth_min) < 0.0
            or isinstance(depth_max, bool)
            or not isinstance(depth_max, (int, float))
            or not math.isfinite(float(depth_max))
            or float(depth_max) <= float(depth_min)
        ):
            raise ValueError("depth range must satisfy 0 <= min < max")
        self.index = discover_nyuv2(root) if index is None else index
        resolved_root = _resolve_root(root)
        if self.index.root != resolved_root:
            raise ValueError("provided NYUv2 index belongs to a different dataset root")
        official_records = self.index.train if split == "train" else self.index.test
        if samples is None:
            self.records = official_records
        else:
            requested = tuple(samples)
            if not requested:
                raise ValueError("explicit NYUv2 sample subset must not be empty")
            if len({sample.key for sample in requested}) != len(requested):
                raise ValueError("explicit NYUv2 sample subset contains duplicates")
            official_by_key = {sample.key: sample for sample in official_records}
            unknown = [
                sample.key
                for sample in requested
                if official_by_key.get(sample.key) != sample
            ]
            if unknown:
                raise ValueError(
                    f"sample subset is outside official {split} membership: "
                    f"{unknown[:5]}"
                )
            self.records = requested
        self.split = split
        self.image_size = _positive_hw(image_size, "image_size")
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.label_mapping, self.source_ignore_labels = _validated_label_contract(
            label_mapping,
            source_ignore_labels=source_ignore_labels,
            num_classes=num_classes,
            ignore_index=ignore_index,
        )
        self.depth_unit_scale = float(depth_unit_scale)
        self.depth_min = float(depth_min)
        self.depth_max = float(depth_max)
        self.augmentation = augmentation
        self.seed = seed
        self.epoch = 0
        self.normalize = normalize
        self._mean = torch.tensor(IMAGENET_RGB_MEAN).reshape(3, 1, 1)
        self._std = torch.tensor(IMAGENET_RGB_STD).reshape(3, 1, 1)

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        """Set the deterministic augmentation epoch (safe with worker copies)."""

        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        self.epoch = epoch

    def _generator(self, index: int) -> torch.Generator:
        digest = hashlib.blake2b(
            f"{self.seed}:{self.epoch}:{index}".encode("ascii"), digest_size=8
        ).digest()
        return torch.Generator().manual_seed(int.from_bytes(digest, "little"))

    def __getitem__(self, index: int) -> tuple[Tensor, dict[str, Tensor]]:
        record = self.records[index]
        rgb = _load_rgb(record.image_path)
        raw_segmentation = read_nyuv2_segmentation(record.segmentation_path)
        segmentation = _map_segmentation(
            raw_segmentation,
            mapping=self.label_mapping,
            ignored=self.source_ignore_labels,
            ignore_index=self.ignore_index,
            path=record.segmentation_path,
        )
        source_depth = read_nyuv2_depth(
            record.depth_path, unit_scale=self.depth_unit_scale
        )
        source_hw = tuple(rgb.shape[-2:])
        if (
            tuple(segmentation.shape) != source_hw
            or tuple(source_depth.shape) != source_hw
        ):
            raise ValueError(
                f"NYUv2 RGB/segmentation/depth shapes disagree for {record.key}: "
                f"{source_hw}, {tuple(segmentation.shape)}, {tuple(source_depth.shape)}"
            )
        valid = (
            np.isfinite(source_depth)
            & (source_depth >= self.depth_min)
            & (source_depth <= self.depth_max)
        )
        source_depth = np.where(valid, source_depth, 0.0).astype(np.float32)
        depth = torch.from_numpy(source_depth).unsqueeze(0)
        depth_valid = torch.from_numpy(valid.copy()).unsqueeze(0)

        generator: torch.Generator | None = None
        scale_multiplier = 1.0
        do_flip = False
        focus_coordinate: tuple[int, int] | None = None
        do_blur = False
        if self.split == "train" and self.augmentation is not None:
            generator = self._generator(index)
            scale_multiplier = _uniform(
                generator, self.augmentation.scale_min, self.augmentation.scale_max
            )
            do_flip = bool(
                torch.rand((), generator=generator)
                < self.augmentation.horizontal_flip_probability
            )
            do_blur = bool(
                torch.rand((), generator=generator)
                < self.augmentation.blur_probability
            )
            if (
                self.augmentation.rare_classes
                and bool(
                    torch.rand((), generator=generator)
                    < self.augmentation.class_aware_crop_probability
                )
            ):
                present = [
                    int(class_id)
                    for class_id in self.augmentation.rare_classes
                    if bool((segmentation == int(class_id)).any())
                ]
                if present:
                    selected_class = present[
                        int(torch.randint(len(present), (), generator=generator))
                    ]
                    coordinates = torch.nonzero(segmentation == selected_class, as_tuple=False)
                    selected = coordinates[
                        int(torch.randint(coordinates.shape[0], (), generator=generator))
                    ]
                    focus_coordinate = (int(selected[0]), int(selected[1]))
        rgb, segmentation, depth, depth_valid = _resize_and_crop(
            rgb,
            segmentation,
            depth,
            depth_valid,
            output_hw=self.image_size,
            scale_multiplier=scale_multiplier,
            generator=generator,
            focus_coordinate=focus_coordinate,
            segmentation_pad_value=self.ignore_index,
        )
        if do_flip:
            rgb = rgb.flip(-1)
            segmentation = segmentation.flip(-1)
            depth = depth.flip(-1)
            depth_valid = depth_valid.flip(-1)
        if generator is not None:
            assert self.augmentation is not None
            rgb = _colour_jitter(rgb, self.augmentation, generator)
            if do_blur:
                rgb = _light_blur(rgb, self.augmentation.blur_kernel_size).clamp_(0.0, 1.0)
        depth = depth.contiguous()
        depth_valid = depth_valid.contiguous()
        depth[~depth_valid] = 0.0
        if self.normalize:
            rgb = (rgb - self._mean) / self._std
        return rgb.contiguous(), {
            "segmentation": segmentation.contiguous(),
            "segmentation_valid": (segmentation != self.ignore_index).contiguous(),
            "depth": depth,
            "depth_valid": depth_valid,
        }

    def sample_provenance(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        return {
            "official_split": self.split,
            "sample_key": record.key,
            "image_path": str(record.image_path),
            "segmentation_path": str(record.segmentation_path),
            "depth_path": str(record.depth_path),
            "depth_unit_scale": self.depth_unit_scale,
        }


def nyuv2_collate(
    samples: Sequence[tuple[Tensor, Mapping[str, Tensor]]],
) -> tuple[Tensor, dict[str, Tensor]]:
    """Stack the static NYUv2 segmentation+depth training contract."""

    if not samples:
        raise ValueError("cannot collate an empty NYUv2 batch")
    inputs = torch.stack([sample[0] for sample in samples])
    names = {"segmentation", "segmentation_valid", "depth", "depth_valid"}
    if any(set(targets) != names for _, targets in samples):
        raise ValueError("NYUv2 target schema mismatch")
    targets = {
        name: torch.stack([sample_targets[name] for _, sample_targets in samples])
        for name in sorted(names)
    }
    return inputs, targets


__all__ = [
    "NYUV2_IMAGE_SIZE",
    "Nyuv2Augmentation",
    "Nyuv2Dataset",
    "Nyuv2Index",
    "Nyuv2Sample",
    "discover_nyuv2",
    "nyuv2_collate",
    "read_nyuv2_depth",
    "read_nyuv2_segmentation",
    "scan_nyuv2_label_ids",
]
