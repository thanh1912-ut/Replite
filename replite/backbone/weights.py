"""Verified weight specifications and loading shared by Replite backbones."""

from __future__ import annotations

import hashlib
import math
import string
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from safetensors import SafetensorError
from safetensors.torch import load as load_safetensors
from torch import Tensor


class CheckpointDownloadError(RuntimeError):
    """Raised when a pretrained checkpoint cannot be obtained or read."""


class ChecksumMismatchError(CheckpointDownloadError):
    """Raised when a checkpoint fails its pinned SHA-256 verification."""


class CheckpointFormatError(RuntimeError):
    """Raised when verified bytes are not a valid safetensors checkpoint."""


@dataclass(frozen=True)
class PretrainedWeightsSpec:
    """Declarative specification of one verified pretrained checkpoint."""

    architecture: str
    repository: str
    revision: str
    sha256: str
    dataset: str = "imagenet-1k"
    input_size: tuple[int, int, int] = (3, 224, 224)
    test_input_size: tuple[int, int, int] | None = None
    fixed_input_size: bool = False
    interpolation: str = "bicubic"
    crop_pct: float = 0.875
    test_crop_pct: float | None = None
    crop_mode: str = "center"
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    filename: str = "model.safetensors"
    license: str = "apache-2.0"

    def __post_init__(self) -> None:
        """Canonicalize mutable inputs and validate checkpoint invariants."""

        for field_name in (
            "architecture",
            "repository",
            "revision",
            "dataset",
            "interpolation",
            "crop_mode",
            "filename",
            "license",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")

        if (
            not isinstance(self.sha256, str)
            or len(self.sha256) != 64
            or any(character not in string.hexdigits for character in self.sha256)
        ):
            raise ValueError("sha256 must contain exactly 64 hexadecimal characters")
        object.__setattr__(self, "sha256", self.sha256.lower())

        def image_size(value: Any, field_name: str) -> tuple[int, int, int]:
            try:
                values = tuple(value)
            except TypeError as exc:
                raise ValueError(f"{field_name} must contain three integers") from exc
            if len(values) != 3 or any(
                isinstance(item, bool) or not isinstance(item, Integral) or item <= 0
                for item in values
            ):
                raise ValueError(f"{field_name} must contain three positive integers")
            return tuple(int(item) for item in values)

        def channel_values(
            value: Any, field_name: str, *, positive: bool
        ) -> tuple[float, float, float]:
            try:
                values = tuple(value)
            except TypeError as exc:
                raise ValueError(f"{field_name} must contain three numbers") from exc
            if len(values) != 3 or any(
                isinstance(item, bool)
                or not isinstance(item, Real)
                or not math.isfinite(float(item))
                or (positive and float(item) <= 0)
                for item in values
            ):
                qualifier = "positive finite" if positive else "finite"
                raise ValueError(f"{field_name} must contain three {qualifier} numbers")
            return tuple(float(item) for item in values)

        def crop(value: Any, field_name: str) -> float:
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                or not 0 < float(value) <= 1
            ):
                raise ValueError(f"{field_name} must be in the interval (0, 1]")
            return float(value)

        object.__setattr__(
            self, "input_size", image_size(self.input_size, "input_size")
        )
        if self.input_size[0] != 3:
            raise ValueError("input_size must describe a three-channel RGB input")
        if self.test_input_size is not None:
            object.__setattr__(
                self,
                "test_input_size",
                image_size(self.test_input_size, "test_input_size"),
            )
            if self.test_input_size[0] != 3:
                raise ValueError(
                    "test_input_size must describe a three-channel RGB input"
                )
        object.__setattr__(
            self, "mean", channel_values(self.mean, "mean", positive=False)
        )
        object.__setattr__(self, "std", channel_values(self.std, "std", positive=True))
        object.__setattr__(self, "crop_pct", crop(self.crop_pct, "crop_pct"))
        if self.test_crop_pct is not None:
            object.__setattr__(
                self,
                "test_crop_pct",
                crop(self.test_crop_pct, "test_crop_pct"),
            )
        if not isinstance(self.fixed_input_size, bool):
            raise ValueError("fixed_input_size must be a boolean")

    def as_pretrained_cfg(self) -> dict[str, Any]:
        """Return complete, mutation-safe preprocessing and weight metadata."""

        return {
            "dataset": self.dataset,
            "architecture": self.architecture,
            "repository": self.repository,
            "revision": self.revision,
            "filename": self.filename,
            "sha256": self.sha256,
            "mean": self.mean,
            "std": self.std,
            "input_size": self.input_size,
            "test_input_size": (
                self.test_input_size
                if self.test_input_size is not None
                else self.input_size
            ),
            "fixed_input_size": self.fixed_input_size,
            "interpolation": self.interpolation,
            "crop_pct": self.crop_pct,
            "test_crop_pct": (
                self.test_crop_pct if self.test_crop_pct is not None else self.crop_pct
            ),
            "crop_mode": self.crop_mode,
            "license": self.license,
        }


def file_sha256(path: str | Path) -> str:
    """Compute the SHA-256 hex digest of a file, streaming it in chunks."""

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_verified_file(path: Path, spec: PretrainedWeightsSpec) -> dict[str, Tensor]:
    """Verify and decode one immutable byte snapshot from ``path``."""

    resolved = path.expanduser().resolve()
    try:
        payload = resolved.read_bytes()
    except OSError as exc:
        raise CheckpointDownloadError(
            f"Failed to read checkpoint at {resolved}: {exc}"
        ) from exc

    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != spec.sha256:
        raise ChecksumMismatchError(
            f"SHA-256 mismatch for {resolved} "
            f"({spec.repository!r}@{spec.revision!r}): expected {spec.sha256}, "
            f"got {actual_sha256}"
        )
    try:
        return load_safetensors(payload)
    except SafetensorError as exc:
        raise CheckpointFormatError(
            f"Verified checkpoint at {resolved} is not valid safetensors: {exc}"
        ) from exc


def _download_from_hub(
    spec: PretrainedWeightsSpec,
    *,
    cache_dir: str | Path | None,
    local_files_only: bool,
    force_download: bool,
) -> Path:
    try:
        return Path(
            hf_hub_download(
                repo_id=spec.repository,
                filename=spec.filename,
                revision=spec.revision,
                cache_dir=str(cache_dir) if cache_dir is not None else None,
                local_files_only=local_files_only,
                force_download=force_download,
            )
        )
    except Exception as exc:
        raise CheckpointDownloadError(
            f"Failed to obtain {spec.filename!r} from {spec.repository!r} at "
            f"revision {spec.revision!r} "
            f"(cache_dir={cache_dir!r}, local_files_only={local_files_only}, "
            f"force_download={force_download}): {exc}"
        ) from exc


def load_verified_state_dict(
    spec: PretrainedWeightsSpec,
    *,
    checkpoint_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    force_download: bool = False,
) -> dict[str, Tensor]:
    """Load the pinned checkpoint from a local path or Hugging Face Hub.

    Hashing and safetensors decoding operate on the same immutable bytes. A
    corrupt Hub cache is retried exactly once when network access is allowed;
    explicit local files are never modified or replaced.
    """

    if checkpoint_path is not None and (
        cache_dir is not None or local_files_only or force_download
    ):
        raise ValueError(
            "checkpoint_path cannot be combined with cache_dir, "
            "local_files_only, or force_download"
        )
    if local_files_only and force_download:
        raise ValueError("local_files_only and force_download are mutually exclusive")

    if checkpoint_path is not None:
        return _load_verified_file(Path(checkpoint_path), spec)

    path = _download_from_hub(
        spec,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        force_download=force_download,
    )
    try:
        return _load_verified_file(path, spec)
    except ChecksumMismatchError as initial_mismatch:
        if local_files_only or force_download:
            raise
        try:
            refreshed_path = _download_from_hub(
                spec,
                cache_dir=cache_dir,
                local_files_only=False,
                force_download=True,
            )
        except CheckpointDownloadError as refresh_error:
            raise CheckpointDownloadError(
                f"Cached checkpoint failed verification ({initial_mismatch}); "
                f"forced refresh also failed ({refresh_error})"
            ) from refresh_error
        return _load_verified_file(refreshed_path, spec)
