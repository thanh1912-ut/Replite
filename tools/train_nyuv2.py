#!/usr/bin/env python3
"""Audited RepLite segmentation+depth training on the NYUD-MT bundle.

The official ``gt_sets/val.txt`` split is treated as a final test set.  Model
selection and early stopping use a deterministic inner-validation subset made
only from ``gt_sets/train.txt``.  The official test images are decoded only
after ``best.pt`` has been checksum-verified and strict-loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tarfile
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from replite.data import (  # noqa: E402
    Nyuv2Augmentation,
    Nyuv2Dataset,
    Nyuv2Index,
    discover_nyuv2,
    nyuv2_collate,
    read_nyuv2_depth,
    read_nyuv2_segmentation,
    scan_nyuv2_label_ids,
)
from replite.multitask import (  # noqa: E402
    RepLiteConfig,
    TaskConfig,
    create_replite_model,
)
from replite.training import (  # noqa: E402
    BalancedBatchSampler,
    CheckpointManager,
    DepthMetrics,
    MultiTaskCriterion,
    MultiTaskMetrics,
    SegmentationMetrics,
    Trainer,
    TrainerConfig,
    TrainingLogger,
    WarmupCosineScheduler,
    YoloProgressReporter,
    create_adamw,
    load_training_checkpoint,
    inverse_sqrt_class_weights,
)


LEGACY_PROTOCOL_ID = "replite-nyuv2-segdepth-v1"
PROTOCOL_ID = "replite-nyuv2-segdepth-v2"
SUPPORTED_PROTOCOL_IDS = (LEGACY_PROTOCOL_ID, PROTOCOL_ID)
ACTIVE_TASKS = ("segmentation", "depth")
MODE_TASKS = {
    "seg-only": ("segmentation",),
    "depth-only": ("depth",),
    "multitask": ACTIVE_TASKS,
}
_MODE_ALIASES = {
    "seg-only": "seg-only",
    "segmentation-only": "seg-only",
    "segmentation_only": "seg-only",
    "depth-only": "depth-only",
    "depth_only": "depth-only",
    "multitask": "multitask",
    "multi-task": "multitask",
}
_CAMPAIGN_STAGES = ("pilot", "ablation", "main")
_DEPTH_BIN_EDGES = (0.1, 1.0, 3.0, 5.0, 10.0)
NYUV2_CLASS_NAMES = (
    "wall",
    "floor",
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "blinds",
    "desk",
    "shelves",
    "curtain",
    "dresser",
    "pillow",
    "mirror",
    "floor mat",
    "clothes",
    "ceiling",
    "books",
    "refrigerator",
    "television",
    "paper",
    "towel",
    "shower curtain",
    "box",
    "whiteboard",
    "person",
    "night stand",
    "toilet",
    "sink",
    "lamp",
    "bathtub",
    "bag",
    "otherstructure",
    "otherfurniture",
    "otherprop",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON artifact contains NaN or infinity")
        return value
    return str(value)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any] | Sequence[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(
                _jsonable(value),
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON: {path}") from exc


def _sha256_file(path: Path, *, progress: bool = False) -> str:
    digest = hashlib.sha256()
    total = path.stat().st_size
    completed = 0
    next_report = 256 * 1024**2
    started = time.monotonic()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
            completed += len(block)
            if progress and (completed >= next_report or completed == total):
                elapsed = max(time.monotonic() - started, 1e-6)
                print(
                    "[sha256] "
                    f"{completed / 1024**3:.2f}/{total / 1024**3:.2f} GiB | "
                    f"{completed / elapsed / 1024**2:.1f} MiB/s",
                    flush=True,
                )
                next_report = completed + 256 * 1024**2
    return digest.hexdigest()


def _plain_name(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value in {".", ".."}:
        raise ValueError(f"{name} must be a non-empty plain name")
    if "/" in value or "\\" in value:
        raise ValueError(f"{name} must be a plain name")
    return value


def _runtime_source_commit() -> str:
    """Return the commit containing the runner currently being executed."""

    try:
        value = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "NYUDv2 v2 must run from a Git checkout so source can be pinned"
        ) from exc
    if len(value) != 40 or any(
        character not in "0123456789abcdef" for character in value.casefold()
    ):
        raise RuntimeError(f"git returned an invalid source commit: {value!r}")
    return value.casefold()


def _campaign_mode(config: Mapping[str, Any]) -> str:
    model = config.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("campaign model must be a mapping")
    raw_tasks = model.get("active_tasks")
    if not isinstance(raw_tasks, list) or any(
        not isinstance(task, str) for task in raw_tasks
    ):
        raise ValueError("model.active_tasks must be a list of task names")
    tasks = tuple(raw_tasks)
    inferred = next(
        (mode for mode, expected in MODE_TASKS.items() if tasks == expected),
        None,
    )
    raw_mode = config.get("mode")
    if raw_mode is None:
        if inferred is None:
            raise ValueError(
                "model.active_tasks must select seg-only, depth-only, or multitask"
            )
        return inferred
    if not isinstance(raw_mode, str) or raw_mode not in _MODE_ALIASES:
        raise ValueError(
            "mode must be 'seg-only', 'depth-only', or 'multitask'"
        )
    mode = _MODE_ALIASES[raw_mode]
    if tasks != MODE_TASKS[mode]:
        raise ValueError(
            f"model.active_tasks {tasks!r} do not match mode {mode!r}"
        )
    return mode


def _active_tasks(config: Mapping[str, Any]) -> tuple[str, ...]:
    return MODE_TASKS[_campaign_mode(config)]


def _uses_v2_contract(config: Mapping[str, Any]) -> bool:
    """Return whether the campaign explicitly selects protocol v2.

    Protocol identity is authoritative.  A v2-labelled file must never fall
    back to legacy checkpoint selection or split semantics merely because an
    optional v2 field is missing; validation should reject that drift instead.
    """

    return config.get("protocol_id") == PROTOCOL_ID


def _campaign_stage(config: Mapping[str, Any]) -> str:
    train = config.get("train")
    if not isinstance(train, Mapping):
        raise ValueError("campaign train must be a mapping")
    value = config.get("stage", train.get("stage", "main"))
    if value not in _CAMPAIGN_STAGES:
        raise ValueError("stage must be 'pilot', 'ablation', or 'main'")
    return str(value)


def _expected_monitor(mode: str) -> tuple[str, str]:
    if mode == "seg-only":
        return "val/segmentation/miou", "max"
    if mode == "depth-only":
        return "val/depth/abs_rel", "min"
    return "val/selection/joint", "max"


def _selection_anchors(config: Mapping[str, Any]) -> dict[str, float] | None:
    train = config["train"]
    value = train.get("single_task_anchors")
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("train.single_task_anchors must be a mapping or null")
    required = ("segmentation_miou", "depth_abs_rel")
    if set(value) != set(required):
        raise ValueError(
            "single_task_anchors requires segmentation_miou and depth_abs_rel"
        )
    result: dict[str, float] = {}
    for name in required:
        raw = value[name]
        if (
            isinstance(raw, bool)
            or not isinstance(raw, (int, float))
            or not math.isfinite(float(raw))
            or float(raw) <= 0.0
        ):
            raise ValueError(f"single_task_anchors.{name} must be positive")
        result[name] = float(raw)
    if result["segmentation_miou"] > 1.0:
        raise ValueError("single_task_anchors.segmentation_miou must be <= 1")
    return result


def load_campaign(filename: str | os.PathLike[str]) -> tuple[Path, dict[str, Any]]:
    """Load and validate the locked NYUDv2 campaign configuration."""

    path = Path(filename).expanduser().resolve()
    value = _read_json(path)
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("unsupported campaign schema")
    protocol_id = value.get("protocol_id")
    if protocol_id not in SUPPORTED_PROTOCOL_IDS:
        raise ValueError(
            "protocol_id must be one of " + repr(SUPPORTED_PROTOCOL_IDS)
        )
    is_v2 = protocol_id == PROTOCOL_ID
    strict_v2 = _uses_v2_contract(value)
    _plain_name(value.get("run_id"), "run_id")
    for section in ("archive", "paths", "model", "data", "train"):
        if not isinstance(value.get(section), dict):
            raise ValueError(f"campaign {section} must be a mapping")

    if strict_v2:
        repository = value.get("source_repository")
        if not isinstance(repository, str) or not repository:
            raise ValueError("source_repository must be a non-empty string")
        source_commit = value.get("source_commit")
        if not isinstance(source_commit, str) or len(source_commit) != 40 or any(
            char not in "0123456789abcdef" for char in source_commit.casefold()
        ):
            raise ValueError("source_commit must be a 40-character Git commit")

    archive = value["archive"]
    if (
        isinstance(archive.get("expected_bytes"), bool)
        or not isinstance(archive.get("expected_bytes"), int)
        or archive["expected_bytes"] <= 0
    ):
        raise ValueError("archive.expected_bytes must be a positive integer")
    digest = archive.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(
        char not in "0123456789abcdef" for char in digest.casefold()
    ):
        raise ValueError("archive.sha256 must be a lowercase SHA-256")
    for name in ("expected_train_samples", "expected_test_samples"):
        if (
            isinstance(archive.get(name), bool)
            or not isinstance(archive.get(name), int)
            or archive[name] <= 0
        ):
            raise ValueError(f"archive.{name} must be a positive integer")

    for name in ("local_dataset_root", "local_work_root", "drive_run_root"):
        if not isinstance(value["paths"].get(name), str) or not value["paths"][name]:
            raise ValueError(f"paths.{name} must be a non-empty path")

    model = value["model"]
    mode = _campaign_mode(value)
    active_tasks = MODE_TASKS[mode]
    stage = _campaign_stage(value) if strict_v2 else "main"
    if not is_v2 and active_tasks != ACTIVE_TASKS:
        raise ValueError(
            "legacy model.active_tasks must be ['segmentation', 'depth']"
        )
    if active_tasks == ACTIVE_TASKS:
        if model.get("dense_fusion_direction") != "seg_to_depth":
            raise ValueError("NYUDv2 requires dense_fusion_direction='seg_to_depth'")
        if model.get("dense_fusion_detach_source") is not True:
            raise ValueError("NYUDv2 requires dense_fusion_detach_source=true")
    if strict_v2:
        auxiliary = model.get("segmentation_auxiliary", False)
        if not isinstance(auxiliary, bool):
            raise ValueError("model.segmentation_auxiliary must be a boolean")
        if "segmentation" not in active_tasks and auxiliary:
            raise ValueError(
                "model.segmentation_auxiliary requires segmentation mode"
            )
        if stage == "main":
            if model.get("dense_decoder") != "dense_v2_s":
                raise ValueError("v2 main campaigns require dense_decoder='dense_v2_s'")
            if "segmentation" in active_tasks and auxiliary is not True:
                raise ValueError(
                    "v2 main segmentation campaigns require segmentation_auxiliary=true"
                )

    data = value["data"]
    if data.get("image_size") != [288, 384]:
        raise ValueError("NYUDv2 input is locked to 288x384 (height x width)")
    if data.get("num_classes") != len(NYUV2_CLASS_NAMES):
        raise ValueError("NYUDv2 segmentation must use the standard 40 classes")
    mapping = data.get("raw_label_mapping")
    expected_mapping = {str(index): index - 1 for index in range(1, 41)}
    if mapping != expected_mapping or data.get("source_ignore_labels") != [0]:
        raise ValueError("NYUDv2 labels must map raw 1..40 to 0..39 and ignore raw 0")
    if float(data.get("depth_unit_scale", 0.0)) != 1.0:
        raise ValueError("the NYUD-MT NPY bundle requires depth_unit_scale=1.0")
    fraction = data.get("inner_validation_fraction")
    if (
        isinstance(fraction, bool)
        or not isinstance(fraction, (int, float))
        or not 0.0 < float(fraction) < 0.5
    ):
        raise ValueError("data.inner_validation_fraction must be in (0,0.5)")
    for name in ("batch_size", "num_workers", "prefetch_factor", "split_seed"):
        item = data.get(name)
        lower = 0 if name == "num_workers" else 1
        if isinstance(item, bool) or not isinstance(item, int) or item < lower:
            raise ValueError(f"data.{name} is invalid")
    augmentation_config = data.get("augmentation")
    if not isinstance(augmentation_config, Mapping):
        raise ValueError("data.augmentation must be a mapping")
    try:
        augmentation = Nyuv2Augmentation(**dict(augmentation_config))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid data.augmentation: {exc}") from exc
    if any(
        class_id >= int(data["num_classes"])
        for class_id in augmentation.rare_classes
    ):
        raise ValueError("data.augmentation.rare_classes exceeds num_classes")
    if strict_v2 and stage == "main" and "segmentation" in active_tasks:
        if (
            augmentation.class_aware_crop_probability <= 0.0
            or not augmentation.rare_classes
        ):
            raise ValueError(
                "v2 main segmentation requires class-aware crop and rare_classes"
            )

    train = value["train"]
    if strict_v2:
        monitor, monitor_mode = _expected_monitor(mode)
        if train.get("monitor") != monitor or train.get("monitor_mode") != monitor_mode:
            raise ValueError(
                f"{mode} checkpoint selection requires {monitor!r} ({monitor_mode})"
            )
        patience = train.get("early_stopping_patience")
        if patience is not None and (
            isinstance(patience, bool)
            or not isinstance(patience, int)
            or patience <= 0
        ):
            raise ValueError(
                "early_stopping_patience must be a positive integer or null"
            )
        if stage in {"pilot", "ablation"} and patience is not None:
            raise ValueError(
                f"early stopping must be disabled for {stage} campaigns"
            )
        if stage == "main" and patience != 10:
            raise ValueError(
                "v2 main early_stopping_patience must be exactly 10"
            )
        min_delta = train.get("early_stopping_min_delta")
        if (
            isinstance(min_delta, bool)
            or not isinstance(min_delta, (int, float))
            or not math.isfinite(float(min_delta))
            or float(min_delta) < 0.0
        ):
            raise ValueError(
                "early_stopping_min_delta must be finite and non-negative"
            )
        task_weights = train.get("task_weights")
        if not isinstance(task_weights, Mapping):
            raise ValueError("train.task_weights must be a mapping")
        unknown_weights = set(task_weights) - set(ACTIVE_TASKS)
        if unknown_weights:
            raise ValueError(
                "unknown NYUDv2 task weights: "
                + ", ".join(sorted(unknown_weights))
            )
        for task in active_tasks:
            weight = task_weights.get(task)
            if (
                isinstance(weight, bool)
                or not isinstance(weight, (int, float))
                or not math.isfinite(float(weight))
                or float(weight) <= 0.0
            ):
                raise ValueError(f"active task weight {task!r} must be positive")
        for task in set(ACTIVE_TASKS) - set(active_tasks):
            if task in task_weights and float(task_weights[task]) != 0.0:
                raise ValueError(f"inactive task weight {task!r} must be zero or absent")
        if stage == "main" and any(
            float(task_weights[task]) != 1.0 for task in active_tasks
        ):
            raise ValueError("v2 main active task weights must all equal 1.0")
        loss = train.get("loss")
        if not isinstance(loss, Mapping):
            raise ValueError("train.loss must be a mapping")
        if stage == "main" and "segmentation" in active_tasks:
            if loss.get("segmentation_normalize_weighted_loss") is not True:
                raise ValueError(
                    "v2 main segmentation requires normalized class weights"
                )
            if float(loss.get("segmentation_lovasz_weight", 0.0)) <= 0.0:
                raise ValueError("v2 main segmentation requires Lovasz loss")
            if float(loss.get("segmentation_auxiliary_weight", 0.0)) <= 0.0:
                raise ValueError(
                    "v2 main segmentation requires auxiliary supervision"
                )
        if stage == "main" and "depth" in active_tasks:
            if loss.get("depth_loss_type") != "per_image_silog_log_l1_gradient":
                raise ValueError(
                    "v2 main depth requires per-image SiLog/log-L1/gradient loss"
                )
            if float(loss.get("depth_gradient_weight", 0.0)) <= 0.0:
                raise ValueError("v2 main depth requires positive gradient loss")
        _selection_anchors(value)
    else:
        if train.get("monitor") != "val/total" or train.get("monitor_mode") != "min":
            raise ValueError("legacy checkpoint selection must minimize val/total")
        if train.get("early_stopping_patience") != 10:
            raise ValueError("legacy early stopping patience is locked to 10 validations")
        if train.get("task_weights") != {"segmentation": 1.0, "depth": 0.25}:
            raise ValueError("legacy task weights must be segmentation=1.0 and depth=0.25")
    if (
        isinstance(train.get("epochs"), bool)
        or not isinstance(train.get("epochs"), int)
        or train["epochs"] < 1
    ):
        raise ValueError("train.epochs must be positive")
    return path, value


def _extraction_marker(root: Path) -> Path:
    return root / ".replite_nyuv2_extract.json"


def _validate_extracted(
    root: Path,
    config: Mapping[str, Any],
) -> tuple[Nyuv2Index, dict[str, Any]]:
    marker_path = _extraction_marker(root)
    if not marker_path.is_file():
        raise ValueError(f"extracted dataset marker is missing: {marker_path}")
    marker = _read_json(marker_path)
    archive = config["archive"]
    required = {
        "schema_version": 1,
        "archive_bytes": archive["expected_bytes"],
        "archive_sha256": archive["sha256"],
        "official_train_samples": archive["expected_train_samples"],
        "official_test_samples": archive["expected_test_samples"],
    }
    if not isinstance(marker, dict) or any(marker.get(key) != item for key, item in required.items()):
        raise ValueError("extracted NYUDv2 marker does not match the locked archive")
    if marker.get("protocol_id") not in SUPPORTED_PROTOCOL_IDS:
        raise ValueError("extracted NYUDv2 marker uses an unsupported protocol")
    index = discover_nyuv2(root)
    if (
        len(index.train) != archive["expected_train_samples"]
        or len(index.test) != archive["expected_test_samples"]
    ):
        raise ValueError("extracted NYUDv2 split counts changed")
    return index, marker


def _safe_member_path(member_name: str) -> PurePosixPath:
    path = PurePosixPath(member_name)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe tar member path: {member_name!r}")
    return path


def _stream_extract_tar(archive: Path, staging: Path) -> dict[str, int]:
    files = 0
    directories = 0
    extracted_bytes = 0
    started = time.monotonic()
    with tarfile.open(archive, mode="r|*") as bundle:
        for member in bundle:
            relative = _safe_member_path(member.name)
            target = staging.joinpath(*relative.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                directories += 1
                continue
            if not member.isreg():
                raise ValueError(
                    f"NYUDv2 archive contains a forbidden non-regular member: {member.name}"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            source = bundle.extractfile(member)
            if source is None:
                raise ValueError(f"cannot read tar member: {member.name}")
            try:
                with target.open("xb") as destination:
                    copied = shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
                    destination.flush()
                # copyfileobj returns None; member.size is the audited byte count.
                del copied
            finally:
                source.close()
            if target.stat().st_size != member.size:
                raise IOError(f"short extraction for tar member: {member.name}")
            files += 1
            extracted_bytes += member.size
            if files == 1 or files % 250 == 0:
                elapsed = max(time.monotonic() - started, 1e-6)
                print(
                    f"[extract] files={files:,} | {extracted_bytes / 1024**3:.2f} GiB | "
                    f"{extracted_bytes / elapsed / 1024**2:.1f} MiB/s",
                    flush=True,
                )
    return {
        "files": files,
        "directories": directories,
        "extracted_bytes": extracted_bytes,
    }


def extract_campaign(filename: str | os.PathLike[str]) -> dict[str, Any]:
    """Verify the Drive archive and extract it directly onto Colab SSD."""

    _, config = load_campaign(filename)
    archive_config = config["archive"]
    archive = Path(archive_config["path"]).expanduser().resolve()
    output = Path(config["paths"]["local_dataset_root"]).expanduser().resolve()
    if output.exists():
        _, marker = _validate_extracted(output, config)
        print("[extract] verified local dataset already exists:", output)
        return marker
    if not archive.is_file():
        raise FileNotFoundError(f"NYUDv2 archive is missing: {archive}")
    actual_bytes = archive.stat().st_size
    if actual_bytes != archive_config["expected_bytes"]:
        raise ValueError(
            f"archive size mismatch: expected {archive_config['expected_bytes']}, got {actual_bytes}"
        )
    print("[extract] verifying archive SHA-256 directly on Drive", flush=True)
    actual_sha = _sha256_file(archive, progress=True)
    if actual_sha != archive_config["sha256"]:
        raise ValueError(
            f"archive SHA-256 mismatch: expected {archive_config['sha256']}, got {actual_sha}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(output.parent).free
    reserve = 2 * 1024**3
    conservative_need = math.ceil(actual_bytes * 3.0) + reserve
    if free < conservative_need:
        raise OSError(
            "not enough Colab SSD for safe NYUDv2 extraction: "
            f"free={free / 1024**3:.1f} GiB, required~={conservative_need / 1024**3:.1f} GiB"
        )
    staging = output.parent / f".{output.name}.extracting.{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)
    published = False
    try:
        stats = _stream_extract_tar(archive, staging)
        index = discover_nyuv2(staging)
        if (
            len(index.train) != archive_config["expected_train_samples"]
            or len(index.test) != archive_config["expected_test_samples"]
        ):
            raise ValueError(
                "archive split counts mismatch: "
                f"train={len(index.train)}, test={len(index.test)}"
            )
        source_root = index.root
        if source_root == staging:
            os.replace(staging, output)
        else:
            os.replace(source_root, output)
            shutil.rmtree(staging)
        published = True
        marker = {
            "schema_version": 1,
            "protocol_id": config["protocol_id"],
            "archive_path": str(archive),
            "archive_bytes": actual_bytes,
            "archive_sha256": actual_sha,
            "official_train_samples": len(index.train),
            "official_test_samples": len(index.test),
            **stats,
        }
        _atomic_json(_extraction_marker(output), marker)
        _validate_extracted(output, config)
        print("[extract] READY:", output, flush=True)
        return marker
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)


@dataclass(frozen=True)
class InnerSplit:
    fit_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    manifest_path: Path
    manifest_sha256: str


def _split_features(
    index: Nyuv2Index,
    data: Mapping[str, Any],
) -> tuple[frozenset[str], ...]:
    """Read official-train labels only and build multi-label split strata."""

    mapping = {int(source): int(target) for source, target in data["raw_label_mapping"].items()}
    ignored = {int(value) for value in data["source_ignore_labels"]}
    depth_min = float(data["depth_min_metres"])
    depth_max = float(data["depth_max_metres"])
    scale = float(data["depth_unit_scale"])
    rows: list[frozenset[str]] = []
    for sample in index.train:
        segmentation = read_nyuv2_segmentation(sample.segmentation_path)
        observed = {int(value) for value in np.unique(segmentation)}
        unknown = observed - set(mapping) - ignored
        if unknown:
            raise ValueError(
                f"split inventory found unmapped labels {sorted(unknown)} in {sample.key}"
            )
        features = {
            f"class:{mapping[raw]}" for raw in observed if raw in mapping
        }
        depth = read_nyuv2_depth(sample.depth_path, unit_scale=scale)
        valid = np.isfinite(depth) & (depth >= depth_min) & (depth <= depth_max)
        for lower, upper in zip(_DEPTH_BIN_EDGES[:-1], _DEPTH_BIN_EDGES[1:]):
            upper_mask = depth <= upper if upper == _DEPTH_BIN_EDGES[-1] else depth < upper
            if bool(np.any(valid & (depth >= lower) & upper_mask)):
                features.add(f"depth:{lower:g}-{upper:g}m")
        rows.append(frozenset(features))
    return tuple(rows)


def _stratified_validation_positions(
    feature_sets: Sequence[frozenset[str]],
    *,
    count: int,
    seed: int,
    keys: Sequence[str],
) -> tuple[frozenset[int], dict[str, dict[str, int]]]:
    """Greedily preserve class/depth-bin presence without emptying fit strata."""

    if len(feature_sets) != len(keys):
        raise ValueError("split feature/key counts differ")
    support: dict[str, int] = {}
    for features in feature_sets:
        for feature in features:
            support[feature] = support.get(feature, 0) + 1
    desired = {
        feature: (
            0
            if total < 2
            else min(total - 1, max(1, round(count * total / len(feature_sets))))
        )
        for feature, total in support.items()
    }
    current = {feature: 0 for feature in support}
    stable_order = sorted(
        range(len(feature_sets)),
        key=lambda position: hashlib.sha256(
            f"{seed}:{keys[position]}".encode("utf-8")
        ).hexdigest(),
    )
    selected: set[int] = set()
    while len(selected) < count:
        candidates = [position for position in stable_order if position not in selected]
        if not candidates:
            raise RuntimeError("cannot complete the inner-validation split")

        def rank(position: int) -> tuple[int, float, float]:
            features = feature_sets[position]
            preserves_fit = all(
                support[feature] - current[feature] > 1 for feature in features
            )
            need = sum(
                max(desired[feature] - current[feature], 0)
                / max(desired[feature], 1)
                for feature in features
                if desired[feature] > 0
            )
            overshoot = sum(
                max(current[feature] + 1 - desired[feature], 0)
                / support[feature]
                for feature in features
                if desired[feature] > 0
            )
            return int(preserves_fit), need, -overshoot

        chosen = max(candidates, key=rank)
        selected.add(chosen)
        for feature in feature_sets[chosen]:
            current[feature] += 1

    summary = {
        feature: {
            "official_train_images": support[feature],
            "fit_images": support[feature] - current[feature],
            "inner_validation_images": current[feature],
        }
        for feature in sorted(support)
    }
    return frozenset(selected), summary


def _create_or_load_inner_split(
    index: Nyuv2Index,
    *,
    fraction: float,
    seed: int,
    manifest_path: Path,
    protocol_id: str,
    data: Mapping[str, Any],
) -> InnerSplit:
    if len(index.train) < 2:
        raise ValueError("official training split is too small for inner validation")
    count = min(len(index.train) - 1, max(1, round(len(index.train) * fraction)))
    strata_summary: dict[str, dict[str, int]] | None = None
    if protocol_id == PROTOCOL_ID:
        feature_sets = _split_features(index, data)
        validation, strata_summary = _stratified_validation_positions(
            feature_sets,
            count=count,
            seed=seed,
            keys=[sample.key for sample in index.train],
        )
        strategy = "greedy_multilabel_class_and_depth_presence_v1"
    else:
        ranked = sorted(
            range(len(index.train)),
            key=lambda position: hashlib.sha256(
                f"{seed}:{index.train[position].key}".encode("utf-8")
            ).hexdigest(),
        )
        validation = frozenset(ranked[:count])
        strategy = "deterministic_key_hash_v1"
    payload = {
        "schema_version": 1,
        "protocol_id": protocol_id,
        "source": "official gt_sets/train.txt only",
        "seed": seed,
        "fraction": fraction,
        "fit_keys": [
            sample.key
            for position, sample in enumerate(index.train)
            if position not in validation
        ],
        "inner_validation_keys": [
            sample.key
            for position, sample in enumerate(index.train)
            if position in validation
        ],
        "official_test_keys_sha256": _canonical_sha256(
            [sample.key for sample in index.test]
        ),
        "official_test_samples": len(index.test),
        "official_test_used": False,
    }
    if protocol_id == PROTOCOL_ID:
        payload["split_strategy"] = strategy
        payload["strata"] = strata_summary
    if manifest_path.exists():
        if _read_json(manifest_path) != payload:
            raise FileExistsError(
                f"existing inner split differs; choose a new run_id: {manifest_path}"
            )
    else:
        _atomic_json(manifest_path, payload)
    return InnerSplit(
        fit_indices=tuple(
            position for position in range(len(index.train)) if position not in validation
        ),
        validation_indices=tuple(
            position for position in range(len(index.train)) if position in validation
        ),
        manifest_path=manifest_path,
        manifest_sha256=_canonical_sha256(payload),
    )


def _compute_fit_statistics(
    index: Nyuv2Index,
    fit_indices: Sequence[int],
    data: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute class/depth statistics from fit samples only.

    The official held-out split and inner-validation samples are never read by
    this function, so class weighting cannot leak benchmark information.
    """

    num_classes = int(data["num_classes"])
    mapping = {int(source): int(target) for source, target in data["raw_label_mapping"].items()}
    ignored = {int(value) for value in data["source_ignore_labels"]}
    class_pixels = np.zeros(num_classes, dtype=np.int64)
    images_per_class = np.zeros(num_classes, dtype=np.int64)
    depth_bins = np.zeros(len(_DEPTH_BIN_EDGES) - 1, dtype=np.int64)
    depth_pixels = 0
    depth_sum = 0.0
    depth_sq_sum = 0.0
    depth_min = float(data["depth_min_metres"])
    depth_max = float(data["depth_max_metres"])
    scale = float(data["depth_unit_scale"])
    for position in fit_indices:
        sample = index.train[int(position)]
        raw_seg = read_nyuv2_segmentation(sample.segmentation_path)
        mapped = np.full(raw_seg.shape, -1, dtype=np.int64)
        for source, target in mapping.items():
            mapped[raw_seg == source] = target
        valid_seg = mapped >= 0
        class_pixels += np.bincount(mapped[valid_seg], minlength=num_classes)
        images_per_class += np.bincount(
            np.unique(mapped[valid_seg]), minlength=num_classes
        )
        depth = read_nyuv2_depth(sample.depth_path, unit_scale=scale)
        valid_depth = np.isfinite(depth) & (depth >= depth_min) & (depth <= depth_max)
        values = depth[valid_depth]
        depth_pixels += int(values.size)
        if values.size:
            depth_sum += float(values.sum())
            depth_sq_sum += float(np.square(values).sum())
            for bin_index, (lower, upper) in enumerate(
                zip(_DEPTH_BIN_EDGES[:-1], _DEPTH_BIN_EDGES[1:])
            ):
                upper_mask = values <= upper if upper == _DEPTH_BIN_EDGES[-1] else values < upper
                depth_bins[bin_index] += int(np.count_nonzero((values >= lower) & upper_mask))
    weights = inverse_sqrt_class_weights(class_pixels.tolist())
    mean = depth_sum / max(depth_pixels, 1)
    variance = max(depth_sq_sum / max(depth_pixels, 1) - mean * mean, 0.0)
    return {
        "schema_version": 1,
        "source": "fit_indices_only",
        "fit_samples": len(fit_indices),
        "segmentation": {
            "pixel_counts": class_pixels.tolist(),
            "images_per_class": images_per_class.tolist(),
            "class_weights": weights.tolist(),
        },
        "depth": {
            "valid_pixels": depth_pixels,
            "mean_metres": mean,
            "std_metres": math.sqrt(variance),
            "bin_edges_metres": list(_DEPTH_BIN_EDGES),
            "bin_pixel_counts": depth_bins.tolist(),
        },
    }


class _EpochSubset(Dataset[Any]):
    """Subset that forwards the deterministic epoch hook to its dataset."""

    def __init__(self, dataset: Dataset[Any], indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = tuple(int(index) for index in indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Any:
        return self.dataset[self.indices[index]]

    def set_epoch(self, epoch: int) -> None:
        method = getattr(self.dataset, "set_epoch", None)
        if callable(method):
            method(epoch)


@dataclass(frozen=True)
class Prepared:
    config_path: Path
    config: dict[str, Any]
    index: Nyuv2Index
    split: InnerSplit
    local_work: Path
    local_run: Path
    drive_run: Path
    context: dict[str, Any]


def _ensure_locked_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        if _read_json(path) != _jsonable(payload):
            raise FileExistsError(f"locked artifact differs; use a new run_id: {path}")
    else:
        _atomic_json(path, payload)


def prepare(filename: str | os.PathLike[str]) -> Prepared:
    config_path, config = load_campaign(filename)
    runtime_source_commit = _runtime_source_commit()
    if (
        _uses_v2_contract(config)
        and str(config["source_commit"]).casefold() != runtime_source_commit
    ):
        raise ValueError(
            "runtime source commit does not match the v2 campaign source_commit: "
            f"{runtime_source_commit} != {config['source_commit']}"
        )
    dataset_root = Path(config["paths"]["local_dataset_root"]).expanduser().resolve()
    index, marker = _validate_extracted(dataset_root, config)
    raw_ids = scan_nyuv2_label_ids(index.train)
    expected_ids = tuple(config["data"].get("expected_raw_label_ids", range(41)))
    if raw_ids != expected_ids:
        raise ValueError(
            f"official-train raw segmentation IDs changed: {raw_ids} != {expected_ids}"
        )
    local_work = Path(config["paths"]["local_work_root"]).expanduser().resolve()
    local_run = local_work / "runs" / config["run_id"]
    drive_run = Path(config["paths"]["drive_run_root"]).expanduser().resolve()
    local_run.mkdir(parents=True, exist_ok=True)
    drive_run.mkdir(parents=True, exist_ok=True)
    split = _create_or_load_inner_split(
        index,
        fraction=float(config["data"]["inner_validation_fraction"]),
        seed=int(config["data"]["split_seed"]),
        manifest_path=drive_run / "inner_split_manifest.json",
        protocol_id=PROTOCOL_ID if _uses_v2_contract(config) else LEGACY_PROTOCOL_ID,
        data=config["data"],
    )
    fit_statistics = _compute_fit_statistics(index, split.fit_indices, config["data"])
    _ensure_locked_json(drive_run / "fit_statistics.json", fit_statistics)
    config = dict(config)
    config["fit_statistics"] = fit_statistics
    _ensure_locked_json(drive_run / "resolved_config.json", config)
    context = {
        "protocol_id": PROTOCOL_ID if _uses_v2_contract(config) else LEGACY_PROTOCOL_ID,
        "run_id": config["run_id"],
        "runtime_source_commit": runtime_source_commit,
        "configured_source_commit": config.get("source_commit"),
        "archive_sha256": marker["archive_sha256"],
        "config_sha256": _canonical_sha256(config),
        "split_sha256": split.manifest_sha256,
        "active_tasks": list(_active_tasks(config)),
        "fit_statistics_sha256": _canonical_sha256(fit_statistics),
        "official_test_used": False,
    }
    _ensure_locked_json(drive_run / "context.json", context)
    audit = {
        "schema_version": 1,
        "official_train_samples": len(index.train),
        "fit_samples": len(split.fit_indices),
        "inner_validation_samples": len(split.validation_indices),
        "official_test_samples_reserved": len(index.test),
        "official_test_used": False,
        "raw_segmentation_ids_official_train": list(raw_ids),
        "label_policy": "raw 0 ignored; raw 1..40 mapped to train IDs 0..39",
        "depth_unit_scale_to_metres": config["data"]["depth_unit_scale"],
        "input_contract": ["B", 3, 288, 384],
    }
    _ensure_locked_json(drive_run / "data_audit.json", audit)
    return Prepared(
        config_path=config_path,
        config=config,
        index=index,
        split=split,
        local_work=local_work,
        local_run=local_run,
        drive_run=drive_run,
        context=context,
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def _model_config(prepared: Prepared) -> RepLiteConfig:
    source = prepared.config["model"]
    data = prepared.config["data"]
    mode = _campaign_mode(prepared.config)
    active = set(_active_tasks(prepared.config))
    return RepLiteConfig(
        tasks=TaskConfig(
            detection_classes=None,
            segmentation_classes=(int(data["num_classes"]) if "segmentation" in active else None),
            depth="depth" in active,
            gated_dense_fusion=mode == "multitask",
            dense_fusion_direction=str(source.get("dense_fusion_direction", "seg_to_depth")),
            dense_fusion_detach_source=bool(source.get("dense_fusion_detach_source", True)),
        ),
        backbone_name=str(source["backbone_name"]),
        pretrained=bool(source["pretrained_in1k"]),
        recurrence_steps=int(source["recurrence_steps"]),
        recurrent_c4_channels=int(source["recurrent_c4_channels"]),
        recurrent_c5_channels=int(source["recurrent_c5_channels"]),
        neck_channels=int(source["neck_channels"]),
        dense_channels=int(source["dense_channels"]),
        task_adapter_channels=int(source["task_adapter_channels"]),
        detection_head_channels=int(source.get("detection_head_channels", 48)),
        detection_head_blocks=int(source.get("detection_head_blocks", 2)),
        detection_reg_max=int(source.get("detection_reg_max", 0)),
        dense_decoder=str(source.get("dense_decoder", "legacy")),
        segmentation_auxiliary=(
            "segmentation" in active
            and bool(source.get("segmentation_auxiliary", False))
        ),
        use_sppf=bool(source["use_sppf"]),
    )


def _create_model(prepared: Prepared) -> nn.Module:
    config = _model_config(prepared)
    weight_options = (
        {"cache_dir": str(prepared.local_work / "pretrained_cache")}
        if config.pretrained
        else {}
    )
    return create_replite_model(config, **weight_options)


def _create_criterion(prepared: Prepared) -> MultiTaskCriterion:
    data = prepared.config["data"]
    train = prepared.config["train"]
    fit_stats = prepared.config.get("fit_statistics", {})
    seg_stats = fit_stats.get("segmentation", {}) if isinstance(fit_stats, Mapping) else {}
    class_weights = seg_stats.get("class_weights") if isinstance(seg_stats, Mapping) else None
    loss_cfg = train.get("loss", {})
    if not isinstance(loss_cfg, Mapping):
        raise ValueError("train.loss must be a mapping")
    return MultiTaskCriterion(
        task_weights=train["task_weights"],
        segmentation_ignore_index=int(data["ignore_index"]),
        segmentation_class_weights=class_weights,
        segmentation_normalize_weighted_loss=bool(loss_cfg.get("segmentation_normalize_weighted_loss", True)),
        segmentation_lovasz_weight=float(loss_cfg.get("segmentation_lovasz_weight", 0.0)),
        segmentation_auxiliary_weight=float(loss_cfg.get("segmentation_auxiliary_weight", 0.0)),
        depth_loss_type=str(loss_cfg.get("depth_loss_type", "per_image_silog_log_l1_gradient")),
        depth_min=float(data["depth_min_metres"]),
        depth_max=float(data["depth_max_metres"]),
        depth_log_l1_weight=float(loss_cfg.get("depth_log_l1_weight", 1.0)),
        depth_silog_weight=float(loss_cfg.get("depth_silog_weight", 1.0)),
        depth_gradient_weight=float(loss_cfg.get("depth_gradient_weight", 0.25)),
        depth_silog_lambda=float(loss_cfg.get("depth_silog_lambda", 0.5)),
    )


def _dataset_kwargs(prepared: Prepared) -> dict[str, Any]:
    data = prepared.config["data"]
    return {
        "num_classes": int(data["num_classes"]),
        "label_mapping": {
            int(source): int(target)
            for source, target in data["raw_label_mapping"].items()
        },
        "source_ignore_labels": tuple(data["source_ignore_labels"]),
        "depth_unit_scale": float(data["depth_unit_scale"]),
        "image_size": tuple(data["image_size"]),
        "ignore_index": int(data["ignore_index"]),
        "depth_min": float(data["depth_min_metres"]),
        "depth_max": float(data["depth_max_metres"]),
        "seed": int(data["split_seed"]),
        "normalize": True,
        "index": prepared.index,
    }


def _loader_options(data: Mapping[str, Any]) -> dict[str, Any]:
    workers = int(data["num_workers"])
    result: dict[str, Any] = {
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": False,
        "collate_fn": nyuv2_collate,
    }
    if workers:
        result["prefetch_factor"] = int(data["prefetch_factor"])
    return result


def create_train_loaders(prepared: Prepared) -> tuple[DataLoader[Any], DataLoader[Any]]:
    data = prepared.config["data"]
    root = prepared.index.root
    augmentation = Nyuv2Augmentation(**data["augmentation"])
    fit_full = Nyuv2Dataset(
        root,
        split="train",
        augmentation=augmentation,
        **_dataset_kwargs(prepared),
    )
    validation_full = Nyuv2Dataset(
        root,
        split="train",
        augmentation=None,
        **_dataset_kwargs(prepared),
    )
    fit = _EpochSubset(fit_full, prepared.split.fit_indices)
    validation = _EpochSubset(validation_full, prepared.split.validation_indices)
    sampler = BalancedBatchSampler(
        fit,
        int(data["batch_size"]),
        shuffle=True,
        seed=int(data["split_seed"]),
    )
    options = _loader_options(data)
    return (
        DataLoader(fit, batch_sampler=sampler, **options),
        DataLoader(
            validation,
            batch_size=int(data["batch_size"]),
            shuffle=False,
            drop_last=False,
            **options,
        ),
    )


def _trainer_config(prepared: Prepared) -> TrainerConfig:
    train = prepared.config["train"]
    patience = train.get("early_stopping_patience")
    return TrainerConfig(
        epochs=int(train["epochs"]),
        grad_accum_steps=int(train["grad_accum_steps"]),
        amp=bool(train["amp"]),
        amp_dtype=str(train["amp_dtype"]),
        grad_clip_norm=float(train["grad_clip_norm"]),
        log_every_n_steps=int(train["progress_every_n_steps"]),
        validate_every_n_epochs=1,
        checkpoint_every_n_epochs=1,
        monitor=str(train["monitor"]),
        monitor_mode=str(train["monitor_mode"]),
        early_stopping_patience=(None if patience is None else int(patience)),
        early_stopping_min_delta=float(train["early_stopping_min_delta"]),
    )


def _create_metrics(prepared: Prepared) -> MultiTaskMetrics:
    data = prepared.config["data"]
    active = set(_active_tasks(prepared.config))
    return MultiTaskMetrics(
        segmentation=SegmentationMetrics(
            int(data["num_classes"]),
            ignore_index=int(data["ignore_index"]),
        ) if "segmentation" in active else None,
        depth=DepthMetrics(
            min_depth=float(data["depth_min_metres"]),
            max_depth=float(data["depth_max_metres"]),
        ) if "depth" in active else None,
        selection_anchors=_selection_anchors(prepared.config),
    )


def _optimizer_scheduler(
    prepared: Prepared,
    model: nn.Module,
    batches_per_epoch: int,
) -> tuple[torch.optim.Optimizer, WarmupCosineScheduler, int, int]:
    train = prepared.config["train"]
    trainer_config = _trainer_config(prepared)
    optimizer = create_adamw(
        model,
        lr=float(train["base_lr"]),
        weight_decay=float(train["weight_decay"]),
        backbone_lr_multiplier=float(train["backbone_lr_multiplier"]),
    )
    updates_per_epoch = math.ceil(
        batches_per_epoch / trainer_config.grad_accum_steps
    )
    total_steps = updates_per_epoch * trainer_config.epochs
    warmup_steps = min(
        total_steps,
        round(total_steps * float(train["warmup_fraction"])),
    )
    scheduler = WarmupCosineScheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=float(train["min_lr_ratio"]),
    )
    return optimizer, scheduler, total_steps, warmup_steps


def _parameter_summary(model: nn.Module) -> dict[str, Any]:
    rows = []
    for name, module in model.named_children():
        total = sum(parameter.numel() for parameter in module.parameters())
        rows.append({"module": name, "parameters": total})
    total = sum(parameter.numel() for parameter in model.parameters())
    return {"total": total, "fp32_mib": total * 4 / 1024**2, "modules": rows}


def inspect_campaign(filename: str | os.PathLike[str]) -> dict[str, Any]:
    prepared = prepare(filename)
    _seed_everything(int(prepared.config["train"]["seed"]))
    train_loader, val_loader = create_train_loaders(prepared)
    model = _create_model(prepared)
    criterion = _create_criterion(prepared)
    optimizer, _, total_steps, warmup_steps = _optimizer_scheduler(
        prepared, model, len(train_loader)
    )
    parameters = _parameter_summary(model)
    batch_sizes = list(train_loader.batch_sampler.batch_sizes)
    report = {
        "schema_version": 1,
        "protocol_id": prepared.context["protocol_id"],
        "mode": _campaign_mode(prepared.config),
        "stage": _campaign_stage(prepared.config),
        "context": prepared.context,
        "data": {
            "official_train": len(prepared.index.train),
            "fit": len(prepared.split.fit_indices),
            "inner_validation": len(prepared.split.validation_indices),
            "official_test_reserved": len(prepared.index.test),
            "official_test_used": False,
            "input_shape": ["B", 3, 288, 384],
            "batches_per_epoch": len(train_loader),
            "validation_batches": len(val_loader),
            "balanced_batch_min": min(batch_sizes),
            "balanced_batch_max": max(batch_sizes),
            "light_augmentation": prepared.config["data"]["augmentation"],
            "fit_statistics": prepared.config.get("fit_statistics"),
        },
        "model": model.config.as_dict(),
        "pretrained": model.backbone.weights_provenance,
        "parameters": parameters,
        "criterion": criterion.loss_metadata,
        "optimizer_groups": [
            {
                "name": group.get("name", str(position)),
                "parameters": sum(parameter.numel() for parameter in group["params"]),
                "lr": float(group["lr"]),
                "weight_decay": float(group["weight_decay"]),
            }
            for position, group in enumerate(optimizer.param_groups)
        ],
        "schedule": {
            "epochs_max": prepared.config["train"]["epochs"],
            "monitor": _trainer_config(prepared).monitor,
            "early_stopping_patience": _trainer_config(prepared).early_stopping_patience,
            "optimizer_updates_total": total_steps,
            "warmup_steps": warmup_steps,
        },
    }
    print("\n========== DATA / BATCH PLAN ==========")
    print(
        f"official train={len(prepared.index.train)} | fit={len(prepared.split.fit_indices)} | "
        f"inner-val={len(prepared.split.validation_indices)} | "
        f"official-test RESERVED={len(prepared.index.test)}"
    )
    print(
        f"input=B×3×288×384 | batches/epoch={len(train_loader)} | "
        f"balanced batch={min(batch_sizes)}..{max(batch_sizes)} | val batches={len(val_loader)}"
    )
    print("\n========== MODEL ==========")
    print(json.dumps(report["model"], indent=2, ensure_ascii=False))
    print("\n========== PARAMETERS ==========")
    for row in parameters["modules"]:
        print(f"{row['module']:<24} {row['parameters']:>12,}")
    print(f"{'ALL':<24} {parameters['total']:>12,} ({parameters['fp32_mib']:.2f} MiB FP32)")
    print("\n========== LOSS / CONFLICT CONTROL ==========")
    print(json.dumps(report["criterion"], indent=2, ensure_ascii=False))
    print(
        f"mode={_campaign_mode(prepared.config)} | active={','.join(_active_tasks(prepared.config))} | "
        "fusion: segmentation -> depth | source detached"
    )
    print("\n========== CHECKPOINT SELECTION ==========")
    trainer_config = _trainer_config(prepared)
    print(
        f"inner-val monitor={trainer_config.monitor} ({trainer_config.monitor_mode}) | "
        f"patience={trainer_config.early_stopping_patience} | "
        "official test is locked until evaluate-test"
    )
    output = prepared.drive_run / "inspection.json"
    _atomic_json(output, report)
    print("Inspection:", output)
    return report


def _build_trainer(
    prepared: Prepared,
    train_loader: DataLoader[Any],
) -> tuple[Trainer, CheckpointManager]:
    model = _create_model(prepared)
    criterion = _create_criterion(prepared)
    optimizer, scheduler, _, _ = _optimizer_scheduler(
        prepared, model, len(train_loader)
    )
    config = _trainer_config(prepared)
    logger = TrainingLogger(
        prepared.local_run,
        run_id=prepared.config["run_id"],
        fsync=False,
    )
    manager = CheckpointManager(prepared.drive_run / "checkpoints")
    trainer = Trainer(
        model,
        criterion,
        optimizer,
        config,
        device="cuda" if torch.cuda.is_available() else "cpu",
        scheduler=scheduler,
        logger=logger,
        checkpoint_manager=manager,
        validation_metrics=_create_metrics(prepared),
        checkpoint_extra=prepared.context,
        event_callback=YoloProgressReporter(
            every_n_steps=int(prepared.config["train"]["progress_every_n_steps"]),
            active_tasks=_active_tasks(prepared.config),
        ),
    )
    trainer.scaler = torch.amp.GradScaler(
        "cuda",
        enabled=trainer.amp_enabled,
        init_scale=float(prepared.config["train"]["amp_initial_scale"]),
    )
    return trainer, manager


def _history_path(prepared: Prepared) -> Path:
    return prepared.drive_run / "history.json"


def _load_history(prepared: Prepared) -> list[dict[str, Any]]:
    path = _history_path(prepared)
    if not path.exists():
        return []
    value = _read_json(path)
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(f"invalid training history: {path}")
    return value


def _official_test_loader(prepared: Prepared) -> DataLoader[Any]:
    # This function is intentionally called only after strict-loading best.pt.
    data = prepared.config["data"]
    dataset = Nyuv2Dataset(
        prepared.index.root,
        split="test",
        augmentation=None,
        **_dataset_kwargs(prepared),
    )
    return DataLoader(
        dataset,
        batch_size=int(data["batch_size"]),
        shuffle=False,
        drop_last=False,
        **_loader_options(data),
    )


def _evaluate_official_test(
    prepared: Prepared,
    trainer: Trainer,
    manager: CheckpointManager,
) -> dict[str, Any]:
    report_path = prepared.drive_run / "official_test_metrics.json"
    if report_path.exists():
        existing = _read_json(report_path)
        if not isinstance(existing, dict) or existing.get("context") != prepared.context:
            raise FileExistsError(f"official test artifact conflicts: {report_path}")
        print("[official-test] already completed; not evaluating twice")
        print(json.dumps(existing, indent=2, ensure_ascii=False))
        return existing
    best = manager.directory / "best.pt"
    if not best.is_file():
        raise FileNotFoundError("best.pt is missing; official test remains locked")
    print("\n[official-test] checksum verification + strict best.pt load", flush=True)
    state = load_training_checkpoint(
        best,
        model=trainer.model,
        optimizer=trainer.optimizer,
        trainer_config=trainer.config,
        scheduler=trainer.scheduler,
        scaler=trainer.scaler,
        criterion=trainer.criterion,
        restore_rng=False,
        expected_extra=prepared.context,
    )
    best_sha = _sha256_file(best)
    test_loader = _official_test_loader(prepared)
    print(
        f"[official-test] UNLOCKED | samples={len(prepared.index.test)} | "
        f"batches={len(test_loader)} | checkpoint epoch={state.next_epoch}",
        flush=True,
    )
    trainer.model.eval()
    result = trainer.validate(test_loader, epoch=max(0, state.next_epoch - 1))
    report = {
        "schema_version": 1,
        "protocol_id": prepared.context["protocol_id"],
        "context": prepared.context,
        "selection": {
            "split": "inner validation derived only from official train",
            "monitor": trainer.config.monitor,
            "best_checkpoint": str(best),
            "best_checkpoint_sha256": best_sha,
            "best_checkpoint_next_epoch": state.next_epoch,
        },
        "official_test": {
            "source": "gt_sets/val.txt (reserved final benchmark split)",
            "samples": len(prepared.index.test),
            "used_once_after_best_strict_load": True,
            "metrics": _jsonable(result),
        },
    }
    _atomic_json(report_path, report)
    print("\n========== OFFICIAL TEST RESULT ==========")
    print(json.dumps(_jsonable(result), indent=2, ensure_ascii=False))
    print("Official test report:", report_path)
    return report


def train_campaign(
    filename: str | os.PathLike[str],
    *,
    resume: bool,
) -> dict[str, Any]:
    prepared = prepare(filename)
    _seed_everything(int(prepared.config["train"]["seed"]))
    train_loader, val_loader = create_train_loaders(prepared)
    trainer, manager = _build_trainer(prepared, train_loader)
    if resume:
        state = trainer.resume()
        print(
            f"[resume] next epoch={state.next_epoch + 1} | global_step={state.global_step} | "
            f"bad_epochs={trainer.early_stopping_bad_epochs}/"
            f"{trainer.config.early_stopping_patience or 'off'}",
            flush=True,
        )
    elif manager.resume_candidates():
        raise FileExistsError(
            "checkpoints already exist; use --resume or choose a new run_id"
        )
    history = _load_history(prepared)
    if history and int(history[-1]["epoch"]) >= trainer.start_epoch:
        raise ValueError("history is ahead of the checksum-valid resume checkpoint")

    print("\n========== TRAINING PLAN ==========")
    print(
        f"epochs {trainer.start_epoch + 1}->{trainer.config.epochs} | "
        f"fit={len(prepared.split.fit_indices)} | inner-val={len(prepared.split.validation_indices)} | "
        f"batch max={prepared.config['data']['batch_size']} | "
        f"batches/epoch={len(train_loader)}"
    )
    print(
        f"monitor={trainer.config.monitor} ({trainer.config.monitor_mode}) | "
        f"patience={trainer.config.early_stopping_patience or 'off'} | "
        f"active={','.join(_active_tasks(prepared.config))} | "
        f"official-test={len(prepared.index.test)} RESERVED"
    )
    while trainer.start_epoch < trainer.config.epochs and not trainer.early_stopping_triggered:
        next_boundary = trainer.start_epoch + 1
        records = trainer.fit(
            train_loader,
            val_loader,
            stop_after_epoch=next_boundary,
        )
        if not records:
            break
        history.extend(_jsonable(records))
        _atomic_json(_history_path(prepared), history)
        latest = records[-1]
        val_total = latest.get("val", {}).get("total")
        monitored = latest.get("val", {}).get(trainer.config.monitor.removeprefix("val/"))
        print(
            f"[epoch {next_boundary}] val/total={float(val_total):.6f} | "
            f"{trainer.config.monitor}={float(monitored):.6f} | "
            f"best={trainer.best_metrics.get(trainer.config.monitor)} | "
            f"no_improve={trainer.early_stopping_bad_epochs}/"
            f"{trainer.config.early_stopping_patience or 'off'}",
            flush=True,
        )

    if not (manager.directory / "best.pt").is_file():
        raise RuntimeError("training ended without a best checkpoint")
    summary = {
        "schema_version": 1,
        "run_id": prepared.config["run_id"],
        "protocol_id": prepared.context["protocol_id"],
        "mode": _campaign_mode(prepared.config),
        "stage": _campaign_stage(prepared.config),
        "training_complete": True,
        "epochs_completed": trainer.start_epoch,
        "early_stopping_triggered": trainer.early_stopping_triggered,
        "monitor": trainer.config.monitor,
        "best_metric": trainer.best_metrics.get(trainer.config.monitor),
        "amp_skip_count": trainer.amp_skip_count,
        "official_test_used": False,
    }
    _atomic_json(prepared.drive_run / "run_summary.json", summary)
    for name in ("events.jsonl", "metrics.csv"):
        source = prepared.local_run / name
        if source.is_file():
            shutil.copy2(source, prepared.drive_run / name)
    print("\nRUN COMPLETE")
    print(json.dumps(_jsonable(summary), indent=2, ensure_ascii=False))
    return summary


def evaluate_test_campaign(filename: str | os.PathLike[str]) -> dict[str, Any]:
    """Run the reserved official test exactly once after training completes."""

    prepared = prepare(filename)
    summary_path = prepared.drive_run / "run_summary.json"
    if not summary_path.is_file():
        raise RuntimeError("official test is locked until a completed training run exists")
    summary = _read_json(summary_path)
    if not isinstance(summary, Mapping) or summary.get("training_complete") is not True:
        raise RuntimeError("official test is locked until training_complete=true")
    _seed_everything(int(prepared.config["train"]["seed"]))
    train_loader, _ = create_train_loaders(prepared)
    trainer, manager = _build_trainer(prepared, train_loader)
    return _evaluate_official_test(prepared, trainer, manager)


def protocol_info() -> dict[str, Any]:
    """Print the runner protocol contract for notebook bootstrap checks."""

    payload = {
        "schema_version": 1,
        "default_protocol_id": PROTOCOL_ID,
        "supported_protocol_ids": list(SUPPORTED_PROTOCOL_IDS),
        "source_commit": _runtime_source_commit(),
    }
    print(json.dumps(payload, sort_keys=True))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("protocol-info")
    for command in ("extract", "inspect"):
        child = subparsers.add_parser(command)
        child.add_argument("--config", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--resume", action="store_true")
    test = subparsers.add_parser("evaluate-test", aliases=["test"])
    test.add_argument("--config", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "protocol-info":
        protocol_info()
    elif args.command == "extract":
        extract_campaign(args.config)
    elif args.command == "inspect":
        inspect_campaign(args.config)
    elif args.command == "train":
        train_campaign(args.config, resume=bool(args.resume))
    else:
        evaluate_test_campaign(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
