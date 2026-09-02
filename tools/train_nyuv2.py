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
)


PROTOCOL_ID = "replite-nyuv2-segdepth-v1"
ACTIVE_TASKS = ("segmentation", "depth")
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


def load_campaign(filename: str | os.PathLike[str]) -> tuple[Path, dict[str, Any]]:
    """Load and validate the locked NYUDv2 campaign configuration."""

    path = Path(filename).expanduser().resolve()
    value = _read_json(path)
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("unsupported campaign schema")
    if value.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"protocol_id must be {PROTOCOL_ID!r}")
    _plain_name(value.get("run_id"), "run_id")
    for section in ("archive", "paths", "model", "data", "train"):
        if not isinstance(value.get(section), dict):
            raise ValueError(f"campaign {section} must be a mapping")

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
    if model.get("active_tasks") != list(ACTIVE_TASKS):
        raise ValueError("model.active_tasks must be ['segmentation', 'depth']")
    if model.get("dense_fusion_direction") != "seg_to_depth":
        raise ValueError("NYUDv2 requires dense_fusion_direction='seg_to_depth'")
    if model.get("dense_fusion_detach_source") is not True:
        raise ValueError("NYUDv2 requires dense_fusion_detach_source=true")

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

    train = value["train"]
    if train.get("monitor") != "val/total" or train.get("monitor_mode") != "min":
        raise ValueError("checkpoint selection must minimize val/total")
    if train.get("early_stopping_patience") != 10:
        raise ValueError("early stopping patience is locked to 10 validations")
    if train.get("task_weights") != {"segmentation": 1.0, "depth": 0.25}:
        raise ValueError("task weights must be segmentation=1.0 and depth=0.25")
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
        "protocol_id": PROTOCOL_ID,
        "archive_bytes": archive["expected_bytes"],
        "archive_sha256": archive["sha256"],
        "official_train_samples": archive["expected_train_samples"],
        "official_test_samples": archive["expected_test_samples"],
    }
    if not isinstance(marker, dict) or any(marker.get(key) != item for key, item in required.items()):
        raise ValueError("extracted NYUDv2 marker does not match the locked archive")
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
            "protocol_id": PROTOCOL_ID,
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


def _create_or_load_inner_split(
    index: Nyuv2Index,
    *,
    fraction: float,
    seed: int,
    manifest_path: Path,
) -> InnerSplit:
    if len(index.train) < 2:
        raise ValueError("official training split is too small for inner validation")
    count = min(len(index.train) - 1, max(1, round(len(index.train) * fraction)))
    ranked = sorted(
        range(len(index.train)),
        key=lambda position: hashlib.sha256(
            f"{seed}:{index.train[position].key}".encode("utf-8")
        ).hexdigest(),
    )
    validation = frozenset(ranked[:count])
    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
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
    _ensure_locked_json(drive_run / "resolved_config.json", config)
    split = _create_or_load_inner_split(
        index,
        fraction=float(config["data"]["inner_validation_fraction"]),
        seed=int(config["data"]["split_seed"]),
        manifest_path=drive_run / "inner_split_manifest.json",
    )
    context = {
        "protocol_id": PROTOCOL_ID,
        "run_id": config["run_id"],
        "archive_sha256": marker["archive_sha256"],
        "config_sha256": _canonical_sha256(config),
        "split_sha256": split.manifest_sha256,
        "active_tasks": list(ACTIVE_TASKS),
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
    return RepLiteConfig(
        tasks=TaskConfig(
            detection_classes=None,
            segmentation_classes=int(data["num_classes"]),
            depth=True,
            gated_dense_fusion=True,
            dense_fusion_direction=str(source["dense_fusion_direction"]),
            dense_fusion_detach_source=bool(source["dense_fusion_detach_source"]),
        ),
        backbone_name=str(source["backbone_name"]),
        pretrained=bool(source["pretrained_in1k"]),
        recurrence_steps=int(source["recurrence_steps"]),
        recurrent_c4_channels=int(source["recurrent_c4_channels"]),
        recurrent_c5_channels=int(source["recurrent_c5_channels"]),
        neck_channels=int(source["neck_channels"]),
        dense_channels=int(source["dense_channels"]),
        task_adapter_channels=int(source["task_adapter_channels"]),
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
    return MultiTaskCriterion(
        task_weights=prepared.config["train"]["task_weights"],
        segmentation_ignore_index=int(data["ignore_index"]),
        depth_loss_type="log_l1_silog",
        depth_min=float(data["depth_min_metres"]),
        depth_max=float(data["depth_max_metres"]),
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
        early_stopping_patience=int(train["early_stopping_patience"]),
        early_stopping_min_delta=float(train["early_stopping_min_delta"]),
    )


def _create_metrics(prepared: Prepared) -> MultiTaskMetrics:
    data = prepared.config["data"]
    return MultiTaskMetrics(
        segmentation=SegmentationMetrics(
            int(data["num_classes"]),
            ignore_index=int(data["ignore_index"]),
        ),
        depth=DepthMetrics(
            min_depth=float(data["depth_min_metres"]),
            max_depth=float(data["depth_max_metres"]),
        ),
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
        "protocol_id": PROTOCOL_ID,
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
            "monitor": "val/total",
            "early_stopping_patience": 10,
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
    print("fusion: segmentation -> depth | source detached | no depth -> segmentation")
    print("\n========== CHECKPOINT SELECTION ==========")
    print("inner-val total loss | patience=10 | official test decoded only after best.pt")
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
            active_tasks=ACTIVE_TASKS,
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
        "protocol_id": PROTOCOL_ID,
        "context": prepared.context,
        "selection": {
            "split": "inner validation derived only from official train",
            "monitor": "val/total",
            "best_checkpoint": str(best),
            "best_checkpoint_sha256": best_sha,
            "best_checkpoint_next_epoch": state.next_epoch,
        },
        "official_test": {
            "source": "gt_sets/val.txt (reserved final benchmark split)",
            "samples": len(prepared.index.test),
            "used_once_after_best_strict_load": True,
            "metrics": result,
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
            f"bad_epochs={trainer.early_stopping_bad_epochs}/10",
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
        "monitor=val/total | patience=10 | task weights seg=1.0 depth=0.25 | "
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
        print(
            f"[epoch {next_boundary}] val/total={float(val_total):.6f} | "
            f"best={trainer.best_metrics.get('val/total')} | "
            f"no_improve={trainer.early_stopping_bad_epochs}/10",
            flush=True,
        )

    if not (manager.directory / "best.pt").is_file():
        raise RuntimeError("training ended without a best checkpoint")
    test_report = _evaluate_official_test(prepared, trainer, manager)
    summary = {
        "schema_version": 1,
        "run_id": prepared.config["run_id"],
        "epochs_completed": trainer.start_epoch,
        "early_stopping_triggered": trainer.early_stopping_triggered,
        "best_val_total": trainer.best_metrics.get("val/total"),
        "amp_skip_count": trainer.amp_skip_count,
        "official_test_report": str(
            prepared.drive_run / "official_test_metrics.json"
        ),
        "official_test_metrics": test_report["official_test"]["metrics"],
    }
    _atomic_json(prepared.drive_run / "run_summary.json", summary)
    for name in ("events.jsonl", "metrics.csv"):
        source = prepared.local_run / name
        if source.is_file():
            shutil.copy2(source, prepared.drive_run / name)
    print("\nRUN COMPLETE")
    print(json.dumps(_jsonable(summary), indent=2, ensure_ascii=False))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("extract", "inspect"):
        child = subparsers.add_parser(command)
        child.add_argument("--config", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "extract":
        extract_campaign(args.config)
    elif args.command == "inspect":
        inspect_campaign(args.config)
    else:
        train_campaign(args.config, resume=bool(args.resume))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
