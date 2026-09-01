#!/usr/bin/env python3
"""Audited SANPO full-data inspection, gated epoch one, and main training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from replite.data import (
    ArchiveCatalog,
    ArchiveGroupSplit,
    ArchiveShardLoader,
    LocalArchiveStage,
    SANPO_DERIVED_DETECTION_CONFIG,
    SANPO_DERIVED_DETECTION_CONFIG_SHA256,
    SANPO_DETECTION_CLASS_NAMES,
    SANPO_SEGMENTATION_CLASS_NAMES,
    SANPO_SEGMENTATION_IGNORE_INDEX,
    canonical_json_sha256,
    create_or_load_group_split,
    load_archive_catalog,
)
from replite.multitask import RepLiteConfig, TaskConfig, create_replite_model
from replite.training import (
    CheckpointManager,
    DepthMetrics,
    DetectionMAP,
    MultiTaskCriterion,
    MultiTaskMetrics,
    SegmentationMetrics,
    SnapshotContext,
    Trainer,
    TrainerConfig,
    TrainingLogger,
    WarmupCosineScheduler,
    YoloProgressReporter,
    create_adamw,
    detection_per_class_rows,
    flatten_epoch_record,
    move_to_device,
    publish_epoch_snapshot,
    restore_latest_snapshot,
    segmentation_per_class_rows,
)


EXPECTED = {
    "records": 234,
    "train_records": 186,
    "test_records": 48,
    "train_frames": 14_718,
    "test_frames": 3_803,
}
PROTOCOL_ID = "replite-sanpo-real-human-v0-session-split-v4"
APPROVAL_KIND = "replite-sanpo-main-epoch1-approval-v3"
DETECTION_MIN_COMPONENT_PIXELS = int(
    SANPO_DERIVED_DETECTION_CONFIG["min_component_pixels"]
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


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    content = (
        json.dumps(
            _jsonable(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    _atomic_bytes(path, content)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_campaign(filename: str | os.PathLike[str]) -> tuple[Path, dict[str, Any]]:
    """Read the resolved notebook config and reject malformed campaigns."""

    path = Path(filename).expanduser().resolve()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read campaign config: {path}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("unsupported campaign config schema")
    for section in ("model", "data", "train", "metrics"):
        if not isinstance(value.get(section), dict):
            raise ValueError(f"campaign {section} must be a mapping")
    for field in (
        "run_id",
        "source_repository",
        "source_commit",
        "drive_data_root",
        "drive_runs_root",
        "local_work_root",
    ):
        if not isinstance(value.get(field), str) or not value[field]:
            raise ValueError(f"campaign {field} must be a non-empty string")
    if value["run_id"] in {".", ".."} or any(
        token in value["run_id"] for token in ("/", "\\")
    ):
        raise ValueError("run_id must be a plain directory name")
    epochs = value["train"].get("epochs")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 2:
        raise ValueError("train.epochs must be at least two")
    size = value["data"].get("image_size")
    if (
        not isinstance(size, list)
        or len(size) != 2
        or any(
            isinstance(item, bool)
            or not isinstance(item, int)
            or item <= 0
            or item % 32
            for item in size
        )
    ):
        raise ValueError("data.image_size must be two positive multiples of 32")
    detection_min_area = value["data"].get("detection_min_component_pixels")
    if (
        isinstance(detection_min_area, bool)
        or not isinstance(detection_min_area, int)
        or detection_min_area != DETECTION_MIN_COMPONENT_PIXELS
    ):
        raise ValueError(
            "data.detection_min_component_pixels must be 100 for the locked "
            "SANPO derived-detection protocol"
        )
    return path, value


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


def _assert_catalog(catalog: ArchiveCatalog) -> None:
    observed = {
        "records": len(catalog.records),
        "train_records": len(catalog.train_records),
        "test_records": len(catalog.test_records),
        "train_frames": sum(item.joint_frames for item in catalog.train_records),
        "test_frames": sum(item.joint_frames for item in catalog.test_records),
    }
    if observed != EXPECTED:
        raise ValueError(
            "catalog differs from the locked 234-archive download: "
            f"observed={observed}, expected={EXPECTED}"
        )
    train_sessions = {item.session_id for item in catalog.train_records}
    test_sessions = {item.session_id for item in catalog.test_records}
    if train_sessions & test_sessions:
        raise ValueError("official train and official test overlap by session_id")


@dataclass(frozen=True)
class Prepared:
    config_path: Path
    config: dict[str, Any]
    data_root: Path
    local_root: Path
    local_run: Path
    drive_run: Path
    catalog: ArchiveCatalog
    split: ArchiveGroupSplit
    context: SnapshotContext
    checkpoint_extra: dict[str, Any]
    train_stage: LocalArchiveStage
    train_loader: ArchiveShardLoader
    val_loader: ArchiveShardLoader


def prepare(filename: str | os.PathLike[str]) -> Prepared:
    config_path, config = load_campaign(filename)
    data_root = Path(config["drive_data_root"]).expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"SANPO data root is missing: {data_root}")
    print(
        "[prepare] 1/3 metadata catalog từ Drive "
        "(không stat 234 archive/sidecar)",
        flush=True,
    )
    catalog = load_archive_catalog(
        data_root,
        validate_archive_files=False,
        validate_sidecars=False,
    )
    if catalog.detection_config_sha256 != SANPO_DERIVED_DETECTION_CONFIG_SHA256:
        raise ValueError(
            "archive catalog detection policy differs from the locked main "
            "training protocol"
        )
    _assert_catalog(catalog)
    print(
        f"[prepare] 2/3 catalog OK | {len(catalog.records)} archives | "
        "byte/SHA vật lý sẽ kiểm đầy đủ khi stage",
        flush=True,
    )
    seed = int(config["data"]["split_seed"])
    fraction = float(config["data"]["validation_fraction"])
    basis_points = round(fraction * 10_000)
    split = create_or_load_group_split(
        catalog,
        data_root
        / "metadata"
        / f"replite_main_split_seed{seed}_val{basis_points:04d}_v2.json",
        seed=seed,
        validation_fraction=fraction,
    )
    train_sessions = {item.session_id for item in split.train_records}
    val_sessions = {item.session_id for item in split.validation_records}
    if train_sessions & val_sessions:
        raise AssertionError("train/validation session leakage")
    if split.official_test_records != catalog.test_records:
        raise AssertionError("official-test identity changed during split creation")
    if {item.key for item in (*split.train_records, *split.validation_records)} != {
        item.key for item in catalog.train_records
    }:
        raise AssertionError("fit plus inner-validation does not cover official train")
    print(
        f"[prepare] 3/3 split OK | fit={len(split.train_records)} | "
        f"inner-val={len(split.validation_records)} | "
        f"official-test={len(split.official_test_records)}",
        flush=True,
    )

    config_sha = canonical_json_sha256(config)
    source_sha = canonical_json_sha256(
        {
            "repository": config["source_repository"],
            "commit": config["source_commit"],
        }
    )
    context = SnapshotContext(
        source_sha256=source_sha,
        config_sha256=config_sha,
        catalog_sha256=catalog.catalog_sha256,
        split_sha256=split.manifest_sha256,
    )
    extra = {
        **context.as_dict(),
        "run_id": config["run_id"],
        "protocol_id": PROTOCOL_ID,
        "official_test_used": False,
    }
    local_root = Path(config["local_work_root"]).expanduser().resolve()
    staging = config["data"].get("local_staging", {})
    if not isinstance(staging, Mapping):
        raise ValueError("data.local_staging must be a mapping")
    expansion_factor = staging.get("expansion_factor", 1.05)
    reserve_gib = staging.get("reserve_gib", 4.0)
    stage_cache_id = staging.get("cache_id", config["run_id"])
    if (
        not isinstance(stage_cache_id, str)
        or not stage_cache_id
        or stage_cache_id in {".", ".."}
        or "/" in stage_cache_id
        or "\\" in stage_cache_id
    ):
        raise ValueError("data.local_staging.cache_id must be one path component")
    if (
        isinstance(expansion_factor, bool)
        or not isinstance(expansion_factor, (int, float))
        or not math.isfinite(float(expansion_factor))
        or float(expansion_factor) < 1.0
    ):
        raise ValueError("data.local_staging.expansion_factor must be at least one")
    if (
        isinstance(reserve_gib, bool)
        or not isinstance(reserve_gib, (int, float))
        or not math.isfinite(float(reserve_gib))
        or float(reserve_gib) < 0.0
    ):
        raise ValueError("data.local_staging.reserve_gib must be non-negative")
    train_stage = LocalArchiveStage(
        catalog.train_records,
        local_root=local_root
        / "stages"
        / stage_cache_id
        / "official_train",
        purpose="official_train",
        expansion_factor=float(expansion_factor),
        reserve_bytes=math.ceil(float(reserve_gib) * 1024**3),
    )
    common = {
        "local_root": local_root / "shards" / config["run_id"],
        "batch_size": int(config["data"]["batch_size"]),
        "image_size": tuple(config["data"]["image_size"]),
        "seed": seed,
        "num_workers": int(config["data"]["num_workers"]),
        "pin_memory": torch.cuda.is_available(),
        "drop_last": False,
        "prefetch_factor": int(config["data"]["prefetch_factor"]),
        "dataset_kwargs": {
            "depth_min": float(config["data"]["depth_min_metres"]),
            "depth_max": float(config["data"]["depth_max_metres"]),
            "detection_min_area": int(
                config["data"]["detection_min_component_pixels"]
            ),
            "normalize": True,
        },
        "local_stage": train_stage,
    }
    return Prepared(
        config_path=config_path,
        config=config,
        data_root=data_root,
        local_root=local_root,
        local_run=local_root / "runs" / config["run_id"],
        drive_run=Path(config["drive_runs_root"]).expanduser().resolve()
        / config["run_id"],
        catalog=catalog,
        split=split,
        context=context,
        checkpoint_extra=extra,
        train_stage=train_stage,
        train_loader=ArchiveShardLoader(split.train_records, shuffle=True, **common),
        val_loader=ArchiveShardLoader(
            split.validation_records, shuffle=False, **common
        ),
    )


def model_config(prepared: Prepared, *, pretrained: bool | None = None) -> RepLiteConfig:
    source = prepared.config["model"]
    return RepLiteConfig(
        tasks=TaskConfig(
            detection_classes=len(SANPO_DETECTION_CLASS_NAMES),
            segmentation_classes=len(SANPO_SEGMENTATION_CLASS_NAMES),
            depth=True,
            gated_dense_fusion=True,
        ),
        backbone_name=str(source["backbone_name"]),
        pretrained=(
            bool(source["pretrained_in1k"]) if pretrained is None else pretrained
        ),
        recurrence_steps=int(source["recurrence_steps"]),
        recurrent_c4_channels=int(source["recurrent_c4_channels"]),
        recurrent_c5_channels=int(source["recurrent_c5_channels"]),
        neck_channels=int(source["neck_channels"]),
        dense_channels=int(source["dense_channels"]),
        task_adapter_channels=int(source["task_adapter_channels"]),
        detection_head_channels=int(source["detection_head_channels"]),
        detection_head_blocks=int(source["detection_head_blocks"]),
        detection_reg_max=int(source["detection_reg_max"]),
        use_sppf=bool(source["use_sppf"]),
    )


def create_model(prepared: Prepared) -> nn.Module:
    return create_replite_model(
        model_config(prepared),
        cache_dir=str(prepared.data_root / "pretrained_cache"),
    )


def create_criterion(prepared: Prepared) -> MultiTaskCriterion:
    data = prepared.config["data"]
    return MultiTaskCriterion(
        detection_num_classes=len(SANPO_DETECTION_CLASS_NAMES),
        detection_reg_max=model_config(prepared).detection_reg_max,
        segmentation_ignore_index=SANPO_SEGMENTATION_IGNORE_INDEX,
        depth_loss_type="log_l1_silog",
        depth_min=float(data["depth_min_metres"]),
        depth_max=float(data["depth_max_metres"]),
    )


def create_trainer_config(prepared: Prepared) -> TrainerConfig:
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
    )


def create_optimizer_schedule(
    prepared: Prepared, model: nn.Module
) -> tuple[torch.optim.Optimizer, WarmupCosineScheduler, int, int]:
    train = prepared.config["train"]
    optimizer = create_adamw(
        model,
        lr=float(train["base_lr"]),
        weight_decay=float(train["weight_decay"]),
        backbone_lr_multiplier=float(train["backbone_lr_multiplier"]),
    )
    trainer_config = create_trainer_config(prepared)
    updates = math.ceil(
        len(prepared.train_loader) / trainer_config.grad_accum_steps
    )
    total = updates * trainer_config.epochs
    warmup = min(total, max(0, round(total * float(train["warmup_fraction"]))))
    scheduler = WarmupCosineScheduler(
        optimizer,
        total_steps=total,
        warmup_steps=warmup,
        min_lr_ratio=float(train["min_lr_ratio"]),
    )
    return optimizer, scheduler, total, warmup


def create_metrics(prepared: Prepared) -> MultiTaskMetrics:
    metric = prepared.config["metrics"]
    data = prepared.config["data"]
    return MultiTaskMetrics(
        detection=DetectionMAP(
            len(SANPO_DETECTION_CLASS_NAMES),
            max_detections=int(metric["detection_max_detections"]),
        ),
        segmentation=SegmentationMetrics(
            len(SANPO_SEGMENTATION_CLASS_NAMES),
            ignore_index=SANPO_SEGMENTATION_IGNORE_INDEX,
        ),
        depth=DepthMetrics(
            min_depth=float(data["depth_min_metres"]),
            max_depth=float(data["depth_max_metres"]),
        ),
        detection_reg_max=model_config(prepared).detection_reg_max,
        detection_score_threshold=float(metric["detection_score_threshold"]),
        detection_nms_iou_threshold=float(
            metric["detection_nms_iou_threshold"]
        ),
    )


@dataclass
class TrainObjects:
    model: nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: WarmupCosineScheduler
    trainer: Trainer
    logger: TrainingLogger
    manager: CheckpointManager
    config: TrainerConfig
    total_steps: int
    warmup_steps: int


def build_train_objects(
    prepared: Prepared,
    *,
    amp_initial_scale: float | None = None,
) -> TrainObjects:
    _seed_everything(int(prepared.config["train"]["seed"]))
    model = create_model(prepared)
    criterion = create_criterion(prepared)
    optimizer, scheduler, total, warmup = create_optimizer_schedule(prepared, model)
    config = create_trainer_config(prepared)
    prepared.local_run.mkdir(parents=True, exist_ok=True)
    logger = TrainingLogger(
        prepared.local_run, run_id=prepared.config["run_id"], fsync=False
    )
    manager = CheckpointManager(prepared.local_run / "checkpoints")
    trainer = Trainer(
        model,
        criterion,
        optimizer,
        config,
        device="cuda" if torch.cuda.is_available() else "cpu",
        scheduler=scheduler,
        logger=logger,
        checkpoint_manager=manager,
        validation_metrics=create_metrics(prepared),
        checkpoint_extra=prepared.checkpoint_extra,
        event_callback=YoloProgressReporter(
            every_n_steps=int(
                prepared.config["train"]["progress_every_n_steps"]
            ),
            reg_max=model_config(prepared).detection_reg_max,
        ),
    )
    trainer.scaler = torch.amp.GradScaler(
        "cuda",
        enabled=trainer.amp_enabled,
        init_scale=(
            float(prepared.config["train"]["amp_initial_scale"])
            if amp_initial_scale is None
            else float(amp_initial_scale)
        ),
    )
    return TrainObjects(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        trainer=trainer,
        logger=logger,
        manager=manager,
        config=config,
        total_steps=total,
        warmup_steps=warmup,
    )


def _parameter_summary(model: nn.Module) -> dict[str, Any]:
    total = sum(item.numel() for item in model.parameters())
    trainable = sum(
        item.numel() for item in model.parameters() if item.requires_grad
    )
    modules = []
    for name, module in model.named_children():
        count = sum(item.numel() for item in module.parameters())
        active = sum(
            item.numel() for item in module.parameters() if item.requires_grad
        )
        modules.append(
            {
                "module": name,
                "total": count,
                "trainable": active,
                "frozen": count - active,
            }
        )
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "fp32_mib": total * 4 / 1024**2,
        "modules": modules,
    }


def _table(
    title: str, headers: Sequence[str], rows: Sequence[Sequence[Any]]
) -> None:
    text = [[str(value) for value in row] for row in rows]
    widths = [len(name) for name in headers]
    for row in text:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    print(f"\n========== {title} ==========")
    print(
        "  ".join(name.ljust(widths[index]) for index, name in enumerate(headers))
    )
    print("  ".join("-" * width for width in widths))
    for row in text:
        print(
            "  ".join(
                value.ljust(widths[index]) for index, value in enumerate(row)
            )
        )


def _print_execution_plan(
    prepared: Prepared,
    objects: TrainObjects,
    *,
    start_epoch: int,
) -> None:
    accumulate = objects.config.grad_accum_steps
    train_batches = len(prepared.train_loader)
    val_batches = len(prepared.val_loader)
    updates = math.ceil(train_batches / accumulate)
    print("\n========== TRAIN / VALIDATION PLAN ==========")
    print(
        f"epochs {start_epoch + 1}->{objects.config.epochs} | "
        f"micro_batch={prepared.config['data']['batch_size']} | "
        f"accumulate={accumulate} | "
        f"effective_batch={int(prepared.config['data']['batch_size']) * accumulate}"
    )
    print(
        f"train={prepared.train_loader.sample_count:,} samples, "
        f"{train_batches:,} batches/epoch | "
        f"optimizer={updates:,} updates/epoch | "
        f"validation={prepared.val_loader.sample_count:,} samples, "
        f"{val_batches:,} batches/epoch"
    )
    print(
        f"campaign optimizer updates={objects.total_steps:,} | "
        f"warmup={objects.warmup_steps:,} | "
        f"progress every={prepared.config['train']['progress_every_n_steps']} batches"
    )


def inspection_payload(
    prepared: Prepared,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
) -> dict[str, Any]:
    train_sessions = {item.session_id for item in prepared.split.train_records}
    val_sessions = {item.session_id for item in prepared.split.validation_records}
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "context": prepared.context.as_dict(),
        "data": {
            "catalog_records": len(prepared.catalog.records),
            "annotation_policy": prepared.catalog.annotation_policy,
            "detection_config_sha256": prepared.catalog.detection_config_sha256,
            "train_archives": len(prepared.split.train_records),
            "val_archives": len(prepared.split.validation_records),
            "official_test_archives_reserved": len(
                prepared.split.official_test_records
            ),
            "train_sessions": len(train_sessions),
            "val_sessions": len(val_sessions),
            "train_samples": prepared.train_loader.sample_count,
            "val_samples": prepared.val_loader.sample_count,
            "official_test_samples_reserved": EXPECTED["test_frames"],
            "official_test_used": False,
            "local_stage_cache_id": prepared.config["data"]
            .get("local_staging", {})
            .get("cache_id", prepared.config["run_id"]),
            "image_size": prepared.config["data"]["image_size"],
            "clip_frames": ["t-2", "t-1", "t"],
            "detection_classes": list(SANPO_DETECTION_CLASS_NAMES),
            "detection_min_component_pixels": prepared.config["data"][
                "detection_min_component_pixels"
            ],
            "detection_archive_sources": {
                source: sum(
                    item.detection_source == source
                    for item in prepared.catalog.records
                )
                for source in ("packaged_json", "panoptic_on_load")
            },
            "segmentation_classes": list(SANPO_SEGMENTATION_CLASS_NAMES),
            "depth_range_metres": [
                prepared.config["data"]["depth_min_metres"],
                prepared.config["data"]["depth_max_metres"],
            ],
        },
        "model_config": model.config.as_dict(),
        "model_metadata": model.model_metadata,
        "pretrained": model.backbone.weights_provenance,
        "feature_stages": model.backbone.feature_info.get_dicts(),
        "parameters": _parameter_summary(model),
        "optimizer_groups": [
            {
                "name": group.get("name", f"group_{index}"),
                "parameters": sum(item.numel() for item in group["params"]),
                "lr": float(group["lr"]),
                "weight_decay": float(group["weight_decay"]),
            }
            for index, group in enumerate(optimizer.param_groups)
        ],
        "schedule": {
            "epochs": prepared.config["train"]["epochs"],
            "micro_batch_size": prepared.config["data"]["batch_size"],
            "grad_accum_steps": prepared.config["train"]["grad_accum_steps"],
            "effective_batch_size": (
                int(prepared.config["data"]["batch_size"])
                * int(prepared.config["train"]["grad_accum_steps"])
            ),
            "batches_per_epoch": len(prepared.train_loader),
            "val_batches": len(prepared.val_loader),
            "optimizer_updates_per_epoch": math.ceil(
                len(prepared.train_loader)
                / int(prepared.config["train"]["grad_accum_steps"])
            ),
            "total_steps": total_steps,
            "warmup_steps": warmup_steps,
            "num_workers": prepared.config["data"]["num_workers"],
            "prefetch_factor": prepared.config["data"]["prefetch_factor"],
            "progress_every_n_batches": prepared.config["train"][
                "progress_every_n_steps"
            ],
        },
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "amp_initial_scale": prepared.config["train"]["amp_initial_scale"],
            "deterministic": "warn_only",
            "tf32": False,
            "persistent_workers": False,
            "archive_policy": (
                "stage all 186 official-train shards once on local SSD; "
                "reuse for fit and inner-validation; official-test deferred"
            ),
            "train_stage": prepared.train_stage.disk_plan(),
        },
    }


def inspect_campaign(filename: str | os.PathLike[str]) -> dict[str, Any]:
    print("[inspect] phase 1/4: metadata + split", flush=True)
    prepared = prepare(filename)
    print("[inspect] phase 2/4: dựng model + pretrained IN-1K", flush=True)
    _seed_everything(int(prepared.config["train"]["seed"]))
    model = create_model(prepared)
    print("[inspect] phase 3/4: optimizer + schedule", flush=True)
    optimizer, _, total, warmup = create_optimizer_schedule(prepared, model)
    print("[inspect] phase 4/4: SSD plan + report", flush=True)
    result = inspection_payload(prepared, model, optimizer, total, warmup)
    _table(
        "DATA AUDIT",
        ("split", "archives", "sessions", "samples", "role"),
        (
            (
                "train",
                result["data"]["train_archives"],
                result["data"]["train_sessions"],
                result["data"]["train_samples"],
                "optimizer",
            ),
            (
                "val",
                result["data"]["val_archives"],
                result["data"]["val_sessions"],
                result["data"]["val_samples"],
                "metrics + best",
            ),
            (
                "official-test",
                result["data"]["official_test_archives_reserved"],
                len(
                    {
                        item.session_id
                        for item in prepared.split.official_test_records
                    }
                ),
                EXPECTED["test_frames"],
                "RESERVED, not opened",
            ),
        ),
    )
    _table(
        "DETECTION LABEL SOURCE",
        ("source", "archives", "policy"),
        (
            (
                "packaged_json",
                result["data"]["detection_archive_sources"]["packaged_json"],
                "versioned JSON inside archive",
            ),
            (
                "panoptic_on_load",
                result["data"]["detection_archive_sources"]["panoptic_on_load"],
                "8-connected, min area 100",
            ),
        ),
    )
    _table(
        "BACKBONE C2-C5",
        ("stage", "module", "channels", "stride"),
        tuple(
            (
                f"C{int(stage['index']) + 2}",
                stage["module"],
                stage["num_chs"],
                stage["reduction"],
            )
            for stage in result["feature_stages"]
        ),
    )
    _table(
        "PARAMETERS",
        ("module", "total", "trainable", "frozen"),
        tuple(
            (
                item["module"],
                f"{item['total']:,}",
                f"{item['trainable']:,}",
                f"{item['frozen']:,}",
            )
            for item in result["parameters"]["modules"]
        )
        + (
            (
                "ALL",
                f"{result['parameters']['total']:,}",
                f"{result['parameters']['trainable']:,}",
                f"{result['parameters']['frozen']:,}",
            ),
        ),
    )
    _table(
        "OPTIMIZER",
        ("group", "parameters", "lr", "weight_decay"),
        tuple(
            (
                group["name"],
                f"{group['parameters']:,}",
                f"{group['lr']:.3g}",
                f"{group['weight_decay']:.3g}",
            )
            for group in result["optimizer_groups"]
        ),
    )
    schedule = result["schedule"]
    _table(
        "TRAINING PLAN (EXACT)",
        ("item", "value"),
        (
            ("train samples", f"{result['data']['train_samples']:,}"),
            ("validation samples", f"{result['data']['val_samples']:,}"),
            ("micro batch / GPU", schedule["micro_batch_size"]),
            ("gradient accumulation", schedule["grad_accum_steps"]),
            ("effective batch", schedule["effective_batch_size"]),
            ("train batches / epoch", f"{schedule['batches_per_epoch']:,}"),
            (
                "optimizer updates / epoch",
                f"{schedule['optimizer_updates_per_epoch']:,}",
            ),
            ("validation batches / epoch", f"{schedule['val_batches']:,}"),
            ("epochs", schedule["epochs"]),
            ("total optimizer updates", f"{schedule['total_steps']:,}"),
            ("warmup updates", f"{schedule['warmup_steps']:,}"),
            (
                "workers / prefetch",
                f"{schedule['num_workers']} / {schedule['prefetch_factor']}",
            ),
            (
                "progress row every batches",
                schedule["progress_every_n_batches"],
            ),
        ),
    )
    print("\nMODEL CONFIG")
    print(json.dumps(result["model_config"], indent=2, ensure_ascii=False))
    print("\nPRETRAINED IN-1K")
    print(json.dumps(result["pretrained"], indent=2, ensure_ascii=False))
    print("\nCONTEXT HASHES")
    print(json.dumps(result["context"], indent=2))
    output = (
        prepared.local_root / "inspections" / f"{prepared.config['run_id']}.json"
    )
    _atomic_json(output, result)
    print("\nInspection JSON:", output)
    print(
        f"Total params {result['parameters']['total']:,} | "
        f"FP32 {result['parameters']['fp32_mib']:.2f} MiB"
    )
    return result


def _one_batch(loader: ArchiveShardLoader) -> tuple[Tensor, dict[str, Any]]:
    iterator = iter(loader)
    try:
        return next(iterator)
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            close()


def _gradient_health(model: nn.Module) -> tuple[int, tuple[str, ...]]:
    """Return connected-gradient count and names containing NaN/Inf values."""

    connected = 0
    nonfinite: list[str] = []
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if not parameter.requires_grad or gradient is None:
            continue
        connected += 1
        if not bool(torch.isfinite(gradient).all()):
            nonfinite.append(name)
    return connected, tuple(nonfinite)


def _finish_preflight_optimizer_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
) -> tuple[bool, int, tuple[str, ...], float, float]:
    """Consume one unscaled preflight gradient set without unsafe updates."""

    connected, nonfinite = _gradient_health(model)
    if connected == 0:
        raise FloatingPointError("preflight loss produced no model gradients")
    old_scale = float(scaler.get_scale())
    if scaler.is_enabled():
        # GradScaler has already inspected the gradients in unscale_(). It
        # atomically skips optimizer.step() when the set contains Inf/NaN.
        scaler.step(optimizer)
        scaler.update()
        new_scale = float(scaler.get_scale())
        stepped = not nonfinite and new_scale >= old_scale
    elif nonfinite:
        new_scale = old_scale
        stepped = False
    else:
        optimizer.step()
        new_scale = old_scale
        stepped = True
    return stepped, connected, nonfinite, old_scale, new_scale


def disposable_preflight(prepared: Prepared) -> dict[str, Any]:
    """Use a throwaway model so production BN buffers and RNG stay pristine."""

    _seed_everything(int(prepared.config["train"]["seed"]))
    prepared.train_loader.set_epoch(0)
    inputs, targets = _one_batch(prepared.train_loader)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(prepared).to(device).train()
    criterion = create_criterion(prepared).to(device)
    optimizer, _, _, _ = create_optimizer_schedule(prepared, model)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda",
        init_scale=float(prepared.config["train"]["amp_initial_scale"]),
    )
    inputs = move_to_device(inputs, device, non_blocking=device.type == "cuda")
    targets = move_to_device(targets, device, non_blocking=device.type == "cuda")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    initial_scale = float(scaler.get_scale())
    overflow_attempts: list[dict[str, Any]] = []
    while True:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            outputs = model(inputs)
            losses = criterion(outputs, targets)
        total = losses["total"]
        if not bool(torch.isfinite(total.detach())):
            raise FloatingPointError("preflight loss is non-finite")
        scaler.scale(total).backward() if scaler.is_enabled() else total.backward()
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        (
            stepped,
            connected_gradients,
            nonfinite_gradients,
            old_scale,
            new_scale,
        ) = _finish_preflight_optimizer_step(model, optimizer, scaler)

        if nonfinite_gradients:
            if not scaler.is_enabled():
                names = ", ".join(nonfinite_gradients[:8])
                raise FloatingPointError(
                    "preflight gradients are non-finite: " + names
                )
            overflow_attempts.append(
                {
                    "scale": old_scale,
                    "next_scale": new_scale,
                    "parameter_count": len(nonfinite_gradients),
                    "parameters": list(nonfinite_gradients[:16]),
                }
            )
            print(
                "[preflight] AMP overflow "
                f"scale={old_scale:g} -> {new_scale:g} | "
                f"nonfinite_gradients={len(nonfinite_gradients)} | "
                f"first={nonfinite_gradients[0]}"
            )
            if new_scale >= old_scale or new_scale < 1.0:
                names = ", ".join(nonfinite_gradients[:8])
                raise FloatingPointError(
                    "preflight AMP could not find a stable scale; "
                    f"last gradients: {names}"
                )
            del outputs, losses, total
            continue

        if scaler.is_enabled() and not stepped:
            raise RuntimeError(
                "GradScaler skipped a finite-gradient preflight update"
            )
        # ``old_scale`` is the scale actually proven finite by this attempt;
        # a growth-interval update could make ``new_scale`` larger but untested.
        stable_scale = old_scale
        break
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak = (
        torch.cuda.max_memory_allocated() / 1024**3
        if device.type == "cuda"
        else 0.0
    )
    result = {
        "input_shape": list(inputs.shape),
        "segmentation_shape": list(outputs.segmentation.shape),
        "depth_shape": list(outputs.depth.shape),
        "detection_levels": [
            list(item.shape) for item in outputs.detection.cls_logits
        ],
        "losses": {
            name: float(value.detach().float().cpu())
            for name, value in losses.items()
            if isinstance(value, Tensor) and value.ndim == 0
        },
        "optimizer_stepped": stepped,
        "amp_initial_scale": initial_scale,
        "amp_stable_scale": stable_scale,
        "amp_backoff_count": len(overflow_attempts),
        "amp_overflow_attempts": overflow_attempts,
        "elapsed_seconds": elapsed,
        "peak_vram_gib": peak,
        "production_model_mutated": False,
    }
    del outputs, losses, inputs, targets
    del optimizer, criterion, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if not stepped:
        raise FloatingPointError("preflight AMP update overflowed")
    return result


def _state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fields))
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field) for field in fields})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_reports(
    prepared: Prepared,
    history: list[dict[str, Any]],
    latest: Mapping[str, Any],
) -> None:
    content = (
        json.dumps(history, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    _atomic_bytes(prepared.local_run / "history.json", content)
    rows = [flatten_epoch_record(item) for item in history]
    if rows:
        _write_csv(prepared.local_run / "history.csv", rows, tuple(rows[0]))
    val = latest.get("val")
    if not isinstance(val, Mapping):
        raise ValueError("epoch record has no full validation result")
    _atomic_json(prepared.local_run / "latest_val_metrics.json", dict(val))
    detection = detection_per_class_rows(val, SANPO_DETECTION_CLASS_NAMES)
    segmentation = segmentation_per_class_rows(
        val, SANPO_SEGMENTATION_CLASS_NAMES
    )
    _write_csv(
        prepared.local_run / "val_detection_per_class.csv",
        detection,
        ("class_id", "class_name", "map50_95", "present"),
    )
    _write_csv(
        prepared.local_run / "val_segmentation_per_class.csv",
        segmentation,
        ("class_id", "class_name", "iou", "present"),
    )
    matrix = val.get("segmentation/confusion_matrix")
    if matrix is not None:
        matrix = _jsonable(matrix)
        fields = (
            "actual_class",
            *[
                str(index)
                for index in range(len(SANPO_SEGMENTATION_CLASS_NAMES))
            ],
        )
        _write_csv(
            prepared.local_run / "val_segmentation_confusion_matrix.csv",
            [
                {
                    "actual_class": index,
                    **{
                        str(column): count
                        for column, count in enumerate(row)
                    },
                }
                for index, row in enumerate(matrix)
            ],
            fields,
        )


def _write_metadata(
    prepared: Prepared, inspection: Mapping[str, Any]
) -> None:
    prepared.local_run.mkdir(parents=True, exist_ok=True)
    _atomic_json(prepared.local_run / "resolved_config.json", prepared.config)
    _atomic_json(
        prepared.local_run / "catalog.json",
        {
            "schema_version": 1,
            "catalog_sha256": prepared.catalog.catalog_sha256,
            "records": [
                {
                    "split": item.split,
                    "session_id": item.session_id,
                    "sensor": item.sensor,
                    "annotation_policy": item.annotation_policy,
                    "selection_sha256": item.selection_sha256,
                    "detection_source": item.detection_source,
                    "detection_config_sha256": item.detection_config_sha256,
                    "package_sha256": item.package_sha256,
                    "archive_name": item.archive_path.name,
                    "archive_bytes": item.archive_bytes,
                    "archive_sha256": item.archive_sha256,
                    "source_bytes": item.source_bytes,
                    "joint_frames": item.joint_frames,
                }
                for item in prepared.catalog.records
            ],
        },
    )
    shutil.copy2(
        prepared.split.manifest_path,
        prepared.local_run / "split_manifest.json",
    )
    _atomic_bytes(
        prepared.local_run / "source_commit.txt",
        (prepared.config["source_commit"] + "\n").encode(),
    )
    _atomic_json(prepared.local_run / "inspection.json", dict(inspection))


def _snapshot_files(run: Path) -> tuple[str, ...]:
    candidates = (
        "checkpoints/best.pt",
        "checkpoints/best.pt.sha256",
        "events.jsonl",
        "metrics.csv",
        "resolved_config.json",
        "catalog.json",
        "split_manifest.json",
        "source_commit.txt",
        "inspection.json",
        "preflight.json",
        "history.json",
        "history.csv",
        "latest_val_metrics.json",
        "val_detection_per_class.csv",
        "val_segmentation_per_class.csv",
        "val_segmentation_confusion_matrix.csv",
        "run_status.json",
        "pilot_report.json",
    )
    return tuple(name for name in candidates if (run / name).is_file())


def _restore(prepared: Prepared) -> Any:
    checkpoint = prepared.local_run / "checkpoints" / "last.pt"
    restored = restore_latest_snapshot(
        prepared.drive_run,
        checkpoint,
        expected_context=prepared.context,
        checkpoint_relative_path="checkpoints/last.pt",
    )
    if restored is None:
        raise FileNotFoundError("no checksum-valid compatible Drive snapshot")
    for record in restored.files:
        if record.path in {
            "checkpoints/last.pt",
            "checkpoints/last.pt.sha256",
        }:
            continue
        relative = PurePosixPath(record.path)
        source = restored.snapshot_dir.joinpath(*relative.parts)
        target = prepared.local_run.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        try:
            shutil.copyfile(source, temporary)
            if (
                temporary.stat().st_size != record.bytes
                or _sha256_file(temporary) != record.sha256
            ):
                raise ValueError(f"restored file failed checksum: {record.path}")
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    return restored


def _enrich(
    record: Mapping[str, Any],
    trainer: Trainer,
    *,
    seconds: float,
    peak_vram: float,
    epoch_skips: int,
) -> dict[str, Any]:
    result = _jsonable(record)
    result.update(
        {
            "global_step": trainer.global_step,
            "amp_skip_count": trainer.amp_skip_count,
            "epoch_amp_skips": epoch_skips,
            "epoch_seconds": seconds,
            "peak_vram_gib": peak_vram,
            "lr": float(trainer.optimizer.param_groups[0]["lr"]),
        }
    )
    return result


def _finite_scalars(record: Mapping[str, Any]) -> bool:
    found = False
    for split in ("train", "val"):
        section = record.get(split)
        if not isinstance(section, Mapping):
            continue
        for value in section.values():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            found = True
            if not math.isfinite(float(value)):
                return False
    return found


def approval_token(
    context: SnapshotContext,
    checkpoint_sha256: str,
    report_sha256: str,
) -> str:
    return canonical_json_sha256(
        {
            "kind": APPROVAL_KIND,
            "context": context.as_dict(),
            "epoch_completed": 0,
            "checkpoint_sha256": checkpoint_sha256,
            "pilot_report_sha256": report_sha256,
        }
    )


def _gate_path(prepared: Prepared) -> Path:
    return prepared.drive_run / "pilot_gate.json"


def _load_gate(prepared: Prepared) -> dict[str, Any] | None:
    path = _gate_path(prepared)
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("pilot_gate.json must contain an object")
    if value.get("context") != prepared.context.as_dict():
        raise ValueError("pilot gate context mismatch")
    return value


def _publish_gate(prepared: Prepared, gate: Mapping[str, Any]) -> None:
    prepared.drive_run.mkdir(parents=True, exist_ok=True)
    _atomic_json(prepared.local_run / "pilot_gate.json", dict(gate))
    _atomic_json(_gate_path(prepared), dict(gate))


def _load_cached_inspection(prepared: Prepared) -> dict[str, Any] | None:
    """Reuse Cell 3's context-bound audit instead of printing it again."""

    path = (
        prepared.local_root
        / "inspections"
        / f"{prepared.config['run_id']}.json"
    )
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        print("Cached inspection is unreadable; rebuilding:", path)
        return None
    required = {
        "data", "model_config", "pretrained", "parameters", "schedule"
    }
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != 1
        or value.get("protocol_id") != PROTOCOL_ID
        or value.get("context") != prepared.context.as_dict()
        or not required.issubset(value)
    ):
        print("Cached inspection is stale/incomplete; rebuilding:", path)
        return None
    print("Reusing context-matched inspection from Cell 3:", path)
    return value


def _duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "--:--"
    rounded = int(round(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def _progress_bar(fraction: float, width: int = 18) -> str:
    checked = min(1.0, max(0.0, fraction))
    filled = min(width, int(checked * width))
    return "█" * filled + "░" * (width - filled)


class StageProgressReporter:
    """Newline-safe YOLO-style stage progress for Colab cells and ``tail -F``."""

    def __init__(
        self,
        local_root: Path,
        *,
        interval_seconds: float = 3.0,
        clock: Any = time.monotonic,
    ) -> None:
        if interval_seconds < 0:
            raise ValueError("interval_seconds must be non-negative")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self.local_root = local_root
        self.interval_seconds = interval_seconds
        self.clock = clock
        self.wall_started = 0.0
        self.eta_started = 0.0
        self.last_rendered = 0.0
        self.phase_started = 0.0
        self.phase = ""
        self.phase_initial_bytes = 0
        self.new_ready_start = 0
        self.disabled = False

    def _print(self, line: str) -> None:
        if self.disabled:
            return
        try:
            print(line, flush=True)
        except BrokenPipeError:
            self.disabled = True

    def _render(self, payload: Mapping[str, Any], *, force: bool = False) -> None:
        if self.disabled:
            return
        now = float(self.clock())
        if not force and now - self.last_rendered < self.interval_seconds:
            return
        self.last_rendered = now
        total_records = int(payload["total_records"])
        ready_records = int(payload["ready_records"])
        total_bytes = int(payload["total_bytes"])
        ready_bytes = int(payload["ready_bytes"])
        fraction = ready_bytes / total_bytes if total_bytes else 1.0
        elapsed = max(0.0, now - self.eta_started)
        newly_ready = max(0, ready_bytes - self.new_ready_start)
        rate = newly_ready / elapsed if elapsed > 0 and newly_ready > 0 else 0.0
        remaining_bytes = max(0, total_bytes - ready_bytes)
        eta = 0.0 if remaining_bytes == 0 else (
            remaining_bytes / rate if rate > 0 else None
        )
        record = payload.get("record")
        current = "-"
        if record is not None:
            current = f"{record.session_id[:12]}/{record.sensor.removeprefix('camera_')}"
        phase = str(payload.get("phase") or payload.get("status") or "ready")
        phase_text = phase
        event = payload.get("event")
        if event in {"resume_check", "final_verify"}:
            index = int(payload["index"])
            scan_elapsed = max(0.0, now - self.phase_started)
            scan_speed = index / scan_elapsed if index > 0 and scan_elapsed > 0 else 0.0
            eta = (
                (total_records - index) / scan_speed
                if scan_speed > 0 and index < total_records
                else 0.0 if index >= total_records else None
            )
            phase_text = f"{phase} {index}/{total_records}"
        completed = payload.get("bytes_completed")
        phase_total = payload.get("bytes_total")
        phase_speed = "--"
        if isinstance(completed, int) and isinstance(phase_total, int) and phase_total > 0:
            phase_fraction = min(1.0, completed / phase_total)
            phase_text = f"{phase} {phase_fraction * 100:5.1f}%"
            phase_elapsed = now - self.phase_started
            phase_delta = max(0, completed - self.phase_initial_bytes)
            if phase_elapsed > 0.05 and phase_delta > 0:
                phase_speed = f"{phase_delta / phase_elapsed / 1024**2:6.1f}MB/s"
        free = shutil.disk_usage(self.local_root).free / 1024**3
        self._print(
            f"{payload.get('purpose', 'stage'):14s} "
            f"{ready_records:3d}/{total_records:<3d} "
            f"[{_progress_bar(fraction)}] {fraction * 100:5.1f}% "
            f"{ready_bytes / 1024**3:6.1f}/{total_bytes / 1024**3:6.1f}G | "
            f"{current:19.19s} | {phase_text:16.16s} | "
            f"{phase_speed:10s} | ETA {_duration(eta):>7s} | free {free:5.1f}G",
        )

    def __call__(self, payload: Mapping[str, Any]) -> None:
        event = payload.get("event")
        now = float(self.clock())
        if event == "stage_start":
            self.wall_started = now
            self.eta_started = now
            self.phase_started = now
            self.last_rendered = 0.0
            self.new_ready_start = int(payload["ready_bytes"])
            self._print(
                "\nStage          Shards [overall progress]       Ready GiB | "
                "Current             | Phase            | Speed      | ETA     | SSD",
            )
            self._render(payload, force=True)
        elif event == "final_verify_start":
            self.phase_started = now
            self._render(payload, force=True)
        elif event in {"resume_check", "final_verify"}:
            index = int(payload["index"])
            if index == 1 or index % 10 == 0 or index == int(payload["total_records"]):
                self._render(payload, force=True)
        elif event == "resume_end":
            # Cached bytes were completed in a previous process and must not
            # inflate this invocation's speed or make its ETA optimistic.
            self.eta_started = now
            self.phase_started = now
            self.new_ready_start = int(payload["ready_bytes"])
            self._render({**dict(payload), "phase": "resume-ready"}, force=True)
        elif event == "record_start":
            self.phase_started = now
            self.phase = str(payload.get("status", "start"))
            self.phase_initial_bytes = 0
            if payload.get("status") != "cached":
                self._render(payload, force=True)
        elif event == "record_progress":
            phase = str(payload.get("phase", "work"))
            if phase != self.phase:
                self.phase = phase
                self.phase_started = now
                self.phase_initial_bytes = int(payload.get("bytes_completed", 0))
                self._render(payload, force=True)
            else:
                self._render(payload)
        elif event == "record_end":
            index = int(payload["index"])
            cached = payload.get("status") == "cached"
            if not cached or index % 10 == 0 or index == int(payload["total_records"]):
                self._render(
                    {**dict(payload), "phase": "cached" if cached else "done"},
                    force=True,
                )
        elif event == "stage_end":
            self._render({**dict(payload), "phase": "complete"}, force=True)
            self._print(f"Stage complete in {_duration(now - self.wall_started)}")


def _print_stage_plan(title: str, plan: Mapping[str, Any]) -> None:
    print(f"\n========== {title} ==========")
    print(
        f"pending={plan['pending_records']} | "
        f"archives={int(plan['archive_bytes']) / 1024**3:.2f} GiB | "
        f"exact selected files={int(plan['exact_source_bytes']) / 1024**3:.2f} GiB "
        f"({plan['source_bytes_records']} records) | "
        f"capacity estimate={int(plan['estimated_unpacked_bytes']) / 1024**3:.2f} GiB"
    )
    print(
        f"required free={int(plan['required_free_bytes']) / 1024**3:.2f} GiB | "
        f"available={int(plan['available_free_bytes']) / 1024**3:.2f} GiB"
    )


def stage_official_train(filename: str | os.PathLike[str]) -> dict[str, Any]:
    """Copy, verify, and extract official-train once for all later epochs."""

    prepared = prepare(filename)
    plan = prepared.train_stage.disk_plan()
    _print_stage_plan("STAGE OFFICIAL-TRAIN TO LOCAL SSD", plan)
    result = prepared.train_stage.prepare_all(
        on_event=StageProgressReporter(prepared.train_stage.local_root)
    )
    if result.get("complete") is not True:
        raise RuntimeError("official-train SSD stage did not complete")
    print(
        "\nTRAIN STAGE READY | "
        f"{result['completed_count']}/{result['record_count']} shards | "
        f"{int(result['completed_extracted_bytes']) / 1024**3:.2f} GiB | "
        f"{prepared.train_stage.local_root}"
    )
    return result


def _official_test_stage(prepared: Prepared) -> LocalArchiveStage:
    staging = prepared.config["data"].get("local_staging", {})
    stage_cache_id = staging.get("cache_id", prepared.config["run_id"])
    return LocalArchiveStage(
        prepared.split.official_test_records,
        local_root=prepared.local_root
        / "stages"
        / stage_cache_id
        / "official_test",
        purpose="official_test",
        expansion_factor=float(staging.get("expansion_factor", 1.05)),
        reserve_bytes=math.ceil(float(staging.get("reserve_gib", 4.0)) * 1024**3),
    )


def stage_official_test(filename: str | os.PathLike[str]) -> dict[str, Any]:
    """Stage the untouched holdout only after a checksum-valid full campaign."""

    prepared = prepare(filename)
    restored = _restore(prepared)
    status_path = prepared.local_run / "run_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    expected_epoch = int(prepared.config["train"]["epochs"]) - 1
    if (
        not isinstance(status, dict)
        or status.get("status") != "complete"
        or status.get("epoch_completed") != expected_epoch
        or restored.epoch_completed != expected_epoch
        or status.get("official_test_used") is not False
    ):
        raise PermissionError(
            "official-test may be staged only after the full checksum-valid "
            "training campaign is complete"
        )

    # The full campaign no longer needs official-train pixels. This is an
    # ownership-checked deletion under /content; Drive archives are untouched.
    prepared.train_stage.cleanup()
    test_stage = _official_test_stage(prepared)
    plan = test_stage.disk_plan()
    _print_stage_plan("STAGE OFFICIAL-TEST TO LOCAL SSD", plan)
    result = test_stage.prepare_all(
        on_event=StageProgressReporter(test_stage.local_root)
    )
    if result.get("complete") is not True:
        raise RuntimeError("official-test SSD stage did not complete")
    print(
        "\nTEST STAGE READY (NOT EVALUATED) | "
        f"{result['completed_count']}/{result['record_count']} shards | "
        f"{int(result['completed_extracted_bytes']) / 1024**3:.2f} GiB | "
        f"{test_stage.local_root}"
    )
    return result


def run_pilot(filename: str | os.PathLike[str]) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("pilot requires an NVIDIA CUDA runtime")
    prepared = prepare(filename)
    existing = _load_gate(prepared)
    if existing is not None:
        restored = _restore(prepared)
        if restored.epoch_completed < 0:
            raise ValueError("pilot gate has no epoch snapshot")
        print("Pilot đã có và snapshot hợp lệ; không train lại.")
        print(json.dumps(existing, indent=2, ensure_ascii=False))
        return existing
    if (prepared.local_run / "checkpoints" / "last.pt").exists():
        raise FileExistsError(
            "unpublished local checkpoint exists; use a new RUN_ID"
        )

    prepared.train_stage.require_complete()

    inspection = _load_cached_inspection(prepared)
    if inspection is None:
        inspection = inspect_campaign(filename)
    _write_metadata(prepared, inspection)
    preflight = disposable_preflight(prepared)
    _atomic_json(prepared.local_run / "preflight.json", preflight)
    # The disposable model consumed RNG and updated its own BN buffers. Reset
    # before constructing the clean production model for campaign epoch zero.
    _seed_everything(int(prepared.config["train"]["seed"]))
    objects = build_train_objects(
        prepared,
        amp_initial_scale=float(preflight["amp_stable_scale"]),
    )
    _print_execution_plan(prepared, objects, start_epoch=0)
    prepared.train_loader.set_epoch(0)
    prepared.val_loader.set_epoch(0)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    skips_before = objects.trainer.amp_skip_count
    started = time.perf_counter()
    result = objects.trainer.fit(
        prepared.train_loader,
        prepared.val_loader,
        stop_after_epoch=1,
    )
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated() / 1024**3
    if len(result) != 1 or result[0]["epoch"] != 0:
        raise RuntimeError("pilot did not stop after exactly epoch one")
    epoch_skips = objects.trainer.amp_skip_count - skips_before
    record = _enrich(
        result[0],
        objects.trainer,
        seconds=seconds,
        peak_vram=peak,
        epoch_skips=epoch_skips,
    )
    print(
        f"[epoch] 1/{objects.config.epochs} complete | wall={_duration(seconds)} | "
        f"peak_vram={peak:.2f}G | amp_skips={epoch_skips} | "
        f"projected_remaining≈{_duration(seconds * (objects.config.epochs - 1))}",
        flush=True,
    )
    before = _state_sha256(objects.model)
    state = objects.trainer.resume(str(objects.manager.last_path))
    checkpoint_roundtrip = (
        before == _state_sha256(objects.model) and state.next_epoch == 1
    )
    _write_reports(prepared, [record], record)
    attempted = math.ceil(
        len(prepared.train_loader) / objects.config.grad_accum_steps
    )
    val = record.get("val", {})
    reasons = []
    if not _finite_scalars(record):
        reasons.append("loss_or_metric_non_finite")
    if not preflight["optimizer_stepped"]:
        reasons.append("preflight_optimizer_did_not_step")
    if objects.trainer.global_step + objects.trainer.amp_skip_count != attempted:
        reasons.append("optimizer_attempt_count_mismatch")
    if epoch_skips:
        reasons.append("amp_skip_in_epoch_1")
    if peak > float(prepared.config["train"]["max_peak_vram_gib"]):
        reasons.append("peak_vram_budget_exceeded")
    if val.get("detection/num_images") != prepared.val_loader.sample_count:
        reasons.append("validation_image_count_mismatch")
    if int(val.get("segmentation/num_pixels", 0)) <= 0:
        reasons.append("segmentation_valid_pixels_missing")
    if int(val.get("depth/num_pixels", 0)) <= 0:
        reasons.append("depth_valid_pixels_missing")
    if not checkpoint_roundtrip:
        reasons.append("checkpoint_roundtrip_failed")
    if not (objects.manager.directory / "best.pt").is_file():
        reasons.append("best_checkpoint_missing")

    report = {
        "schema_version": 1,
        "epoch_completed": 0,
        "display_epoch": 1,
        "context": prepared.context.as_dict(),
        "preflight": preflight,
        "epoch": record,
        "attempted_updates": attempted,
        "checkpoint_roundtrip": checkpoint_roundtrip,
        "gate_reasons": reasons,
        "official_test_used": False,
        "estimated_remaining_hours": seconds
        * (objects.config.epochs - 1)
        / 3600.0,
    }
    _atomic_json(prepared.local_run / "pilot_report.json", report)
    _atomic_json(
        prepared.local_run / "run_status.json",
        {
            "schema_version": 1,
            "status": "pilot_pass" if not reasons else "pilot_fail",
            "next_epoch": 1,
            "official_test_used": False,
        },
    )
    objects.logger.close()
    snapshot = publish_epoch_snapshot(
        prepared.local_run,
        prepared.drive_run,
        epoch_completed=0,
        context=prepared.context,
        checkpoint_relative_path="checkpoints/last.pt",
        files=_snapshot_files(prepared.local_run),
    )
    report_sha = canonical_json_sha256(report)
    token = (
        approval_token(prepared.context, snapshot.checkpoint_sha256, report_sha)
        if not reasons
        else None
    )
    gate = {
        "schema_version": 1,
        "status": "pass" if not reasons else "fail",
        "approval_kind": APPROVAL_KIND,
        "approval_token": token,
        "context": prepared.context.as_dict(),
        "checkpoint_sha256": snapshot.checkpoint_sha256,
        "pilot_report_sha256": report_sha,
        "snapshot": str(snapshot.snapshot_dir),
        "gate_reasons": reasons,
        "metrics": record["val"],
        "elapsed_minutes": seconds / 60.0,
        "peak_vram_gib": peak,
        "amp_skips_epoch_1": epoch_skips,
        "next_epoch": 1,
        "official_test_used": False,
    }
    _publish_gate(prepared, gate)
    print("\n========== PILOT EPOCH 1 ==========")
    print(json.dumps(_jsonable(gate), indent=2, ensure_ascii=False))
    if reasons:
        raise RuntimeError("pilot gate failed: " + ", ".join(reasons))
    return gate


def _verify_approval(prepared: Prepared, supplied: str) -> dict[str, Any]:
    gate = _load_gate(prepared)
    if gate is None or gate.get("status") != "pass":
        raise PermissionError("a passing pilot gate is required")
    expected = approval_token(
        prepared.context,
        str(gate["checkpoint_sha256"]),
        str(gate["pilot_report_sha256"]),
    )
    if gate.get("approval_token") != expected or supplied != expected:
        raise PermissionError("approval token is invalid or stale")
    return gate


def _load_history(run: Path) -> list[dict[str, Any]]:
    path = run / "history.json"
    if not path.exists():
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("history.json must contain a list")
    return value


def run_main(filename: str | os.PathLike[str], supplied_token: str) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("main training requires an NVIDIA CUDA runtime")
    prepared = prepare(filename)
    prepared.train_stage.require_complete()
    _verify_approval(prepared, supplied_token)
    restored = _restore(prepared)
    objects = build_train_objects(prepared)
    state = objects.trainer.resume(str(restored.local_checkpoint_path))
    if state.next_epoch != restored.epoch_completed + 1:
        raise ValueError("snapshot and checkpoint epoch disagree")
    history = _load_history(prepared.local_run)
    if history and history[-1]["epoch"] + 1 != state.next_epoch:
        raise ValueError("history and checkpoint epoch disagree")
    _print_execution_plan(prepared, objects, start_epoch=state.next_epoch)

    while objects.trainer.start_epoch < objects.config.epochs:
        epoch = objects.trainer.start_epoch
        prepared.train_loader.set_epoch(epoch)
        prepared.val_loader.set_epoch(epoch)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        skips_before = objects.trainer.amp_skip_count
        started = time.perf_counter()
        result = objects.trainer.fit(
            prepared.train_loader,
            prepared.val_loader,
            stop_after_epoch=epoch + 1,
        )
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        peak = torch.cuda.max_memory_allocated() / 1024**3
        if len(result) != 1 or result[0]["epoch"] != epoch:
            raise RuntimeError("campaign epoch boundary was not respected")
        record = _enrich(
            result[0],
            objects.trainer,
            seconds=seconds,
            peak_vram=peak,
            epoch_skips=objects.trainer.amp_skip_count - skips_before,
        )
        if not _finite_scalars(record):
            raise FloatingPointError(f"epoch {epoch + 1} is non-finite")
        remaining_epochs = objects.config.epochs - epoch - 1
        print(
            f"[epoch] {epoch + 1}/{objects.config.epochs} complete | "
            f"wall={_duration(seconds)} | peak_vram={peak:.2f}G | "
            f"amp_skips={record['epoch_amp_skips']} | "
            f"projected_remaining≈{_duration(seconds * remaining_epochs)}",
            flush=True,
        )
        history.append(record)
        _write_reports(prepared, history, record)
        _atomic_json(
            prepared.local_run / "run_status.json",
            {
                "schema_version": 1,
                "status": (
                    "complete"
                    if epoch + 1 == objects.config.epochs
                    else "training"
                ),
                "epoch_completed": epoch,
                "next_epoch": epoch + 1,
                "global_step": objects.trainer.global_step,
                "amp_skip_count": objects.trainer.amp_skip_count,
                "best_metrics": objects.trainer.best_metrics,
                "official_test_used": False,
            },
        )
        snapshot = publish_epoch_snapshot(
            prepared.local_run,
            prepared.drive_run,
            epoch_completed=epoch,
            context=prepared.context,
            checkpoint_relative_path="checkpoints/last.pt",
            files=_snapshot_files(prepared.local_run),
        )
        print(
            f"[snapshot] epoch {epoch + 1}/{objects.config.epochs} | "
            f"{snapshot.snapshot_dir} | {snapshot.checkpoint_sha256}"
        )
    objects.logger.close()
    prepared.train_stage.cleanup()
    print("\nTRAINING COMPLETE")
    print("Drive run:", prepared.drive_run)
    print("Best:", objects.trainer.best_metrics)
    print("Official-train local SSD stage đã được xoá; Drive archives còn nguyên.")
    print("Official-test vẫn chưa được dùng.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "stage-train", "pilot", "stage-test"):
        child = commands.add_parser(name)
        child.add_argument("--config", required=True)
    train = commands.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--approval-token", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "inspect":
        inspect_campaign(args.config)
    elif args.command == "stage-train":
        stage_official_train(args.config)
    elif args.command == "pilot":
        run_pilot(args.config)
    elif args.command == "train":
        run_main(args.config, args.approval_token)
    elif args.command == "stage-test":
        stage_official_test(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
