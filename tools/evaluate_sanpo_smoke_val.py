"""Evaluate a SANPO smoke run's selected checkpoint on its held-out val session.

Use this in the still-live Colab namespace after pulling the latest repository::

    !git -C /content/Replite pull --ff-only
    %run -i /content/Replite/tools/evaluate_sanpo_smoke_val.py

This performs inference only. It constructs a separate model, loads ``best.pt``
strictly, evaluates only ``val_loader``, restores all RNG state, and publishes
a versioned checksummed evaluation bundle without rewriting the base run.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import SequentialSampler
from tqdm.auto import tqdm

import replite.training.metrics as _metrics_module
from replite.data import (
    SANPO_DETECTION_CLASS_NAMES,
    SANPO_SEGMENTATION_CLASS_NAMES,
    SANPO_SEGMENTATION_IGNORE_INDEX,
)


DETECTION_SCORE_THRESHOLD = 0.001
DETECTION_NMS_IOU_THRESHOLD = 0.6
DETECTION_MAX_DETECTIONS = 300
EXPECTED_VAL_SAMPLES = 73

_REQUIRED_GLOBALS = {
    "DEPTH_MAX_METRES",
    "DEPTH_MIN_METRES",
    "DRIVE_RUN_DIR",
    "IMAGE_HEIGHT",
    "IMAGE_WIDTH",
    "LOCAL_RUN_DIR",
    "REPO_DIR",
    "RUN_ID",
    "SOURCE_COMMIT",
    "create_replite_model",
    "model_config",
    "move_to_device",
    "trainer",
    "val_dataset",
    "val_loader",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_sidecar(path: Path) -> str:
    sidecar = path.with_name(path.name + ".sha256")
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    if len(fields) != 2 or fields[1] != path.name:
        raise RuntimeError(f"Invalid SHA-256 sidecar: {sidecar}")
    actual = _sha256_file(path)
    if fields[0] != actual:
        raise RuntimeError(f"SHA-256 mismatch: {path}")
    return actual


def _jsonable(value: object) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _json_text(value: object) -> str:
    return json.dumps(
        _jsonable(value), indent=2, sort_keys=True, allow_nan=False
    )


def _manifest_records(manifest: object, *, run_id: str) -> dict[str, dict[str, Any]]:
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise RuntimeError("Unsupported base artifact manifest")
    if manifest.get("run_id") != run_id:
        raise RuntimeError("Base artifact manifest has the wrong run_id")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise RuntimeError("Base artifact manifest is missing artifacts")
    records: dict[str, dict[str, Any]] = {}
    for raw in artifacts:
        if not isinstance(raw, dict):
            raise RuntimeError("Base artifact manifest contains a non-object record")
        relative = raw.get("path")
        posix = PurePosixPath(relative) if isinstance(relative, str) else None
        if (
            posix is None
            or not relative
            or posix.is_absolute()
            or any(part in {"", ".", ".."} for part in posix.parts)
            or relative in records
        ):
            raise RuntimeError("Base artifact manifest contains an unsafe/duplicate path")
        size, sha256 = raw.get("bytes"), raw.get("sha256")
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            raise RuntimeError(f"Invalid base artifact record: {relative}")
        records[relative] = raw
    return records


def _verify_manifest_record(
    root: Path, relative: str, records: dict[str, dict[str, Any]]
) -> Path:
    if relative not in records:
        raise RuntimeError(f"Base manifest does not list required artifact: {relative}")
    path = root.joinpath(*PurePosixPath(relative).parts)
    record = records[relative]
    if not path.is_file() or path.stat().st_size != record["bytes"]:
        raise RuntimeError(f"Base artifact size mismatch: {path}")
    if _sha256_file(path) != record["sha256"]:
        raise RuntimeError(f"Base artifact SHA-256 mismatch: {path}")
    return path


def _model_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


_missing = sorted(name for name in _REQUIRED_GLOBALS if name not in globals())
if _missing:
    raise RuntimeError(
        "Evaluator phải chạy bằng %run -i trong runtime SANPO hiện tại; thiếu: "
        + ", ".join(_missing)
    )
if not torch.cuda.is_available():
    raise RuntimeError("Val evaluator requires the live CUDA runtime")
if trainer.device.type != "cuda":
    raise RuntimeError("Live trainer is not on CUDA")

local_run_dir = Path(LOCAL_RUN_DIR).resolve()
drive_run_dir = Path(DRIVE_RUN_DIR).resolve()
repo_dir = Path(REPO_DIR).resolve()
if RUN_ID != local_run_dir.name or RUN_ID != drive_run_dir.name:
    raise RuntimeError("RUN_ID does not match the local and Drive run paths")
if not local_run_dir.is_dir() or not drive_run_dir.is_dir():
    raise RuntimeError("Base smoke run is missing locally or on Drive")
if len(val_dataset) != EXPECTED_VAL_SAMPLES:
    raise RuntimeError(f"Expected 73 validation samples, got {len(val_dataset)}")
if val_dataset.info.official_split != "train":
    raise RuntimeError("Pilot val must be the held-out official-train session")
if val_loader.dataset is not val_dataset:
    raise RuntimeError("val_loader does not wrap the live val_dataset")
if val_loader.drop_last or not isinstance(val_loader.sampler, SequentialSampler):
    raise RuntimeError("Validation loader must be sequential with drop_last=False")

local_manifest_path = local_run_dir / "artifact_manifest.json"
drive_manifest_path = drive_run_dir / "artifact_manifest.json"
base_manifest_sha256 = _sha256_file(local_manifest_path)
if _sha256_file(drive_manifest_path) != base_manifest_sha256:
    raise RuntimeError("Local and Drive base artifact manifests disagree")
base_manifest = json.loads(drive_manifest_path.read_text(encoding="utf-8"))
base_records = _manifest_records(base_manifest, run_id=RUN_ID)
for required_artifact in (
    "source_commit.txt",
    "resolved_config.json",
    "run_summary.json",
    "checkpoints/best.pt.sha256",
):
    _verify_manifest_record(local_run_dir, required_artifact, base_records)
    _verify_manifest_record(drive_run_dir, required_artifact, base_records)

training_source_path = local_run_dir / "source_commit.txt"
if training_source_path.read_text(encoding="utf-8").strip() != SOURCE_COMMIT:
    raise RuntimeError("Live training source commit disagrees with the base run")
base_config = json.loads(
    (local_run_dir / "resolved_config.json").read_text(encoding="utf-8")
)
run_summary = json.loads(
    (local_run_dir / "run_summary.json").read_text(encoding="utf-8")
)
if base_config.get("run_id") != RUN_ID or base_config.get("source_commit") != SOURCE_COMMIT:
    raise RuntimeError("Resolved config has the wrong run/source identity")
if run_summary.get("status") not in {"smoke_pass", "smoke_pass_amp_recovered"}:
    raise RuntimeError("Base smoke run did not finish with a passing status")
base_epochs = run_summary.get("epochs")
if isinstance(base_epochs, bool) or not isinstance(base_epochs, int) or not 2 <= base_epochs <= 5:
    raise RuntimeError("Base run has an invalid smoke epoch count")
if base_config["split_protocol"]["validation_session"] != val_dataset.info.session_id:
    raise RuntimeError("Live validation session disagrees with resolved_config.json")
if base_config["split_protocol"]["train_session"] == val_dataset.info.session_id:
    raise RuntimeError("Training and validation sessions must be disjoint")
if tuple(base_config["image_size_hw"]) != tuple(val_dataset.image_size):
    raise RuntimeError("Validation image size disagrees with resolved_config.json")
if tuple(base_config["depth_range_metres"]) != (
    val_dataset.depth_min,
    val_dataset.depth_max,
):
    raise RuntimeError("Validation depth range disagrees with resolved_config.json")
if not val_dataset.normalize:
    raise RuntimeError("Validation dataset must use the training normalization")
if val_loader.batch_size != base_config["batch_size"]:
    raise RuntimeError("Validation batch size disagrees with resolved_config.json")
if bool(base_config["trainer_config"]["amp"]) != bool(trainer.amp_enabled):
    raise RuntimeError("Validation AMP mode disagrees with resolved_config.json")
if base_config["trainer_config"]["amp_dtype"] != trainer.config.amp_dtype:
    raise RuntimeError("Validation AMP dtype disagrees with resolved_config.json")
validation_manifest_sha256 = _sha256_file(val_dataset.info.manifest_path)
validation_frame_ids = [
    int(sample["target_frame"]) for sample in val_dataset.manifest["samples"]
]
if len(validation_frame_ids) != len(set(validation_frame_ids)):
    raise RuntimeError("Validation manifest contains duplicate target frames")
validation_sample_ids_sha256 = hashlib.sha256(
    json.dumps(
        validation_frame_ids, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
).hexdigest()

local_best = local_run_dir / "checkpoints" / "best.pt"
drive_best = drive_run_dir / "checkpoints" / "best.pt"
best_sha256 = _verify_sidecar(local_best)
if _verify_sidecar(drive_best) != best_sha256:
    raise RuntimeError("Local and Drive best checkpoints disagree")
best_record = base_records.get("checkpoints/best.pt")
if (
    best_record is None
    or best_record["sha256"] != best_sha256
    or local_best.stat().st_size != best_record["bytes"]
    or drive_best.stat().st_size != best_record["bytes"]
):
    raise RuntimeError("Best checkpoint disagrees with the base artifact manifest")
checkpoint = torch.load(local_best, map_location="cpu", weights_only=False)
if checkpoint.get("schema_version") != 1 or checkpoint.get(
    "checkpoint_kind"
) != "replite_training_epoch_boundary":
    raise RuntimeError("Unsupported best checkpoint schema/kind")
progress = checkpoint.get("progress")
if not isinstance(progress, dict) or (
    progress.get("batch_in_epoch") != 0
    or progress.get("accumulation_index") != 0
):
    raise RuntimeError("Best checkpoint is not a clean epoch boundary")
best_epoch = int(progress["epoch_completed"]) + 1
if not 1 <= best_epoch <= base_epochs:
    raise RuntimeError("Best checkpoint has an invalid completed epoch")

saved_config = dict(checkpoint["model_metadata"]["config"])
expected_config = model_config.as_dict()
saved_config.pop("pretrained", None)
expected_config.pop("pretrained", None)
if saved_config != expected_config:
    raise RuntimeError("Live model architecture disagrees with best.pt")
selected_val_total = float(checkpoint["best_metrics"]["val/total"])
if not math.isfinite(selected_val_total):
    raise RuntimeError("Best checkpoint has a non-finite selection metric")
summary_best = float(run_summary["best_metrics"]["val/total"])
if summary_best != selected_val_total:
    raise RuntimeError("Run summary and best checkpoint selection metrics disagree")
checkpoint_model_state = checkpoint.get("model")
if not isinstance(checkpoint_model_state, dict):
    raise RuntimeError("Best checkpoint is missing model state")
del checkpoint

evaluator_commit = subprocess.check_output(
    ["git", "-C", str(repo_dir), "rev-parse", "HEAD"], text=True
).strip()
script_sha256 = _sha256_file(Path(__file__).resolve())

metrics_source_path = Path(_metrics_module.__file__).resolve()
try:
    metrics_source_path.relative_to(repo_dir)
except ValueError as exc:
    raise RuntimeError("Reloaded metrics module is outside REPO_DIR") from exc
metrics_source_sha256 = _sha256_file(metrics_source_path)
# Pulling source in a live notebook leaves the old metrics classes cached and
# re-exported from replite.training. Load the current file under an isolated
# package-qualified name instead of importlib.reload(), which would mutate the
# cached module and fracture class identity for subsequent notebook code.
isolated_metrics_name = (
    "replite.training._sanpo_val_metrics_" + metrics_source_sha256[:16]
)
isolated_metrics_spec = importlib.util.spec_from_file_location(
    isolated_metrics_name, metrics_source_path
)
if isolated_metrics_spec is None or isolated_metrics_spec.loader is None:
    raise RuntimeError("Cannot create isolated metrics module spec")
isolated_metrics_module = importlib.util.module_from_spec(isolated_metrics_spec)
sys.modules[isolated_metrics_name] = isolated_metrics_module
try:
    isolated_metrics_spec.loader.exec_module(isolated_metrics_module)
finally:
    sys.modules.pop(isolated_metrics_name, None)
_metrics_module = isolated_metrics_module
ignore_probe = _metrics_module.DetectionMAP(1, iou_thresholds=(0.5,))
ignore_probe.update(
    [
        {
            "boxes": torch.tensor(
                [[20.0, 20.0, 30.0, 30.0], [0.0, 0.0, 10.0, 10.0]]
            ),
            "scores": torch.tensor([0.9, 0.8]),
            "labels": torch.tensor([0, 0], dtype=torch.long),
        }
    ],
    [
        {
            "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
            "labels": torch.tensor([0], dtype=torch.long),
            "ignore_boxes": torch.tensor([[20.0, 20.0, 30.0, 30.0]]),
        }
    ],
)
if ignore_probe.compute()["map50"] != 1.0:
    raise RuntimeError("Reloaded DetectionMAP does not implement ignore-box matching")
del ignore_probe

protocol = {
    "selection_split": "held-out session from SANPO official-train",
    "selection_metric": "minimum val/total",
    "runtime": {
        "device": torch.cuda.get_device_name(0),
        "amp_enabled": bool(trainer.amp_enabled),
        "amp_dtype": trainer.config.amp_dtype,
        "metrics_source_sha256": metrics_source_sha256,
    },
    "detection": {
        "ground_truth": "boxes derived from SANPO panoptic thing components",
        "ap": "101-point interpolation at IoU 0.50:0.05:0.95",
        "score": "sqrt(sigmoid(class_logit) * sigmoid(quality_logit))",
        "score_threshold": DETECTION_SCORE_THRESHOLD,
        "nms": "class-aware pure-Torch NMS",
        "nms_iou_threshold": DETECTION_NMS_IOU_THRESHOLD,
        "pre_nms_topk": 1000,
        "max_detections_per_image": DETECTION_MAX_DETECTIONS,
        "max_detection_policy": (
            "score-order truncation occurs before ignore matching; ignored "
            "detections therefore consume the per-image cap"
        ),
        "ignore_policy": (
            "at each AP IoU threshold: positive match first; otherwise IoU "
            ">= that threshold with class-agnostic small-component "
            "ignore_boxes is neither TP nor FP"
        ),
        "official_coco_metric": False,
    },
    "segmentation": {
        "resolution_hw": [IMAGE_HEIGHT, IMAGE_WIDTH],
        "ignore_index": SANPO_SEGMENTATION_IGNORE_INDEX,
        "miou_average": "classes with nonzero ground-truth/prediction union",
        "confusion_rows": "ground truth",
        "confusion_columns": "prediction",
    },
    "depth": {
        "resolution_hw": [IMAGE_HEIGHT, IMAGE_WIDTH],
        "unit": "metres",
        "valid_target": (
            f"depth_valid & finite(target) & target>{DEPTH_MIN_METRES} "
            f"& target<={DEPTH_MAX_METRES}"
        ),
        "aggregation": "global valid-pixel weighted",
        "prediction_clipping": None,
        "median_scaling": False,
        "delta1": "max(pred/target,target/pred) < 1.25",
    },
}
protocol_sha256 = hashlib.sha256(
    json.dumps(
        protocol, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
).hexdigest()
evaluation_id = (
    f"val_best_e{best_epoch:03d}_{best_sha256[:10]}_"
    f"{script_sha256[:8]}_{protocol_sha256[:8]}"
)
local_evaluation_dir = (
    local_run_dir.parent / "replite_sanpo_val_evaluations" / RUN_ID / evaluation_id
)
drive_evaluations_root = drive_run_dir / "evaluations"
drive_evaluation_dir = drive_evaluations_root / evaluation_id
uploading = drive_evaluation_dir.with_name(drive_evaluation_dir.name + ".uploading")
if drive_evaluation_dir.exists():
    checkpoint_model_state = None
    raise FileExistsError(
        "Evaluation bundle đã tồn tại; xem lại JSON, không chạy/ghi đè: "
        + str(drive_evaluation_dir / "val_metrics_best.json")
    )
for stale_stage in (local_evaluation_dir, uploading):
    if stale_stage.exists():
        shutil.rmtree(stale_stage)

validation_metrics = _metrics_module.MultiTaskMetrics(
    detection=_metrics_module.DetectionMAP(
        len(SANPO_DETECTION_CLASS_NAMES),
        max_detections=DETECTION_MAX_DETECTIONS,
    ),
    segmentation=_metrics_module.SegmentationMetrics(
        len(SANPO_SEGMENTATION_CLASS_NAMES),
        ignore_index=SANPO_SEGMENTATION_IGNORE_INDEX,
    ),
    depth=_metrics_module.DepthMetrics(
        min_depth=DEPTH_MIN_METRES,
        max_depth=DEPTH_MAX_METRES,
    ),
    detection_reg_max=model_config.detection_reg_max,
    detection_score_threshold=DETECTION_SCORE_THRESHOLD,
    detection_nms_iou_threshold=DETECTION_NMS_IOU_THRESHOLD,
)

python_rng = random.getstate()
numpy_rng = np.random.get_state()
torch_rng = torch.get_rng_state()
cuda_rng = torch.cuda.get_rng_state_all()
live_model_before = _model_state_sha256(trainer.model)
live_progress_before = (int(trainer.global_step), int(trainer.amp_skip_count))
evaluation_model = None
inputs = targets = outputs = losses = None
try:
    evaluation_config = replace(model_config, pretrained=False)
    evaluation_model = create_replite_model(evaluation_config)
    evaluation_model.load_state_dict(checkpoint_model_state, strict=True)
    checkpoint_model_state = None
    evaluation_model = evaluation_model.to("cuda").eval()
    validation_metrics.reset()
    loss_sums: dict[str, float] = {}
    batch_count = 0
    expected_segmentation_pixels = 0
    expected_depth_pixels = 0
    torch.cuda.synchronize()
    evaluation_started = time.perf_counter()
    with torch.inference_mode():
        for inputs, targets in tqdm(val_loader, desc="best.pt val metrics", leave=True):
            inputs = move_to_device(inputs, torch.device("cuda"), non_blocking=True)
            targets = move_to_device(targets, torch.device("cuda"), non_blocking=True)
            expected_segmentation_pixels += int(
                (
                    targets["segmentation_valid"]
                    & targets["segmentation"].ne(SANPO_SEGMENTATION_IGNORE_INDEX)
                ).sum().detach().cpu()
            )
            expected_depth_pixels += int(
                (
                    targets["depth_valid"]
                    & torch.isfinite(targets["depth"])
                    & targets["depth"].gt(DEPTH_MIN_METRES)
                    & targets["depth"].le(DEPTH_MAX_METRES)
                ).sum().detach().cpu()
            )
            with torch.autocast(
                device_type="cuda",
                dtype=trainer.amp_dtype,
                enabled=trainer.amp_enabled,
            ):
                outputs = evaluation_model(inputs)
                losses = trainer.criterion(outputs, targets)
            validation_metrics.update(outputs, targets)
            for name, value in losses.items():
                if isinstance(value, torch.Tensor) and value.ndim == 0:
                    loss_sums[name] = loss_sums.get(name, 0.0) + float(
                        value.detach().float().cpu()
                    )
            batch_count += 1
    torch.cuda.synchronize()
    evaluation_seconds = time.perf_counter() - evaluation_started
    val_losses = {name: value / batch_count for name, value in loss_sums.items()}
    raw_metrics = validation_metrics.compute()
finally:
    random.setstate(python_rng)
    np.random.set_state(numpy_rng)
    torch.set_rng_state(torch_rng)
    torch.cuda.set_rng_state_all(cuda_rng)
    checkpoint_model_state = None
    evaluation_model = None
    inputs = targets = outputs = losses = None
    torch.cuda.empty_cache()

live_model_after = _model_state_sha256(trainer.model)
live_progress_after = (int(trainer.global_step), int(trainer.amp_skip_count))
if live_model_after != live_model_before or live_progress_after != live_progress_before:
    raise RuntimeError("Val evaluation mutated the live training state")
if batch_count != len(val_loader):
    raise RuntimeError("Val evaluator did not consume the complete loader")
if raw_metrics["detection/num_images"] != len(val_dataset):
    raise RuntimeError("Detection metric image count is incomplete")
if (
    raw_metrics["segmentation/num_pixels"] != expected_segmentation_pixels
    or raw_metrics["depth/num_pixels"] != expected_depth_pixels
    or expected_segmentation_pixels <= 0
    or expected_depth_pixels <= 0
):
    raise RuntimeError("Dense metric valid-pixel counts are incomplete")

if not math.isclose(
    val_losses["total"], selected_val_total, rel_tol=1e-5, abs_tol=1e-6
):
    raise RuntimeError(
        "Recomputed val/total does not match checkpoint selection metric: "
        f"{val_losses['total']} vs {selected_val_total}"
    )

detection_support = torch.zeros(
    len(SANPO_DETECTION_CLASS_NAMES), dtype=torch.int64
)
ignored_box_count = 0
for target in validation_metrics.detection.targets:
    detection_support += torch.bincount(
        target["labels"], minlength=len(SANPO_DETECTION_CLASS_NAMES)
    )
    ignored_box_count += int(target["ignore_boxes"].shape[0])
detection_ap = raw_metrics["detection/per_class_map"]
detection_rows = [
    {
        "class_id": class_id,
        "class_name": class_name,
        "num_targets": int(detection_support[class_id]),
        "present_in_val_gt": bool(detection_support[class_id] > 0),
        "map50_95": (
            float(detection_ap[class_id]) if class_id in detection_ap else None
        ),
    }
    for class_id, class_name in enumerate(SANPO_DETECTION_CLASS_NAMES)
]

confusion = raw_metrics["segmentation/confusion_matrix"].to(torch.int64)
segmentation_iou = raw_metrics["segmentation/per_class_iou"]
segmentation_present = raw_metrics["segmentation/present_classes"]
segmentation_rows = [
    {
        "class_id": class_id,
        "class_name": class_name,
        "num_ground_truth_pixels": int(confusion[class_id].sum()),
        "num_predicted_pixels": int(confusion[:, class_id].sum()),
        "present_in_val_gt": bool(confusion[class_id].sum() > 0),
        "predicted_in_val": bool(confusion[:, class_id].sum() > 0),
        "present_union": bool(segmentation_present[class_id]),
        "iou": (
            float(segmentation_iou[class_id])
            if segmentation_present[class_id]
            else None
        ),
    }
    for class_id, class_name in enumerate(SANPO_SEGMENTATION_CLASS_NAMES)
]
del validation_metrics
del _metrics_module, isolated_metrics_module, isolated_metrics_spec

summary_rows = [
    {"task": "detection", "metric": "mAP50", "value": raw_metrics["detection/map50"], "unit": "ratio"},
    {"task": "detection", "metric": "mAP50-95", "value": raw_metrics["detection/map50_95"], "unit": "ratio"},
    {"task": "segmentation", "metric": "mIoU", "value": raw_metrics["segmentation/miou"], "unit": "ratio"},
    {"task": "segmentation", "metric": "pixel accuracy", "value": raw_metrics["segmentation/pixel_accuracy"], "unit": "ratio"},
    {"task": "depth", "metric": "AbsRel", "value": raw_metrics["depth/abs_rel"], "unit": "ratio"},
    {"task": "depth", "metric": "RMSE", "value": raw_metrics["depth/rmse"], "unit": "metres"},
    {"task": "depth", "metric": "delta1", "value": raw_metrics["depth/delta1"], "unit": "ratio"},
]

local_evaluation_dir.mkdir(parents=True, exist_ok=False)

evaluation_result = {
    "schema_version": 1,
    "evaluation_id": evaluation_id,
    "run_id": RUN_ID,
    "data_role": "held-out validation session from SANPO official-train",
    "validation_session": val_dataset.info.session_id,
    "validation_sensor": val_dataset.info.sensor,
    "num_validation_samples": len(val_dataset),
    "validation_manifest_sha256": validation_manifest_sha256,
    "validation_sample_ids_sha256": validation_sample_ids_sha256,
    "training_source_commit": SOURCE_COMMIT,
    "evaluator_source_commit": evaluator_commit,
    "evaluator_script_sha256": script_sha256,
    "metrics_source_path": metrics_source_path.relative_to(repo_dir).as_posix(),
    "metrics_source_sha256": metrics_source_sha256,
    "base_artifact_manifest_sha256": base_manifest_sha256,
    "checkpoint": {
        "path_in_base_run": "checkpoints/best.pt",
        "sha256": best_sha256,
        "completed_epoch_one_based": best_epoch,
        "selection_val_total": selected_val_total,
        "recomputed_val_total": val_losses["total"],
    },
    "protocol_sha256": protocol_sha256,
    "protocol": protocol,
    "evaluation_seconds": evaluation_seconds,
    "val_losses": val_losses,
    "overall": {row["metric"]: row["value"] for row in summary_rows},
    "raw_metrics": _jsonable(raw_metrics),
    "detection_per_class": detection_rows,
    "detection_ignored_box_count": ignored_box_count,
    "segmentation_per_class": segmentation_rows,
    "expected_valid_pixels": {
        "segmentation": expected_segmentation_pixels,
        "depth": expected_depth_pixels,
    },
    "live_state_unchanged": True,
    "notes": [
        "Descriptive smoke metrics from one 73-frame session; no uncertainty interval.",
        "Detection boxes are derived from panoptic masks, not official SANPO detection GT.",
        "This is not an official SANPO benchmark result.",
    ],
}
(local_evaluation_dir / "val_metrics_best.json").write_text(
    _json_text(evaluation_result),
    encoding="utf-8",
)
_write_csv(
    local_evaluation_dir / "val_metrics_summary.csv",
    summary_rows,
    ["task", "metric", "value", "unit"],
)
_write_csv(
    local_evaluation_dir / "val_detection_per_class.csv",
    detection_rows,
    ["class_id", "class_name", "num_targets", "present_in_val_gt", "map50_95"],
)
_write_csv(
    local_evaluation_dir / "val_segmentation_per_class.csv",
    segmentation_rows,
    [
        "class_id",
        "class_name",
        "num_ground_truth_pixels",
        "num_predicted_pixels",
        "present_in_val_gt",
        "predicted_in_val",
        "present_union",
        "iou",
    ],
)
confusion_rows = []
for class_id, class_name in enumerate(SANPO_SEGMENTATION_CLASS_NAMES):
    row = {"ground_truth_class_id": class_id, "ground_truth_class_name": class_name}
    row.update({f"pred_{index}": int(value) for index, value in enumerate(confusion[class_id])})
    confusion_rows.append(row)
_write_csv(
    local_evaluation_dir / "val_segmentation_confusion_matrix.csv",
    confusion_rows,
    ["ground_truth_class_id", "ground_truth_class_name"]
    + [f"pred_{index}" for index in range(len(SANPO_SEGMENTATION_CLASS_NAMES))],
)

# Honest provisional dashboard: all rate bars use a zero-to-one baseline;
# incompatible depth units are shown as labeled text rather than dual axes.
fig = plt.figure(figsize=(17, 18), layout="constrained", facecolor="white")
grid = fig.add_gridspec(2, 2, height_ratios=(0.8, 1.4))
rate_ax = fig.add_subplot(grid[0, 0])
depth_ax = fig.add_subplot(grid[0, 1])
detection_ax = fig.add_subplot(grid[1, 0])
segmentation_ax = fig.add_subplot(grid[1, 1])

rate_names = ["mAP50", "mAP50-95", "mIoU", "pixel accuracy", "delta1"]
rate_values = [evaluation_result["overall"][name] for name in rate_names]
rate_colors = ["#0072B2", "#56B4E9", "#009E73", "#CC79A7", "#E69F00"]
rate_ax.bar(range(len(rate_names)), rate_values, color=rate_colors, edgecolor="black")
rate_ax.set_ylim(0.0, 1.0)
rate_ax.set_xticks(range(len(rate_names)), rate_names, rotation=24, ha="right")
rate_ax.set_ylabel("Score (0–1)")
rate_ax.set_title("Overall validation rates")
rate_ax.grid(axis="y", alpha=0.25)
for index, value in enumerate(rate_values):
    rate_ax.text(index, min(value + 0.025, 0.97), f"{value:.3f}", ha="center")

depth_ax.axis("off")
depth_text = (
    "Depth metrics (global valid pixels)\n\n"
    f"AbsRel: {raw_metrics['depth/abs_rel']:.4f}\n"
    f"RMSE: {raw_metrics['depth/rmse']:.3f} m\n"
    f"RMSE-log: {raw_metrics['depth/rmse_log']:.4f}\n"
    f"δ1: {raw_metrics['depth/delta1']:.4f}\n"
    f"δ2: {raw_metrics['depth/delta2']:.4f}\n"
    f"δ3: {raw_metrics['depth/delta3']:.4f}\n"
    f"Valid pixels: {raw_metrics['depth/num_pixels']:,}\n\n"
    "No clipping; no median scaling; target range "
    f"({DEPTH_MIN_METRES:g}, {DEPTH_MAX_METRES:g}] m."
)
depth_ax.text(0.02, 0.98, depth_text, va="top", family="monospace", fontsize=12)

def _per_class_panel(
    ax,
    rows,
    value_key,
    support_key,
    title,
    color,
    *,
    missing_label,
    support_label,
    secondary_support_key=None,
    secondary_support_label=None,
):
    positions = np.arange(len(rows))
    present = [row[value_key] is not None for row in rows]
    values = [row[value_key] if row[value_key] is not None else 0.0 for row in rows]
    bars = ax.barh(positions, values, color=color, edgecolor="black", alpha=0.88)
    for bar, is_present in zip(bars, present):
        if not is_present:
            bar.set_visible(False)
    ax.set_yticks(positions, [row["class_name"] for row in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Score (0–1)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    for position, row in zip(positions, rows):
        if row[value_key] is None:
            ax.text(0.01, position, missing_label, va="center", color="#666666")
        else:
            support = f"{support_label}={row[support_key]:,}"
            if secondary_support_key is not None:
                support += (
                    f", {secondary_support_label}="
                    f"{row[secondary_support_key]:,}"
                )
            ax.text(
                0.98,
                position,
                f"{float(row[value_key]):.3f}  {support}",
                va="center",
                ha="right",
                fontsize=7,
            )

_per_class_panel(
    detection_ax, detection_rows, "map50_95", "num_targets",
    "Detection per-class mAP50–95 (derived boxes)", "#0072B2",
    missing_label="no val GT", support_label="nGT",
)
_per_class_panel(
    segmentation_ax, segmentation_rows, "iou", "num_ground_truth_pixels",
    "Segmentation per-class IoU", "#009E73",
    missing_label="absent from GT and predictions", support_label="nGT",
    secondary_support_key="num_predicted_pixels", secondary_support_label="nPred",
)
fig.suptitle(
    f"RepLite SANPO pilot val — best epoch {best_epoch}\n"
    "1 held-out official-train session, n=73 frames; descriptive smoke metrics, no CI",
    fontsize=15,
)
figure_path = local_evaluation_dir / "val_metrics_best.png"
figure_svg_path = local_evaluation_dir / "val_metrics_best.svg"
fig.savefig(figure_path, dpi=180, facecolor="white")
fig.savefig(figure_svg_path, facecolor="white")
alt_text = (
    f"Four-panel validation dashboard for 73 SANPO pilot frames at best epoch {best_epoch}. "
    f"Overall detection mAP50 is {raw_metrics['detection/map50']:.3f}, mAP50-95 is "
    f"{raw_metrics['detection/map50_95']:.3f}, segmentation mIoU is "
    f"{raw_metrics['segmentation/miou']:.3f}, depth AbsRel is "
    f"{raw_metrics['depth/abs_rel']:.3f}, RMSE is {raw_metrics['depth/rmse']:.3f} metres, "
    f"and delta1 is {raw_metrics['depth/delta1']:.3f}. Per-class detection and "
    "segmentation panels explicitly mark classes absent from the validation session."
)
(local_evaluation_dir / "val_metrics_figure_alt.txt").write_text(
    alt_text + "\n", encoding="utf-8"
)

manifest_path = local_evaluation_dir / "artifact_manifest.json"
artifact_records = []
for path in sorted(item for item in local_evaluation_dir.rglob("*") if item.is_file()):
    if path == manifest_path:
        continue
    artifact_records.append({
        "path": path.relative_to(local_evaluation_dir).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    })
manifest = {
    "schema_version": 1,
    "evaluation_id": evaluation_id,
    "base_run_id": RUN_ID,
    "base_artifact_manifest_sha256": base_manifest_sha256,
    "artifacts": artifact_records,
}
manifest_path.write_text(
    _json_text(manifest), encoding="utf-8"
)
manifest_size = manifest_path.stat().st_size
manifest_sha256 = _sha256_file(manifest_path)

drive_evaluations_root.mkdir(parents=True, exist_ok=True)
try:
    shutil.copytree(local_evaluation_dir, uploading)
    for record in artifact_records:
        copied = uploading / record["path"]
        if copied.stat().st_size != record["bytes"]:
            raise RuntimeError(f"Drive size mismatch: {record['path']}")
        if _sha256_file(copied) != record["sha256"]:
            raise RuntimeError(f"Drive SHA-256 mismatch: {record['path']}")
    copied_manifest = uploading / manifest_path.name
    if copied_manifest.stat().st_size != manifest_size:
        raise RuntimeError("Drive evaluation manifest size mismatch")
    if _sha256_file(copied_manifest) != manifest_sha256:
        raise RuntimeError("Drive evaluation manifest SHA-256 mismatch")
    os.replace(uploading, drive_evaluation_dir)
except Exception:
    if uploading.exists():
        shutil.rmtree(uploading, ignore_errors=True)
    raise

VAL_METRICS_RESULT = evaluation_result
print("\n========== BEST CHECKPOINT — HELD-OUT VAL ==========")
print(f"Best epoch       : {best_epoch}")
print(f"Validation frames: {len(val_dataset)} ({val_dataset.info.session_id})")
print(f"Detection mAP50  : {raw_metrics['detection/map50']:.6f}")
print(f"Detection mAP50-95: {raw_metrics['detection/map50_95']:.6f}")
print(f"Segmentation mIoU: {raw_metrics['segmentation/miou']:.6f}")
print(f"Depth AbsRel      : {raw_metrics['depth/abs_rel']:.6f}")
print(f"Depth RMSE (m)    : {raw_metrics['depth/rmse']:.6f}")
print(f"Depth delta1      : {raw_metrics['depth/delta1']:.6f}")
print("Official-test không được dùng trong evaluation này.")
print("Evaluation bundle:", drive_evaluation_dir)
print("Full metrics JSON:", drive_evaluation_dir / "val_metrics_best.json")
print("Dashboard:", drive_evaluation_dir / figure_path.name)
plt.show()
plt.close(fig)
del fig, grid, rate_ax, depth_ax, detection_ax, segmentation_ax
