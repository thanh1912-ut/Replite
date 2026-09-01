"""Audit and publish the completed first SANPO smoke run without retraining.

Run this only in the still-live Colab namespace after the original three
epochs completed and the old Cell 7 stopped at its zero-AMP-skip assertion::

    !git -C /content/Replite pull --ff-only
    %run -i /content/Replite/tools/recover_sanpo_smoke_amp.py

Do not rerun setup Cell 1 or training Cell 7 first. This script does not call
``fit``, ``train_epoch``, ``backward``, or an optimizer step. It validates the
live state against atomic checkpoints, proves that the last epoch had no new
AMP skips, performs one inference-only official-test batch, and publishes a
checksummed run bundle.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


EXPECTED_TRAINING_SOURCE_COMMIT = "86a64260bf2b43e5c092e202dfe80dc022132482"
AMP_INITIAL_SCALE = 65536.0  # actual setting used by the original run
MAX_RECOVERABLE_AMP_SKIP_RATE = 0.05

_REQUIRED_GLOBALS = {
    "BATCH_SIZE",
    "BUNDLES",
    "DEPTH_MAX_METRES",
    "DEPTH_MIN_METRES",
    "DRIVE_RUN_DIR",
    "DRIVE_RUNS_ROOT",
    "IMAGE_HEIGHT",
    "IMAGE_WIDTH",
    "LOCAL_RUN_DIR",
    "NUM_WORKERS",
    "REPO_DIR",
    "REPO_URL",
    "RUN_ID",
    "SEED",
    "SMOKE_EPOCHS",
    "SOURCE_COMMIT",
    "checkpoint_manager",
    "elapsed_seconds",
    "history",
    "model_config",
    "official_test_dataset",
    "official_test_loader",
    "pretrained_fallback",
    "total_steps",
    "train_dataset",
    "train_loader",
    "trainer",
    "trainer_config",
    "val_dataset",
    "warmup_steps",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_checkpoint(path: Path) -> str:
    path = Path(path)
    sidecar = path.with_name(path.name + ".sha256")
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    if len(fields) != 2 or fields[1] != path.name:
        raise RuntimeError(f"Invalid checkpoint checksum sidecar: {sidecar}")
    actual = _sha256_file(path)
    if actual != fields[0]:
        raise RuntimeError(f"Checkpoint SHA-256 mismatch: {path}")
    return actual


def _payload(path: Path) -> dict[str, Any]:
    _verify_checkpoint(path)
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise RuntimeError(f"Unsupported checkpoint schema: {path}")
    if value.get("checkpoint_kind") != "replite_training_epoch_boundary":
        raise RuntimeError(f"Unsupported checkpoint kind: {path}")
    return value


def _jsonable(value: object) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


_missing = sorted(name for name in _REQUIRED_GLOBALS if name not in globals())
if _missing:
    raise RuntimeError(
        "Recovery phải chạy bằng %run -i trong đúng runtime vừa train; thiếu: "
        + ", ".join(_missing)
    )

if SOURCE_COMMIT != EXPECTED_TRAINING_SOURCE_COMMIT:
    raise RuntimeError(
        "Sai source commit của run cần cứu; không được gán provenance mới cho "
        f"checkpoint cũ: {SOURCE_COMMIT}"
    )
if SMOKE_EPOCHS != 3:
    raise RuntimeError(f"Recovery này chỉ dành cho pilot 3 epoch, got {SMOKE_EPOCHS}")
if not torch.cuda.is_available() or trainer.device.type != "cuda":
    raise RuntimeError("Recovery phải chạy trong chính CUDA runtime đã train")

local_run_dir = Path(LOCAL_RUN_DIR).resolve()
drive_runs_root = Path(DRIVE_RUNS_ROOT).resolve()
drive_run_dir = Path(DRIVE_RUN_DIR).resolve()
checkpoint_dir = Path(checkpoint_manager.directory).resolve()
uploading = drive_run_dir.with_name(drive_run_dir.name + ".uploading")
if checkpoint_dir != local_run_dir / "checkpoints":
    raise RuntimeError("Checkpoint directory does not belong to LOCAL_RUN_DIR")
if RUN_ID != local_run_dir.name or RUN_ID != drive_run_dir.name:
    raise RuntimeError("RUN_ID does not match local/Drive run directories")
if drive_run_dir.parent != drive_runs_root:
    raise RuntimeError("DRIVE_RUN_DIR is outside DRIVE_RUNS_ROOT")
if drive_run_dir.exists() or uploading.exists():
    raise RuntimeError(
        f"Drive target or temporary upload already exists; refusing overwrite: {drive_run_dir}"
    )
if (len(train_dataset), len(val_dataset), len(official_test_dataset)) != (107, 73, 78):
    raise RuntimeError("Live dataset cardinalities differ from the original pilot")

if len(history) != SMOKE_EPOCHS:
    raise RuntimeError(f"Expected {SMOKE_EPOCHS} completed epochs, got {len(history)}")
if [int(record.get("epoch", -1)) for record in history] != list(range(SMOKE_EPOCHS)):
    raise RuntimeError("Smoke history is not the complete ordered epoch range")

scalar_history: list[dict[str, Any]] = []
for record in history:
    compact: dict[str, Any] = {"epoch": int(record["epoch"])}
    for split in ("train", "val"):
        compact[split] = {
            key: float(value)
            for key, value in record[split].items()
            if isinstance(value, (int, float))
        }
        if not compact[split] or not all(
            np.isfinite(tuple(compact[split].values()))
        ):
            raise RuntimeError(f"Non-finite or empty {split} history")
    scalar_history.append(compact)

last_path = Path(checkpoint_manager.last_path)
previous_path = Path(checkpoint_manager.previous_path)
best_path = checkpoint_dir / "best.pt"
last_payload = _payload(last_path)
previous_payload = _payload(previous_path)
_payload(best_path)

if last_payload["progress"]["epoch_completed"] != SMOKE_EPOCHS - 1:
    raise RuntimeError("last.pt is not the final completed smoke epoch")
if previous_payload["progress"]["epoch_completed"] != SMOKE_EPOCHS - 2:
    raise RuntimeError("last.prev.pt is not the penultimate smoke epoch")

final_skips = int(last_payload["extra"].get("amp_skip_count", -1))
previous_skips = int(previous_payload["extra"].get("amp_skip_count", -1))
if not 0 <= previous_skips <= final_skips:
    raise RuntimeError("Checkpoint AMP skip counters are invalid or non-monotonic")
if final_skips != int(trainer.amp_skip_count):
    raise RuntimeError("Live trainer and final checkpoint disagree on AMP skips")
if int(last_payload["progress"]["global_step"]) != int(trainer.global_step):
    raise RuntimeError("Live trainer and final checkpoint disagree on global step")
final_epoch_skips = final_skips - previous_skips

checkpoint_scaler = last_payload.get("scaler")
if not isinstance(checkpoint_scaler, dict) or "scale" not in checkpoint_scaler:
    raise RuntimeError("Final checkpoint does not contain an AMP scaler state")
final_amp_scale = float(trainer.scaler.get_scale())
checkpoint_amp_scale = float(checkpoint_scaler["scale"])
if final_amp_scale != checkpoint_amp_scale:
    raise RuntimeError("Live trainer and final checkpoint disagree on AMP scale")

checkpoint_model = last_payload.get("model")
live_model = trainer.model.state_dict()
if not isinstance(checkpoint_model, dict) or checkpoint_model.keys() != live_model.keys():
    raise RuntimeError("Live model/checkpoint state keys differ")
for name, saved_tensor in checkpoint_model.items():
    live_tensor = live_model[name].detach().cpu()
    if not isinstance(saved_tensor, torch.Tensor) or not torch.equal(live_tensor, saved_tensor):
        raise RuntimeError(f"Live model differs from last.pt at tensor: {name}")

attempted_updates = SMOKE_EPOCHS * math.ceil(
    len(train_loader) / trainer_config.grad_accum_steps
)
if trainer.global_step + final_skips != attempted_updates:
    raise RuntimeError(
        "Successful + skipped optimizer updates do not match attempted updates"
    )
amp_skip_rate = final_skips / attempted_updates
if not (
    0 < final_skips
    and amp_skip_rate <= MAX_RECOVERABLE_AMP_SKIP_RATE
    and final_epoch_skips == 0
    and np.isfinite(final_amp_scale)
    and final_amp_scale >= 1.0
):
    raise RuntimeError(
        "AMP chưa phục hồi; không publish: "
        f"skips={final_skips}/{attempted_updates}, "
        f"final_epoch_skips={final_epoch_skips}, scale={final_amp_scale}"
    )

# The original notebook did not retain exact skip deltas for epochs 0 and 1.
# Atomic penultimate/final checkpoints still prove the final epoch delta.
amp_skips_by_epoch: list[int | None] = [None, None, final_epoch_skips]
amp_epoch_skip_history_complete = False
smoke_status = "smoke_pass_amp_recovered"

# All state/provenance checks above must pass before this single held-out
# inference batch. It cannot update weights, optimizer, scheduler, or scaler.
trainer.model.eval()
test_inputs, _ = next(iter(official_test_loader))
with torch.inference_mode(), trainer._autocast():
    test_outputs = trainer.model(test_inputs.to(trainer.device, non_blocking=True))
if test_outputs.segmentation.shape[-2:] != (IMAGE_HEIGHT, IMAGE_WIDTH):
    raise RuntimeError("Official-test segmentation output shape mismatch")
if test_outputs.depth.shape[-2:] != (IMAGE_HEIGHT, IMAGE_WIDTH):
    raise RuntimeError("Official-test depth output shape mismatch")
del test_inputs, test_outputs

peak_vram_gib = torch.cuda.max_memory_allocated() / 1024**3
recovery_source_commit = subprocess.check_output(
    ["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"], text=True
).strip()
recovery_audit = {
    "schema_version": 1,
    "status": smoke_status,
    "training_source_commit": SOURCE_COMMIT,
    "recovery_source_commit": recovery_source_commit,
    "attempted_updates": attempted_updates,
    "successful_updates": int(trainer.global_step),
    "amp_skip_count": final_skips,
    "amp_skip_rate": amp_skip_rate,
    "penultimate_cumulative_amp_skips": previous_skips,
    "final_epoch_amp_skips": final_epoch_skips,
    "amp_epoch_skip_history_complete": amp_epoch_skip_history_complete,
    "final_amp_scale": final_amp_scale,
    "checkpoint_amp_scale": checkpoint_amp_scale,
    "gate": {
        "max_skip_rate": MAX_RECOVERABLE_AMP_SKIP_RATE,
        "requires_zero_skips_in_final_epoch": True,
        "requires_finite_final_scale_at_least": 1.0,
    },
    "note": (
        "The original notebook used GradScaler initial_scale=65536. "
        f"{final_skips} early updates were skipped while scale backed off; "
        "the final epoch had zero new skips. No training was repeated by "
        "this recovery."
    ),
}
(local_run_dir / "amp_recovery_audit.json").write_text(
    json.dumps(recovery_audit, indent=2, sort_keys=True), encoding="utf-8"
)

resolved = {
    "schema_version": 1,
    "run_id": RUN_ID,
    "source_repository": REPO_URL,
    "source_commit": SOURCE_COMMIT,
    "recovery_source_commit": recovery_source_commit,
    "seed": SEED,
    "model_config": model_config.as_dict(),
    "trainer_config": asdict(trainer_config),
    "optimizer": {
        "name": "AdamW", "lr": 3e-4, "weight_decay": 1e-2,
        "backbone_lr_multiplier": 0.1,
    },
    "scheduler": {
        "name": "warmup_cosine", "total_steps": total_steps,
        "warmup_steps": warmup_steps, "min_lr_ratio": 0.05,
    },
    "image_size_hw": [IMAGE_HEIGHT, IMAGE_WIDTH],
    "batch_size": BATCH_SIZE,
    "num_workers": NUM_WORKERS,
    "depth_range_metres": [DEPTH_MIN_METRES, DEPTH_MAX_METRES],
    "amp_initial_scale": AMP_INITIAL_SCALE,
    "amp_recovery_gate": recovery_audit["gate"],
    "pretrained_random_init_fallback": pretrained_fallback,
    "split_protocol": {
        "train_session": train_dataset.info.session_id,
        "validation_session": val_dataset.info.session_id,
        "official_test_session": official_test_dataset.info.session_id,
        "official_test_role": (
            "one-batch forward-only pipeline QA; never checkpoint selection"
        ),
    },
    "archives": [
        {
            "split": item["entry"]["split"],
            "session_id": item["entry"]["session_id"],
            "sensor": item["entry"]["sensor"],
            "archive_sha256": item["entry"]["archive_sha256"],
            "selection_sha256": item["entry"]["selection_sha256"],
        }
        for item in BUNDLES
    ],
}
(local_run_dir / "resolved_config.json").write_text(
    json.dumps(_jsonable(resolved), indent=2, sort_keys=True), encoding="utf-8"
)
(local_run_dir / "history.json").write_text(
    json.dumps(scalar_history, indent=2, sort_keys=True), encoding="utf-8"
)
(local_run_dir / "run_summary.json").write_text(
    json.dumps(
        {
            "schema_version": 1,
            "status": smoke_status,
            "epochs": SMOKE_EPOCHS,
            "global_step": trainer.global_step,
            "amp_skip_count": trainer.amp_skip_count,
            "amp_skip_rate": amp_skip_rate,
            "amp_skips_by_epoch": amp_skips_by_epoch,
            "amp_epoch_skip_history_complete": amp_epoch_skip_history_complete,
            "final_amp_scale": final_amp_scale,
            "elapsed_seconds": elapsed_seconds,
            "peak_vram_gib": peak_vram_gib,
            "best_metrics": trainer.best_metrics,
        },
        indent=2, sort_keys=True,
    ), encoding="utf-8"
)
(local_run_dir / "source_commit.txt").write_text(
    SOURCE_COMMIT + "\n", encoding="utf-8"
)
(local_run_dir / "recovery_source_commit.txt").write_text(
    recovery_source_commit + "\n", encoding="utf-8"
)

manifest_path = local_run_dir / "artifact_manifest.json"
artifact_records = []
for path in sorted(item for item in local_run_dir.rglob("*") if item.is_file()):
    relative = path.relative_to(local_run_dir).as_posix()
    if relative == manifest_path.name:
        continue
    artifact_records.append({
        "path": relative, "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    })
artifact_manifest = {
    "schema_version": 1,
    "run_id": RUN_ID,
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "artifacts": artifact_records,
}
manifest_path.write_text(
    json.dumps(artifact_manifest, indent=2, sort_keys=True), encoding="utf-8"
)
manifest_bytes = manifest_path.stat().st_size
manifest_sha256 = _sha256_file(manifest_path)

drive_runs_root.mkdir(parents=True, exist_ok=True)
try:
    shutil.copytree(local_run_dir, uploading)
    for record in artifact_records:
        copied = uploading / record["path"]
        if copied.stat().st_size != record["bytes"]:
            raise RuntimeError(f"Drive size mismatch: {record['path']}")
        if _sha256_file(copied) != record["sha256"]:
            raise RuntimeError(f"Drive SHA-256 mismatch: {record['path']}")
    copied_manifest = uploading / manifest_path.name
    if copied_manifest.stat().st_size != manifest_bytes:
        raise RuntimeError("Drive artifact_manifest.json size mismatch")
    if _sha256_file(copied_manifest) != manifest_sha256:
        raise RuntimeError("Drive artifact_manifest.json SHA-256 mismatch")
    os.replace(uploading, drive_run_dir)
except Exception:
    if uploading.exists():
        shutil.rmtree(uploading, ignore_errors=True)
    raise

print(
    "AMP RECOVERED | "
    f"{final_skips}/{attempted_updates} skips ({amp_skip_rate:.2%}) | "
    f"final epoch skips={final_epoch_skips} | final scale={final_amp_scale:.0f}"
)
print("Không train lại; official-test chỉ chạy một batch inference.")
print("HOÀN TẤT")
print("Local run:", local_run_dir)
print("Drive run:", drive_run_dir)
print("Preview:", drive_run_dir / "sanpo_data_preview.png")
print("Best checkpoint:", drive_run_dir / "checkpoints" / "best.pt")
