"""CPU-backed integration test for the Colab SANPO val evaluator."""

from __future__ import annotations

import hashlib
import json
import runpy
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from replite import RepLiteConfig, TaskConfig
from replite.data import sanpo_joint_collate
from replite.multitask.heads import DetectionOutput
from replite.multitask.model import RepLiteOutput
import replite.training.metrics as canonical_metrics


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "tools" / "evaluate_sanpo_smoke_val.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _TinyValDataset(Dataset):
    def __init__(self, manifest_path: Path) -> None:
        self.image_size = (32, 32)
        self.depth_min = 0.1
        self.depth_max = 80.0
        self.normalize = True
        self.manifest = {
            "samples": [{"target_frame": index} for index in range(73)]
        }
        manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
        self.info = SimpleNamespace(
            official_split="train",
            session_id="held-out-session",
            sensor="camera_head",
            manifest_path=manifest_path,
        )

    def __len__(self) -> int:
        return 73

    def __getitem__(self, index: int):
        clip = torch.zeros(3, 3, 1, 1)
        return clip, {
            "detection": {
                "boxes": torch.empty(0, 4),
                "labels": torch.empty(0, dtype=torch.long),
                "valid_size": self.image_size,
                "ignore_boxes": torch.empty(0, 4),
            },
            "segmentation": torch.zeros(1, 1, dtype=torch.long),
            "segmentation_valid": torch.ones(1, 1, dtype=torch.bool),
            "depth": torch.ones(1, 1, 1),
            "depth_valid": torch.ones(1, 1, 1, dtype=torch.bool),
        }


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))

    def to(self, *args, **kwargs):
        # The test exercises the full evaluator on CPU while monkeypatching the
        # CUDA runtime surface. Production models use normal nn.Module.to().
        return self

    def forward(self, inputs: torch.Tensor) -> RepLiteOutput:
        batch = inputs.shape[0]
        shapes = ((4, 4), (2, 2), (1, 1))
        classes = tuple(
            torch.full((batch, 15, height, width), -20.0)
            for height, width in shapes
        )
        boxes = tuple(
            torch.ones(batch, 4, height, width) for height, width in shapes
        )
        quality = tuple(
            torch.full((batch, 1, height, width), -20.0)
            for height, width in shapes
        )
        segmentation = torch.zeros(batch, 30, 1, 1)
        segmentation[:, 0] = 1.0
        depth = torch.ones(batch, 1, 1, 1)
        return RepLiteOutput(
            DetectionOutput(classes, boxes, quality),
            segmentation,
            depth,
            None,
        )


class _TinyCriterion:
    def __call__(self, outputs, targets):
        return {"total": outputs.depth.mean() * 0.0 + 1.0}


def _write_base_run(
    local_run: Path,
    drive_run: Path,
    *,
    run_id: str,
    source_commit: str,
    model_config: RepLiteConfig,
    model: nn.Module,
) -> None:
    checkpoints = local_run / "checkpoints"
    checkpoints.mkdir(parents=True)
    (local_run / "source_commit.txt").write_text(
        source_commit + "\n", encoding="utf-8"
    )
    resolved = {
        "schema_version": 1,
        "run_id": run_id,
        "source_commit": source_commit,
        "image_size_hw": [32, 32],
        "depth_range_metres": [0.1, 80.0],
        "batch_size": 73,
        "trainer_config": {"amp": False, "amp_dtype": "float16"},
        "split_protocol": {
            "train_session": "train-session",
            "validation_session": "held-out-session",
        },
    }
    (local_run / "resolved_config.json").write_text(
        json.dumps(resolved, sort_keys=True), encoding="utf-8"
    )
    summary = {
        "schema_version": 1,
        "status": "smoke_pass",
        "epochs": 3,
        "best_metrics": {"val/total": 1.0},
    }
    (local_run / "run_summary.json").write_text(
        json.dumps(summary, sort_keys=True), encoding="utf-8"
    )
    checkpoint = {
        "schema_version": 1,
        "checkpoint_kind": "replite_training_epoch_boundary",
        "progress": {
            "epoch_completed": 2,
            "next_epoch": 3,
            "batch_in_epoch": 0,
            "accumulation_index": 0,
            "global_step": 3,
        },
        "model": model.state_dict(),
        "model_metadata": {"config": model_config.as_dict()},
        "best_metrics": {"val/total": 1.0},
    }
    best = checkpoints / "best.pt"
    torch.save(checkpoint, best)
    (checkpoints / "best.pt.sha256").write_text(
        f"{_sha256(best)}  best.pt\n", encoding="utf-8"
    )
    records = []
    for relative in (
        "source_commit.txt",
        "resolved_config.json",
        "run_summary.json",
        "checkpoints/best.pt",
        "checkpoints/best.pt.sha256",
    ):
        path = local_run / relative
        records.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": _sha256(path)}
        )
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "artifacts": records,
    }
    (local_run / "artifact_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    drive_run.parent.mkdir(parents=True)
    shutil.copytree(local_run, drive_run)


def test_val_evaluator_runs_end_to_end_and_publishes_finite_bundle(
    tmp_path: Path, monkeypatch
) -> None:
    run_id = "sanpo_smoke_test"
    source_commit = "1" * 40
    local_run = tmp_path / "local_runs" / run_id
    drive_run = tmp_path / "drive_runs" / run_id
    model_config = RepLiteConfig(
        tasks=TaskConfig(detection_classes=15, segmentation_classes=30, depth=True)
    )
    live_model = _TinyModel()
    _write_base_run(
        local_run,
        drive_run,
        run_id=run_id,
        source_commit=source_commit,
        model_config=model_config,
        model=live_model,
    )
    dataset = _TinyValDataset(tmp_path / "val_manifest.json")
    loader = DataLoader(
        dataset,
        batch_size=73,
        shuffle=False,
        drop_last=False,
        collate_fn=sanpo_joint_collate,
    )
    trainer = SimpleNamespace(
        model=live_model,
        criterion=_TinyCriterion(),
        device=SimpleNamespace(type="cuda"),
        amp_enabled=False,
        amp_dtype=torch.float16,
        config=SimpleNamespace(amp_dtype="float16"),
        global_step=3,
        amp_skip_count=0,
    )

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "Fake CUDA")
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda state: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(
        subprocess, "check_output", lambda *args, **kwargs: "2" * 40 + "\n"
    )
    monkeypatch.setattr(plt, "show", lambda: None)

    canonical_detection_map = canonical_metrics.DetectionMAP
    namespace = runpy.run_path(
        str(EVALUATOR),
        init_globals={
            "DEPTH_MAX_METRES": 80.0,
            "DEPTH_MIN_METRES": 0.1,
            "DRIVE_RUN_DIR": drive_run,
            "IMAGE_HEIGHT": 32,
            "IMAGE_WIDTH": 32,
            "LOCAL_RUN_DIR": local_run,
            "REPO_DIR": ROOT,
            "RUN_ID": run_id,
            "SOURCE_COMMIT": source_commit,
            "create_replite_model": lambda config: _TinyModel(),
            "model_config": model_config,
            "move_to_device": lambda value, device, non_blocking: value,
            "trainer": trainer,
            "val_dataset": dataset,
            "val_loader": loader,
        },
    )

    result = namespace["VAL_METRICS_RESULT"]
    assert result["overall"]["mIoU"] == 1.0
    assert result["overall"]["AbsRel"] == 0.0
    assert result["live_state_unchanged"] is True
    assert namespace["checkpoint_model_state"] is None
    assert all(namespace[name] is None for name in ("inputs", "targets", "outputs", "losses"))
    assert "validation_metrics" not in namespace
    assert "fig" not in namespace
    assert canonical_metrics.DetectionMAP is canonical_detection_map
    assert not any(
        name.startswith("replite.training._sanpo_val_metrics_")
        for name in sys.modules
    )
    bundles = list((drive_run / "evaluations").iterdir())
    assert len(bundles) == 1
    payload_text = (bundles[0] / "val_metrics_best.json").read_text(encoding="utf-8")
    assert "NaN" not in payload_text and "Infinity" not in payload_text
    payload = json.loads(payload_text)
    assert payload["expected_valid_pixels"] == {
        "segmentation": 73,
        "depth": 73,
    }
    assert (bundles[0] / "val_segmentation_confusion_matrix.csv").is_file()
    assert (bundles[0] / "val_metrics_best.png").is_file()
