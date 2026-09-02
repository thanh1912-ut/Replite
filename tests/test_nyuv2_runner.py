"""Protocol and extraction tests for the NYUDv2 training runner."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from collections import namedtuple
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tools import train_nyuv2 as runner


_DiskUsage = namedtuple("_DiskUsage", ("total", "used", "free"))


@pytest.fixture(autouse=True)
def _stable_test_disk_capacity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep synthetic extraction tests independent of host SSD pressure."""

    gib = 1024**3
    monkeypatch.setattr(
        runner.shutil,
        "disk_usage",
        lambda _path: _DiskUsage(total=128 * gib, used=1 * gib, free=127 * gib),
    )


def test_protocol_info_is_machine_readable_and_reports_runtime_source(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert runner.main(["protocol-info"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "schema_version": 1,
        "default_protocol_id": runner.PROTOCOL_ID,
        "supported_protocol_ids": list(runner.SUPPORTED_PROTOCOL_IDS),
        "source_commit": runner._runtime_source_commit(),
    }


def _write_sample(root: Path, key: str, label: int) -> None:
    for name in ("images", "segmentation", "depth", "gt_sets"):
        (root / name).mkdir(parents=True, exist_ok=True)
    image = np.zeros((8, 12, 3), dtype=np.uint8)
    image[..., 0] = 40 + label
    Image.fromarray(image).save(root / "images" / f"{key}.jpg")
    segmentation = np.full((8, 12), label, dtype=np.uint8)
    segmentation[0, 0] = 0
    Image.fromarray(segmentation).save(root / "segmentation" / f"{key}.png")
    np.save(root / "depth" / f"{key}.npy", np.full((8, 12), 2.0, np.float32))


def _archive(tmp_path: Path) -> Path:
    source = tmp_path / "source" / "NYUD_MT"
    _write_sample(source, "0001", 1)
    _write_sample(source, "0002", 1)
    _write_sample(source, "0003", 1)
    (source / "gt_sets" / "train.txt").write_text("0001\n0002\n", encoding="utf-8")
    (source / "gt_sets" / "val.txt").write_text("0003\n", encoding="utf-8")
    archive = tmp_path / "NYUDv2.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(source, arcname="NYUD_MT")
    return archive


def _config(tmp_path: Path, archive: Path) -> dict[str, object]:
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    return {
        "schema_version": 1,
        "protocol_id": runner.LEGACY_PROTOCOL_ID,
        "run_id": "nyu_unit_seed42",
        "source_repository": "https://example.test/Replite.git",
        "source_commit": "a" * 40,
        "archive": {
            "path": str(archive),
            "expected_bytes": archive.stat().st_size,
            "sha256": digest,
            "expected_train_samples": 2,
            "expected_test_samples": 1,
        },
        "paths": {
            "local_dataset_root": str(tmp_path / "local" / "nyudv2"),
            "local_work_root": str(tmp_path / "work"),
            "drive_run_root": str(tmp_path / "drive" / "run"),
        },
        "model": {
            "active_tasks": ["segmentation", "depth"],
            "backbone_name": "mobilenetv4_conv_small",
            "pretrained_in1k": False,
            "recurrence_steps": 3,
            "recurrent_c4_channels": 48,
            "recurrent_c5_channels": 64,
            "neck_channels": 48,
            "dense_channels": 32,
            "task_adapter_channels": 32,
            "use_sppf": False,
            "dense_fusion_direction": "seg_to_depth",
            "dense_fusion_detach_source": True,
        },
        "data": {
            "image_size": [288, 384],
            "batch_size": 16,
            "num_workers": 0,
            "prefetch_factor": 2,
            "num_classes": 40,
            "ignore_index": 255,
            "raw_label_mapping": {str(index): index - 1 for index in range(1, 41)},
            "source_ignore_labels": [0],
            "expected_raw_label_ids": [0, 1],
            "depth_unit_scale": 1.0,
            "depth_min_metres": 0.1,
            "depth_max_metres": 10.0,
            "inner_validation_fraction": 0.25,
            "split_seed": 42,
            "augmentation": {
                "horizontal_flip_probability": 0.5,
                "scale_min": 1.0,
                "scale_max": 1.1,
                "brightness": 0.1,
                "contrast": 0.1,
                "saturation": 0.08,
            },
        },
        "train": {
            "epochs": 2,
            "seed": 42,
            "base_lr": 3e-4,
            "backbone_lr_multiplier": 0.1,
            "weight_decay": 1e-2,
            "warmup_fraction": 0.05,
            "min_lr_ratio": 0.05,
            "grad_accum_steps": 1,
            "grad_clip_norm": 1.0,
            "amp": True,
            "amp_dtype": "float16",
            "amp_initial_scale": 1024.0,
            "progress_every_n_steps": 10,
            "monitor": "val/total",
            "monitor_mode": "min",
            "early_stopping_patience": 10,
            "early_stopping_min_delta": 0.0,
            "task_weights": {"segmentation": 1.0, "depth": 0.25},
        },
    }


def _v2_config(
    tmp_path: Path,
    archive: Path,
    *,
    mode: str = "multitask",
    stage: str = "main",
) -> dict[str, object]:
    value = _config(tmp_path, archive)
    tasks = {
        "seg-only": ["segmentation"],
        "depth-only": ["depth"],
        "multitask": ["segmentation", "depth"],
    }[mode]
    monitor = {
        "seg-only": ("val/segmentation/miou", "max"),
        "depth-only": ("val/depth/abs_rel", "min"),
        "multitask": ("val/selection/joint", "max"),
    }[mode]
    value.update(
        protocol_id=runner.PROTOCOL_ID,
        mode=mode,
        stage=stage,
        source_commit=runner._runtime_source_commit(),
    )
    value["model"].update(  # type: ignore[union-attr]
        active_tasks=tasks,
        dense_decoder="dense_v2_s",
        segmentation_auxiliary="segmentation" in tasks,
    )
    value["data"]["augmentation"].update(  # type: ignore[index,union-attr]
        scale_min=0.75,
        scale_max=1.25,
        class_aware_crop_probability=0.35,
        rare_classes=[0],
        blur_probability=0.08,
        blur_kernel_size=3,
    )
    value["train"].update(  # type: ignore[union-attr]
        monitor=monitor[0],
        monitor_mode=monitor[1],
        early_stopping_patience=10 if stage == "main" else None,
        task_weights={task: 1.0 for task in tasks},
        single_task_anchors=None,
        loss={
            "segmentation_normalize_weighted_loss": True,
            "segmentation_lovasz_weight": 0.25,
            "segmentation_auxiliary_weight": 0.25,
            "depth_loss_type": "per_image_silog_log_l1_gradient",
            "depth_log_l1_weight": 0.5,
            "depth_silog_weight": 1.0,
            "depth_gradient_weight": 0.25,
            "depth_silog_lambda": 0.5,
        },
    )
    return value


def _write_config(tmp_path: Path, value: dict[str, object]) -> Path:
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_extracts_directly_to_local_disk_and_is_idempotent(tmp_path: Path) -> None:
    archive = _archive(tmp_path)
    config = _config(tmp_path, archive)
    path = _write_config(tmp_path, config)

    first = runner.extract_campaign(path)
    second = runner.extract_campaign(path)

    root = Path(config["paths"]["local_dataset_root"])  # type: ignore[index]
    assert first == second
    assert first["official_train_samples"] == 2
    assert first["official_test_samples"] == 1
    assert (root / "images" / "0001.jpg").is_file()
    assert not (root.parent / archive.name).exists()


def test_prepare_freezes_inner_validation_without_test_leakage(tmp_path: Path) -> None:
    archive = _archive(tmp_path)
    config = _config(tmp_path, archive)
    path = _write_config(tmp_path, config)
    runner.extract_campaign(path)

    prepared = runner.prepare(path)
    fit_keys = {prepared.index.train[index].key for index in prepared.split.fit_indices}
    val_keys = {
        prepared.index.train[index].key
        for index in prepared.split.validation_indices
    }
    test_keys = {sample.key for sample in prepared.index.test}

    assert len(fit_keys) == len(val_keys) == 1
    assert not fit_keys & val_keys
    assert not (fit_keys | val_keys) & test_keys
    manifest = json.loads(prepared.split.manifest_path.read_text(encoding="utf-8"))
    assert manifest["official_test_used"] is False
    assert manifest["source"] == "official gt_sets/train.txt only"


def test_runner_builds_directional_detached_model_and_weighted_loss(
    tmp_path: Path,
) -> None:
    archive = _archive(tmp_path)
    path = _write_config(tmp_path, _config(tmp_path, archive))
    runner.extract_campaign(path)
    prepared = runner.prepare(path)

    model_config = runner._model_config(prepared)
    criterion = runner._create_criterion(prepared)

    assert model_config.active_tasks == ("segmentation", "depth")
    assert model_config.tasks.detection_classes is None
    assert model_config.tasks.dense_fusion_direction == "seg_to_depth"
    assert model_config.tasks.dense_fusion_detach_source is True
    assert criterion.resolved_task_weights["segmentation"] == 1.0
    assert criterion.resolved_task_weights["depth"] == 0.25


@pytest.mark.parametrize(
    ("mode", "expected_tasks", "monitor"),
    [
        ("seg-only", ("segmentation",), "val/segmentation/miou"),
        ("depth-only", ("depth",), "val/depth/abs_rel"),
        ("multitask", ("segmentation", "depth"), "val/selection/joint"),
    ],
)
def test_v2_modes_build_only_requested_tasks_and_metric(
    tmp_path: Path, mode: str, expected_tasks: tuple[str, ...], monitor: str
) -> None:
    archive = _archive(tmp_path)
    value = _v2_config(tmp_path, archive, mode=mode)
    # Explicit inactive zero is accepted as documented.
    inactive = set(runner.ACTIVE_TASKS) - set(expected_tasks)
    value["train"]["task_weights"].update(  # type: ignore[index,union-attr]
        {task: 0.0 for task in inactive}
    )
    path = _write_config(tmp_path, value)
    runner.extract_campaign(path)
    prepared = runner.prepare(path)

    model_config = runner._model_config(prepared)
    criterion = runner._create_criterion(prepared)
    metrics = runner._create_metrics(prepared)

    assert model_config.active_tasks == expected_tasks
    assert runner._trainer_config(prepared).monitor == monitor
    assert (metrics.segmentation is not None) == ("segmentation" in expected_tasks)
    assert (metrics.depth is not None) == ("depth" in expected_tasks)
    assert model_config.segmentation_auxiliary == ("segmentation" in expected_tasks)
    assert set(criterion.task_weights) >= set(expected_tasks)


def test_v2_prepare_rejects_runtime_source_commit_mismatch(tmp_path: Path) -> None:
    archive = _archive(tmp_path)
    value = _v2_config(tmp_path, archive)
    value["source_commit"] = "b" * 40
    path = _write_config(tmp_path, value)
    runner.extract_campaign(path)

    with pytest.raises(ValueError, match="runtime source commit"):
        runner.prepare(path)


def test_v2_official_test_is_locked_without_completed_training(tmp_path: Path) -> None:
    archive = _archive(tmp_path)
    path = _write_config(tmp_path, _v2_config(tmp_path, archive))
    runner.extract_campaign(path)

    with pytest.raises(RuntimeError, match="locked until a completed training"):
        runner.evaluate_test_campaign(path)


def test_v2_one_epoch_train_aliases_and_one_shot_official_test(
    tmp_path: Path,
) -> None:
    archive = _archive(tmp_path)
    value = _v2_config(tmp_path, archive)
    value["model"].update(  # type: ignore[union-attr]
        recurrence_steps=1,
        recurrent_c4_channels=8,
        recurrent_c5_channels=8,
        neck_channels=8,
        dense_channels=8,
        task_adapter_channels=8,
    )
    value["data"].update(batch_size=2, num_workers=0)  # type: ignore[union-attr]
    value["train"].update(  # type: ignore[union-attr]
        epochs=1,
        amp=False,
        progress_every_n_steps=100,
    )
    path = _write_config(tmp_path, value)
    runner.extract_campaign(path)

    summary = runner.train_campaign(path, resume=False)
    drive_run = Path(value["paths"]["drive_run_root"])  # type: ignore[index]
    checkpoint_dir = drive_run / "checkpoints"

    assert summary["training_complete"] is True
    assert summary["official_test_used"] is False
    for name in ("best.pt", "best_miou.pt", "best_absrel.pt", "best_joint.pt", "last.pt"):
        assert (checkpoint_dir / name).is_file()
        assert (checkpoint_dir / f"{name}.sha256").is_file()

    first = runner.evaluate_test_campaign(path)
    second = runner.evaluate_test_campaign(path)
    assert second == first
    assert first["official_test"]["used_once_after_best_strict_load"] is True
    metrics = first["official_test"]["metrics"]
    assert "segmentation/miou" in metrics
    assert "depth/abs_rel" in metrics
    assert "selection/joint" in metrics


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda value: value["model"].update(dense_decoder="legacy"),
            "dense_decoder",
        ),
        (
            lambda value: value["model"].update(segmentation_auxiliary=False),
            "segmentation_auxiliary",
        ),
        (
            lambda value: value["train"]["loss"].update(  # type: ignore[index]
                segmentation_lovasz_weight=0.0
            ),
            "Lovasz",
        ),
        (
            lambda value: value["train"]["loss"].update(  # type: ignore[index]
                depth_loss_type="l1"
            ),
            "per-image SiLog",
        ),
        (
            lambda value: value["data"]["augmentation"].update(  # type: ignore[index]
                class_aware_crop_probability=0.0
            ),
            "class-aware crop",
        ),
        (
            lambda value: value["train"].update(early_stopping_patience=None),
            "exactly 10",
        ),
    ],
)
def test_v2_main_rejects_disabled_improvement_components(
    tmp_path: Path, mutate, message: str
) -> None:
    archive = _archive(tmp_path)
    value = _v2_config(tmp_path, archive)
    mutate(value)

    with pytest.raises(ValueError, match=message):
        runner.load_campaign(_write_config(tmp_path, value))


def test_v2_ablation_allows_components_off_but_disables_early_stop(
    tmp_path: Path,
) -> None:
    archive = _archive(tmp_path)
    value = _v2_config(tmp_path, archive, stage="ablation")
    value["model"].update(  # type: ignore[union-attr]
        dense_decoder="legacy",
        segmentation_auxiliary=False,
    )
    value["data"]["augmentation"].update(  # type: ignore[index,union-attr]
        class_aware_crop_probability=0.0,
        rare_classes=[],
    )
    value["train"]["loss"].update(  # type: ignore[index,union-attr]
        segmentation_lovasz_weight=0.0,
        segmentation_auxiliary_weight=0.0,
        depth_loss_type="l1",
        depth_gradient_weight=0.0,
    )

    _, loaded = runner.load_campaign(_write_config(tmp_path, value))
    assert loaded["stage"] == "ablation"


def test_inspection_and_static_loader_contract_run_end_to_end(tmp_path: Path) -> None:
    archive = _archive(tmp_path)
    path = _write_config(tmp_path, _config(tmp_path, archive))
    runner.extract_campaign(path)

    report = runner.inspect_campaign(path)
    prepared = runner.prepare(path)
    train_loader, val_loader = runner.create_train_loaders(prepared)
    inputs, targets = next(iter(train_loader))
    model = runner._create_model(prepared)
    outputs = model(inputs)
    losses = runner._create_criterion(prepared)(outputs, targets)

    assert report["data"]["official_test_used"] is False
    assert report["data"]["input_shape"] == ["B", 3, 288, 384]
    assert report["model"]["tasks"]["detection_classes"] is None
    assert report["model"]["tasks"]["dense_fusion_direction"] == "seg_to_depth"
    assert len(train_loader) == len(val_loader) == 1
    assert inputs.shape == (1, 3, 288, 384)
    assert targets["segmentation"].shape == (1, 288, 384)
    assert targets["depth"].shape == (1, 1, 288, 384)
    assert outputs.detection is None
    assert outputs.segmentation.shape == (1, 40, 288, 384)
    assert outputs.depth.shape == (1, 1, 288, 384)
    assert losses["total"].isfinite()


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda value: value["data"].update(image_size=[288, 512]), "288x384"),
        (
            lambda value: value["model"].update(
                active_tasks=["detection", "segmentation", "depth"]
            ),
            "active_tasks",
        ),
        (
            lambda value: value["model"].update(
                dense_fusion_direction="bidirectional"
            ),
            "seg_to_depth",
        ),
        (
            lambda value: value["train"].update(monitor="train/total"),
            "val/total",
        ),
        (
            lambda value: value["train"].update(early_stopping_patience=9),
            "patience",
        ),
    ),
)
def test_load_campaign_rejects_protocol_drift(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    archive = _archive(tmp_path)
    value = _config(tmp_path, archive)
    mutation(value)
    path = _write_config(tmp_path, value)
    with pytest.raises(ValueError, match=message):
        runner.load_campaign(path)


def test_stream_extractor_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.tar"
    with tarfile.open(archive, "w") as handle:
        content = b"owned"
        info = tarfile.TarInfo("../outside.txt")
        info.size = len(content)
        handle.addfile(info, io.BytesIO(content))
    staging = tmp_path / "staging"
    staging.mkdir()

    with pytest.raises(ValueError, match="unsafe tar member"):
        runner._stream_extract_tar(archive, staging)
    assert not (tmp_path / "outside.txt").exists()
