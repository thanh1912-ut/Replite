from __future__ import annotations

from io import StringIO

import pytest

from replite.training.progress import (
    EPOCH_ROW_FIELDS,
    TRAIN_PROGRESS_COLUMNS,
    VALIDATION_COLUMNS,
    VALIDATION_PROGRESS_COLUMNS,
    YoloProgressReporter,
    detection_per_class_rows,
    flatten_epoch_record,
    format_validation_table,
    segmentation_per_class_rows,
)


def _validation_result():
    return {
        "total": 1.25,
        "detection": 0.5,
        "segmentation": 0.4,
        "depth": 0.35,
        "detection/num_images": 12,
        "detection/num_targets": 34,
        "detection/map50": 0.61,
        "detection/map50_95": 0.42,
        "detection/per_class_map": {0: 0.7, 2: 0.14},
        "segmentation/miou": 0.53,
        "segmentation/pixel_accuracy": 0.91,
        "segmentation/per_class_iou": [0.9, 0.0, 0.4],
        "segmentation/present_classes": [True, False, True],
        "depth/abs_rel": 0.12,
        "depth/rmse": 2.5,
        "depth/delta1": 0.83,
    }


def test_plain_reporter_emits_deterministic_batch_progress_speed_and_eta() -> None:
    stream = StringIO()
    ticks = iter((0.0, 2.0, 4.0, 6.0))
    reporter = YoloProgressReporter(
        every_n_steps=2,
        reg_max=0,
        stream=stream,
        use_tqdm=False,
        clock=lambda: next(ticks),
    )
    reporter("train_epoch_start", {"epoch": 0, "total_epochs": 3, "total_batches": 3})
    common = {
        "epoch": 0,
        "total_epochs": 3,
        "total_batches": 3,
        "running_losses": {
            "total": 2.0,
            "detection_box": 0.2,
            "detection_classification": 0.3,
            "detection_quality": 0.1,
            "segmentation": 0.8,
            "depth": 0.6,
        },
        "instances": 7,
        "image_size": (512, 1024),
        "lr": [0.0002],
        "gpu_memory_bytes": 2 * 1024**3,
    }
    reporter("train_batch_end", {**common, "batches_completed": 1})
    reporter("train_batch_end", {**common, "batches_completed": 2})
    reporter("train_batch_end", {**common, "batches_completed": 3})
    reporter("train_epoch_end", {})

    lines = stream.getvalue().splitlines()
    assert lines[0] == "\t".join(TRAIN_PROGRESS_COLUMNS)
    assert len(lines) == 4  # header, first row, cadence row, forced last row
    fields = lines[1].split("\t")
    assert len(fields) == len(TRAIN_PROGRESS_COLUMNS)
    assert fields[0] == "1/3"
    assert fields[1] == "2.00G"
    assert fields[TRAIN_PROGRESS_COLUMNS.index("Batch")] == "1/3"
    assert fields[TRAIN_PROGRESS_COLUMNS.index("Progress")] == " 33.3%"
    assert fields[TRAIN_PROGRESS_COLUMNS.index("Speed")] == "0.50it/s"
    assert fields[TRAIN_PROGRESS_COLUMNS.index("ETA")] == "00:04"
    assert fields[TRAIN_PROGRESS_COLUMNS.index("dfl")] == "0.0000"
    assert fields[TRAIN_PROGRESS_COLUMNS.index("Inst")] == "7"
    assert fields[TRAIN_PROGRESS_COLUMNS.index("Size")] == "512x1024"
    assert fields[TRAIN_PROGRESS_COLUMNS.index("lr")] == "0.000200"
    cadence = lines[2].split("\t")
    assert cadence[TRAIN_PROGRESS_COLUMNS.index("Batch")] == "2/3"
    assert cadence[TRAIN_PROGRESS_COLUMNS.index("Progress")] == " 66.7%"
    assert cadence[TRAIN_PROGRESS_COLUMNS.index("Speed")] == "0.50it/s"
    assert cadence[TRAIN_PROGRESS_COLUMNS.index("ETA")] == "00:02"
    last = lines[3].split("\t")
    assert last[TRAIN_PROGRESS_COLUMNS.index("Batch")] == "3/3"
    assert last[TRAIN_PROGRESS_COLUMNS.index("Progress")] == "100.0%"
    assert last[TRAIN_PROGRESS_COLUMNS.index("Speed")] == "0.50it/s"
    assert last[TRAIN_PROGRESS_COLUMNS.index("ETA")] == "00:00"


def test_validation_progress_reports_first_cadence_last_and_metrics() -> None:
    stream = StringIO()
    ticks = iter((10.0, 12.0, 14.0, 20.0))
    reporter = YoloProgressReporter(
        every_n_steps=2,
        stream=stream,
        use_tqdm=False,
        clock=lambda: next(ticks),
    )
    reporter(
        "validation_start",
        {"epoch": 1, "total_epochs": 3, "total_batches": 4},
    )
    common = {"epoch": 1, "total_epochs": 3, "total_batches": 4}
    for completed in range(1, 5):
        reporter(
            "validation_batch_end",
            {**common, "batches_completed": completed},
        )
    reporter("validation_end", {"result": _validation_result()})

    lines = stream.getvalue().splitlines()
    assert lines[0] == "\t".join(VALIDATION_PROGRESS_COLUMNS)
    assert lines[1].split("\t") == ["2/3", "1/4", " 25.0%", "0.50it/s", "00:06"]
    assert lines[2].split("\t") == ["2/3", "2/4", " 50.0%", "0.50it/s", "00:04"]
    assert lines[3].split("\t") == ["2/3", "4/4", "100.0%", "0.40it/s", "00:00"]
    assert lines[4] == "\t".join(VALIDATION_COLUMNS)
    assert lines[5].startswith("all\t12\t34\t0.6100\t0.4200")


def test_validation_table_has_exact_fields_and_na_for_missing_tasks() -> None:
    table = format_validation_table(
        {
            "detection/num_images": 5,
            "detection/num_targets": 9,
            "detection/map50": 0.5,
            "detection/map50_95": 0.25,
        }
    )
    header, row = table.splitlines()
    assert header == "\t".join(VALIDATION_COLUMNS)
    assert row.split("\t") == [
        "all",
        "5",
        "9",
        "0.5000",
        "0.2500",
        "N/A",
        "N/A",
        "N/A",
        "N/A",
        "N/A",
    ]


def test_validation_end_prints_all_task_metrics() -> None:
    stream = StringIO()
    reporter = YoloProgressReporter(stream=stream, use_tqdm=False)
    reporter("validation_end", {"result": _validation_result()})
    lines = stream.getvalue().splitlines()
    assert lines[0] == "\t".join(VALIDATION_COLUMNS)
    assert lines[1] == "all\t12\t34\t0.6100\t0.4200\t0.5300\t0.9100\t0.1200\t2.5000\t0.8300"


def test_flatten_epoch_record_has_stable_complete_schema() -> None:
    record = {
        "epoch": 0,
        "global_step": 10,
        "amp_skip_count": 0,
        "epoch_seconds": 60.5,
        "peak_vram_gib": 4.25,
        "lr": 0.001,
        "train": {
            "total": 2.0,
            "detection_box": 0.3,
            "detection_dfl": 0.0,
        },
        "val": _validation_result(),
    }
    row = flatten_epoch_record(record)
    assert tuple(row) == EPOCH_ROW_FIELDS
    assert row["schema_version"] == 1
    assert row["epoch"] == 0
    assert row["train_total"] == 2.0
    assert row["train_detection_dfl"] == 0.0
    assert row["train_segmentation"] is None
    assert row["val_images"] == 12
    assert row["val_map50_95"] == 0.42
    assert row["val_rmse_m"] == 2.5


def test_per_class_rows_keep_absent_detection_and_segmentation_classes() -> None:
    result = _validation_result()
    detection = detection_per_class_rows(result, ["person", "car", "bike"])
    assert detection == [
        {"class_id": 0, "class_name": "person", "map50_95": 0.7, "present": True},
        {"class_id": 1, "class_name": "car", "map50_95": None, "present": False},
        {"class_id": 2, "class_name": "bike", "map50_95": 0.14, "present": True},
    ]
    segmentation = segmentation_per_class_rows(result, ["road", "sky", "person"])
    assert segmentation == [
        {"class_id": 0, "class_name": "road", "iou": 0.9, "present": True},
        {"class_id": 1, "class_name": "sky", "iou": None, "present": False},
        {"class_id": 2, "class_name": "person", "iou": 0.4, "present": True},
    ]


def test_progress_configuration_and_nonfinite_exports_are_rejected() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        YoloProgressReporter(every_n_steps=0)
    with pytest.raises(ValueError, match="finite"):
        flatten_epoch_record({"epoch": 0, "train": {"total": float("nan")}})
    with pytest.raises(TypeError, match="clock"):
        YoloProgressReporter(clock=None)
