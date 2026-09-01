"""YOLO-style terminal progress and stable epoch-report helpers.

The :class:`YoloProgressReporter` is deliberately only an observer: it can be
passed directly as ``Trainer(..., event_callback=reporter)`` and never mutates
the model, optimizer, metrics, or checkpoints.  The pure helpers in this
module keep notebook CSV/JSON exports deterministic and make missing task
metrics explicit instead of silently turning them into zeros.
"""

from __future__ import annotations

import math
import sys
import time
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any, TextIO


TRAIN_PROGRESS_COLUMNS = (
    "Epoch",
    "GPU_mem",
    "Batch",
    "Progress",
    "Speed",
    "ETA",
    "total",
    "box",
    "cls",
    "qual",
    "dfl",
    "seg",
    "depth",
    "Inst",
    "Size",
    "lr",
)

_TASK_ORDER = ("detection", "segmentation", "depth", "classification")
_TRAIN_PROGRESS_PREFIX = (
    "Epoch",
    "GPU_mem",
    "Batch",
    "Progress",
    "Speed",
    "ETA",
    "total",
)
_TRAIN_TASK_COLUMNS = {
    "detection": ("box", "cls", "qual", "dfl"),
    "segmentation": ("seg",),
    "depth": ("depth",),
    "classification": ("class",),
}
_TRAIN_PROGRESS_SUFFIX = ("Size", "lr")

VALIDATION_COLUMNS = (
    "Task",
    "Images",
    "Targets",
    "mAP50",
    "mAP50-95",
    "mIoU",
    "pixel accuracy",
    "AbsRel",
    "RMSE(m)",
    "delta1",
)

_VALIDATION_TASK_COLUMNS = {
    "detection": ("Images", "Targets", "mAP50", "mAP50-95"),
    "segmentation": ("mIoU", "pixel accuracy"),
    "depth": ("AbsRel", "RMSE(m)", "delta1"),
    "classification": ("Samples", "top1 accuracy"),
}

VALIDATION_PROGRESS_COLUMNS = ("Val", "Batch", "Progress", "Speed", "ETA")

EPOCH_ROW_FIELDS = (
    "schema_version",
    "epoch",
    "global_step",
    "amp_skip_count",
    "epoch_seconds",
    "peak_vram_gib",
    "lr",
    "train_total",
    "train_detection",
    "train_detection_box",
    "train_detection_classification",
    "train_detection_quality",
    "train_detection_dfl",
    "train_segmentation",
    "train_depth",
    "val_total",
    "val_detection",
    "val_detection_box",
    "val_detection_classification",
    "val_detection_quality",
    "val_detection_dfl",
    "val_segmentation",
    "val_depth",
    "val_images",
    "val_targets",
    "val_map50",
    "val_map50_95",
    "val_miou",
    "val_pixel_accuracy",
    "val_abs_rel",
    "val_rmse_m",
    "val_delta1",
)

_LOSS_NAMES = (
    "total",
    "detection",
    "detection_box",
    "detection_classification",
    "detection_quality",
    "detection_dfl",
    "segmentation",
    "depth",
)


def _as_optional_number(value: Any, *, name: str) -> int | float | None:
    """Convert a scalar-like value into a finite JSON/CSV-safe number."""

    if value is None:
        return None
    # Scalar tensors and NumPy scalars both expose ``item``.  Avoid importing
    # either dependency in this terminal-only module.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            value = item()
        except (TypeError, ValueError, RuntimeError):
            pass
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar or None")
    if isinstance(value, Integral):
        return int(value)
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _metric(result: Mapping[str, Any], name: str) -> int | float | None:
    return _as_optional_number(result.get(name), name=name)


def validation_summary(
    result: Mapping[str, Any],
) -> dict[str, int | float | str | None]:
    """Return the single compact multi-task validation row used by the UI."""

    if not isinstance(result, Mapping):
        raise TypeError("validation result must be a mapping")
    return {
        "Task": "all",
        "Images": _metric(result, "detection/num_images"),
        "Targets": _metric(result, "detection/num_targets"),
        "mAP50": _metric(result, "detection/map50"),
        "mAP50-95": _metric(result, "detection/map50_95"),
        "mIoU": _metric(result, "segmentation/miou"),
        "pixel accuracy": _metric(result, "segmentation/pixel_accuracy"),
        "AbsRel": _metric(result, "depth/abs_rel"),
        "RMSE(m)": _metric(result, "depth/rmse"),
        "delta1": _metric(result, "depth/delta1"),
    }


def _format_cell(value: Any, *, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, str):
        return value
    if isinstance(value, Integral) and not isinstance(value, bool):
        return str(int(value))
    return f"{float(value):.{digits}f}"


def _format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "--:--"
    rounded = int(round(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    return (
        f"{hours:d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"
    )


def _normalize_active_tasks(
    active_tasks: Sequence[str] | None,
) -> tuple[str, ...] | None:
    """Validate and canonicalize an optional active-task selection.

    ``None`` intentionally preserves the historical all-task progress schema.
    Explicit task selections are emitted in the same stable order as
    :class:`~replite.multitask.config.TaskConfig`, irrespective of caller
    ordering.
    """

    if active_tasks is None:
        return None
    if isinstance(active_tasks, (str, bytes)) or not isinstance(active_tasks, Sequence):
        raise TypeError("active_tasks must be a non-empty sequence or None")
    requested = tuple(active_tasks)
    if not requested:
        raise ValueError("active_tasks must not be empty")
    if any(not isinstance(task, str) for task in requested):
        raise TypeError("active_tasks entries must be strings")
    if len(set(requested)) != len(requested):
        raise ValueError("active_tasks must not contain duplicates")
    unknown = sorted(set(requested) - set(_TASK_ORDER))
    if unknown:
        raise ValueError("unknown active_tasks: " + ", ".join(unknown))
    selected = set(requested)
    return tuple(task for task in _TASK_ORDER if task in selected)


def _train_progress_columns(active_tasks: tuple[str, ...] | None) -> tuple[str, ...]:
    if active_tasks is None:
        return TRAIN_PROGRESS_COLUMNS
    columns = list(_TRAIN_PROGRESS_PREFIX)
    for task in active_tasks:
        columns.extend(_TRAIN_TASK_COLUMNS[task])
    if "detection" in active_tasks:
        columns.append("Inst")
    columns.extend(_TRAIN_PROGRESS_SUFFIX)
    return tuple(columns)


def _validation_columns(active_tasks: tuple[str, ...] | None) -> tuple[str, ...]:
    if active_tasks is None:
        return VALIDATION_COLUMNS
    columns = ["Task"]
    for task in active_tasks:
        columns.extend(_VALIDATION_TASK_COLUMNS[task])
    return tuple(columns)


def format_validation_table(
    result: Mapping[str, Any],
    *,
    active_tasks: Sequence[str] | None = None,
) -> str:
    """Format validation metrics as a tab-separated, YOLO-like task table."""

    normalized_tasks = _normalize_active_tasks(active_tasks)
    row = validation_summary(result)
    if normalized_tasks is not None and "classification" in normalized_tasks:
        row.update(
            {
                "Samples": _metric(result, "classification/num_samples"),
                "top1 accuracy": _metric(result, "classification/top1_accuracy"),
            }
        )
    columns = _validation_columns(normalized_tasks)
    return "\n".join(
        (
            "\t".join(columns),
            "\t".join(_format_cell(row[name]) for name in columns),
        )
    )


def flatten_epoch_record(record: Mapping[str, Any]) -> dict[str, int | float | None]:
    """Flatten one ``Trainer.fit`` history record into a stable wide row.

    Every invocation returns all :data:`EPOCH_ROW_FIELDS` in the same order.
    Unavailable tasks are represented by ``None`` so CSV writers emit an empty
    cell and JSON writers emit ``null``.
    """

    if not isinstance(record, Mapping):
        raise TypeError("epoch record must be a mapping")
    row: dict[str, int | float | None] = {name: None for name in EPOCH_ROW_FIELDS}
    row["schema_version"] = 1
    for name in (
        "epoch",
        "global_step",
        "amp_skip_count",
        "epoch_seconds",
        "peak_vram_gib",
        "lr",
    ):
        row[name] = _as_optional_number(record.get(name), name=name)

    for split in ("train", "val"):
        values = record.get(split)
        if values is None:
            continue
        if not isinstance(values, Mapping):
            raise TypeError(f"{split} result must be a mapping or None")
        for loss_name in _LOSS_NAMES:
            row[f"{split}_{loss_name}"] = _metric(values, loss_name)

    val = record.get("val")
    if isinstance(val, Mapping):
        summary = validation_summary(val)
        row.update(
            {
                "val_images": summary["Images"],
                "val_targets": summary["Targets"],
                "val_map50": summary["mAP50"],
                "val_map50_95": summary["mAP50-95"],
                "val_miou": summary["mIoU"],
                "val_pixel_accuracy": summary["pixel accuracy"],
                "val_abs_rel": summary["AbsRel"],
                "val_rmse_m": summary["RMSE(m)"],
                "val_delta1": summary["delta1"],
            }
        )
    # Guard against an accidental field-order change during later refactors.
    return {name: row[name] for name in EPOCH_ROW_FIELDS}


def _class_catalog(
    class_names: Sequence[str] | Mapping[int, str],
) -> list[tuple[int, str]]:
    if isinstance(class_names, Mapping):
        result: list[tuple[int, str]] = []
        for raw_id, raw_name in class_names.items():
            if isinstance(raw_id, bool) or not isinstance(raw_id, Integral):
                raise TypeError("class-name mapping keys must be integer class IDs")
            result.append((int(raw_id), str(raw_name)))
        return sorted(result)
    if isinstance(class_names, (str, bytes)) or not isinstance(class_names, Sequence):
        raise TypeError("class_names must be a sequence or integer-keyed mapping")
    return [(class_id, str(name)) for class_id, name in enumerate(class_names)]


def _integer_key_mapping(value: Any, *, name: str) -> dict[int, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    result: dict[int, Any] = {}
    for raw_key, item in value.items():
        if isinstance(raw_key, bool):
            raise TypeError(f"{name} keys must be integer class IDs")
        try:
            key = int(raw_key)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} keys must be integer class IDs") from exc
        result[key] = item
    return result


def detection_per_class_rows(
    result: Mapping[str, Any],
    class_names: Sequence[str] | Mapping[int, str],
) -> list[dict[str, int | float | str | bool | None]]:
    """Return one detection AP row per configured class, including absences."""

    if not isinstance(result, Mapping):
        raise TypeError("validation result must be a mapping")
    values = _integer_key_mapping(
        result.get("detection/per_class_map"), name="detection/per_class_map"
    )
    rows: list[dict[str, int | float | str | bool | None]] = []
    for class_id, class_name in _class_catalog(class_names):
        value = (
            _as_optional_number(values[class_id], name=f"detection AP class {class_id}")
            if class_id in values
            else None
        )
        rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "map50_95": value,
                "present": value is not None,
            }
        )
    return rows


def segmentation_per_class_rows(
    result: Mapping[str, Any],
    class_names: Sequence[str] | Mapping[int, str],
) -> list[dict[str, int | float | str | bool | None]]:
    """Return one segmentation IoU row per class; absent classes have ``None``."""

    if not isinstance(result, Mapping):
        raise TypeError("validation result must be a mapping")
    raw_values = result.get("segmentation/per_class_iou")
    raw_present = result.get("segmentation/present_classes")
    if raw_values is not None and (
        isinstance(raw_values, (str, bytes)) or not isinstance(raw_values, Sequence)
    ):
        raise TypeError("segmentation/per_class_iou must be a sequence")
    if raw_present is not None and (
        isinstance(raw_present, (str, bytes)) or not isinstance(raw_present, Sequence)
    ):
        raise TypeError("segmentation/present_classes must be a sequence")
    values = list(raw_values or ())
    present = list(raw_present or ())
    rows: list[dict[str, int | float | str | bool | None]] = []
    for class_id, class_name in _class_catalog(class_names):
        is_present = bool(present[class_id]) if class_id < len(present) else False
        value = None
        if is_present and class_id < len(values):
            value = _as_optional_number(
                values[class_id], name=f"segmentation IoU class {class_id}"
            )
        rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "iou": value,
                "present": is_present and value is not None,
            }
        )
    return rows


class YoloProgressReporter:
    """Render :class:`Trainer` events as a YOLO-style live progress table."""

    def __init__(
        self,
        *,
        every_n_steps: int = 20,
        reg_max: int = 0,
        active_tasks: Sequence[str] | None = None,
        stream: TextIO | None = None,
        use_tqdm: bool | None = None,
        clock: Any = time.monotonic,
    ) -> None:
        if (
            isinstance(every_n_steps, bool)
            or not isinstance(every_n_steps, Integral)
            or every_n_steps <= 0
        ):
            raise ValueError("every_n_steps must be a positive integer")
        if (
            isinstance(reg_max, bool)
            or not isinstance(reg_max, Integral)
            or reg_max < 0
        ):
            raise ValueError("reg_max must be a non-negative integer")
        if use_tqdm is not None and not isinstance(use_tqdm, bool):
            raise TypeError("use_tqdm must be a boolean or None")
        self.every_n_steps = int(every_n_steps)
        self.reg_max = int(reg_max)
        self.active_tasks = _normalize_active_tasks(active_tasks)
        self.train_progress_columns = _train_progress_columns(self.active_tasks)
        self.validation_columns = _validation_columns(self.active_tasks)
        self.stream = sys.stdout if stream is None else stream
        self.use_tqdm = use_tqdm
        if not callable(clock):
            raise TypeError("clock must be callable")
        self.clock = clock
        self._bar: Any = None
        self._bar_completed = 0
        self._phase_started = 0.0

    def _print(self, line: str) -> None:
        self.stream.write(line + "\n")
        flush = getattr(self.stream, "flush", None)
        if callable(flush):
            flush()

    def _wants_tqdm(self) -> bool:
        if self.use_tqdm is not None:
            return self.use_tqdm
        isatty = getattr(self.stream, "isatty", None)
        return bool(callable(isatty) and isatty())

    @staticmethod
    def _gpu_memory(value: Any) -> str:
        number = _as_optional_number(value, name="gpu_memory_bytes")
        if number is None or float(number) <= 0.0:
            return "0G"
        return f"{float(number) / 1024**3:.2f}G"

    def _progress_values(self, payload: Mapping[str, Any]) -> dict[str, str]:
        losses = payload.get("running_losses") or payload.get("losses") or {}
        if not isinstance(losses, Mapping):
            losses = {}
        epoch = int(payload.get("epoch", 0)) + 1
        total_epochs = payload.get("total_epochs")
        epoch_text = (
            f"{epoch}/{total_epochs}" if total_epochs is not None else str(epoch)
        )

        def loss(name: str) -> str:
            value = losses.get(name)
            if (
                value is None
                and name == "detection_dfl"
                and self.reg_max == 0
                and (self.active_tasks is None or "detection" in self.active_tasks)
            ):
                value = 0.0
            try:
                checked = _as_optional_number(value, name=name)
            except (TypeError, ValueError):
                checked = None
            return _format_cell(checked)

        image_size = payload.get("image_size")
        if (
            isinstance(image_size, Sequence)
            and not isinstance(image_size, (str, bytes))
            and len(image_size) == 2
        ):
            size = f"{int(image_size[0])}x{int(image_size[1])}"
        else:
            size = "N/A"
        raw_lr = payload.get("lr")
        if isinstance(raw_lr, Sequence) and not isinstance(raw_lr, (str, bytes)):
            raw_lr = raw_lr[0] if raw_lr else None
        checked_lr = (
            _as_optional_number(raw_lr, name="lr") if raw_lr is not None else None
        )
        completed = int(payload.get("batches_completed", 0))
        total_batches_raw = payload.get("total_batches")
        total_batches = (
            int(total_batches_raw) if total_batches_raw is not None else None
        )
        elapsed = max(0.0, float(self.clock()) - self._phase_started)
        speed = completed / elapsed if completed > 0 and elapsed > 0 else 0.0
        fraction = (
            completed / total_batches
            if total_batches is not None and total_batches > 0
            else 0.0
        )
        eta = (
            (total_batches - completed) / speed
            if total_batches is not None and speed > 0
            else None
        )
        return {
            "Epoch": epoch_text,
            "GPU_mem": self._gpu_memory(payload.get("gpu_memory_bytes")),
            "Batch": (
                f"{completed}/{total_batches}"
                if total_batches is not None
                else str(completed)
            ),
            "Progress": f"{fraction * 100:5.1f}%",
            "Speed": f"{speed:.2f}it/s" if speed > 0 else "--",
            "ETA": _format_duration(eta),
            "total": loss("total"),
            "box": loss("detection_box"),
            "cls": loss("detection_classification"),
            "qual": loss("detection_quality"),
            "dfl": loss("detection_dfl"),
            "seg": loss("segmentation"),
            "depth": loss("depth"),
            "class": loss("classification"),
            "Inst": _format_cell(payload.get("instances")),
            "Size": size,
            "lr": _format_cell(checked_lr, digits=6),
        }

    @staticmethod
    def _validation_progress_values(
        payload: Mapping[str, Any], elapsed: float
    ) -> dict[str, str]:
        epoch = payload.get("epoch")
        total_epochs = payload.get("total_epochs")
        val = "-" if epoch is None else str(int(epoch) + 1)
        if epoch is not None and total_epochs is not None:
            val = f"{int(epoch) + 1}/{int(total_epochs)}"
        completed = int(payload.get("batches_completed", 0))
        raw_total = payload.get("total_batches")
        total = int(raw_total) if raw_total is not None else None
        speed = completed / elapsed if completed > 0 and elapsed > 0 else 0.0
        fraction = completed / total if total is not None and total > 0 else 0.0
        eta = (total - completed) / speed if total is not None and speed > 0 else None
        return {
            "Val": val,
            "Batch": f"{completed}/{total}" if total is not None else str(completed),
            "Progress": f"{fraction * 100:5.1f}%",
            "Speed": f"{speed:.2f}it/s" if speed > 0 else "--",
            "ETA": _format_duration(eta),
        }

    def _close_bar(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None
            self._bar_completed = 0

    def __call__(self, event: str, payload: Mapping[str, Any]) -> None:
        """Consume one event emitted by :class:`~replite.training.Trainer`."""

        if not isinstance(event, str) or not isinstance(payload, Mapping):
            raise TypeError("progress callback expects (str, mapping)")
        if event == "train_epoch_start":
            self._close_bar()
            self._phase_started = float(self.clock())
            self._print("\t".join(self.train_progress_columns))
            if self._wants_tqdm():
                try:
                    from tqdm.auto import tqdm
                except ImportError:
                    if self.use_tqdm:
                        raise
                else:
                    self._bar = tqdm(
                        total=payload.get("total_batches"),
                        file=self.stream,
                        dynamic_ncols=True,
                        leave=True,
                    )
                    self._bar_completed = 0
            return

        if event == "train_batch_end":
            completed = int(payload.get("batches_completed", 0))
            total_batches = payload.get("total_batches")
            last = total_batches is not None and completed >= int(total_batches)
            should_render = (
                completed == 1 or completed % self.every_n_steps == 0 or last
            )
            if self._bar is not None:
                delta = max(0, completed - self._bar_completed)
                if delta:
                    self._bar.update(delta)
                    self._bar_completed = completed
                if should_render:
                    values = self._progress_values(payload)
                    self._bar.set_postfix_str(
                        " ".join(
                            f"{name}={values[name]}"
                            for name in self.train_progress_columns
                        )
                    )
            elif should_render:
                values = self._progress_values(payload)
                self._print(
                    "\t".join(values[name] for name in self.train_progress_columns)
                )
            return

        if event == "train_epoch_end":
            self._close_bar()
            return

        if event == "validation_start":
            self._close_bar()
            self._phase_started = float(self.clock())
            self._print("\t".join(VALIDATION_PROGRESS_COLUMNS))
            return

        if event == "validation_batch_end":
            completed = int(payload.get("batches_completed", 0))
            total_batches = payload.get("total_batches")
            last = total_batches is not None and completed >= int(total_batches)
            if completed == 1 or completed % self.every_n_steps == 0 or last:
                values = self._validation_progress_values(
                    payload,
                    max(0.0, float(self.clock()) - self._phase_started),
                )
                self._print(
                    "\t".join(values[name] for name in VALIDATION_PROGRESS_COLUMNS)
                )
            return

        if event == "validation_end":
            result = payload.get("result", {})
            if not isinstance(result, Mapping):
                raise TypeError("validation_end result must be a mapping")
            self._print(format_validation_table(result, active_tasks=self.active_tasks))


__all__ = [
    "EPOCH_ROW_FIELDS",
    "TRAIN_PROGRESS_COLUMNS",
    "VALIDATION_COLUMNS",
    "VALIDATION_PROGRESS_COLUMNS",
    "YoloProgressReporter",
    "detection_per_class_rows",
    "flatten_epoch_record",
    "format_validation_table",
    "segmentation_per_class_rows",
    "validation_summary",
]
