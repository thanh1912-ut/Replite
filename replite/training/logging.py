"""Append-only, machine-readable training logs.

JSONL is the source of truth.  A long-form CSV is written alongside it so one
metric occupies one row and can be inspected without flattening evolving
schemas.  The logger intentionally opens files per event: this makes notebook
interruptions and resumed processes less likely to lose buffered records.
"""

from __future__ import annotations

import csv
import json
import math
import os
from datetime import datetime, timezone
from numbers import Real
from pathlib import Path
from typing import Any, Mapping

import torch


_CSV_FIELDS = (
    "timestamp_utc",
    "run_id",
    "event",
    "split",
    "epoch",
    "global_step",
    "metric",
    "value",
)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _scalar(value: object, name: str) -> int | float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"metric {name!r} must be scalar")
        value = value.detach().cpu().item()
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"metric {name!r} must be a real scalar")
    result: int | float
    result = int(value) if isinstance(value, int) else float(value)
    if isinstance(result, float) and not math.isfinite(result):
        raise ValueError(f"metric {name!r} must be finite")
    return result


def _jsonable(value: object) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return _scalar(value, "extra")
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("extra fields must not contain NaN or infinity")
        return value
    return str(value)


def _append_bytes(path: Path, payload: bytes, *, fsync: bool) -> None:
    flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
    descriptor = os.open(path, flags, 0o644)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        if fsync:
            os.fsync(descriptor)
    finally:
        os.close(descriptor)


class TrainingLogger:
    """Write append-only ``events.jsonl`` and long-form ``metrics.csv``.

    Args:
        output_dir: Run directory receiving both files.
        run_id: Stable identifier included in every row.
        fsync: Force durable storage after each event.  This is safer but can be
            slow on network filesystems, so callers commonly enable it only for
            epoch/checkpoint events.

    Only rank zero should instantiate this class in distributed training.
    """

    def __init__(
        self,
        output_dir: str | os.PathLike[str],
        *,
        run_id: str = "run",
        fsync: bool = False,
    ) -> None:
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must be a non-empty string")
        if not isinstance(fsync, bool):
            raise ValueError("fsync must be a boolean")
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.fsync = fsync
        self.jsonl_path = self.output_dir / "events.jsonl"
        self.csv_path = self.output_dir / "metrics.csv"
        self._closed = False
        self._ensure_csv_header()

    def _ensure_csv_header(self) -> None:
        if self.csv_path.exists() and self.csv_path.stat().st_size > 0:
            with self.csv_path.open("r", encoding="utf-8", newline="") as handle:
                first = next(csv.reader(handle), None)
            if tuple(first or ()) != _CSV_FIELDS:
                raise ValueError(
                    f"existing CSV has an incompatible header: {self.csv_path}"
                )
            return
        buffer = ",".join(_CSV_FIELDS) + "\n"
        _append_bytes(self.csv_path, buffer.encode("utf-8"), fsync=self.fsync)

    def log(
        self,
        event: str,
        metrics: Mapping[str, object] | None = None,
        *,
        epoch: int | None = None,
        global_step: int | None = None,
        split: str | None = None,
        extra: Mapping[str, object] | None = None,
    ) -> dict[str, Any]:
        """Append one event and return the JSON-compatible record.

        ``metrics`` values must be finite scalar tensors or real numbers.
        Nested/non-scalar diagnostics belong in ``extra`` and are kept only in
        JSONL.  Epoch numbering is caller-defined but must remain consistent
        within a run.
        """

        if not isinstance(event, str) or not event.strip():
            raise ValueError("event must be a non-empty string")
        if self._closed:
            raise RuntimeError("logger is closed")
        if epoch is not None and (
            isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0
        ):
            raise ValueError("epoch must be a non-negative integer or None")
        if global_step is not None and (
            isinstance(global_step, bool)
            or not isinstance(global_step, int)
            or global_step < 0
        ):
            raise ValueError("global_step must be a non-negative integer or None")
        if split is not None and (not isinstance(split, str) or not split):
            raise ValueError("split must be a non-empty string or None")

        checked_metrics = {
            str(name): _scalar(value, str(name))
            for name, value in (metrics or {}).items()
        }
        timestamp = _timestamp()
        record: dict[str, Any] = {
            "schema_version": 1,
            "timestamp_utc": timestamp,
            "run_id": self.run_id,
            "event": event,
            "split": split,
            "epoch": epoch,
            "global_step": global_step,
            "metrics": checked_metrics,
        }
        if extra:
            record["extra"] = _jsonable(extra)
        encoded = (
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        _append_bytes(self.jsonl_path, encoded, fsync=self.fsync)

        if checked_metrics:
            rows: list[str] = []
            for metric, value in checked_metrics.items():
                # csv.writer handles commas and quotes in names/run identifiers.
                from io import StringIO

                buffer = StringIO()
                writer = csv.DictWriter(buffer, fieldnames=_CSV_FIELDS)
                writer.writerow(
                    {
                        "timestamp_utc": timestamp,
                        "run_id": self.run_id,
                        "event": event,
                        "split": "" if split is None else split,
                        "epoch": "" if epoch is None else epoch,
                        "global_step": (
                            "" if global_step is None else global_step
                        ),
                        "metric": metric,
                        "value": value,
                    }
                )
                rows.append(buffer.getvalue())
            _append_bytes(
                self.csv_path,
                "".join(rows).encode("utf-8"),
                fsync=self.fsync,
            )
        return record

    def close(self) -> None:
        """Close the logical logger.

        Files are opened atomically for each append, so there is no buffered
        descriptor to flush. The closed flag still catches accidental writes
        after a run has been finalized.
        """

        self._closed = True

    def __enter__(self) -> "TrainingLogger":
        if self._closed:
            raise RuntimeError("logger is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


__all__ = ["TrainingLogger"]
