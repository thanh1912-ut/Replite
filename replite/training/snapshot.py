"""Checksummed, versioned filesystem snapshots for durable epoch resume.

The helpers in this module deliberately use only :mod:`pathlib`/filesystem
semantics.  A mounted Google Drive directory is therefore just another
destination: files are copied into an ``.uploading`` sibling, the manifest is
written last, and the complete directory is atomically renamed into place.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping


SNAPSHOT_SCHEMA_VERSION = 1
SNAPSHOT_KIND = "replite_epoch_snapshot"
SNAPSHOT_MANIFEST = "snapshot_manifest.json"
_CONTEXT_FIELDS = (
    "source_sha256",
    "config_sha256",
    "catalog_sha256",
    "split_sha256",
)


class SnapshotError(RuntimeError):
    """Base class for snapshot publication and restoration failures."""


class SnapshotValidationError(SnapshotError):
    """A completed-looking snapshot is incomplete, corrupt, or incompatible."""


class SnapshotConflictError(SnapshotError):
    """A completed version already exists with different immutable content."""


@dataclass(frozen=True)
class SnapshotContext:
    """Immutable hashes tying a checkpoint to its exact experiment context."""

    source_sha256: str
    config_sha256: str
    catalog_sha256: str
    split_sha256: str

    def as_dict(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in _CONTEXT_FIELDS}


@dataclass(frozen=True)
class SnapshotFile:
    """One checksummed file recorded in a snapshot manifest."""

    path: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class PublishedSnapshot:
    """Metadata for a newly published or idempotently reused snapshot."""

    snapshot_dir: Path
    manifest_path: Path
    epoch_completed: int
    checkpoint_relative_path: str
    checkpoint_sha256: str
    context: SnapshotContext
    files: tuple[SnapshotFile, ...]
    idempotent: bool


@dataclass(frozen=True)
class RestoredSnapshot:
    """Metadata for the newest valid checkpoint restored to local storage."""

    snapshot_dir: Path
    manifest_path: Path
    epoch_completed: int
    source_checkpoint_path: Path
    local_checkpoint_path: Path
    checkpoint_sha256: str
    context: SnapshotContext
    files: tuple[SnapshotFile, ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _valid_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _normalize_context(
    value: SnapshotContext | Mapping[str, str],
) -> SnapshotContext:
    if isinstance(value, SnapshotContext):
        values = value.as_dict()
    elif isinstance(value, Mapping):
        if set(value) != set(_CONTEXT_FIELDS):
            missing = sorted(set(_CONTEXT_FIELDS) - set(value))
            extra = sorted(set(value) - set(_CONTEXT_FIELDS))
            details = []
            if missing:
                details.append("missing " + ", ".join(missing))
            if extra:
                details.append("unexpected " + ", ".join(extra))
            raise ValueError("snapshot context fields must match exactly (" + "; ".join(details) + ")")
        values = {name: value[name] for name in _CONTEXT_FIELDS}
    else:
        raise TypeError("context must be SnapshotContext or a mapping")
    normalized: dict[str, str] = {}
    for name, digest in values.items():
        if not _valid_digest(digest):
            raise ValueError(f"{name} must be a 64-character SHA-256 digest")
        normalized[name] = digest.lower()
    return SnapshotContext(**normalized)


def _relative_path(value: str | os.PathLike[str]) -> str:
    raw = os.fspath(value)
    if not raw or "\\" in raw:
        raise ValueError(f"snapshot paths must be non-empty POSIX relative paths: {raw!r}")
    path = PurePosixPath(raw)
    if path.is_absolute() or path == PurePosixPath(".") or ".." in path.parts:
        raise ValueError(f"snapshot path escapes its root: {raw!r}")
    return path.as_posix()


def _local_source(root: Path, relative: str) -> Path:
    candidate = root.joinpath(*PurePosixPath(relative).parts)
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
    except (FileNotFoundError, ValueError) as exc:
        raise SnapshotValidationError(f"snapshot source is missing or escapes run root: {relative}") from exc
    if not resolved.is_file() or candidate.is_symlink():
        raise SnapshotValidationError(f"snapshot source must be a regular non-symlink file: {relative}")
    return resolved


def _file_record(path: Path, relative: str) -> SnapshotFile:
    try:
        size = path.stat().st_size
        digest = _sha256_file(path)
    except OSError as exc:
        raise SnapshotValidationError(f"cannot read snapshot file: {relative}") from exc
    return SnapshotFile(path=relative, bytes=size, sha256=digest)


def _checkpoint_sidecar_relative(checkpoint_relative: str) -> str:
    return checkpoint_relative + ".sha256"


def _verify_checkpoint_sidecar(
    checkpoint: Path,
    sidecar: Path,
    digest: str,
    *,
    checkpoint_name: str | None = None,
) -> None:
    try:
        fields = sidecar.read_text(encoding="utf-8").strip().split()
    except OSError as exc:
        raise SnapshotValidationError(f"missing or unreadable checkpoint sidecar: {sidecar}") from exc
    expected_name = checkpoint.name if checkpoint_name is None else checkpoint_name
    if len(fields) != 2 or fields[1] != expected_name:
        raise SnapshotValidationError(f"invalid checkpoint sidecar: {sidecar}")
    if not _valid_digest(fields[0]) or fields[0].lower() != digest:
        raise SnapshotValidationError(f"checkpoint sidecar SHA-256 mismatch: {sidecar}")


def _snapshot_name(epoch_completed: int) -> str:
    if isinstance(epoch_completed, bool) or not isinstance(epoch_completed, int):
        raise ValueError("epoch_completed must be an integer")
    if epoch_completed < 0:
        raise ValueError("epoch_completed must be non-negative")
    return f"epoch_{epoch_completed:06d}"


def _canonical_manifest(
    *,
    epoch_completed: int,
    context: SnapshotContext,
    checkpoint_relative: str,
    files: Iterable[SnapshotFile],
) -> dict[str, object]:
    records = sorted(files, key=lambda item: item.path)
    checkpoint = next(item for item in records if item.path == checkpoint_relative)
    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "snapshot_kind": SNAPSHOT_KIND,
        "epoch_completed": epoch_completed,
        "context": context.as_dict(),
        "checkpoint": {
            "path": checkpoint_relative,
            "sidecar_path": _checkpoint_sidecar_relative(checkpoint_relative),
            "bytes": checkpoint.bytes,
            "sha256": checkpoint.sha256,
        },
        "files": [
            {"path": item.path, "bytes": item.bytes, "sha256": item.sha256}
            for item in records
        ],
    }


def _manifest_bytes(manifest: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _fsync_directory_best_effort(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        try:
            os.fsync(descriptor)
        except OSError as exc:
            if exc.errno not in {errno.EINVAL, errno.ENOTSUP, errno.EBADF}:
                raise
    finally:
        os.close(descriptor)


def _copy_verified(source: Path, destination: Path, expected: SnapshotFile) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    try:
        with source.open("rb") as source_handle, destination.open("xb") as target_handle:
            while block := source_handle.read(16 * 1024 * 1024):
                target_handle.write(block)
                digest.update(block)
                size += len(block)
            target_handle.flush()
            os.fsync(target_handle.fileno())
    except OSError as exc:
        raise SnapshotError(f"failed to copy snapshot file: {expected.path}") from exc
    if size != expected.bytes or digest.hexdigest() != expected.sha256:
        raise SnapshotValidationError(f"source changed while publishing: {expected.path}")


def _read_manifest(snapshot_dir: Path) -> dict[str, object]:
    path = snapshot_dir / SNAPSHOT_MANIFEST
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SnapshotValidationError(f"missing or invalid snapshot manifest: {path}") from exc
    if not isinstance(value, dict):
        raise SnapshotValidationError(f"snapshot manifest must be an object: {path}")
    return value


def _decode_and_verify_manifest(
    snapshot_dir: Path,
    *,
    expected_context: SnapshotContext | Mapping[str, str] | None,
) -> tuple[dict[str, object], SnapshotContext, tuple[SnapshotFile, ...]]:
    manifest = _read_manifest(snapshot_dir)
    if manifest.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise SnapshotValidationError("unsupported snapshot schema")
    if manifest.get("snapshot_kind") != SNAPSHOT_KIND:
        raise SnapshotValidationError("unexpected snapshot kind")
    epoch = manifest.get("epoch_completed")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise SnapshotValidationError("invalid snapshot epoch")
    if snapshot_dir.name != _snapshot_name(epoch):
        raise SnapshotValidationError("snapshot directory name/epoch mismatch")
    try:
        context = _normalize_context(manifest.get("context", {}))
    except (TypeError, ValueError) as exc:
        raise SnapshotValidationError("invalid snapshot context") from exc
    if expected_context is not None and context != _normalize_context(expected_context):
        raise SnapshotValidationError("snapshot context mismatch")

    raw_files = manifest.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise SnapshotValidationError("snapshot manifest has no files")
    records: list[SnapshotFile] = []
    seen: set[str] = set()
    for raw in raw_files:
        if not isinstance(raw, dict):
            raise SnapshotValidationError("invalid snapshot file record")
        try:
            relative = _relative_path(raw["path"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SnapshotValidationError("invalid snapshot file path") from exc
        size, digest = raw.get("bytes"), raw.get("sha256")
        if relative in seen or isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise SnapshotValidationError(f"invalid/duplicate snapshot file record: {relative}")
        if not _valid_digest(digest):
            raise SnapshotValidationError(f"invalid snapshot file SHA-256: {relative}")
        seen.add(relative)
        record = SnapshotFile(relative, size, digest.lower())
        path = snapshot_dir.joinpath(*PurePosixPath(relative).parts)
        if path.is_symlink() or not path.is_file():
            raise SnapshotValidationError(f"snapshot file is missing or not regular: {relative}")
        actual = _file_record(path, relative)
        if actual != record:
            raise SnapshotValidationError(f"snapshot file size/SHA-256 mismatch: {relative}")
        records.append(record)

    checkpoint = manifest.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise SnapshotValidationError("snapshot checkpoint metadata is missing")
    try:
        checkpoint_relative = _relative_path(checkpoint["path"])
        sidecar_relative = _relative_path(checkpoint["sidecar_path"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SnapshotValidationError("invalid snapshot checkpoint paths") from exc
    if sidecar_relative != _checkpoint_sidecar_relative(checkpoint_relative):
        raise SnapshotValidationError("checkpoint sidecar path mismatch")
    by_path = {record.path: record for record in records}
    if checkpoint_relative not in by_path or sidecar_relative not in by_path:
        raise SnapshotValidationError("checkpoint or checksum sidecar is not in snapshot files")
    checkpoint_record = by_path[checkpoint_relative]
    if checkpoint.get("bytes") != checkpoint_record.bytes or checkpoint.get("sha256") != checkpoint_record.sha256:
        raise SnapshotValidationError("checkpoint metadata/file record mismatch")
    checkpoint_path = snapshot_dir.joinpath(*PurePosixPath(checkpoint_relative).parts)
    sidecar_path = snapshot_dir.joinpath(*PurePosixPath(sidecar_relative).parts)
    _verify_checkpoint_sidecar(checkpoint_path, sidecar_path, checkpoint_record.sha256)
    return manifest, context, tuple(sorted(records, key=lambda item: item.path))


def _published_result(
    snapshot_dir: Path,
    manifest: Mapping[str, object],
    context: SnapshotContext,
    files: tuple[SnapshotFile, ...],
    *,
    idempotent: bool,
) -> PublishedSnapshot:
    checkpoint = manifest["checkpoint"]
    assert isinstance(checkpoint, dict)
    return PublishedSnapshot(
        snapshot_dir=snapshot_dir,
        manifest_path=snapshot_dir / SNAPSHOT_MANIFEST,
        epoch_completed=int(manifest["epoch_completed"]),
        checkpoint_relative_path=str(checkpoint["path"]),
        checkpoint_sha256=str(checkpoint["sha256"]),
        context=context,
        files=files,
        idempotent=idempotent,
    )


def publish_epoch_snapshot(
    local_run_dir: str | os.PathLike[str],
    destination_run_root: str | os.PathLike[str],
    *,
    epoch_completed: int,
    context: SnapshotContext | Mapping[str, str],
    checkpoint_relative_path: str | os.PathLike[str] = "checkpoints/last.pt",
    files: Iterable[str | os.PathLike[str]] = (),
) -> PublishedSnapshot:
    """Publish one immutable epoch snapshot under ``destination_run_root``.

    ``checkpoint_relative_path`` and its ``.sha256`` sidecar are always copied.
    Additional ``files`` are relative to ``local_run_dir``.  A completed
    ``epoch_XXXXXX`` directory is never overwritten; the call is idempotent
    only when its manifest and every byte are already exactly identical.
    """

    snapshot_name = _snapshot_name(epoch_completed)
    normalized_context = _normalize_context(context)
    local_root = Path(local_run_dir).expanduser().resolve(strict=True)
    destination_root = Path(destination_run_root).expanduser().resolve()
    checkpoint_relative = _relative_path(checkpoint_relative_path)
    sidecar_relative = _checkpoint_sidecar_relative(checkpoint_relative)
    selected = [checkpoint_relative, sidecar_relative]
    selected.extend(_relative_path(path) for path in files)
    if SNAPSHOT_MANIFEST in selected:
        raise ValueError(f"{SNAPSHOT_MANIFEST} is reserved")
    if len(selected) != len(set(selected)):
        raise ValueError("snapshot file selection contains duplicates")

    records = tuple(
        sorted(
            (_file_record(_local_source(local_root, relative), relative) for relative in selected),
            key=lambda item: item.path,
        )
    )
    by_path = {record.path: record for record in records}
    checkpoint_source = _local_source(local_root, checkpoint_relative)
    sidecar_source = _local_source(local_root, sidecar_relative)
    _verify_checkpoint_sidecar(
        checkpoint_source,
        sidecar_source,
        by_path[checkpoint_relative].sha256,
    )
    expected_manifest = _canonical_manifest(
        epoch_completed=epoch_completed,
        context=normalized_context,
        checkpoint_relative=checkpoint_relative,
        files=records,
    )

    snapshots_root = destination_root / "snapshots"
    snapshots_root.mkdir(parents=True, exist_ok=True)
    final = snapshots_root / snapshot_name
    staging = snapshots_root / f"{snapshot_name}.uploading"
    if final.exists():
        actual_manifest, actual_context, actual_files = _decode_and_verify_manifest(
            final, expected_context=normalized_context
        )
        if actual_manifest != expected_manifest:
            raise SnapshotConflictError(f"completed snapshot already differs: {final}")
        return _published_result(
            final, actual_manifest, actual_context, actual_files, idempotent=True
        )
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=False)
    try:
        for record in records:
            source = _local_source(local_root, record.path)
            destination = staging.joinpath(*PurePosixPath(record.path).parts)
            _copy_verified(source, destination, record)
        # This is intentionally the final file created inside the staging tree.
        manifest_path = staging / SNAPSHOT_MANIFEST
        with manifest_path.open("xb") as handle:
            handle.write(_manifest_bytes(expected_manifest))
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory_best_effort(staging)
        try:
            os.rename(staging, final)
        except FileExistsError:
            # A concurrent publisher won. Accept it only if it is byte-exact.
            actual_manifest, actual_context, actual_files = _decode_and_verify_manifest(
                final, expected_context=normalized_context
            )
            if actual_manifest != expected_manifest:
                raise SnapshotConflictError(f"completed snapshot already differs: {final}")
            return _published_result(
                final, actual_manifest, actual_context, actual_files, idempotent=True
            )
        _fsync_directory_best_effort(snapshots_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    actual_manifest, actual_context, actual_files = _decode_and_verify_manifest(
        final, expected_context=normalized_context
    )
    if actual_manifest != expected_manifest:
        raise SnapshotValidationError("published snapshot differs from its source manifest")
    return _published_result(
        final, actual_manifest, actual_context, actual_files, idempotent=False
    )


def _snapshot_candidates(destination_run_root: Path) -> list[Path]:
    root = destination_run_root / "snapshots"
    if not root.is_dir():
        return []
    candidates: list[tuple[int, Path]] = []
    for path in root.iterdir():
        if not path.is_dir() or path.name.endswith(".uploading"):
            continue
        prefix = "epoch_"
        suffix = path.name[len(prefix) :] if path.name.startswith(prefix) else ""
        if len(suffix) >= 6 and suffix.isdigit():
            candidates.append((int(suffix), path))
    return [path for _, path in sorted(candidates, reverse=True)]


def _replace_checkpoint_pair(
    source_checkpoint: Path,
    source_sidecar: Path,
    destination_checkpoint: Path,
    expected_digest: str,
) -> None:
    destination_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    destination_sidecar = destination_checkpoint.with_name(destination_checkpoint.name + ".sha256")
    nonce = uuid.uuid4().hex
    checkpoint_temp = destination_checkpoint.with_name(f".{destination_checkpoint.name}.{nonce}.tmp")
    sidecar_temp = destination_sidecar.with_name(f".{destination_sidecar.name}.{nonce}.tmp")
    checkpoint_backup = destination_checkpoint.with_name(f".{destination_checkpoint.name}.{nonce}.bak")
    sidecar_backup = destination_sidecar.with_name(f".{destination_sidecar.name}.{nonce}.bak")
    try:
        expected_checkpoint = SnapshotFile(
            path=destination_checkpoint.name,
            bytes=source_checkpoint.stat().st_size,
            sha256=expected_digest,
        )
        _copy_verified(source_checkpoint, checkpoint_temp, expected_checkpoint)
        expected_sidecar = _file_record(source_sidecar, source_sidecar.name)
        _copy_verified(source_sidecar, sidecar_temp, expected_sidecar)
        _verify_checkpoint_sidecar(
            checkpoint_temp,
            sidecar_temp,
            expected_digest,
            checkpoint_name=source_checkpoint.name,
        )

        if destination_checkpoint.exists():
            os.replace(destination_checkpoint, checkpoint_backup)
        if destination_sidecar.exists():
            os.replace(destination_sidecar, sidecar_backup)
        try:
            os.replace(checkpoint_temp, destination_checkpoint)
            os.replace(sidecar_temp, destination_sidecar)
            _fsync_directory_best_effort(destination_checkpoint.parent)
        except Exception:
            destination_checkpoint.unlink(missing_ok=True)
            destination_sidecar.unlink(missing_ok=True)
            if checkpoint_backup.exists():
                os.replace(checkpoint_backup, destination_checkpoint)
            if sidecar_backup.exists():
                os.replace(sidecar_backup, destination_sidecar)
            raise
    finally:
        checkpoint_temp.unlink(missing_ok=True)
        sidecar_temp.unlink(missing_ok=True)
        checkpoint_backup.unlink(missing_ok=True)
        sidecar_backup.unlink(missing_ok=True)


def restore_latest_snapshot(
    destination_run_root: str | os.PathLike[str],
    local_checkpoint_path: str | os.PathLike[str],
    *,
    expected_context: SnapshotContext | Mapping[str, str],
    checkpoint_relative_path: str | os.PathLike[str] | None = None,
) -> RestoredSnapshot | None:
    """Restore the newest valid compatible snapshot, falling back on damage.

    Candidates are checked newest-to-oldest.  Incomplete ``.uploading`` trees,
    corrupt bytes, unsupported schemas, and context mismatches are skipped.
    ``None`` is returned when no snapshot is safe to resume.
    """

    root = Path(destination_run_root).expanduser().resolve()
    destination = Path(local_checkpoint_path).expanduser().resolve()
    context = _normalize_context(expected_context)
    requested = None if checkpoint_relative_path is None else _relative_path(checkpoint_relative_path)
    for candidate in _snapshot_candidates(root):
        try:
            manifest, saved_context, files = _decode_and_verify_manifest(
                candidate, expected_context=context
            )
        except SnapshotValidationError:
            continue
        checkpoint = manifest["checkpoint"]
        assert isinstance(checkpoint, dict)
        relative = str(checkpoint["path"])
        if requested is not None and relative != requested:
            continue
        sidecar_relative = str(checkpoint["sidecar_path"])
        source_checkpoint = candidate.joinpath(*PurePosixPath(relative).parts)
        source_sidecar = candidate.joinpath(*PurePosixPath(sidecar_relative).parts)
        digest = str(checkpoint["sha256"])
        try:
            _replace_checkpoint_pair(
                source_checkpoint,
                source_sidecar,
                destination,
                digest,
            )
        except (OSError, SnapshotError):
            continue
        return RestoredSnapshot(
            snapshot_dir=candidate,
            manifest_path=candidate / SNAPSHOT_MANIFEST,
            epoch_completed=int(manifest["epoch_completed"]),
            source_checkpoint_path=source_checkpoint,
            local_checkpoint_path=destination,
            checkpoint_sha256=digest,
            context=saved_context,
            files=files,
        )
    return None


__all__ = [
    "PublishedSnapshot",
    "RestoredSnapshot",
    "SNAPSHOT_KIND",
    "SNAPSHOT_MANIFEST",
    "SNAPSHOT_SCHEMA_VERSION",
    "SnapshotConflictError",
    "SnapshotContext",
    "SnapshotError",
    "SnapshotFile",
    "SnapshotValidationError",
    "publish_epoch_snapshot",
    "restore_latest_snapshot",
]
