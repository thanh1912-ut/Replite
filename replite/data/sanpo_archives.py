"""Verified, out-of-core access to downloader-produced SANPO archives.

The SANPO download notebook stores one ``.tar.zst`` archive per
``(official split, session, camera)`` on Google Drive.  This module joins the
selection manifest to the append-only archive ledger, freezes a leak-free
train/validation split by *session*, and feeds one verified archive at a time
to :class:`~replite.data.sanpo_joint.SanpoJointDataset`.

Absolute archive paths recorded by Colab are deliberately treated as
provenance only.  A catalog always resolves an archive below the caller's
current ``drive_root/archives/<split>`` using only the validated basename.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import shutil
import tarfile
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

import torch
from torch.utils.data import DataLoader

from .sanpo_joint import SanpoJointDataset, load_sanpo_joint_manifest, sanpo_joint_collate


_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_ARCHIVE_NAME_RE = re.compile(r"[A-Za-z0-9._-]+\.tar\.zst")
_SPLITS = frozenset({"train", "test"})
_SENSORS = frozenset({"camera_head", "camera_chest"})
_STAGE_MARKER = ".replite_owned_sanpo_stage.json"


def canonical_json_bytes(value: object) -> bytes:
    """Return the canonical UTF-8 representation used by manifest hashes."""

    try:
        text = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("value is not canonical JSON data") from exc
    return text.encode("utf-8")


def canonical_json_sha256(value: object) -> str:
    """Hash JSON independently of whitespace and mapping insertion order."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _read_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {description}: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must contain a JSON object")
    return payload


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _identity(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(f"{name} is invalid")
    return value


def _record_id(record: "SanpoArchiveRecord") -> dict[str, object]:
    return {
        "split": record.split,
        "session_id": record.session_id,
        "sensor": record.sensor,
        "annotation_policy": record.annotation_policy,
        "selection_sha256": record.selection_sha256,
        "detection_config_sha256": record.detection_config_sha256,
        "package_sha256": record.package_sha256,
        "archive_name": record.archive_path.name,
        "archive_sha256": record.archive_sha256,
        "archive_bytes": record.archive_bytes,
        "joint_frames": record.joint_frames,
    }


def _package_sha256(selection_sha256: str, detection_config_sha256: str) -> str:
    return canonical_json_sha256(
        {
            "selection_sha256": selection_sha256,
            "detection_config_sha256": detection_config_sha256,
        }
    )


def _detection_config_sha256_hint(path: Path) -> str | None:
    """Return a validated config hint when the metadata wrapper carries one.

    Early downloader metadata contained only the derived class taxonomy.  In
    that case the exact config can still be recovered from the unique package
    family that covers the complete immutable selection in the append-only
    ledger, so absence of these optional fields is not an error.
    """

    payload = _read_json_object(path, "SANPO derived-detection manifest")
    schema_version = payload.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version < 1
    ):
        raise ValueError(
            "derived-detection manifest schema_version must be a positive integer"
        )
    # The downloader has used more than one wrapper schema.  Package selection
    # depends only on these two structurally validated fields, not on the
    # wrapper version, so compatible old/new manifests remain reproducible.
    config = payload.get("detection_config")
    declared_raw = payload.get("detection_config_sha256")
    if config is None and declared_raw is None:
        return None
    if config is None:
        return _sha256(
            declared_raw, "derived-detection detection_config_sha256"
        )
    if not isinstance(config, Mapping):
        raise ValueError("derived-detection detection_config must be a mapping")
    declared = _sha256(
        declared_raw, "derived-detection detection_config_sha256"
    )
    actual = canonical_json_sha256(dict(config))
    if actual != declared:
        raise ValueError("derived-detection config SHA-256 does not match its content")
    return declared


@dataclass(frozen=True)
class SanpoArchiveRecord:
    """Validated identity and local Drive path for one session-camera shard."""

    split: str
    session_id: str
    sensor: str
    annotation_policy: str
    selection_sha256: str
    detection_config_sha256: str
    package_sha256: str
    archive_path: Path
    archive_bytes: int
    archive_sha256: str
    joint_frames: int

    @property
    def selection_key(self) -> tuple[str, str, str, str]:
        """Return the source-selection identity, excluding derived labels."""

        return self.split, self.session_id, self.sensor, self.selection_sha256

    @property
    def key(self) -> tuple[str, str, str, str, str]:
        """Return the immutable packaged-shard identity."""

        return (*self.selection_key, self.package_sha256)


@dataclass(frozen=True)
class ArchiveCatalog:
    """Immutable view of the archives selected by the downloader."""

    drive_root: Path
    records: tuple[SanpoArchiveRecord, ...]
    annotation_policy: str
    detection_config_sha256: str
    catalog_sha256: str

    @property
    def train_records(self) -> tuple[SanpoArchiveRecord, ...]:
        return tuple(record for record in self.records if record.split == "train")

    @property
    def test_records(self) -> tuple[SanpoArchiveRecord, ...]:
        return tuple(record for record in self.records if record.split == "test")


def _selection_record(raw: object, index: int) -> dict[str, object]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"selection record {index} must be a mapping")
    split = raw.get("split")
    sensor = raw.get("sensor")
    if split not in _SPLITS:
        raise ValueError(f"selection record {index} has an invalid split")
    if sensor not in _SENSORS:
        raise ValueError(f"selection record {index} has an invalid sensor")
    return {
        "split": split,
        "session_id": _identity(raw.get("session_id"), "selection session_id"),
        "sensor": sensor,
        "selection_sha256": _sha256(
            raw.get("selection_sha256"), "selection selection_sha256"
        ),
        "joint_frames": _positive_int(
            raw.get("joint_frames"), "selection joint_frames"
        ),
    }


def _ledger_record(
    raw: object,
    *,
    drive_root: Path,
    expected_annotation_policy: str,
    expected_detection_config_sha256: str,
) -> SanpoArchiveRecord:
    if not isinstance(raw, Mapping):
        raise ValueError("archive ledger entry must be a mapping")
    split = raw.get("split")
    sensor = raw.get("sensor")
    if split not in _SPLITS:
        raise ValueError("archive ledger entry has an invalid split")
    if sensor not in _SENSORS:
        raise ValueError("archive ledger entry has an invalid sensor")
    session_id = _identity(raw.get("session_id"), "archive session_id")
    annotation_policy = _identity(
        raw.get("annotation_policy"), "archive annotation_policy"
    )
    if annotation_policy != expected_annotation_policy:
        raise ValueError("archive annotation_policy disagrees with selection")
    selection_sha256 = _sha256(
        raw.get("selection_sha256"), "archive selection_sha256"
    )
    detection_config_sha256 = _sha256(
        raw.get("detection_config_sha256"),
        "archive detection_config_sha256",
    )
    if detection_config_sha256 != expected_detection_config_sha256:
        raise ValueError("archive detection config is not the active Drive config")
    package_sha256 = _sha256(raw.get("package_sha256"), "archive package_sha256")
    expected_package_sha256 = _package_sha256(
        selection_sha256, detection_config_sha256
    )
    if package_sha256 != expected_package_sha256:
        raise ValueError("archive package_sha256 does not match its provenance")
    archive_value = raw.get("archive")
    if not isinstance(archive_value, str) or not archive_value:
        raise ValueError("archive ledger path must be a non-empty string")
    archive_name = Path(archive_value).name
    if _ARCHIVE_NAME_RE.fullmatch(archive_name) is None:
        raise ValueError("archive ledger path has an invalid basename")

    # Never trust the absolute path persisted by a previous Colab mount.
    archive_path = (drive_root / "archives" / split / archive_name).resolve()
    expected_parent = (drive_root / "archives" / split).resolve()
    try:
        archive_path.relative_to(expected_parent)
    except ValueError as exc:  # defensive; the basename regex already prevents this
        raise ValueError("resolved archive escapes its official split") from exc

    record = SanpoArchiveRecord(
        split=split,
        session_id=session_id,
        sensor=sensor,
        annotation_policy=annotation_policy,
        selection_sha256=selection_sha256,
        detection_config_sha256=detection_config_sha256,
        package_sha256=package_sha256,
        archive_path=archive_path,
        archive_bytes=_positive_int(raw.get("archive_bytes"), "archive_bytes"),
        archive_sha256=_sha256(raw.get("archive_sha256"), "archive_sha256"),
        joint_frames=_positive_int(raw.get("joint_frames"), "archive joint_frames"),
    )
    if not record.archive_path.is_file():
        raise FileNotFoundError(f"SANPO archive is missing: {record.archive_path}")
    if record.archive_path.stat().st_size != record.archive_bytes:
        raise ValueError(f"SANPO archive byte count disagrees with ledger: {record.archive_path}")
    return record


def _validate_sidecars(record: SanpoArchiveRecord) -> None:
    manifest_path = record.archive_path.with_name(record.archive_path.name + ".manifest.json")
    if manifest_path.exists():
        payload = _read_json_object(manifest_path, "SANPO archive sidecar")
        if payload.get("schema_version") != 1 or not isinstance(payload.get("entry"), Mapping):
            raise ValueError(f"unsupported SANPO archive sidecar: {manifest_path}")
        entry = payload["entry"]
        comparisons = {
            "split": record.split,
            "session_id": record.session_id,
            "sensor": record.sensor,
            "annotation_policy": record.annotation_policy,
            "selection_sha256": record.selection_sha256,
            "detection_config_sha256": record.detection_config_sha256,
            "package_sha256": record.package_sha256,
            "archive_bytes": record.archive_bytes,
            "archive_sha256": record.archive_sha256,
            "joint_frames": record.joint_frames,
        }
        for name, expected in comparisons.items():
            if entry.get(name) != expected:
                raise ValueError(f"archive sidecar field {name!r} disagrees with ledger")

    sha_path = record.archive_path.with_name(record.archive_path.name + ".sha256")
    if sha_path.exists():
        try:
            fields = sha_path.read_text(encoding="utf-8").strip().split()
        except OSError as exc:
            raise ValueError(f"cannot read archive SHA sidecar: {sha_path}") from exc
        if fields != [record.archive_sha256, record.archive_path.name]:
            raise ValueError(f"archive SHA sidecar disagrees with ledger: {sha_path}")


def load_archive_catalog(
    drive_root: str | os.PathLike[str],
    *,
    selection_path: str | os.PathLike[str] | None = None,
    ledger_path: str | os.PathLike[str] | None = None,
    detection_manifest_path: str | os.PathLike[str] | None = None,
    validate_sidecars: bool = True,
) -> ArchiveCatalog:
    """Resolve the active packaged archive for every selected source shard.

    ``selection_sha256`` intentionally excludes the derived-box policy, so an
    append-only ledger may contain several legitimate packages for the same
    source frames.  A structurally valid metadata hint is preferred; otherwise
    the active config is the unique package family covering every record in the
    immutable current selection.  Historical partial package families are
    retained on Drive but ignored here.
    """

    root = Path(drive_root).expanduser().resolve()
    selection_file = (
        Path(selection_path).expanduser().resolve()
        if selection_path is not None
        else root / "metadata" / "current_download_selection.json"
    )
    ledger_file = (
        Path(ledger_path).expanduser().resolve()
        if ledger_path is not None
        else root / "archive_manifest.json"
    )
    detection_manifest_file = (
        Path(detection_manifest_path).expanduser().resolve()
        if detection_manifest_path is not None
        else root / "metadata" / "derived_detection_classes.json"
    )
    selection = _read_json_object(selection_file, "SANPO download selection")
    ledger = _read_json_object(ledger_file, "SANPO archive ledger")
    detection_config_hint = (
        _detection_config_sha256_hint(detection_manifest_file)
        if detection_manifest_file.is_file()
        else None
    )
    if selection.get("schema_version") != 1:
        raise ValueError("unsupported SANPO download selection schema")
    if ledger.get("schema_version") != 2:
        raise ValueError("unsupported SANPO archive ledger schema")
    annotation_policy = _identity(
        selection.get("annotation_policy"), "selection annotation_policy"
    )
    if ledger.get("dataset") != f"SANPO-Real-v0-joint-{annotation_policy}":
        raise ValueError("archive ledger dataset disagrees with selection policy")
    if not isinstance(selection.get("records"), list) or not selection["records"]:
        raise ValueError("SANPO download selection has no records")
    if not isinstance(ledger.get("archives"), Mapping):
        raise ValueError("SANPO archive ledger archives must be a mapping")

    selected = [
        _selection_record(raw, index)
        for index, raw in enumerate(selection["records"])
    ]
    selected_keys = [
        (item["split"], item["session_id"], item["sensor"], item["selection_sha256"])
        for item in selected
    ]
    if len(selected_keys) != len(set(selected_keys)):
        raise ValueError("SANPO download selection contains duplicate join keys")
    if (
        "session_camera_count" in selection
        and selection["session_camera_count"] != len(selected)
    ):
        raise ValueError("selection session_camera_count does not match records")
    if "joint_target_count" in selection and selection["joint_target_count"] != sum(
        int(item["joint_frames"]) for item in selected
    ):
        raise ValueError("selection joint_target_count does not match records")

    selected_key_set = set(selected_keys)
    package_families: dict[
        str,
        dict[tuple[str, str, str, str], list[Mapping[str, object]]],
    ] = {}
    for raw in ledger["archives"].values():
        # The ledger is append-only and can retain records from older download
        # selections.  Resolve/stat only entries that can join the current
        # immutable selection; unrelated historical files may no longer exist.
        if not isinstance(raw, Mapping):
            continue
        raw_key = (
            raw.get("split"),
            raw.get("session_id"),
            raw.get("sensor"),
            raw.get("selection_sha256"),
        )
        if raw_key not in selected_key_set:
            continue
        if raw.get("annotation_policy") != annotation_policy:
            continue
        try:
            candidate_config = _sha256(
                raw.get("detection_config_sha256"),
                "archive detection_config_sha256",
            )
            candidate_package = _sha256(
                raw.get("package_sha256"), "archive package_sha256"
            )
        except ValueError:
            continue
        expected_package = _package_sha256(str(raw_key[3]), candidate_config)
        if candidate_package != expected_package:
            continue
        package_families.setdefault(candidate_config, {}).setdefault(
            raw_key, []
        ).append(raw)

    complete_families = sorted(
        config_sha
        for config_sha, by_key in package_families.items()
        if set(by_key) == selected_key_set
    )
    if detection_config_hint in complete_families:
        detection_config_sha256 = str(detection_config_hint)
    elif len(complete_families) == 1:
        detection_config_sha256 = complete_families[0]
    else:
        coverage = ", ".join(
            f"{config_sha[:12]}={len(by_key)}/{len(selected_key_set)}"
            for config_sha, by_key in sorted(package_families.items())
        ) or "none"
        raise ValueError(
            "cannot resolve one SANPO detection package family covering the "
            f"complete selection; metadata_hint={detection_config_hint!r}, "
            f"coverage=[{coverage}]"
        )

    ledger_by_key: dict[tuple[str, str, str, str], list[SanpoArchiveRecord]] = {}
    for raw_key, candidates in package_families[detection_config_sha256].items():
        for raw in candidates:
            record = _ledger_record(
                raw,
                drive_root=root,
                expected_annotation_policy=annotation_policy,
                expected_detection_config_sha256=detection_config_sha256,
            )
            ledger_by_key.setdefault(raw_key, []).append(record)

    joined: list[SanpoArchiveRecord] = []
    for item, key in zip(selected, selected_keys):
        matches = ledger_by_key.get(key, [])
        distinct = {
            canonical_json_sha256(_record_id(record)): record
            for record in matches
        }
        if len(distinct) != 1:
            raise ValueError(
                "each selected SANPO shard needs exactly one active package; "
                f"found {len(distinct)} for {key[:3]} with detection config "
                f"{detection_config_sha256[:12]}"
            )
        record = next(iter(distinct.values()))
        if record.joint_frames != item["joint_frames"]:
            raise ValueError(f"selection and ledger joint_frames disagree for {key[:3]}")
        if validate_sidecars:
            _validate_sidecars(record)
        joined.append(record)

    records = tuple(
        sorted(
            joined,
            key=lambda item: (
                item.split,
                item.session_id,
                item.sensor,
                item.archive_sha256,
            ),
        )
    )
    fingerprint_payload = {
        "schema_version": 1,
        "annotation_policy": annotation_policy,
        "detection_config_sha256": detection_config_sha256,
        "records": [_record_id(record) for record in records],
    }
    return ArchiveCatalog(
        drive_root=root,
        records=records,
        annotation_policy=annotation_policy,
        detection_config_sha256=detection_config_sha256,
        catalog_sha256=canonical_json_sha256(fingerprint_payload),
    )


@dataclass(frozen=True)
class ArchiveGroupSplit:
    """A frozen official-train split plus untouched official-test records."""

    train_records: tuple[SanpoArchiveRecord, ...]
    validation_records: tuple[SanpoArchiveRecord, ...]
    official_test_records: tuple[SanpoArchiveRecord, ...]
    seed: int
    validation_fraction: float
    catalog_sha256: str
    manifest_path: Path
    manifest_sha256: str


def _group_split_payload(
    catalog: ArchiveCatalog,
    *,
    seed: int,
    validation_fraction: float,
) -> tuple[dict[str, object], set[str], set[str]]:
    train_sessions = sorted({record.session_id for record in catalog.train_records})
    if len(train_sessions) < 2:
        raise ValueError("at least two official-train sessions are required")
    ordered = sorted(
        train_sessions,
        key=lambda session_id: (
            hashlib.sha256(
                f"replite-sanpo-split-v1:{seed}:{session_id}".encode()
            ).hexdigest(),
            session_id,
        ),
    )
    validation_count = max(
        1,
        min(len(ordered) - 1, round(len(ordered) * validation_fraction)),
    )
    validation_sessions = set(ordered[:validation_count])
    fit_sessions = set(ordered[validation_count:])
    test_sessions = sorted({record.session_id for record in catalog.test_records})
    if (fit_sessions | validation_sessions) & set(test_sessions):
        raise ValueError("official test session_id overlaps official train")
    payload: dict[str, object] = {
        "schema_version": 1,
        "dataset": "SANPO-Real-v0-joint",
        "grouping_unit": "session_id",
        "ordering": "sha256(replite-sanpo-split-v1:seed:session_id)",
        "seed": seed,
        "validation_fraction": validation_fraction,
        "catalog_sha256": catalog.catalog_sha256,
        "train_session_ids": sorted(fit_sessions),
        "validation_session_ids": sorted(validation_sessions),
        "official_test_session_ids": test_sessions,
        "protocol_note": "Official test is excluded from training, validation, and selection.",
    }
    payload["manifest_sha256"] = canonical_json_sha256(payload)
    return payload, fit_sessions, validation_sessions


def _write_json_immutable(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        existing = _read_json_object(path, "immutable SANPO split manifest")
        if existing != dict(payload):
            raise FileExistsError(
                f"existing SANPO split manifest differs; choose a new path: {path}"
            ) from None


def create_or_load_group_split(
    catalog: ArchiveCatalog,
    manifest_path: str | os.PathLike[str],
    *,
    seed: int = 42,
    validation_fraction: float = 0.15,
) -> ArchiveGroupSplit:
    """Create or verify an immutable, deterministic session-level split."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if (
        isinstance(validation_fraction, bool)
        or not isinstance(validation_fraction, (int, float))
        or not math.isfinite(float(validation_fraction))
        or not 0.0 < float(validation_fraction) < 1.0
    ):
        raise ValueError("validation_fraction must be finite and between zero and one")
    fraction = float(validation_fraction)
    payload, train_sessions, validation_sessions = _group_split_payload(
        catalog, seed=seed, validation_fraction=fraction
    )
    path = Path(manifest_path).expanduser().resolve()
    _write_json_immutable(path, payload)

    train_records = tuple(
        record for record in catalog.train_records if record.session_id in train_sessions
    )
    validation_records = tuple(
        record for record in catalog.train_records if record.session_id in validation_sessions
    )
    if not train_records or not validation_records:
        raise ValueError("session grouping produced an empty training or validation set")
    if {record.session_id for record in train_records} & {
        record.session_id for record in validation_records
    }:
        raise AssertionError("session-level train/validation leakage")
    return ArchiveGroupSplit(
        train_records=train_records,
        validation_records=validation_records,
        official_test_records=catalog.test_records,
        seed=seed,
        validation_fraction=fraction,
        catalog_sha256=catalog.catalog_sha256,
        manifest_path=path,
        manifest_sha256=str(payload["manifest_sha256"]),
    )


def _copy_verified(record: SanpoArchiveRecord, destination: Path) -> None:
    digest = hashlib.sha256()
    size = 0
    with record.archive_path.open("rb") as source, destination.open("xb") as target:
        while block := source.read(16 * 1024 * 1024):
            target.write(block)
            digest.update(block)
            size += len(block)
        target.flush()
        os.fsync(target.fileno())
    if size != record.archive_bytes or digest.hexdigest() != record.archive_sha256:
        raise ValueError(f"SANPO archive failed byte/SHA verification: {record.archive_path}")


@contextlib.contextmanager
def _zstd_stream_reader(path: Path) -> Iterator[BinaryIO]:
    try:
        import zstandard
    except ImportError as exc:  # pragma: no cover - depends on the caller environment
        raise RuntimeError(
            "Archive extraction requires `zstandard`; install it before creating a loader"
        ) from exc
    with path.open("rb") as compressed:
        with zstandard.ZstdDecompressor().stream_reader(compressed) as reader:
            yield reader


def _safe_tar_parts(name: str) -> tuple[str, ...]:
    if not name or "\x00" in name or "\\" in name:
        raise ValueError(f"unsafe tar member name: {name!r}")
    trimmed = name.rstrip("/")
    if not trimmed:
        raise ValueError(f"unsafe tar member name: {name!r}")
    posix = PurePosixPath(trimmed)
    raw_parts = trimmed.split("/")
    if (
        posix.is_absolute()
        or any(part in {"", ".", ".."} for part in raw_parts)
        or not raw_parts
        or raw_parts[0] != "sanpo-real"
    ):
        raise ValueError(f"tar member is not safely rooted under sanpo-real: {name!r}")
    return tuple(raw_parts)


def _extract_tar_zst_safely(archive: Path, destination: Path) -> int:
    seen: set[tuple[str, ...]] = set()
    member_count = 0
    with _zstd_stream_reader(archive) as stream:
        with tarfile.open(fileobj=stream, mode="r|") as tar:
            for member in tar:
                parts = _safe_tar_parts(member.name)
                if parts in seen:
                    raise ValueError(f"duplicate tar member: {member.name!r}")
                seen.add(parts)
                target = destination.joinpath(*parts)
                try:
                    target.resolve(strict=False).relative_to(destination.resolve())
                except ValueError as exc:
                    raise ValueError(
                        f"tar member escapes extraction root: {member.name!r}"
                    ) from exc
                if member.isdir():
                    if target.exists() and not target.is_dir():
                        raise ValueError(
                            f"tar directory collides with a file: {member.name!r}"
                        )
                    target.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    source = tar.extractfile(member)
                    if source is None:
                        raise ValueError(f"cannot read regular tar member: {member.name!r}")
                    with source, target.open("xb") as output:
                        shutil.copyfileobj(source, output, length=1024 * 1024)
                else:
                    raise ValueError(
                        f"tar links, devices, and special members are forbidden: {member.name!r}"
                    )
                member_count += 1
    if member_count == 0:
        raise ValueError("SANPO archive is empty")
    return member_count


class ArchiveMaterializer:
    """Copy, verify, and safely extract exactly one SANPO archive.

    The returned joint-manifest path remains valid only inside the context.
    Both successful and failed exits remove the uniquely owned stage.
    """

    def __init__(
        self,
        record: SanpoArchiveRecord,
        *,
        local_root: str | os.PathLike[str],
    ) -> None:
        if not isinstance(record, SanpoArchiveRecord):
            raise TypeError("record must be a SanpoArchiveRecord")
        self.record = record
        self.local_root = Path(local_root).expanduser().resolve()
        self._stage: Path | None = None

    def _cleanup(self) -> None:
        stage = self._stage
        if stage is None or not stage.exists():
            self._stage = None
            return
        try:
            stage.resolve().relative_to(self.local_root)
        except ValueError as exc:
            raise RuntimeError("refusing to clean a stage outside local_root") from exc
        marker = stage / _STAGE_MARKER
        try:
            marker_payload = _read_json_object(marker, "owned SANPO stage marker")
        except ValueError as exc:
            raise RuntimeError(f"refusing to clean an unowned SANPO stage: {stage}") from exc
        if marker_payload.get("archive_sha256") != self.record.archive_sha256:
            raise RuntimeError(f"refusing to clean a SANPO stage with a foreign marker: {stage}")
        shutil.rmtree(stage)
        self._stage = None

    def __enter__(self) -> Path:
        if self._stage is not None:
            raise RuntimeError("ArchiveMaterializer cannot be entered twice")
        self.local_root.mkdir(parents=True, exist_ok=True)
        self._stage = Path(tempfile.mkdtemp(prefix="sanpo-shard-", dir=self.local_root))
        marker = self._stage / _STAGE_MARKER
        marker.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "archive_sha256": self.record.archive_sha256,
                    "record_key": list(self.record.key),
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        try:
            local_archive = self._stage / self.record.archive_path.name
            _copy_verified(self.record, local_archive)
            extracted = self._stage / "extracted"
            extracted.mkdir()
            _extract_tar_zst_safely(local_archive, extracted)
            local_archive.unlink()
            manifests = list(extracted.rglob("_sanpo_joint_manifest.json"))
            if len(manifests) != 1:
                raise ValueError(
                    "each SANPO archive must contain exactly one _sanpo_joint_manifest.json"
                )
            manifest, info = load_sanpo_joint_manifest(manifests[0])
            detection = manifest.get("detection")
            if (
                info.official_split != self.record.split
                or info.session_id != self.record.session_id
                or info.sensor != self.record.sensor
                or info.sample_count != self.record.joint_frames
                or manifest.get("annotation_policy") != self.record.annotation_policy
                or manifest.get("selection_sha256") != self.record.selection_sha256
                or not isinstance(detection, Mapping)
                or detection.get("config_sha256")
                != self.record.detection_config_sha256
            ):
                raise ValueError("extracted SANPO manifest disagrees with archive catalog")
            return manifests[0]
        except BaseException:
            self._cleanup()
            raise

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self._cleanup()


def _seed_from_parts(*parts: object) -> int:
    encoded = ":".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") % (2**63)


class ArchiveShardLoader:
    """Iterable of batches backed by one verified archive at a time.

    ``len(loader)`` is exact for the per-archive batching policy.  With
    ``drop_last=True`` each shard drops its own incomplete final batch.
    ``persistent_workers`` is intentionally fixed to ``False`` so an archive
    can be removed as soon as its iterator finishes.
    """

    def __init__(
        self,
        records: Sequence[SanpoArchiveRecord],
        *,
        local_root: str | os.PathLike[str],
        batch_size: int,
        image_size: Sequence[int] = (288, 512),
        seed: int = 42,
        shuffle: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
        prefetch_factor: int = 2,
        dataset_kwargs: Mapping[str, object] | None = None,
    ) -> None:
        self.records = tuple(records)
        if any(not isinstance(record, SanpoArchiveRecord) for record in self.records):
            raise TypeError("records must contain only SanpoArchiveRecord values")
        if len({record.key for record in self.records}) != len(self.records):
            raise ValueError("records contains duplicate archive join keys")
        self.local_root = Path(local_root).expanduser().resolve()
        self.batch_size = _positive_int(batch_size, "batch_size")
        try:
            checked_image_size = tuple(image_size)
        except TypeError as exc:
            raise ValueError("image_size must contain two positive integers") from exc
        if len(checked_image_size) != 2 or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in checked_image_size
        ):
            raise ValueError("image_size must contain two positive integers")
        self.image_size = tuple(int(value) for value in checked_image_size)
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if (
            isinstance(num_workers, bool)
            or not isinstance(num_workers, int)
            or num_workers < 0
        ):
            raise ValueError("num_workers must be a non-negative integer")
        self.seed = seed
        self.shuffle = bool(shuffle)
        self.num_workers = num_workers
        self.pin_memory = bool(pin_memory)
        self.drop_last = bool(drop_last)
        self.prefetch_factor = _positive_int(prefetch_factor, "prefetch_factor")
        self.dataset_kwargs = dict(dataset_kwargs or {})
        if "image_size" in self.dataset_kwargs:
            raise ValueError("pass image_size directly, not through dataset_kwargs")
        self._epoch = 0
        self._iterating = False

    @property
    def sample_count(self) -> int:
        return sum(record.joint_frames for record in self.records)

    @property
    def epoch(self) -> int:
        return self._epoch

    def __len__(self) -> int:
        if self.drop_last:
            return sum(record.joint_frames // self.batch_size for record in self.records)
        return sum(math.ceil(record.joint_frames / self.batch_size) for record in self.records)

    def set_epoch(self, epoch: int) -> None:
        """Select the deterministic shard and sample ordering for an epoch."""

        if self._iterating:
            raise RuntimeError("cannot change epoch during ArchiveShardLoader iteration")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        self._epoch = epoch

    def _ordered_records(self) -> tuple[SanpoArchiveRecord, ...]:
        if not self.shuffle:
            return tuple(
                sorted(self.records, key=lambda record: (record.session_id, record.sensor))
            )
        return tuple(
            sorted(
                self.records,
                key=lambda record: (
                    _seed_from_parts(
                        "replite-sanpo-shards-v1",
                        self.seed,
                        self._epoch,
                        *record.key,
                    ),
                    record.key,
                ),
            )
        )

    def __iter__(self) -> Iterator[tuple[torch.Tensor, dict[str, Any]]]:
        if self._iterating:
            raise RuntimeError("ArchiveShardLoader does not support concurrent iteration")
        self._iterating = True
        try:
            for record in self._ordered_records():
                with ArchiveMaterializer(record, local_root=self.local_root) as manifest_path:
                    dataset = SanpoJointDataset(
                        manifest_path,
                        image_size=self.image_size,
                        **self.dataset_kwargs,
                    )
                    if len(dataset) != record.joint_frames:
                        raise ValueError("materialized dataset length disagrees with archive record")
                    generator = torch.Generator().manual_seed(
                        _seed_from_parts(
                            "replite-sanpo-samples-v1",
                            self.seed,
                            self._epoch,
                            *record.key,
                        )
                    )
                    kwargs: dict[str, object] = {}
                    if self.num_workers > 0:
                        kwargs["prefetch_factor"] = self.prefetch_factor
                    loader = DataLoader(
                        dataset,
                        batch_size=self.batch_size,
                        shuffle=self.shuffle,
                        num_workers=self.num_workers,
                        pin_memory=self.pin_memory,
                        drop_last=self.drop_last,
                        persistent_workers=False,
                        collate_fn=sanpo_joint_collate,
                        generator=generator,
                        **kwargs,
                    )
                    try:
                        yield from loader
                    finally:
                        del loader
                        del dataset
        finally:
            self._iterating = False


__all__ = [
    "ArchiveCatalog",
    "ArchiveGroupSplit",
    "ArchiveMaterializer",
    "ArchiveShardLoader",
    "SanpoArchiveRecord",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "create_or_load_group_split",
    "load_archive_catalog",
]
