"""Verified, out-of-core access to downloader-produced SANPO archives.

The SANPO download notebook stores one ``.tar.zst`` archive per
``(official split, session, camera)`` on Google Drive.  This module joins the
selection manifest to the append-only archive ledger, freezes a leak-free
train/validation split by *session*, and either feeds one verified archive at a
time or persistently stages a complete official split on local SSD for
:class:`~replite.data.sanpo_joint.SanpoJointDataset`.

Absolute archive paths recorded by Colab are deliberately treated as
provenance only.  A catalog always resolves an archive below the caller's
current ``drive_root/archives/<split>`` using only the validated basename.
"""

from __future__ import annotations

import bisect
import contextlib
import hashlib
import json
import math
import os
import re
import shutil
import tarfile
import tempfile
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, BinaryIO, Callable, Literal

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from .sanpo_joint import (
    SanpoJointDataset,
    load_sanpo_joint_manifest,
    sanpo_joint_collate,
)


_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_ARCHIVE_NAME_RE = re.compile(r"[A-Za-z0-9._-]+\.tar\.zst")
_SPLITS = frozenset({"train", "test"})
_SENSORS = frozenset({"camera_head", "camera_chest"})
_STAGE_MARKER = ".replite_owned_sanpo_stage.json"
_PERSISTENT_STAGE_MARKER = ".replite_sanpo_persistent_stage.json"
_PERSISTENT_SHARD_COMPLETE = ".replite_sanpo_shard_complete.json"

DetectionSource = Literal["packaged_json", "panoptic_on_load"]
StagePurpose = Literal["official_train", "official_test"]
_StageProgressCallback = Callable[[Mapping[str, object]], None]
LedgerCandidate = tuple[str, Mapping[str, object]]


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


SANPO_DERIVED_DETECTION_CONFIG: Mapping[str, object] = MappingProxyType(
    {
        "schema_version": 1,
        "bbox_format": "XYXY half-open, absolute target-RGB pixels",
        "component_connectivity": 8,
        "min_component_pixels": 100,
        "small_component_policy": "ignore_boxes",
        "instance_zero_policy": "keep_as_valid_panoptic_instance",
        "crowd_policy": "none; SANPO does not publish iscrowd",
    }
)
SANPO_DERIVED_DETECTION_CONFIG_SHA256 = canonical_json_sha256(
    dict(SANPO_DERIVED_DETECTION_CONFIG)
)


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
        "detection_source": record.detection_source,
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


def _detection_config_sha256_hint(path: Path) -> str:
    """Validate metadata against the locked derived-box policy.

    Early downloader metadata contained only the class taxonomy.  Such a
    wrapper still uses the versioned RepLite runtime policy below; missing
    fields must never cause a policy to be guessed from whatever packages
    happen to remain in an append-only ledger.
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
        return SANPO_DERIVED_DETECTION_CONFIG_SHA256
    if config is None:
        declared = _sha256(declared_raw, "derived-detection detection_config_sha256")
        if declared != SANPO_DERIVED_DETECTION_CONFIG_SHA256:
            raise ValueError(
                "derived-detection metadata is not the locked RepLite policy"
            )
        return declared
    if not isinstance(config, Mapping):
        raise ValueError("derived-detection detection_config must be a mapping")
    declared = _sha256(declared_raw, "derived-detection detection_config_sha256")
    actual = canonical_json_sha256(dict(config))
    if actual != declared:
        raise ValueError("derived-detection config SHA-256 does not match its content")
    if declared != SANPO_DERIVED_DETECTION_CONFIG_SHA256:
        raise ValueError("derived-detection metadata is not the locked RepLite policy")
    return declared


@dataclass(frozen=True)
class SanpoArchiveRecord:
    """Validated identity and local Drive path for one session-camera shard."""

    split: str
    session_id: str
    sensor: str
    annotation_policy: str
    selection_sha256: str
    detection_source: DetectionSource
    detection_config_sha256: str | None
    package_sha256: str | None
    archive_path: Path
    archive_bytes: int
    archive_sha256: str
    joint_frames: int
    source_bytes: int | None = None

    @property
    def selection_key(self) -> tuple[str, str, str, str]:
        """Return the source-selection identity, excluding derived labels."""

        return self.split, self.session_id, self.sensor, self.selection_sha256

    @property
    def key(self) -> tuple[str, str, str, str, str]:
        """Return the exact selected archive identity."""

        return (*self.selection_key, self.archive_sha256)


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


@dataclass(frozen=True)
class _LedgerIdentity:
    split: str
    session_id: str
    sensor: str
    annotation_policy: str
    selection_sha256: str
    detection_source: DetectionSource
    detection_config_sha256: str | None
    package_sha256: str | None

    @property
    def selection_key(self) -> tuple[str, str, str, str]:
        return self.split, self.session_id, self.sensor, self.selection_sha256


def _ledger_identity(
    raw: object,
    *,
    ledger_key: object,
    expected_annotation_policy: str,
) -> _LedgerIdentity:
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
    selection_sha256 = _sha256(raw.get("selection_sha256"), "archive selection_sha256")
    has_detection_config = "detection_config_sha256" in raw
    has_package = "package_sha256" in raw
    if has_detection_config != has_package:
        raise ValueError(
            "archive detection_config_sha256 and package_sha256 must appear together"
        )
    if has_detection_config:
        detection_source: DetectionSource = "packaged_json"
        detection_config_sha256 = _sha256(
            raw.get("detection_config_sha256"),
            "archive detection_config_sha256",
        )
        package_sha256 = _sha256(raw.get("package_sha256"), "archive package_sha256")
        expected_package_sha256 = _package_sha256(
            selection_sha256, detection_config_sha256
        )
        if package_sha256 != expected_package_sha256:
            raise ValueError("archive package_sha256 does not match its provenance")
        ledger_digest = package_sha256
    else:
        # Legacy archives predate packaged detection JSON.  Keep the missing
        # digests as ``None`` instead of attributing the active runtime config
        # to bytes that never declared it.
        detection_source = "panoptic_on_load"
        detection_config_sha256 = None
        package_sha256 = None
        ledger_digest = selection_sha256

    expected_ledger_key = f"{split}/{session_id}/{sensor}/{ledger_digest}"
    if ledger_key != expected_ledger_key:
        raise ValueError(
            "archive ledger key does not match entry provenance: "
            f"expected {expected_ledger_key!r}"
        )
    return _LedgerIdentity(
        split=split,
        session_id=session_id,
        sensor=sensor,
        annotation_policy=annotation_policy,
        selection_sha256=selection_sha256,
        detection_source=detection_source,
        detection_config_sha256=detection_config_sha256,
        package_sha256=package_sha256,
    )


def _ledger_record(
    raw: object,
    *,
    ledger_key: object,
    drive_root: Path,
    expected_annotation_policy: str,
    validate_archive_file: bool,
) -> SanpoArchiveRecord:
    identity = _ledger_identity(
        raw,
        ledger_key=ledger_key,
        expected_annotation_policy=expected_annotation_policy,
    )
    assert isinstance(raw, Mapping)  # established by _ledger_identity
    archive_value = raw.get("archive")
    if not isinstance(archive_value, str) or not archive_value:
        raise ValueError("archive ledger path must be a non-empty string")
    archive_name = Path(archive_value).name
    if _ARCHIVE_NAME_RE.fullmatch(archive_name) is None:
        raise ValueError("archive ledger path has an invalid basename")

    # Never trust the absolute path persisted by a previous Colab mount.
    # `drive_root` is already absolute and `archive_name` is a validated
    # basename. Avoid Path.resolve() here: on Colab's Drive FUSE it performs
    # remote metadata lookups once per archive even when physical validation
    # is intentionally deferred to local-SSD staging.
    archive_path = drive_root / "archives" / identity.split / archive_name

    record = SanpoArchiveRecord(
        split=identity.split,
        session_id=identity.session_id,
        sensor=identity.sensor,
        annotation_policy=identity.annotation_policy,
        selection_sha256=identity.selection_sha256,
        detection_source=identity.detection_source,
        detection_config_sha256=identity.detection_config_sha256,
        package_sha256=identity.package_sha256,
        archive_path=archive_path,
        archive_bytes=_positive_int(raw.get("archive_bytes"), "archive_bytes"),
        archive_sha256=_sha256(raw.get("archive_sha256"), "archive_sha256"),
        joint_frames=_positive_int(raw.get("joint_frames"), "archive joint_frames"),
        source_bytes=(
            _positive_int(raw.get("source_bytes"), "source_bytes")
            if raw.get("source_bytes") is not None
            else None
        ),
    )
    if validate_archive_file:
        try:
            observed_bytes = record.archive_path.stat().st_size
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"SANPO archive is missing: {record.archive_path}"
            ) from exc
        if observed_bytes != record.archive_bytes:
            raise ValueError(
                f"SANPO archive byte count disagrees with ledger: {record.archive_path}"
            )
    return record


def _validate_sidecars(record: SanpoArchiveRecord) -> None:
    manifest_path = record.archive_path.with_name(
        record.archive_path.name + ".manifest.json"
    )
    if manifest_path.exists():
        payload = _read_json_object(manifest_path, "SANPO archive sidecar")
        if payload.get("schema_version") != 1 or not isinstance(
            payload.get("entry"), Mapping
        ):
            raise ValueError(f"unsupported SANPO archive sidecar: {manifest_path}")
        entry = payload["entry"]
        comparisons = {
            "split": record.split,
            "session_id": record.session_id,
            "sensor": record.sensor,
            "annotation_policy": record.annotation_policy,
            "selection_sha256": record.selection_sha256,
            "archive_bytes": record.archive_bytes,
            "archive_sha256": record.archive_sha256,
            "joint_frames": record.joint_frames,
        }
        if record.source_bytes is not None:
            comparisons["source_bytes"] = record.source_bytes
        if record.detection_source == "packaged_json":
            comparisons.update(
                detection_config_sha256=record.detection_config_sha256,
                package_sha256=record.package_sha256,
            )
        else:
            for name in ("detection_config_sha256", "package_sha256"):
                if name in entry:
                    raise ValueError(
                        f"legacy archive sidecar unexpectedly declares {name!r}"
                    )
        for name, expected in comparisons.items():
            if entry.get(name) != expected:
                raise ValueError(
                    f"archive sidecar field {name!r} disagrees with ledger"
                )

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
    validate_archive_files: bool = True,
    validate_sidecars: bool = True,
) -> ArchiveCatalog:
    """Resolve the active packaged archive for every selected source shard.

    ``selection_sha256`` intentionally excludes the derived-box policy, so an
    append-only ledger may contain several legitimate packages for the same
    source frames.  For each selected source key, a modern package matching the
    active detection config is preferred; otherwise its exact-keyed legacy
    archive is used and detection is derived from panoptic masks on load.

    ``validate_archive_files=False, validate_sidecars=False`` is the metadata-
    only mode for slow mounted filesystems. It still validates the immutable
    selection/ledger identities and byte/SHA declarations. The persistent SSD
    stage later opens every selected archive, verifies its exact byte count and
    SHA-256 while copying, and validates the extracted manifest/payloads.
    """

    if not isinstance(validate_archive_files, bool):
        raise TypeError("validate_archive_files must be boolean")
    if not isinstance(validate_sidecars, bool):
        raise TypeError("validate_sidecars must be boolean")

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
        else SANPO_DERIVED_DETECTION_CONFIG_SHA256
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
        _selection_record(raw, index) for index, raw in enumerate(selection["records"])
    ]
    selected_keys = [
        (item["split"], item["session_id"], item["sensor"], item["selection_sha256"])
        for item in selected
    ]
    if len(selected_keys) != len(set(selected_keys)):
        raise ValueError("SANPO download selection contains duplicate join keys")
    if "session_camera_count" in selection and selection["session_camera_count"] != len(
        selected
    ):
        raise ValueError("selection session_camera_count does not match records")
    if "joint_target_count" in selection and selection["joint_target_count"] != sum(
        int(item["joint_frames"]) for item in selected
    ):
        raise ValueError("selection joint_target_count does not match records")

    selected_key_set = set(selected_keys)
    package_families: dict[
        str,
        dict[tuple[str, str, str, str], list[LedgerCandidate]],
    ] = {}
    legacy_by_key: dict[tuple[str, str, str, str], list[LedgerCandidate]] = {}
    for ledger_key, raw in ledger["archives"].items():
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
        identity = _ledger_identity(
            raw,
            ledger_key=ledger_key,
            expected_annotation_policy=annotation_policy,
        )
        candidate = (ledger_key, raw)
        if identity.detection_source == "packaged_json":
            assert identity.detection_config_sha256 is not None
            package_families.setdefault(
                identity.detection_config_sha256, {}
            ).setdefault(identity.selection_key, []).append(candidate)
        else:
            legacy_by_key.setdefault(identity.selection_key, []).append(candidate)

    detection_config_sha256 = detection_config_hint

    ledger_by_key: dict[tuple[str, str, str, str], list[SanpoArchiveRecord]] = {}
    active_modern = package_families.get(detection_config_sha256, {})
    for raw_key in selected_keys:
        candidates = active_modern.get(raw_key) or legacy_by_key.get(raw_key, [])
        for ledger_key, raw in candidates:
            record = _ledger_record(
                raw,
                ledger_key=ledger_key,
                drive_root=root,
                expected_annotation_policy=annotation_policy,
                validate_archive_file=validate_archive_files,
            )
            ledger_by_key.setdefault(raw_key, []).append(record)

    joined: list[SanpoArchiveRecord] = []
    for item, key in zip(selected, selected_keys):
        matches = ledger_by_key.get(key, [])
        distinct = {
            canonical_json_sha256(_record_id(record)): record for record in matches
        }
        if len(distinct) != 1:
            raise ValueError(
                "each selected SANPO shard needs exactly one active archive; "
                f"found {len(distinct)} for {key[:3]} with detection config "
                f"{detection_config_sha256[:12]}"
            )
        record = next(iter(distinct.values()))
        if record.joint_frames != item["joint_frames"]:
            raise ValueError(
                f"selection and ledger joint_frames disagree for {key[:3]}"
            )
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
    """A frozen official-train split plus a reserved official-test holdout."""

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
    session_limit: int | None = None,
    validation_session_count: int | None = None,
    official_test_session_limit: int | None = None,
) -> tuple[dict[str, object], set[str], set[str], set[str]]:
    all_train_sessions = sorted({record.session_id for record in catalog.train_records})
    if len(all_train_sessions) < 2:
        raise ValueError("at least two official-train sessions are required")
    if session_limit is not None:
        if (
            isinstance(session_limit, bool)
            or not isinstance(session_limit, int)
            or session_limit < 2
            or session_limit > len(all_train_sessions)
        ):
            raise ValueError(
                "session_limit must be between two and the available "
                "official-train session count"
            )
    deterministic_train_order = sorted(
        all_train_sessions,
        key=lambda session_id: (
            hashlib.sha256(
                f"replite-sanpo-split-v1:{seed}:{session_id}".encode()
            ).hexdigest(),
            session_id,
        ),
    )
    if session_limit is None:
        ordered = deterministic_train_order
        selected_train_sessions = sorted(ordered)
        session_selection_ordering = None
    elif validation_session_count is not None:
        # A clean subset campaign must not depend on which prefix happened to
        # be staged in an earlier ephemeral Colab runtime.  Freeze the sample
        # from the entire official-train pool using the same seed-bound hash
        # ordering that assigns fit versus inner-validation sessions.
        ordered = deterministic_train_order[:session_limit]
        selected_train_sessions = sorted(ordered)
        session_selection_ordering = "sha256(replite-sanpo-split-v1:seed:session_id)"
    else:
        # Preserve the legacy prefix-selection contract for old immutable
        # manifests that used session_limit together with a fraction.
        selected_train_sessions = all_train_sessions[:session_limit]
        selected_set = set(selected_train_sessions)
        ordered = [
            session_id
            for session_id in deterministic_train_order
            if session_id in selected_set
        ]
        session_selection_ordering = "lexicographic session_id"
    if validation_session_count is None:
        validation_count = max(
            1,
            min(len(ordered) - 1, round(len(ordered) * validation_fraction)),
        )
    else:
        if (
            isinstance(validation_session_count, bool)
            or not isinstance(validation_session_count, int)
            or validation_session_count < 1
            or validation_session_count >= len(ordered)
        ):
            raise ValueError(
                "validation_session_count must be between one and one less "
                "than the selected official-train session count"
            )
        validation_count = validation_session_count
    validation_sessions = set(ordered[:validation_count])
    fit_sessions = set(ordered[validation_count:])
    all_test_sessions = sorted({record.session_id for record in catalog.test_records})
    if (fit_sessions | validation_sessions) & set(all_test_sessions):
        raise ValueError("official test session_id overlaps official train")
    if official_test_session_limit is not None:
        if (
            isinstance(official_test_session_limit, bool)
            or not isinstance(official_test_session_limit, int)
            or official_test_session_limit < 1
            or official_test_session_limit > len(all_test_sessions)
        ):
            raise ValueError(
                "official_test_session_limit must be between one and the "
                "available official-test session count"
            )
        deterministic_test_order = sorted(
            all_test_sessions,
            key=lambda session_id: (
                hashlib.sha256(
                    f"replite-sanpo-test-holdout-v1:{seed}:{session_id}".encode()
                ).hexdigest(),
                session_id,
            ),
        )
        selected_test_sessions = set(
            deterministic_test_order[:official_test_session_limit]
        )
    else:
        selected_test_sessions = set(all_test_sessions)
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
        "official_test_session_ids": sorted(selected_test_sessions),
        "protocol_note": "Official test is excluded from training, validation, and selection.",
    }
    if session_limit is not None:
        payload.update(
            {
                "subset_schema_version": 1,
                "session_limit": session_limit,
                "session_selection_ordering": session_selection_ordering,
                "selected_official_train_session_ids": selected_train_sessions,
            }
        )
    if validation_session_count is not None:
        payload.update(
            {
                "fit_session_count": len(fit_sessions),
                "validation_session_count": validation_count,
            }
        )
    if official_test_session_limit is not None:
        payload.update(
            {
                "official_test_session_limit": official_test_session_limit,
                "official_test_pool_session_count": len(all_test_sessions),
                "selected_official_test_session_ids": sorted(selected_test_sessions),
                "official_test_selection_ordering": (
                    "sha256(replite-sanpo-test-holdout-v1:seed:session_id)"
                ),
            }
        )
    payload["manifest_sha256"] = canonical_json_sha256(payload)
    return payload, fit_sessions, validation_sessions, selected_test_sessions


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
    session_limit: int | None = None,
    validation_session_count: int | None = None,
    official_test_session_limit: int | None = None,
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
    (
        payload,
        train_sessions,
        validation_sessions,
        official_test_sessions,
    ) = _group_split_payload(
        catalog,
        seed=seed,
        validation_fraction=fraction,
        session_limit=session_limit,
        validation_session_count=validation_session_count,
        official_test_session_limit=official_test_session_limit,
    )
    path = Path(manifest_path).expanduser().resolve()
    _write_json_immutable(path, payload)

    train_records = tuple(
        record
        for record in catalog.train_records
        if record.session_id in train_sessions
    )
    validation_records = tuple(
        record
        for record in catalog.train_records
        if record.session_id in validation_sessions
    )
    official_test_records = tuple(
        record
        for record in catalog.test_records
        if record.session_id in official_test_sessions
    )
    if not train_records or not validation_records:
        raise ValueError(
            "session grouping produced an empty training or validation set"
        )
    if {record.session_id for record in train_records} & {
        record.session_id for record in validation_records
    }:
        raise AssertionError("session-level train/validation leakage")
    return ArchiveGroupSplit(
        train_records=train_records,
        validation_records=validation_records,
        official_test_records=official_test_records,
        seed=seed,
        validation_fraction=fraction,
        catalog_sha256=catalog.catalog_sha256,
        manifest_path=path,
        manifest_sha256=str(payload["manifest_sha256"]),
    )


def _copy_verified(
    record: SanpoArchiveRecord,
    destination: Path,
    progress: Callable[[int, int], None] | None = None,
) -> None:
    digest = hashlib.sha256()
    size = 0
    with record.archive_path.open("rb") as source, destination.open("xb") as target:
        while block := source.read(16 * 1024 * 1024):
            target.write(block)
            digest.update(block)
            size += len(block)
            if progress is not None:
                progress(size, record.archive_bytes)
        target.flush()
        os.fsync(target.fileno())
    if size != record.archive_bytes or digest.hexdigest() != record.archive_sha256:
        raise ValueError(
            f"SANPO archive failed byte/SHA verification: {record.archive_path}"
        )


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


def _extract_tar_zst_safely(
    archive: Path,
    destination: Path,
    progress: Callable[[int, int], None] | None = None,
) -> int:
    seen: set[tuple[str, ...]] = set()
    member_count = 0
    extracted_bytes = 0
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
                        raise ValueError(
                            f"cannot read regular tar member: {member.name!r}"
                        )
                    with source, target.open("xb") as output:
                        while block := source.read(16 * 1024 * 1024):
                            output.write(block)
                            extracted_bytes += len(block)
                            if progress is not None:
                                progress(extracted_bytes, member_count + 1)
                else:
                    raise ValueError(
                        f"tar links, devices, and special members are forbidden: {member.name!r}"
                    )
                member_count += 1
    if member_count == 0:
        raise ValueError("SANPO archive is empty")
    return member_count


def _validate_manifest_identity(
    record: SanpoArchiveRecord,
    manifest_path: Path,
) -> Path:
    """Validate one known manifest against its immutable archive record."""

    manifest, info = load_sanpo_joint_manifest(manifest_path)
    manifest_policy = manifest.get("annotation_policy")
    policy_matches = manifest_policy == record.annotation_policy
    if record.detection_source == "panoptic_on_load" and manifest_policy is None:
        # Legacy manifests predate this redundant descriptive field.  The
        # selected sample set remains bound by selection_sha256.
        policy_matches = True
    if (
        info.official_split != record.split
        or info.session_id != record.session_id
        or info.sensor != record.sensor
        or info.sample_count != record.joint_frames
        or not policy_matches
        or manifest.get("selection_sha256") != record.selection_sha256
    ):
        raise ValueError("extracted SANPO manifest disagrees with archive catalog")
    if record.detection_source == "packaged_json":
        detection = manifest.get("detection")
        if (
            not isinstance(detection, Mapping)
            or detection.get("config_sha256") != record.detection_config_sha256
        ):
            raise ValueError(
                "extracted SANPO detection provenance disagrees with archive catalog"
            )
    return manifest_path


def _referenced_payload_signature(
    record: SanpoArchiveRecord,
    manifest_path: Path,
) -> dict[str, int | str]:
    """Validate referenced files and bind their relative paths and byte sizes."""

    payload, info = load_sanpo_joint_manifest(manifest_path)
    required: set[str] = set()
    for sample in payload["samples"]:
        assert isinstance(sample, Mapping)
        context = sample["rgb_context_paths"]
        assert isinstance(context, list)
        required.update(str(item) for item in context)
        required.add(str(sample["panoptic_path"]))
        required.add(str(sample["depth_path"]))
        if record.detection_source == "packaged_json":
            detection = sample.get("detection_path")
            if not isinstance(detection, str) or not detection:
                raise ValueError("packaged SANPO manifest sample has no detection_path")
            required.add(detection)

    sizes: list[tuple[str, int]] = []
    for relative in sorted(required):
        path = info.session_root.joinpath(*PurePosixPath(relative).parts)
        if not path.is_file():
            raise ValueError(
                f"extracted SANPO archive is missing referenced file: {relative}"
            )
        size = path.stat().st_size
        if size <= 0:
            raise ValueError(
                f"extracted SANPO archive has empty referenced file: {relative}"
            )
        sizes.append((relative, size))
    return {
        "referenced_file_count": len(sizes),
        "referenced_file_bytes": sum(size for _, size in sizes),
        "referenced_file_sizes_sha256": canonical_json_sha256(sizes),
    }


def _validate_materialized_manifest(
    record: SanpoArchiveRecord,
    extracted: Path,
) -> Path:
    """Find the sole manifest and verify every selected payload file exists."""

    manifests = list(extracted.rglob("_sanpo_joint_manifest.json"))
    if len(manifests) != 1:
        raise ValueError(
            "each SANPO archive must contain exactly one _sanpo_joint_manifest.json"
        )
    manifest_path = _validate_manifest_identity(record, manifests[0])
    _referenced_payload_signature(record, manifest_path)
    return manifest_path


def _atomic_json_file(path: Path, payload: Mapping[str, object]) -> None:
    encoded = (
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _tree_file_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


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
            raise RuntimeError(
                f"refusing to clean an unowned SANPO stage: {stage}"
            ) from exc
        if marker_payload.get("archive_sha256") != self.record.archive_sha256:
            raise RuntimeError(
                f"refusing to clean a SANPO stage with a foreign marker: {stage}"
            )
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
            return _validate_materialized_manifest(self.record, extracted)
        except BaseException:
            self._cleanup()
            raise

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self._cleanup()


class LocalArchiveStage:
    """Persistent, verified extraction of one leak-safe SANPO split on local SSD.

    Each archive is copied from Drive exactly once, hashing while copying, then
    extracted into an owned temporary directory.  A completed shard is
    atomically renamed into place and reused for every later epoch.  An
    interrupted stage therefore resumes at the first incomplete archive.

    ``official_train`` accepts only official-train records (both fit and
    inner-validation groups).  ``official_test`` accepts only official-test
    records, which lets the campaign runner keep the holdout entirely absent
    from local disk until an explicit final-evaluation action.
    """

    def __init__(
        self,
        records: Sequence[SanpoArchiveRecord],
        *,
        local_root: str | os.PathLike[str],
        purpose: StagePurpose,
        expansion_factor: float = 1.05,
        reserve_bytes: int = 4 * 1024**3,
    ) -> None:
        self.records = tuple(records)
        if not self.records or any(
            not isinstance(record, SanpoArchiveRecord) for record in self.records
        ):
            raise ValueError("a local SANPO stage requires archive records")
        if len({record.key for record in self.records}) != len(self.records):
            raise ValueError("local SANPO stage records contain duplicate keys")
        if purpose not in {"official_train", "official_test"}:
            raise ValueError("invalid local SANPO stage purpose")
        expected_split = "train" if purpose == "official_train" else "test"
        if any(record.split != expected_split for record in self.records):
            raise ValueError(
                f"{purpose} stage may contain only official-{expected_split} archives"
            )
        if (
            isinstance(expansion_factor, bool)
            or not isinstance(expansion_factor, (int, float))
            or not math.isfinite(float(expansion_factor))
            or float(expansion_factor) < 1.0
        ):
            raise ValueError("expansion_factor must be finite and at least one")
        if (
            isinstance(reserve_bytes, bool)
            or not isinstance(reserve_bytes, int)
            or reserve_bytes < 0
        ):
            raise ValueError("reserve_bytes must be a non-negative integer")
        self.local_root = Path(local_root).expanduser().resolve()
        self.purpose: StagePurpose = purpose
        self.expansion_factor = float(expansion_factor)
        self.reserve_bytes = reserve_bytes
        ordered = sorted(
            (_record_id(record) for record in self.records),
            key=lambda x: (
                str(x["split"]),
                str(x["session_id"]),
                str(x["sensor"]),
                str(x["archive_sha256"]),
            ),
        )
        self.stage_sha256 = canonical_json_sha256(
            {
                "schema_version": 1,
                "purpose": purpose,
                "records": ordered,
            }
        )
        self._records_by_key = {record.key: record for record in self.records}

    def _select_records(
        self,
        records: Sequence[SanpoArchiveRecord] | None,
    ) -> tuple[SanpoArchiveRecord, ...]:
        selected = self.records if records is None else tuple(records)
        if not selected:
            raise ValueError("local SANPO stage selection cannot be empty")
        if any(not isinstance(record, SanpoArchiveRecord) for record in selected):
            raise TypeError("local SANPO stage selection must contain archive records")
        if len({record.key for record in selected}) != len(selected):
            raise ValueError("local SANPO stage selection contains duplicate keys")
        if any(self._records_by_key.get(record.key) != record for record in selected):
            raise ValueError("record selection is not a subset of this SANPO stage")
        return selected

    @property
    def marker_path(self) -> Path:
        return self.local_root / _PERSISTENT_STAGE_MARKER

    @property
    def status_path(self) -> Path:
        return self.local_root / "stage_manifest.json"

    @property
    def _lock_path(self) -> Path:
        # Keep the lock outside the deletable stage tree so cleanup and a
        # concurrent direct CLI invocation can never lock different inodes.
        return self.local_root.parent / (
            f".{self.local_root.name}.{self.stage_sha256[:16]}.lock"
        )

    @property
    def _shards_root(self) -> Path:
        return self.local_root / "shards"

    @property
    def _partials_root(self) -> Path:
        return self.local_root / "partials"

    def _marker_payload(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "stage_sha256": self.stage_sha256,
            "purpose": self.purpose,
            "record_count": len(self.records),
        }

    def _ensure_owned_root(self) -> None:
        self.local_root.mkdir(parents=True, exist_ok=True)
        if self.marker_path.exists():
            if (
                _read_json_object(self.marker_path, "persistent SANPO stage marker")
                != self._marker_payload()
            ):
                raise RuntimeError(
                    "local SANPO stage belongs to a different record selection"
                )
        else:
            if any(self.local_root.iterdir()):
                raise RuntimeError(
                    "refusing to claim a non-empty unowned SANPO stage directory"
                )
            _atomic_json_file(self.marker_path, self._marker_payload())
        self._shards_root.mkdir(exist_ok=True)
        self._partials_root.mkdir(exist_ok=True)

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        try:
            import fcntl
        except ImportError as exc:  # pragma: no cover - Colab/Linux path
            raise RuntimeError(
                "persistent SANPO SSD staging requires POSIX file locking"
            ) from exc
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock_path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                self._ensure_owned_root()
                if (
                    _read_json_object(self.marker_path, "persistent SANPO stage marker")
                    != self._marker_payload()
                ):
                    raise RuntimeError("local SANPO stage marker changed while waiting")
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _record_stage_id(record: SanpoArchiveRecord) -> str:
        return canonical_json_sha256(_record_id(record))

    def _record_root(self, record: SanpoArchiveRecord) -> Path:
        return self._shards_root / self._record_stage_id(record)

    def _complete_payload(
        self,
        record: SanpoArchiveRecord,
        *,
        manifest_relative_path: str,
        extracted_bytes: int,
        payload_signature: Mapping[str, int | str],
    ) -> dict[str, object]:
        return {
            "schema_version": 1,
            "stage_sha256": self.stage_sha256,
            "record": _record_id(record),
            "manifest_relative_path": manifest_relative_path,
            "extracted_bytes": extracted_bytes,
            **dict(payload_signature),
        }

    def _existing_manifest(
        self,
        record: SanpoArchiveRecord,
        *,
        verify_payloads: bool = False,
    ) -> Path | None:
        root = self._record_root(record)
        if not root.exists():
            return None
        complete_path = root / _PERSISTENT_SHARD_COMPLETE
        complete = _read_json_object(complete_path, "completed SANPO shard marker")
        if (
            complete.get("schema_version") != 1
            or complete.get("stage_sha256") != self.stage_sha256
            or complete.get("record") != _record_id(record)
        ):
            raise RuntimeError(f"persistent SANPO shard has foreign identity: {root}")
        relative_raw = complete.get("manifest_relative_path")
        if not isinstance(relative_raw, str):
            raise RuntimeError("completed SANPO shard has no manifest path")
        relative = PurePosixPath(relative_raw)
        if relative.is_absolute() or any(
            part in {"", ".", ".."} for part in relative.parts
        ):
            raise RuntimeError("completed SANPO shard manifest path is unsafe")
        manifest = root.joinpath(*relative.parts)
        try:
            manifest.resolve().relative_to(root.resolve())
        except ValueError as exc:
            raise RuntimeError(
                "completed SANPO shard manifest escapes its root"
            ) from exc
        if not manifest.is_file():
            raise RuntimeError("completed SANPO shard manifest is missing")
        validated = _validate_manifest_identity(record, manifest)
        if validated.resolve() != manifest.resolve():
            raise RuntimeError("completed SANPO shard points at the wrong manifest")
        signature_fields = {
            "referenced_file_count",
            "referenced_file_bytes",
            "referenced_file_sizes_sha256",
        }
        if not signature_fields.issubset(complete):
            raise RuntimeError("completed SANPO shard has no payload signature")
        if verify_payloads:
            observed = _referenced_payload_signature(record, manifest)
            expected = {name: complete[name] for name in signature_fields}
            if observed != expected:
                raise RuntimeError(
                    f"completed SANPO shard payload signature changed: {root}"
                )
        return manifest

    def _write_status(self) -> dict[str, object]:
        expected = {self._record_stage_id(record): record for record in self.records}
        observed = {path.name for path in self._shards_root.iterdir() if path.is_dir()}
        foreign = observed - set(expected)
        if foreign:
            raise RuntimeError(
                "persistent SANPO stage contains foreign shard directories"
            )
        completed_ids = sorted(observed)
        extracted_bytes = 0
        for record_id in completed_ids:
            complete = _read_json_object(
                self._shards_root / record_id / _PERSISTENT_SHARD_COMPLETE,
                "completed SANPO shard marker",
            )
            value = complete.get("extracted_bytes")
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RuntimeError(
                    "completed SANPO shard has invalid extracted byte count"
                )
            extracted_bytes += value
        payload: dict[str, object] = {
            "schema_version": 1,
            "stage_sha256": self.stage_sha256,
            "purpose": self.purpose,
            "record_count": len(self.records),
            "completed_count": len(completed_ids),
            "complete": len(completed_ids) == len(self.records),
            "completed_record_ids": completed_ids,
            "completed_extracted_bytes": extracted_bytes,
        }
        _atomic_json_file(self.status_path, payload)
        return payload

    def _selection_status(
        self,
        records: Sequence[SanpoArchiveRecord],
        global_status: Mapping[str, object],
    ) -> dict[str, object]:
        selected = self._select_records(records)
        if selected == self.records:
            return dict(global_status)
        completed_ids: list[str] = []
        extracted_bytes = 0
        for record in selected:
            record_id = self._record_stage_id(record)
            root = self._shards_root / record_id
            if not root.is_dir():
                continue
            complete = _read_json_object(
                root / _PERSISTENT_SHARD_COMPLETE,
                "completed SANPO shard marker",
            )
            value = complete.get("extracted_bytes")
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RuntimeError(
                    "completed SANPO shard has invalid extracted byte count"
                )
            completed_ids.append(record_id)
            extracted_bytes += value
        completed_ids.sort()
        return {
            "schema_version": 1,
            "stage_sha256": self.stage_sha256,
            "purpose": self.purpose,
            "selection_sha256": canonical_json_sha256(
                [_record_id(record) for record in selected]
            ),
            "record_count": len(selected),
            "completed_count": len(completed_ids),
            "complete": len(completed_ids) == len(selected),
            "completed_record_ids": completed_ids,
            "completed_extracted_bytes": extracted_bytes,
            "stage_record_count": global_status["record_count"],
            "stage_completed_count": global_status["completed_count"],
        }

    def _remove_owned_partials(self) -> None:
        for partial in self._partials_root.iterdir():
            if not partial.is_dir():
                raise RuntimeError(f"unexpected file in SANPO partial area: {partial}")
            marker = _read_json_object(
                partial / _STAGE_MARKER, "partial SANPO stage marker"
            )
            if marker.get("stage_sha256") != self.stage_sha256:
                raise RuntimeError(
                    f"refusing to remove foreign SANPO partial: {partial}"
                )
            shutil.rmtree(partial)

    def _estimated_unpacked_bytes(self, record: SanpoArchiveRecord) -> int:
        # Downloader ledgers bind the exact byte sum of selected RGB,
        # panoptic, and depth objects.  The archive-ratio allowance covers the
        # small generated manifests/detection JSON and filesystem overhead.
        ratio_estimate = math.ceil(record.archive_bytes * self.expansion_factor)
        return max(ratio_estimate, record.source_bytes or 0)

    def _disk_plan_unlocked(
        self,
        records: Sequence[SanpoArchiveRecord] | None = None,
    ) -> dict[str, int | float]:
        selected = self._select_records(records)
        pending = [
            record for record in selected if self._existing_manifest(record) is None
        ]
        archive_bytes = sum(record.archive_bytes for record in pending)
        exact_source_bytes = sum(record.source_bytes or 0 for record in pending)
        source_bytes_records = sum(
            record.source_bytes is not None for record in pending
        )
        unpacked_estimate = sum(
            self._estimated_unpacked_bytes(record) for record in pending
        )
        largest_temporary_archive = max(
            (record.archive_bytes for record in pending), default=0
        )
        required = unpacked_estimate + largest_temporary_archive + self.reserve_bytes
        usage = shutil.disk_usage(self.local_root)
        return {
            "pending_records": len(pending),
            "archive_bytes": archive_bytes,
            "exact_source_bytes": exact_source_bytes,
            "source_bytes_records": source_bytes_records,
            "estimated_unpacked_bytes": unpacked_estimate,
            "largest_temporary_archive_bytes": largest_temporary_archive,
            "reserve_bytes": self.reserve_bytes,
            "required_free_bytes": required,
            "available_free_bytes": usage.free,
            "expansion_factor": self.expansion_factor,
        }

    def disk_plan(
        self,
        records: Sequence[SanpoArchiveRecord] | None = None,
    ) -> dict[str, int | float]:
        """Return capacity bytes needed to finish all missing shards."""

        with self._locked():
            return self._disk_plan_unlocked(records)

    def _assert_disk_capacity(
        self,
        records: Sequence[SanpoArchiveRecord] | None = None,
    ) -> dict[str, int | float]:
        plan = self._disk_plan_unlocked(records)
        if int(plan["available_free_bytes"]) < int(plan["required_free_bytes"]):
            raise OSError(
                "insufficient local SSD for SANPO stage: "
                f"need {int(plan['required_free_bytes']) / 1024**3:.2f} GiB free, "
                f"have {int(plan['available_free_bytes']) / 1024**3:.2f} GiB; "
                "free /content space or lower the selected dataset size"
            )
        return plan

    def materialize(
        self,
        record: SanpoArchiveRecord,
        *,
        progress: _StageProgressCallback | None = None,
    ) -> Path:
        """Return a persistent manifest, extracting atomically if necessary."""

        selected = self._records_by_key.get(record.key)
        if selected != record:
            raise ValueError("record does not belong to this local SANPO stage")
        with self._locked():
            existing = self._existing_manifest(record)
            if existing is not None:
                return existing
            self._remove_owned_partials()
            usage = shutil.disk_usage(self.local_root)
            per_record_required = (
                self._estimated_unpacked_bytes(record)
                + record.archive_bytes
                + self.reserve_bytes
            )
            if usage.free < per_record_required:
                raise OSError(
                    "insufficient local SSD to materialize next SANPO archive: "
                    f"need {per_record_required / 1024**3:.2f} GiB free, "
                    f"have {usage.free / 1024**3:.2f} GiB"
                )
            partial = self._partials_root / (
                f".{self._record_stage_id(record)}.{uuid.uuid4().hex}.partial"
            )
            partial.mkdir()
            _atomic_json_file(
                partial / _STAGE_MARKER,
                {
                    "schema_version": 1,
                    "stage_sha256": self.stage_sha256,
                    "record": _record_id(record),
                },
            )
            try:
                local_archive = partial / record.archive_path.name
                if progress is None:
                    _copy_verified(record, local_archive)
                else:
                    _copy_verified(
                        record,
                        local_archive,
                        progress=lambda completed, total: progress(
                            {
                                "event": "record_progress",
                                "phase": "copy+sha",
                                "record": record,
                                "bytes_completed": completed,
                                "bytes_total": total,
                            }
                        ),
                    )
                extracted = partial / "extracted"
                extracted.mkdir()
                if progress is None:
                    _extract_tar_zst_safely(local_archive, extracted)
                else:
                    _extract_tar_zst_safely(
                        local_archive,
                        extracted,
                        progress=lambda completed, members: progress(
                            {
                                "event": "record_progress",
                                "phase": "extract",
                                "record": record,
                                "bytes_completed": completed,
                                "bytes_total": self._estimated_unpacked_bytes(record),
                                "members_completed": members,
                            }
                        ),
                    )
                local_archive.unlink()
                if progress is not None:
                    progress(
                        {
                            "event": "record_progress",
                            "phase": "verify",
                            "record": record,
                        }
                    )
                manifest = _validate_materialized_manifest(record, extracted)
                relative = manifest.relative_to(partial).as_posix()
                payload_signature = _referenced_payload_signature(record, manifest)
                complete = self._complete_payload(
                    record,
                    manifest_relative_path=relative,
                    extracted_bytes=_tree_file_bytes(extracted),
                    payload_signature=payload_signature,
                )
                _atomic_json_file(partial / _PERSISTENT_SHARD_COMPLETE, complete)
                if progress is not None:
                    progress(
                        {
                            "event": "record_progress",
                            "phase": "publish",
                            "record": record,
                        }
                    )
                final = self._record_root(record)
                if final.exists():
                    raise RuntimeError(f"SANPO shard appeared concurrently: {final}")
                os.replace(partial, final)
                self._write_status()
                return final.joinpath(*PurePosixPath(relative).parts)
            except BaseException:
                if partial.exists():
                    shutil.rmtree(partial)
                raise

    def prepare_all(
        self,
        progress: Callable[[int, int, SanpoArchiveRecord, str], None] | None = None,
        *,
        on_event: _StageProgressCallback | None = None,
        records: Sequence[SanpoArchiveRecord] | None = None,
    ) -> dict[str, object]:
        """Preflight disk, resume missing shards, and return the stage manifest.

        ``progress`` preserves the original per-record callback contract.
        ``on_event`` is the detailed byte/phase stream used by the live Colab
        dashboard.  Keeping them separate avoids silently breaking callers
        that still expect ``(index, total, record, status)``.
        """

        selected = self._select_records(records)
        with self._locked():
            self._remove_owned_partials()
            plan = self._assert_disk_capacity(selected)
        total = len(selected)
        total_bytes = sum(self._estimated_unpacked_bytes(record) for record in selected)
        ready_records = 0
        ready_bytes = 0
        if on_event is not None:
            on_event(
                {
                    "event": "stage_start",
                    "purpose": self.purpose,
                    "total_records": total,
                    "ready_records": ready_records,
                    "total_bytes": total_bytes,
                    "ready_bytes": ready_bytes,
                }
            )
        existing_by_key: dict[tuple[str, str, str, str, str], Path | None] = {}
        for index, record in enumerate(selected, start=1):
            existing = self._existing_manifest(record, verify_payloads=True)
            existing_by_key[record.key] = existing
            if existing is not None:
                ready_records += 1
                ready_bytes += self._estimated_unpacked_bytes(record)
            if on_event is not None:
                on_event(
                    {
                        "event": "resume_check",
                        "purpose": self.purpose,
                        "index": index,
                        "total_records": total,
                        "record": record,
                        "phase": "resume-check",
                        "ready_records": ready_records,
                        "ready_bytes": ready_bytes,
                        "total_bytes": total_bytes,
                    }
                )
        if on_event is not None:
            on_event(
                {
                    "event": "resume_end",
                    "purpose": self.purpose,
                    "total_records": total,
                    "ready_records": ready_records,
                    "total_bytes": total_bytes,
                    "ready_bytes": ready_bytes,
                }
            )
        for index, record in enumerate(selected, start=1):
            existing = existing_by_key[record.key]
            status = "cached" if existing is not None else "extract"
            if progress is not None:
                progress(
                    index,
                    total,
                    record,
                    status,
                )
            if on_event is not None:
                on_event(
                    {
                        "event": "record_start",
                        "purpose": self.purpose,
                        "index": index,
                        "total_records": total,
                        "record": record,
                        "status": status,
                        "ready_records": ready_records,
                        "ready_bytes": ready_bytes,
                        "total_bytes": total_bytes,
                    }
                )
            if existing is None:
                self.materialize(
                    record,
                    progress=(
                        None
                        if on_event is None
                        else lambda payload, index=index: on_event(
                            {
                                **dict(payload),
                                "purpose": self.purpose,
                                "index": index,
                                "total_records": total,
                                "ready_records": ready_records,
                                "ready_bytes": ready_bytes,
                                "total_bytes": total_bytes,
                            }
                        )
                    ),
                )
                ready_records += 1
                ready_bytes += self._estimated_unpacked_bytes(record)
            if on_event is not None:
                on_event(
                    {
                        "event": "record_end",
                        "purpose": self.purpose,
                        "index": index,
                        "total_records": total,
                        "record": record,
                        "status": status,
                        "ready_records": ready_records,
                        "ready_bytes": ready_bytes,
                        "total_bytes": total_bytes,
                    }
                )
        if on_event is not None:
            on_event(
                {
                    "event": "final_verify_start",
                    "purpose": self.purpose,
                    "total_records": total,
                    "ready_records": ready_records,
                    "total_bytes": total_bytes,
                    "ready_bytes": ready_bytes,
                    "phase": "final-verify",
                }
            )
        with self._locked():
            for index, record in enumerate(selected, start=1):
                self._existing_manifest(record, verify_payloads=True)
                if on_event is not None:
                    on_event(
                        {
                            "event": "final_verify",
                            "purpose": self.purpose,
                            "index": index,
                            "total_records": total,
                            "record": record,
                            "phase": "final-verify",
                            "ready_records": ready_records,
                            "ready_bytes": ready_bytes,
                            "total_bytes": total_bytes,
                        }
                    )
            global_status = self._write_status()
            payload = self._selection_status(selected, global_status)
        if on_event is not None:
            on_event(
                {
                    "event": "stage_end",
                    "purpose": self.purpose,
                    "total_records": total,
                    "ready_records": ready_records,
                    "total_bytes": total_bytes,
                    "ready_bytes": ready_bytes,
                }
            )
        return {**payload, "disk_plan": plan}

    def status(
        self,
        records: Sequence[SanpoArchiveRecord] | None = None,
    ) -> dict[str, object]:
        """Validate completed shards and return the resumable stage status."""

        selected = self._select_records(records)
        with self._locked():
            for record in selected:
                self._existing_manifest(record, verify_payloads=True)
            global_status = self._write_status()
            return self._selection_status(selected, global_status)

    def require_complete(
        self,
        records: Sequence[SanpoArchiveRecord] | None = None,
    ) -> dict[str, object]:
        """Reject training until every selected archive is present on SSD."""

        payload = self.status(records)
        if payload.get("complete") is not True:
            raise RuntimeError(
                "local SANPO stage is incomplete: "
                f"{payload['completed_count']}/{payload['record_count']} shards; "
                "run the stage-train command first"
            )
        return payload

    def cleanup(self) -> None:
        """Delete only this exact owned stage; archives on Drive are untouched."""

        if not self.local_root.exists():
            return
        with self._locked():
            marker = _read_json_object(
                self.marker_path, "persistent SANPO stage marker"
            )
            if marker != self._marker_payload():
                raise RuntimeError("refusing to clean a foreign SANPO stage")
            shutil.rmtree(self.local_root)


def _seed_from_parts(*parts: object) -> int:
    encoded = ":".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") % (2**63)


class _CombinedSanpoDataset(Dataset[tuple[torch.Tensor, dict[str, Any]]]):
    """One map-style index space spanning immutable staged SANPO shards."""

    def __init__(self, datasets: Sequence[Dataset]) -> None:
        self.datasets = tuple(datasets)
        if not self.datasets:
            raise ValueError("combined SANPO dataset requires at least one shard")
        cumulative: list[int] = []
        total = 0
        for dataset in self.datasets:
            length = len(dataset)
            if length <= 0:
                raise ValueError("combined SANPO dataset contains an empty shard")
            total += length
            cumulative.append(total)
        self.cumulative_sizes = tuple(cumulative)

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any]]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("combined SANPO sample index is out of range")
        shard_index = bisect.bisect_right(self.cumulative_sizes, index)
        previous = 0 if shard_index == 0 else self.cumulative_sizes[shard_index - 1]
        return self.datasets[shard_index][index - previous]


class _EpochIndexSampler(Sampler[int]):
    """Stateless epoch-bound permutation for exact epoch-boundary resume."""

    def __init__(self, size: int, *, seed: int, shuffle: bool) -> None:
        if size <= 0:
            raise ValueError("epoch sampler size must be positive")
        self.size = size
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.size

    def __iter__(self) -> Iterator[int]:
        if not self.shuffle:
            yield from range(self.size)
            return
        generator = torch.Generator().manual_seed(
            _seed_from_parts(
                "replite-sanpo-global-samples-v1",
                self.seed,
                self.epoch,
            )
        )
        yield from torch.randperm(self.size, generator=generator).tolist()


class ArchiveShardLoader:
    """Iterable of batches backed by verified local archive extraction.

    A persistent :class:`LocalArchiveStage` uses one global map-style dataset,
    one sampler, and one DataLoader across all selected shards. Consequently,
    batches cross session boundaries, there is only one tail batch per split,
    and worker processes persist safely because preprocessing is deterministic.
    The epoch sampler is stateless and seed-bound, so setting the resumed epoch
    recreates exactly the same global sample order without saving worker state.

    The non-persistent materializer fallback retains per-shard streaming so it
    never expands every archive onto local disk at the same time.
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
        local_stage: LocalArchiveStage | None = None,
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
        if local_stage is not None:
            if not isinstance(local_stage, LocalArchiveStage):
                raise TypeError("local_stage must be a LocalArchiveStage")
            missing = [
                record.key
                for record in self.records
                if local_stage._records_by_key.get(record.key) != record
            ]
            if missing:
                raise ValueError("loader records are absent from its local SANPO stage")
        self.local_stage = local_stage
        if "image_size" in self.dataset_kwargs:
            raise ValueError("pass image_size directly, not through dataset_kwargs")
        if "use_packaged_detection" in self.dataset_kwargs:
            raise ValueError(
                "use_packaged_detection is selected from archive provenance"
            )
        if "prepared_cache_dir" in self.dataset_kwargs:
            raise ValueError(
                "prepared_cache_dir is owned by the persistent local stage"
            )
        detection_min_area = self.dataset_kwargs.get("detection_min_area", 100)
        if detection_min_area != SANPO_DERIVED_DETECTION_CONFIG["min_component_pixels"]:
            raise ValueError(
                "archive loading requires the locked 100-pixel detection policy"
            )
        self._epoch = 0
        self._iterating = False
        self._staged_dataset: _CombinedSanpoDataset | None = None
        self._staged_sampler: _EpochIndexSampler | None = None
        self._staged_loader: DataLoader | None = None

    @property
    def sample_count(self) -> int:
        return sum(record.joint_frames for record in self.records)

    @property
    def epoch(self) -> int:
        return self._epoch

    def __len__(self) -> int:
        if self.local_stage is not None:
            if self.drop_last:
                return self.sample_count // self.batch_size
            return math.ceil(self.sample_count / self.batch_size)
        if self.drop_last:
            return sum(
                record.joint_frames // self.batch_size for record in self.records
            )
        return sum(
            math.ceil(record.joint_frames / self.batch_size) for record in self.records
        )

    def set_epoch(self, epoch: int) -> None:
        """Select the deterministic shard and sample ordering for an epoch."""

        if self._iterating:
            raise RuntimeError(
                "cannot change epoch during ArchiveShardLoader iteration"
            )
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        self._epoch = epoch
        if self._staged_sampler is not None:
            self._staged_sampler.set_epoch(epoch)

    @property
    def persistent_workers(self) -> bool:
        """Whether the global staged loader keeps its workers across epochs."""

        return self.local_stage is not None and self.num_workers > 0

    def _fixed_records(self) -> tuple[SanpoArchiveRecord, ...]:
        return tuple(sorted(self.records, key=lambda record: record.key))

    def _ensure_staged_loader(self) -> DataLoader:
        if self.local_stage is None:
            raise RuntimeError("global staged loader requires LocalArchiveStage")
        if self._staged_loader is not None:
            return self._staged_loader

        datasets: list[Dataset] = []
        prepared_root = self.local_stage.local_root / "prepared_samples"
        for record in self._fixed_records():
            manifest_path = self.local_stage.materialize(record)
            dataset = SanpoJointDataset(
                manifest_path,
                image_size=self.image_size,
                use_packaged_detection=(record.detection_source == "packaged_json"),
                prepared_cache_dir=(
                    prepared_root / self.local_stage._record_stage_id(record)
                ),
                **self.dataset_kwargs,
            )
            if len(dataset) != record.joint_frames:
                raise ValueError(
                    "materialized dataset length disagrees with archive record"
                )
            datasets.append(dataset)

        combined = _CombinedSanpoDataset(datasets)
        if len(combined) != self.sample_count:
            raise RuntimeError("global SANPO dataset length is inconsistent")
        sampler = _EpochIndexSampler(
            len(combined),
            seed=self.seed,
            shuffle=self.shuffle,
        )
        sampler.set_epoch(self._epoch)
        kwargs: dict[str, object] = {}
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = self.prefetch_factor
        worker_generator = torch.Generator().manual_seed(
            _seed_from_parts("replite-sanpo-workers-v1", self.seed)
        )
        loader = DataLoader(
            combined,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            persistent_workers=self.persistent_workers,
            collate_fn=sanpo_joint_collate,
            generator=worker_generator,
            **kwargs,
        )
        self._staged_dataset = combined
        self._staged_sampler = sampler
        self._staged_loader = loader
        return loader

    def prepared_cache_plan(self) -> dict[str, object]:
        """Return cache readiness and local-SSD bytes before a warm pass."""

        if self.local_stage is None:
            raise RuntimeError("prepared sample cache requires LocalArchiveStage")
        self._ensure_staged_loader()
        assert self._staged_dataset is not None
        statuses = [
            dataset.prepared_cache_status()
            for dataset in self._staged_dataset.datasets
            if isinstance(dataset, SanpoJointDataset)
        ]
        if len(statuses) != len(self._staged_dataset.datasets):
            raise RuntimeError("staged SANPO dataset does not support prepared caching")
        pending_bytes = sum(int(status["estimated_pending_bytes"]) for status in statuses)
        max_sample_bytes = max(
            (int(status["estimated_sample_bytes"]) for status in statuses),
            default=0,
        )
        ready_samples = sum(int(status["ready_samples"]) for status in statuses)
        pending_samples = sum(int(status["pending_samples"]) for status in statuses)
        cached_bytes = sum(int(status["cached_bytes"]) for status in statuses)
        usage = shutil.disk_usage(self.local_stage.local_root)
        required_free_bytes = (
            pending_bytes + max_sample_bytes + self.local_stage.reserve_bytes
        )
        return {
            "schema_version": 1,
            "sample_count": self.sample_count,
            "ready_samples": ready_samples,
            "pending_samples": pending_samples,
            "cached_bytes": cached_bytes,
            "estimated_pending_bytes": pending_bytes,
            "largest_atomic_sample_bytes": max_sample_bytes,
            "reserve_bytes": self.local_stage.reserve_bytes,
            "required_free_bytes": required_free_bytes,
            "available_free_bytes": usage.free,
            "complete_by_count": pending_samples == 0,
        }

    def warm_prepared_cache(
        self,
        *,
        on_event: _StageProgressCallback | None = None,
    ) -> dict[str, object]:
        """Build/verify local resized samples once before a measured epoch.

        The pass uses the same global DataLoader and worker pool as training,
        but never touches CUDA. Official-test data remains absent unless a
        caller explicitly constructs and warms an official-test loader.
        """

        plan = self.prepared_cache_plan()
        total_batches = len(self)
        if on_event is not None:
            on_event(
                {
                    "event": "cache_warm_start",
                    "total_batches": total_batches,
                    **plan,
                }
            )
        if bool(plan["complete_by_count"]):
            result = {**plan, "cache_reused": True}
            if on_event is not None:
                on_event(
                    {
                        "event": "cache_warm_end",
                        "total_batches": total_batches,
                        "completed_samples": self.sample_count,
                        **result,
                    }
                )
            return result
        if int(plan["available_free_bytes"]) < int(plan["required_free_bytes"]):
            raise OSError(
                "insufficient local SSD for SANPO prepared cache: "
                f"need {int(plan['required_free_bytes']) / 1024**3:.2f} GiB free, "
                f"have {int(plan['available_free_bytes']) / 1024**3:.2f} GiB; "
                "free /content space or reduce the selected session count"
            )
        completed_samples = 0
        for batch_index, (inputs, _) in enumerate(self, start=1):
            completed_samples += int(inputs.shape[0])
            if on_event is not None:
                on_event(
                    {
                        "event": "cache_warm_progress",
                        "batch_index": batch_index,
                        "total_batches": total_batches,
                        "completed_samples": completed_samples,
                        "sample_count": self.sample_count,
                    }
                )
        result = {**self.prepared_cache_plan(), "cache_reused": False}
        if (
            int(result["ready_samples"]) != self.sample_count
            or int(result["pending_samples"]) != 0
        ):
            raise RuntimeError("SANPO prepared cache warm pass is incomplete")
        if on_event is not None:
            on_event(
                {
                    "event": "cache_warm_end",
                    "total_batches": total_batches,
                    "completed_samples": completed_samples,
                    **result,
                }
            )
        return result

    def _ordered_records(self) -> tuple[SanpoArchiveRecord, ...]:
        if not self.shuffle:
            return tuple(
                sorted(
                    self.records, key=lambda record: (record.session_id, record.sensor)
                )
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
            raise RuntimeError(
                "ArchiveShardLoader does not support concurrent iteration"
            )
        self._iterating = True
        try:
            if self.local_stage is not None:
                yield from self._ensure_staged_loader()
                return
            for record in self._ordered_records():
                manifest_context = ArchiveMaterializer(
                    record, local_root=self.local_root
                )
                with manifest_context as manifest_path:
                    dataset = SanpoJointDataset(
                        manifest_path,
                        image_size=self.image_size,
                        use_packaged_detection=(
                            record.detection_source == "packaged_json"
                        ),
                        **self.dataset_kwargs,
                    )
                    if len(dataset) != record.joint_frames:
                        raise ValueError(
                            "materialized dataset length disagrees with archive record"
                        )
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
    "LocalArchiveStage",
    "SANPO_DERIVED_DETECTION_CONFIG",
    "SANPO_DERIVED_DETECTION_CONFIG_SHA256",
    "SanpoArchiveRecord",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "create_or_load_group_split",
    "load_archive_catalog",
]
