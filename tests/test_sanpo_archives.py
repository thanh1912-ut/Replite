"""Tests for verified, out-of-core SANPO archive loading."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import Dataset

import replite.data.sanpo_archives as archives_module
from replite.data import (
    ArchiveMaterializer,
    ArchiveShardLoader,
    LocalArchiveStage,
    SanpoArchiveRecord,
    canonical_json_sha256,
    create_or_load_group_split,
    load_archive_catalog,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


_ANNOTATION_POLICY = "human_only"
_DETECTION_CONFIG = {
    "schema_version": 1,
    "bbox_format": "XYXY half-open, absolute target-RGB pixels",
    "component_connectivity": 8,
    "min_component_pixels": 100,
    "small_component_policy": "ignore_boxes",
    "instance_zero_policy": "keep_as_valid_panoptic_instance",
    "crowd_policy": "none; SANPO does not publish iscrowd",
}
_DETECTION_CONFIG_SHA = canonical_json_sha256(_DETECTION_CONFIG)


def _package_sha(selection_sha: str, detection_sha: str = _DETECTION_CONFIG_SHA) -> str:
    return canonical_json_sha256(
        {
            "selection_sha256": selection_sha,
            "detection_config_sha256": detection_sha,
        }
    )


def _ledger_key(entry: dict[str, object]) -> str:
    digest = entry.get("package_sha256", entry["selection_sha256"])
    return "/".join(
        str(entry[field]) for field in ("split", "session_id", "sensor")
    ) + f"/{digest}"


def _first_ledger_entry(
    ledger: dict[str, object],
) -> tuple[str, dict[str, object]]:
    archives = ledger["archives"]
    assert isinstance(archives, dict)
    key = next(iter(archives))
    entry = archives[key]
    assert isinstance(entry, dict)
    return key, entry


def _catalog_fixture(tmp_path: Path):
    root = tmp_path / "mounted-drive"
    specifications = (
        ("train", "session-a", "camera_head", 3),
        ("train", "session-a", "camera_chest", 2),
        ("train", "session-b", "camera_head", 4),
        ("train", "session-c", "camera_chest", 5),
        ("test", "session-test", "camera_head", 2),
    )
    selection_records = []
    ledger_entries = {}
    for index, (split, session_id, sensor, frames) in enumerate(specifications):
        selection_sha = _sha(f"selection-{index}")
        package_sha = _package_sha(selection_sha)
        name = f"{session_id}__{sensor}__{package_sha[:12]}.tar.zst"
        archive = root / "archives" / split / name
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_bytes(f"archive-{index}".encode())
        archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
        selection_records.append(
            {
                "split": split,
                "session_id": session_id,
                "sensor": sensor,
                "selection_sha256": selection_sha,
                "joint_frames": frames,
            }
        )
        entry = {
            "split": split,
            "session_id": session_id,
            "sensor": sensor,
            "annotation_policy": _ANNOTATION_POLICY,
            "selection_sha256": selection_sha,
            "detection_config_sha256": _DETECTION_CONFIG_SHA,
            "package_sha256": package_sha,
            # This old absolute mount path must never be trusted.
            "archive": f"/content/drive/old-root/{name}",
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": archive_sha,
            "source_bytes": archive.stat().st_size,
            "joint_frames": frames,
        }
        ledger_entries[_ledger_key(entry)] = entry
        _write_json(
            archive.with_name(archive.name + ".manifest.json"),
            {"schema_version": 1, "entry": entry},
        )
        archive.with_name(archive.name + ".sha256").write_text(
            f"{archive_sha}  {name}\n", encoding="utf-8"
        )
    _write_json(
        root / "metadata" / "current_download_selection.json",
        {
            "schema_version": 1,
            "annotation_policy": _ANNOTATION_POLICY,
            "session_camera_count": len(selection_records),
            "joint_target_count": sum(item[3] for item in specifications),
            "records": selection_records,
        },
    )
    _write_json(
        root / "metadata" / "derived_detection_classes.json",
        {
            "schema_version": 2,
            "detection_config": _DETECTION_CONFIG,
            "detection_config_sha256": _DETECTION_CONFIG_SHA,
        },
    )
    _write_json(
        root / "archive_manifest.json",
        {
            "schema_version": 2,
            "dataset": "SANPO-Real-v0-joint-human_only",
            "archives": ledger_entries,
        },
    )
    return root, specifications


def _legacy_variant(
    root: Path, modern: dict[str, object], *, payload: bytes
) -> tuple[str, dict[str, object]]:
    legacy = dict(modern)
    legacy.pop("detection_config_sha256")
    legacy.pop("package_sha256")
    split = str(legacy["split"])
    name = (
        f"{legacy['session_id']}__{legacy['sensor']}__"
        f"{str(legacy['selection_sha256'])[:12]}.tar.zst"
    )
    archive = root / "archives" / split / name
    archive.write_bytes(payload)
    legacy.update(
        archive=f"/content/drive/old-root/{name}",
        archive_bytes=archive.stat().st_size,
        archive_sha256=hashlib.sha256(payload).hexdigest(),
    )
    _write_json(
        archive.with_name(archive.name + ".manifest.json"),
        {"schema_version": 1, "entry": legacy},
    )
    archive.with_name(archive.name + ".sha256").write_text(
        f"{legacy['archive_sha256']}  {name}\n", encoding="utf-8"
    )
    return _ledger_key(legacy), legacy


def test_catalog_joins_manifests_and_resolves_only_below_current_drive_root(
    tmp_path: Path,
) -> None:
    root, specifications = _catalog_fixture(tmp_path)
    catalog = load_archive_catalog(root)

    assert len(catalog.records) == len(specifications)
    assert len(catalog.train_records) == 4
    assert len(catalog.test_records) == 1
    assert len(catalog.catalog_sha256) == 64
    for record in catalog.records:
        assert record.archive_path.is_relative_to(root / "archives" / record.split)
        assert "/old-root/" not in str(record.archive_path)


def test_catalog_matches_real_mixed_legacy_and_modern_ledger_shape(
    tmp_path: Path,
) -> None:
    root, specifications = _catalog_fixture(tmp_path)
    ledger_path = root / "archive_manifest.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    modern_items = list(ledger["archives"].items())
    for index, (modern_key, modern) in enumerate(modern_items):
        legacy_key, legacy = _legacy_variant(
            root, modern, payload=f"legacy-{index}".encode()
        )
        ledger["archives"][legacy_key] = legacy
        # Keep one modern duplicate, just as the real ledger keeps three
        # repackaged pilots alongside 234 source-keyed legacy archives.
        if index:
            del ledger["archives"][modern_key]
    _write_json(ledger_path, ledger)

    catalog = load_archive_catalog(root)

    assert len(ledger["archives"]) == len(specifications) + 1
    assert len(catalog.records) == len(specifications)
    assert sum(item.detection_source == "packaged_json" for item in catalog.records) == 1
    assert sum(item.detection_source == "panoptic_on_load" for item in catalog.records) == 4
    assert catalog.detection_config_sha256 == _DETECTION_CONFIG_SHA


def test_catalog_supports_complete_legacy_selection_without_repackaging(
    tmp_path: Path,
) -> None:
    root, specifications = _catalog_fixture(tmp_path)
    ledger_path = root / "archive_manifest.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    replacement: dict[str, object] = {}
    for index, modern in enumerate(ledger["archives"].values()):
        legacy_key, legacy = _legacy_variant(
            root, modern, payload=f"legacy-only-{index}".encode()
        )
        replacement[legacy_key] = legacy
    ledger["archives"] = replacement
    _write_json(ledger_path, ledger)

    catalog = load_archive_catalog(root)

    assert len(catalog.records) == len(specifications)
    assert all(item.detection_source == "panoptic_on_load" for item in catalog.records)
    assert catalog.detection_config_sha256 == _DETECTION_CONFIG_SHA


def test_catalog_rejects_ledger_alias_that_disagrees_with_provenance(
    tmp_path: Path,
) -> None:
    root, _ = _catalog_fixture(tmp_path)
    ledger_path = root / "archive_manifest.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    _, first = _first_ledger_entry(ledger)
    ledger["archives"]["alias"] = dict(first)
    _write_json(ledger_path, ledger)

    with pytest.raises(ValueError, match="ledger key does not match"):
        load_archive_catalog(root)


def test_catalog_rejects_conflicting_active_package_entries(tmp_path: Path) -> None:
    root, _ = _catalog_fixture(tmp_path)
    ledger_path = root / "archive_manifest.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    _, first = _first_ledger_entry(ledger)
    conflicting = dict(first)
    conflicting_name = "session-a__camera_head__conflict.tar.zst"
    conflicting_archive = root / "archives" / "train" / conflicting_name
    conflicting_archive.write_bytes(b"different-active-package-bytes")
    conflicting.update(
        archive=f"/content/drive/old-root/{conflicting_name}",
        archive_bytes=conflicting_archive.stat().st_size,
        archive_sha256=hashlib.sha256(conflicting_archive.read_bytes()).hexdigest(),
    )
    ledger["archives"]["conflicting"] = conflicting
    _write_json(ledger_path, ledger)

    with pytest.raises(ValueError, match="ledger key does not match"):
        load_archive_catalog(root)


def test_catalog_selects_package_for_active_detection_config(tmp_path: Path) -> None:
    root, _ = _catalog_fixture(tmp_path)
    detection_manifest_path = root / "metadata" / "derived_detection_classes.json"
    detection_manifest = json.loads(
        detection_manifest_path.read_text(encoding="utf-8")
    )
    # Reproduce early downloader metadata: taxonomy exists, but the wrapper
    # does not expose the derived-box config or its digest.
    detection_manifest.pop("detection_config")
    detection_manifest.pop("detection_config_sha256")
    _write_json(detection_manifest_path, detection_manifest)
    ledger_path = root / "archive_manifest.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    _, first = _first_ledger_entry(ledger)
    historical = dict(first)
    historical_detection_sha = _sha("old-detection-config")
    historical.update(
        detection_config_sha256=historical_detection_sha,
        package_sha256=_package_sha(
            historical["selection_sha256"], historical_detection_sha
        ),
        archive="/content/drive/old-root/deleted-historical-package.tar.zst",
        archive_bytes=123,
        archive_sha256=_sha("deleted-historical-package"),
    )
    ledger["archives"][_ledger_key(historical)] = historical
    _write_json(ledger_path, ledger)

    catalog = load_archive_catalog(root)

    selected = next(
        record
        for record in catalog.records
        if record.session_id == "session-a" and record.sensor == "camera_head"
    )
    assert selected.detection_config_sha256 == _DETECTION_CONFIG_SHA
    assert selected.package_sha256 == _package_sha(selected.selection_sha256)


def test_catalog_ignores_foreign_package_family_without_metadata_config_fields(
    tmp_path: Path,
) -> None:
    root, _ = _catalog_fixture(tmp_path)
    detection_manifest_path = root / "metadata" / "derived_detection_classes.json"
    detection_manifest = json.loads(
        detection_manifest_path.read_text(encoding="utf-8")
    )
    detection_manifest.pop("detection_config")
    detection_manifest.pop("detection_config_sha256")
    _write_json(detection_manifest_path, detection_manifest)

    ledger_path = root / "archive_manifest.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    second_config = _sha("second-complete-config")
    for key, entry in list(ledger["archives"].items()):
        alternative = dict(entry)
        alternative["detection_config_sha256"] = second_config
        alternative["package_sha256"] = _package_sha(
            alternative["selection_sha256"], second_config
        )
        alternative["archive"] = (
            f"/content/drive/old-root/{key}-second-family.tar.zst"
        )
        ledger["archives"][_ledger_key(alternative)] = alternative
    _write_json(ledger_path, ledger)

    catalog = load_archive_catalog(root)

    assert catalog.detection_config_sha256 == _DETECTION_CONFIG_SHA
    assert all(
        record.detection_config_sha256 == _DETECTION_CONFIG_SHA
        for record in catalog.records
    )


def test_catalog_rejects_bad_sha_sidecar(tmp_path: Path) -> None:
    root, _ = _catalog_fixture(tmp_path)
    ledger_path = root / "archive_manifest.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))

    _, first = _first_ledger_entry(ledger)
    first_name = Path(first["archive"]).name
    sidecar = root / "archives" / "train" / f"{first_name}.sha256"
    sidecar.write_text(f"{'0' * 64}  {first_name}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA sidecar"):
        load_archive_catalog(root)


def test_catalog_rejects_tampered_active_detection_config(tmp_path: Path) -> None:
    root, _ = _catalog_fixture(tmp_path)
    manifest_path = root / "metadata" / "derived_detection_classes.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["detection_config"]["min_component_pixels"] = 101
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="config SHA-256"):
        load_archive_catalog(root)


@pytest.mark.parametrize("schema_version", [1, 3])
def test_catalog_accepts_structurally_compatible_detection_manifest_versions(
    tmp_path: Path, schema_version: int
) -> None:
    root, specifications = _catalog_fixture(tmp_path)
    manifest_path = root / "metadata" / "derived_detection_classes.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = schema_version
    _write_json(manifest_path, manifest)

    catalog = load_archive_catalog(root)

    assert len(catalog.records) == len(specifications)
    assert catalog.detection_config_sha256 == _DETECTION_CONFIG_SHA


def test_catalog_rejects_invalid_active_package_provenance(tmp_path: Path) -> None:
    root, _ = _catalog_fixture(tmp_path)
    ledger_path = root / "archive_manifest.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    _, first = _first_ledger_entry(ledger)
    first["package_sha256"] = "0" * 64
    _write_json(ledger_path, ledger)

    with pytest.raises(ValueError, match="package_sha256 does not match"):
        load_archive_catalog(root)


def test_catalog_rejects_sidecar_package_provenance_mismatch(tmp_path: Path) -> None:
    root, _ = _catalog_fixture(tmp_path)
    ledger = json.loads((root / "archive_manifest.json").read_text(encoding="utf-8"))
    _, entry = _first_ledger_entry(ledger)
    archive_name = Path(entry["archive"]).name
    sidecar_path = root / "archives" / "train" / f"{archive_name}.manifest.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["entry"]["detection_config_sha256"] = _sha("foreign-config")
    _write_json(sidecar_path, sidecar)

    with pytest.raises(ValueError, match="detection_config_sha256"):
        load_archive_catalog(root)


def test_catalog_ignores_missing_unrelated_historical_ledger_entry(
    tmp_path: Path,
) -> None:
    root, specifications = _catalog_fixture(tmp_path)
    ledger_path = root / "archive_manifest.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["archives"]["historical"] = {
        "split": "train",
        "session_id": "old-session",
        "sensor": "camera_head",
        "selection_sha256": _sha("obsolete-selection"),
        "archive": "/content/drive/old-root/deleted.tar.zst",
        "archive_bytes": 123,
        "archive_sha256": _sha("deleted-archive"),
        "joint_frames": 1,
    }
    _write_json(ledger_path, ledger)

    catalog = load_archive_catalog(root)

    assert len(catalog.records) == len(specifications)
    assert all(record.session_id != "old-session" for record in catalog.records)


def test_group_split_is_session_stable_excludes_test_and_is_immutable(tmp_path: Path) -> None:
    root, _ = _catalog_fixture(tmp_path)
    catalog = load_archive_catalog(root)
    manifest = tmp_path / "splits" / "seed42.json"
    first = create_or_load_group_split(
        catalog, manifest, seed=42, validation_fraction=0.34
    )
    second = create_or_load_group_split(
        catalog, manifest, seed=42, validation_fraction=0.34
    )

    assert first.manifest_sha256 == second.manifest_sha256
    assert first.train_records == second.train_records
    train_sessions = {record.session_id for record in first.train_records}
    val_sessions = {record.session_id for record in first.validation_records}
    test_sessions = {record.session_id for record in first.official_test_records}
    assert train_sessions and val_sessions
    assert train_sessions.isdisjoint(val_sessions | test_sessions)
    assert val_sessions.isdisjoint(test_sessions)
    # Both cameras from session-a can never be split across train and val.
    assert "session-a" in train_sessions ^ val_sessions
    assert sum(record.session_id == "session-a" for record in first.train_records) in {0, 2}
    assert sum(record.session_id == "session-a" for record in first.validation_records) in {0, 2}

    with pytest.raises(FileExistsError, match="differs"):
        create_or_load_group_split(
            catalog, manifest, seed=7, validation_fraction=0.34
        )


def _joint_manifest_bytes(
    *,
    split: str,
    session_id: str,
    sensor: str,
    selection_sha: str,
    schema_version: int = 2,
    detection_config_sha: str | None = _DETECTION_CONFIG_SHA,
    include_detection_paths: bool = True,
    include_annotation_policy: bool = True,
    frames: int = 1,
) -> bytes:
    samples = []
    for index in range(frames):
        sample = {
            "target_frame": index + 2,
            "rgb_context_paths": [
                f"{sensor}/left/rgb/{index}.png",
                f"{sensor}/left/rgb/{index + 1}.png",
                f"{sensor}/left/rgb/{index + 2}.png",
            ],
            "panoptic_path": f"{sensor}/left/panoptic/{index + 2}.png",
            "depth_path": f"{sensor}/left/depth/{index + 2}.gz",
        }
        if include_detection_paths:
            sample["detection_path"] = (
                f"{sensor}/left/detection/{index + 2}.json"
            )
        samples.append(sample)
    manifest = {
        "schema_version": schema_version,
        "dataset": "SANPO-Real-v0-joint",
        "official_split": split,
        "session_id": session_id,
        "sensor": sensor,
        "selection_sha256": selection_sha,
        "joint_frames": frames,
        "samples": samples,
    }
    if include_annotation_policy:
        manifest["annotation_policy"] = _ANNOTATION_POLICY
    if detection_config_sha is not None:
        manifest["detection"] = {
            "derived": True,
            "config_sha256": detection_config_sha,
        }
    return json.dumps(manifest).encode()


def _write_tar_archive(
    path: Path,
    *,
    split: str = "train",
    session_id: str = "session-a",
    sensor: str = "camera_head",
    selection_sha: str,
    unsafe_name: str | None = None,
    detection_source: str = "packaged_json",
    manifest_schema_version: int = 2,
    manifest_detection_config_sha: str | None = _DETECTION_CONFIG_SHA,
    include_detection_paths: bool = True,
    include_annotation_policy: bool = True,
) -> SanpoArchiveRecord:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as tar:
        name = unsafe_name or (
            f"sanpo-real/{session_id}/{sensor}/left/_sanpo_joint_manifest.json"
        )
        payload = b"unsafe" if unsafe_name else _joint_manifest_bytes(
            split=split,
            session_id=session_id,
            sensor=sensor,
            selection_sha=selection_sha,
            schema_version=manifest_schema_version,
            detection_config_sha=manifest_detection_config_sha,
            include_detection_paths=include_detection_paths,
            include_annotation_policy=include_annotation_policy,
        )
        info = tarfile.TarInfo(name)
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
        if unsafe_name is None:
            manifest = json.loads(payload)
            referenced: set[str] = set()
            for sample in manifest["samples"]:
                referenced.update(sample["rgb_context_paths"])
                referenced.add(sample["panoptic_path"])
                referenced.add(sample["depth_path"])
                if "detection_path" in sample:
                    referenced.add(sample["detection_path"])
            for relative in sorted(referenced):
                content = b"fixture"
                member = tarfile.TarInfo(
                    f"sanpo-real/{session_id}/{relative}"
                )
                member.size = len(content)
                tar.addfile(member, io.BytesIO(content))
    return SanpoArchiveRecord(
        split=split,
        session_id=session_id,
        sensor=sensor,
        annotation_policy=_ANNOTATION_POLICY,
        selection_sha256=selection_sha,
        detection_source=detection_source,
        detection_config_sha256=(
            _DETECTION_CONFIG_SHA if detection_source == "packaged_json" else None
        ),
        package_sha256=(
            _package_sha(selection_sha) if detection_source == "packaged_json" else None
        ),
        archive_path=path.resolve(),
        archive_bytes=path.stat().st_size,
        archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        joint_frames=1,
    )


@contextlib.contextmanager
def _plain_tar_reader(path: Path):
    with path.open("rb") as handle:
        yield handle


def test_materializer_verifies_extracts_and_cleans_owned_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(archives_module, "_zstd_stream_reader", _plain_tar_reader)
    record = _write_tar_archive(
        tmp_path / "source" / "archive.tar.zst", selection_sha=_sha("selection")
    )
    local_root = tmp_path / "cache"

    with ArchiveMaterializer(record, local_root=local_root) as manifest:
        assert manifest.is_file()
        assert manifest.name == "_sanpo_joint_manifest.json"
        assert any(local_root.iterdir())
    assert list(local_root.iterdir()) == []

    broken = SanpoArchiveRecord(
        **{
            **record.__dict__,
            "archive_sha256": "0" * 64,
        }
    )
    with pytest.raises(ValueError, match="SHA verification"):
        with ArchiveMaterializer(broken, local_root=local_root):
            pass
    assert list(local_root.iterdir()) == []


@pytest.mark.parametrize("schema_version", [1, 2])
def test_materializer_accepts_legacy_manifest_without_detection_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    schema_version: int,
) -> None:
    monkeypatch.setattr(archives_module, "_zstd_stream_reader", _plain_tar_reader)
    record = _write_tar_archive(
        tmp_path / "source" / f"legacy-v{schema_version}.tar.zst",
        selection_sha=_sha("legacy-selection"),
        detection_source="panoptic_on_load",
        manifest_schema_version=schema_version,
        manifest_detection_config_sha=None,
        include_detection_paths=False,
        include_annotation_policy=False,
    )
    local_root = tmp_path / "cache"

    with ArchiveMaterializer(record, local_root=local_root) as manifest_path:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["schema_version"] == schema_version
        assert "detection" not in manifest
        assert "detection_path" not in manifest["samples"][0]
    assert list(local_root.iterdir()) == []


def test_materializer_ignores_unversioned_legacy_detection_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(archives_module, "_zstd_stream_reader", _plain_tar_reader)
    record = _write_tar_archive(
        tmp_path / "source" / "legacy-unversioned-detection.tar.zst",
        selection_sha=_sha("legacy-selection"),
        detection_source="panoptic_on_load",
        manifest_schema_version=1,
        manifest_detection_config_sha=_sha("untrusted-legacy-config"),
    )

    with ArchiveMaterializer(record, local_root=tmp_path / "cache"):
        pass


@pytest.mark.parametrize("manifest_detection_config_sha", [None, _sha("foreign")])
def test_materializer_requires_modern_detection_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    manifest_detection_config_sha: str | None,
) -> None:
    monkeypatch.setattr(archives_module, "_zstd_stream_reader", _plain_tar_reader)
    record = _write_tar_archive(
        tmp_path / "source" / "modern-bad-provenance.tar.zst",
        selection_sha=_sha("selection"),
        manifest_detection_config_sha=manifest_detection_config_sha,
    )
    local_root = tmp_path / "cache"

    with pytest.raises(ValueError, match="detection provenance"):
        with ArchiveMaterializer(record, local_root=local_root):
            pass
    assert list(local_root.iterdir()) == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("annotation_policy", "foreign_policy"),
        ("selection_sha256", _sha("foreign-selection")),
    ],
)
def test_materializer_keeps_legacy_identity_validation_strict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
) -> None:
    monkeypatch.setattr(archives_module, "_zstd_stream_reader", _plain_tar_reader)
    record = _write_tar_archive(
        tmp_path / "source" / f"legacy-bad-{field}.tar.zst",
        selection_sha=_sha("selection"),
        detection_source="panoptic_on_load",
        manifest_schema_version=1,
        manifest_detection_config_sha=None,
        include_detection_paths=False,
    )
    conflicting = SanpoArchiveRecord(**{**record.__dict__, field: value})
    local_root = tmp_path / "cache"

    with pytest.raises(ValueError, match="manifest disagrees"):
        with ArchiveMaterializer(conflicting, local_root=local_root):
            pass
    assert list(local_root.iterdir()) == []


def test_materializer_rejects_tar_traversal_and_cleans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(archives_module, "_zstd_stream_reader", _plain_tar_reader)
    record = _write_tar_archive(
        tmp_path / "source" / "unsafe.tar.zst",
        selection_sha=_sha("selection"),
        unsafe_name="sanpo-real/../escape.txt",
    )
    local_root = tmp_path / "cache"
    with pytest.raises(ValueError, match="safely rooted"):
        with ArchiveMaterializer(record, local_root=local_root):
            pass
    assert list(local_root.iterdir()) == []
    assert not (tmp_path / "escape.txt").exists()


def test_persistent_stage_extracts_once_resumes_and_cleans_owned_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(archives_module, "_zstd_stream_reader", _plain_tar_reader)
    records = (
        _write_tar_archive(
            tmp_path / "source" / "first.tar.zst",
            session_id="session-a",
            sensor="camera_head",
            selection_sha=_sha("stage-a"),
        ),
        _write_tar_archive(
            tmp_path / "source" / "second.tar.zst",
            session_id="session-b",
            sensor="camera_chest",
            selection_sha=_sha("stage-b"),
        ),
    )
    copied: list[str] = []
    original_copy = archives_module._copy_verified

    def counted_copy(record, destination):
        copied.append(record.session_id)
        return original_copy(record, destination)

    monkeypatch.setattr(archives_module, "_copy_verified", counted_copy)
    root = tmp_path / "ssd-stage"
    stage = LocalArchiveStage(
        records,
        local_root=root,
        purpose="official_train",
        expansion_factor=1.0,
        reserve_bytes=0,
    )
    progress: list[tuple[int, str]] = []
    first = stage.prepare_all(
        lambda index, total, record, status: progress.append((index, status))
    )

    assert first["complete"] is True
    assert first["completed_count"] == 2
    assert first["completed_extracted_bytes"] > 0
    assert copied == ["session-a", "session-b"]
    assert progress == [(1, "extract"), (2, "extract")]
    assert stage.require_complete()["record_count"] == 2
    for record in records:
        assert stage.materialize(record).is_file()

    progress.clear()
    second = stage.prepare_all(
        lambda index, total, record, status: progress.append((index, status))
    )
    assert second["complete"] is True
    assert copied == ["session-a", "session-b"]
    assert progress == [(1, "cached"), (2, "cached")]

    stage.cleanup()
    assert not root.exists()
    assert all(record.archive_path.is_file() for record in records)


def test_persistent_stage_enforces_official_split_and_complete_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(archives_module, "_zstd_stream_reader", _plain_tar_reader)
    train = _write_tar_archive(
        tmp_path / "source" / "train.tar.zst",
        selection_sha=_sha("train-stage"),
    )
    test = _write_tar_archive(
        tmp_path / "source" / "test.tar.zst",
        split="test",
        session_id="test-session",
        selection_sha=_sha("test-stage"),
    )
    with pytest.raises(ValueError, match="official-test"):
        LocalArchiveStage(
            (train,),
            local_root=tmp_path / "wrong-test",
            purpose="official_test",
        )
    with pytest.raises(ValueError, match="official-train"):
        LocalArchiveStage(
            (test,),
            local_root=tmp_path / "wrong-train",
            purpose="official_train",
        )

    stage = LocalArchiveStage(
        (train,),
        local_root=tmp_path / "incomplete",
        purpose="official_train",
        expansion_factor=1.0,
        reserve_bytes=0,
    )
    with pytest.raises(RuntimeError, match="stage-train"):
        stage.require_complete()


def test_persistent_stage_gate_detects_changed_payload_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(archives_module, "_zstd_stream_reader", _plain_tar_reader)
    record = _write_tar_archive(
        tmp_path / "source" / "integrity.tar.zst",
        selection_sha=_sha("integrity-stage"),
    )
    stage = LocalArchiveStage(
        (record,),
        local_root=tmp_path / "integrity-stage",
        purpose="official_train",
        expansion_factor=1.0,
        reserve_bytes=0,
    )
    stage.prepare_all()
    manifest = stage.materialize(record)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    changed = manifest.parents[2] / payload["samples"][0]["rgb_context_paths"][0]
    changed.write_bytes(b"x")

    with pytest.raises(RuntimeError, match="payload signature changed"):
        stage.require_complete()


def test_persistent_stage_disk_plan_uses_exact_source_bytes(
    tmp_path: Path,
) -> None:
    original = _write_tar_archive(
        tmp_path / "source" / "source-size.tar.zst",
        selection_sha=_sha("source-size-stage"),
    )
    record = SanpoArchiveRecord(
        **{**original.__dict__, "source_bytes": original.archive_bytes * 4}
    )
    stage = LocalArchiveStage(
        (record,),
        local_root=tmp_path / "source-size-stage",
        purpose="official_train",
        expansion_factor=1.0,
        reserve_bytes=0,
    )
    plan = stage.disk_plan()
    assert plan["source_bytes_records"] == 1
    assert plan["exact_source_bytes"] == record.source_bytes
    assert plan["estimated_unpacked_bytes"] == record.source_bytes


def test_persistent_stage_checks_aggregate_disk_before_copying(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(archives_module, "_zstd_stream_reader", _plain_tar_reader)
    record = _write_tar_archive(
        tmp_path / "source" / "disk.tar.zst",
        selection_sha=_sha("disk-stage"),
    )
    copied = False

    def unexpected_copy(record, destination):
        nonlocal copied
        copied = True

    monkeypatch.setattr(archives_module, "_copy_verified", unexpected_copy)
    monkeypatch.setattr(
        archives_module.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(total=100, used=99, free=1),
    )
    stage = LocalArchiveStage(
        (record,),
        local_root=tmp_path / "too-small",
        purpose="official_train",
        expansion_factor=1.0,
        reserve_bytes=1,
    )
    with pytest.raises(OSError, match="insufficient local SSD"):
        stage.prepare_all()
    assert copied is False


def test_persistent_stage_cleanup_refuses_foreign_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(archives_module, "_zstd_stream_reader", _plain_tar_reader)
    record = _write_tar_archive(
        tmp_path / "source" / "owned.tar.zst",
        selection_sha=_sha("owned-stage"),
    )
    root = tmp_path / "owned-stage"
    stage = LocalArchiveStage(
        (record,),
        local_root=root,
        purpose="official_train",
        expansion_factor=1.0,
        reserve_bytes=0,
    )
    stage.prepare_all()
    marker = json.loads(stage.marker_path.read_text(encoding="utf-8"))
    marker["stage_sha256"] = "0" * 64
    stage.marker_path.write_text(json.dumps(marker), encoding="utf-8")

    with pytest.raises(RuntimeError, match="different record selection"):
        stage.cleanup()
    assert root.exists()


def test_persistent_stage_never_claims_nonempty_unowned_directory(
    tmp_path: Path,
) -> None:
    record = _write_tar_archive(
        tmp_path / "source" / "unowned.tar.zst",
        selection_sha=_sha("unowned-stage"),
    )
    root = tmp_path / "unowned"
    root.mkdir()
    sentinel = root / "user-file.txt"
    sentinel.write_text("keep", encoding="utf-8")
    stage = LocalArchiveStage(
        (record,),
        local_root=root,
        purpose="official_train",
        expansion_factor=1.0,
        reserve_bytes=0,
    )
    with pytest.raises(RuntimeError, match="non-empty unowned"):
        stage.disk_plan()
    assert sentinel.read_text(encoding="utf-8") == "keep"


class _FakeDataset(Dataset):
    lengths: dict[str, int] = {}
    detection_modes: list[tuple[str, bool]] = []

    def __init__(
        self,
        manifest_path: Path,
        *,
        use_packaged_detection: bool,
        **_: object,
    ) -> None:
        self.name = Path(manifest_path).name
        self.detection_modes.append((self.name, use_packaged_detection))

    def __len__(self) -> int:
        return self.lengths[self.name]

    def __getitem__(self, index: int):
        identity = int(self.name.removeprefix("shard-")) * 100 + index
        clip = torch.full((3, 3, 2, 2), float(identity))
        targets = {
            "detection": {
                "boxes": torch.empty((0, 4)),
                "labels": torch.empty((0,), dtype=torch.long),
            },
            "segmentation": torch.zeros((2, 2), dtype=torch.long),
            "segmentation_valid": torch.ones((2, 2), dtype=torch.bool),
            "depth": torch.ones((1, 2, 2)),
            "depth_valid": torch.ones((1, 2, 2), dtype=torch.bool),
        }
        return clip, targets


class _FakeMaterializer:
    order: list[str] = []

    def __init__(self, record: SanpoArchiveRecord, *, local_root: Path) -> None:
        self.record = record

    def __enter__(self) -> Path:
        self.order.append(self.record.session_id)
        return Path(f"shard-{self.record.session_id.removeprefix('session-')}")

    def __exit__(self, *args: object) -> None:
        return None


def _fake_records(tmp_path: Path) -> tuple[SanpoArchiveRecord, ...]:
    records = []
    for index, frames in enumerate((3, 5, 4, 2)):
        path = tmp_path / f"{index}.tar.zst"
        path.write_bytes(str(index).encode())
        records.append(
            SanpoArchiveRecord(
                split="train",
                session_id=f"session-{index}",
                sensor="camera_head",
                annotation_policy=_ANNOTATION_POLICY,
                selection_sha256=_sha(f"selection-{index}"),
                detection_source="packaged_json",
                detection_config_sha256=_DETECTION_CONFIG_SHA,
                package_sha256=_package_sha(_sha(f"selection-{index}")),
                archive_path=path,
                archive_bytes=path.stat().st_size,
                archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                joint_frames=frames,
            )
        )
    return tuple(records)


def test_archive_shard_loader_len_epoch_shuffle_and_epoch_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(archives_module, "ArchiveMaterializer", _FakeMaterializer)
    monkeypatch.setattr(archives_module, "SanpoJointDataset", _FakeDataset)
    records = _fake_records(tmp_path)
    _FakeDataset.lengths = {
        f"shard-{index}": record.joint_frames for index, record in enumerate(records)
    }
    loader = ArchiveShardLoader(
        records,
        local_root=tmp_path / "cache",
        batch_size=2,
        image_size=(2, 2),
        seed=42,
        shuffle=True,
    )
    assert len(loader) == 2 + 3 + 2 + 1
    assert loader.sample_count == 14

    def identities() -> list[int]:
        return [int(value) for batch, _ in loader for value in batch[:, 0, 0, 0, 0]]

    epoch_zero = identities()
    repeated = identities()
    assert epoch_zero == repeated
    loader.set_epoch(1)
    epoch_one = identities()
    assert sorted(epoch_zero) == sorted(epoch_one)
    assert epoch_zero != epoch_one
    assert not loader._iterating

    dropped = ArchiveShardLoader(
        records,
        local_root=tmp_path / "cache",
        batch_size=2,
        image_size=(2, 2),
        drop_last=True,
    )
    assert len(dropped) == 1 + 2 + 2 + 1


def test_archive_shard_loader_routes_record_detection_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(archives_module, "ArchiveMaterializer", _FakeMaterializer)
    monkeypatch.setattr(archives_module, "SanpoJointDataset", _FakeDataset)
    modern = _fake_records(tmp_path)[:2]
    legacy = SanpoArchiveRecord(
        **{
            **modern[0].__dict__,
            "detection_source": "panoptic_on_load",
            "detection_config_sha256": None,
            "package_sha256": None,
        }
    )
    records = (legacy, modern[1])
    _FakeDataset.lengths = {
        f"shard-{index}": record.joint_frames for index, record in enumerate(records)
    }
    _FakeDataset.detection_modes = []

    loader = ArchiveShardLoader(
        records,
        local_root=tmp_path / "cache",
        batch_size=2,
        image_size=(2, 2),
    )
    list(loader)

    assert _FakeDataset.detection_modes == [
        ("shard-0", False),
        ("shard-1", True),
    ]


@pytest.mark.parametrize(
    "dataset_kwargs",
    [
        {"use_packaged_detection": True},
        {"detection_min_area": 101},
    ],
)
def test_archive_shard_loader_locks_detection_policy(
    tmp_path: Path, dataset_kwargs: dict[str, object]
) -> None:
    with pytest.raises(ValueError, match="detection"):
        ArchiveShardLoader(
            _fake_records(tmp_path),
            local_root=tmp_path / "cache",
            batch_size=2,
            dataset_kwargs=dataset_kwargs,
        )


def test_canonical_hash_ignores_mapping_order_and_rejects_nan() -> None:
    assert canonical_json_sha256({"a": 1, "b": 2}) == canonical_json_sha256(
        {"b": 2, "a": 1}
    )
    with pytest.raises(ValueError, match="canonical JSON"):
        canonical_json_sha256({"bad": float("nan")})
