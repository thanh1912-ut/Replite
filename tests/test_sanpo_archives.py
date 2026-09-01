"""Tests for verified, out-of-core SANPO archive loading."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset

import replite.data.sanpo_archives as archives_module
from replite.data import (
    ArchiveMaterializer,
    ArchiveShardLoader,
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
    "component_connectivity": 8,
    "min_component_pixels": 100,
}
_DETECTION_CONFIG_SHA = canonical_json_sha256(_DETECTION_CONFIG)


def _package_sha(selection_sha: str, detection_sha: str = _DETECTION_CONFIG_SHA) -> str:
    return canonical_json_sha256(
        {
            "selection_sha256": selection_sha,
            "detection_config_sha256": detection_sha,
        }
    )


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
            "joint_frames": frames,
        }
        ledger_entries[f"entry-{index}"] = entry
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


def test_catalog_coalesces_equivalent_duplicate_ledger_entries(tmp_path: Path) -> None:
    root, _ = _catalog_fixture(tmp_path)
    baseline = load_archive_catalog(root)
    ledger_path = root / "archive_manifest.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["archives"]["duplicate"] = dict(ledger["archives"]["entry-0"])
    ledger["archives"]["duplicate"]["archive"] = ledger["archives"]["duplicate"][
        "archive"
    ].replace("/old-root/", "/another-old-mount/")
    ledger["archives"]["duplicate"]["verified_utc"] = "2000-01-01T00:00:00Z"
    _write_json(ledger_path, ledger)

    duplicate = load_archive_catalog(root)

    assert duplicate.records == baseline.records
    assert duplicate.catalog_sha256 == baseline.catalog_sha256


def test_catalog_rejects_conflicting_active_package_entries(tmp_path: Path) -> None:
    root, _ = _catalog_fixture(tmp_path)
    ledger_path = root / "archive_manifest.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    conflicting = dict(ledger["archives"]["entry-0"])
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

    with pytest.raises(ValueError, match="exactly one active package"):
        load_archive_catalog(root)


def test_catalog_selects_package_for_active_detection_config(tmp_path: Path) -> None:
    root, _ = _catalog_fixture(tmp_path)
    ledger_path = root / "archive_manifest.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    historical = dict(ledger["archives"]["entry-0"])
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
    ledger["archives"]["historical-package"] = historical
    _write_json(ledger_path, ledger)

    catalog = load_archive_catalog(root)

    selected = next(
        record
        for record in catalog.records
        if record.session_id == "session-a" and record.sensor == "camera_head"
    )
    assert selected.detection_config_sha256 == _DETECTION_CONFIG_SHA
    assert selected.package_sha256 == _package_sha(selected.selection_sha256)


def test_catalog_rejects_bad_sha_sidecar(tmp_path: Path) -> None:
    root, _ = _catalog_fixture(tmp_path)
    ledger_path = root / "archive_manifest.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))

    first_name = Path(ledger["archives"]["entry-0"]["archive"]).name
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


def test_catalog_rejects_invalid_active_package_provenance(tmp_path: Path) -> None:
    root, _ = _catalog_fixture(tmp_path)
    ledger_path = root / "archive_manifest.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["archives"]["entry-0"]["package_sha256"] = "0" * 64
    _write_json(ledger_path, ledger)

    with pytest.raises(ValueError, match="invalid package_sha256"):
        load_archive_catalog(root)


def test_catalog_rejects_sidecar_package_provenance_mismatch(tmp_path: Path) -> None:
    root, _ = _catalog_fixture(tmp_path)
    ledger = json.loads((root / "archive_manifest.json").read_text(encoding="utf-8"))
    entry = ledger["archives"]["entry-0"]
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
    detection_config_sha: str = _DETECTION_CONFIG_SHA,
    frames: int = 1,
) -> bytes:
    samples = [
        {
            "target_frame": index + 2,
            "rgb_context_paths": [
                f"{sensor}/left/rgb/{index}.png",
                f"{sensor}/left/rgb/{index + 1}.png",
                f"{sensor}/left/rgb/{index + 2}.png",
            ],
            "panoptic_path": f"{sensor}/left/panoptic/{index + 2}.png",
            "depth_path": f"{sensor}/left/depth/{index + 2}.gz",
            "detection_path": f"{sensor}/left/detection/{index + 2}.json",
        }
        for index in range(frames)
    ]
    return json.dumps(
        {
            "schema_version": 2,
            "dataset": "SANPO-Real-v0-joint",
            "official_split": split,
            "session_id": session_id,
            "sensor": sensor,
            "annotation_policy": _ANNOTATION_POLICY,
            "selection_sha256": selection_sha,
            "joint_frames": frames,
            "samples": samples,
            "detection": {
                "derived": True,
                "config_sha256": detection_config_sha,
            },
        }
    ).encode()


def _write_tar_archive(
    path: Path,
    *,
    split: str = "train",
    session_id: str = "session-a",
    sensor: str = "camera_head",
    selection_sha: str,
    unsafe_name: str | None = None,
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
        )
        info = tarfile.TarInfo(name)
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    return SanpoArchiveRecord(
        split=split,
        session_id=session_id,
        sensor=sensor,
        annotation_policy=_ANNOTATION_POLICY,
        selection_sha256=selection_sha,
        detection_config_sha256=_DETECTION_CONFIG_SHA,
        package_sha256=_package_sha(selection_sha),
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


class _FakeDataset(Dataset):
    lengths: dict[str, int] = {}

    def __init__(self, manifest_path: Path, **_: object) -> None:
        self.name = Path(manifest_path).name

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


def test_canonical_hash_ignores_mapping_order_and_rejects_nan() -> None:
    assert canonical_json_sha256({"a": 1, "b": 2}) == canonical_json_sha256(
        {"b": 2, "a": 1}
    )
    with pytest.raises(ValueError, match="canonical JSON"):
        canonical_json_sha256({"bad": float("nan")})
