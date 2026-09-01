"""Durable versioned snapshot publication and resume tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from replite.training.snapshot import (
    SNAPSHOT_MANIFEST,
    SnapshotConflictError,
    SnapshotContext,
    SnapshotValidationError,
    publish_epoch_snapshot,
    restore_latest_snapshot,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _context(character: str = "a") -> SnapshotContext:
    return SnapshotContext(
        source_sha256=character * 64,
        config_sha256="b" * 64,
        catalog_sha256="c" * 64,
        split_sha256="d" * 64,
    )


def _local_run(root: Path, payload: bytes = b"epoch checkpoint") -> Path:
    checkpoint = root / "checkpoints" / "last.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(payload)
    digest = _sha256(checkpoint)
    checkpoint.with_name("last.pt.sha256").write_text(
        f"{digest}  last.pt\n", encoding="utf-8"
    )
    (root / "resolved_config.json").write_text('{"seed":42}\n', encoding="utf-8")
    (root / "timing.csv").write_text("epoch,seconds\n1,12.5\n", encoding="utf-8")
    return root


def test_publish_snapshot_stages_manifest_last_and_is_idempotent(tmp_path: Path) -> None:
    local = _local_run(tmp_path / "local")
    remote = tmp_path / "drive" / "run"
    published = publish_epoch_snapshot(
        local,
        remote,
        epoch_completed=1,
        context=_context(),
        files=("resolved_config.json", "timing.csv"),
    )

    assert published.snapshot_dir == remote / "snapshots" / "epoch_000001"
    assert published.idempotent is False
    assert not (remote / "snapshots" / "epoch_000001.uploading").exists()
    manifest = json.loads((published.snapshot_dir / SNAPSHOT_MANIFEST).read_text())
    assert manifest["context"] == _context().as_dict()
    assert manifest["checkpoint"]["path"] == "checkpoints/last.pt"
    assert [entry["path"] for entry in manifest["files"]] == [
        "checkpoints/last.pt",
        "checkpoints/last.pt.sha256",
        "resolved_config.json",
        "timing.csv",
    ]
    for record in manifest["files"]:
        artifact = published.snapshot_dir / record["path"]
        assert artifact.stat().st_size == record["bytes"]
        assert _sha256(artifact) == record["sha256"]

    repeated = publish_epoch_snapshot(
        local,
        remote,
        epoch_completed=1,
        context=_context(),
        files=("resolved_config.json", "timing.csv"),
    )
    assert repeated.idempotent is True
    assert repeated.snapshot_dir == published.snapshot_dir


def test_completed_version_is_never_overwritten(tmp_path: Path) -> None:
    local = _local_run(tmp_path / "local")
    remote = tmp_path / "remote"
    published = publish_epoch_snapshot(
        local, remote, epoch_completed=3, context=_context()
    )
    original = (published.snapshot_dir / "checkpoints" / "last.pt").read_bytes()
    _local_run(local, b"different checkpoint")

    with pytest.raises(SnapshotConflictError, match="already differs"):
        publish_epoch_snapshot(local, remote, epoch_completed=3, context=_context())
    assert (published.snapshot_dir / "checkpoints" / "last.pt").read_bytes() == original


def test_bad_local_checksum_fails_without_deleting_safe_checkpoint(tmp_path: Path) -> None:
    local = _local_run(tmp_path / "local")
    checkpoint = local / "checkpoints" / "last.pt"
    checkpoint.with_name("last.pt.sha256").write_text(
        f"{'0' * 64}  last.pt\n", encoding="utf-8"
    )

    with pytest.raises(SnapshotValidationError, match="sidecar SHA-256 mismatch"):
        publish_epoch_snapshot(
            local, tmp_path / "remote", epoch_completed=1, context=_context()
        )
    assert checkpoint.read_bytes() == b"epoch checkpoint"
    assert checkpoint.exists()


def test_restore_falls_back_from_corrupt_newest_snapshot(tmp_path: Path) -> None:
    local = _local_run(tmp_path / "local", b"epoch one")
    remote = tmp_path / "remote"
    first = publish_epoch_snapshot(
        local, remote, epoch_completed=1, context=_context()
    )
    _local_run(local, b"epoch two")
    newest = publish_epoch_snapshot(
        local, remote, epoch_completed=2, context=_context()
    )
    (newest.snapshot_dir / "checkpoints" / "last.pt").write_bytes(b"corrupt")
    # Incomplete uploads must not participate in resume selection either.
    incomplete = remote / "snapshots" / "epoch_000003.uploading"
    incomplete.mkdir()
    (incomplete / "checkpoints.pt").write_bytes(b"not complete")

    destination = tmp_path / "restored" / "checkpoints" / "last.pt"
    restored = restore_latest_snapshot(
        remote, destination, expected_context=_context()
    )

    assert restored is not None
    assert restored.snapshot_dir == first.snapshot_dir
    assert restored.epoch_completed == 1
    assert destination.read_bytes() == b"epoch one"
    assert destination.with_name("last.pt.sha256").read_text().endswith("  last.pt\n")
    assert _sha256(destination) == restored.checkpoint_sha256


def test_context_mismatch_is_never_accepted(tmp_path: Path) -> None:
    local = _local_run(tmp_path / "local")
    remote = tmp_path / "remote"
    publish_epoch_snapshot(local, remote, epoch_completed=1, context=_context("a"))
    destination = tmp_path / "restore" / "last.pt"

    assert (
        restore_latest_snapshot(
            remote, destination, expected_context=_context("f")
        )
        is None
    )
    assert not destination.exists()


def test_newest_compatible_snapshot_skips_newer_other_context(tmp_path: Path) -> None:
    local = _local_run(tmp_path / "local", b"compatible")
    remote = tmp_path / "remote"
    compatible = publish_epoch_snapshot(
        local, remote, epoch_completed=4, context=_context("a")
    )
    _local_run(local, b"other experiment")
    publish_epoch_snapshot(local, remote, epoch_completed=5, context=_context("f"))

    destination = tmp_path / "restore" / "last.pt"
    restored = restore_latest_snapshot(
        remote, destination, expected_context=_context("a")
    )
    assert restored is not None
    assert restored.snapshot_dir == compatible.snapshot_dir
    assert destination.read_bytes() == b"compatible"


def test_rejects_path_traversal_and_inexact_context_fields(tmp_path: Path) -> None:
    local = _local_run(tmp_path / "local")
    with pytest.raises(ValueError, match="escapes"):
        publish_epoch_snapshot(
            local,
            tmp_path / "remote",
            epoch_completed=1,
            context=_context(),
            files=("../secret",),
        )
    incomplete_context = _context().as_dict()
    incomplete_context.pop("split_sha256")
    with pytest.raises(ValueError, match="match exactly"):
        publish_epoch_snapshot(
            local,
            tmp_path / "remote",
            epoch_completed=1,
            context=incomplete_context,
        )


def test_restore_preserves_existing_local_pair_when_no_snapshot_matches(tmp_path: Path) -> None:
    local = _local_run(tmp_path / "local", b"remote")
    remote = tmp_path / "remote"
    publish_epoch_snapshot(local, remote, epoch_completed=1, context=_context())
    destination_root = _local_run(tmp_path / "existing", b"safe existing")
    destination = destination_root / "checkpoints" / "last.pt"
    existing_sidecar = destination.with_name("last.pt.sha256").read_bytes()

    result = restore_latest_snapshot(
        remote, destination, expected_context=_context("f")
    )
    assert result is None
    assert destination.read_bytes() == b"safe existing"
    assert destination.with_name("last.pt.sha256").read_bytes() == existing_sidecar
