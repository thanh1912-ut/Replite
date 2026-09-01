"""Pure/static contracts for the SANPO main campaign runner."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from replite.training import SnapshotContext
from tools import train_sanpo_main as runner


def _config(tmp_path: Path) -> dict[str, object]:
    return {
        "schema_version": 1,
        "run_id": "unit_seed42",
        "source_repository": "https://example.test/replite.git",
        "source_commit": "a" * 40,
        "drive_data_root": str(tmp_path / "data"),
        "drive_runs_root": str(tmp_path / "runs"),
        "local_work_root": str(tmp_path / "local"),
        "model": {},
        "data": {
            "image_size": [288, 512],
            "detection_min_component_pixels": 100,
        },
        "train": {"epochs": 50},
        "metrics": {},
    }


def test_load_campaign_accepts_visible_notebook_shape(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    expected = _config(tmp_path)
    path.write_text(json.dumps(expected), encoding="utf-8")
    resolved, actual = runner.load_campaign(path)
    assert resolved == path.resolve()
    assert actual == expected


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda value: value.update(schema_version=2), "schema"),
        (lambda value: value.update(run_id="../escape"), "run_id"),
        (lambda value: value["train"].update(epochs=1), "epochs"),
        (lambda value: value["data"].update(image_size=[287, 512]), "image_size"),
        (
            lambda value: value["data"].update(
                detection_min_component_pixels=101
            ),
            "detection_min_component_pixels",
        ),
        (
            lambda value: value["data"].update(
                detection_min_component_pixels=100.0
            ),
            "detection_min_component_pixels",
        ),
    ),
)
def test_load_campaign_rejects_unsafe_or_incompatible_config(
    tmp_path: Path, mutation, message: str
) -> None:
    value = _config(tmp_path)
    mutation(value)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        runner.load_campaign(path)


def test_approval_token_is_deterministic_and_context_bound() -> None:
    context = SnapshotContext(
        source_sha256="a" * 64,
        config_sha256="b" * 64,
        catalog_sha256="c" * 64,
        split_sha256="d" * 64,
    )
    token = runner.approval_token(context, "e" * 64, "f" * 64)
    assert len(token) == 64
    assert token == runner.approval_token(context, "e" * 64, "f" * 64)
    changed = SnapshotContext(
        source_sha256="0" * 64,
        config_sha256="b" * 64,
        catalog_sha256="c" * 64,
        split_sha256="d" * 64,
    )
    assert token != runner.approval_token(changed, "e" * 64, "f" * 64)


def test_catalog_contract_locks_full_download_and_no_split_overlap() -> None:
    def records(count: int, frames: int, prefix: str):
        base, remainder = divmod(frames, count)
        return tuple(
            SimpleNamespace(
                joint_frames=base + (index < remainder),
                session_id=f"{prefix}-{index}",
            )
            for index in range(count)
        )

    train = records(186, 14_718, "train")
    test = records(48, 3_803, "test")
    catalog = SimpleNamespace(
        records=train + test,
        train_records=train,
        test_records=test,
    )
    runner._assert_catalog(catalog)
    broken = SimpleNamespace(
        records=catalog.records[:-1],
        train_records=train,
        test_records=test[:-1],
    )
    with pytest.raises(ValueError, match="234-archive"):
        runner._assert_catalog(broken)


def test_runner_never_builds_a_loader_from_official_test() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert "ArchiveShardLoader(split.train_records" in source
    assert "split.validation_records, shuffle=False" in source
    assert "ArchiveShardLoader(split.official_test_records" not in source
    assert '"official_test_used": False' in source
    assert "stop_after_epoch=1" in source
    assert "publish_epoch_snapshot(" in source
    assert "restore_latest_snapshot(" in source
    assert '"DETECTION LABEL SOURCE"' in source


def test_cli_requires_explicit_approval_for_main() -> None:
    parser = runner.build_parser()
    inspect = parser.parse_args(["inspect", "--config", "x.json"])
    assert inspect.command == "inspect"
    with pytest.raises(SystemExit):
        parser.parse_args(["train", "--config", "x.json"])
    train = parser.parse_args(
        ["train", "--config", "x.json", "--approval-token", "token"]
    )
    assert train.approval_token == "token"
