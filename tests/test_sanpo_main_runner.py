"""Pure/static contracts for the SANPO main campaign runner."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

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
        "model": {
            "active_tasks": ["segmentation", "depth"],
            "backbone_name": "mobilenetv4_conv_small",
            "pretrained_in1k": False,
            "recurrence_steps": 3,
            "recurrent_c4_channels": 48,
            "recurrent_c5_channels": 64,
            "neck_channels": 48,
            "dense_channels": 32,
            "task_adapter_channels": 32,
            "use_sppf": False,
        },
        "data": {
            "image_size": [288, 512],
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


def test_load_campaign_accepts_locked_twenty_one_one_subset(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.json"
    expected = _config(tmp_path)
    expected["data"].update(  # type: ignore[union-attr]
        fit_session_count=20,
        validation_session_count=1,
        official_test_session_count=1,
    )
    path.write_text(json.dumps(expected), encoding="utf-8")
    _, actual = runner.load_campaign(path)
    assert actual["data"]["fit_session_count"] == 20


def test_sanpo_model_loss_and_metrics_are_dense_only(tmp_path: Path) -> None:
    prepared = SimpleNamespace(config=_config(tmp_path))
    prepared.config["data"].update(  # type: ignore[union-attr]
        depth_min_metres=0.1,
        depth_max_metres=80.0,
    )

    config = runner.model_config(prepared, pretrained=False)
    criterion = runner.create_criterion(prepared)

    assert config.active_tasks == ("segmentation", "depth")
    assert config.tasks.detection_classes is None
    assert criterion.detection_criterion is None

    metrics = runner.create_metrics(prepared)
    assert metrics.detection is None
    assert metrics.segmentation is not None
    assert metrics.depth is not None


def test_sanpo_dense_only_model_physically_prunes_detection_path(
    tmp_path: Path,
) -> None:
    prepared = SimpleNamespace(config=_config(tmp_path))
    model = runner.create_replite_model(runner.model_config(prepared, pretrained=False))

    assert model.active_tasks == ("segmentation", "depth")
    assert model.backbone.out_indices == (0, 1, 2)
    assert not hasattr(model, "detection_head")
    assert not hasattr(model.neck, "detection_path")
    assert not hasattr(model.neck, "recurrent5")
    assert sum(parameter.numel() for parameter in model.parameters()) == 363_969


@pytest.mark.parametrize(
    "counts",
    (
        {"fit_session_count": 20},
        {
            "fit_session_count": 20,
            "validation_session_count": 0,
            "official_test_session_count": 1,
        },
        {
            "fit_session_count": True,
            "validation_session_count": 1,
            "official_test_session_count": 1,
        },
    ),
)
def test_load_campaign_rejects_incomplete_or_invalid_subset_counts(
    tmp_path: Path,
    counts: dict[str, object],
) -> None:
    path = tmp_path / "bad-subset.json"
    value = _config(tmp_path)
    value["data"].update(counts)  # type: ignore[union-attr]
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="fit_session_count"):
        runner.load_campaign(path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda value: value.update(schema_version=2), "schema"),
        (lambda value: value.update(run_id="../escape"), "run_id"),
        (lambda value: value["train"].update(epochs=1), "epochs"),
        (lambda value: value["data"].update(image_size=[287, 512]), "image_size"),
        (
            lambda value: value["model"].update(
                active_tasks=["detection", "segmentation", "depth"]
            ),
            "active_tasks",
        ),
        (
            lambda value: value["model"].update(active_tasks=["depth", "segmentation"]),
            "active_tasks",
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


def test_state_sha256_supports_scalar_integer_buffers() -> None:
    class ScalarBufferModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([1.0, 2.0]))
            self.register_buffer("counter", torch.tensor(7, dtype=torch.long))

    model = ScalarBufferModel()
    first = runner._state_sha256(model)
    assert first == runner._state_sha256(model)
    assert len(first) == 64

    model.counter.add_(1)
    assert runner._state_sha256(model) != first


def test_preflight_amp_overflow_backs_off_without_mutating_model() -> None:
    model = nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(1.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler(
        "cpu", enabled=True, init_scale=8.0, growth_interval=100
    )
    before = model.weight.detach().clone()

    # The unscaled forward loss is finite, but scaling makes its gradient
    # overflow. This is the exact recoverable state the CUDA preflight sees.
    loss = model(torch.ones(1, 1)).sum() * 1e38
    assert bool(torch.isfinite(loss))
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    stepped, connected, nonfinite, old_scale, new_scale = (
        runner._finish_preflight_optimizer_step(model, optimizer, scaler)
    )

    assert not stepped
    assert connected == 1
    assert nonfinite == ("weight",)
    assert (old_scale, new_scale) == (8.0, 4.0)
    torch.testing.assert_close(model.weight, before)

    optimizer.zero_grad(set_to_none=True)
    stable_loss = model(torch.ones(1, 1)).sum()
    scaler.scale(stable_loss).backward()
    scaler.unscale_(optimizer)
    stepped, connected, nonfinite, old_scale, new_scale = (
        runner._finish_preflight_optimizer_step(model, optimizer, scaler)
    )
    assert stepped
    assert connected == 1
    assert nonfinite == ()
    assert (old_scale, new_scale) == (4.0, 4.0)
    assert not torch.equal(model.weight, before)


def test_preflight_gradient_health_distinguishes_missing_and_nonfinite() -> None:
    model = nn.Linear(2, 1)
    assert runner._gradient_health(model) == (0, ())
    model(torch.ones(1, 2)).sum().backward()
    assert runner._gradient_health(model) == (2, ())
    model.weight.grad[0, 0] = torch.inf
    assert runner._gradient_health(model) == (2, ("weight",))


@pytest.mark.parametrize(
    ("proven", "backoffs", "expected"),
    ((4096.0, 2, 1024.0), (4.0, 2, 1.0), (1.0, 2, 1.0), (8.0, 0, 8.0)),
)
def test_production_amp_scale_has_recorded_safety_margin(
    proven: float, backoffs: int, expected: float
) -> None:
    assert runner._production_amp_scale(proven, safety_backoffs=backoffs) == expected


def test_disposable_preflight_reports_proven_amp_scale(monkeypatch) -> None:
    class OneBatch:
        def set_epoch(self, epoch: int) -> None:
            assert epoch == 0

        def __iter__(self):
            yield torch.ones(1, 1, 2, 2), {}

    class ToyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(()))

        def forward(self, inputs):
            value = inputs * self.weight
            return SimpleNamespace(
                segmentation=value,
                depth=value,
                detection=None,
            )

    class ToyCriterion(nn.Module):
        def forward(self, outputs, targets):
            loss = outputs.segmentation.square().mean()
            return {"total": loss, "toy": loss}

    monkeypatch.setattr(runner, "create_model", lambda prepared: ToyModel())
    monkeypatch.setattr(runner, "create_criterion", lambda prepared: ToyCriterion())

    def schedule(prepared, model):
        return torch.optim.SGD(model.parameters(), lr=0.1), None, 1, 0

    monkeypatch.setattr(runner, "create_optimizer_schedule", schedule)
    prepared = SimpleNamespace(
        config={"train": {"seed": 42, "amp_initial_scale": 4096.0}},
        train_loader=OneBatch(),
    )
    result = runner.disposable_preflight(prepared)
    assert result["optimizer_stepped"] is True
    assert result["amp_stable_scale"] >= 1.0
    assert result["amp_production_scale"] == runner._production_amp_scale(
        result["amp_stable_scale"]
    )
    assert result["amp_safety_backoff_count"] == 2
    assert result["amp_safety_margin_factor"] == 4
    assert result["amp_backoff_count"] == len(result["amp_overflow_attempts"])
    assert result["production_model_mutated"] is False
    assert result["active_tasks"] == ["segmentation", "depth"]
    assert result["detection_disabled"] is True


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
    assert '"include_detection": False' in source
    assert "publish_epoch_snapshot(" in source
    assert "restore_latest_snapshot(" in source
    assert '"ACTIVE TASKS"' in source
    assert '"DETECTION LABEL SOURCE"' not in source


def test_dense_only_reports_never_publish_detection_csv(tmp_path: Path) -> None:
    prepared = SimpleNamespace(local_run=tmp_path)
    latest = {
        "epoch": 0,
        "val": {
            "total": 1.0,
            "segmentation/per_class_iou": [],
            "segmentation/present_classes": [],
            "segmentation/num_pixels": 1,
            "depth/num_pixels": 1,
        },
    }

    runner._write_reports(prepared, [latest], latest)

    assert (tmp_path / "val_segmentation_per_class.csv").is_file()
    assert not (tmp_path / "val_detection_per_class.csv").exists()
    assert "val_detection_per_class.csv" not in runner._snapshot_files(tmp_path)


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
    assert parser.parse_args(["stage-train", "--config", "x.json"]).command == (
        "stage-train"
    )
    assert parser.parse_args(["stage-test", "--config", "x.json"]).command == (
        "stage-test"
    )


def _stage_plan() -> dict[str, int | float]:
    return {
        "pending_records": 1,
        "archive_bytes": 10,
        "exact_source_bytes": 9,
        "source_bytes_records": 1,
        "estimated_unpacked_bytes": 11,
        "largest_temporary_archive_bytes": 10,
        "reserve_bytes": 0,
        "required_free_bytes": 21,
        "available_free_bytes": 100,
        "expansion_factor": 1.05,
    }


def test_stage_train_wires_detailed_progress_reporter(
    tmp_path: Path, monkeypatch
) -> None:
    first = SimpleNamespace(key=("train", "session-a", "head", "s", "a"))
    second = SimpleNamespace(key=("train", "session-b", "head", "s", "b"))

    class Stage:
        local_root = tmp_path / "unit-stage"

        def __init__(self) -> None:
            self.calls: list[str] = []
            self.reporter = None

        def disk_plan(self, records):
            self.calls.append("plan")
            assert records == (first, second)
            return _stage_plan()

        def prepare_all(self, *, on_event, records):
            self.calls.append("prepare")
            self.reporter = on_event
            assert records == (first, second)
            return {
                "complete": True,
                "completed_count": 2,
                "record_count": 2,
                "completed_extracted_bytes": 123,
            }

    class Loader:
        def __init__(self, samples: int) -> None:
            self.samples = samples
            self.reporter = None

        def warm_prepared_cache(self, *, on_event):
            self.reporter = on_event
            return {"ready_samples": self.samples}

    stage = Stage()
    train_loader = Loader(11)
    val_loader = Loader(7)
    monkeypatch.setattr(
        runner,
        "prepare",
        lambda filename: SimpleNamespace(
            train_stage=stage,
            train_loader=train_loader,
            val_loader=val_loader,
            catalog=SimpleNamespace(train_records=(first, second)),
            split=SimpleNamespace(
                train_records=(first,),
                validation_records=(second,),
            ),
        ),
    )
    result = runner.stage_official_train("config.json")
    assert result["complete"] is True
    assert stage.calls == ["plan", "prepare"]
    assert isinstance(stage.reporter, runner.StageProgressReporter)
    assert isinstance(train_loader.reporter, runner.PreparedCacheProgressReporter)
    assert isinstance(val_loader.reporter, runner.PreparedCacheProgressReporter)


def test_prepared_cache_reporter_is_newline_safe_and_shows_eta(capsys) -> None:
    ticks = iter((0.0, 1.0, 3.0))
    reporter = runner.PreparedCacheProgressReporter(
        "fit", interval_seconds=0.0, clock=lambda: next(ticks)
    )
    reporter(
        {
            "event": "cache_warm_start",
            "sample_count": 32,
            "estimated_pending_bytes": 2 * 1024**3,
            "available_free_bytes": 100 * 1024**3,
        }
    )
    reporter(
        {
            "event": "cache_warm_progress",
            "batch_index": 1,
            "total_batches": 2,
            "completed_samples": 16,
            "sample_count": 32,
        }
    )
    reporter(
        {
            "event": "cache_warm_end",
            "ready_samples": 32,
            "sample_count": 32,
            "cached_bytes": 2 * 1024**3,
        }
    )
    output = capsys.readouterr().out
    assert "CACHE fit" in output
    assert "1/2" in output
    assert "Samples/s" in output
    assert "ETA" in output
    assert "CACHE fit READY" in output
    assert "\r" not in output
    assert "\x1b[" not in output


def test_stage_progress_reporter_is_newline_safe_and_shows_overall_eta(
    tmp_path: Path, capsys
) -> None:
    tick = -1.0

    def clock() -> float:
        nonlocal tick
        tick += 1.0
        return tick

    reporter = runner.StageProgressReporter(
        tmp_path,
        interval_seconds=0.0,
        clock=clock,
    )
    gib = 1024**3
    record = SimpleNamespace(session_id="session-123456789", sensor="camera_head")
    common = {
        "purpose": "official_train",
        "total_records": 2,
        "total_bytes": 2 * gib,
    }
    reporter(
        {
            **common,
            "event": "stage_start",
            "ready_records": 0,
            "ready_bytes": 0,
        }
    )
    reporter(
        {
            **common,
            "event": "resume_end",
            "ready_records": 0,
            "ready_bytes": 0,
        }
    )
    reporter(
        {
            **common,
            "event": "record_start",
            "record": record,
            "status": "extract",
            "ready_records": 0,
            "ready_bytes": 0,
        }
    )
    for completed in (gib // 4, 3 * gib // 4):
        reporter(
            {
                **common,
                "event": "record_progress",
                "record": record,
                "phase": "copy+sha",
                "bytes_completed": completed,
                "bytes_total": gib,
                "ready_records": 0,
                "ready_bytes": 0,
            }
        )
    reporter(
        {
            **common,
            "event": "record_end",
            "record": record,
            "index": 1,
            "status": "extract",
            "ready_records": 1,
            "ready_bytes": gib,
        }
    )
    reporter(
        {
            **common,
            "event": "stage_end",
            "ready_records": 2,
            "ready_bytes": 2 * gib,
        }
    )

    output = capsys.readouterr().out
    assert "Stage          Shards [overall progress]" in output
    assert "1/2" in output
    assert " 50.0%" in output
    assert "copy+sha" in output
    assert "MB/s" in output
    assert "ETA" in output
    assert "free" in output
    assert "Stage complete in" in output
    assert "\r" not in output
    assert "\x1b[" not in output


def test_stage_test_requires_complete_campaign_then_cleans_train(
    tmp_path: Path, monkeypatch
) -> None:
    events: list[str] = []

    class TrainStage:
        def cleanup(self) -> None:
            events.append("cleanup-train")

    class TestStage:
        local_root = tmp_path / "test-stage"

        def disk_plan(self):
            events.append("plan-test")
            return _stage_plan()

        def prepare_all(self, *, on_event):
            events.append("prepare-test")
            assert isinstance(on_event, runner.StageProgressReporter)
            return {
                "complete": True,
                "completed_count": 48,
                "record_count": 48,
                "completed_extracted_bytes": 456,
            }

    local_run = tmp_path / "run"
    local_run.mkdir()
    (local_run / "run_status.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "epoch_completed": 49,
                "official_test_used": False,
            }
        ),
        encoding="utf-8",
    )
    prepared = SimpleNamespace(
        local_run=local_run,
        train_stage=TrainStage(),
        config={"train": {"epochs": 50}},
    )
    monkeypatch.setattr(runner, "prepare", lambda filename: prepared)
    monkeypatch.setattr(
        runner,
        "_restore",
        lambda value: SimpleNamespace(epoch_completed=49),
    )
    monkeypatch.setattr(runner, "_official_test_stage", lambda value: TestStage())

    result = runner.stage_official_test("config.json")
    assert result["complete"] is True
    assert events == ["cleanup-train", "plan-test", "prepare-test"]


def test_subset_campaign_never_deletes_shared_stage(capsys) -> None:
    class Stage:
        def cleanup(self) -> None:
            raise AssertionError("shared stage must not be deleted")

    prepared = SimpleNamespace(
        train_stage=Stage(),
        config={
            "run_id": "subset20-v1",
            "data": {"local_staging": {"cache_id": "full-v4"}},
        },
    )
    assert runner._cleanup_private_train_stage(prepared) is False
    assert "Shared official-train SSD cache" in capsys.readouterr().out


def test_stage_test_does_not_delete_train_before_full_campaign(
    tmp_path: Path, monkeypatch
) -> None:
    cleaned = False

    class TrainStage:
        def cleanup(self) -> None:
            nonlocal cleaned
            cleaned = True

    local_run = tmp_path / "run"
    local_run.mkdir()
    (local_run / "run_status.json").write_text(
        json.dumps(
            {
                "status": "training",
                "epoch_completed": 12,
                "official_test_used": False,
            }
        ),
        encoding="utf-8",
    )
    prepared = SimpleNamespace(
        local_run=local_run,
        train_stage=TrainStage(),
        config={"train": {"epochs": 50}},
    )
    monkeypatch.setattr(runner, "prepare", lambda filename: prepared)
    monkeypatch.setattr(
        runner,
        "_restore",
        lambda value: SimpleNamespace(epoch_completed=12),
    )
    with pytest.raises(PermissionError, match="full checksum-valid"):
        runner.stage_official_test("config.json")
    assert cleaned is False


def test_official_test_stage_contains_only_the_frozen_holdout(
    tmp_path: Path,
    monkeypatch,
) -> None:
    holdout = SimpleNamespace(session_id="test-holdout", key=("test", "holdout"))
    captured: dict[str, object] = {}

    class Stage:
        def __init__(self, records, **kwargs) -> None:
            captured["records"] = tuple(records)
            captured.update(kwargs)

    monkeypatch.setattr(runner, "LocalArchiveStage", Stage)
    prepared = SimpleNamespace(
        split=SimpleNamespace(official_test_records=(holdout,)),
        local_root=tmp_path,
        config={
            "run_id": "twenty-one-one",
            "data": {
                "local_staging": {
                    "cache_id": "twenty-one-one",
                    "expansion_factor": 1.03,
                    "reserve_gib": 4.0,
                }
            },
        },
    )

    runner._official_test_stage(prepared)

    assert captured["records"] == (holdout,)
    assert captured["purpose"] == "official_test"
    assert str(captured["local_root"]).endswith("/stages/twenty-one-one/official_test")


def test_cached_inspection_is_reused_only_when_context_matches(
    tmp_path: Path,
) -> None:
    context = SnapshotContext(
        source_sha256="a" * 64,
        config_sha256="b" * 64,
        catalog_sha256="c" * 64,
        split_sha256="d" * 64,
    )
    prepared = SimpleNamespace(
        local_root=tmp_path,
        config={"run_id": "unit_seed42"},
        context=context,
    )
    path = tmp_path / "inspections" / "unit_seed42.json"
    path.parent.mkdir(parents=True)
    payload = {
        "schema_version": 1,
        "protocol_id": runner.PROTOCOL_ID,
        "context": context.as_dict(),
        "data": {},
        "model_config": {},
        "pretrained": {},
        "parameters": {},
        "schedule": {},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert runner._load_cached_inspection(prepared) == payload

    payload["context"]["source_sha256"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert runner._load_cached_inspection(prepared) is None


def test_execution_plan_prints_exact_batches_updates_and_log_cadence(
    capsys,
) -> None:
    class Loader:
        def __init__(self, batches: int, samples: int) -> None:
            self.batches = batches
            self.sample_count = samples

        def __len__(self) -> int:
            return self.batches

    prepared = SimpleNamespace(
        config={
            "data": {"batch_size": 4},
            "train": {"progress_every_n_steps": 10},
        },
        train_loader=Loader(125, 500),
        val_loader=Loader(25, 100),
        split=SimpleNamespace(
            train_records=(
                SimpleNamespace(session_id="fit-a", joint_frames=250),
                SimpleNamespace(session_id="fit-b", joint_frames=250),
            ),
            validation_records=(SimpleNamespace(session_id="val-a", joint_frames=100),),
            official_test_records=(
                SimpleNamespace(session_id="test-a", joint_frames=9),
            ),
        ),
    )
    objects = SimpleNamespace(
        config=SimpleNamespace(grad_accum_steps=2, epochs=50),
        total_steps=3_150,
        warmup_steps=158,
    )

    runner._print_execution_plan(prepared, objects, start_epoch=1)

    output = capsys.readouterr().out
    assert "epochs 2->50" in output
    assert "micro_batch=4" in output
    assert "effective_batch=8" in output
    assert "train=500 samples, 125 batches/epoch" in output
    assert "optimizer=63 updates/epoch" in output
    assert "validation=100 samples, 25 batches/epoch" in output
    assert "sessions fit=2 | validation=1 | official-test holdout=1" in output
    assert "9 samples, 3 future batches; NOT LOADED" in output
    assert "campaign optimizer updates=3,150" in output
    assert "warmup=158" in output
    assert "progress every=10 batches" in output
