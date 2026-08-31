"""Training optimizer, log, and checkpoint reliability tests."""

from __future__ import annotations

import csv
import json
import random

import pytest
import torch
from torch import nn

from replite.training.checkpoint import (
    CheckpointIntegrityError,
    CheckpointManager,
    load_training_checkpoint,
    save_training_checkpoint,
)
from replite.training.logging import TrainingLogger
from replite.training.optim import WarmupCosineScheduler, build_adamw_param_groups


class MetadataModel(nn.Module):
    def __init__(self, width: int = 2, *, pretrained: bool = False) -> None:
        super().__init__()
        self.linear = nn.Linear(width, 1)
        self.width = width
        self.pretrained = pretrained

    @property
    def model_metadata(self):
        return {
            "model": "test",
            "config": {"width": self.width, "pretrained": self.pretrained},
        }

    def forward(self, inputs):
        return self.linear(inputs)


def _objects(*, width: int = 2, pretrained: bool = False):
    model = MetadataModel(width, pretrained=pretrained)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = WarmupCosineScheduler(optimizer, total_steps=4, warmup_steps=1)
    criterion = nn.MSELoss()
    return model, optimizer, scheduler, criterion


def _one_update(model, optimizer, scheduler):
    loss = model(torch.ones(2, model.width)).square().mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()


def test_checkpoint_round_trip_restores_every_state_and_rng(tmp_path) -> None:
    torch.manual_seed(10)
    random.seed(10)
    model, optimizer, scheduler, criterion = _objects()
    _one_update(model, optimizer, scheduler)
    expected_parameters = {name: value.detach().clone() for name, value in model.state_dict().items()}
    path = tmp_path / "epoch.pt"
    digest = save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        epoch_completed=2,
        global_step=7,
        trainer_config={"epochs": 4},
        best_metrics={"val/total": 0.4},
        extra={"amp_skip_count": 1},
    )
    assert len(digest) == 64
    assert (tmp_path / "epoch.pt.sha256").read_text().startswith(digest)
    expected_torch = torch.rand(3)
    expected_python = random.random()

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(100)
    torch.manual_seed(99)
    random.seed(99)
    state = load_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        trainer_config={"epochs": 4},
    )
    assert state.next_epoch == 3
    assert state.global_step == 7
    assert state.best_metrics == {"val/total": 0.4}
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected_parameters[name], rtol=0, atol=0)
    torch.testing.assert_close(torch.rand(3), expected_torch, rtol=0, atol=0)
    assert random.random() == expected_python


def test_pretrained_initialization_flag_does_not_change_architecture_signature(tmp_path) -> None:
    source, optimizer, scheduler, criterion = _objects(pretrained=True)
    path = tmp_path / "model.pt"
    save_training_checkpoint(
        path,
        model=source,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        epoch_completed=0,
        global_step=0,
        trainer_config={"same": True},
    )
    restored, restored_optimizer, restored_scheduler, restored_criterion = _objects(pretrained=False)
    load_training_checkpoint(
        path,
        model=restored,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        criterion=restored_criterion,
        trainer_config={"same": True},
    )


def test_config_mismatch_and_checksum_corruption_are_rejected(tmp_path) -> None:
    model, optimizer, scheduler, criterion = _objects()
    path = tmp_path / "model.pt"
    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        epoch_completed=0,
        global_step=0,
        trainer_config={"seed": 42},
    )
    with pytest.raises(ValueError, match="config hash"):
        load_training_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            trainer_config={"seed": 7},
        )
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(CheckpointIntegrityError, match="SHA-256"):
        load_training_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            trainer_config={"seed": 42},
        )


def test_manager_falls_back_to_checksum_valid_previous_checkpoint(tmp_path) -> None:
    model, optimizer, scheduler, criterion = _objects()
    manager = CheckpointManager(tmp_path)
    common = {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "criterion": criterion,
        "trainer_config": {"seed": 42},
    }
    manager.save_last(epoch_completed=0, global_step=1, **common)
    with torch.no_grad():
        model.linear.weight.add_(5)
    manager.save_last(epoch_completed=1, global_step=2, **common)
    manager.last_path.write_bytes(manager.last_path.read_bytes() + b"broken")

    state = manager.load_latest(**common)
    assert state.checkpoint_path == manager.previous_path
    assert state.next_epoch == 1
    assert state.global_step == 1


def test_logger_appends_jsonl_and_long_form_csv_and_closes(tmp_path) -> None:
    logger = TrainingLogger(tmp_path, run_id="run-42", fsync=True)
    logger.log(
        "train_step",
        {"loss": torch.tensor(1.25), "lr": 0.01},
        epoch=0,
        global_step=1,
        split="train",
        extra={"shape": (2, 3)},
    )
    logger.log("epoch_end", {"loss": 1.0}, epoch=0, global_step=2)
    records = [json.loads(line) for line in logger.jsonl_path.read_text().splitlines()]
    assert [record["event"] for record in records] == ["train_step", "epoch_end"]
    assert records[0]["metrics"] == {"loss": 1.25, "lr": 0.01}
    with logger.csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["metric"] for row in rows] == ["loss", "lr", "loss"]
    logger.close()
    with pytest.raises(RuntimeError, match="closed"):
        logger.log("late")

    with TrainingLogger(tmp_path / "context") as context_logger:
        context_logger.log("start")
    with pytest.raises(RuntimeError, match="closed"):
        context_logger.log("late")


def test_optimizer_groups_are_disjoint_and_scheduler_reaches_minimum() -> None:
    class Grouped(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4))
            self.head = nn.Linear(4, 2)

    model = Grouped()
    groups = build_adamw_param_groups(
        model, lr=1e-3, weight_decay=0.1, backbone_lr_multiplier=0.2
    )
    parameter_ids = [id(parameter) for group in groups for parameter in group["params"]]
    assert len(parameter_ids) == len(set(parameter_ids))
    assert set(parameter_ids) == {id(parameter) for parameter in model.parameters()}
    assert {group["lr"] for group in groups if group["name"].startswith("backbone")} == {2e-4}
    assert all(group["weight_decay"] == 0.0 for group in groups if group["name"].endswith("no_decay"))

    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    scheduler = WarmupCosineScheduler(
        optimizer, total_steps=4, warmup_steps=1, min_lr_ratio=0.1
    )
    rates = [scheduler.get_last_lr()[0]]
    for _ in range(3):
        scheduler.step()
        rates.append(scheduler.get_last_lr()[0])
    assert rates[0] == pytest.approx(1.0)
    assert rates[-1] == pytest.approx(0.1)

