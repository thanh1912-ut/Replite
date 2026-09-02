"""CPU smoke and invariant tests for the generic training engine."""

from __future__ import annotations

import json
import math

import pytest
import torch
from torch import nn

from replite.multitask.config import RepLiteConfig, TaskConfig
from replite.multitask.model import RepLiteMultiTaskModel
from replite.training.checkpoint import CheckpointManager
from replite.training.losses import MultiTaskCriterion
from replite.training.logging import TrainingLogger
from replite.training.sampling import BalancedBatchSampler, balanced_batch_sizes
from replite.training.trainer import Trainer, TrainerConfig, move_to_device


class ScalarModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0))

    @property
    def model_metadata(self):
        return {"model": "scalar", "config": {"pretrained": False}}

    def forward(self, inputs):
        return inputs.reshape(inputs.shape[0]) * self.weight


class MSECriterion(nn.Module):
    def forward(self, prediction, target):
        loss = (prediction - target).square().mean()
        return {"total": loss, "mse": loss}


class CountingScheduler:
    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1

    def state_dict(self):
        return {"steps": self.steps}

    def load_state_dict(self, state):
        self.steps = state["steps"]


class SequenceMetric:
    def __init__(self, name: str, values: list[float]) -> None:
        self.name = name
        self.values = values
        self.index = 0

    def reset(self):
        return None

    def update(self, prediction, target):
        return None

    def compute(self):
        value = self.values[self.index]
        self.index += 1
        return {self.name: value}


def _batch(x: float, y: float):
    return torch.tensor([[[[x]]]]), torch.tensor([y])


def test_short_final_accumulation_window_is_normalized_by_actual_size() -> None:
    model = ScalarModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = CountingScheduler()
    trainer = Trainer(
        model,
        MSECriterion(),
        optimizer,
        TrainerConfig(
            epochs=1,
            grad_accum_steps=2,
            amp=False,
            grad_clip_norm=None,
        ),
        scheduler=scheduler,
        device="cpu",
    )
    result = trainer.train_epoch([_batch(1, 1), _batch(1, 3), _batch(1, 5)], epoch=0)
    assert result["total"] > 0
    assert model.weight.item() == pytest.approx(1.32)
    assert trainer.global_step == 2
    assert scheduler.steps == 2
    assert model.weight.grad is None


def test_unequal_microbatches_and_epoch_losses_are_sample_weighted() -> None:
    model = ScalarModel()
    trainer = Trainer(
        model,
        MSECriterion(),
        torch.optim.SGD(model.parameters(), lr=0.1),
        TrainerConfig(
            epochs=1,
            grad_accum_steps=2,
            amp=False,
            grad_clip_norm=None,
        ),
        device="cpu",
    )
    batches = [
        (
            torch.ones(2, 1, 1, 1),
            torch.tensor([1.0, 1.0]),
        ),
        (
            torch.ones(1, 1, 1, 1),
            torch.tensor([3.0]),
        ),
    ]
    result = trainer.train_epoch(batches, epoch=0)

    # Combined mean loss is (1 + 1 + 9) / 3.  Its gradient at w=0 is
    # (-2 - 2 - 6) / 3, so SGD(lr=.1) reaches exactly 1/3.
    assert result["total"] == pytest.approx(11.0 / 3.0)
    assert model.weight.item() == pytest.approx(1.0 / 3.0)

    validation_model = ScalarModel()
    validation = Trainer(
        validation_model,
        MSECriterion(),
        torch.optim.SGD(validation_model.parameters(), lr=0.1),
        TrainerConfig(epochs=1, amp=False),
        device="cpu",
    )
    assert validation.validate(batches)["total"] == pytest.approx(11.0 / 3.0)


def test_balanced_batch_sampler_uses_all_samples_without_singleton_tail() -> None:
    sizes = balanced_batch_sizes(1809, 16)
    assert len(sizes) == 114
    assert sum(sizes) == 1809
    assert min(sizes) == 15
    assert max(sizes) == 16

    sampler = BalancedBatchSampler(1809, 16, shuffle=True, seed=42)
    epoch_zero = list(sampler)
    assert [len(batch) for batch in epoch_zero] == list(sizes)
    assert sorted(index for batch in epoch_zero for index in batch) == list(range(1809))
    assert list(sampler) == epoch_zero

    sampler.set_epoch(1)
    epoch_one = list(sampler)
    assert epoch_one != epoch_zero
    assert sorted(index for batch in epoch_one for index in batch) == list(range(1809))


def test_trainer_sets_epoch_on_balanced_sampler_and_dataset() -> None:
    class EpochDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.epoch = None

        def __len__(self):
            return 5

        def __getitem__(self, index):
            return torch.tensor([[[float(index + 1)]]]), torch.tensor(float(index))

        def set_epoch(self, epoch):
            self.epoch = epoch

    dataset = EpochDataset()
    sampler = BalancedBatchSampler(dataset, 3, shuffle=True, seed=11)
    loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)
    model = ScalarModel()
    trainer = Trainer(
        model,
        MSECriterion(),
        torch.optim.SGD(model.parameters(), lr=0.01),
        TrainerConfig(epochs=5, amp=False, grad_clip_norm=None),
        device="cpu",
    )
    trainer.train_epoch(loader, epoch=4)
    assert dataset.epoch == 4
    assert sampler.epoch == 4


@pytest.mark.parametrize(
    ("num_samples", "batch_size", "expected"),
    [
        (0, 16, ()),
        (1, 16, (1,)),
        (16, 16, (16,)),
        (17, 16, (9, 8)),
        (31, 16, (16, 15)),
    ],
)
def test_balanced_batch_sizes_edges(num_samples, batch_size, expected) -> None:
    assert balanced_batch_sizes(num_samples, batch_size) == expected


@pytest.mark.parametrize(
    ("metric_name", "alias", "monitor_mode", "values"),
    [
        ("segmentation/miou", "best_miou.pt", "max", [0.40, 0.45]),
        ("depth/abs_rel", "best_absrel.pt", "min", [0.40, 0.35]),
        ("selection/joint", "best_joint.pt", "max", [0.40, 0.45]),
    ],
)
def test_task_alias_tracks_literal_best_independent_of_monitor_delta(
    tmp_path, metric_name, alias, monitor_mode, values
) -> None:
    model = ScalarModel()
    manager = CheckpointManager(tmp_path)
    trainer = Trainer(
        model,
        MSECriterion(),
        torch.optim.SGD(model.parameters(), lr=0.01),
        TrainerConfig(
            epochs=2,
            amp=False,
            monitor=f"val/{metric_name}",
            monitor_mode=monitor_mode,
            early_stopping_min_delta=0.1,
        ),
        device="cpu",
        checkpoint_manager=manager,
        validation_metrics=SequenceMetric(metric_name, values),
    )

    trainer.fit([_batch(1, 1)], [_batch(1, 1)])

    alias_path = tmp_path / alias
    assert alias_path.is_file()
    assert alias_path.with_name(alias_path.name + ".sha256").is_file()
    payload = torch.load(alias_path, map_location="cpu", weights_only=False)
    assert payload["progress"]["next_epoch"] == 2
    assert trainer.alias_best_metrics[f"val/{metric_name}"] == values[-1]
    # The early-stop monitor still honors min_delta rather than being mutated
    # by the independently tracked literal-best alias.
    assert trainer.best_metrics[f"val/{metric_name}"] == values[0]


def test_validation_restores_model_mode_and_updates_metric_adapter() -> None:
    class Metric:
        def reset(self):
            self.count = 0

        def update(self, prediction, target):
            self.count += prediction.numel()

        def compute(self):
            return {"count": self.count}

    model = ScalarModel().train()
    trainer = Trainer(
        model,
        MSECriterion(),
        torch.optim.SGD(model.parameters(), lr=0.1),
        TrainerConfig(epochs=1, amp=False),
        device="cpu",
        validation_metrics=Metric(),
    )
    result = trainer.validate([_batch(1, 1), _batch(2, 2)], epoch=0)
    assert model.training
    assert result["count"] == 2
    assert "total" in result


def test_mapping_batch_and_five_dimensional_clip_reach_model_unchanged() -> None:
    class ClipModel(ScalarModel):
        def forward(self, inputs):
            self.seen_shape = tuple(inputs.shape)
            return inputs[:, -1].reshape(inputs.shape[0]) * self.weight

    model = ClipModel()
    trainer = Trainer(
        model,
        MSECriterion(),
        torch.optim.SGD(model.parameters(), lr=0.1),
        TrainerConfig(epochs=1, amp=False),
        device="cpu",
    )
    clip = torch.arange(3.0).reshape(1, 3, 1, 1, 1)
    trainer.train_epoch([{"inputs": clip, "targets": torch.tensor([1.0])}], epoch=0)
    assert model.seen_shape == (1, 3, 1, 1, 1)


def test_event_callback_receives_yolo_progress_fields() -> None:
    class MappingCriterion(nn.Module):
        def forward(self, prediction, target):
            del target
            loss = prediction.square().mean()
            return {"total": loss}

    events = []
    model = ScalarModel()
    trainer = Trainer(
        model,
        MappingCriterion(),
        torch.optim.SGD(model.parameters(), lr=0.1),
        TrainerConfig(epochs=1, amp=False),
        device="cpu",
        event_callback=lambda name, payload: events.append((name, payload)),
    )
    targets = {
        "detection": [
            {
                "labels": torch.tensor([0, 1]),
                "ignore_boxes": torch.zeros(3, 4),
            }
        ]
    }
    trainer.train_epoch([(torch.ones(1, 1, 1, 1), targets)], epoch=0)
    names = [name for name, _ in events]
    assert names == ["train_epoch_start", "train_batch_end", "train_epoch_end"]
    batch = events[1][1]
    assert batch["instances"] == 2
    assert batch["ignored_instances"] == 3
    assert batch["image_size"] == (1, 1)
    assert batch["running_losses"]["total"] >= 0


def test_fit_checkpoint_and_resume_continue_at_next_epoch(tmp_path) -> None:
    config = TrainerConfig(epochs=2, amp=False, monitor="val/total")
    manager = CheckpointManager(tmp_path)
    model = ScalarModel()
    trainer = Trainer(
        model,
        MSECriterion(),
        torch.optim.SGD(model.parameters(), lr=0.1),
        config,
        device="cpu",
        checkpoint_manager=manager,
    )
    history = trainer.fit([_batch(1, 1)], [_batch(1, 1)])
    assert len(history) == 2
    assert manager.last_path.exists()
    assert (tmp_path / "best.pt").exists()

    restored = ScalarModel()
    resumed = Trainer(
        restored,
        MSECriterion(),
        torch.optim.SGD(restored.parameters(), lr=0.1),
        config,
        device="cpu",
        checkpoint_manager=manager,
    )
    state = resumed.resume()
    assert state.next_epoch == 2
    assert resumed.start_epoch == 2
    assert resumed.global_step == trainer.global_step
    torch.testing.assert_close(restored.weight, model.weight, rtol=0, atol=0)


def test_fit_can_pause_after_first_campaign_epoch_and_resume_same_config(tmp_path) -> None:
    config = TrainerConfig(
        epochs=3,
        amp=False,
        validate_every_n_epochs=3,
        checkpoint_every_n_epochs=3,
    )
    manager = CheckpointManager(tmp_path)
    model = ScalarModel()
    trainer = Trainer(
        model,
        MSECriterion(),
        torch.optim.SGD(model.parameters(), lr=0.1),
        config,
        device="cpu",
        checkpoint_manager=manager,
        checkpoint_extra={"split_sha256": "a" * 64},
    )

    pilot = trainer.fit(
        [_batch(1, 1)],
        [_batch(1, 1)],
        stop_after_epoch=1,
    )
    assert [record["epoch"] for record in pilot] == [0]
    assert "val" in pilot[0]
    assert trainer.start_epoch == 1
    assert manager.last_path.is_file()

    resumed_model = ScalarModel()
    resumed = Trainer(
        resumed_model,
        MSECriterion(),
        torch.optim.SGD(resumed_model.parameters(), lr=0.1),
        config,
        device="cpu",
        checkpoint_manager=manager,
        checkpoint_extra={"split_sha256": "a" * 64},
    )
    state = resumed.resume()
    assert state.next_epoch == 1
    assert state.extra["split_sha256"] == "a" * 64
    remainder = resumed.fit(
        [_batch(1, 1)],
        [_batch(1, 1)],
        stop_after_epoch=3,
    )
    assert [record["epoch"] for record in remainder] == [1, 2]
    assert resumed.start_epoch == 3


def test_early_stopping_counter_resumes_exactly_and_saves_stop_boundary(tmp_path) -> None:
    class ScriptedTrainer(Trainer):
        def __init__(self, *args, validation_losses, **kwargs):
            super().__init__(*args, **kwargs)
            self.validation_losses = validation_losses

        def train_epoch(self, loader, *, epoch):
            del loader, epoch
            self.global_step += 1
            return {"total": 1.0}

        def validate(self, loader, *, epoch=None):
            del loader
            assert epoch is not None
            return {"total": self.validation_losses[epoch]}

    losses = [1.0, 1.1, 1.2, 1.3, 0.5, 0.4]
    config = TrainerConfig(
        epochs=6,
        amp=False,
        monitor="val/total",
        early_stopping_patience=3,
        checkpoint_every_n_epochs=6,
    )
    manager = CheckpointManager(tmp_path)
    model = ScalarModel()
    first = ScriptedTrainer(
        model,
        MSECriterion(),
        torch.optim.SGD(model.parameters(), lr=0.1),
        config,
        validation_losses=losses,
        device="cpu",
        checkpoint_manager=manager,
    )
    first_history = first.fit([_batch(1, 1)], [_batch(1, 1)], stop_after_epoch=2)
    assert [record["epoch"] for record in first_history] == [0, 1]
    assert first.early_stopping_bad_epochs == 1
    assert not first.early_stopping_triggered
    assert manager.last_path.is_file()
    assert (tmp_path / "best.pt").is_file()

    resumed_model = ScalarModel()
    resumed = ScriptedTrainer(
        resumed_model,
        MSECriterion(),
        torch.optim.SGD(resumed_model.parameters(), lr=0.1),
        config,
        validation_losses=losses,
        device="cpu",
        checkpoint_manager=manager,
    )
    state = resumed.resume()
    assert state.next_epoch == 2
    assert resumed.early_stopping_bad_epochs == 1
    assert not resumed.early_stopping_triggered

    remainder = resumed.fit([_batch(1, 1)], [_batch(1, 1)])
    assert [record["epoch"] for record in remainder] == [2, 3]
    assert remainder[-1]["early_stopping"]["triggered"] is True
    assert resumed.start_epoch == 4
    assert resumed.early_stopping_bad_epochs == 3
    assert resumed.early_stopping_triggered

    terminal_model = ScalarModel()
    terminal = ScriptedTrainer(
        terminal_model,
        MSECriterion(),
        torch.optim.SGD(terminal_model.parameters(), lr=0.1),
        config,
        validation_losses=losses,
        device="cpu",
        checkpoint_manager=manager,
    )
    terminal_state = terminal.resume()
    assert terminal_state.next_epoch == 4
    assert terminal.early_stopping_bad_epochs == 3
    assert terminal.early_stopping_triggered
    assert terminal.fit([_batch(1, 1)], [_batch(1, 1)]) == []


def test_early_stopping_requires_validation_and_honors_min_delta() -> None:
    config = TrainerConfig(
        epochs=2,
        amp=False,
        early_stopping_patience=1,
        early_stopping_min_delta=0.1,
    )
    model = ScalarModel()
    trainer = Trainer(
        model,
        MSECriterion(),
        torch.optim.SGD(model.parameters(), lr=0.1),
        config,
        device="cpu",
    )
    with pytest.raises(ValueError, match="validation loader"):
        trainer.fit([_batch(1, 1)])
    trainer.best_metrics[config.monitor] = 1.0
    assert not trainer._is_better(0.95)
    assert trainer._is_better(0.89)


def test_resume_rejects_checkpoint_context_before_mutating_model(tmp_path) -> None:
    config = TrainerConfig(epochs=2, amp=False)
    source = ScalarModel()
    manager = CheckpointManager(tmp_path)
    source_trainer = Trainer(
        source,
        MSECriterion(),
        torch.optim.SGD(source.parameters(), lr=0.1),
        config,
        device="cpu",
        checkpoint_manager=manager,
        checkpoint_extra={"split_sha256": "a" * 64},
    )
    source_trainer.fit([_batch(1, 1)], stop_after_epoch=1)

    target = ScalarModel()
    before = target.weight.detach().clone()
    target_trainer = Trainer(
        target,
        MSECriterion(),
        torch.optim.SGD(target.parameters(), lr=0.1),
        config,
        device="cpu",
        checkpoint_manager=manager,
        checkpoint_extra={"split_sha256": "b" * 64},
    )
    with pytest.raises(ValueError, match="split_sha256"):
        target_trainer.resume()
    torch.testing.assert_close(target.weight, before, rtol=0, atol=0)


@pytest.mark.parametrize("value", [0, 4, True, 1.5])
def test_fit_rejects_invalid_pause_epoch(value) -> None:
    model = ScalarModel()
    trainer = Trainer(
        model,
        MSECriterion(),
        torch.optim.SGD(model.parameters(), lr=0.1),
        TrainerConfig(epochs=3, amp=False),
        device="cpu",
    )
    with pytest.raises(ValueError, match="stop_after_epoch"):
        trainer.fit([_batch(1, 1)], stop_after_epoch=value)


def test_non_finite_loss_stops_before_optimizer_step() -> None:
    model = ScalarModel()
    trainer = Trainer(
        model,
        MSECriterion(),
        torch.optim.SGD(model.parameters(), lr=0.1),
        TrainerConfig(epochs=1, amp=False),
        device="cpu",
    )
    with pytest.raises(FloatingPointError, match="non-finite"):
        trainer.train_epoch([_batch(float("nan"), 1)], epoch=0)
    assert trainer.global_step == 0


def test_fit_fails_when_validation_monitor_is_missing() -> None:
    model = ScalarModel()
    trainer = Trainer(
        model,
        MSECriterion(),
        torch.optim.SGD(model.parameters(), lr=0.1),
        TrainerConfig(epochs=1, amp=False, monitor="val/not_a_metric"),
        device="cpu",
    )
    with pytest.raises(KeyError, match="not_a_metric"):
        trainer.fit([_batch(1, 1)], [_batch(1, 1)])


def test_skipped_optimizer_update_is_logged_with_scale(tmp_path) -> None:
    model = ScalarModel()
    logger = TrainingLogger(tmp_path, run_id="amp-skip")
    trainer = Trainer(
        model,
        MSECriterion(),
        torch.optim.SGD(model.parameters(), lr=0.1),
        TrainerConfig(epochs=1, amp=False, grad_accum_steps=2),
        device="cpu",
        logger=logger,
    )

    def force_skip(microbatches):
        assert microbatches == 1
        trainer.amp_skip_count += 1
        trainer.optimizer.zero_grad(set_to_none=True)
        return False, None

    trainer._optimizer_step = force_skip
    # An iterator has no len(), so its partial accumulation is flushed after
    # the loop. The skip must still be observable in the durable event log.
    trainer.train_epoch(iter([_batch(1, 1)]), epoch=0)
    logger.close()

    events = [json.loads(line) for line in logger.jsonl_path.read_text().splitlines()]
    assert [event["event"] for event in events] == ["amp_overflow_skip"]
    assert events[0]["metrics"] == {"amp_scale": 1.0, "amp_skip_count": 1.0}
    assert events[0]["extra"] == {"batch_index": 0}


def test_move_to_device_preserves_nested_non_tensor_values() -> None:
    value = {"tensor": torch.ones(1), "list": [torch.zeros(1), "id"], "none": None}
    moved = move_to_device(value, torch.device("cpu"))
    assert moved["tensor"].device.type == "cpu"
    assert moved["list"][1] == "id"
    assert moved["none"] is None


def test_real_replite_joint_cpu_training_step_is_finite() -> None:
    config = RepLiteConfig(
        backbone_name="mobilenetv3_small_050",
        tasks=TaskConfig(detection_classes=2, segmentation_classes=3, depth=True),
        recurrent_c4_channels=12,
        recurrent_c5_channels=16,
        neck_channels=12,
        dense_channels=8,
        task_adapter_channels=8,
        detection_head_channels=12,
        detection_head_blocks=1,
        recurrence_steps=1,
    )
    model = RepLiteMultiTaskModel(config)
    before = model.detection_head.class_predictors[0].weight.detach().clone()
    criterion = MultiTaskCriterion(
        detection_num_classes=2,
        depth_loss_type="l1",
    )
    trainer = Trainer(
        model,
        criterion,
        torch.optim.SGD(model.parameters(), lr=1e-3),
        TrainerConfig(epochs=1, amp=False, grad_clip_norm=1.0),
        device="cpu",
    )
    images = torch.randn(1, 3, 64, 96)
    targets = {
        "detection": [
            {
                "boxes": torch.tensor([[8.0, 8.0, 40.0, 40.0]]),
                "labels": torch.tensor([1], dtype=torch.long),
                "valid_size": (64, 96),
            }
        ],
        "segmentation": torch.randint(0, 3, (1, 64, 96)),
        "depth": torch.full((1, 1, 64, 96), 2.0),
        "depth_valid": torch.ones(1, 1, 64, 96, dtype=torch.bool),
    }
    result = trainer.train_epoch([(images, targets)], epoch=0)
    assert all(torch.isfinite(torch.tensor(value)) for value in result.values())
    assert trainer.global_step == 1
    assert not torch.equal(before, model.detection_head.class_predictors[0].weight)


def test_uninterrupted_and_epoch_boundary_resume_are_exact(tmp_path) -> None:
    config = TrainerConfig(
        epochs=2,
        amp=False,
        grad_clip_norm=None,
        checkpoint_every_n_epochs=1,
    )
    torch.manual_seed(123)
    initial = ScalarModel().state_dict()

    continuous_model = ScalarModel()
    continuous_model.load_state_dict(initial)
    continuous_scheduler = CountingScheduler()
    continuous = Trainer(
        continuous_model,
        MSECriterion(),
        torch.optim.SGD(continuous_model.parameters(), lr=0.1, momentum=0.9),
        config,
        scheduler=continuous_scheduler,
        device="cpu",
    )
    continuous.train_epoch([_batch(1, 1), _batch(2, 3)], epoch=0)
    continuous.train_epoch([_batch(1, 2), _batch(2, 4)], epoch=1)

    interrupted_model = ScalarModel()
    interrupted_model.load_state_dict(initial)
    interrupted_scheduler = CountingScheduler()
    manager = CheckpointManager(tmp_path)
    interrupted = Trainer(
        interrupted_model,
        MSECriterion(),
        torch.optim.SGD(interrupted_model.parameters(), lr=0.1, momentum=0.9),
        config,
        scheduler=interrupted_scheduler,
        checkpoint_manager=manager,
        device="cpu",
    )
    interrupted.train_epoch([_batch(1, 1), _batch(2, 3)], epoch=0)
    manager.save_last(
        model=interrupted.model,
        optimizer=interrupted.optimizer,
        scheduler=interrupted.scheduler,
        scaler=interrupted.scaler,
        criterion=interrupted.criterion,
        epoch_completed=0,
        global_step=interrupted.global_step,
        trainer_config=config,
        extra={"amp_skip_count": interrupted.amp_skip_count},
    )

    resumed_model = ScalarModel()
    resumed_scheduler = CountingScheduler()
    resumed = Trainer(
        resumed_model,
        MSECriterion(),
        torch.optim.SGD(resumed_model.parameters(), lr=0.1, momentum=0.9),
        config,
        scheduler=resumed_scheduler,
        checkpoint_manager=manager,
        device="cpu",
    )
    resumed.resume()
    resumed.train_epoch([_batch(1, 2), _batch(2, 4)], epoch=1)

    torch.testing.assert_close(
        resumed_model.weight,
        continuous_model.weight,
        rtol=0,
        atol=0,
    )
    assert resumed.global_step == continuous.global_step
    assert resumed_scheduler.steps == continuous_scheduler.steps


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_joint_cuda_amp_smoke() -> None:
    config = RepLiteConfig(
        backbone_name="mobilenetv3_small_050",
        tasks=TaskConfig(detection_classes=2, segmentation_classes=3, depth=True),
        recurrent_c4_channels=12,
        recurrent_c5_channels=16,
        neck_channels=12,
        dense_channels=8,
        task_adapter_channels=8,
        detection_head_channels=12,
        detection_head_blocks=1,
        recurrence_steps=1,
    )
    model = RepLiteMultiTaskModel(config)
    trainer = Trainer(
        model,
        MultiTaskCriterion(detection_num_classes=2, depth_loss_type="l1"),
        torch.optim.SGD(model.parameters(), lr=1e-3),
        TrainerConfig(epochs=1, amp=True),
        device="cuda",
    )
    targets = {
        "detection": [
            {
                "boxes": torch.tensor([[8.0, 8.0, 40.0, 40.0]]),
                "labels": torch.tensor([1], dtype=torch.long),
                "valid_size": (64, 96),
            }
        ],
        "segmentation": torch.randint(0, 3, (1, 64, 96)),
        "depth": torch.full((1, 1, 64, 96), 2.0),
        "depth_valid": torch.ones(1, 1, 64, 96, dtype=torch.bool),
    }
    result = trainer.train_epoch([(torch.randn(1, 3, 64, 96), targets)], epoch=0)
    assert trainer.amp_enabled
    assert all(math.isfinite(value) for value in result.values())
