"""Small, deterministic training and validation engine for RepLite.

The engine stays dataset-agnostic. A loader yields ``(inputs, targets)`` or a
mapping with ``inputs``/``images`` and ``targets``. Five-dimensional clips are
passed through unchanged, so the model predicts targets for the final frame.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer

from .checkpoint import (
    CheckpointManager,
    ResumeState,
    load_training_checkpoint,
)
from .logging import TrainingLogger


def move_to_device(value: Any, device: torch.device, *, non_blocking: bool = False) -> Any:
    """Recursively move tensors while preserving ordinary container shapes."""

    if isinstance(value, Tensor):
        return value.to(device=device, non_blocking=non_blocking)
    if isinstance(value, Mapping):
        return {key: move_to_device(item, device, non_blocking=non_blocking) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device, non_blocking=non_blocking) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device, non_blocking=non_blocking) for item in value]
    return value


@dataclass(frozen=True)
class TrainerConfig:
    """Runtime settings whose exact values are stored in checkpoints."""

    epochs: int = 1
    grad_accum_steps: int = 1
    amp: bool = True
    amp_dtype: str = "float16"
    grad_clip_norm: float | None = 1.0
    log_every_n_steps: int = 20
    validate_every_n_epochs: int = 1
    checkpoint_every_n_epochs: int = 1
    monitor: str = "val/total"
    monitor_mode: str = "min"
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "epochs",
            "grad_accum_steps",
            "log_every_n_steps",
            "validate_every_n_epochs",
            "checkpoint_every_n_epochs",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.amp, bool):
            raise ValueError("amp must be a boolean")
        if self.amp_dtype not in {"float16", "bfloat16"}:
            raise ValueError("amp_dtype must be 'float16' or 'bfloat16'")
        if self.grad_clip_norm is not None and (
            isinstance(self.grad_clip_norm, bool)
            or not isinstance(self.grad_clip_norm, Real)
            or not math.isfinite(float(self.grad_clip_norm))
            or float(self.grad_clip_norm) <= 0.0
        ):
            raise ValueError("grad_clip_norm must be finite and positive or None")
        if not isinstance(self.monitor, str) or not self.monitor:
            raise ValueError("monitor must be a non-empty string")
        if self.monitor_mode not in {"min", "max"}:
            raise ValueError("monitor_mode must be 'min' or 'max'")
        if self.early_stopping_patience is not None and (
            isinstance(self.early_stopping_patience, bool)
            or not isinstance(self.early_stopping_patience, Integral)
            or self.early_stopping_patience <= 0
        ):
            raise ValueError("early_stopping_patience must be a positive integer or None")
        if (
            isinstance(self.early_stopping_min_delta, bool)
            or not isinstance(self.early_stopping_min_delta, Real)
            or not math.isfinite(float(self.early_stopping_min_delta))
            or float(self.early_stopping_min_delta) < 0.0
        ):
            raise ValueError("early_stopping_min_delta must be finite and non-negative")
        if self.early_stopping_patience is not None and not self.monitor.startswith("val/"):
            raise ValueError("early stopping requires a validation monitor beginning with 'val/'")


def _split_batch(batch: Any) -> tuple[Tensor, Any]:
    if isinstance(batch, Mapping):
        key = "inputs" if "inputs" in batch else "images" if "images" in batch else None
        if key is None or "targets" not in batch:
            raise ValueError("batch mapping requires inputs/images and targets")
        inputs, targets = batch[key], batch["targets"]
    elif isinstance(batch, Sequence) and not isinstance(batch, (str, bytes)) and len(batch) == 2:
        inputs, targets = batch
    else:
        raise TypeError("batch must be (inputs, targets) or a mapping")
    if not isinstance(inputs, Tensor) or inputs.ndim not in (4, 5):
        raise ValueError("inputs must have shape B,C,H,W or B,T,C,H,W")
    return inputs, targets


def _loss_mapping(result: Any) -> tuple[Tensor, dict[str, Tensor]]:
    if isinstance(result, Tensor):
        total, values = result, {"total": result}
    elif isinstance(result, Mapping):
        if "total" not in result:
            raise ValueError("criterion mapping must contain 'total'")
        total = result["total"]
        values = {str(key): value for key, value in result.items() if isinstance(value, Tensor) and value.ndim == 0}
    elif hasattr(result, "total"):
        total = result.total
        if hasattr(result, "_asdict"):
            values = {
                str(key): value
                for key, value in result._asdict().items()
                if isinstance(value, Tensor) and value.ndim == 0
            }
        elif isinstance(getattr(result, "losses", None), Mapping):
            values = {
                str(key): value
                for key, value in result.losses.items()
                if isinstance(value, Tensor) and value.ndim == 0
            }
            values.setdefault("total", total)
        else:
            values = {"total": total}
    else:
        raise TypeError("criterion must return a scalar tensor, mapping, or object with .total")
    if not isinstance(total, Tensor) or total.ndim != 0:
        raise TypeError("total loss must be a scalar tensor")
    if not bool(torch.isfinite(total.detach())):
        raise FloatingPointError("non-finite total loss")
    return total, values


def _scalar_metrics(values: Mapping[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, value in values.items():
        if isinstance(value, Tensor) and value.numel() == 1:
            value = value.detach().cpu().item()
        if isinstance(value, bool) or not isinstance(value, Real):
            continue
        number = float(value)
        if math.isfinite(number):
            result[str(name)] = number
    return result


def _loss_scalars(values: Mapping[str, Tensor]) -> dict[str, float]:
    """Snapshot scalar losses with one device-to-host synchronization.

    Loss dictionaries contain several zero-dimensional CUDA tensors.  Calling
    ``.cpu()`` for each entry serializes the stream once per loss component.
    Stacking detached FP32 values preserves the previous scalar/logging
    contract while paying for a single, small host transfer per batch.
    """

    if not values:
        return {}
    names = tuple(values)
    tensors = tuple(values[name].detach().float() for name in names)
    device = tensors[0].device
    if any(value.device != device for value in tensors):
        # Criterion losses are expected to be colocated.  Keep a defensive
        # fallback for custom criteria rather than silently moving tensors.
        return {name: float(value.cpu()) for name, value in zip(names, tensors)}
    numbers = torch.stack(tensors).cpu().tolist()
    return {name: float(number) for name, number in zip(names, numbers)}


class Trainer:
    """Train, validate, log, and checkpoint one RepLite-compatible model."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optimizer,
        config: TrainerConfig,
        *,
        device: torch.device | str | None = None,
        scheduler: object | None = None,
        logger: TrainingLogger | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        validation_metrics: object | None = None,
        checkpoint_extra: Mapping[str, Any] | None = None,
        event_callback: object | None = None,
    ) -> None:
        if not isinstance(model, nn.Module) or not isinstance(criterion, nn.Module):
            raise TypeError("model and criterion must be nn.Module instances")
        if not isinstance(optimizer, Optimizer):
            raise TypeError("optimizer must be a torch optimizer")
        if not isinstance(config, TrainerConfig):
            raise TypeError("config must be TrainerConfig")
        if logger is not None and not isinstance(logger, TrainingLogger):
            raise TypeError("logger must be TrainingLogger or None")
        if checkpoint_manager is not None and not isinstance(checkpoint_manager, CheckpointManager):
            raise TypeError("checkpoint_manager must be CheckpointManager or None")
        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer
        self.config = config
        self.scheduler = scheduler
        self.logger = logger
        self.checkpoint_manager = checkpoint_manager
        self.validation_metrics = validation_metrics
        if checkpoint_extra is not None and not isinstance(checkpoint_extra, Mapping):
            raise TypeError("checkpoint_extra must be a mapping or None")
        self.checkpoint_extra = dict(checkpoint_extra or {})
        if event_callback is not None and not callable(event_callback):
            raise TypeError("event_callback must be callable or None")
        self.event_callback = event_callback
        self.amp_enabled = bool(config.amp and self.device.type == "cuda")
        self.amp_dtype = torch.float16 if config.amp_dtype == "float16" else torch.bfloat16
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        self.global_step = 0
        self.start_epoch = 0
        self.amp_skip_count = 0
        self.best_metrics: dict[str, float] = {}
        self.alias_best_metrics: dict[str, float] = {}
        self.early_stopping_bad_epochs = 0
        self.early_stopping_triggered = False

    def _emit(self, event: str, **payload: Any) -> None:
        if self.event_callback is not None:
            self.event_callback(event, dict(payload))

    @staticmethod
    def _batch_summary(inputs: Tensor, targets: Any) -> dict[str, Any]:
        instances = 0
        ignored_instances = 0
        if isinstance(targets, Mapping):
            detection = targets.get("detection")
            if isinstance(detection, Sequence) and not isinstance(
                detection, (str, bytes, Tensor)
            ):
                for item in detection:
                    if not isinstance(item, Mapping):
                        continue
                    labels = item.get("labels")
                    ignore_boxes = item.get("ignore_boxes")
                    if isinstance(labels, Tensor):
                        instances += int(labels.numel())
                    if isinstance(ignore_boxes, Tensor) and ignore_boxes.ndim >= 1:
                        ignored_instances += int(ignore_boxes.shape[0])
        return {
            "batch_size": int(inputs.shape[0]),
            "clip_length": int(inputs.shape[1]) if inputs.ndim == 5 else 1,
            "image_size": (int(inputs.shape[-2]), int(inputs.shape[-1])),
            "instances": instances,
            "ignored_instances": ignored_instances,
        }

    def _autocast(self):
        if not self.amp_enabled:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.amp_dtype)

    def _divide_gradients(self, divisor: int) -> None:
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(float(divisor))

    def _optimizer_step(self, accumulated_samples: int) -> tuple[bool, Tensor | None]:
        if accumulated_samples <= 0:
            raise RuntimeError("optimizer step requires accumulated samples")
        if self.amp_enabled:
            self.scaler.unscale_(self.optimizer)
        self._divide_gradients(accumulated_samples)
        grad_norm: Tensor | None = None
        if self.config.grad_clip_norm is not None:
            norm = clip_grad_norm_(self.model.parameters(), float(self.config.grad_clip_norm))
            # Keep the scalar on device.  It is materialized only on durable
            # log steps instead of forcing an extra CUDA synchronization after
            # every optimizer update.
            grad_norm = norm.detach()
        if self.amp_enabled:
            old_scale = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            stepped = self.scaler.get_scale() >= old_scale
        else:
            self.optimizer.step()
            stepped = True
        self.optimizer.zero_grad(set_to_none=True)
        if stepped:
            self.global_step += 1
            if self.scheduler is not None:
                step = getattr(self.scheduler, "step", None)
                if not callable(step):
                    raise TypeError("scheduler must provide step()")
                step()
        else:
            self.amp_skip_count += 1
        return stepped, grad_norm

    def _log_amp_overflow_skip(self, *, epoch: int, batch_index: int) -> None:
        if self.logger is None:
            return
        self.logger.log(
            "amp_overflow_skip",
            {
                "amp_skip_count": float(self.amp_skip_count),
                "amp_scale": float(self.scaler.get_scale()),
            },
            epoch=epoch,
            global_step=self.global_step,
            split="train",
            extra={"batch_index": batch_index},
        )

    def train_epoch(self, loader: Any, *, epoch: int) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        # DistributedSampler and BalancedBatchSampler both use this standard
        # hook.  Deriving sampler order from the campaign epoch makes a clean
        # epoch-boundary resume reproduce the uninterrupted ordering.
        seen_epoch_aware: set[int] = set()
        for epoch_aware in (
            getattr(loader, "batch_sampler", None),
            getattr(loader, "sampler", None),
            getattr(loader, "dataset", None),
        ):
            if epoch_aware is None or id(epoch_aware) in seen_epoch_aware:
                continue
            seen_epoch_aware.add(id(epoch_aware))
            set_epoch = getattr(epoch_aware, "set_epoch", None)
            if callable(set_epoch):
                set_epoch(epoch)
        sums: dict[str, float] = {}
        batches = 0
        samples = 0
        microbatches = 0
        accumulated_samples = 0
        try:
            total_batches = len(loader)
        except TypeError:
            total_batches = None
        self._emit(
            "train_epoch_start",
            epoch=epoch,
            total_epochs=self.config.epochs,
            total_batches=total_batches,
        )
        for batch_index, batch in enumerate(loader):
            inputs, targets = _split_batch(batch)
            batch_summary = self._batch_summary(inputs, targets)
            inputs = move_to_device(inputs, self.device, non_blocking=self.device.type == "cuda")
            targets = move_to_device(targets, self.device, non_blocking=self.device.type == "cuda")
            with self._autocast():
                outputs = self.model(inputs)
                total, losses = _loss_mapping(self.criterion(outputs, targets))
            if not total.requires_grad:
                raise RuntimeError("training loss is disconnected from the model graph")
            batch_size = int(inputs.shape[0])
            # Criteria return a mean loss.  Accumulate its sample sum, then
            # divide gradients by the exact number of samples at the optimizer
            # boundary.  Unequal microbatches therefore cannot overweight a
            # short tail.
            sample_sum_loss = total * batch_size
            if self.amp_enabled:
                self.scaler.scale(sample_sum_loss).backward()
            else:
                sample_sum_loss.backward()
            microbatches += 1
            accumulated_samples += batch_size
            batches += 1
            samples += batch_size
            current_losses = _loss_scalars(losses)
            for name, scalar in current_losses.items():
                sums[name] = sums.get(name, 0.0) + scalar * batch_size
            is_last = total_batches is not None and batch_index + 1 == total_batches
            boundary = microbatches == self.config.grad_accum_steps or is_last
            if boundary:
                stepped, grad_norm = self._optimizer_step(accumulated_samples)
                microbatches = 0
                accumulated_samples = 0
                if not stepped:
                    self._log_amp_overflow_skip(epoch=epoch, batch_index=batch_index)
                if stepped and self.logger is not None and self.global_step % self.config.log_every_n_steps == 0:
                    metrics = {f"loss/{name}": value / samples for name, value in sums.items()}
                    metrics.update(
                        {f"lr/group_{index}": float(group["lr"]) for index, group in enumerate(self.optimizer.param_groups)}
                    )
                    if grad_norm is not None:
                        grad_norm_value = float(grad_norm.float().cpu())
                        if math.isfinite(grad_norm_value):
                            metrics["grad_norm"] = grad_norm_value
                    metrics["amp_skip_count"] = float(self.amp_skip_count)
                    if self.amp_enabled:
                        metrics["amp_scale"] = float(self.scaler.get_scale())
                    self.logger.log(
                        "train_step",
                        metrics,
                        epoch=epoch,
                        global_step=self.global_step,
                        split="train",
                    )
            self._emit(
                "train_batch_end",
                epoch=epoch,
                total_epochs=self.config.epochs,
                batch_index=batch_index,
                batches_completed=batches,
                total_batches=total_batches,
                losses=current_losses,
                running_losses={name: value / samples for name, value in sums.items()},
                global_step=self.global_step,
                amp_skip_count=self.amp_skip_count,
                lr=[float(group["lr"]) for group in self.optimizer.param_groups],
                gpu_memory_bytes=(
                    int(torch.cuda.memory_reserved(self.device))
                    if self.device.type == "cuda"
                    else 0
                ),
                **batch_summary,
            )
        if microbatches:
            stepped, _ = self._optimizer_step(accumulated_samples)
            if not stepped:
                self._log_amp_overflow_skip(epoch=epoch, batch_index=batch_index)
        if batches == 0:
            raise ValueError("training loader is empty")
        result = {name: value / samples for name, value in sums.items()}
        self._emit(
            "train_epoch_end",
            epoch=epoch,
            total_epochs=self.config.epochs,
            batches=batches,
            samples=samples,
            result=result,
            global_step=self.global_step,
            amp_skip_count=self.amp_skip_count,
        )
        return result

    def validate(self, loader: Any, *, epoch: int | None = None) -> dict[str, Any]:
        was_training = self.model.training
        self.model.eval()
        if self.validation_metrics is not None:
            reset = getattr(self.validation_metrics, "reset", None)
            if not callable(reset):
                raise TypeError("validation_metrics must provide reset/update/compute")
            reset()
        sums: dict[str, float] = {}
        batches = 0
        samples = 0
        try:
            total_batches = len(loader)
        except TypeError:
            total_batches = None
        self._emit(
            "validation_start",
            epoch=epoch,
            total_epochs=self.config.epochs,
            total_batches=total_batches,
        )
        try:
            with torch.inference_mode():
                for batch in loader:
                    inputs, targets = _split_batch(batch)
                    host_targets = targets
                    inputs = move_to_device(inputs, self.device, non_blocking=self.device.type == "cuda")
                    targets = move_to_device(targets, self.device, non_blocking=self.device.type == "cuda")
                    with self._autocast():
                        outputs = self.model(inputs)
                        _, losses = _loss_mapping(self.criterion(outputs, targets))
                    batch_size = int(inputs.shape[0])
                    for name, scalar in _loss_scalars(losses).items():
                        sums[name] = sums.get(name, 0.0) + scalar * batch_size
                    if self.validation_metrics is not None:
                        metric_targets = targets
                        # Detection metrics retain targets on CPU for the full
                        # validation epoch.  Reuse the loader's host copy rather
                        # than copying every box/label tensor back from CUDA.
                        # Dense targets remain device-resident because their
                        # confusion/error reductions run on the accelerator.
                        if (
                            isinstance(host_targets, Mapping)
                            and isinstance(targets, Mapping)
                            and "detection" in host_targets
                        ):
                            metric_targets = dict(targets)
                            metric_targets["detection"] = host_targets["detection"]
                        self.validation_metrics.update(outputs, metric_targets)
                    batches += 1
                    samples += batch_size
                    self._emit(
                        "validation_batch_end",
                        epoch=epoch,
                        total_epochs=self.config.epochs,
                        batches_completed=batches,
                        total_batches=total_batches,
                    )
        finally:
            self.model.train(was_training)
        if batches == 0:
            raise ValueError("validation loader is empty")
        result: dict[str, Any] = {name: value / samples for name, value in sums.items()}
        if self.validation_metrics is not None:
            computed = self.validation_metrics.compute()
            if not isinstance(computed, Mapping):
                raise TypeError("validation metric compute() must return a mapping")
            for name, value in computed.items():
                key = str(name) if str(name) not in result else f"metric_{name}"
                result[key] = value
        if self.logger is not None:
            self.logger.log(
                "validation",
                _scalar_metrics(result),
                epoch=epoch,
                global_step=self.global_step,
                split="val",
            )
        self._emit(
            "validation_end",
            epoch=epoch,
            total_epochs=self.config.epochs,
            batches=batches,
            result=result,
            global_step=self.global_step,
        )
        return result

    def resume(self, path: str | None = None) -> ResumeState:
        kwargs = {
            "model": self.model,
            "optimizer": self.optimizer,
            "trainer_config": self.config,
            "scheduler": self.scheduler,
            "scaler": self.scaler,
            "criterion": self.criterion,
            "expected_extra": self.checkpoint_extra,
        }
        if path is None:
            if self.checkpoint_manager is None:
                raise ValueError("path or checkpoint_manager is required")
            state = self.checkpoint_manager.load_latest(**kwargs)
        else:
            state = load_training_checkpoint(path, **kwargs)
        self.start_epoch = state.next_epoch
        self.global_step = state.global_step
        self.best_metrics = dict(state.best_metrics)
        raw_alias_metrics = state.extra.get("alias_best_metrics", {})
        if not isinstance(raw_alias_metrics, Mapping):
            raise ValueError("checkpoint alias_best_metrics must be a mapping")
        self.alias_best_metrics = {
            str(name): float(value)
            for name, value in raw_alias_metrics.items()
        }
        if any(
            not math.isfinite(value)
            for value in self.alias_best_metrics.values()
        ):
            raise ValueError("checkpoint alias_best_metrics must be finite")
        self.amp_skip_count = int(state.extra.get("amp_skip_count", 0))
        self.early_stopping_bad_epochs = int(
            state.extra.get("early_stopping_bad_epochs", 0)
        )
        self.early_stopping_triggered = bool(
            state.extra.get("early_stopping_triggered", False)
        )
        if self.logger is not None:
            resume_metrics = {
                "amp_skip_count": self.amp_skip_count,
                "early_stopping_bad_epochs": self.early_stopping_bad_epochs,
            }
            if self.amp_enabled:
                resume_metrics["amp_scale"] = float(self.scaler.get_scale())
            self.logger.log(
                "resume",
                resume_metrics,
                epoch=self.start_epoch,
                global_step=self.global_step,
                extra={"checkpoint": str(state.checkpoint_path)},
            )
        return state

    def _is_better(self, value: float) -> bool:
        previous = self.best_metrics.get(self.config.monitor)
        if previous is None:
            return True
        delta = float(self.config.early_stopping_min_delta)
        if self.config.monitor_mode == "min":
            return value < previous - delta
        return value > previous + delta

    def fit(
        self,
        train_loader: Any,
        val_loader: Any | None = None,
        *,
        stop_after_epoch: int | None = None,
    ) -> list[dict[str, Any]]:
        """Fit through an epoch boundary without changing campaign semantics.

        ``stop_after_epoch`` is the one-based number of completed epochs at
        which to pause.  The stored :class:`TrainerConfig`, optimizer schedule,
        and checkpoint hash still describe the complete campaign.  This makes
        a gated first epoch resumable as epoch two instead of creating a
        different one-epoch experiment.  A pause boundary always triggers
        validation and a ``last.pt`` checkpoint when those objects are
        configured.
        """

        if stop_after_epoch is None:
            end_epoch = self.config.epochs
        else:
            if (
                isinstance(stop_after_epoch, bool)
                or not isinstance(stop_after_epoch, Integral)
                or not 1 <= int(stop_after_epoch) <= self.config.epochs
            ):
                raise ValueError(
                    "stop_after_epoch must be in [1, config.epochs] or None"
                )
            end_epoch = int(stop_after_epoch)
        if end_epoch < self.start_epoch:
            raise ValueError(
                "stop_after_epoch precedes the next epoch in the resumed state"
            )
        if self.config.early_stopping_patience is not None and val_loader is None:
            raise ValueError("early stopping requires a validation loader")
        if self.early_stopping_triggered:
            # A checkpoint written at the stopping boundary is terminal for
            # this exact campaign.  Starting another epoch after resume would
            # violate early-stopping parity with the uninterrupted run.
            return []
        history: list[dict[str, Any]] = []
        for epoch in range(self.start_epoch, end_epoch):
            train_result = self.train_epoch(train_loader, epoch=epoch)
            record: dict[str, Any] = {"epoch": epoch, "train": train_result}
            validation_due = val_loader is not None and (
                (epoch + 1) % self.config.validate_every_n_epochs == 0
                or epoch + 1 == end_epoch
            )
            val_result = self.validate(val_loader, epoch=epoch) if validation_due else None
            if val_result is not None:
                record["val"] = val_result
            combined = {f"train/{key}": value for key, value in _scalar_metrics(train_result).items()}
            if val_result is not None:
                combined.update({f"val/{key}": value for key, value in _scalar_metrics(val_result).items()})
            improved = False
            if val_result is not None and self.config.monitor not in combined:
                raise KeyError(
                    f"configured monitor {self.config.monitor!r} is absent from "
                    "the epoch metrics"
                )
            if self.config.monitor in combined:
                monitored = float(combined[self.config.monitor])
                improved = self._is_better(monitored)
                if improved:
                    self.best_metrics[self.config.monitor] = monitored
                    self.early_stopping_bad_epochs = 0
                elif (
                    val_result is not None
                    and self.config.early_stopping_patience is not None
                ):
                    self.early_stopping_bad_epochs += 1
            should_stop = bool(
                self.config.early_stopping_patience is not None
                and self.early_stopping_bad_epochs
                >= self.config.early_stopping_patience
            )
            if should_stop:
                self.early_stopping_triggered = True
                record["early_stopping"] = {
                    "triggered": True,
                    "bad_epochs": self.early_stopping_bad_epochs,
                    "patience": self.config.early_stopping_patience,
                    "monitor": self.config.monitor,
                    "best": self.best_metrics.get(self.config.monitor),
                }
            if self.logger is not None:
                self.logger.log(
                    "epoch_end",
                    combined,
                    epoch=epoch,
                    global_step=self.global_step,
                )
                if should_stop:
                    self.logger.log(
                        "early_stopping",
                        {
                            "bad_epochs": float(self.early_stopping_bad_epochs),
                            "patience": float(self.config.early_stopping_patience),
                            "best": float(self.best_metrics[self.config.monitor]),
                            "current": float(combined[self.config.monitor]),
                        },
                        epoch=epoch,
                        global_step=self.global_step,
                        split="val",
                        extra={"monitor": self.config.monitor},
                    )
            if self.checkpoint_manager is not None:
                checkpoint_kwargs = {
                    "model": self.model,
                    "optimizer": self.optimizer,
                    "epoch_completed": epoch,
                    "global_step": self.global_step,
                    "trainer_config": self.config,
                    "scheduler": self.scheduler,
                    "scaler": self.scaler,
                    "criterion": self.criterion,
                    "best_metrics": self.best_metrics,
                    "extra": {
                        **self.checkpoint_extra,
                        "amp_skip_count": self.amp_skip_count,
                        "amp_scale": float(self.scaler.get_scale()),
                        "early_stopping_bad_epochs": self.early_stopping_bad_epochs,
                        "early_stopping_triggered": self.early_stopping_triggered,
                        "alias_best_metrics": dict(self.alias_best_metrics),
                    },
                }
                if improved:
                    self.checkpoint_manager.save_named("best.pt", **checkpoint_kwargs)
                # Keep task-specific aliases alongside the campaign monitor.
                # This makes the selected segmentation/depth checkpoints
                # auditable without changing which metric controls stopping.
                aliases = (
                    ("best_miou.pt", "val/segmentation/miou", "max"),
                    ("best_absrel.pt", "val/depth/abs_rel", "min"),
                    ("best_joint.pt", "val/selection/joint", "max"),
                )
                for alias, metric_name, mode in aliases:
                    if metric_name not in combined:
                        continue
                    current = float(combined[metric_name])
                    previous = self.alias_best_metrics.get(metric_name)
                    # Aliases mean literal per-task best, independent of the
                    # campaign monitor's early-stopping min_delta.
                    alias_improved = previous is None or (
                        current < previous
                        if mode == "min"
                        else current > previous
                    )
                    if alias_improved:
                        self.alias_best_metrics[metric_name] = current
                        checkpoint_kwargs["extra"]["alias_best_metrics"] = dict(
                            self.alias_best_metrics
                        )
                        self.checkpoint_manager.save_named(alias, **checkpoint_kwargs)
                if (
                    (epoch + 1) % self.config.checkpoint_every_n_epochs == 0
                    or epoch + 1 == end_epoch
                    or should_stop
                ):
                    self.checkpoint_manager.save_last(**checkpoint_kwargs)
            history.append(record)
            self.start_epoch = epoch + 1
            if should_stop:
                self._emit(
                    "early_stopping",
                    epoch=epoch,
                    total_epochs=self.config.epochs,
                    monitor=self.config.monitor,
                    best=self.best_metrics.get(self.config.monitor),
                    bad_epochs=self.early_stopping_bad_epochs,
                    patience=self.config.early_stopping_patience,
                    global_step=self.global_step,
                )
                break
        return history


__all__ = ["Trainer", "TrainerConfig", "move_to_device"]
