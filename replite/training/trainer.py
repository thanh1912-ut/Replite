"""Small, deterministic training and validation engine for RepLite.

The engine stays dataset-agnostic. A loader yields ``(inputs, targets)`` or a
mapping with ``inputs``/``images`` and ``targets``. Five-dimensional clips are
passed through unchanged, so the model predicts targets for the final frame.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
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
        self.amp_enabled = bool(config.amp and self.device.type == "cuda")
        self.amp_dtype = torch.float16 if config.amp_dtype == "float16" else torch.bfloat16
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        self.global_step = 0
        self.start_epoch = 0
        self.amp_skip_count = 0
        self.best_metrics: dict[str, float] = {}

    def _autocast(self):
        if not self.amp_enabled:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.amp_dtype)

    def _divide_gradients(self, divisor: int) -> None:
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(float(divisor))

    def _optimizer_step(self, microbatches: int) -> tuple[bool, float | None]:
        if microbatches <= 0:
            raise RuntimeError("optimizer step requires accumulated gradients")
        if self.amp_enabled:
            self.scaler.unscale_(self.optimizer)
        self._divide_gradients(microbatches)
        grad_norm: float | None = None
        if self.config.grad_clip_norm is not None:
            norm = clip_grad_norm_(self.model.parameters(), float(self.config.grad_clip_norm))
            grad_norm = float(norm.detach().cpu())
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

    def train_epoch(self, loader: Any, *, epoch: int) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        sums: dict[str, float] = {}
        batches = 0
        microbatches = 0
        try:
            total_batches = len(loader)
        except TypeError:
            total_batches = None
        for batch_index, batch in enumerate(loader):
            inputs, targets = _split_batch(batch)
            inputs = move_to_device(inputs, self.device, non_blocking=self.device.type == "cuda")
            targets = move_to_device(targets, self.device, non_blocking=self.device.type == "cuda")
            with self._autocast():
                outputs = self.model(inputs)
                total, losses = _loss_mapping(self.criterion(outputs, targets))
            if not total.requires_grad:
                raise RuntimeError("training loss is disconnected from the model graph")
            self.scaler.scale(total).backward() if self.amp_enabled else total.backward()
            microbatches += 1
            batches += 1
            for name, value in losses.items():
                sums[name] = sums.get(name, 0.0) + float(value.detach().float().cpu())
            is_last = total_batches is not None and batch_index + 1 == total_batches
            boundary = microbatches == self.config.grad_accum_steps or is_last
            if boundary:
                stepped, grad_norm = self._optimizer_step(microbatches)
                microbatches = 0
                if stepped and self.logger is not None and self.global_step % self.config.log_every_n_steps == 0:
                    metrics = {f"loss/{name}": value / batches for name, value in sums.items()}
                    metrics.update(
                        {f"lr/group_{index}": float(group["lr"]) for index, group in enumerate(self.optimizer.param_groups)}
                    )
                    if grad_norm is not None and math.isfinite(grad_norm):
                        metrics["grad_norm"] = grad_norm
                    metrics["amp_skip_count"] = float(self.amp_skip_count)
                    self.logger.log(
                        "train_step",
                        metrics,
                        epoch=epoch,
                        global_step=self.global_step,
                        split="train",
                    )
        if microbatches:
            self._optimizer_step(microbatches)
        if batches == 0:
            raise ValueError("training loader is empty")
        return {name: value / batches for name, value in sums.items()}

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
        try:
            with torch.inference_mode():
                for batch in loader:
                    inputs, targets = _split_batch(batch)
                    inputs = move_to_device(inputs, self.device, non_blocking=self.device.type == "cuda")
                    targets = move_to_device(targets, self.device, non_blocking=self.device.type == "cuda")
                    with self._autocast():
                        outputs = self.model(inputs)
                        _, losses = _loss_mapping(self.criterion(outputs, targets))
                    for name, value in losses.items():
                        sums[name] = sums.get(name, 0.0) + float(value.detach().float().cpu())
                    if self.validation_metrics is not None:
                        self.validation_metrics.update(outputs, targets)
                    batches += 1
        finally:
            self.model.train(was_training)
        if batches == 0:
            raise ValueError("validation loader is empty")
        result: dict[str, Any] = {name: value / batches for name, value in sums.items()}
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
        return result

    def resume(self, path: str | None = None) -> ResumeState:
        kwargs = {
            "model": self.model,
            "optimizer": self.optimizer,
            "trainer_config": self.config,
            "scheduler": self.scheduler,
            "scaler": self.scaler,
            "criterion": self.criterion,
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
        self.amp_skip_count = int(state.extra.get("amp_skip_count", 0))
        if self.logger is not None:
            self.logger.log(
                "resume",
                {"amp_skip_count": self.amp_skip_count},
                epoch=self.start_epoch,
                global_step=self.global_step,
                extra={"checkpoint": str(state.checkpoint_path)},
            )
        return state

    def _is_better(self, value: float) -> bool:
        previous = self.best_metrics.get(self.config.monitor)
        if previous is None:
            return True
        return value < previous if self.config.monitor_mode == "min" else value > previous

    def fit(self, train_loader: Any, val_loader: Any | None = None) -> list[dict[str, Any]]:
        history: list[dict[str, Any]] = []
        for epoch in range(self.start_epoch, self.config.epochs):
            train_result = self.train_epoch(train_loader, epoch=epoch)
            record: dict[str, Any] = {"epoch": epoch, "train": train_result}
            validation_due = val_loader is not None and (
                (epoch + 1) % self.config.validate_every_n_epochs == 0
                or epoch + 1 == self.config.epochs
            )
            val_result = self.validate(val_loader, epoch=epoch) if validation_due else None
            if val_result is not None:
                record["val"] = val_result
            combined = {f"train/{key}": value for key, value in _scalar_metrics(train_result).items()}
            if val_result is not None:
                combined.update({f"val/{key}": value for key, value in _scalar_metrics(val_result).items()})
            improved = False
            if self.config.monitor in combined:
                monitored = float(combined[self.config.monitor])
                improved = self._is_better(monitored)
                if improved:
                    self.best_metrics[self.config.monitor] = monitored
            if self.logger is not None:
                self.logger.log(
                    "epoch_end",
                    combined,
                    epoch=epoch,
                    global_step=self.global_step,
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
                    "extra": {"amp_skip_count": self.amp_skip_count},
                }
                if improved:
                    self.checkpoint_manager.save_named("best.pt", **checkpoint_kwargs)
                if (
                    (epoch + 1) % self.config.checkpoint_every_n_epochs == 0
                    or epoch + 1 == self.config.epochs
                ):
                    self.checkpoint_manager.save_last(**checkpoint_kwargs)
            history.append(record)
        return history


__all__ = ["Trainer", "TrainerConfig", "move_to_device"]
