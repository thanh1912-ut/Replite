"""Optimizer and learning-rate scheduling utilities for RepLite training.

The helpers in this module deliberately operate on ordinary ``nn.Module`` and
``Optimizer`` instances.  They therefore remain useful for task-specific
wrappers and small test models in addition to the complete RepLite network.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

from torch import nn
from torch.optim import AdamW, Optimizer


def _positive_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive finite number")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return value


def _non_negative_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite non-negative number")
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return value


def _canonical_parameter_name(name: str) -> str:
    """Remove a single DDP prefix without changing nested module names."""

    return name[7:] if name.startswith("module.") else name


def build_adamw_param_groups(
    model: nn.Module,
    *,
    lr: float,
    weight_decay: float,
    backbone_lr_multiplier: float = 0.1,
) -> list[dict[str, Any]]:
    """Build disjoint AdamW groups for backbone/main and decay/no-decay.

    Biases, scalar parameters, and one-dimensional parameters (including
    normalization scales) receive no weight decay.  Parameters whose canonical
    name starts with ``backbone.`` receive ``backbone_lr_multiplier`` times the
    base learning rate.  Frozen parameters are omitted.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("model must be an nn.Module")
    base_lr = _positive_float(lr, "lr")
    decay = _non_negative_float(weight_decay, "weight_decay")
    backbone_multiplier = _positive_float(
        backbone_lr_multiplier,
        "backbone_lr_multiplier",
    )

    grouped: dict[tuple[str, bool], list[nn.Parameter]] = {
        ("backbone", True): [],
        ("backbone", False): [],
        ("main", True): [],
        ("main", False): [],
    }
    seen: set[int] = set()
    trainable_count = 0
    for raw_name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        trainable_count += 1
        identifier = id(parameter)
        if identifier in seen:
            # Tied parameters should appear exactly once in the optimizer.
            continue
        seen.add(identifier)
        name = _canonical_parameter_name(raw_name)
        section = "backbone" if name.startswith("backbone.") else "main"
        use_decay = parameter.ndim > 1 and not name.endswith(".bias")
        grouped[(section, use_decay)].append(parameter)

    if not seen:
        raise ValueError("model has no trainable parameters")

    parameter_groups: list[dict[str, Any]] = []
    for section, use_decay in (
        ("backbone", True),
        ("backbone", False),
        ("main", True),
        ("main", False),
    ):
        parameters = grouped[(section, use_decay)]
        if not parameters:
            continue
        lr_scale = backbone_multiplier if section == "backbone" else 1.0
        group_lr = base_lr * lr_scale
        parameter_groups.append(
            {
                "name": f"{section}_{'decay' if use_decay else 'no_decay'}",
                "params": parameters,
                "lr": group_lr,
                "initial_lr": group_lr,
                "lr_scale": lr_scale,
                "weight_decay": decay if use_decay else 0.0,
            }
        )

    grouped_ids = {
        id(parameter)
        for group in parameter_groups
        for parameter in group["params"]
    }
    if grouped_ids != seen:
        raise RuntimeError("optimizer parameter grouping is incomplete")
    if len(grouped_ids) > trainable_count:
        raise RuntimeError("optimizer parameter grouping contains duplicates")
    return parameter_groups


def create_adamw(
    model: nn.Module,
    *,
    lr: float,
    weight_decay: float = 1e-2,
    backbone_lr_multiplier: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> AdamW:
    """Create AdamW with RepLite-safe parameter grouping."""

    if (
        not isinstance(betas, tuple)
        or len(betas) != 2
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0.0 <= float(value) < 1.0
            for value in betas
        )
    ):
        raise ValueError("betas must contain two values in [0, 1)")
    epsilon = _positive_float(eps, "eps")
    groups = build_adamw_param_groups(
        model,
        lr=lr,
        weight_decay=weight_decay,
        backbone_lr_multiplier=backbone_lr_multiplier,
    )
    return AdamW(
        groups,
        lr=float(lr),
        betas=(float(betas[0]), float(betas[1])),
        eps=epsilon,
    )


class WarmupCosineScheduler:
    """Step-based linear-warmup then cosine-decay scheduler.

    The learning rate stored in the optimizer always applies to the *next*
    optimizer update.  Construction configures update zero.  Call ``step``
    exactly once after each successful optimizer update; AMP-overflow skips
    must not advance this scheduler.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        total_steps: int,
        warmup_steps: int = 0,
        min_lr_ratio: float = 0.0,
    ) -> None:
        if not isinstance(optimizer, Optimizer):
            raise TypeError("optimizer must be a torch Optimizer")
        if isinstance(total_steps, bool) or not isinstance(total_steps, int):
            raise ValueError("total_steps must be a positive integer")
        if total_steps <= 0:
            raise ValueError("total_steps must be a positive integer")
        if isinstance(warmup_steps, bool) or not isinstance(warmup_steps, int):
            raise ValueError("warmup_steps must be a non-negative integer")
        if warmup_steps < 0 or warmup_steps > total_steps:
            raise ValueError("warmup_steps must be in [0, total_steps]")
        ratio = _non_negative_float(min_lr_ratio, "min_lr_ratio")
        if ratio > 1.0:
            raise ValueError("min_lr_ratio must be at most 1")

        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr_ratio = ratio
        self.base_lrs = [
            float(group.get("initial_lr", group["lr"]))
            for group in optimizer.param_groups
        ]
        self.step_count = 0
        self._apply(self.step_count)

    def _scale(self, step: int) -> float:
        if self.warmup_steps and step < self.warmup_steps:
            return float(step + 1) / float(self.warmup_steps)
        if self.total_steps <= self.warmup_steps:
            return self.min_lr_ratio
        decay_positions = self.total_steps - self.warmup_steps
        denominator = max(decay_positions - 1, 1)
        progress = min(max(step - self.warmup_steps, 0), denominator) / denominator
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

    def _apply(self, step: int) -> None:
        scale = self._scale(step)
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * scale

    def step(self) -> None:
        """Advance to the learning rate for the next optimizer update."""

        self.step_count += 1
        self._apply(self.step_count)

    def get_last_lr(self) -> list[float]:
        """Return current group learning rates."""

        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def state_dict(self) -> dict[str, Any]:
        """Return a fully self-validating scheduler state."""

        return {
            "schema_version": 1,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr_ratio": self.min_lr_ratio,
            "base_lrs": list(self.base_lrs),
            "step_count": self.step_count,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore state, rejecting schedules with different semantics."""

        expected = {
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr_ratio": self.min_lr_ratio,
        }
        for name, value in expected.items():
            if state_dict.get(name) != value:
                raise ValueError(
                    f"scheduler {name} mismatch: "
                    f"checkpoint={state_dict.get(name)!r}, current={value!r}"
                )
        base_lrs = [float(value) for value in state_dict.get("base_lrs", ())]
        if len(base_lrs) != len(self.optimizer.param_groups):
            raise ValueError("scheduler optimizer group count mismatch")
        if base_lrs != self.base_lrs:
            raise ValueError("scheduler base learning rates mismatch")
        step_count = state_dict.get("step_count")
        if isinstance(step_count, bool) or not isinstance(step_count, int):
            raise ValueError("invalid scheduler step_count")
        if step_count < 0:
            raise ValueError("invalid scheduler step_count")
        self.step_count = step_count
        self._apply(self.step_count)


__all__ = [
    "WarmupCosineScheduler",
    "build_adamw_param_groups",
    "create_adamw",
]
