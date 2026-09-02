"""Immutable configuration for the modular RepLite multi-task model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


_TASK_ORDER = ("detection", "segmentation", "depth", "classification")
_DENSE_FUSION_DIRECTIONS = (
    "bidirectional",
    "seg_to_depth",
    "depth_to_seg",
)


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _optional_classes(value: object, name: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, name)


@dataclass(frozen=True)
class TaskConfig:
    """Task heads built into one static RepLite model artifact.

    A ``None`` class count removes that head completely. Depth is represented
    by a boolean because its output contract is always one positive channel.
    Segmentation covers semantic, lane, drivable-area, or another dense label
    space; callers choose the class count and loss outside the model.

    Dense fusion defaults to the historical bidirectional graph. Directional
    modes physically keep only one cross-task projection. When
    ``dense_fusion_detach_source`` is true, the destination task can learn from
    a source feature without sending its loss gradient into the source adapter.
    """

    detection_classes: int | None = None
    segmentation_classes: int | None = None
    depth: bool = False
    classification_classes: int | None = None
    gated_dense_fusion: bool = True
    dense_fusion_direction: str = "bidirectional"
    dense_fusion_detach_source: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "detection_classes",
            _optional_classes(self.detection_classes, "detection_classes"),
        )
        object.__setattr__(
            self,
            "segmentation_classes",
            _optional_classes(self.segmentation_classes, "segmentation_classes"),
        )
        object.__setattr__(
            self,
            "classification_classes",
            _optional_classes(self.classification_classes, "classification_classes"),
        )
        if not isinstance(self.depth, bool):
            raise ValueError("depth must be a boolean")
        if not isinstance(self.gated_dense_fusion, bool):
            raise ValueError("gated_dense_fusion must be a boolean")
        if self.dense_fusion_direction not in _DENSE_FUSION_DIRECTIONS:
            raise ValueError(
                "dense_fusion_direction must be 'bidirectional', "
                "'seg_to_depth', or 'depth_to_seg'"
            )
        if not isinstance(self.dense_fusion_detach_source, bool):
            raise ValueError("dense_fusion_detach_source must be a boolean")
        if not self.active_tasks:
            raise ValueError("at least one task head must be enabled")

    @property
    def active_tasks(self) -> tuple[str, ...]:
        """Enabled task names in a stable public order."""

        enabled = {
            "detection": self.detection_classes is not None,
            "segmentation": self.segmentation_classes is not None,
            "depth": self.depth,
            "classification": self.classification_classes is not None,
        }
        return tuple(task for task in _TASK_ORDER if enabled[task])

    @property
    def uses_dense_path(self) -> bool:
        return self.segmentation_classes is not None or self.depth

    @property
    def uses_dense_fusion(self) -> bool:
        return (
            self.gated_dense_fusion
            and self.segmentation_classes is not None
            and self.depth
        )

    def subset(self, tasks: tuple[str, ...] | list[str]) -> "TaskConfig":
        """Return a deployment config containing only selected active tasks."""

        try:
            requested = tuple(tasks)
        except TypeError as exc:
            raise ValueError(
                "tasks must be a non-empty sequence of task names"
            ) from exc
        if not requested or any(not isinstance(task, str) for task in requested):
            raise ValueError("tasks must be a non-empty sequence of task names")
        if len(set(requested)) != len(requested):
            raise ValueError("tasks must not contain duplicates")
        unavailable = sorted(set(requested) - set(self.active_tasks))
        if unavailable:
            raise ValueError(
                "cannot select tasks absent from the source config: "
                + ", ".join(unavailable)
            )
        selected = set(requested)
        return TaskConfig(
            detection_classes=(
                self.detection_classes if "detection" in selected else None
            ),
            segmentation_classes=(
                self.segmentation_classes if "segmentation" in selected else None
            ),
            depth="depth" in selected,
            classification_classes=(
                self.classification_classes if "classification" in selected else None
            ),
            gated_dense_fusion=self.gated_dense_fusion,
            dense_fusion_direction=self.dense_fusion_direction,
            dense_fusion_detach_source=self.dense_fusion_detach_source,
        )


@dataclass(frozen=True)
class RepLiteConfig:
    """Architecture configuration independent of checkpoint location."""

    tasks: TaskConfig
    backbone_name: str = "mobilenetv4_conv_small"
    pretrained: bool = False
    recurrence_steps: int = 3
    recurrent_c4_channels: int = 48
    recurrent_c5_channels: int = 64
    neck_channels: int = 48
    dense_channels: int = 32
    task_adapter_channels: int = 32
    detection_head_channels: int = 48
    detection_head_blocks: int = 2
    detection_reg_max: int = 0
    use_sppf: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.backbone_name, str) or not self.backbone_name.strip():
            raise ValueError("backbone_name must be a non-empty string")
        if not isinstance(self.pretrained, bool):
            raise ValueError("pretrained must be a boolean")
        if not isinstance(self.tasks, TaskConfig):
            raise ValueError("tasks must be a TaskConfig")
        for name in (
            "recurrence_steps",
            "recurrent_c4_channels",
            "recurrent_c5_channels",
            "neck_channels",
            "dense_channels",
            "task_adapter_channels",
            "detection_head_channels",
            "detection_head_blocks",
        ):
            _positive_int(getattr(self, name), name)
        if (
            isinstance(self.detection_reg_max, bool)
            or not isinstance(self.detection_reg_max, int)
            or self.detection_reg_max < 0
        ):
            raise ValueError("detection_reg_max must be a non-negative integer")
        if not isinstance(self.use_sppf, bool):
            raise ValueError("use_sppf must be a boolean")

    @property
    def active_tasks(self) -> tuple[str, ...]:
        return self.tasks.active_tasks

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-serializable architecture metadata for checkpoints."""

        return asdict(self)

    def for_tasks(self, tasks: tuple[str, ...] | list[str]) -> "RepLiteConfig":
        """Return the same architecture with a statically pruned task set."""

        values = asdict(self)
        values["tasks"] = self.tasks.subset(tasks)
        return RepLiteConfig(**values)


__all__ = ["RepLiteConfig", "TaskConfig"]
