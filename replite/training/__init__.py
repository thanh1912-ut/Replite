"""Training, validation, post-processing, and checkpoint utilities."""

from .checkpoint import (
    CheckpointIntegrityError,
    CheckpointManager,
    ResumeState,
    capture_rng_state,
    load_training_checkpoint,
    restore_rng_state,
    save_training_checkpoint,
)
from .detection import (
    DEFAULT_SIZE_RANGES,
    DEFAULT_STRIDES,
    DetectionCriterion,
    DetectionLosses,
    DetectionPoints,
    FCOSAssignment,
    assign_fcos_targets,
    box_iou,
    class_aware_nms,
    decode_box_regression,
    decode_detections,
    generalized_box_iou_aligned,
    make_detection_points,
    nms,
)
from .logging import TrainingLogger
from .losses import (
    MultiTaskCriterion,
    classification_loss,
    masked_depth_loss,
    segmentation_loss,
)
from .metrics import (
    ClassificationMetrics,
    DepthMetrics,
    DetectionMAP,
    MultiTaskMetrics,
    SegmentationMetrics,
)
from .optim import (
    WarmupCosineScheduler,
    build_adamw_param_groups,
    create_adamw,
)
from .trainer import Trainer, TrainerConfig, move_to_device

__all__ = [
    "DEFAULT_SIZE_RANGES",
    "DEFAULT_STRIDES",
    "CheckpointIntegrityError",
    "CheckpointManager",
    "ClassificationMetrics",
    "DepthMetrics",
    "DetectionCriterion",
    "DetectionLosses",
    "DetectionMAP",
    "DetectionPoints",
    "FCOSAssignment",
    "MultiTaskCriterion",
    "MultiTaskMetrics",
    "ResumeState",
    "SegmentationMetrics",
    "Trainer",
    "TrainerConfig",
    "TrainingLogger",
    "WarmupCosineScheduler",
    "assign_fcos_targets",
    "box_iou",
    "build_adamw_param_groups",
    "capture_rng_state",
    "class_aware_nms",
    "classification_loss",
    "create_adamw",
    "decode_box_regression",
    "decode_detections",
    "generalized_box_iou_aligned",
    "load_training_checkpoint",
    "make_detection_points",
    "masked_depth_loss",
    "move_to_device",
    "nms",
    "restore_rng_state",
    "save_training_checkpoint",
    "segmentation_loss",
]
