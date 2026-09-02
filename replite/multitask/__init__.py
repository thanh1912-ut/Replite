"""Modular lightweight multi-task perception model."""

from .blocks import ConvBNAct, DepthwiseSeparableConv, RepDepthwiseBlock
from .config import RepLiteConfig, TaskConfig
from .heads import (
    ClassificationHead,
    DensePredictionHead,
    DepthHead,
    DetectionHead,
    DetectionOutput,
    ResidualGatedFusion,
    TaskAdapter,
)
from .model import (
    RepLiteMultiTaskModel,
    RepLiteOutput,
    TaskExportWrapper,
    create_replite_model,
    detach_state,
)
from .neck import LiteASPP, NeckOutput, NeckState, RecurrentMultiTaskNeck
from .recurrent import (
    LSTMState,
    LiteConvLSTM,
    LiteConvLSTMCell,
    ResidualLiteConvLSTM,
)

__all__ = [
    "ClassificationHead",
    "ConvBNAct",
    "DensePredictionHead",
    "DepthHead",
    "DepthwiseSeparableConv",
    "DetectionHead",
    "DetectionOutput",
    "LSTMState",
    "LiteConvLSTM",
    "LiteConvLSTMCell",
    "LiteASPP",
    "NeckOutput",
    "NeckState",
    "RepDepthwiseBlock",
    "RepLiteConfig",
    "RepLiteMultiTaskModel",
    "RepLiteOutput",
    "RecurrentMultiTaskNeck",
    "ResidualGatedFusion",
    "ResidualLiteConvLSTM",
    "TaskAdapter",
    "TaskConfig",
    "TaskExportWrapper",
    "create_replite_model",
    "detach_state",
]
