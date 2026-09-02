"""Complete modular RepLite multi-task perception model."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import NamedTuple

import torch
from torch import Tensor, nn

from replite.backbone import create_backbone

from .blocks import RepDepthwiseBlock
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
from .neck import NeckOutput, NeckState, RecurrentMultiTaskNeck


class RepLiteOutput(NamedTuple):
    """Predictions from statically enabled heads.

    Disabled task entries are ``None``. Detection predictions are raw
    anchor-free pyramid outputs; decoding, assignment, losses and NMS belong to
    the training/deployment application because their contracts are
    dataset-specific.
    """

    detection: DetectionOutput | None
    segmentation: Tensor | None
    depth: Tensor | None
    classification: Tensor | None


def detach_state(state: NeckState) -> NeckState:
    """Detach an explicit recurrent state at a truncated-BPTT boundary."""

    if not isinstance(state, NeckState):
        raise TypeError("state must be a NeckState")
    level4 = (
        None
        if state.level4 is None
        else (state.level4[0].detach(), state.level4[1].detach())
    )
    level5 = (
        None
        if state.level5 is None
        else (state.level5[0].detach(), state.level5[1].detach())
    )
    return NeckState(level4=level4, level5=level5)


class RepLiteMultiTaskModel(nn.Module):
    """Backbone, always-on LiteConvLSTM neck, and prunable task heads.

    ``forward`` uses iterative recurrent refinement for a 4D image and final
    frame prediction for a 5D ``B,T,C,H,W`` clip. Streaming applications should
    call :meth:`forward_step` and carry the returned :class:`NeckState` across
    frames. Static-image refinement and sequence processing share all weights.
    """

    def __init__(
        self,
        config: RepLiteConfig,
        *,
        checkpoint_path: str | None = None,
        cache_dir: str | None = None,
        local_files_only: bool = False,
        force_download: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(config, RepLiteConfig):
            raise TypeError("config must be a RepLiteConfig")
        self.config = config

        needs_level4 = (
            config.tasks.detection_classes is not None or config.tasks.uses_dense_path
        )
        needs_level5 = (
            config.tasks.detection_classes is not None
            or config.tasks.classification_classes is not None
        )
        # A dense-only artifact ends at C4.  BackboneBase trims every deeper
        # block group, so C5 is absent from parameters, state dict and compute.
        backbone_out_indices = (0, 1, 2, 3) if needs_level5 else (0, 1, 2)
        self.backbone = create_backbone(
            config.backbone_name,
            pretrained=config.pretrained,
            out_indices=backbone_out_indices,
            checkpoint_path=checkpoint_path,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            force_download=force_download,
        )
        backbone_channels = tuple(self.backbone.feature_info.channels())
        self.neck = RecurrentMultiTaskNeck(
            in_channels=backbone_channels,
            recurrent_channels=(
                config.recurrent_c4_channels,
                config.recurrent_c5_channels,
            ),
            detection_channels=config.neck_channels,
            dense_channels=config.dense_channels,
            refine_steps=config.recurrence_steps,
            use_sppf=config.use_sppf,
            enable_detection=config.tasks.detection_classes is not None,
            enable_dense=config.tasks.uses_dense_path,
            enable_level4=needs_level4,
            enable_level5=needs_level5,
        )

        if config.tasks.detection_classes is not None:
            self.detection_head = DetectionHead(
                in_channels=config.neck_channels,
                num_classes=config.tasks.detection_classes,
                head_channels=config.detection_head_channels,
                num_convs=config.detection_head_blocks,
                reg_max=config.detection_reg_max,
            )

        if config.tasks.segmentation_classes is not None:
            self.segmentation_adapter = TaskAdapter(
                config.dense_channels,
                config.task_adapter_channels,
            )
            self.segmentation_head = DensePredictionHead(
                config.task_adapter_channels,
                config.tasks.segmentation_classes,
            )

        if config.tasks.depth:
            self.depth_adapter = TaskAdapter(
                config.dense_channels,
                config.task_adapter_channels,
            )
            self.depth_head = DepthHead(config.task_adapter_channels)

        if config.tasks.uses_dense_fusion:
            self.dense_fusion = ResidualGatedFusion(
                config.task_adapter_channels,
                config.task_adapter_channels,
                direction=config.tasks.dense_fusion_direction,
                detach_source=config.tasks.dense_fusion_detach_source,
            )

        if config.tasks.classification_classes is not None:
            self.classification_head = ClassificationHead(
                config.recurrent_c5_channels,
                config.tasks.classification_classes,
            )

    @property
    def active_tasks(self) -> tuple[str, ...]:
        return self.config.active_tasks

    @property
    def model_metadata(self) -> dict[str, object]:
        """JSON-serializable architecture and initialization provenance."""

        return {
            "schema_version": 1,
            "model": "replite_multitask",
            "config": self.config.as_dict(),
            "backbone": self.backbone.backbone_config,
        }

    @staticmethod
    def _validate_image(images: Tensor) -> None:
        if not isinstance(images, Tensor):
            raise TypeError("images must be a torch.Tensor")
        if images.ndim != 4:
            raise ValueError("images must have shape B,3,H,W")
        if torch.jit.is_tracing():
            return
        if images.shape[0] <= 0 or images.shape[2] <= 0 or images.shape[3] <= 0:
            raise ValueError("image batch and spatial dimensions must be non-zero")
        if images.shape[1] != 3:
            raise ValueError(f"images must have 3 RGB channels, got {images.shape[1]}")
        if not images.is_floating_point():
            raise TypeError("images must use a floating-point dtype")

    @classmethod
    def _validate_clip(cls, frames: Tensor) -> None:
        if not isinstance(frames, Tensor):
            raise TypeError("frames must be a torch.Tensor")
        if frames.ndim != 5:
            raise ValueError("frames must have shape B,T,3,H,W")
        if frames.shape[1] <= 0:
            raise ValueError("frame sequence length must be non-zero")
        cls._validate_image(frames[:, 0])

    def _predict(
        self,
        neck_output: NeckOutput,
        output_size: tuple[int, int],
        requested_tasks: tuple[str, ...] | None = None,
    ) -> RepLiteOutput:
        if requested_tasks is None:
            requested_tasks = self.active_tasks
        requested = set(requested_tasks)
        unavailable = requested - set(self.active_tasks)
        if unavailable:
            raise ValueError(
                "requested tasks are not active: " + ", ".join(sorted(unavailable))
            )

        detection: DetectionOutput | None = None
        segmentation: Tensor | None = None
        depth: Tensor | None = None
        classification: Tensor | None = None

        if "detection" in requested:
            pyramid = neck_output.detection
            if pyramid is None:
                raise RuntimeError("detection head is enabled but neck path is absent")
            detection = self.detection_head(pyramid)

        dense_feature = neck_output.f2
        segmentation_feature: Tensor | None = None
        depth_feature: Tensor | None = None
        wants_segmentation = "segmentation" in requested
        wants_depth = "depth" in requested
        needs_depth_for_segmentation = (
            wants_segmentation
            and hasattr(self, "dense_fusion")
            and self.dense_fusion.enabled
            and self.dense_fusion.uses_depth_to_seg
        )
        needs_segmentation_for_depth = (
            wants_depth
            and hasattr(self, "dense_fusion")
            and self.dense_fusion.enabled
            and self.dense_fusion.uses_seg_to_depth
        )
        needs_fusion_pair = needs_depth_for_segmentation or needs_segmentation_for_depth
        if wants_segmentation or needs_segmentation_for_depth:
            if dense_feature is None:
                raise RuntimeError(
                    "segmentation head is enabled but dense path is absent"
                )
            segmentation_feature = self.segmentation_adapter(dense_feature)
        if wants_depth or needs_depth_for_segmentation:
            if dense_feature is None:
                raise RuntimeError("depth head is enabled but dense path is absent")
            depth_feature = self.depth_adapter(dense_feature)

        if needs_fusion_pair:
            if segmentation_feature is None or depth_feature is None:
                raise RuntimeError(
                    "dense fusion requires segmentation and depth features"
                )
            if wants_segmentation and wants_depth:
                segmentation_feature, depth_feature = self.dense_fusion(
                    segmentation_feature,
                    depth_feature,
                )
            elif wants_segmentation:
                segmentation_feature = self.dense_fusion.forward_segmentation(
                    segmentation_feature,
                    depth_feature,
                )
            else:
                depth_feature = self.dense_fusion.forward_depth(
                    segmentation_feature,
                    depth_feature,
                )

        if wants_segmentation:
            if segmentation_feature is None:
                raise RuntimeError("segmentation feature is unexpectedly absent")
            segmentation = self.segmentation_head(
                segmentation_feature,
                output_size,
            )
        if wants_depth:
            if depth_feature is None:
                raise RuntimeError("depth feature is unexpectedly absent")
            depth = self.depth_head(depth_feature, output_size)
        if "classification" in requested:
            if neck_output.r5 is None:
                raise RuntimeError(
                    "classification head is enabled but recurrent level 5 is absent"
                )
            classification = self.classification_head(neck_output.r5)

        return RepLiteOutput(
            detection=detection,
            segmentation=segmentation,
            depth=depth,
            classification=classification,
        )

    def _forward_static_task(self, images: Tensor, task: str) -> RepLiteOutput:
        """Execute only the dependencies of one active static-image task."""

        if task not in self.active_tasks:
            raise ValueError(f"task must be active; got {task!r}")
        self._validate_image(images)
        features = self.backbone(images)
        is_detection = task == "detection"
        is_dense = task in ("segmentation", "depth")
        is_classification = task == "classification"
        neck_output, _ = self.neck.refine(
            features,
            decode_detection=is_detection,
            decode_dense=is_dense,
            include_level4=is_detection or is_dense,
            include_level5=is_detection or is_classification,
        )
        return self._predict(
            neck_output,
            tuple(images.shape[-2:]),
            requested_tasks=(task,),
        )

    def forward_static(
        self,
        images: Tensor,
        *,
        steps: int | None = None,
    ) -> RepLiteOutput:
        """Refine one image repeatedly; the backbone executes exactly once."""

        self._validate_image(images)
        features = self.backbone(images)
        neck_output, _ = self.neck.refine(features, steps=steps)
        return self._predict(neck_output, tuple(images.shape[-2:]))

    def forward_step(
        self,
        image: Tensor,
        state: NeckState | None = None,
    ) -> tuple[RepLiteOutput, NeckState]:
        """Process one frame and return explicit state for the next frame."""

        self._validate_image(image)
        features = self.backbone(image)
        neck_output, next_state = self.neck.step(features, state=state)
        return self._predict(neck_output, tuple(image.shape[-2:])), next_state

    def forward_sequence(
        self,
        frames: Tensor,
        state: NeckState | None = None,
    ) -> tuple[RepLiteOutput, NeckState]:
        """Aggregate a clip and return predictions for its final frame.

        The backbone sees the flattened ``B*T`` batch once. Recurrent state is
        advanced in time order, while task predictions are returned only for
        the final frame, matching a ``t-2,t-1,t -> output_t`` training setup.
        """

        self._validate_clip(frames)
        batch_size, sequence_length, channels, height, width = frames.shape
        flattened = frames.reshape(
            batch_size * sequence_length,
            channels,
            height,
            width,
        )
        flat_features = self.backbone(flattened)
        sequences = tuple(
            feature.reshape(
                batch_size,
                sequence_length,
                feature.shape[1],
                feature.shape[2],
                feature.shape[3],
            )
            for feature in flat_features
        )

        next_state = state
        for index in range(sequence_length - 1):
            step_features = tuple(feature[:, index] for feature in sequences)
            next_state = self.neck.advance(step_features, state=next_state)
        final_features = tuple(feature[:, -1] for feature in sequences)
        neck_output, next_state = self.neck.step(
            final_features,
            state=next_state,
        )
        return self._predict(neck_output, (height, width)), next_state

    def forward(self, inputs: Tensor) -> RepLiteOutput:
        """Dispatch a 4D image to refinement or a 5D clip to sequence mode."""

        if not isinstance(inputs, Tensor):
            raise TypeError("inputs must be a torch.Tensor")
        if inputs.ndim == 4:
            return self.forward_static(inputs)
        if inputs.ndim == 5:
            output, _ = self.forward_sequence(inputs)
            return output
        raise ValueError("inputs must have shape B,3,H,W or B,T,3,H,W")

    def export_task(self, task: str) -> "TaskExportWrapper":
        """Create a tensor-only wrapper for one active task."""

        return TaskExportWrapper(self, task)

    def switch_to_deploy(self) -> "RepLiteMultiTaskModel":
        """Fuse every RepDepthwiseBlock after the model is put in eval mode."""

        if self.training:
            raise RuntimeError("call eval() before switch_to_deploy()")
        for module in tuple(self.modules()):
            if isinstance(module, RepDepthwiseBlock):
                module.switch_to_deploy()
        return self


class TaskExportWrapper(nn.Module):
    """Independent task-pruned static-image snapshot for export runtimes."""

    def __init__(self, model: RepLiteMultiTaskModel, task: str) -> None:
        super().__init__()
        if not isinstance(model, RepLiteMultiTaskModel):
            raise TypeError("model must be a RepLiteMultiTaskModel")
        if not isinstance(task, str) or task not in model.active_tasks:
            raise ValueError(
                f"task must be active; got {task!r}, active={model.active_tasks}"
            )
        self.task = task
        self._task_model = self._build_pruned_model(model, task)
        self._export_config = model.config.for_tasks([task]).as_dict()
        self._source_metadata = deepcopy(model.model_metadata)
        self._uses_fusion_dependency = (
            hasattr(model, "dense_fusion")
            and model.dense_fusion.enabled
            and (
                (task == "segmentation" and model.dense_fusion.uses_depth_to_seg)
                or (task == "depth" and model.dense_fusion.uses_seg_to_depth)
            )
        )
        # A newly-created nn.Module defaults to training mode. Mirror the
        # wrapped model immediately so ONNX/export utilities cannot restore the
        # wrapper to ``train`` and recursively flip an eval model's BatchNorms.
        self.train(model.training)

    @property
    def active_tasks(self) -> tuple[str, ...]:
        """The single tensor contract exposed by this wrapper."""

        return (self.task,)

    @property
    def model_metadata(self) -> dict[str, object]:
        """JSON-serializable export config and source-weight provenance."""

        return {
            "schema_version": 1,
            "model": "replite_task_export",
            "task": self.task,
            "config": deepcopy(self._export_config),
            "source": deepcopy(self._source_metadata),
            "dependencies": {
                "dense_fusion": self._uses_fusion_dependency,
                "weight_semantics": "snapshot_at_wrapper_creation",
            },
        }

    @staticmethod
    def _build_pruned_model(
        source: RepLiteMultiTaskModel,
        task: str,
    ) -> RepLiteMultiTaskModel:
        """Copy current weights into the smallest semantics-preserving model."""

        source_tasks = source.config.tasks
        uses_fusion_dependency = (
            hasattr(source, "dense_fusion")
            and source.dense_fusion.enabled
            and (
                (
                    task == "segmentation"
                    and source.dense_fusion.uses_depth_to_seg
                )
                or (task == "depth" and source.dense_fusion.uses_seg_to_depth)
            )
        )
        if uses_fusion_dependency:
            # Both adapters and the required directional fusion branch are
            # part of the exported dense task's semantics. The unused final
            # predictor is deleted after loading.
            target_tasks = TaskConfig(
                segmentation_classes=source_tasks.segmentation_classes,
                depth=True,
                gated_dense_fusion=True,
                dense_fusion_direction=source_tasks.dense_fusion_direction,
                dense_fusion_detach_source=source_tasks.dense_fusion_detach_source,
            )
        else:
            target_tasks = source_tasks.subset([task])
        target_config = replace(
            source.config,
            tasks=target_tasks,
            pretrained=False,
        )
        source_parameter = next(source.parameters())
        rng_devices: list[int] = []
        if source_parameter.is_cuda:
            rng_devices.append(
                torch.cuda.current_device()
                if source_parameter.device.index is None
                else source_parameter.device.index
            )
        # Constructing the independent snapshot initializes temporary random
        # weights before the strict copy. Preserve the caller's RNG streams so
        # exporting during an experiment cannot perturb later sampling.
        with torch.random.fork_rng(devices=rng_devices):
            target = RepLiteMultiTaskModel(target_config)

        source_rep_blocks = [
            module
            for module in source.modules()
            if isinstance(module, RepDepthwiseBlock)
        ]
        if source_rep_blocks and any(block.deploy for block in source_rep_blocks):
            if not all(block.deploy for block in source_rep_blocks):
                raise RuntimeError(
                    "cannot export a partially deployed RepDepthwiseBlock model"
                )
            target.eval().switch_to_deploy()

        target.to(device=source_parameter.device, dtype=source_parameter.dtype)
        if any(
            parameter.ndim == 4
            and parameter.is_contiguous(memory_format=torch.channels_last)
            and not parameter.is_contiguous()
            for parameter in source.parameters()
        ):
            target.to(memory_format=torch.channels_last)
        source_state = source.state_dict()
        target_state = target.state_dict()
        missing = sorted(set(target_state) - set(source_state))
        if missing:
            raise RuntimeError(
                "source model cannot initialize pruned export: " + ", ".join(missing)
            )
        target.load_state_dict(
            {name: source_state[name] for name in target_state},
            strict=True,
        )

        source_parameters = dict(source.named_parameters())
        for name, parameter in target.named_parameters():
            if name in source_parameters:
                parameter.requires_grad_(source_parameters[name].requires_grad)

        if uses_fusion_dependency:
            target.dense_fusion.enabled = source.dense_fusion.enabled
            if task == "segmentation":
                del target.depth_head
                if target.dense_fusion.uses_seg_to_depth:
                    del target.dense_fusion.seg_to_depth
                    del target.dense_fusion.seg_to_depth_scale
            else:
                del target.segmentation_head
                if target.dense_fusion.uses_depth_to_seg:
                    del target.dense_fusion.depth_to_seg
                    del target.dense_fusion.depth_to_seg_scale
        # The private execution model exposes only the wrapper task. Extra
        # directional fusion modules above are retained dependencies, not an
        # additional output contract.
        target.config = source.config.for_tasks([task])
        target.backbone._weights_loaded = source.backbone.weights_loaded
        target.backbone._weights_source = source.backbone.weights_source
        target.backbone._checkpoint_path = source.backbone._checkpoint_path
        target.train(source.training)
        return target

    def forward(self, images: Tensor):
        output = self._task_model._forward_static_task(images, self.task)
        if self.task == "detection":
            detection = output.detection
            if detection is None:
                raise RuntimeError("detection output is unexpectedly absent")
            return (
                *detection.cls_logits,
                *detection.box_regression,
                *detection.quality,
            )
        if self.task == "segmentation":
            if output.segmentation is None:
                raise RuntimeError("segmentation output is unexpectedly absent")
            return output.segmentation
        if self.task == "depth":
            if output.depth is None:
                raise RuntimeError("depth output is unexpectedly absent")
            return output.depth
        if output.classification is None:
            raise RuntimeError("classification output is unexpectedly absent")
        return output.classification


def create_replite_model(
    config: RepLiteConfig,
    **weight_options,
) -> RepLiteMultiTaskModel:
    """Construct a RepLite model from an immutable config."""

    return RepLiteMultiTaskModel(config, **weight_options)


__all__ = [
    "RepLiteMultiTaskModel",
    "RepLiteOutput",
    "TaskExportWrapper",
    "create_replite_model",
    "detach_state",
]
