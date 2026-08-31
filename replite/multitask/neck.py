"""Recurrent feature neck shared by RepLite perception tasks.

The neck has no hidden recurrent state.  Static-image refinement and streaming
video inference therefore share exactly the same weights while callers retain
explicit control over sequence boundaries through :class:`NeckState`.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .blocks import ConvBNAct, DepthwiseSeparableConv, RepDepthwiseBlock
from .recurrent import LiteConvLSTM


RecurrentLevelState = tuple[Tensor, Tensor]


class NeckState(NamedTuple):
    """Explicit ConvLSTM states for the stride-16 and stride-32 levels."""

    level4: RecurrentLevelState | None
    level5: RecurrentLevelState | None


class NeckOutput(NamedTuple):
    """Task features produced from the final recurrent state.

    Detection features and the dense feature are optional because task-specific
    model artifacts physically omit disabled paths. ``r4`` and ``r5`` are
    present only when the static task set requires their recurrent level.
    """

    d3: Tensor | None
    d4: Tensor | None
    d5: Tensor | None
    f2: Tensor | None
    r4: Tensor | None
    r5: Tensor | None

    @property
    def detection(self) -> tuple[Tensor, Tensor, Tensor] | None:
        """Return the detection pyramid, or ``None`` when it is pruned."""

        if self.d3 is None:
            return None
        # All three values are constructed together by the detection path.
        assert self.d4 is not None and self.d5 is not None
        return self.d3, self.d4, self.d5


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _channel_tuple(
    value: object,
    *,
    name: str,
    length: int,
) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must contain exactly {length} positive integers")
    channels = tuple(value)
    if len(channels) != length:
        raise ValueError(f"{name} must contain exactly {length} positive integers")
    for channel in channels:
        _positive_int(channel, name)
    return channels


def _resize_like(source: Tensor, reference: Tensor) -> Tensor:
    """Nearest-neighbour resize using an explicit reference shape.

    Referencing the destination tensor rather than a scale factor keeps fusion
    well-defined for odd feature-map dimensions.
    """

    return F.interpolate(source, size=reference.shape[-2:], mode="nearest")


class _LiteSPPF(nn.Module):
    """Small spatial-pyramid pooling block with only two learned projections."""

    def __init__(self, channels: int, kernel_size: int = 5) -> None:
        super().__init__()
        hidden_channels = max(channels // 2, 1)
        self.reduce = ConvBNAct(
            channels, hidden_channels, kernel_size=1, activation="silu"
        )
        self.pool = nn.MaxPool2d(
            kernel_size=kernel_size, stride=1, padding=kernel_size // 2
        )
        self.expand = ConvBNAct(
            hidden_channels * 4, channels, kernel_size=1, activation="silu"
        )

    def forward(self, x: Tensor) -> Tensor:
        x0 = self.reduce(x)
        x1 = self.pool(x0)
        x2 = self.pool(x1)
        x3 = self.pool(x2)
        return self.expand(torch.cat((x0, x1, x2, x3), dim=1))


class RecurrentMultiTaskNeck(nn.Module):
    """LiteConvLSTM feature refinement plus prunable task-specific decoders.

    Args:
        in_channels: Native backbone channels for ``(C2, C3, C4, C5)``.
        recurrent_channels: Hidden widths for recurrent levels C4 and C5.
        detection_channels: Common width of D3, D4 and D5.
        dense_channels: Width of the shared stride-4 dense feature F2.
        refine_steps: Default number of repeated recurrent steps for an image.
        use_sppf: Add lightweight SPPF at P5 when detection is enabled.
        enable_detection: Materialize and execute the detection path.
        enable_dense: Materialize and execute the dense prediction path.
        enable_level4: Materialize recurrent refinement at stride 16.
        enable_level5: Materialize recurrent refinement at stride 32.

    Inputs are the native ``(C2, C3, C4, C5)`` pyramid when level 5 is enabled,
    or ``(C2, C3, C4)`` for a dense-only artifact that physically prunes C5.
    Each stage must be the ceil-half spatial shape of the preceding stage,
    matching stride-4/8/16/32 padded convolutional backbones.
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        recurrent_channels: Sequence[int] = (48, 64),
        detection_channels: int = 48,
        dense_channels: int = 32,
        refine_steps: int = 3,
        use_sppf: bool = False,
        enable_detection: bool = True,
        enable_dense: bool = True,
        enable_level4: bool = True,
        enable_level5: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(in_channels, (str, bytes)) or not isinstance(
            in_channels, Sequence
        ):
            raise ValueError("in_channels must contain three or four positive integers")
        self.in_channels = tuple(in_channels)
        if len(self.in_channels) not in (3, 4):
            raise ValueError("in_channels must contain three or four positive integers")
        for channel in self.in_channels:
            _positive_int(channel, "in_channels")
        recurrent = _channel_tuple(
            recurrent_channels, name="recurrent_channels", length=2
        )
        self.recurrent_channels = (recurrent[0], recurrent[1])
        self.detection_channels = _positive_int(
            detection_channels, "detection_channels"
        )
        self.dense_channels = _positive_int(dense_channels, "dense_channels")
        self.refine_steps = _positive_int(refine_steps, "refine_steps")
        if not isinstance(use_sppf, bool):
            raise ValueError("use_sppf must be a boolean")
        if not isinstance(enable_detection, bool):
            raise ValueError("enable_detection must be a boolean")
        if not isinstance(enable_dense, bool):
            raise ValueError("enable_dense must be a boolean")
        if not isinstance(enable_level4, bool):
            raise ValueError("enable_level4 must be a boolean")
        if not isinstance(enable_level5, bool):
            raise ValueError("enable_level5 must be a boolean")
        if enable_detection and not (enable_level4 and enable_level5):
            raise ValueError("detection requires recurrent levels 4 and 5")
        if enable_dense and not enable_level4:
            raise ValueError("the dense path requires recurrent level 4")
        if enable_level5 and len(self.in_channels) != 4:
            raise ValueError("recurrent level 5 requires C5 input channels")
        if not enable_level4 and not enable_level5:
            raise ValueError("at least one recurrent level must be enabled")
        self.use_sppf = use_sppf
        self.enable_detection = enable_detection
        self.enable_dense = enable_dense
        self.enable_level4 = enable_level4
        self.enable_level5 = enable_level5

        c2_channels, c3_channels, c4_channels = self.in_channels[:3]
        c5_channels = self.in_channels[3] if len(self.in_channels) == 4 else None
        r4_channels, r5_channels = self.recurrent_channels
        if self.enable_level4:
            self.c4_projection = ConvBNAct(
                c4_channels, r4_channels, kernel_size=1, activation="silu"
            )
            self.recurrent4 = LiteConvLSTM(
                r4_channels, r4_channels, steps=self.refine_steps
            )
        if self.enable_level5:
            assert c5_channels is not None
            self.c5_projection = ConvBNAct(
                c5_channels, r5_channels, kernel_size=1, activation="silu"
            )
            self.recurrent5 = LiteConvLSTM(
                r5_channels, r5_channels, steps=self.refine_steps
            )

        if self.enable_detection:
            detection_path: dict[str, nn.Module] = {
                "lateral5": ConvBNAct(
                    r5_channels,
                    self.detection_channels,
                    kernel_size=1,
                    activation="silu",
                ),
                "lateral4": ConvBNAct(
                    r4_channels,
                    self.detection_channels,
                    kernel_size=1,
                    activation="silu",
                ),
                "lateral3": ConvBNAct(
                    c3_channels,
                    self.detection_channels,
                    kernel_size=1,
                    activation="silu",
                ),
                "top4": RepDepthwiseBlock(self.detection_channels),
                "top3": RepDepthwiseBlock(self.detection_channels),
                "down3": DepthwiseSeparableConv(
                    self.detection_channels,
                    self.detection_channels,
                    kernel_size=3,
                    stride=2,
                    activation="silu",
                ),
                "pan4": RepDepthwiseBlock(self.detection_channels),
                "down4": DepthwiseSeparableConv(
                    self.detection_channels,
                    self.detection_channels,
                    kernel_size=3,
                    stride=2,
                    activation="silu",
                ),
                "pan5": RepDepthwiseBlock(self.detection_channels),
            }
            if self.use_sppf:
                detection_path["sppf"] = _LiteSPPF(self.detection_channels)
            self.detection_path = nn.ModuleDict(detection_path)

        if self.enable_dense:
            self.dense_path = nn.ModuleDict(
                {
                    "lateral4": ConvBNAct(
                        r4_channels,
                        self.dense_channels,
                        kernel_size=1,
                        activation="silu",
                    ),
                    "lateral3": ConvBNAct(
                        c3_channels,
                        self.dense_channels,
                        kernel_size=1,
                        activation="silu",
                    ),
                    "refine3": RepDepthwiseBlock(self.dense_channels),
                    "lateral2": ConvBNAct(
                        c2_channels,
                        self.dense_channels,
                        kernel_size=1,
                        activation="silu",
                    ),
                    "refine2": RepDepthwiseBlock(self.dense_channels),
                }
            )

    def _validate_features(
        self, features: Sequence[Tensor]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        if isinstance(features, Tensor) or not isinstance(features, Sequence):
            raise ValueError("features must be a sequence matching in_channels")
        if len(features) != len(self.in_channels):
            expected = "C2, C3, C4, C5" if len(self.in_channels) == 4 else "C2, C3, C4"
            raise ValueError(f"features must contain exactly {expected}")
        if torch.jit.is_tracing():
            c5 = features[3] if len(features) == 4 else None
            return features[0], features[1], features[2], c5

        checked: list[Tensor] = []
        for index, (feature, channels) in enumerate(
            zip(features, self.in_channels), start=2
        ):
            if not isinstance(feature, Tensor):
                raise ValueError(f"C{index} must be a torch.Tensor")
            if feature.ndim != 4:
                raise ValueError(f"C{index} must be a 4D NCHW tensor")
            if feature.shape[1] != channels:
                raise ValueError(
                    f"C{index} must have {channels} channels, got {feature.shape[1]}"
                )
            if feature.shape[0] <= 0 or min(feature.shape[-2:]) <= 0:
                raise ValueError(f"C{index} must have non-empty batch and spatial axes")
            if not feature.is_floating_point():
                raise ValueError(f"C{index} must use a floating-point dtype")
            checked.append(feature)

        reference = checked[0]
        for index, feature in enumerate(checked[1:], start=3):
            if feature.shape[0] != reference.shape[0]:
                raise ValueError("all feature levels must have the same batch size")
            if feature.device != reference.device:
                raise ValueError("all feature levels must be on the same device")
            if feature.dtype != reference.dtype:
                raise ValueError("all feature levels must use the same dtype")

            previous = checked[index - 3]
            expected = tuple((dimension + 1) // 2 for dimension in previous.shape[-2:])
            if feature.shape[-2:] != expected:
                raise ValueError(
                    f"C{index} spatial shape must be ceil-half of C{index - 1}; "
                    f"expected {expected}, got {tuple(feature.shape[-2:])}"
                )

        c5 = checked[3] if len(checked) == 4 else None
        return checked[0], checked[1], checked[2], c5

    def _validate_state(
        self,
        state: NeckState | tuple[RecurrentLevelState, RecurrentLevelState] | None,
        x4: Tensor | None,
        x5: Tensor | None,
    ) -> NeckState | None:
        if state is None:
            return None
        if not isinstance(state, (tuple, list)) or len(state) != 2:
            raise ValueError("state must contain recurrent level4 and level5 states")

        validated: list[RecurrentLevelState | None] = []
        for name, level_state, feature in (
            ("level4", state[0], x4),
            ("level5", state[1], x5),
        ):
            if feature is None:
                if level_state is not None:
                    raise ValueError(f"state.{name} must be None when disabled")
                validated.append(None)
                continue
            if level_state is None:
                raise ValueError(f"state.{name} must be an (h, c) pair")
            if not isinstance(level_state, (tuple, list)) or len(level_state) != 2:
                raise ValueError(f"state.{name} must be an (h, c) pair")
            expected_shape = tuple(feature.shape)
            pair: list[Tensor] = []
            for tensor_name, tensor in zip(("h", "c"), level_state):
                if not isinstance(tensor, Tensor):
                    raise ValueError(f"state.{name}.{tensor_name} must be a tensor")
                if tuple(tensor.shape) != expected_shape:
                    raise ValueError(
                        f"state.{name}.{tensor_name} must have shape "
                        f"{expected_shape}, got {tuple(tensor.shape)}"
                    )
                if tensor.device != feature.device or tensor.dtype != feature.dtype:
                    raise ValueError(
                        f"state.{name}.{tensor_name} must match its feature "
                        "dtype/device"
                    )
                pair.append(tensor)
            validated.append((pair[0], pair[1]))
        return NeckState(validated[0], validated[1])

    @staticmethod
    def _resolve_path_flag(value: bool | None, enabled: bool, name: str) -> bool:
        if value is None:
            return enabled
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be a boolean or None")
        if value and not enabled:
            raise ValueError(f"{name} requested a path absent from this neck")
        return value

    def _advance_recurrent(
        self,
        c4: Tensor,
        c5: Tensor | None,
        state: NeckState | tuple[RecurrentLevelState, RecurrentLevelState] | None,
        *,
        steps: int,
        include_level4: bool,
        include_level5: bool,
    ) -> tuple[Tensor | None, Tensor | None, NeckState]:
        if include_level4 and not self.enable_level4:
            raise ValueError("recurrent level 4 is absent from this neck")
        if include_level5 and not self.enable_level5:
            raise ValueError("recurrent level 5 is absent from this neck")

        x4 = self.c4_projection(c4) if include_level4 else None
        if include_level5:
            if c5 is None:
                raise ValueError("recurrent level 5 requires a C5 feature")
            x5 = self.c5_projection(c5)
        else:
            x5 = None

        # Validate against projected tensors: under autocast these may use a
        # different dtype from the raw backbone features, and returned state
        # must match the tensors entering the recurrent cells.
        state = self._validate_state(state, x4, x5)
        level4 = None if state is None else state.level4
        level5 = None if state is None else state.level5

        r4: Tensor | None = None
        r5: Tensor | None = None
        state4: RecurrentLevelState | None = None
        state5: RecurrentLevelState | None = None
        if x4 is not None:
            r4, state4 = self.recurrent4.forward_final(x4, state=level4, steps=steps)
        if x5 is not None:
            r5, state5 = self.recurrent5.forward_final(x5, state=level5, steps=steps)
        return r4, r5, NeckState(state4, state5)

    def _decode(
        self,
        c2: Tensor,
        c3: Tensor,
        r4: Tensor | None,
        r5: Tensor | None,
        *,
        decode_detection: bool | None = None,
        decode_dense: bool | None = None,
    ) -> NeckOutput:
        d3: Tensor | None = None
        d4: Tensor | None = None
        d5: Tensor | None = None
        f2: Tensor | None = None

        decode_detection = self._resolve_path_flag(
            decode_detection, self.enable_detection, "decode_detection"
        )
        decode_dense = self._resolve_path_flag(
            decode_dense, self.enable_dense, "decode_dense"
        )

        if decode_detection:
            if r4 is None or r5 is None:
                raise RuntimeError(
                    "detection decoding requires recurrent levels 4 and 5"
                )
            path = self.detection_path
            p5 = path["lateral5"](r5)
            if self.use_sppf:
                p5 = path["sppf"](p5)
            p4_lateral = path["lateral4"](r4)
            p4 = path["top4"](p4_lateral + _resize_like(p5, p4_lateral))
            p3_lateral = path["lateral3"](c3)
            p3 = path["top3"](p3_lateral + _resize_like(p4, p3_lateral))

            d3 = p3
            down3 = _resize_like(path["down3"](d3), p4)
            d4 = path["pan4"](p4 + down3)
            down4 = _resize_like(path["down4"](d4), p5)
            d5 = path["pan5"](p5 + down4)

        if decode_dense:
            if r4 is None:
                raise RuntimeError("dense decoding requires recurrent level 4")
            path = self.dense_path
            f3_lateral = path["lateral3"](c3)
            f4_lateral = path["lateral4"](r4)
            f3 = path["refine3"](f3_lateral + _resize_like(f4_lateral, f3_lateral))
            f2_lateral = path["lateral2"](c2)
            f2 = path["refine2"](f2_lateral + _resize_like(f3, f2_lateral))

        return NeckOutput(d3=d3, d4=d4, d5=d5, f2=f2, r4=r4, r5=r5)

    def refine(
        self,
        features: Sequence[Tensor],
        steps: int | None = None,
        *,
        decode_detection: bool | None = None,
        decode_dense: bool | None = None,
        include_level4: bool | None = None,
        include_level5: bool | None = None,
    ) -> tuple[NeckOutput, NeckState]:
        """Repeatedly refine one static feature pyramid from zero state.

        Optional route flags let task export omit unrelated recurrent levels
        and decoders.  ``None`` preserves the statically configured default.
        """

        c2, c3, c4, c5 = self._validate_features(features)
        if steps is None:
            steps = self.refine_steps
        steps = _positive_int(steps, "steps")
        include_level4 = self._resolve_path_flag(
            include_level4, self.enable_level4, "include_level4"
        )
        include_level5 = self._resolve_path_flag(
            include_level5, self.enable_level5, "include_level5"
        )
        r4, r5, state = self._advance_recurrent(
            c4,
            c5,
            None,
            steps=steps,
            include_level4=include_level4,
            include_level5=include_level5,
        )
        return (
            self._decode(
                c2,
                c3,
                r4,
                r5,
                decode_detection=decode_detection,
                decode_dense=decode_dense,
            ),
            state,
        )

    def advance(
        self,
        features: Sequence[Tensor],
        state: (
            NeckState | tuple[RecurrentLevelState, RecurrentLevelState] | None
        ) = None,
    ) -> NeckState:
        """Advance recurrent state for one frame without running any decoder."""

        _, _, c4, c5 = self._validate_features(features)
        _, _, next_state = self._advance_recurrent(
            c4,
            c5,
            state,
            steps=1,
            include_level4=self.enable_level4,
            include_level5=self.enable_level5,
        )
        return next_state

    def step(
        self,
        features: Sequence[Tensor],
        state: (
            NeckState | tuple[RecurrentLevelState, RecurrentLevelState] | None
        ) = None,
    ) -> tuple[NeckOutput, NeckState]:
        """Process one streaming frame with optional explicit prior state."""

        c2, c3, c4, c5 = self._validate_features(features)
        r4, r5, next_state = self._advance_recurrent(
            c4,
            c5,
            state,
            steps=1,
            include_level4=self.enable_level4,
            include_level5=self.enable_level5,
        )
        return self._decode(c2, c3, r4, r5), next_state

    def forward(
        self,
        features: Sequence[Tensor],
        steps: int | None = None,
    ) -> tuple[NeckOutput, NeckState]:
        """Alias for static-image :meth:`refine`."""

        return self.refine(features, steps=steps)

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, "
            f"recurrent_channels={self.recurrent_channels}, "
            f"detection_channels={self.detection_channels}, "
            f"dense_channels={self.dense_channels}, "
            f"refine_steps={self.refine_steps}, "
            f"use_sppf={self.use_sppf}, "
            f"enable_detection={self.enable_detection}, "
            f"enable_dense={self.enable_dense}, "
            f"enable_level4={self.enable_level4}, "
            f"enable_level5={self.enable_level5}"
        )


__all__ = [
    "NeckOutput",
    "NeckState",
    "RecurrentLevelState",
    "RecurrentMultiTaskNeck",
]
