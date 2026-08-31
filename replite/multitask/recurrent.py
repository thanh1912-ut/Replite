"""Lightweight convolutional LSTM feature refinement.

The same module supports two use cases:

* a single 4D feature map is recurrently refined for a configured number of
  steps while the input feature is reused; and
* a 5D ``B,T,C,H,W`` feature sequence is aggregated over its real time axis.

Only the input/hidden concatenation is spatially filtered.  A depthwise 3x3
convolution performs that filtering and a pointwise convolution emits all four
LSTM gates, keeping the recurrent cell small and deployment friendly.
"""

from __future__ import annotations

from typing import TypeAlias

import torch
from torch import Tensor, nn


LSTMState: TypeAlias = tuple[Tensor, Tensor]


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


class LiteConvLSTMCell(nn.Module):
    """Depthwise-separable ConvLSTM cell with no normalization in its gates."""

    def __init__(self, input_channels: int, hidden_channels: int) -> None:
        super().__init__()
        input_channels = _positive_int(input_channels, "input_channels")
        hidden_channels = _positive_int(hidden_channels, "hidden_channels")
        combined_channels = input_channels + hidden_channels

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.depthwise = nn.Conv2d(
            combined_channels,
            combined_channels,
            kernel_size=3,
            padding=1,
            groups=combined_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(
            combined_channels,
            4 * hidden_channels,
            kernel_size=1,
            bias=True,
        )

        # A positive forget bias is a stable default for recurrent refinement.
        # Gate order is input, forget, output, candidate.
        with torch.no_grad():
            self.pointwise.bias[hidden_channels : 2 * hidden_channels].fill_(1.0)

    def zero_state(
        self,
        batch_size: int,
        spatial_size: tuple[int, int],
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> LSTMState:
        """Create an independent all-zero hidden/cell state.

        When device or dtype is omitted it follows the cell parameters.  The
        explicit shape validation makes configuration errors fail before a
        convolution or concatenation emits a less actionable error.
        """

        batch_size = _positive_int(batch_size, "batch_size")
        if not isinstance(spatial_size, tuple) or len(spatial_size) != 2:
            raise TypeError("spatial_size must be a (height, width) tuple")
        height = _positive_int(spatial_size[0], "spatial_size[0]")
        width = _positive_int(spatial_size[1], "spatial_size[1]")
        if dtype is not None and not isinstance(dtype, torch.dtype):
            raise TypeError("dtype must be a torch.dtype or None")

        parameter = self.pointwise.weight
        state_device = parameter.device if device is None else torch.device(device)
        state_dtype = parameter.dtype if dtype is None else dtype
        probe = torch.empty((), device=state_device, dtype=state_dtype)
        if not probe.is_floating_point():
            raise TypeError("state dtype must be floating point")
        shape = (batch_size, self.hidden_channels, height, width)
        hidden = torch.zeros(shape, device=state_device, dtype=state_dtype)
        cell = torch.zeros(shape, device=state_device, dtype=state_dtype)
        return hidden, cell

    def _validate_input(self, x: Tensor) -> None:
        if not isinstance(x, Tensor):
            raise TypeError("x must be a torch.Tensor")
        if x.ndim != 4:
            raise ValueError("LiteConvLSTMCell input must have shape B,C,H,W")
        if torch.jit.is_tracing():
            return
        if x.shape[1] != self.input_channels:
            raise ValueError(
                f"expected {self.input_channels} input channels, got {x.shape[1]}"
            )
        if x.shape[0] <= 0 or x.shape[2] <= 0 or x.shape[3] <= 0:
            raise ValueError("input batch and spatial dimensions must be non-zero")
        if not x.is_floating_point():
            raise TypeError("x must have a floating-point dtype")

    def _validate_state(self, x: Tensor, state: LSTMState) -> None:
        if torch.jit.is_tracing():
            return
        if not isinstance(state, tuple) or len(state) != 2:
            raise TypeError("state must be a (hidden, cell) tuple")
        hidden, cell = state
        if not isinstance(hidden, Tensor) or not isinstance(cell, Tensor):
            raise TypeError("hidden and cell state must be torch.Tensor objects")
        expected = (
            x.shape[0],
            self.hidden_channels,
            x.shape[2],
            x.shape[3],
        )
        if tuple(hidden.shape) != expected:
            raise ValueError(
                f"hidden state must have shape {expected}, got {tuple(hidden.shape)}"
            )
        if tuple(cell.shape) != expected:
            raise ValueError(
                f"cell state must have shape {expected}, got {tuple(cell.shape)}"
            )
        if hidden.device != x.device or cell.device != x.device:
            raise ValueError("state and input must be on the same device")
        if hidden.dtype != x.dtype or cell.dtype != x.dtype:
            raise ValueError("state and input must have the same dtype")
        if not hidden.is_floating_point() or not cell.is_floating_point():
            raise TypeError("state tensors must have a floating-point dtype")

    def forward(self, x: Tensor, state: LSTMState | None = None) -> LSTMState:
        """Advance the cell by one step and return ``(hidden, cell)``."""

        self._validate_input(x)
        if state is None:
            # Derive the shape from ``x`` without converting symbolic sizes to
            # Python integers. This path remains traceable for dynamic
            # torch.export/ONNX graphs while ``zero_state`` keeps its strict
            # user-facing argument validation.
            shape = (
                x.shape[0],
                self.hidden_channels,
                x.shape[2],
                x.shape[3],
            )
            hidden = x.new_zeros(shape)
            cell = x.new_zeros(shape)
        else:
            self._validate_state(x, state)
            hidden, cell = state

        recurrent_input = torch.cat((x, hidden), dim=1)
        gates = self.pointwise(self.depthwise(recurrent_input))
        input_gate, forget_gate, output_gate, candidate = gates.chunk(4, dim=1)
        next_cell = torch.sigmoid(forget_gate) * cell + torch.sigmoid(
            input_gate
        ) * torch.tanh(candidate)
        next_hidden = torch.sigmoid(output_gate) * torch.tanh(next_cell)
        return next_hidden, next_cell

    def extra_repr(self) -> str:
        return (
            f"input_channels={self.input_channels}, "
            f"hidden_channels={self.hidden_channels}, kernel_size=3"
        )


class LiteConvLSTM(nn.Module):
    """Unroll :class:`LiteConvLSTMCell` over static or sequential features.

    Args:
        input_channels: Channel count in each input feature map.
        hidden_channels: Channel count of hidden, cell and output features.
        steps: Number of recurrent refinements for a 4D static feature map.
            A 5D input always uses its explicit sequence length instead.
    """

    def __init__(
        self, input_channels: int, hidden_channels: int, steps: int = 3
    ) -> None:
        super().__init__()
        input_channels = _positive_int(input_channels, "input_channels")
        hidden_channels = _positive_int(hidden_channels, "hidden_channels")
        steps = _positive_int(steps, "steps")
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.steps = steps
        self.cell = LiteConvLSTMCell(input_channels, hidden_channels)

    def zero_state(
        self,
        batch_size: int,
        spatial_size: tuple[int, int],
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> LSTMState:
        """Delegate zero-state creation to the underlying cell."""

        return self.cell.zero_state(
            batch_size, spatial_size, device=device, dtype=dtype
        )

    def forward(
        self,
        x: Tensor,
        state: LSTMState | None = None,
        *,
        steps: int | None = None,
    ) -> tuple[Tensor, LSTMState, Tensor]:
        """Run recurrent refinement or sequence aggregation.

        Args:
            x: Either ``B,C,H,W`` (repeat the same feature) or
                ``B,T,C,H,W`` (consume an explicit sequence).
            state: Optional initial ``(hidden, cell)`` state.
            steps: Optional per-call repeat count for a 4D input.  Supplying it
                for a 5D input is rejected because the sequence already defines
                the number of recurrent steps.

        Returns:
            ``(final_hidden, (final_hidden, final_cell), outputs)`` where
            ``outputs`` has shape ``B,T,hidden_channels,H,W``.
        """

        if not isinstance(x, Tensor):
            raise TypeError("x must be a torch.Tensor")
        if x.ndim not in (4, 5):
            raise ValueError("input must have shape B,C,H,W or B,T,C,H,W")
        if not x.is_floating_point():
            raise TypeError("x must have a floating-point dtype")

        if x.ndim == 4:
            repeat_steps = (
                self.steps if steps is None else _positive_int(steps, "steps")
            )
            frame = x
            frames = (frame for _ in range(repeat_steps))
        else:
            if steps is not None:
                raise ValueError("steps may not be overridden for a 5D input")
            if x.shape[1] <= 0:
                raise ValueError("sequence length must be non-zero")
            frames = (x[:, index] for index in range(x.shape[1]))

        current_state = state
        outputs: list[Tensor] = []
        for frame in frames:
            current_state = self.cell(frame, current_state)
            outputs.append(current_state[0])

        # Both input modes guarantee at least one step above.
        final_hidden, final_cell = current_state
        step_outputs = torch.stack(outputs, dim=1)
        return final_hidden, (final_hidden, final_cell), step_outputs

    def forward_final(
        self,
        x: Tensor,
        state: LSTMState | None = None,
        *,
        steps: int | None = None,
    ) -> tuple[Tensor, LSTMState]:
        """Run recurrence without materializing intermediate hidden outputs.

        This is the low-allocation path used by the neck, whose decoders only
        consume the final hidden state.  :meth:`forward` remains available for
        callers that explicitly need the complete temporal output sequence.
        """

        if not isinstance(x, Tensor):
            raise TypeError("x must be a torch.Tensor")
        if x.ndim not in (4, 5):
            raise ValueError("input must have shape B,C,H,W or B,T,C,H,W")
        if not x.is_floating_point():
            raise TypeError("x must have a floating-point dtype")

        current_state = state
        if x.ndim == 4:
            repeat_steps = (
                self.steps if steps is None else _positive_int(steps, "steps")
            )
            for _ in range(repeat_steps):
                current_state = self.cell(x, current_state)
        else:
            if steps is not None:
                raise ValueError("steps may not be overridden for a 5D input")
            if x.shape[1] <= 0:
                raise ValueError("sequence length must be non-zero")
            for index in range(x.shape[1]):
                current_state = self.cell(x[:, index], current_state)

        # Both input modes guarantee at least one step above.
        final_hidden, final_cell = current_state
        return final_hidden, (final_hidden, final_cell)

    def extra_repr(self) -> str:
        return (
            f"input_channels={self.input_channels}, "
            f"hidden_channels={self.hidden_channels}, steps={self.steps}"
        )


__all__ = ["LSTMState", "LiteConvLSTMCell", "LiteConvLSTM"]
