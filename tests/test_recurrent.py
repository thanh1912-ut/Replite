"""Unit tests for mobile blocks and lightweight recurrent refinement."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from replite.multitask.blocks import (
    ConvBNAct,
    DepthwiseSeparableConv,
    RepDepthwiseBlock,
)
from replite.multitask.recurrent import LiteConvLSTM, LiteConvLSTMCell


def _input(*shape: int, seed: int = 2026) -> torch.Tensor:
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


def test_mobile_convolution_blocks_have_expected_shapes() -> None:
    x = _input(2, 8, 16, 20)
    projection = ConvBNAct(8, 12, kernel_size=1, activation="relu")
    downsample = DepthwiseSeparableConv(8, 10, stride=2)

    assert projection(x).shape == (2, 12, 16, 20)
    assert downsample(x).shape == (2, 10, 8, 10)
    assert downsample.depthwise.conv.groups == 8
    assert downsample.depthwise.conv.bias is None
    assert downsample.pointwise.conv.kernel_size == (1, 1)


def test_rep_depthwise_deploy_fusion_preserves_eval_output() -> None:
    torch.manual_seed(7)
    block = RepDepthwiseBlock(6, activation="silu")

    # Give all three BN branches non-default running statistics before fusion.
    block.train()
    with torch.no_grad():
        for seed in range(4):
            block(_input(4, 6, 13, 17, seed=seed))

    block.eval()
    x = _input(2, 6, 13, 17, seed=99)
    with torch.no_grad():
        reference = block(x)
        returned = block.switch_to_deploy()
        deployed = block(x)

    assert returned is block
    assert block.deploy
    assert block.reparam_conv.groups == 6
    assert not hasattr(block, "dw_branch")
    torch.testing.assert_close(reference, deployed, rtol=1e-5, atol=2e-6)

    # Conversion is deliberately idempotent.
    assert block.switch_to_deploy() is block
    with torch.no_grad():
        torch.testing.assert_close(deployed, block(x), rtol=0.0, atol=0.0)


def test_rep_depthwise_deploy_preserves_eval_and_frozen_state() -> None:
    block = RepDepthwiseBlock(6).eval().requires_grad_(False)
    block.switch_to_deploy()

    assert not block.training
    assert not block.reparam_conv.training
    assert all(not parameter.requires_grad for parameter in block.parameters())


def test_lite_convlstm_cell_architecture_and_forward_shape() -> None:
    cell = LiteConvLSTMCell(input_channels=5, hidden_channels=7)
    x = _input(2, 5, 11, 13)
    hidden, state = cell(x)

    assert hidden.shape == state.shape == (2, 7, 11, 13)
    assert cell.depthwise.kernel_size == (3, 3)
    assert cell.depthwise.groups == 12
    assert cell.pointwise.in_channels == 12
    assert cell.pointwise.out_channels == 28
    assert not any(
        isinstance(module, nn.modules.batchnorm._BatchNorm) for module in cell.modules()
    )


def test_cell_zero_state_matches_explicit_state_exactly() -> None:
    torch.manual_seed(8)
    cell = LiteConvLSTMCell(4, 6).eval()
    x = _input(2, 4, 9, 7)
    explicit = cell.zero_state(2, (9, 7), device=x.device, dtype=x.dtype)

    with torch.no_grad():
        implicit_output = cell(x)
        explicit_output = cell(x, explicit)
        repeated_output = cell(x, explicit)

    for implicit, supplied, repeated in zip(
        implicit_output, explicit_output, repeated_output
    ):
        torch.testing.assert_close(implicit, supplied, rtol=0.0, atol=0.0)
        torch.testing.assert_close(supplied, repeated, rtol=0.0, atol=0.0)


def test_cell_equations_with_deterministic_parameters() -> None:
    cell = LiteConvLSTMCell(2, 3)
    with torch.no_grad():
        for parameter in cell.parameters():
            parameter.zero_()

    x = torch.ones(1, 2, 4, 5)
    hidden = torch.zeros(1, 3, 4, 5)
    previous_cell = torch.ones(1, 3, 4, 5)
    next_hidden, next_cell = cell(x, (hidden, previous_cell))

    expected_cell = torch.full_like(previous_cell, 0.5)
    expected_hidden = 0.5 * torch.tanh(expected_cell)
    torch.testing.assert_close(next_cell, expected_cell, rtol=0.0, atol=0.0)
    torch.testing.assert_close(next_hidden, expected_hidden, rtol=0.0, atol=0.0)


def test_static_repeat_and_explicit_sequence_are_exactly_equivalent() -> None:
    torch.manual_seed(123)
    recurrent = LiteConvLSTM(4, 6, steps=4).eval()
    x = _input(2, 4, 8, 10)
    initial_state = (
        _input(2, 6, 8, 10, seed=40),
        _input(2, 6, 8, 10, seed=41),
    )
    sequence = x.unsqueeze(1).repeat(1, 4, 1, 1, 1)

    with torch.no_grad():
        static_result = recurrent(x, initial_state)
        sequence_result = recurrent(sequence, initial_state)

    for static, explicit in zip(
        (
            static_result[0],
            static_result[1][0],
            static_result[1][1],
            static_result[2],
        ),
        (
            sequence_result[0],
            sequence_result[1][0],
            sequence_result[1][1],
            sequence_result[2],
        ),
    ):
        torch.testing.assert_close(static, explicit, rtol=0.0, atol=0.0)


def test_sequence_shapes_and_stateful_continuation() -> None:
    torch.manual_seed(9)
    recurrent = LiteConvLSTM(3, 5, steps=2).eval()
    sequence = _input(2, 4, 3, 7, 9)

    with torch.no_grad():
        final_full, state_full, outputs_full = recurrent(sequence)
        _, state_prefix, outputs_prefix = recurrent(sequence[:, :2])
        final_suffix, state_suffix, outputs_suffix = recurrent(
            sequence[:, 2:], state_prefix
        )

    assert final_full.shape == (2, 5, 7, 9)
    assert state_full[0].shape == state_full[1].shape == (2, 5, 7, 9)
    assert outputs_full.shape == (2, 4, 5, 7, 9)
    assert final_full is state_full[0]
    torch.testing.assert_close(final_full, final_suffix, rtol=0.0, atol=0.0)
    torch.testing.assert_close(state_full[0], state_suffix[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(state_full[1], state_suffix[1], rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        outputs_full,
        torch.cat((outputs_prefix, outputs_suffix), dim=1),
        rtol=0.0,
        atol=0.0,
    )


def test_static_steps_can_be_overridden_per_call() -> None:
    recurrent = LiteConvLSTM(3, 4, steps=3)
    final, state, outputs = recurrent(_input(1, 3, 6, 8), steps=1)
    assert final.shape == state[0].shape == state[1].shape == (1, 4, 6, 8)
    assert outputs.shape == (1, 1, 4, 6, 8)


@pytest.mark.parametrize("rank", [4, 5])
def test_final_only_recurrence_matches_full_sequence_result(rank) -> None:
    recurrent = LiteConvLSTM(3, 4, steps=3).eval()
    x = _input(2, 3, 6, 8)
    if rank == 5:
        x = x[:, None].repeat(1, 3, 1, 1, 1)

    with torch.no_grad():
        expected_hidden, expected_state, _ = recurrent(x)
        actual_hidden, actual_state = recurrent.forward_final(x)

    torch.testing.assert_close(actual_hidden, expected_hidden, rtol=0.0, atol=0.0)
    for actual, expected in zip(actual_state, expected_state):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_all_recurrent_parameters_receive_finite_gradients() -> None:
    torch.manual_seed(10)
    recurrent = LiteConvLSTM(4, 7, steps=3)
    x = _input(2, 4, 8, 9).requires_grad_(True)
    final, (_, cell), outputs = recurrent(x)
    (final.square().mean() + cell.abs().mean() + outputs.square().mean()).backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, parameter in recurrent.named_parameters():
        assert parameter.grad is not None, f"unused recurrent parameter: {name}"
        assert torch.isfinite(parameter.grad).all(), name


@pytest.mark.parametrize(
    ("state", "error", "message"),
    [
        ([torch.zeros(1), torch.zeros(1)], TypeError, "tuple"),
        ((torch.zeros(1),), TypeError, "tuple"),
        (("hidden", "cell"), TypeError, "Tensor"),
        (
            (torch.zeros(2, 4, 6, 7), torch.zeros(2, 5, 6, 7)),
            ValueError,
            "hidden state",
        ),
        (
            (torch.zeros(2, 5, 6, 7), torch.zeros(2, 5, 6, 8)),
            ValueError,
            "cell state",
        ),
        (
            (
                torch.zeros(2, 5, 6, 7, dtype=torch.float64),
                torch.zeros(2, 5, 6, 7, dtype=torch.float64),
            ),
            ValueError,
            "dtype",
        ),
    ],
)
def test_invalid_states_fail_early(state, error, message) -> None:
    cell = LiteConvLSTMCell(3, 5)
    with pytest.raises(error, match=message):
        cell(_input(2, 3, 6, 7), state)


@pytest.mark.parametrize(
    "bad_input",
    [
        torch.zeros(2, 3, 4),
        torch.zeros(2, 1, 3, 4, 5, 6),
        torch.zeros(2, 3, 4, 5, dtype=torch.int64),
        torch.zeros(2, 0, 3, 4, 5),
    ],
)
def test_invalid_recurrent_inputs_fail_early(bad_input) -> None:
    recurrent = LiteConvLSTM(3, 5)
    with pytest.raises((TypeError, ValueError)):
        recurrent(bad_input)


def test_invalid_channels_steps_and_zero_state_are_rejected() -> None:
    with pytest.raises(ValueError, match="input_channels"):
        LiteConvLSTMCell(0, 4)
    with pytest.raises(ValueError, match="hidden_channels"):
        LiteConvLSTM(3, True)
    with pytest.raises(ValueError, match="steps"):
        LiteConvLSTM(3, 4, steps=0)

    recurrent = LiteConvLSTM(3, 4)
    with pytest.raises(ValueError, match="steps"):
        recurrent(_input(1, 3, 5, 5), steps=False)
    with pytest.raises(ValueError, match="5D input"):
        recurrent(_input(1, 2, 3, 5, 5), steps=2)
    with pytest.raises(ValueError, match="batch_size"):
        recurrent.zero_state(0, (5, 5))
    with pytest.raises(TypeError, match="tuple"):
        recurrent.zero_state(1, [5, 5])
    with pytest.raises(TypeError, match="floating"):
        recurrent.zero_state(1, (5, 5), dtype=torch.int64)


def test_channel_and_state_spatial_mismatches_are_rejected() -> None:
    cell = LiteConvLSTMCell(3, 4)
    with pytest.raises(ValueError, match="input channels"):
        cell(_input(2, 2, 6, 7))

    wrong_state = cell.zero_state(2, (5, 7))
    with pytest.raises(ValueError, match="hidden state"):
        cell(_input(2, 3, 6, 7), wrong_state)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ConvBNAct(3, 4, kernel_size=2),
        lambda: ConvBNAct(3, 4, groups=2),
        lambda: DepthwiseSeparableConv(3, 4, pointwise_activation=1),
        lambda: RepDepthwiseBlock(4, kernel_size=2),
    ],
)
def test_invalid_block_configuration_is_rejected(factory) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory()
