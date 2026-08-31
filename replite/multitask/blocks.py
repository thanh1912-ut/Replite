"""Mobile-friendly convolution blocks used by the multi-task network.

The primitives in this module deliberately use deployment-friendly operators:
convolutions, BatchNorm, elementwise additions and common activations.  The
reparameterizable depthwise block has a multi-branch training form and can be
collapsed into one depthwise convolution for inference.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Final

import torch
from torch import Tensor, nn
from torch.nn import functional as F


_ACTIVATION_NAMES: Final = {
    "identity",
    "none",
    "relu",
    "relu6",
    "silu",
    "hardswish",
}


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _activation(activation: str | nn.Module | bool | None) -> nn.Module:
    """Build an activation without relying on framework-specific factories."""

    if isinstance(activation, nn.Module):
        return deepcopy(activation)
    if activation is True:
        activation = "silu"
    if activation is False or activation is None:
        activation = "identity"
    if not isinstance(activation, str):
        raise TypeError("activation must be a string, nn.Module, bool, or None")
    name = activation.lower()
    if name not in _ACTIVATION_NAMES:
        choices = ", ".join(sorted(_ACTIVATION_NAMES))
        raise ValueError(
            f"unknown activation {activation!r}; expected one of {choices}"
        )
    if name in {"identity", "none"}:
        return nn.Identity()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "relu6":
        return nn.ReLU6(inplace=True)
    if name == "silu":
        return nn.SiLU(inplace=True)
    return nn.Hardswish(inplace=True)


class ConvBNAct(nn.Module):
    """Convolution followed by BatchNorm and an optional activation.

    Padding defaults to ``same`` for odd kernels.  Bias is intentionally
    disabled because BatchNorm supplies the affine transform and can be fused
    with the convolution by deployment toolchains.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        *,
        padding: int | None = None,
        groups: int = 1,
        dilation: int = 1,
        activation: str | nn.Module | bool | None = "silu",
    ) -> None:
        super().__init__()
        in_channels = _positive_int(in_channels, "in_channels")
        out_channels = _positive_int(out_channels, "out_channels")
        kernel_size = _positive_int(kernel_size, "kernel_size")
        stride = _positive_int(stride, "stride")
        groups = _positive_int(groups, "groups")
        dilation = _positive_int(dilation, "dilation")
        if in_channels % groups != 0 or out_channels % groups != 0:
            raise ValueError("groups must divide both in_channels and out_channels")
        if padding is None:
            if kernel_size % 2 == 0:
                raise ValueError(
                    "automatic padding requires an odd kernel_size; pass padding explicitly"
                )
            padding = dilation * (kernel_size - 1) // 2
        if isinstance(padding, bool) or not isinstance(padding, int) or padding < 0:
            raise ValueError("padding must be a non-negative integer or None")

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = _activation(activation)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))


class DepthwiseSeparableConv(nn.Module):
    """Depthwise spatial convolution followed by a pointwise projection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        *,
        dilation: int = 1,
        activation: str | nn.Module | bool | None = "silu",
        pointwise_activation: bool = True,
    ) -> None:
        super().__init__()
        in_channels = _positive_int(in_channels, "in_channels")
        out_channels = _positive_int(out_channels, "out_channels")
        kernel_size = _positive_int(kernel_size, "kernel_size")
        stride = _positive_int(stride, "stride")
        dilation = _positive_int(dilation, "dilation")
        if not isinstance(pointwise_activation, bool):
            raise TypeError("pointwise_activation must be a bool")

        self.depthwise = ConvBNAct(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            groups=in_channels,
            dilation=dilation,
            activation=activation,
        )
        self.pointwise = ConvBNAct(
            in_channels,
            out_channels,
            1,
            activation=activation if pointwise_activation else None,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.pointwise(self.depthwise(x))


class RepDepthwiseBlock(nn.Module):
    """Reparameterizable, shape-preserving depthwise convolution block.

    During training, the block sums an odd-kernel depthwise Conv-BN branch, a
    1x1 depthwise Conv-BN branch and an identity-BN branch.  In eval mode those
    branches are algebraically equivalent to one biased depthwise convolution.
    :meth:`switch_to_deploy` performs that fusion in place.

    Call ``eval()`` before comparing the train and deploy structures: BatchNorm
    uses batch statistics while training but its stored statistics are what can
    be folded into a convolution.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        *,
        activation: str | nn.Module | bool | None = "silu",
        deploy: bool = False,
    ) -> None:
        super().__init__()
        channels = _positive_int(channels, "channels")
        kernel_size = _positive_int(kernel_size, "kernel_size")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        if not isinstance(deploy, bool):
            raise TypeError("deploy must be a bool")

        self.channels = channels
        self.kernel_size = kernel_size
        self.deploy = deploy
        self.act = _activation(activation)
        padding = kernel_size // 2

        if deploy:
            self.reparam_conv = nn.Conv2d(
                channels,
                channels,
                kernel_size,
                padding=padding,
                groups=channels,
                bias=True,
            )
        else:
            self.dw_branch = nn.Sequential(
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size,
                    padding=padding,
                    groups=channels,
                    bias=False,
                ),
                nn.BatchNorm2d(channels),
            )
            self.point_branch = nn.Sequential(
                nn.Conv2d(
                    channels,
                    channels,
                    1,
                    groups=channels,
                    bias=False,
                ),
                nn.BatchNorm2d(channels),
            )
            self.identity_branch = nn.BatchNorm2d(channels)

    @staticmethod
    def _fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> tuple[Tensor, Tensor]:
        if bn.running_mean is None or bn.running_var is None:
            raise RuntimeError("BatchNorm must track running statistics for fusion")
        kernel = conv.weight
        if conv.bias is None:
            bias = torch.zeros(
                conv.out_channels, device=kernel.device, dtype=kernel.dtype
            )
        else:
            bias = conv.bias
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        return (
            kernel * scale.reshape(-1, 1, 1, 1),
            bn.bias + (bias - bn.running_mean) * scale,
        )

    def _fuse_identity_bn(self, bn: nn.BatchNorm2d) -> tuple[Tensor, Tensor]:
        kernel = bn.weight.new_zeros(
            self.channels, 1, self.kernel_size, self.kernel_size
        )
        kernel[:, 0, self.kernel_size // 2, self.kernel_size // 2] = 1.0
        identity = nn.Conv2d(
            self.channels,
            self.channels,
            self.kernel_size,
            padding=self.kernel_size // 2,
            groups=self.channels,
            bias=False,
            device=kernel.device,
            dtype=kernel.dtype,
        )
        with torch.no_grad():
            identity.weight.copy_(kernel)
        return self._fuse_conv_bn(identity, bn)

    def get_equivalent_kernel_bias(self) -> tuple[Tensor, Tensor]:
        """Return the single-convolution parameters equivalent in eval mode."""

        if self.deploy:
            return self.reparam_conv.weight, self.reparam_conv.bias

        kernel_main, bias_main = self._fuse_conv_bn(*self.dw_branch)
        kernel_point, bias_point = self._fuse_conv_bn(*self.point_branch)
        kernel_identity, bias_identity = self._fuse_identity_bn(self.identity_branch)
        pad = (self.kernel_size - 1) // 2
        kernel_point = F.pad(kernel_point, (pad, pad, pad, pad))
        return (
            kernel_main + kernel_point + kernel_identity,
            bias_main + bias_point + bias_identity,
        )

    def switch_to_deploy(self) -> "RepDepthwiseBlock":
        """Fuse all training branches into one depthwise convolution in place."""

        if self.deploy:
            return self

        was_training = self.training
        requires_grad = any(parameter.requires_grad for parameter in self.parameters())
        kernel, bias = self.get_equivalent_kernel_bias()
        reparam_conv = nn.Conv2d(
            self.channels,
            self.channels,
            self.kernel_size,
            padding=self.kernel_size // 2,
            groups=self.channels,
            bias=True,
            device=kernel.device,
            dtype=kernel.dtype,
        )
        with torch.no_grad():
            reparam_conv.weight.copy_(kernel)
            reparam_conv.bias.copy_(bias)
        reparam_conv.train(was_training)
        reparam_conv.requires_grad_(requires_grad)
        self.reparam_conv = reparam_conv
        del self.dw_branch
        del self.point_branch
        del self.identity_branch
        self.deploy = True
        return self

    def forward(self, x: Tensor) -> Tensor:
        if self.deploy:
            return self.act(self.reparam_conv(x))
        return self.act(
            self.dw_branch(x) + self.point_branch(x) + self.identity_branch(x)
        )


__all__ = ["ConvBNAct", "DepthwiseSeparableConv", "RepDepthwiseBlock"]
