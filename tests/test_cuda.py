"""Optional CUDA, AMP, and channels-last smoke tests."""

from __future__ import annotations

import pytest
import torch

from replite.backbone import create_backbone
from replite.multitask import RepLiteConfig, RepLiteMultiTaskModel, TaskConfig


pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
]


def _normalized_cuda_input(batch: int = 2) -> torch.Tensor:
    generator = torch.Generator().manual_seed(2026)
    inputs = torch.rand(batch, 3, 128, 256, generator=generator)
    mean = inputs.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = inputs.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    return ((inputs - mean) / std).cuda()


def test_channels_last_cuda_matches_contiguous(backbone_name) -> None:
    contiguous = create_backbone(backbone_name).eval().cuda()
    channels_last = create_backbone(backbone_name).eval()
    channels_last.load_state_dict(contiguous.state_dict())
    channels_last = channels_last.cuda().to(memory_format=torch.channels_last)

    inputs = _normalized_cuda_input()
    channels_last_inputs = inputs.contiguous(memory_format=torch.channels_last)
    with torch.no_grad():
        reference = contiguous(inputs)
        actual = channels_last(channels_last_inputs)
    assert len(reference) == len(actual)
    for expected, observed in zip(reference, actual):
        torch.testing.assert_close(expected, observed, rtol=1e-3, atol=1e-4)
        assert torch.isfinite(observed).all()


def test_cuda_amp_channels_last_forward_backward(backbone_name) -> None:
    model = create_backbone(backbone_name).cuda()
    model = model.to(memory_format=torch.channels_last).train()
    inputs = (
        _normalized_cuda_input()
        .contiguous(memory_format=torch.channels_last)
        .requires_grad_(True)
    )

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        outputs = model(inputs)
        loss = sum(output.float().square().mean() for output in outputs)
    loss.backward()

    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"unused CUDA parameter: {name}"
        assert torch.isfinite(parameter.grad).all(), f"non-finite CUDA gradient: {name}"


def _cuda_multitask_model() -> RepLiteMultiTaskModel:
    config = RepLiteConfig(
        tasks=TaskConfig(
            detection_classes=3,
            segmentation_classes=2,
            depth=True,
        ),
        backbone_name="mobilenetv4_conv_small",
        recurrent_c4_channels=16,
        recurrent_c5_channels=24,
        neck_channels=16,
        dense_channels=12,
        task_adapter_channels=12,
        detection_head_channels=16,
    )
    return RepLiteMultiTaskModel(config)


def test_multitask_cuda_amp_channels_last_forward_backward() -> None:
    model = _cuda_multitask_model().cuda().to(memory_format=torch.channels_last)
    images = (
        _normalized_cuda_input()
        .contiguous(memory_format=torch.channels_last)
        .requires_grad_(True)
    )
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output = model(images)
        assert output.detection is not None
        assert output.segmentation is not None
        assert output.depth is not None
        loss = output.segmentation.float().mean() + output.depth.float().mean()
        for group in output.detection:
            loss = loss + sum(tensor.float().mean() for tensor in group)
    loss.backward()

    assert images.grad is not None and torch.isfinite(images.grad).all()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"unused CUDA parameter: {name}"
        assert torch.isfinite(parameter.grad).all(), f"non-finite CUDA gradient: {name}"


def test_multitask_streaming_state_stays_on_cuda_with_amp() -> None:
    model = _cuda_multitask_model().eval().cuda()
    frame = _normalized_cuda_input(batch=1)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        _, state = model.forward_step(frame)
        output, next_state = model.forward_step(frame, state)

    assert output.depth is not None and torch.isfinite(output.depth).all()
    for level in next_state:
        for tensor in level:
            assert tensor.is_cuda
            assert tensor.dtype == torch.float16
            assert torch.isfinite(tensor).all()
