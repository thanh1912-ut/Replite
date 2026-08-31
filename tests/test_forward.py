"""Forward/backward and BatchNorm freshness tests."""

from __future__ import annotations

import torch
from torch.nn.modules.batchnorm import _BatchNorm

from replite.backbone import create_backbone


def test_forward_backward_runs_on_cpu(backbone_name, make_input) -> None:
    model = create_backbone(backbone_name)
    model.train()
    inputs = make_input(2, 3, 64, 96).requires_grad_(True)

    features = model(inputs)
    loss = sum(feature.float().pow(2).mean() for feature in features)
    loss.backward()

    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    assert model.conv_stem.weight.grad is not None
    assert torch.isfinite(model.conv_stem.weight.grad).all()
    last_group = model.blocks[-1]
    last_parameters = [p for p in last_group.parameters() if p.requires_grad]
    assert last_parameters and last_parameters[0].grad is not None
    for parameter in model.parameters():
        assert parameter.grad is not None


@torch.no_grad()
def _assert_no_nonfinite_gradients(model: torch.nn.Module) -> None:
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"unused trainable parameter: {name}"
        assert torch.isfinite(parameter.grad).all(), f"non-finite gradient: {name}"


@torch.enable_grad()
def test_every_subset_has_no_unused_trainable_parameters(
    backbone_name, make_input
) -> None:
    for out_indices in ((0,), (1,), (2,), (3,), (0, 2), (1, 3)):
        model = create_backbone(backbone_name, out_indices=out_indices)
        model.train()
        outputs = model(make_input(2, 3, 64, 96))
        sum(output.float().square().mean() for output in outputs).backward()
        _assert_no_nonfinite_gradients(model)


def test_eval_forward_is_deterministic(backbone_name, make_input) -> None:
    model = create_backbone(backbone_name)
    model.eval()
    inputs = make_input(1, 3, 96, 128)
    with torch.no_grad():
        first = model(inputs)
        second = model(inputs)
    for a, b in zip(first, second):
        torch.testing.assert_close(a, b, rtol=0.0, atol=0.0)


def test_constructor_does_not_update_batchnorm(backbone_name) -> None:
    """No dummy forward may run inside the constructor (BN stats must be fresh)."""
    model = create_backbone(backbone_name)
    batch_norms = [
        module for module in model.modules() if isinstance(module, _BatchNorm)
    ]
    assert batch_norms, "expected BatchNorm layers in the trunk"
    for norm in batch_norms:
        assert norm.num_batches_tracked.item() == 0
        if norm.running_mean is not None:
            assert torch.all(norm.running_mean == 0)
            assert torch.all(norm.running_var == 1)


def test_batchnorm_unchanged_after_eval_forward(backbone_name, make_input) -> None:
    model = create_backbone(backbone_name)
    model.eval()
    before = {
        name: buffer.clone()
        for name, buffer in model.named_buffers()
        if "running" in name or "num_batches" in name
    }
    with torch.no_grad():
        model(make_input(1, 3, 64, 64))
    after = {
        name: buffer.clone()
        for name, buffer in model.named_buffers()
        if "running" in name or "num_batches" in name
    }
    assert before.keys() == after.keys()
    for name in before:
        torch.testing.assert_close(before[name], after[name], rtol=0.0, atol=0.0)
